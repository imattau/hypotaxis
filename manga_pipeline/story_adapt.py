from __future__ import annotations

import json
import re
import threading
from functools import lru_cache
from pathlib import Path
from typing import Callable

from .captioner import Captioner
from .layouts import LAYOUTS, is_wide_box
from .llm import SmallLLM, get_embedder
from .registry import CharacterRegistry
from .schema import DialogueLine, Page, Panel, Story
from .train_captioner import parse_caption_and_camera

_REPORTING_VERBS = {
    "say", "ask", "reply", "whisper", "shout", "call", "murmur", "mutter", "cry",
    "yell", "answer", "continue", "add", "admit", "insist", "wonder", "exclaim",
    "snap", "sigh", "respond", "declare", "protest", "note", "observe", "think",
}

_TEMPLATES_BY_COUNT: dict[int, list[str]] = {}
for _name, _boxes in LAYOUTS.items():
    _TEMPLATES_BY_COUNT.setdefault(len(_boxes), []).append(_name)

# base preference order for page pacing (favor typical page sizes over
# cramming everything into one big grid just because the count happens to
# divide evenly) - pack_into_pages rotates this per group rather than
# always trying it in this fixed order, see the comment there
_SUPPORTED_COUNTS = [c for c in (3, 4, 2, 9) if c in _TEMPLATES_BY_COUNT]
_DATASET_LOCK = threading.Lock()

# the two camera hints found (via a real page-generation test) to need a
# landscape-oriented box - see is_wide_box(). over-the-shoulder/bird's-eye
# view/close-up/medium shot aren't here: nothing in real testing showed them
# needing a wide box specifically, only these two.
_WIDE_CAMERA_HINTS = {"wide two-shot", "wide establishing shot"}


def _layout_fits(layout: str, group: list[Panel]) -> bool:
    """True if every panel in `group` that needs a wide box (see
    _WIDE_CAMERA_HINTS) actually gets one from this layout, matched by
    position - pipeline.run() zips a page's panels with boxes_for(layout) in
    the same reading order, so box i is where panel i actually renders."""
    return all(
        panel.camera_hint not in _WIDE_CAMERA_HINTS or is_wide_box(box) for panel, box in zip(group, LAYOUTS[layout])
    )
_MAX_SEGMENT_SENTENCES = 1500


_TITLE_ABBREVIATIONS = ("Dr", "Mr", "Mrs", "Ms", "Prof", "Sr", "Jr", "St", "Capt", "Lt", "Col", "Gen", "Rev", "Hon", "Mt")


def split_sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text.strip())
    # protect a title abbreviation's period ("Dr. Osei") from being treated
    # as a sentence boundary - naive period-based splitting mistook it for
    # one, fragmenting "Dr. Osei jogged..." into "...as Dr." + "Osei
    # jogged..." (found via real caption-quality testing on generated data:
    # the orphaned "Dr." fragment produced a useless, ungrounded caption).
    # \x00 marks a protected space so the split regex below skips it, since
    # Python's re module doesn't support variable-length lookbehind for the
    # alternation of abbreviations this would otherwise need.
    for abbr in _TITLE_ABBREVIATIONS:
        text = re.sub(rf"\b{abbr}\.\s+", f"{abbr}.\x00", text)
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.replace("\x00", " ").strip() for s in sentences if s.strip()]


def score_sentence_importance(sentences: list[str], embeddings) -> list[float]:
    """LexRank-style importance score per sentence: build a graph where edge
    weight is embedding cosine similarity, then rank sentences by graph
    centrality (PageRank). This is the deterministic, embeddings-only
    equivalent of asking an LLM 'which sentences matter most' - no extra
    model or LLM call needed, since Stage A already computes these sentence
    embeddings for segmentation.
    """
    import networkx as nx

    n = len(sentences)
    if n <= 1:
        return [1.0] * n

    similarity = embeddings @ embeddings.T
    graph = nx.Graph()
    graph.add_nodes_from(range(n))
    threshold = 0.15
    for i in range(n):
        for j in range(i + 1, n):
            weight = float(similarity[i, j])
            if weight > threshold:
                graph.add_edge(i, j, weight=weight)

    if graph.number_of_edges() == 0:
        return [1.0] * n
    scores = nx.pagerank(graph, weight="weight")
    return [scores[i] for i in range(n)]


def target_panel_count(word_count: int, min_panels: int = 6, max_panels: int = 90, words_per_panel: int = 50) -> int:
    """Panel budget for a chapter, independent of how many sentences it
    happens to contain. Without this, panel count tracks sentence count
    1:1 regardless of chapter length - a real multi-thousand-word chapter
    would produce hundreds of panels instead of a paced adaptation. This
    keeps panel density roughly constant with floor/ceiling bounds so very
    short or very long input still yields a normal-looking manga chapter.

    words_per_panel=50 is calibrated against typical manga panel density
    (roughly 40-60 words of source prose per panel), not chosen arbitrarily -
    an earlier default of 130 was tested and compressed too aggressively,
    losing real plot beats and causing the caption LLM to hallucinate when
    forced to summarize an overly dense merged chunk in one sentence.
    """
    return max(min_panels, min(max_panels, round(word_count / words_per_panel)))


_MAX_DIALOGUE_PER_CHUNK = 3


def _dialogue_span_count(text: str) -> int:
    """Counts quoted speech spans and italicized thought spans in `text` -
    the same two span kinds split_dialogue() (further below) turns into a
    DialogueLine, and so into a rendered bubble. A cheap over-count (it
    doesn't apply split_dialogue's own filters, e.g. the "*emphasis on a
    quote*"/single-word-italics exclusions) is fine here: this only feeds a
    chunk-boundary heuristic, not the actual dialogue extraction."""
    return len(_QUOTE_RE.findall(text)) + len(_THOUGHT_RE.findall(text))


def segment_text(
    text: str, drop_threshold: float = 0.15, max_sentences: int = 3, target_panels: int | None = None
) -> list[str]:
    """Split prose into panel-sized chunks via sentence-embedding similarity
    drops (TextTiling-style), rather than fixed-size or purely rule-based
    chunking.

    If target_panels is given and the initial topic-boundary segmentation
    produces more chunks than that, compress by repeatedly merging the
    adjacent pair of chunks with the lowest combined LexRank importance
    score - i.e. low-importance/descriptive filler gets folded into a
    neighboring panel first, while higher-importance beats keep their own
    panel for as long as the budget allows.

    A chunk this dense in quoted/thought dialogue produces a panel with a
    tower of speech bubbles - real dialogue-heavy prose has plenty of very
    short exchanges ("Yeah." "Sure," "Weird?"), so a handful of them still
    fits comfortably under a word-count-only budget even though they'd
    render as far too many bubbles for one panel. _MAX_DIALOGUE_PER_CHUNK
    is therefore enforced as a second, independent hard boundary alongside
    max_sentences during initial segmentation, and again as a floor the
    target_panels compression step below refuses to merge past - so a
    dialogue-dense passage ends up with more (smaller) panels than the raw
    word budget alone would give it, rather than one overcrowded panel.
    """
    sentences = split_sentences(text)
    if not sentences:
        raise ValueError("story text must contain at least one sentence")
    if len(sentences) > _MAX_SEGMENT_SENTENCES:
        raise ValueError(f"story is too long; maximum is {_MAX_SEGMENT_SENTENCES} sentences")
    if len(sentences) <= 1:
        return sentences

    embedder = get_embedder()
    embeddings = embedder.encode(sentences, normalize_embeddings=True)
    sims = [float(embeddings[i] @ embeddings[i + 1]) for i in range(len(embeddings) - 1)]
    dialogue_counts = [_dialogue_span_count(s) for s in sentences]

    # build initial chunks as (start, end) sentence-index ranges rather than
    # joined strings, so they can still be merged during compression below
    ranges: list[tuple[int, int]] = []
    start = 0
    running_dialogue = 0
    for i, sim in enumerate(sims):
        boundary = sim < (1 - drop_threshold)
        running_dialogue += dialogue_counts[i]
        current_len = i + 1 - start
        if boundary or current_len >= max_sentences or running_dialogue > _MAX_DIALOGUE_PER_CHUNK:
            ranges.append((start, i + 1))
            start = i + 1
            running_dialogue = 0
    ranges.append((start, len(sentences)))

    if target_panels is not None and len(ranges) > target_panels:
        importance = score_sentence_importance(sentences, embeddings)

        def chunk_score(r: tuple[int, int]) -> float:
            lo, hi = r
            return sum(importance[lo:hi]) / (hi - lo)

        def chunk_dialogue(r: tuple[int, int]) -> int:
            lo, hi = r
            return sum(dialogue_counts[lo:hi])

        while len(ranges) > target_panels:
            mergeable = [
                i
                for i in range(len(ranges) - 1)
                if chunk_dialogue(ranges[i]) + chunk_dialogue(ranges[i + 1]) <= _MAX_DIALOGUE_PER_CHUNK
            ]
            if not mergeable:
                # every remaining adjacent pair would exceed the per-panel
                # dialogue cap if merged - stop compressing early and accept
                # more panels than target_panels rather than crowd one
                break
            pair_scores = [chunk_score(ranges[i]) + chunk_score(ranges[i + 1]) for i in mergeable]
            merge_at = mergeable[min(range(len(pair_scores)), key=lambda k: pair_scores[k])]
            merged = (ranges[merge_at][0], ranges[merge_at + 1][1])
            ranges[merge_at : merge_at + 2] = [merged]

    return [" ".join(sentences[lo:hi]) for lo, hi in ranges]


_MARKDOWN_HEADER_RE = re.compile(r"^#{1,6}[ \t]+.*$", re.MULTILINE)
_BLOCKQUOTE_MARKER_RE = re.compile(r"^[ \t]*>[ \t]?", re.MULTILINE)
_SCENE_BREAK_RE = re.compile(r"^[ \t]*(?:[-*_][ \t]*){3,}$", re.MULTILINE)


def _strip_markdown_structure(text: str) -> str:
    """Strip markdown structural markup that isn't narrative prose - a
    chapter heading ("# Chapter 1: Soft Reset") or a blockquote marker
    ("> ") would otherwise get swept into the sentence stream as literal
    text, polluting the first real sentence or an in-story quoted note
    (found on a real manuscript: the chapter heading became part of the
    opening sentence's embedding/caption input). Scene-break markers
    ("---", "***", "___" alone on a line) are handled separately by
    _split_on_scene_breaks, not here, since they need to survive as
    boundary signals rather than just being deleted.
    """
    text = _MARKDOWN_HEADER_RE.sub("", text)
    text = _BLOCKQUOTE_MARKER_RE.sub("", text)
    return text


def _split_on_scene_breaks(text: str) -> list[str]:
    """Split raw prose on standalone scene-break lines ("---", "***", "___")
    into independent sections. Without this, a scene break is invisible to
    segment_text's embedding-similarity chunking - it isn't sentence-ending
    punctuation, so it never creates a chunk boundary on its own, and the
    scenes on either side of it can still get merged into the same panel
    during panel-budget compression. Segmenting each section independently
    (see _segment_with_scene_breaks) makes every scene break a hard
    boundary no single panel can straddle.
    """
    sections = _SCENE_BREAK_RE.split(text)
    return [s.strip() for s in sections if s.strip()]


def _segment_with_scene_breaks(text: str, target_panels: int) -> list[str]:
    """segment_text(), but scene breaks are hard panel boundaries - see
    _split_on_scene_breaks. The panel budget is split across sections in
    proportion to each section's word count, with every non-empty section
    guaranteed at least one panel."""
    sections = _split_on_scene_breaks(text)
    if len(sections) <= 1:
        return segment_text(sections[0] if sections else text, target_panels=target_panels)

    word_counts = [max(1, len(s.split())) for s in sections]
    total_words = sum(word_counts)
    shares = [max(1, round(target_panels * wc / total_words)) for wc in word_counts]

    chunks: list[str] = []
    for section, share in zip(sections, shares):
        chunks.extend(segment_text(section, target_panels=share))
    return chunks


def _parse_profile_sheet(text: str) -> dict[str, str]:
    """Parse an author-supplied profile sheet: one 'Name: description' per
    line (blank lines and '#' comments ignored). Trusted input, so it
    bypasses NER entirely - used both for character cast sheets and for
    location/prop sheets, since the format and purpose are identical:
    guarantee an asset is recognized regardless of what automatic detection
    can find, and seed its registry description/reference image from the
    author's own words instead of a generated guess.
    """
    profiles: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        name, description = line.split(":", 1)
        name = name.strip()
        description = description.strip()
        if name and description:
            profiles[name] = description
    return profiles


_NO_FORM_TAG_RE = re.compile(r"^\[no-form\]\s*", re.IGNORECASE)


def parse_character_profiles(text: str) -> tuple[dict[str, str], set[str]]:
    """See _parse_profile_sheet. Character names have no reliable automatic
    detection substitute for NER's blind spots (non-Western names, etc.),
    so this is the primary mitigation for that gap.

    A description starting with the literal tag "[no-form]" marks that
    character as having no physical/humanoid appearance at all (an AI, a
    voice, a presence) - returned separately as a set of names, since
    CharacterRegistry needs to carry this flag too (Stage C runs as a
    separate invocation from Stage A, sometimes a different day). See
    backends.py: forcing the normal "character reference portrait, front-
    facing" template onto something explicitly described as having no body
    just produces a person anyway (confirmed empirically), so an abstract
    character skips reference-portrait generation and IP-Adapter
    conditioning entirely and is text-anchored into the prompt instead,
    the same way props are.
    """
    raw = _parse_profile_sheet(text)
    abstract_names: set[str] = set()
    profiles: dict[str, str] = {}
    for name, description in raw.items():
        match = _NO_FORM_TAG_RE.match(description)
        if match:
            abstract_names.add(name)
            description = description[match.end():].strip()
        profiles[name] = description
    return profiles, abstract_names


def parse_location_profiles(text: str) -> dict[str, str]:
    """See _parse_profile_sheet. Locations have no automatic detection at
    all (NER only catches named places like 'Tokyo', not generic-but-
    important scene settings like 'the abandoned mill'), so this is the
    only way to register them, not just a mitigation."""
    return _parse_profile_sheet(text)


def parse_prop_profiles(text: str) -> dict[str, str]:
    """See _parse_profile_sheet. Props are kept as a separate registry from
    locations even though the file format is identical, because they're
    handled completely differently downstream: a location is a whole
    backdrop and gets image-conditioned via IP-Adapter, while a small
    portable object gets only text-anchored into the generation prompt (see
    backends.py) - testing showed a 'wide establishing shot' reference for
    a small prop produces a cluttered, unusable scene rather than a clean
    single-object reference, and whole-image IP-Adapter conditioning on
    such a reference would distort a panel's entire composition around one
    incidental object. This also avoids a real structural problem: a
    location and a prop routinely appear in the same panel ('the letter, in
    the mill'), and if they shared one registry/slot, that case would fall
    into 'multiple names - skip conditioning entirely'."""
    return _parse_profile_sheet(text)


@lru_cache(maxsize=None)
def get_nlp():
    import spacy

    return spacy.load("en_core_web_sm")


def extract_person_entities(text: str) -> list[str]:
    """Real statistical NER (spaCy) for character names, replacing an
    earlier capitalized-word heuristic. Run once over the whole chapter
    (not per-panel chunk) since a short isolated chunk gives the model too
    little context to reliably recognize names - NER accuracy improves
    noticeably with surrounding paragraph context.
    """
    doc = get_nlp()(text)
    names: list[str] = []
    for ent in doc.ents:
        if ent.label_ == "PERSON" and ent.text not in names:
            names.append(ent.text)
    return names


def _merge_person_aliases(names: list[str], priority_names: list[str] | None = None) -> tuple[list[str], dict[str, str]]:
    """NER emits a separate span for each way a person is referred to across
    a document - "Dr. Mina Park", "Mina Park", "Park" - plus stray possessive
    spans like "Mina Park's". Left unmerged, each becomes its own bogus
    character. Collapse same-cluster spans into one canonical (longest) name,
    picking the longest form first so shorter aliases get absorbed into it
    rather than the reverse.

    priority_names (author-supplied character profiles / already-registered
    names) are always canonical and never absorbed into an NER-derived form,
    even a longer one - spaCy's PERSON span usually drops a leading title
    ("Dr."), so an NER hit like "Mina Park" needs to merge into the profile's
    "Dr. Mina Park" rather than the reverse.

    Returns (canonical_names, alias_to_canonical) rather than just a merged
    list: a chunk of prose that only uses the short form ("Park") needs to
    still be tagged under the canonical name, so callers must check both.
    """
    cleaned = [n[:-2] if n.endswith("'s") else n for n in names]
    cleaned = [n.strip() for n in cleaned if n.strip()]
    unique = list(dict.fromkeys(cleaned))
    by_length = sorted(unique, key=len, reverse=True)

    canonical_forms: list[str] = list(dict.fromkeys(priority_names or []))
    alias_to_canonical: dict[str, str] = {name: name for name in canonical_forms}
    for name in by_length:
        if name in alias_to_canonical:
            continue
        owner = next(
            (c for c in canonical_forms if re.search(rf"\b{re.escape(name)}\b", c, re.IGNORECASE)),
            None,
        )
        alias_to_canonical[name] = owner if owner is not None else name
        if owner is None:
            canonical_forms.append(name)

    ordered_canonical: list[str] = []
    for name in unique:
        canon = alias_to_canonical[name]
        if canon not in ordered_canonical:
            ordered_canonical.append(canon)

    aliases_only = {alias: canon for alias, canon in alias_to_canonical.items() if alias != canon}
    return ordered_canonical, aliases_only


def names_in_chunk(chunk: str, known_names: list[str]) -> list[str]:
    """Which of a known set of names (characters or locations/props) are
    actually mentioned in this panel's chunk - a simple substring check,
    since we already know these are real names and just need to know where
    each one appears.
    """
    return [name for name in known_names if name in chunk]


# kept as an alias: existing call sites/tests refer to this by its original,
# character-specific name
characters_in_chunk = names_in_chunk


_QUOTE_RE = re.compile(r'"([^"]+)"')

# markdown-italicized text ("*Or loving me.*") as an internal-thought marker -
# the negative lookaround on both sides excludes **bold** (double asterisks),
# which is emphasis, not the single-asterisk italic convention prose uses for
# a character's unspoken thought
_THOUGHT_RE = re.compile(r"(?<!\*)\*(?!\*)([^*\n]+)(?<!\*)\*(?!\*)")

_QUOTE_NORMALIZE_TABLE = str.maketrans({
    "“": '"',  # left double quotation mark
    "”": '"',  # right double quotation mark
    "‘": "'",  # left single quotation mark
    "’": "'",  # right single quotation mark / apostrophe
})


def _normalize_quotes(text: str) -> str:
    """Word/Docs/Notes/many markdown editors write curly typographic quotes
    by default - straight to _QUOTE_RE (which only matches straight `"`)
    those come through as plain prose with no detectable dialogue at all.
    Normalizing to straight quotes/apostrophes up front (before any other
    Stage A processing) fixes dialogue extraction and also stops spaCy's
    NER from occasionally splitting a curly-apostrophe possessive like
    "Jules’s" into its own bogus entity distinct from "Jules"."""
    return text.translate(_QUOTE_NORMALIZE_TABLE)


def _propn_subject_of_any_verb(sentence, known_characters: list[str]) -> str | None:
    for tok in sentence:
        if tok.pos_ != "VERB":
            continue
        subject = next((c for c in tok.children if c.dep_ in ("nsubj", "nsubjpass")), None)
        if subject is None:
            continue
        candidate = subject
        if subject.pos_ != "PROPN":
            # "Dr. Park's face hovered..." - subject is "face" (NOUN), with
            # "Park" attached to it as a possessive modifier. The sentence
            # is about the possessor, not the body part/object, so treat a
            # possessive PROPN child as the effective subject too.
            candidate = next((c for c in subject.children if c.dep_ == "poss" and c.pos_ == "PROPN"), None)
            if candidate is None:
                continue
        for name in known_characters:
            if candidate.text in name or name in candidate.text:
                return name
        return candidate.text
    return None


def _resolve_speaker_via_parse(doc, quote_start: int, quote_end: int, known_characters: list[str]) -> str | None:
    """Find the grammatical subject of a reporting verb (said/whispered/
    called/...) in the sentence containing this quote - the speaker is
    reliably that subject in English regardless of which character name
    happens to sit textually closer to the quote (the failure mode of a
    nearest-name heuristic, e.g. Ren calling out 'Aiko?' was previously
    misattributed to Aiko since her name is the only one adjacent to the
    quote).

    Falls back to a second check when the quote's own sentence has no
    reporting verb at all - a very common prose pattern is a short action
    beat immediately before a bare quote ('Jules shrugged. "I dreamed."'),
    where "shrugged" isn't a reporting verb but its subject is still
    unambiguously the speaker of the quote that follows. Only applies when
    the beat sentence is immediately adjacent to the quote's sentence (nothing
    but whitespace between them) - a name appearing in some earlier,
    unrelated sentence is not the same signal.

    Returns None (not 'unknown') when neither check resolves to a name, so
    the caller can fall back to its own heuristics rather than treating an
    unresolved pronoun as a name.
    """
    # the sentence containing the quote's *opening* - not one spanning the
    # whole quote, since spaCy splits a multi-sentence quoted line ("Nova is
    # programmed... Maybe she's responding...") into multiple sentences of
    # its own, and no single sentence would span end-to-end in that case
    sentence = next((s for s in doc.sents if s.start_char <= quote_start < s.end_char), None)
    if sentence is None:
        return None
    for tok in sentence:
        if tok.pos_ != "VERB" or tok.lemma_.lower() not in _REPORTING_VERBS:
            continue
        subject = next((c for c in tok.children if c.dep_ in ("nsubj", "nsubjpass")), None)
        if subject is None:
            continue
        if subject.pos_ != "PROPN":
            return None  # pronoun subject - needs coreference we don't have; let caller fall back
        for name in known_characters:
            if subject.text in name or name in subject.text:
                return name
        return subject.text

    prev_sentence = None
    for s in doc.sents:
        if s.end_char <= sentence.start_char:
            prev_sentence = s
        else:
            break
    if prev_sentence is not None and sentence.start_char - prev_sentence.end_char <= 2:
        return _propn_subject_of_any_verb(prev_sentence, known_characters)
    return None


def split_dialogue(
    chunk: str, characters: list[str], default_speaker: str | None = None, doc=None
) -> list[DialogueLine]:
    lines: list[DialogueLine] = []
    # prefer rolling context from earlier panels over "first character named
    # in this chunk" - a chunk can mention multiple characters from
    # different beats once panel-budget compression merges adjacent chunks
    # together, and the first-named one isn't necessarily who's speaking
    last_speaker = default_speaker or (characters[0] if characters else "unknown")
    # the most recent speaker *distinct* from last_speaker - not simply the
    # previous line's speaker, since two consecutive lines can genuinely
    # belong to the same person ("Maybe." Jules hesitated. "It didn't feel
    # programmed.") and that shouldn't erase the record of who the other
    # active participant is for ping-pong alternation below
    other_speaker: str | None = None
    prior_spans: list[tuple[int, int]] = []

    # speech (quoted) and thought (markdown-italicized, e.g. "*Or loving
    # me.*") spans are interleaved by position and run through the same
    # speaker-resolution pipeline - a thought's speaker is attributed
    # exactly the same grammatical way a quote's is ("Jules almost said,
    # *Or loving me.*" resolves via the same reporting-verb check)
    quote_matches = list(_QUOTE_RE.finditer(chunk))
    thought_matches = [
        m
        for m in _THOUGHT_RE.finditer(chunk)
        # italics wrapped *around* a quote ("*"I've always seen you."*") is
        # emphasis on spoken dialogue, not a separate unspoken thought -
        # without this, prose that italicizes a line of spoken dialogue for
        # emphasis produced two duplicate bubbles for the same line
        if not any(m.start() < q.end() and q.start() < m.end() for q in quote_matches)
        # prose italicizes plenty of things that aren't a character's
        # internal thought - a single emphasized word ("the way she said
        # *love*"), a painting/room title ("*Resonance*", "*Continuum*").
        # A real thought is a sentence-like fragment, not one bare word -
        # this cheap filter isn't perfect (it can't catch multi-word titles,
        # and won't tell a genuine thought from italicized reported speech)
        # but it removes the clearest false-positive class.
        and len(m.group(1).split()) >= 3
    ]
    matches = sorted(
        [(m, "speech") for m in quote_matches] + [(m, "thought") for m in thought_matches],
        key=lambda pair: pair[0].start(),
    )

    for match, kind in matches:
        text = match.group(1).strip()
        speaker = None
        if doc is not None:
            speaker = _resolve_speaker_via_parse(doc, match.start(), match.end(), characters)
        if speaker is None:
            # never look further back than the immediately preceding
            # speech/thought span - a name attached to an earlier beat
            # ("Jules shrugged.") is that beat's attribution cue, not a
            # floating cue that should keep bleeding forward into every
            # later line within 40 chars of it
            prior_end = prior_spans[-1][1] if prior_spans else 0
            window_start = max(prior_end, match.start() - 40)
            window = chunk[window_start : match.start()]
            speaker = next((name for name in reversed(characters) if name in window), None)
        if speaker is None and other_speaker is not None:
            # no explicit signal at all for this line - unattributed dialogue
            # in fiction overwhelmingly alternates between the two active
            # speakers rather than one character monologuing indefinitely, so
            # flip to the other active participant instead of just repeating
            # last_speaker (which produced long wrong streaks on real
            # dialogue-dense chapters - see the pipeline review)
            speaker = other_speaker
        if speaker is None:
            speaker = last_speaker
        lines.append(DialogueLine(speaker=speaker, text=text, kind=kind))
        prior_spans.append((match.start(), match.end()))
        if speaker != last_speaker:
            other_speaker = last_speaker
        last_speaker = speaker
    return lines


# camera-hint keyword fallback and CAPTION:/CAMERA: response parsing now
# live in train_captioner.py (as CAMERA_HINTS / guess_camera_hint /
# parse_caption_and_camera) - shared with Captioner.generate() so the
# trained-captioner path gets the same content-aware camera framing as the
# bridge-LLM path below, not just the flat keyword heuristic.


def _merge_panel(base: Panel, extra: Panel) -> Panel:
    return Panel(
        scene_description=f"{base.scene_description} {extra.scene_description}",
        characters=list(dict.fromkeys(base.characters + extra.characters)),
        locations=list(dict.fromkeys(base.locations + extra.locations)),
        props=list(dict.fromkeys(base.props + extra.props)),
        camera_hint=base.camera_hint,
        dialogue=base.dialogue + extra.dialogue,
    )


def pack_into_pages(panels: list[Panel]) -> list[Page]:
    if len(panels) < 2:
        raise ValueError("story too short to segment into panels - need at least 2 segments")

    groups: list[list[Panel]] = []
    i = 0
    n = len(panels)
    group_index = 0
    while i < n:
        remaining = n - i
        # rotate which count gets tried first, per group, instead of always
        # trying _SUPPORTED_COUNTS in the same fixed order - found via real
        # generation testing that always preferring 3 first (the previous
        # behavior) made virtually every page a 3-panel page, since
        # "3 <= remaining" is true on nearly every iteration for any story
        # of reasonable length, leaving 4-panel/9-panel layouts effectively
        # unreachable. This keeps pagination fully deterministic (the same
        # story always paginates the same way) while giving real page-size
        # variety instead of a fixed count picked once and reused forever.
        offset = group_index % len(_SUPPORTED_COUNTS)
        rotated_counts = _SUPPORTED_COUNTS[offset:] + _SUPPORTED_COUNTS[:offset]
        count = next((c for c in rotated_counts if c <= remaining), None)
        if count is None:
            groups[-1][-1] = _merge_panel(groups[-1][-1], panels[i])
            i += 1
            continue
        groups.append(panels[i : i + count])
        i += count
        group_index += 1

    pages = []
    for page_index, group in enumerate(groups):
        templates = _TEMPLATES_BY_COUNT[len(group)]
        # prefer a template whose box shapes actually fit this group's
        # camera hints (see _layout_fits) - falls back to every template of
        # this count if none fit (e.g. a wide-shot panel in a page size that
        # has no landscape-oriented box at all), same as before this existed
        fitting = [t for t in templates if _layout_fits(t, group)]
        candidates = fitting or templates
        layout = candidates[page_index % len(candidates)]
        pages.append(Page(layout=layout, panels=group))
    return pages


_CAPTION_PROMPT = """You are converting prose fiction into a single manga panel description for an image generator.
Passage: {chunk}
Characters present: {characters}
Known appearance (mention only briefly if relevant, do not dwell on it): {appearance_notes}
Reply with exactly two lines:
CAPTION: one sentence, under 25 words, describing only the setting, action, and expression stated in the passage. Do not invent objects, locations, or events not in the passage. Do not include dialogue. Do not add characters not listed.
CAMERA: pick exactly one of these camera framings, whichever best fits the moment described - extreme close-up, close-up, medium shot, wide two-shot, wide establishing shot, over-the-shoulder, bird's-eye view"""

_DESCRIPTION_PROMPT = """Passage: {chunk}
Based only on this passage, describe {name}'s visual appearance in under 15 words (hair, clothing, build). If not stated, invent one simple consistent appearance. Reply with the description only."""


def _append_jsonl(path: str | Path, record: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _DATASET_LOCK:
        with path.open("a") as f:
            f.write(json.dumps(record) + "\n")


def _caption_input(chunk: str, character_notes: str) -> str:
    """The caption model's raw (unprefixed) input text - shared between
    dataset harvesting (written as the {"input": ...} field a human later
    reviews) and Captioner.generate() at inference time, so a trained
    captioner sees text in exactly the same shape it was fine-tuned on."""
    return f"{chunk}\nKnown appearances: {character_notes}" if character_notes else chunk


def adapt_story(
    text: str,
    story_id: str,
    title: str,
    style_prompt: str,
    registry: CharacterRegistry,
    llm: SmallLLM,
    dataset_path: str | Path | None = None,
    on_progress: "Callable[[str], None] | None" = None,
    character_profiles: dict[str, str] | None = None,
    abstract_characters: set[str] | None = None,
    location_registry: CharacterRegistry | None = None,
    location_profiles: dict[str, str] | None = None,
    prop_registry: CharacterRegistry | None = None,
    prop_profiles: dict[str, str] | None = None,
    captioner: "Captioner | None" = None,
) -> Story:
    """dataset_path: if given, append every {input, characters, target} caption
    example produced by the bridge LLM to this JSONL file (Phase 3 harvesting) -
    this is the training data collected across Phase 2 runs to eventually
    LoRA-fine-tune a small seq2seq model and shrink/replace the bridge LLM.
    Ignored when captioner is given - harvesting is meant to capture the
    bridge LLM's raw output for later human curation, not the trained
    captioner's own output (which would be a pointless self-distillation
    loop back into its own training data).

    captioner: optional trained LoRA captioner (see manga_pipeline/captioner.py,
    train_captioner.py) - when given, used instead of prompting the full
    bridge LLM for each panel's caption. Much faster and lighter on VRAM
    than an instruct-model prompt per panel, once a curated dataset exists
    to train it from (see README's "Curating a clean caption dataset"). If
    the captioner was fine-tuned on the CAPTION:/CAMERA: joint target format
    it produces content-aware camera framing directly (see Captioner.generate);
    otherwise (an older adapter trained on caption-only targets) it falls
    back to the guess_camera_hint() keyword heuristic the same way the
    bridge-LLM path does when its own CAMERA: line is missing or
    unrecognized. Character descriptions still go through `llm` either way -
    the captioner was never trained for that task.

    on_progress: optional callback invoked with a short human-readable status
    string as chunks are processed - purely cosmetic (e.g. for a UI progress
    display), no effect on the result.

    character_profiles: optional {name: description} from an author-supplied
    cast sheet (see parse_character_profiles) - these names are guaranteed
    to be recognized regardless of NER, and their descriptions seed the
    registry directly instead of being generated by the bridge LLM.

    abstract_characters: optional set of names (the second element
    parse_character_profiles returns) with no physical/humanoid appearance -
    recorded on the registry so Stage C can skip portrait/IP-Adapter
    conditioning for them (see CharacterRegistry.set_is_abstract).

    location_registry/location_profiles: same idea as character_profiles,
    but for locations (see parse_location_profiles). Unlike characters,
    locations have no automatic-detection fallback at all - NER only
    catches named places, not generic-but-important settings like "the
    abandoned mill" - so a location only ever gets tagged on a panel if it
    was listed in location_profiles.

    prop_registry/prop_profiles: same mechanism as locations (see
    parse_prop_profiles), but kept as a separate registry/tag since props
    are handled differently downstream - see parse_prop_profiles for why.
    """

    def report(msg: str) -> None:
        if on_progress is not None:
            on_progress(msg)

    text = _normalize_quotes(text)
    text = _strip_markdown_structure(text)

    for name, description in (character_profiles or {}).items():
        if registry.get(name) is None:
            registry.set_description(name, description)
    for name in abstract_characters or ():
        registry.set_is_abstract(name, True)

    if location_registry is not None:
        for name, description in (location_profiles or {}).items():
            if location_registry.get(name) is None:
                location_registry.set_description(name, description)
    known_locations = list(location_profiles or {})

    if prop_registry is not None:
        for name, description in (prop_profiles or {}).items():
            if prop_registry.get(name) is None:
                prop_registry.set_description(name, description)
    known_props = list(prop_profiles or {})

    report("segmenting story into panels")
    budget = target_panel_count(len(text.split()))
    chunks = _segment_with_scene_breaks(text, budget)
    panels: list[Panel] = []

    # whole-text NER pass (not per-chunk) since a short isolated chunk gives
    # spaCy too little context to reliably recognize names; author-supplied
    # profile names are folded in unconditionally, ahead of NER results, so
    # they're never missed regardless of what the NER model does or doesn't
    # recognize. NER hits that collide with a known location name are
    # dropped - e.g. spaCy's PERSON tagger false-positived on "Mill" in
    # testing, and an author-confirmed location name is authoritative over
    # an automatic (and here, wrong) NER guess for the same string.
    person_entities = [
        name for name in extract_person_entities(text) if name not in known_locations and name not in known_props
    ]
    # collapse NER spans that refer to the same person under different
    # wording ("Dr. Mina Park" / "Mina Park" / "Park") into one canonical
    # name - person_aliases lets a chunk using only the short form still
    # get tagged under the canonical name (see _merge_person_aliases).
    # Author-supplied/already-registered names are given as priority so an
    # NER-derived variant merges into e.g. a profile's "Dr. Mina Park"
    # rather than creating its own separate canonical form.
    already_known = list(dict.fromkeys(list(character_profiles or {}) + list(registry.all())))
    person_entities, person_aliases = _merge_person_aliases(person_entities, priority_names=already_known)
    known_characters = list(dict.fromkeys(already_known + person_entities))

    last_speaker: str | None = next(iter(known_characters), None)
    for chunk_index, chunk in enumerate(chunks):
        report(f"writing panel {chunk_index + 1}/{len(chunks)}")
        characters = names_in_chunk(chunk, known_characters)
        for alias, canonical in person_aliases.items():
            if canonical in known_characters and canonical not in characters and alias in chunk:
                characters.append(canonical)
        locations = names_in_chunk(chunk, known_locations)
        props = names_in_chunk(chunk, known_props)
        for name in characters:
            if registry.get(name) is None:
                description = llm.generate(_DESCRIPTION_PROMPT.format(chunk=chunk, name=name), max_new_tokens=30)
                registry.set_description(name, description)

        character_notes = "; ".join(f"{name} ({registry.get(name).description})" for name in characters)
        caption_input = _caption_input(chunk, character_notes)
        if captioner is not None:
            caption, camera_hint = captioner.generate(caption_input, characters, chunk, len(characters))
        else:
            # kept as a separate prompt field rather than folded into the passage text -
            # when it was part of "Passage", the tiny LLM tended to fixate on restating
            # appearance instead of the scene's actual action (see caption quality notes)
            raw_response = llm.generate(
                _CAPTION_PROMPT.format(
                    chunk=chunk, characters=", ".join(characters) or "none", appearance_notes=character_notes or "none"
                ),
                max_new_tokens=80,
            )
            caption, camera_hint = parse_caption_and_camera(raw_response, chunk, len(characters))
            if dataset_path is not None:
                _append_jsonl(
                    dataset_path,
                    {"input": caption_input, "characters": characters, "target": caption, "camera": camera_hint},
                )

        chunk_doc = get_nlp()(chunk) if ('"' in chunk or _THOUGHT_RE.search(chunk)) else None
        dialogue = split_dialogue(chunk, characters, default_speaker=last_speaker, doc=chunk_doc)
        if characters:
            last_speaker = characters[-1]
        elif dialogue:
            last_speaker = dialogue[-1].speaker
        panels.append(
            Panel(
                scene_description=caption,
                characters=characters,
                locations=locations,
                props=props,
                camera_hint=camera_hint,
                dialogue=dialogue,
            )
        )

    report("laying out pages")
    pages = pack_into_pages(panels)
    return Story(id=story_id, title=title, style_prompt=style_prompt, pages=pages)
