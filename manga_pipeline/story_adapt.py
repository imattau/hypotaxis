from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Callable

from .layouts import LAYOUTS
from .llm import SmallLLM, get_embedder
from .registry import CharacterRegistry
from .schema import DialogueLine, Page, Panel, Story

_REPORTING_VERBS = {
    "say", "ask", "reply", "whisper", "shout", "call", "murmur", "mutter", "cry",
    "yell", "answer", "continue", "add", "admit", "insist", "wonder", "exclaim",
    "snap", "sigh", "respond", "declare", "protest", "note", "observe",
}

_TEMPLATES_BY_COUNT: dict[int, list[str]] = {}
for _name, _boxes in LAYOUTS.items():
    _TEMPLATES_BY_COUNT.setdefault(len(_boxes), []).append(_name)

# preference order for page pacing: favor typical page sizes over cramming
# everything into one big grid just because the count happens to divide evenly
_SUPPORTED_COUNTS = [c for c in (3, 4, 2, 9) if c in _TEMPLATES_BY_COUNT]


def split_sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text.strip())
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if s.strip()]


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
    """
    sentences = split_sentences(text)
    if len(sentences) <= 1:
        return sentences

    embedder = get_embedder()
    embeddings = embedder.encode(sentences, normalize_embeddings=True)
    sims = [float(embeddings[i] @ embeddings[i + 1]) for i in range(len(embeddings) - 1)]

    # build initial chunks as (start, end) sentence-index ranges rather than
    # joined strings, so they can still be merged during compression below
    ranges: list[tuple[int, int]] = []
    start = 0
    for i, sim in enumerate(sims):
        boundary = sim < (1 - drop_threshold)
        current_len = i + 1 - start
        if boundary or current_len >= max_sentences:
            ranges.append((start, i + 1))
            start = i + 1
    ranges.append((start, len(sentences)))

    if target_panels is not None and len(ranges) > target_panels:
        importance = score_sentence_importance(sentences, embeddings)

        def chunk_score(r: tuple[int, int]) -> float:
            lo, hi = r
            return sum(importance[lo:hi]) / (hi - lo)

        while len(ranges) > target_panels:
            pair_scores = [chunk_score(ranges[i]) + chunk_score(ranges[i + 1]) for i in range(len(ranges) - 1)]
            merge_at = min(range(len(pair_scores)), key=lambda i: pair_scores[i])
            merged = (ranges[merge_at][0], ranges[merge_at + 1][1])
            ranges[merge_at : merge_at + 2] = [merged]

    return [" ".join(sentences[lo:hi]) for lo, hi in ranges]


def parse_character_profiles(text: str) -> dict[str, str]:
    """Parse an author-supplied cast sheet: one 'Name: description' per line
    (blank lines and '#' comments ignored). This is trusted input, so it
    bypasses both weak points NER has for this domain - it guarantees a
    character is recognized even if the NER model's training distribution
    under-represents their name (e.g. non-Western names), and it replaces
    the bridge LLM's unreliable self-generated appearance descriptions
    (observed to sometimes return a scene sentence instead of an actual
    appearance) with the author's own, which also improves the Stage B/
    Phase 4 reference portraits generated from that description.
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


def characters_in_chunk(chunk: str, known_characters: list[str]) -> list[str]:
    """Which of the chapter's known characters (from the whole-text NER
    pass) are actually mentioned in this panel's chunk - a simple substring
    check, since we already know these are real names and just need to
    know where each one appears.
    """
    return [name for name in known_characters if name in chunk]


_QUOTE_RE = re.compile(r'"([^"]+)"')


def _resolve_speaker_via_parse(doc, quote_start: int, quote_end: int, known_characters: list[str]) -> str | None:
    """Find the grammatical subject of a reporting verb (said/whispered/
    called/...) in the sentence containing this quote - the speaker is
    reliably that subject in English regardless of which character name
    happens to sit textually closer to the quote (the failure mode of a
    nearest-name heuristic, e.g. Ren calling out 'Aiko?' was previously
    misattributed to Aiko since her name is the only one adjacent to the
    quote). Returns None (not 'unknown') when the subject is a pronoun or
    no reporting verb is found, so the caller can fall back to its own
    heuristics rather than treating an unresolved pronoun as a name.
    """
    sentence = next((s for s in doc.sents if s.start_char <= quote_start and s.end_char >= quote_end), None)
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
    for match in _QUOTE_RE.finditer(chunk):
        text = match.group(1).strip()
        speaker = None
        if doc is not None:
            speaker = _resolve_speaker_via_parse(doc, match.start(), match.end(), characters)
        if speaker is None:
            window = chunk[max(0, match.start() - 40) : match.start()]
            speaker = next((name for name in reversed(characters) if name in window), last_speaker)
        lines.append(DialogueLine(speaker=speaker, text=text, kind="speech"))
        last_speaker = speaker
    return lines


def guess_camera_hint(chunk: str, character_count: int) -> str:
    lowered = chunk.lower()
    if "close" in lowered or "hand" in lowered or "eyes" in lowered:
        return "close-up"
    if character_count >= 2:
        return "wide two-shot"
    if "stood" in lowered or "stands" in lowered or "platform" in lowered or "room" in lowered:
        return "wide establishing shot"
    return "medium shot"


def _merge_panel(base: Panel, extra: Panel) -> Panel:
    return Panel(
        scene_description=f"{base.scene_description} {extra.scene_description}",
        characters=list(dict.fromkeys(base.characters + extra.characters)),
        camera_hint=base.camera_hint,
        dialogue=base.dialogue + extra.dialogue,
    )


def pack_into_pages(panels: list[Panel]) -> list[Page]:
    if len(panels) < 2:
        raise ValueError("story too short to segment into panels - need at least 2 segments")

    groups: list[list[Panel]] = []
    i = 0
    n = len(panels)
    while i < n:
        remaining = n - i
        count = next((c for c in _SUPPORTED_COUNTS if c <= remaining), None)
        if count is None:
            groups[-1][-1] = _merge_panel(groups[-1][-1], panels[i])
            i += 1
            continue
        groups.append(panels[i : i + count])
        i += count

    pages = []
    for page_index, group in enumerate(groups):
        templates = _TEMPLATES_BY_COUNT[len(group)]
        layout = templates[page_index % len(templates)]
        pages.append(Page(layout=layout, panels=group))
    return pages


_CAPTION_PROMPT = """You are converting prose fiction into a single manga panel description for an image generator.
Passage: {chunk}
Characters present: {characters}
Write ONE sentence, under 25 words, describing only the visual moment to draw (setting, characters, action, expression). Do not include dialogue. Do not add characters not listed."""

_DESCRIPTION_PROMPT = """Passage: {chunk}
Based only on this passage, describe {name}'s visual appearance in under 15 words (hair, clothing, build). If not stated, invent one simple consistent appearance. Reply with the description only."""


def _append_jsonl(path: str | Path, record: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(record) + "\n")


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
) -> Story:
    """dataset_path: if given, append every {input, characters, target} caption
    example produced by the bridge LLM to this JSONL file (Phase 3 harvesting) -
    this is the training data collected across Phase 2 runs to eventually
    LoRA-fine-tune a small seq2seq model and shrink/replace the bridge LLM.

    on_progress: optional callback invoked with a short human-readable status
    string as chunks are processed - purely cosmetic (e.g. for a UI progress
    display), no effect on the result.

    character_profiles: optional {name: description} from an author-supplied
    cast sheet (see parse_character_profiles) - these names are guaranteed
    to be recognized regardless of NER, and their descriptions seed the
    registry directly instead of being generated by the bridge LLM.
    """

    def report(msg: str) -> None:
        if on_progress is not None:
            on_progress(msg)

    for name, description in (character_profiles or {}).items():
        if registry.get(name) is None:
            registry.set_description(name, description)

    report("segmenting story into panels")
    budget = target_panel_count(len(text.split()))
    chunks = segment_text(text, target_panels=budget)
    panels: list[Panel] = []

    # whole-text NER pass (not per-chunk) since a short isolated chunk gives
    # spaCy too little context to reliably recognize names; author-supplied
    # profile names are folded in unconditionally, ahead of NER results, so
    # they're never missed regardless of what the NER model does or doesn't
    # recognize
    known_characters = list(
        dict.fromkeys(list(character_profiles or {}) + list(registry.all()) + extract_person_entities(text))
    )

    last_speaker: str | None = next(iter(known_characters), None)
    for chunk_index, chunk in enumerate(chunks):
        report(f"writing panel {chunk_index + 1}/{len(chunks)}")
        characters = characters_in_chunk(chunk, known_characters)
        for name in characters:
            if registry.get(name) is None:
                description = llm.generate(_DESCRIPTION_PROMPT.format(chunk=chunk, name=name), max_new_tokens=30)
                registry.set_description(name, description)

        character_notes = "; ".join(f"{name} ({registry.get(name).description})" for name in characters)
        caption_input = f"{chunk}\nKnown appearances: {character_notes}" if character_notes else chunk
        caption = llm.generate(
            _CAPTION_PROMPT.format(chunk=caption_input, characters=", ".join(characters) or "none"),
            max_new_tokens=60,
        )
        if dataset_path is not None:
            _append_jsonl(
                dataset_path,
                {"input": caption_input, "characters": characters, "target": caption},
            )

        chunk_doc = get_nlp()(chunk) if '"' in chunk else None
        dialogue = split_dialogue(chunk, characters, default_speaker=last_speaker, doc=chunk_doc)
        if characters:
            last_speaker = characters[-1]
        elif dialogue:
            last_speaker = dialogue[-1].speaker
        camera_hint = guess_camera_hint(chunk, len(characters))
        panels.append(Panel(scene_description=caption, characters=characters, camera_hint=camera_hint, dialogue=dialogue))

    report("laying out pages")
    pages = pack_into_pages(panels)
    return Story(id=story_id, title=title, style_prompt=style_prompt, pages=pages)
