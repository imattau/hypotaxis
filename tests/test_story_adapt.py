"""Regression tests for the pure-logic pieces of Stage A (manga_pipeline/story_adapt.py).

These lock in a series of real bugs found and fixed while testing the pipeline
against an actual manuscript ("Soft Reset") - see the project's memory/commit
history for the failure cases these were pulled from. Anything requiring the
sentence-embedding model (segment_text, score_sentence_importance) is left
untested here since it needs a network download; everything below only needs
spaCy's en_core_web_sm, which the project already requires locally.
"""

from __future__ import annotations

from manga_pipeline.captioner import build_captioner_source
from manga_pipeline.story_adapt import (
    _caption_input,
    _merge_person_aliases,
    _normalize_quotes,
    _split_on_scene_breaks,
    _strip_markdown_structure,
    get_nlp,
    names_in_chunk,
    pack_into_pages,
    parse_character_profiles,
    parse_location_profiles,
    split_dialogue,
    split_sentences,
)
from manga_pipeline.schema import DialogueLine, Panel
from manga_pipeline.train_captioner import _sanitize_caption, guess_camera_hint, parse_caption_and_camera


def test_caption_input_appends_known_appearances_when_present():
    assert _caption_input("Jules walked in.", "Jules (auburn hair, leather jacket)") == (
        "Jules walked in.\nKnown appearances: Jules (auburn hair, leather jacket)"
    )


def test_caption_input_omits_known_appearances_when_absent():
    # no character_notes (e.g. a chunk with no recognized characters) -
    # must match what train_captioner.py's records look like for such rows,
    # or a trained captioner sees an input shape it never saw in training
    assert _caption_input("Rain fell on the empty street.", "") == "Rain fell on the empty street."


def test_captioner_source_matches_training_format():
    # this must stay byte-for-byte identical to train_captioner.py's
    # to_examples() - see manga_pipeline/captioner.py's build_captioner_source
    # docstring for why a mismatch here silently degrades captioner quality
    # instead of erroring
    source = build_captioner_source("Jules walked in.", ["Jules", "Priya"])
    assert source == "caption: characters: Jules, Priya\nJules walked in."


def test_captioner_source_uses_none_for_no_characters():
    assert build_captioner_source("Rain fell.", []) == "caption: characters: none\nRain fell."


def test_split_sentences_does_not_split_on_title_abbreviation():
    # regression: "Dr. Osei jogged..." used to fragment into "...as Dr."
    # + "Osei jogged..." because the naive period-based splitter treated
    # the abbreviation's period as a sentence boundary - found via a real
    # generated caption for the orphaned "Dr." fragment coming out useless
    text = "Fluorescent lights buzzed overhead as Dr. Osei jogged down the hallway. She reached bed six."
    assert split_sentences(text) == [
        "Fluorescent lights buzzed overhead as Dr. Osei jogged down the hallway.",
        "She reached bed six.",
    ]


def test_split_sentences_still_splits_normal_sentences():
    text = "The rain fell. Kessa walked home."
    assert split_sentences(text) == ["The rain fell.", "Kessa walked home."]


def test_normalize_quotes_converts_curly_to_straight():
    text = "“Good morning, Jules,” Nova said. It’s a fine day."
    normalized = _normalize_quotes(text)
    assert '"Good morning, Jules,"' in normalized
    assert "It's a fine day." in normalized
    assert "“" not in normalized and "’" not in normalized


def test_normalize_quotes_is_idempotent_on_plain_text():
    text = 'Plain "already straight" text.'
    assert _normalize_quotes(text) == text


def test_merge_person_aliases_collapses_ner_variants():
    names = ["Dr. Mina Park", "Mina Park's", "Park", "Jules", "Nova"]
    canonical, aliases = _merge_person_aliases(names)
    assert canonical == ["Dr. Mina Park", "Jules", "Nova"]
    assert aliases == {"Mina Park": "Dr. Mina Park", "Park": "Dr. Mina Park"}


def test_merge_person_aliases_prefers_priority_name_over_longer_ner_form():
    # spaCy's PERSON span usually drops a leading title ("Dr."), so the raw
    # NER hit ("Mina Park") is shorter than the profile-supplied canonical
    # name ("Dr. Mina Park") - the profile name must still win.
    names = ["Mina Park's", "Park"]
    canonical, aliases = _merge_person_aliases(names, priority_names=["Dr. Mina Park"])
    assert canonical == ["Dr. Mina Park"]
    assert aliases == {"Mina Park": "Dr. Mina Park", "Park": "Dr. Mina Park"}


def test_merge_person_aliases_no_false_merge_for_unrelated_names():
    canonical, aliases = _merge_person_aliases(["Aiko", "Ren"])
    assert canonical == ["Aiko", "Ren"]
    assert aliases == {}


def test_strip_markdown_structure_removes_headers_and_blockquote_markers():
    text = "# Chapter 1: Soft Reset\n\nThe day began.\n\n> A note from Nova.\n"
    stripped = _strip_markdown_structure(text)
    assert "# Chapter" not in stripped
    assert ">" not in stripped
    assert "The day began." in stripped
    assert "A note from Nova." in stripped


def test_strip_markdown_structure_leaves_scene_breaks_intact():
    # scene breaks are handled separately by _split_on_scene_breaks - they
    # must survive this pass as a boundary signal, not get deleted here
    text = "First scene.\n\n---\n\nSecond scene."
    assert "---" in _strip_markdown_structure(text)


def test_split_on_scene_breaks_separates_sections():
    text = "First scene.\n\n---\n\nSecond scene.\n\n***\n\nThird scene."
    assert _split_on_scene_breaks(text) == ["First scene.", "Second scene.", "Third scene."]


def test_split_on_scene_breaks_no_break_returns_one_section():
    assert _split_on_scene_breaks("Just one scene, no breaks.") == ["Just one scene, no breaks."]


def test_split_on_scene_breaks_does_not_match_prose_dashes():
    # a real sentence using dashes for punctuation (not a standalone
    # scene-break line) must not get treated as a scene boundary
    text = "It was—unexpectedly—quiet. The end."
    assert _split_on_scene_breaks(text) == ["It was—unexpectedly—quiet. The end."]


def test_names_in_chunk_substring_match():
    assert names_in_chunk("Kessa walked to the Mill.", ["Kessa", "Ren", "Mill"]) == ["Kessa", "Mill"]


def test_parse_character_profiles_strips_no_form_tag():
    text = "Nova: [no-form] no physical body; visualise only as light\nJules: young woman, dark hair"
    profiles, abstract_names = parse_character_profiles(text)
    assert profiles["Nova"] == "no physical body; visualise only as light"
    assert profiles["Jules"] == "young woman, dark hair"
    assert abstract_names == {"Nova"}


def test_parse_character_profiles_ignores_comments_and_blank_lines():
    text = "# a comment\n\nAiko: young woman, black bob haircut\n"
    profiles, abstract_names = parse_character_profiles(text)
    assert profiles == {"Aiko": "young woman, black bob haircut"}
    assert abstract_names == set()


def test_parse_location_profiles_shares_generic_format():
    profiles = parse_location_profiles("Mill: old wooden watermill beside a river")
    assert profiles == {"Mill": "old wooden watermill beside a river"}


def test_sanitize_caption_strips_leaked_prompt_instructions():
    # reproduces the exact malformed response a real 3B-model run produced
    # on a real manuscript panel - the model echoed its own instructions
    # back as if they were caption content
    bad = (
        "EXT. WIDE ESTABLISHING SHOT: A room bathed in silence, the air heavy with "
        "anticipation following a soft pause. One sentence, under 25 words, describing "
        "only the setting, action, and expression stated in the passage. Do not invent "
        "objects, locations, or events not in the passage. Do not include dialogue. "
        "Do not add characters not listed."
    )
    cleaned = _sanitize_caption(bad)
    assert cleaned == "A room bathed in silence, the air heavy with anticipation following a soft pause."
    assert "do not invent" not in cleaned.lower()
    assert "EXT." not in cleaned


def test_sanitize_caption_leaves_clean_caption_untouched():
    clean = "Jules stares out the window, her expression distant."
    assert _sanitize_caption(clean) == clean


def test_sanitize_caption_never_returns_empty():
    # if the leak marker is at the very start, there's no sentence left to
    # back up to - must fall back to the original text rather than produce
    # an empty caption that would break Panel/generation downstream
    assert _sanitize_caption("Do not invent anything at all.") != ""


def test_parse_caption_and_camera_well_formed():
    response = "CAPTION: Jules stares out the window.\nCAMERA: close-up"
    caption, camera = parse_caption_and_camera(response, "chunk text", 1)
    assert caption == "Jules stares out the window."
    assert camera == "close-up"


def test_parse_caption_and_camera_missing_caption_prefix():
    # a 3B model doesn't always follow the two-line format exactly - an
    # unprefixed line must still be treated as caption text, not dropped
    response = "Jules stares out the window.\nCAMERA: wide two-shot"
    caption, camera = parse_caption_and_camera(response, "chunk text", 2)
    assert caption == "Jules stares out the window."
    assert camera == "wide two-shot"


def test_parse_caption_and_camera_unknown_camera_falls_back_to_heuristic():
    response = "CAPTION: Jules stares out the window.\nCAMERA: dutch angle drone shot"
    caption, camera = parse_caption_and_camera(response, "she stood in the room", 1)
    assert caption == "Jules stares out the window."
    assert camera == guess_camera_hint("she stood in the room", 1)


def test_parse_caption_and_camera_missing_camera_line_falls_back_to_heuristic():
    response = "CAPTION: Jules stares out the window."
    caption, camera = parse_caption_and_camera(response, "she stood in the room", 1)
    assert caption == "Jules stares out the window."
    assert camera == guess_camera_hint("she stood in the room", 1)


def test_guess_camera_hint_close_up_keyword():
    assert guess_camera_hint("She looked at her hands.", 1) == "close-up"


def test_guess_camera_hint_two_shot_for_multiple_characters():
    assert guess_camera_hint("They stood together in the yard.", 2) == "wide two-shot"


def test_pack_into_pages_assigns_supported_layout_sizes():
    panels = [Panel(scene_description=f"panel {i}") for i in range(5)]
    pages = pack_into_pages(panels)
    total_panels = sum(len(p.panels) for p in pages)
    assert total_panels == 5
    for page in pages:
        assert len(page.panels) in (2, 3)


def test_pack_into_pages_rejects_too_short_story():
    try:
        pack_into_pages([Panel(scene_description="only one")])
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_split_dialogue_reporting_verb_in_own_sentence():
    chunk = 'Aiko smiled. "Hello there."'
    doc = get_nlp()(chunk)
    lines = split_dialogue(chunk, ["Aiko", "Ren"], default_speaker="Aiko", doc=doc)
    assert [l.speaker for l in lines] == ["Aiko"]


def test_split_dialogue_prefers_reporting_verb_subject_over_nearest_name():
    # the failure mode a nearest-name heuristic gets wrong: Ren is the
    # speaker (subject of "called out"), but Aiko's name sits textually
    # closer to the quote.
    chunk = "Ren spotted her across the platform and called out. \"Aiko?\""
    doc = get_nlp()(chunk)
    lines = split_dialogue(chunk, ["Aiko", "Ren"], default_speaker="Ren", doc=doc)
    assert [l.speaker for l in lines] == ["Ren"]


def test_split_dialogue_action_beat_before_bare_quote():
    # "shrugged" is not a reporting verb, so this only resolves via the
    # adjacent-sentence fallback, not the same-sentence reporting-verb check.
    chunk = 'Jules shrugged. "I dreamed."'
    doc = get_nlp()(chunk)
    lines = split_dialogue(chunk, ["Jules", "Nova"], default_speaker="Nova", doc=doc)
    assert [l.speaker for l in lines] == ["Jules"]


def test_split_dialogue_possessive_subject_of_action_beat():
    # subject of "hovered" is "face" (a NOUN), with "Park" attached as a
    # possessive modifier - the sentence is about the possessor, not the
    # body part.
    chunk = "Dr. Mina Park's face hovered in the corner of Jules' screen. \"Any changes in sleep?\""
    doc = get_nlp()(chunk)
    lines = split_dialogue(chunk, ["Jules", "Dr. Mina Park"], default_speaker="Jules", doc=doc)
    assert [l.speaker for l in lines] == ["Dr. Mina Park"]


def test_split_dialogue_full_exchange_matches_ground_truth():
    """The exact 10-line Dr. Park / Jules exchange from the real manuscript
    that originally triggered the speaker-attribution investigation - this
    is the end-to-end regression test for all three bugs fixed together."""
    chunk = _normalize_quotes(
        "Later, Dr. Mina Park’s face hovered in the corner of Jules’ screen. "
        '"Any changes in sleep? Mood?"\n\n'
        'Jules shrugged. "I dreamed."\n\n'
        '"That’s a change."\n\n'
        '"I think Nova’s been weird."\n\n'
        '"Weird?"\n\n'
        '"She… she wrote a poem. Sort of."\n\n'
        'Dr. Park smiled faintly. "Nova is programmed for linguistic adaptation. '
        'Maybe she’s responding to your art style."\n\n'
        '"Maybe." Jules hesitated. "It didn’t feel programmed."\n\n'
        '"Then perhaps she’s learning you."'
    )
    doc = get_nlp()(chunk)
    characters = ["Jules", "Nova", "Dr. Mina Park"]
    lines = split_dialogue(chunk, characters, default_speaker="Jules", doc=doc)
    assert [l.speaker for l in lines] == [
        "Dr. Mina Park",
        "Jules",
        "Dr. Mina Park",
        "Jules",
        "Dr. Mina Park",
        "Jules",
        "Dr. Mina Park",
        "Jules",
        "Jules",
        "Dr. Mina Park",
    ]


def test_split_dialogue_window_does_not_bleed_into_prior_quote_content():
    # "Nova" appears inside Jules' own spoken line here - it must not hijack
    # attribution for the very next (unrelated) quote.
    chunk = '"I think Nova’s been weird." "Weird?"'
    doc = get_nlp()(chunk)
    lines = split_dialogue(chunk, ["Jules", "Nova", "Dr. Mina Park"], default_speaker="Jules", doc=doc)
    # "Weird?" has no resolvable signal at all here (no beat, no reporting
    # verb, and the window must not pick up "Nova" from inside the prior
    # quote) - it should fall back to a genuine "no signal" default
    # (last_speaker), not be hijacked into "Nova".
    assert lines[1].speaker != "Nova"


def test_split_dialogue_detects_italicized_thought():
    chunk = "Jules almost said, *Or loving me.* But she didn't."
    doc = get_nlp()(chunk)
    lines = split_dialogue(chunk, ["Jules", "Nova"], default_speaker="Nova", doc=doc)
    assert lines == [DialogueLine(speaker="Jules", text="Or loving me.", kind="thought")]


def test_split_dialogue_thought_verb_with_propn_subject():
    chunk = "Jules thought, *This can't be real.*"
    doc = get_nlp()(chunk)
    lines = split_dialogue(chunk, ["Jules", "Nova"], default_speaker="Nova", doc=doc)
    assert lines == [DialogueLine(speaker="Jules", text="This can't be real.", kind="thought")]


def test_split_dialogue_does_not_mistake_bold_for_a_thought():
    chunk = "This is **bold** text, not a thought."
    doc = get_nlp()(chunk)
    assert split_dialogue(chunk, [], doc=doc) == []


def test_split_dialogue_italicized_quote_is_not_double_counted():
    # italics wrapped *around* a quote is emphasis on spoken dialogue, not
    # a separate unspoken thought - this must produce exactly one line, not
    # a duplicate "speech" + "thought" pair for the same text
    chunk = _normalize_quotes("Nova’s presence quivered like light. *“I’ve always seen you.”*")
    doc = get_nlp()(chunk)
    lines = split_dialogue(chunk, ["Jules", "Nova"], default_speaker="Jules", doc=doc)
    assert len(lines) == 1
    assert lines[0].kind == "speech"
    assert lines[0].text == "I've always seen you."


def test_split_dialogue_interleaves_speech_and_thought_by_position():
    chunk = 'Aiko whispered, "Are you there?" *He never answers,* she thought.'
    doc = get_nlp()(chunk)
    lines = split_dialogue(chunk, ["Aiko", "Ren"], default_speaker="Aiko", doc=doc)
    assert [l.kind for l in lines] == ["speech", "thought"]
    assert lines[0].text == "Are you there?"
    assert lines[1].text == "He never answers,"
