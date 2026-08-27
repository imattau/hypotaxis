"""Regression tests for the pure-logic pieces of manga_pipeline/backends.py -
identity-conditioning policy, prop/abstract-character text-anchoring, and the
SDXL panel-size rounding. None of this touches torch/diffusers (DiffusersBackend
only imports those lazily inside _load()), so these run without a GPU or any
model download.
"""

from __future__ import annotations

from manga_pipeline.backends import (
    DiffusersBackend,
    _abstract_character_note,
    _dominant,
    _dominant_character,
    _dominant_location,
    _prop_notes,
    _round_to_8,
)
from manga_pipeline.config import PipelineConfig
from manga_pipeline.registry import CharacterRegistry
from manga_pipeline.schema import Page, Panel


def test_round_to_8_rounds_to_nearest_multiple():
    assert _round_to_8(100) == 96
    assert _round_to_8(101) == 104
    assert _round_to_8(4) == 8  # floor of 8, never rounds down to 0


def test_dominant_picks_most_frequent_name():
    assert _dominant([["Aiko"], ["Aiko", "Ren"], ["Ren"], ["Aiko"]]) == "Aiko"


def test_dominant_returns_none_when_nothing_tagged():
    assert _dominant([[], []]) is None


def test_dominant_character_and_location_use_page_panels():
    page = Page(
        layout="H3",
        panels=[
            Panel(scene_description="a", characters=["Aiko"], locations=["Mill"]),
            Panel(scene_description="b", characters=["Aiko"], locations=["Mill"]),
            Panel(scene_description="c", characters=["Ren"], locations=[]),
        ],
    )
    assert _dominant_character(page) == "Aiko"
    assert _dominant_location(page) == "Mill"


def test_prop_notes_only_includes_panel_tagged_props(tmp_path):
    registry = CharacterRegistry(tmp_path / "props.json")
    registry.set_description("Letter", "a folded letter with a wax seal")
    panel = Panel(scene_description="x", props=["Letter"])
    assert _prop_notes(panel, registry) == "Letter (a folded letter with a wax seal)"

    empty_panel = Panel(scene_description="x", props=[])
    assert _prop_notes(empty_panel, registry) == ""


def test_prop_notes_handles_missing_registry():
    panel = Panel(scene_description="x", props=["Letter"])
    assert _prop_notes(panel, None) == ""


def test_abstract_character_note_formats_name_and_description(tmp_path):
    registry = CharacterRegistry(tmp_path / "chars.json")
    registry.set_description("Nova", "a soft shifting LED glow")
    assert _abstract_character_note("Nova", registry) == "Nova (a soft shifting LED glow)"
    assert _abstract_character_note(None, registry) == ""
    assert _abstract_character_note("Nova", None) == ""


def test_resolve_target_single_name_conditions_on_it():
    backend = DiffusersBackend(PipelineConfig())
    assert backend._resolve_target(["Aiko"], dominant="Ren") == "Aiko"


def test_resolve_target_zero_names_falls_back_to_dominant():
    backend = DiffusersBackend(PipelineConfig())
    assert backend._resolve_target([], dominant="Ren") == "Ren"


def test_resolve_target_multiple_names_skips_conditioning():
    backend = DiffusersBackend(PipelineConfig())
    assert backend._resolve_target(["Aiko", "Ren"], dominant="Ren") is None
