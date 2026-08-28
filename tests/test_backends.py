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
    _build_prompt,
    _dominant,
    _dominant_character,
    _dominant_location,
    _prop_notes,
    _round_to_8,
)
from manga_pipeline.character_lora import (
    build_training_caption,
    default_lora_output_dir,
    sanitize_adapter_name,
)
from manga_pipeline.config import PipelineConfig
from manga_pipeline.registry import CharacterEntry, CharacterRegistry
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


def test_build_prompt_includes_camera_hint_between_style_and_scene():
    panel = Panel(scene_description="Aiko stares out the window", camera_hint="close-up")
    assert (
        _build_prompt("manga style, ink wash", panel, None)
        == "manga style, ink wash, close-up, Aiko stares out the window"
    )


def test_build_prompt_includes_camera_hint_without_style_prompt():
    panel = Panel(scene_description="a train pulls into the station", camera_hint="medium shot")
    assert _build_prompt("", panel, None) == "medium shot, a train pulls into the station"


def test_build_prompt_expands_camera_hints_known_to_need_it():
    # found via real generation comparisons: sdxl-turbo mostly ignores the
    # terse two-to-three-word form of these hints at production steps/
    # guidance settings - see _CAMERA_HINT_PROMPT_EXPANSIONS
    panel = Panel(scene_description="they talk in the hallway", camera_hint="wide establishing shot")
    prompt = _build_prompt("", panel, None)
    assert prompt.startswith("wide establishing shot, entire room visible")
    assert prompt.endswith("they talk in the hallway")


def test_build_prompt_leaves_unexpanded_camera_hints_as_is():
    # extreme close-up/close-up/medium shot already render correctly from
    # the terse hint alone, so they're deliberately not in the expansion table
    panel = Panel(scene_description="x", camera_hint="extreme close-up")
    assert _build_prompt("", panel, None) == "extreme close-up, x"


def test_build_prompt_passes_through_unrecognized_camera_hint():
    # a custom/legacy value not in CAMERA_HINTS or the expansion table must
    # not crash - falls through unchanged, same as before this table existed
    panel = Panel(scene_description="x", camera_hint="dutch angle")
    assert _build_prompt("", panel, None) == "dutch angle, x"


def test_build_prompt_appends_prop_notes_after_camera_hint(tmp_path):
    registry = CharacterRegistry(tmp_path / "props.json")
    registry.set_description("Letter", "a folded letter with a wax seal")
    panel = Panel(scene_description="x", camera_hint="close-up", props=["Letter"])
    assert (
        _build_prompt("manga style", panel, registry)
        == "manga style, close-up, x, featuring Letter (a folded letter with a wax seal)"
    )


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


# ---------- character_lora.py pure helpers ----------


def test_sanitize_adapter_name_strips_punctuation():
    assert sanitize_adapter_name("Jules'") == "Jules"
    assert sanitize_adapter_name("Dr. Mina Park") == "Dr._Mina_Park"


def test_sanitize_adapter_name_falls_back_for_empty_input():
    assert sanitize_adapter_name("") == "character"
    assert sanitize_adapter_name("'''") == "character"


def test_build_training_caption_includes_all_parts():
    caption = build_training_caption(
        "Aiko", "shoulder-length black hair, red satchel", "monochrome manga style", "profile view, plain background"
    )
    assert caption == (
        "monochrome manga style, character reference portrait of Aiko, "
        "shoulder-length black hair, red satchel, profile view, plain background"
    )


def test_build_training_caption_omits_empty_parts():
    caption = build_training_caption("Aiko", "", "", "front-facing portrait, plain background")
    assert caption == "character reference portrait of Aiko, front-facing portrait, plain background"


def test_default_lora_output_dir_uses_sanitized_name(tmp_path):
    result = default_lora_output_dir(tmp_path, "rain_letter", "Jules'")
    assert result == tmp_path / "character_loras" / "rain_letter" / "Jules"


# ---------- DiffusersBackend._activate_character_lora ----------


class _FakePipe:
    def __init__(self):
        self.load_calls: list[tuple[str, str]] = []
        self.set_adapters_calls: list[tuple[list[str], list[float]]] = []
        self.disable_calls = 0

    def load_lora_weights(self, path, adapter_name):
        self.load_calls.append((path, adapter_name))

    def set_adapters(self, names, adapter_weights):
        self.set_adapters_calls.append((names, adapter_weights))

    def disable_lora(self):
        self.disable_calls += 1


def test_activate_character_lora_noop_when_feature_disabled(tmp_path):
    lora_dir = tmp_path / "aiko_lora"
    lora_dir.mkdir()
    cfg = PipelineConfig(use_character_lora=False)
    backend = DiffusersBackend(cfg)
    backend._base_pipe = _FakePipe()
    entry = CharacterEntry(name="Aiko", lora_path=str(lora_dir))

    backend._activate_character_lora("Aiko", entry)

    assert backend._base_pipe.load_calls == []
    assert backend._base_pipe.set_adapters_calls == []


def test_activate_character_lora_noop_when_no_lora_path(tmp_path):
    cfg = PipelineConfig(use_character_lora=True)
    backend = DiffusersBackend(cfg)
    backend._base_pipe = _FakePipe()
    entry = CharacterEntry(name="Aiko")  # lora_path defaults to ""

    backend._activate_character_lora("Aiko", entry)

    assert backend._base_pipe.load_calls == []


def test_activate_character_lora_noop_when_path_missing_on_disk():
    cfg = PipelineConfig(use_character_lora=True)
    backend = DiffusersBackend(cfg)
    backend._base_pipe = _FakePipe()
    entry = CharacterEntry(name="Aiko", lora_path="/nonexistent/path")

    backend._activate_character_lora("Aiko", entry)

    assert backend._base_pipe.load_calls == []


def test_activate_character_lora_loads_and_activates_once(tmp_path):
    lora_dir = tmp_path / "aiko_lora"
    lora_dir.mkdir()
    cfg = PipelineConfig(use_character_lora=True, character_lora_scale=0.75)
    backend = DiffusersBackend(cfg)
    backend._base_pipe = _FakePipe()
    entry = CharacterEntry(name="Aiko", lora_path=str(lora_dir))

    backend._activate_character_lora("Aiko", entry)
    backend._activate_character_lora("Aiko", entry)  # second call: already active, no duplicate work

    assert backend._base_pipe.load_calls == [(str(lora_dir), "Aiko")]
    assert backend._base_pipe.set_adapters_calls == [(["Aiko"], [0.75])]


def test_activate_character_lora_disables_when_switching_to_no_lora(tmp_path):
    lora_dir = tmp_path / "aiko_lora"
    lora_dir.mkdir()
    cfg = PipelineConfig(use_character_lora=True)
    backend = DiffusersBackend(cfg)
    backend._base_pipe = _FakePipe()
    entry = CharacterEntry(name="Aiko", lora_path=str(lora_dir))

    backend._activate_character_lora("Aiko", entry)
    backend._activate_character_lora(None, None)

    assert backend._base_pipe.disable_calls == 1
