from __future__ import annotations

from dataclasses import dataclass
import os
import tempfile
from pathlib import Path


def resolve_device(device: str) -> str:
    """Resolve the portable ``auto`` setting without importing torch at startup."""
    if device != "auto":
        return device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def atomic_write_text(path: str | Path, content: str) -> None:
    """Write a text file without leaving a truncated destination on failure."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent, text=True)
    try:
        with os.fdopen(fd, "w") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


@dataclass
class PipelineConfig:
    backend: str = "mock"  # mock | diffusers
    # base SDXL (not the turbo distillation) with real classifier-free
    # guidance: A/B tested against sdxl-turbo (checkpoint=stabilityai/sdxl-turbo,
    # steps=4, guidance_scale=1.0) on the same page of a real manuscript -
    # turbo's near-zero guidance_scale is what makes it fast, but it also
    # means the text prompt barely competes with the base model's own prior,
    # and tight framings (close-up/extreme close-up) kept drifting to a
    # photorealistic look regardless of an explicit manga style_prompt (see
    # story_adapt.py/bubbles.py history). 30 steps at guidance_scale=7.0
    # produced consistently on-style output across every panel in that same
    # test page, at roughly 2x the per-panel generation time - a page took
    # ~9-10s/panel instead of ~4-5s/panel, not the naive 7-8x steps ratio
    # would suggest, since fixed per-call overhead dominates less at scale.
    checkpoint: str = "stabilityai/stable-diffusion-xl-base-1.0"
    device: str = "auto"
    steps: int = 30
    guidance_scale: float = 7.0
    page_width: int = 1024
    page_height: int = 1536
    output_dir: str = "output"
    seed: int = 0
    registry_dir: str = "registry"
    use_identity_adapter: bool = True
    identity_adapter_scale: float = 0.6
    # per-character LoRA (see manga_pipeline/character_lora.py,
    # train_character_lora.py) - on by default, but a no-op for any
    # character without a separately trained adapter: _activate_character_lora
    # checks the registry's lora_path exists on disk before doing anything,
    # so this flag alone changes nothing for a story with none trained yet
    # (as of this default flip, none of the soft_reset* stories have any).
    # Complements identity_adapter rather than replacing it: IP-Adapter above
    # stays a lightweight always-available fallback, while a trained LoRA is
    # a stronger, character-specific identity signal once someone has
    # invested the training time for that character.
    use_character_lora: bool = True
    character_lora_scale: float = 0.8
    # Optional verified adapter-bank composition manifest. When set, this
    # takes precedence over the per-character LoRA for generated panels.
    adapter_composition_path: str = ""
    # pose-ControlNet conditioning for multi-character panels (see
    # manga_pipeline/pose_skeleton.py). Found via real generation comparisons
    # that no amount of "exactly two people" prompt wording reliably stopped
    # SDXL from dropping or duplicating figures in a panel with 2+ tagged
    # characters (see DiffusersBackend._resolve_target: identity
    # conditioning is already skipped entirely for such panels, since
    # blending multiple identities is a separate unsolved problem) - a
    # synthetic pose skeleton with exactly the right figure count fixed that
    # specific failure reliably where prompt tuning couldn't. On by default
    # since two blockers that previously kept this opt-in are now fixed: the
    # ControlNet checkpoint (xinsir/controlnet-openpose-sdxl-1.0, see
    # backends.py) loads with use_safetensors=True enforced rather than
    # silently falling back to an unsafe pickle checkpoint, and generate_panel
    # now skips this path on close-up/extreme-close-up panels, where a
    # full-body pose skeleton produced a distant, awkward composition that
    # ignored the panel's own tight framing. It also no longer carries a
    # distinct step-count cost - cfg.checkpoint's default (base SDXL,
    # steps=30) already matches what this path needs, so what's left is a
    # second pipeline's worth of one-time load/VRAM overhead per story, not a
    # per-panel multiplier.
    #
    # Still an open problem, not fixed by this path: which figure gets which
    # character's look. The pose skeleton only guarantees headcount/position;
    # a real comparison run swapped gender/identity between the two figures.
    # pose_controlnet_scale=0.5 was the better of two tested values (0.5 vs
    # 0.65): strong enough to lock headcount/position, loose enough that the
    # text prompt still drives what it can of each figure's look rather than
    # the model defaulting both figures to the same appearance.
    use_pose_controlnet: bool = True
    pose_controlnet_scale: float = 0.5
