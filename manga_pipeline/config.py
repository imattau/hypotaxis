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
    checkpoint: str = "stabilityai/sdxl-turbo"
    device: str = "auto"
    steps: int = 4
    guidance_scale: float = 1.0
    page_width: int = 1024
    page_height: int = 1536
    output_dir: str = "output"
    seed: int = 0
    registry_dir: str = "registry"
    use_identity_adapter: bool = True
    identity_adapter_scale: float = 0.6
    # opt-in per-character LoRA (see manga_pipeline/character_lora.py,
    # train_character_lora.py) - off by default since it requires a
    # separately trained adapter per character and changes nothing for a
    # story with none trained yet. Complements identity_adapter rather than
    # replacing it: IP-Adapter above stays a lightweight always-available
    # fallback, while a trained LoRA is a stronger, character-specific
    # identity signal once someone has invested the training time.
    use_character_lora: bool = False
    character_lora_scale: float = 0.8
