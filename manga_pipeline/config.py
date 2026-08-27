from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PipelineConfig:
    backend: str = "mock"  # mock | diffusers
    checkpoint: str = "stabilityai/sdxl-turbo"
    device: str = "cuda"
    steps: int = 4
    guidance_scale: float = 1.0
    edit_strength: float = 0.55
    page_width: int = 1024
    page_height: int = 1536
    output_dir: str = "output"
    seed: int = 0
    registry_dir: str = "registry"
    use_identity_adapter: bool = True
    identity_adapter_scale: float = 0.6
