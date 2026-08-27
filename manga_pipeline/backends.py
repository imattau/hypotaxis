from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod

from PIL import Image, ImageDraw

from pathlib import Path

from .config import PipelineConfig
from .fonts import load_font, wrap_to_width
from .registry import CharacterRegistry
from .schema import Panel, Page


def _seed_for(story_id: str, page_index: int, panel_index: int = -1) -> int:
    key = f"{story_id}:{page_index}:{panel_index}"
    return int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)


class ImageBackend(ABC):
    def prepare_characters(self, story_id: str, registry: CharacterRegistry, style_prompt: str) -> None:
        """Generate/refresh any per-character identity assets (Stage B) up
        front, right after story adaptation and before any page is
        generated - rather than lazily on first appearance during Stage C.
        This guarantees every character in the registry gets a reference
        regardless of whether they happen to first appear in a solo panel,
        and keeps Stage C's per-panel loop free of first-use branching.
        Default no-op; MockBackend has nothing to prepare.
        """

    @abstractmethod
    def generate_base(
        self, story_id: str, page_index: int, page: Page, size: tuple[int, int], style_prompt: str
    ) -> Image.Image: ...

    @abstractmethod
    def edit_panel(
        self,
        base: Image.Image,
        story_id: str,
        page_index: int,
        panel_index: int,
        panel: Panel,
        size: tuple[int, int],
        style_prompt: str,
        registry: CharacterRegistry | None,
    ) -> Image.Image: ...


class MockBackend(ImageBackend):
    """No-download, no-GPU stand-in so the layout/assembly/bubble stages
    can be validated on any machine before wiring up a real diffusion model.
    """

    def generate_base(
        self, story_id: str, page_index: int, page: Page, size: tuple[int, int], style_prompt: str
    ) -> Image.Image:
        seed = _seed_for(story_id, page_index)
        hue = seed % 360
        color = _hsv_to_rgb(hue, 0.25, 0.85)
        return Image.new("RGB", size, color)

    def edit_panel(
        self,
        base: Image.Image,
        story_id: str,
        page_index: int,
        panel_index: int,
        panel: Panel,
        size: tuple[int, int],
        style_prompt: str,
        registry: CharacterRegistry | None,
    ) -> Image.Image:
        seed = _seed_for(story_id, page_index, panel_index)
        hue = seed % 360
        color = _hsv_to_rgb(hue, 0.45, 0.9)
        img = Image.new("RGB", size, color)
        draw = ImageDraw.Draw(img)
        font = load_font(max(10, min(size) // 18))
        label = f"[{panel.camera_hint}] " + panel.scene_description
        wrapped = wrap_to_width(draw, label, font, size[0] - 20)
        text_h = draw.multiline_textbbox((0, 0), wrapped, font=font)[3]
        draw.multiline_text((10, size[1] - text_h - 10), wrapped, fill=(20, 20, 20), font=font)
        return img


class DiffusersBackend(ImageBackend):
    """Real generation backend. Imports torch/diffusers lazily so the mock
    backend keeps working on machines without them installed.

    edit_panel uses a generic img2img pass as a placeholder for a proper
    multi-panel edit model (e.g. Qwen-Edit); swap in a purpose-built editor
    once one is wired in.

    Identity adapter (Stage B/Phase 4): when a panel has exactly one named
    character, condition edit_panel on a persistent per-character reference
    image via IP-Adapter, generated once on first appearance and reused for
    every later panel/page. This targets the specific gap Phase 2 testing
    found - text-only character descriptions were not enough to keep a
    character's appearance consistent across pages. Panels with zero or
    multiple characters fall back to text-only conditioning (IP-Adapter
    scale 0) since blending multiple identities into one panel is a harder
    problem left for a future pass.
    """

    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        self._base_pipe = None
        self._edit_pipe = None
        self._ip_adapter_loaded = False
        self._neutral_ip_image = None

    def _load(self):
        if self._base_pipe is not None:
            return
        import torch
        from diffusers import AutoPipelineForImage2Image, AutoPipelineForText2Image

        dtype = torch.float16 if self.cfg.device.startswith("cuda") else torch.float32
        variant = "fp16" if dtype is torch.float16 else None
        try:
            self._base_pipe = AutoPipelineForText2Image.from_pretrained(
                self.cfg.checkpoint, torch_dtype=dtype, variant=variant
            )
        except (OSError, ValueError):
            # checkpoint has no fp16-variant weights published; fall back to default
            self._base_pipe = AutoPipelineForText2Image.from_pretrained(self.cfg.checkpoint, torch_dtype=dtype)
        # keep VRAM usage low for modest-hardware targets and shared/contended GPUs
        self._base_pipe.vae.enable_slicing()
        self._base_pipe.vae.enable_tiling()

        if self.cfg.use_identity_adapter:
            # load before from_pipe() so the shared unet/image_encoder carry the adapter over;
            # works fine before device placement since it's just loading state dicts
            self._base_pipe.load_ip_adapter("h94/IP-Adapter", subfolder="sdxl_models", weight_name="ip-adapter_sdxl.bin")
            self._base_pipe.set_ip_adapter_scale(0.0)
            self._ip_adapter_loaded = True
            # enable_attention_slicing() unconditionally overwrites every cross-attention
            # processor with a plain SlicedAttnProcessor, which clobbers the IP-Adapter-aware
            # processors load_ip_adapter() installs - the two are mutually exclusive here.
            # The identity adapter's extra CLIP vision encoder (~3.7GB) is enough on its own
            # to reintroduce the VAE-decode OOM the slicing/tiling above was fixing, so trade
            # speed for memory instead: keep only the actively-computing submodule on GPU.
            # enable_model_cpu_offload() manages device placement itself - don't call .to() first.
            if self.cfg.device.startswith("cuda"):
                self._base_pipe.enable_model_cpu_offload(device=self.cfg.device)
            else:
                self._base_pipe.to(self.cfg.device)
        else:
            self._base_pipe.to(self.cfg.device)
            self._base_pipe.enable_attention_slicing()

        # share weights with the base pipeline instead of loading a second full copy
        self._edit_pipe = AutoPipelineForImage2Image.from_pipe(self._base_pipe)

    def _character_dir(self, story_id: str) -> Path:
        path = Path(self.cfg.output_dir) / story_id / "characters"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _neutral_image(self):
        if self._neutral_ip_image is None:
            self._neutral_ip_image = Image.new("RGB", (224, 224), (128, 128, 128))
        return self._neutral_ip_image

    def _ip_kwargs(self, pipe, image: Image.Image | None, scale: float) -> dict:
        # once IP-Adapter is loaded, every call on this unet must supply an
        # ip_adapter_image - a neutral placeholder at scale=0.0 is a
        # mathematically-inert stand-in for "no identity conditioning here"
        if not self._ip_adapter_loaded:
            return {}
        pipe.set_ip_adapter_scale(scale)
        return {"ip_adapter_image": image if image is not None else self._neutral_image()}

    def prepare_characters(self, story_id: str, registry: CharacterRegistry, style_prompt: str) -> None:
        if not self.cfg.use_identity_adapter:
            return
        self._load()
        for name in registry.all():
            self._reference_image(name, story_id, style_prompt, registry)

    def _reference_image(self, name: str, story_id: str, style_prompt: str, registry: CharacterRegistry):
        from PIL import Image as PILImage

        entry = registry.get(name)
        if entry and entry.reference_image and Path(entry.reference_image).exists():
            return PILImage.open(entry.reference_image).convert("RGB")

        description = entry.description if entry else name
        prompt = f"{style_prompt}, character reference portrait of {name}, {description}, plain background, front-facing"
        import torch

        generator = torch.Generator(device=self.cfg.device).manual_seed(
            _seed_for(f"{story_id}:character", 0, hash(name) % 1000)
        )
        result = self._base_pipe(
            prompt=prompt,
            num_inference_steps=self.cfg.steps,
            guidance_scale=self.cfg.guidance_scale,
            width=512,
            height=512,
            generator=generator,
            **self._ip_kwargs(self._base_pipe, None, 0.0),
        )
        image = result.images[0]
        path = self._character_dir(story_id) / f"{name}.png"
        image.save(path)
        registry.set_reference_image(name, str(path))
        return image

    def generate_base(
        self, story_id: str, page_index: int, page: Page, size: tuple[int, int], style_prompt: str
    ) -> Image.Image:
        self._load()
        import torch

        seed = _seed_for(story_id, page_index)
        generator = torch.Generator(device=self.cfg.device).manual_seed(seed)
        scene = page.panels[0].scene_description if page.panels else "manga page"
        prompt = f"{style_prompt}, {scene}" if style_prompt else scene
        result = self._base_pipe(
            prompt=prompt,
            num_inference_steps=self.cfg.steps,
            guidance_scale=self.cfg.guidance_scale,
            width=size[0],
            height=size[1],
            generator=generator,
            **self._ip_kwargs(self._base_pipe, None, 0.0),
        )
        return result.images[0]

    def edit_panel(
        self,
        base: Image.Image,
        story_id: str,
        page_index: int,
        panel_index: int,
        panel: Panel,
        size: tuple[int, int],
        style_prompt: str,
        registry: CharacterRegistry | None,
    ) -> Image.Image:
        self._load()
        import torch

        seed = _seed_for(story_id, page_index, panel_index)
        generator = torch.Generator(device=self.cfg.device).manual_seed(seed)
        prompt = f"{style_prompt}, {panel.scene_description}" if style_prompt else panel.scene_description

        if registry is not None and len(panel.characters) == 1:
            ref_image = self._reference_image(panel.characters[0], story_id, style_prompt, registry)
            ip_kwargs = self._ip_kwargs(self._edit_pipe, ref_image, self.cfg.identity_adapter_scale)
        else:
            ip_kwargs = self._ip_kwargs(self._edit_pipe, None, 0.0)

        result = self._edit_pipe(
            prompt=prompt,
            image=base.resize(size),
            strength=self.cfg.edit_strength,
            num_inference_steps=self.cfg.steps,
            guidance_scale=self.cfg.guidance_scale,
            generator=generator,
            **ip_kwargs,
        )
        return result.images[0]


def _hsv_to_rgb(h: float, s: float, v: float) -> tuple[int, int, int]:
    import colorsys

    r, g, b = colorsys.hsv_to_rgb((h % 360) / 360, s, v)
    return int(r * 255), int(g * 255), int(b * 255)


def build_backend(cfg: PipelineConfig) -> ImageBackend:
    if cfg.backend == "mock":
        return MockBackend()
    if cfg.backend == "diffusers":
        return DiffusersBackend(cfg)
    raise ValueError(f"unknown backend: {cfg.backend}")
