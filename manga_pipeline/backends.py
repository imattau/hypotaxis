from __future__ import annotations

import hashlib
import json
import re
from abc import ABC, abstractmethod

from PIL import Image, ImageDraw

from pathlib import Path

from .character_lora import TRAINING_VIEW_PROMPTS, sanitize_adapter_name
from .adapter_distribution import resolve_composition_paths
from .config import PipelineConfig, resolve_device
from .fonts import load_font, wrap_to_width
from .pose_skeleton import multi_person_skeleton
from .registry import CharacterRegistry
from .schema import Panel, Page

# pose-ControlNet conditioning for multi-character panels (see
# PipelineConfig.use_pose_controlnet, DiffusersBackend._generate_with_pose_controlnet).
# This ControlNet checkpoint is trained against full SDXL base, not the
# lighter sdxl-turbo checkpoint the rest of this backend defaults to -
# loaded as a separate pipeline (_load_pose_pipe) rather than swapping
# cfg.checkpoint's model for the whole story.
_POSE_CONTROLNET_MODEL = "thibaud/controlnet-openpose-sdxl-1.0"
_POSE_CONTROLNET_BASE_CHECKPOINT = "stabilityai/stable-diffusion-xl-base-1.0"
# found via real generation comparisons, not guessed: full SDXL base's
# normal (non-turbo) steps/guidance was needed for legible in-style output -
# a first attempt with a lower step count and no negative prompt produced
# dark, underexposed silhouettes instead of visible characters
_POSE_CONTROLNET_STEPS = 30
_POSE_CONTROLNET_GUIDANCE_SCALE = 8.5
_POSE_CONTROLNET_NEGATIVE_PROMPT = (
    "silhouette, dark, underexposed, shadow figures, black shapes, photorealistic, photo, "
    "color, blurry, extra limbs, extra people, deformed, low contrast"
)


def _seed_for(story_id: str, page_index: int, panel_index: int = -1) -> int:
    key = f"{story_id}:{page_index}:{panel_index}"
    return int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)


def _dominant(names_per_panel: list[list[str]]) -> str | None:
    """Name appearing in the most panels, or None if there are none at all.
    Shared by _dominant_character and _dominant_location."""
    counts: dict[str, int] = {}
    for names in names_per_panel:
        for name in names:
            counts[name] = counts.get(name, 0) + 1
    if not counts:
        return None
    return max(counts, key=counts.get)


def _dominant_character(page: Page) -> str | None:
    """Character appearing in the most panels on this page, or None if the
    page has no characters at all. Used to identity-condition the page's
    base image itself, not just the per-panel edits."""
    return _dominant([panel.characters for panel in page.panels])


def _dominant_location(page: Page) -> str | None:
    """Same idea as _dominant_character, but for locations - only ever
    populated from author-supplied location profiles, since there's no
    automatic detection for generic-but-important settings."""
    return _dominant([panel.locations for panel in page.panels])


def _prop_notes(panel: Panel, prop_registry: CharacterRegistry | None) -> str:
    """Text-anchoring for props: unlike characters/locations, props are not
    image-conditioned via IP-Adapter (see parse_prop_profiles for why - a
    'wide establishing shot' reference for a small object produces a
    cluttered unusable scene, and whole-image conditioning on a close-up
    object shot would distort a panel's entire composition around one
    incidental item). Instead, whenever Stage A tagged a prop on this
    specific panel, its description is woven directly into the generation
    prompt, the same trick Stage A already uses for character appearance
    notes. No dominant-page fallback the way characters/locations get one:
    over-including a prop's description on panels that don't actually
    contain it would introduce stray, wrong mentions of the object.
    """
    if prop_registry is None or not panel.props:
        return ""
    notes = []
    for name in panel.props:
        entry = prop_registry.get(name)
        notes.append(f"{name} ({entry.description})" if entry else name)
    return ", ".join(notes)


# Expands a handful of CAMERA_HINTS (train_captioner.py) into longer,
# more explicit compositional phrasing for the SDXL prompt only - found via
# real generation comparisons (not just eyeballing the two-to-three-word
# hint) that sdxl-turbo at its production steps/guidance settings mostly
# ignores the terse form of these four: "wide two-shot" and "wide
# establishing shot" both rendered as an ordinary close/medium two-shot,
# and "bird's-eye view" produced no overhead angle at all. Spelling out
# what the shot actually contains (full room vs. close crop, camera
# position, what's blurred vs. sharp) fixed three of the four outright on
# the same cheap checkpoint - no checkpoint switch, no LoRA training
# needed. "over-the-shoulder" only partially improved (both faces turned
# to profile, but no real foreground shoulder/head occlusion) and pushing
# the phrasing further overcorrected into a single-subject blurred
# close-up that lost the second character and drifted off the manga
# line-art style entirely - so it's left at the milder, partial-improvement
# phrasing rather than chasing a further fix here.
#
# The short form is deliberately kept as panel.camera_hint's actual value
# (unaffected by this table) - it's what parse_caption_and_camera parses
# and what the captioner is trained to predict; this expansion only
# happens at prompt-build time, for the categories known to need it.
# extreme close-up/close-up/medium shot aren't here since they already
# render correctly from the terse hint alone.
_CAMERA_HINT_PROMPT_EXPANSIONS = {
    "wide two-shot": (
        "wide shot, both characters visible full-body, standing apart, "
        "wide angle lens, lots of empty space around them"
    ),
    "wide establishing shot": (
        "wide establishing shot, entire room visible, small distant figures, "
        "long shot, environment fills the frame"
    ),
    "over-the-shoulder": (
        "over-the-shoulder shot, blurred silhouette of shoulder and back of "
        "head filling the foreground, other character facing camera in sharp focus"
    ),
    "bird's-eye view": (
        "bird's-eye view, directly overhead camera angle, top-down aerial "
        "perspective looking straight down at the scene"
    ),
}


def _build_prompt(style_prompt: str, panel: Panel, prop_registry: CharacterRegistry | None) -> str:
    """The real SDXL generation prompt for one panel.

    camera_hint (Stage A's content-aware shot framing, e.g. "close-up",
    "wide establishing shot", "bird's-eye view" - see CAMERA_HINTS in
    train_captioner.py) is expanded via _CAMERA_HINT_PROMPT_EXPANSIONS (for
    the handful of hints known to need it) and placed right after
    style_prompt and before the scene description: MockBackend already
    showed the raw hint as a cosmetic text label, but DiffusersBackend was
    silently dropping it entirely, and shot-type phrasing early in an SDXL
    prompt tends to influence composition more reliably than the same words
    buried after a long scene description.
    """
    camera_phrase = _CAMERA_HINT_PROMPT_EXPANSIONS.get(panel.camera_hint, panel.camera_hint)
    prompt = ", ".join(part for part in (style_prompt, camera_phrase, panel.scene_description) if part)
    prop_notes = _prop_notes(panel, prop_registry)
    if prop_notes:
        prompt = f"{prompt}, featuring {prop_notes}"
    return prompt


def _abstract_character_note(name: str | None, registry: CharacterRegistry | None) -> str:
    """Text-anchoring for a character marked abstract/no-form (see
    parse_character_profiles) - same trick as _prop_notes, for the same
    reason: an SDXL 'character reference portrait' forces a human figure
    regardless of what the description says, so an entity explicitly
    written as having no body is never image-conditioned, only described
    in the prompt on the panel(s) it's actually in."""
    if name is None or registry is None:
        return ""
    entry = registry.get(name)
    return f"{name} ({entry.description})" if entry else name


class ImageBackend(ABC):
    def prepare_characters(
        self, story_id: str, registry: CharacterRegistry, style_prompt: str, force: bool = False
    ) -> None:
        """Generate/refresh any per-character identity assets (Stage B) up
        front, right after story adaptation and before any page is
        generated - rather than lazily on first appearance during Stage C.
        This guarantees every character in the registry gets a reference
        regardless of whether they happen to first appear in a solo panel,
        and keeps Stage C's per-panel loop free of first-use branching.
        Also callable standalone (studio's "Generate Cast" step) so a user
        can preview/approve identities before spending time on full page
        generation. force=True regenerates even names that already have a
        reference image, for a manual "Regenerate" request; the implicit
        call from run_pipeline always leaves force=False so it stays a
        no-op idempotent pass over whatever the standalone step already
        produced. Default no-op; MockBackend has nothing to prepare.
        """

    def prepare_locations(
        self, story_id: str, location_registry: CharacterRegistry, style_prompt: str, force: bool = False
    ) -> None:
        """Same idea as prepare_characters, for locations. Default no-op;
        MockBackend has nothing to prepare."""

    def prepare_props(
        self, story_id: str, prop_registry: CharacterRegistry, style_prompt: str, force: bool = False
    ) -> None:
        """Generates a reference image per prop for human review in the
        studio UI only - props are never image-conditioned during
        generation (see parse_prop_profiles), so this reference is purely
        documentation, not fed back into generate_panel. Default no-op;
        MockBackend has nothing to prepare."""

    def generate_character_lora_images(
        self, story_id: str, name: str, style_prompt: str, registry: CharacterRegistry, count: int = 8, seed: int = 0
    ) -> list:
        """Bootstrap a small set of varied training images for
        train_character_lora.py's per-character LoRA trainer - see
        manga_pipeline/character_lora.py. Default no-op (empty list);
        MockBackend has no real generation to bootstrap from."""
        return []

    @abstractmethod
    def generate_panel(
        self,
        story_id: str,
        page_index: int,
        page: Page,
        panel_index: int,
        panel: Panel,
        size: tuple[int, int],
        style_prompt: str,
        registry: CharacterRegistry | None,
        location_registry: CharacterRegistry | None = None,
        prop_registry: CharacterRegistry | None = None,
    ) -> Image.Image:
        """Generate a single panel image at its own target size (the actual
        pixel dimensions of its box in the page layout). Each panel is
        generated independently rather than warped out of one shared
        whole-page image: an earlier design generated one full-page base
        image and stretched/squashed it into each panel's differently
        shaped box (a thin V3 strip, a G22 quadrant, ...) via img2img,
        which is what produced visibly elongated/squashed/repeated panels.
        compose_page() still crops each returned image to fit its box
        exactly, so panels don't need to be pixel-perfect on size, just the
        right aspect ballpark.
        """


class MockBackend(ImageBackend):
    """No-download, no-GPU stand-in so the layout/assembly/bubble stages
    can be validated on any machine before wiring up a real diffusion model.
    """

    def generate_panel(
        self,
        story_id: str,
        page_index: int,
        page: Page,
        panel_index: int,
        panel: Panel,
        size: tuple[int, int],
        style_prompt: str,
        registry: CharacterRegistry | None,
        location_registry: CharacterRegistry | None = None,
        prop_registry: CharacterRegistry | None = None,
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

    generate_panel() generates each panel independently via text2img, sized
    to that panel's own layout box (rounded to a multiple of 8 for SDXL,
    then resized back to the exact target for compose_page). An earlier
    version generated one shared full-page "base" image and img2img'd it
    into each panel's box via a plain resize - since panels on the same page
    can have very different aspect ratios (a thin V3 strip vs. a G22
    quadrant), that resize visibly stretched/squashed/repeated the same
    page content across panels. Generating each panel at its own aspect
    ratio from the start removes that distortion entirely.

    Identity adapter (Stage B/Phase 4): two IP-Adapter slots are loaded
    simultaneously - one for character identity, one for location/prop
    identity - so a generation can be conditioned on both at once (verified
    empirically: a character reference and a location reference combined
    both come through recognizably in the same image, neither one
    drowning out the other). For characters: a panel with exactly one named
    character conditions on that character's reference; zero characters
    falls back to the page's dominant character (Stage A leaving a panel
    untagged usually means an ongoing scene, not a new one); multiple
    characters skip identity conditioning entirely, since blending several
    identities into one panel is a harder unsolved problem. Locations follow
    the identical three-way policy on panel.locations, using the page's
    dominant location as the zero-location fallback. Cross-panel/cross-page
    continuity within a page now rests entirely on this identity
    conditioning plus shared prompt wording, since there's no longer a
    shared base image's pixels to inherit from.

    A character whose profile is tagged "[no-form]" (see
    parse_character_profiles) - an AI, a voice, a presence with no physical
    body - is excluded from this whole policy and handled like a prop
    instead (see _abstract_character_note): no reference portrait, no
    IP-Adapter slot. Confirmed empirically that skipping this matters: the
    normal "character reference portrait, front-facing" template produces a
    photorealistic human face regardless of what the description says, so
    without this carve-out an explicitly bodiless character still gets
    rendered - and then IP-Adapter-conditioned - as a person.

    Props are handled entirely differently: no IP-Adapter slot, no dominant-
    page fallback. Testing showed a small portable object doesn't survive
    IP-Adapter image conditioning well - a "wide establishing shot" reference
    (the location template) produces a cluttered scene rather than a clean
    single-object reference, and even a clean close-up reference would risk
    distorting a panel's whole composition around one incidental item. So a
    prop is only ever text-anchored: its description is woven directly into
    generate_panel's prompt on exactly the panels Stage A tagged it on (see
    _prop_notes), the same mechanism Stage A already uses for character
    appearance notes.
    """

    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        self.device = resolve_device(cfg.device)
        self._base_pipe = None
        self._ip_adapter_loaded = False
        self._neutral_ip_image = None
        # per-character LoRA (see character_lora.py) - tracks which adapters
        # are already loaded into the pipe (loading is comparatively slow;
        # switching the *active* one via set_adapters()/disable_lora() is
        # cheap) and which one, if any, is currently active
        self._loaded_lora_adapters: set[str] = set()
        self._active_lora_adapter: str | None = None
        self._active_composition_path: str | None = None
        # separate pose-ControlNet pipe (see _load_pose_pipe) - only loaded
        # lazily, the first time a panel actually needs it
        self._pose_pipe = None

    def _load(self):
        if self._base_pipe is not None:
            return
        import torch
        from diffusers import AutoPipelineForText2Image

        dtype = torch.float16 if self.device.startswith("cuda") else torch.float32
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
            # two slots loaded from the same checkpoint file: slot 0 = character
            # identity, slot 1 = location/prop identity - diffusers supports this
            # natively via list arguments to load_ip_adapter/set_ip_adapter_scale
            # and a list of images to ip_adapter_image, one entry per slot.
            # Loaded before from_pipe() so the shared unet/image_encoder carry
            # the adapters over; works fine before device placement since it's
            # just loading state dicts.
            self._base_pipe.load_ip_adapter(
                ["h94/IP-Adapter", "h94/IP-Adapter"],
                subfolder=["sdxl_models", "sdxl_models"],
                weight_name=["ip-adapter_sdxl.bin", "ip-adapter_sdxl.bin"],
            )
            self._base_pipe.set_ip_adapter_scale([0.0, 0.0])
            self._ip_adapter_loaded = True
            # enable_attention_slicing() unconditionally overwrites every cross-attention
            # processor with a plain SlicedAttnProcessor, which clobbers the IP-Adapter-aware
            # processors load_ip_adapter() installs - the two are mutually exclusive here.
            # The identity adapter's extra CLIP vision encoder (~3.7GB) is enough on its own
            # to reintroduce the VAE-decode OOM the slicing/tiling above was fixing, so trade
            # speed for memory instead: keep only the actively-computing submodule on GPU.
            # enable_model_cpu_offload() manages device placement itself - don't call .to() first.
            if self.device.startswith("cuda"):
                self._base_pipe.enable_model_cpu_offload(device=self.device)
            else:
                self._base_pipe.to(self.device)
        else:
            self._base_pipe.to(self.device)
            self._base_pipe.enable_attention_slicing()

    def _asset_dir(self, story_id: str, kind: str) -> Path:
        path = Path(self.cfg.output_dir) / story_id / kind
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _neutral_image(self):
        if self._neutral_ip_image is None:
            self._neutral_ip_image = Image.new("RGB", (224, 224), (128, 128, 128))
        return self._neutral_ip_image

    def _ip_kwargs(
        self,
        pipe,
        char_image: Image.Image | None,
        char_scale: float,
        loc_image: Image.Image | None,
        loc_scale: float,
    ) -> dict:
        # once IP-Adapter is loaded, every call on this unet must supply an
        # ip_adapter_image for every slot - a neutral placeholder at scale=0.0
        # is a mathematically-inert stand-in for "no identity conditioning
        # in this slot"
        if not self._ip_adapter_loaded:
            return {}
        pipe.set_ip_adapter_scale([char_scale, loc_scale])
        return {
            "ip_adapter_image": [
                char_image if char_image is not None else self._neutral_image(),
                loc_image if loc_image is not None else self._neutral_image(),
            ]
        }

    def prepare_characters(
        self, story_id: str, registry: CharacterRegistry, style_prompt: str, force: bool = False
    ) -> None:
        self._load()
        for name, entry in registry.all().items():
            if entry.is_abstract:
                # like props: this reference is purely for the studio UI to
                # display, never fed into IP-Adapter conditioning, so it's
                # generated regardless of use_identity_adapter
                self._reference_image(name, story_id, style_prompt, registry, force=force)
                continue
            if not self.cfg.use_identity_adapter:
                continue
            self._reference_image(name, story_id, style_prompt, registry, force=force)

    def prepare_locations(
        self, story_id: str, location_registry: CharacterRegistry, style_prompt: str, force: bool = False
    ) -> None:
        if not self.cfg.use_identity_adapter:
            return
        self._load()
        for name in location_registry.all():
            self._location_reference_image(name, story_id, style_prompt, location_registry, force=force)

    def prepare_props(
        self, story_id: str, prop_registry: CharacterRegistry, style_prompt: str, force: bool = False
    ) -> None:
        # not gated on use_identity_adapter: props are never image-conditioned
        # during generation, so this reference is purely for the studio UI to
        # display, independent of whether the identity-adapter feature is on
        self._load()
        for name in prop_registry.all():
            self._prop_reference_image(name, story_id, style_prompt, prop_registry, force=force)

    def generate_character_lora_images(
        self, story_id: str, name: str, style_prompt: str, registry: CharacterRegistry, count: int = 8
    ) -> list[Path]:
        """Bootstrap `count` varied portrait images of `name` for
        train_character_lora.py, since there's no photo set to train a
        character-identity LoRA from - only the character's text
        description and whatever the base model already renders for it.
        Each image uses a different TRAINING_VIEW_PROMPTS entry (own seed)
        so the LoRA sees the character across several angles/expressions
        rather than memorizing one static pose (see character_lora.py's
        TRAINING_VIEW_PROMPTS docstring). Saved under
        <output_dir>/<story_id>/characters/<name>/lora_training/, distinct
        from the single reference image _reference_image() maintains.
        """
        self._load()
        import torch

        entry = registry.get(name)
        description = entry.description if entry else name
        views = TRAINING_VIEW_PROMPTS[: max(1, min(count, len(TRAINING_VIEW_PROMPTS)))]

        out_dir = self._asset_dir(story_id, "characters") / sanitize_adapter_name(name) / "lora_training"
        out_dir.mkdir(parents=True, exist_ok=True)

        paths: list[Path] = []
        for i, view in enumerate(views):
            prompt = f"{style_prompt}, character reference portrait of {name}, {description}, {view}"
            generator = torch.Generator(device=self.device).manual_seed(_seed_for(f"{story_id}:lora_training:{name}:{seed}", i))
            result = self._base_pipe(
                prompt=prompt,
                num_inference_steps=self.cfg.steps,
                guidance_scale=self.cfg.guidance_scale,
                width=768,
                height=768,
                generator=generator,
                **self._ip_kwargs(self._base_pipe, None, 0.0, None, 0.0),
            )
            path = out_dir / f"{i:02d}.png"
            result.images[0].save(path)
            paths.append(path)
        return paths

    def _activate_character_lora(self, char_name: str | None, char_entry) -> None:
        """Load/switch the active per-character LoRA adapter for the panel
        about to be generated (see character_lora.py, train_character_lora.py).
        No-op unless cfg.use_character_lora is on and this character has a
        trained adapter on disk - a story with none trained yet behaves
        identically to before this feature existed."""
        lora_path = char_entry.lora_path if char_entry else ""
        if not self.cfg.use_character_lora or not lora_path or not Path(lora_path).exists():
            if self._active_lora_adapter is not None:
                self._base_pipe.disable_lora()
                self._active_lora_adapter = None
            return

        adapter_name = sanitize_adapter_name(char_name or "")
        if adapter_name not in self._loaded_lora_adapters:
            self._base_pipe.load_lora_weights(lora_path, adapter_name=adapter_name)
            self._loaded_lora_adapters.add(adapter_name)
        if self._active_lora_adapter != adapter_name:
            self._base_pipe.set_adapters([adapter_name], adapter_weights=[self.cfg.character_lora_scale])
            self._active_lora_adapter = adapter_name

    def _activate_composition(self) -> bool:
        """Load and activate the configured, integrity-checked adapter bank."""

        composition_path = self.cfg.adapter_composition_path
        if not composition_path:
            if self._active_composition_path is not None:
                self._base_pipe.disable_lora()
                self._active_composition_path = None
            return False
        path = Path(composition_path).resolve()
        if self._active_composition_path == str(path):
            return True
        composition = json.loads(path.read_text(encoding="utf-8"))
        components = resolve_composition_paths(composition, path.parent.parent)
        names = []
        weights = []
        for component in components:
            adapter_name = sanitize_adapter_name(f"{composition['name']}_{component['name']}_{component['version']}")
            if adapter_name not in self._loaded_lora_adapters:
                self._base_pipe.load_lora_weights(component["path"], adapter_name=adapter_name)
                self._loaded_lora_adapters.add(adapter_name)
            names.append(adapter_name)
            weights.append(component["weight"])
        self._base_pipe.set_adapters(names, adapter_weights=weights)
        self._active_composition_path = str(path)
        self._active_lora_adapter = None
        return True

    def _load_pose_pipe(self) -> None:
        """Lazily loads the separate pose-ControlNet pipeline (see
        cfg.use_pose_controlnet) - a real second full-SDXL pipeline in
        memory, only paid for on the first panel that actually needs it, not
        for every story regardless of whether any panel ever tags 2+
        characters.

        enable_model_cpu_offload() rather than .to(device): this pipe
        coexists with self._base_pipe (never unloaded once a story starts
        generating), and a real run hit CUDA OOM at VAE decode with both
        fully resident on a 16GB card - unsurprising in hindsight, since
        cfg.use_identity_adapter (on by default) already puts _base_pipe
        itself on cpu-offload for the same reason (see _load). Slower per
        pose-conditioned panel, but that path is already ~7-8x the normal
        per-panel cost regardless, so this isn't the dominant cost.
        """
        if self._pose_pipe is not None:
            return
        import torch
        from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline

        dtype = torch.float16 if self.device.startswith("cuda") else torch.float32
        controlnet = ControlNetModel.from_pretrained(_POSE_CONTROLNET_MODEL, torch_dtype=dtype)
        self._pose_pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            _POSE_CONTROLNET_BASE_CHECKPOINT, controlnet=controlnet, torch_dtype=dtype
        )
        self._pose_pipe.vae.enable_slicing()
        self._pose_pipe.vae.enable_tiling()
        if self.device.startswith("cuda"):
            self._pose_pipe.enable_model_cpu_offload(device=self.device)
        else:
            self._pose_pipe.to(self.device)

    def _generate_with_pose_controlnet(self, prompt: str, count: int, width: int, height: int, generator) -> Image.Image:
        """Real generation comparisons (not just prompt wording) found no
        amount of "exactly two people, nobody else in frame" phrasing
        reliably stopped SDXL from dropping or duplicating figures in a
        multi-character panel. A synthetic OpenPose skeleton with exactly
        `count` figures (see pose_skeleton.multi_person_skeleton), fed to
        this ControlNet, made the headcount and rough positioning a
        structural guarantee instead - the actual fix this whole path
        exists for.

        Doesn't assign a specific character's identity to a specific figure
        position - each figure's look is still whatever the shared text
        prompt implies (one real comparison run got genders swapped between
        the two figures at a higher controlnet_conditioning_scale). That's
        the same open problem the per-character LoRA solves for a
        single-subject panel, not yet extended to a multi-figure one - see
        cfg.pose_controlnet_scale's docstring in config.py for the
        scale/identity-following tradeoff this was tuned against.

        Runs against a full SDXL base checkpoint (_POSE_CONTROLNET_BASE_CHECKPOINT),
        not cfg.checkpoint (sdxl-turbo by default) - the ControlNet checkpoint is
        trained against that specific base and won't work with an unrelated one, and
        that base's normal (non-turbo) steps/guidance also happens to be what real
        testing showed was needed for legible, in-style output here, not silhouettes.
        """
        self._load_pose_pipe()
        pose_image = multi_person_skeleton(width, height, count)
        result = self._pose_pipe(
            prompt=prompt,
            negative_prompt=_POSE_CONTROLNET_NEGATIVE_PROMPT,
            image=pose_image,
            num_inference_steps=_POSE_CONTROLNET_STEPS,
            guidance_scale=_POSE_CONTROLNET_GUIDANCE_SCALE,
            controlnet_conditioning_scale=self.cfg.pose_controlnet_scale,
            generator=generator,
            width=width,
            height=height,
        )
        return result.images[0]

    def _generate_reference(self, prompt: str, story_id: str, name: str, kind: str, registry: CharacterRegistry):
        import torch

        generator = torch.Generator(device=self.device).manual_seed(_seed_for(f"{story_id}:{kind}:{name}", 0))
        result = self._base_pipe(
            prompt=prompt,
            num_inference_steps=self.cfg.steps,
            guidance_scale=self.cfg.guidance_scale,
            width=512,
            height=512,
            generator=generator,
            **self._ip_kwargs(self._base_pipe, None, 0.0, None, 0.0),
        )
        image = result.images[0]
        safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._") or "asset"
        name_suffix = hashlib.sha256(name.encode("utf-8")).hexdigest()[:10]
        path = self._asset_dir(story_id, kind) / f"{safe_name[:80]}-{name_suffix}.png"
        image.save(path)
        registry.set_reference_image(name, str(path))
        return image

    def _reference_image(
        self, name: str, story_id: str, style_prompt: str, registry: CharacterRegistry, force: bool = False
    ):
        from PIL import Image as PILImage

        entry = registry.get(name)
        if not force and entry and entry.reference_image and Path(entry.reference_image).exists():
            return PILImage.open(entry.reference_image).convert("RGB")

        description = entry.description if entry else name
        if entry and entry.is_abstract:
            # no "character portrait, front-facing" template here - that
            # template forces a human figure regardless of the description
            # (confirmed empirically), which is exactly wrong for something
            # explicitly written as having no body
            prompt = (
                f"{style_prompt}, abstract visual motif representing {name}: {description}, "
                "no face, no human figure, no body, plain background"
            )
        else:
            prompt = f"{style_prompt}, character reference portrait of {name}, {description}, plain background, front-facing"
        return self._generate_reference(prompt, story_id, name, "characters", registry)

    def _location_reference_image(
        self,
        name: str,
        story_id: str,
        style_prompt: str,
        location_registry: CharacterRegistry,
        force: bool = False,
    ):
        from PIL import Image as PILImage

        entry = location_registry.get(name)
        if not force and entry and entry.reference_image and Path(entry.reference_image).exists():
            return PILImage.open(entry.reference_image).convert("RGB")

        description = entry.description if entry else name
        prompt = f"{style_prompt}, location reference: {name}, {description}, wide establishing shot, no characters"
        return self._generate_reference(prompt, story_id, name, "locations", location_registry)

    def _prop_reference_image(
        self, name: str, story_id: str, style_prompt: str, prop_registry: CharacterRegistry, force: bool = False
    ):
        from PIL import Image as PILImage

        entry = prop_registry.get(name)
        if not force and entry and entry.reference_image and Path(entry.reference_image).exists():
            return PILImage.open(entry.reference_image).convert("RGB")

        # close-up isolated object shot, not a wide establishing shot - testing
        # showed the location-style wide-shot template produces a cluttered,
        # unusable scene for a small object (see parse_prop_profiles)
        description = entry.description if entry else name
        prompt = f"{style_prompt}, prop reference: {name}, {description}, close-up, isolated object, plain background"
        return self._generate_reference(prompt, story_id, name, "props", prop_registry)

    def _resolve_target(self, names: list[str], dominant: str | None) -> str | None:
        if len(names) == 1:
            return names[0]
        if not names:
            return dominant  # untagged panel: assume continuity with the page's dominant asset
        return None  # multiple: blending several identities into one panel is unsolved, skip

    def generate_panel(
        self,
        story_id: str,
        page_index: int,
        page: Page,
        panel_index: int,
        panel: Panel,
        size: tuple[int, int],
        style_prompt: str,
        registry: CharacterRegistry | None,
        location_registry: CharacterRegistry | None = None,
        prop_registry: CharacterRegistry | None = None,
    ) -> Image.Image:
        self._load()
        import torch

        seed = _seed_for(story_id, page_index, panel_index)
        generator = torch.Generator(device=self.device).manual_seed(seed)
        prompt = _build_prompt(style_prompt, panel, prop_registry)

        char_name = self._resolve_target(panel.characters, _dominant_character(page))
        char_entry = registry.get(char_name) if (registry and char_name) else None
        if char_entry and char_entry.is_abstract:
            # text-anchored like a prop, never image-conditioned - see
            # _abstract_character_note
            char_image = None
            char_scale = 0.0
            abstract_note = _abstract_character_note(char_name, registry)
            if abstract_note:
                prompt = f"{prompt}, featuring {abstract_note}"
        else:
            char_image = self._reference_image(char_name, story_id, style_prompt, registry) if (registry and char_name) else None
            char_scale = self.cfg.identity_adapter_scale if char_image is not None else 0.0

        loc_name = self._resolve_target(panel.locations, _dominant_location(page))
        loc_image = (
            self._location_reference_image(loc_name, story_id, style_prompt, location_registry)
            if (location_registry and loc_name)
            else None
        )
        loc_scale = self.cfg.identity_adapter_scale if loc_image is not None else 0.0

        # a trained LoRA only ever applies to the single resolved character
        # (never abstract/no-form ones, same as IP-Adapter conditioning) -
        # see _activate_character_lora
        if not self._activate_composition():
            self._activate_character_lora(char_name if not (char_entry and char_entry.is_abstract) else None, char_entry)

        # SDXL requires width/height to be multiples of 8; a panel's own box in
        # the page layout (e.g. a third of the page width for an H3 layout)
        # rarely lands on one, so generate at the nearest multiple of 8 and let
        # compose_page's crop-to-fit handle the sub-pixel difference.
        gen_width, gen_height = _round_to_8(size[0]), _round_to_8(size[1])

        if self.cfg.use_pose_controlnet and len(panel.characters) >= 2:
            # char_name is already None here (_resolve_target skips identity
            # conditioning for 2+ characters) - this path replaces that gap
            # with a structural headcount/position guarantee instead, see
            # _generate_with_pose_controlnet
            image = self._generate_with_pose_controlnet(prompt, len(panel.characters), gen_width, gen_height, generator)
        else:
            result = self._base_pipe(
                prompt=prompt,
                num_inference_steps=self.cfg.steps,
                guidance_scale=self.cfg.guidance_scale,
                width=gen_width,
                height=gen_height,
                generator=generator,
                **self._ip_kwargs(self._base_pipe, char_image, char_scale, loc_image, loc_scale),
            )
            image = result.images[0]
        if (gen_width, gen_height) != size:
            image = image.resize(size)
        return image


def _round_to_8(value: int) -> int:
    return max(8, round(value / 8) * 8)


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
