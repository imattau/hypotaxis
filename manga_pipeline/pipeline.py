from __future__ import annotations

from pathlib import Path
from dataclasses import asdict
import hashlib
import json
from typing import Callable

from PIL import Image

from .assembly import compose_page
from .backends import build_backend
from .bubbles import render_bubbles
from .config import PipelineConfig, atomic_write_text
from .layouts import boxes_for, panel_count
from .registry import CharacterRegistry
from .schema import Story


def _production_signature(story: Story, cfg: PipelineConfig, registries: list[CharacterRegistry]) -> str:
    """Return the identity of the inputs that affect rendered pages.

    Reference-image paths are intentionally excluded: generation itself writes
    those paths into the registries, and they are an output, not an input.
    """
    registry_inputs = []
    for registry in registries:
        registry_inputs.append([
            {
                "name": name,
                "description": entry.description,
                "is_abstract": entry.is_abstract,
                "lora_path": entry.lora_path,
            }
            for name, entry in sorted(registry.all().items())
        ])
    payload = {
        "format": 1,
        "story": asdict(story),
        "config": {
            key: getattr(cfg, key)
            for key in (
                "backend", "checkpoint", "device", "steps", "guidance_scale",
                "page_width", "page_height", "seed", "use_identity_adapter",
                "identity_adapter_scale", "use_character_lora", "character_lora_scale",
                "adapter_composition_path", "use_pose_controlnet", "pose_controlnet_scale",
                "use_quality_review", "quality_review_max_retries",
            )
        },
        "registries": registry_inputs,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _page_signature(story: Story, page_index: int, production_signature: str) -> str:
    payload = {"production": production_signature, "page_index": page_index, "page": asdict(story.pages[page_index])}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def prepare_cast(
    story: Story, cfg: PipelineConfig, on_progress: "Callable[[str], None] | None" = None, force: bool = False
) -> None:
    """Standalone Stage B step: generate/refresh character, location, and
    prop reference images without touching page generation. Lets the studio
    UI show a cast preview the user can review (and, with force=True,
    regenerate) before committing GPU time to a full page run. `run()` below
    also calls prepare_* itself so a direct page-generation request still
    works without this step, but that call always uses force=False, so it's
    a no-op over whatever this step already produced.
    """

    def report(msg: str) -> None:
        if on_progress is not None:
            on_progress(msg)

    backend = build_backend(cfg)
    registry = CharacterRegistry(Path(cfg.registry_dir) / f"{story.id}.json")
    location_registry = CharacterRegistry(Path(cfg.registry_dir) / f"{story.id}_locations.json")
    prop_registry = CharacterRegistry(Path(cfg.registry_dir) / f"{story.id}_props.json")

    report("designing characters")
    backend.prepare_characters(story.id, registry, story.style_prompt, force=force)
    report("designing locations")
    backend.prepare_locations(story.id, location_registry, story.style_prompt, force=force)
    report("designing props")
    backend.prepare_props(story.id, prop_registry, story.style_prompt, force=force)
    report("done")


def run(
    story: Story,
    cfg: PipelineConfig,
    on_progress: "Callable[[str], None] | None" = None,
    force: bool = False,
) -> Path:
    """force=False (the default) makes this resumable: a page whose PNG
    already exists on disk is loaded rather than regenerated. Page
    generation is the single most expensive, most crash-prone step in the
    pipeline (real GPU time, real risk of OOM on the modest hardware this
    project targets) - without this, a job that dies on page 12 of 17
    forces a full from-scratch redo of the 11 pages that already succeeded.
    Panels within a page are still always generated together (no per-panel
    resume): compose_page/render_bubbles need the whole set at once, and a
    single page's worth of GPU work is small enough that re-doing it isn't
    the expensive part. Pass force=True for a deliberate "regenerate
    everything" request.
    """

    def report(msg: str) -> None:
        if on_progress is not None:
            on_progress(msg)

    lettering_warnings: list[str] = []

    def report_lettering_warning(msg: str) -> None:
        lettering_warnings.append(msg)
        report(f"warning: {msg}")

    # validate every page's layout up front, before spending any GPU time -
    # previously this was only checked as each page was reached, so a bad
    # page late in a long story wasted all the GPU work on the pages before it
    for page_index, page in enumerate(story.pages):
        expected = panel_count(page.layout)
        if len(page.panels) != expected:
            raise ValueError(
                f"page {page_index}: layout '{page.layout}' expects {expected} panels, got {len(page.panels)}"
            )

    backend = build_backend(cfg)
    size = (cfg.page_width, cfg.page_height)
    out_dir = Path(cfg.output_dir) / story.id
    out_dir.mkdir(parents=True, exist_ok=True)
    registry = CharacterRegistry(Path(cfg.registry_dir) / f"{story.id}.json")
    location_registry = CharacterRegistry(Path(cfg.registry_dir) / f"{story.id}_locations.json")
    prop_registry = CharacterRegistry(Path(cfg.registry_dir) / f"{story.id}_props.json")
    production_signature = _production_signature(story, cfg, [registry, location_registry, prop_registry])
    manifest_path = out_dir / "production.json"
    manifest = {"format": 1, "signature": production_signature, "pages": {}}
    if not force and manifest_path.exists():
        try:
            loaded = json.loads(manifest_path.read_text())
            if loaded.get("signature") == production_signature and isinstance(loaded.get("pages"), dict):
                manifest = loaded
        except (OSError, json.JSONDecodeError):
            pass

    report("designing characters")
    backend.prepare_characters(story.id, registry, story.style_prompt)
    report("designing locations")
    backend.prepare_locations(story.id, location_registry, story.style_prompt)
    report("designing props")
    backend.prepare_props(story.id, prop_registry, story.style_prompt)

    page_images: list[Image.Image] = []
    for page_index, page in enumerate(story.pages):
        page_path = out_dir / f"page_{page_index:02d}.png"
        page_signature = _page_signature(story, page_index, production_signature)
        if not force and page_path.exists() and manifest["pages"].get(str(page_index)) == page_signature:
            report(f"page {page_index + 1}/{len(story.pages)} already generated, skipping")
            page_images.append(Image.open(page_path).convert("RGB"))
            continue

        report(f"generating page {page_index + 1}/{len(story.pages)}")
        boxes = boxes_for(page.layout)
        panel_images = [
            backend.generate_panel(
                story.id,
                page_index,
                page,
                panel_index,
                panel,
                (max(1, round(w * size[0])), max(1, round(h * size[1]))),
                story.style_prompt,
                registry,
                location_registry,
                prop_registry,
                cfg.seed,
            )
            for panel_index, (panel, (_, _, w, h)) in enumerate(zip(page.panels, boxes))
        ]
        composed = compose_page(page.layout, panel_images, size)
        final = render_bubbles(
            composed,
            page,
            size,
            panel_images=panel_images,
            # MockBackend intentionally prints the scene description into the
            # artwork as a caption. Keep dialogue out of that reserved area;
            # diffusion panels do not contain this debug caption.
            caption_reserve_fraction=0.30 if cfg.backend == "mock" else 0.0,
            on_warning=report_lettering_warning,
        )
        final.save(page_path)
        page_images.append(final)
        manifest["pages"][str(page_index)] = page_signature
        atomic_write_text(manifest_path, json.dumps(manifest, indent=2) + "\n")

    report("assembling PDF")
    pdf_path = out_dir / f"{story.id}.pdf"
    if page_images:
        page_images[0].save(pdf_path, save_all=True, append_images=page_images[1:])

    report(f"done ({len(lettering_warnings)} lettering warnings)" if lettering_warnings else "done")
    return pdf_path
