from __future__ import annotations

from pathlib import Path
from typing import Callable

from PIL import Image

from .assembly import compose_page
from .backends import build_backend
from .bubbles import render_bubbles
from .config import PipelineConfig
from .layouts import boxes_for, panel_count
from .registry import CharacterRegistry
from .schema import Story


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

    report("designing characters")
    backend.prepare_characters(story.id, registry, story.style_prompt)
    report("designing locations")
    backend.prepare_locations(story.id, location_registry, story.style_prompt)
    report("designing props")
    backend.prepare_props(story.id, prop_registry, story.style_prompt)

    page_images: list[Image.Image] = []
    for page_index, page in enumerate(story.pages):
        page_path = out_dir / f"page_{page_index:02d}.png"
        if not force and page_path.exists():
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
            )
            for panel_index, (panel, (_, _, w, h)) in enumerate(zip(page.panels, boxes))
        ]
        composed = compose_page(page.layout, panel_images, size)
        final = render_bubbles(composed, page, size, panel_images=panel_images)
        final.save(page_path)
        page_images.append(final)

    report("assembling PDF")
    pdf_path = out_dir / f"{story.id}.pdf"
    if page_images:
        page_images[0].save(pdf_path, save_all=True, append_images=page_images[1:])

    report("done")
    return pdf_path
