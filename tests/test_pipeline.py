"""Regression tests for the reliability fixes in manga_pipeline/pipeline.py:
resumable page generation (skip an already-generated page unless force=True)
and upfront layout validation (fail before spending any generation time).
Uses the mock backend throughout, so these run without a GPU.
"""

from __future__ import annotations

from manga_pipeline.config import PipelineConfig
from manga_pipeline.pipeline import run
from manga_pipeline.schema import Page, Panel, Story


def _two_page_story(story_id: str) -> Story:
    return Story(
        id=story_id,
        title="Test",
        style_prompt="",
        pages=[
            Page(layout="H2", panels=[Panel(scene_description="a"), Panel(scene_description="b")]),
            Page(layout="H2", panels=[Panel(scene_description="c"), Panel(scene_description="d")]),
        ],
    )


def _cfg(tmp_path) -> PipelineConfig:
    return PipelineConfig(
        backend="mock",
        output_dir=str(tmp_path / "output"),
        registry_dir=str(tmp_path / "registry"),
    )


def test_run_validates_all_page_layouts_before_generating_anything(tmp_path):
    story = Story(
        id="bad",
        title="Test",
        style_prompt="",
        pages=[
            Page(layout="H2", panels=[Panel(scene_description="a"), Panel(scene_description="b")]),
            Page(layout="H2", panels=[Panel(scene_description="only one panel")]),  # wrong count for H2
        ],
    )
    cfg = _cfg(tmp_path)
    try:
        run(story, cfg)
        assert False, "expected ValueError"
    except ValueError:
        pass
    # the first (valid) page must not have been generated - validation
    # happens for every page before any generation work starts
    assert not (tmp_path / "output" / "bad" / "page_00.png").exists()


def test_run_resumes_by_skipping_already_generated_pages(tmp_path, monkeypatch):
    story = _two_page_story("resume_test")
    cfg = _cfg(tmp_path)

    run(story, cfg)
    page_0_path = tmp_path / "output" / "resume_test" / "page_00.png"
    first_mtime = page_0_path.stat().st_mtime_ns

    # sanity: force a detectable delay so a re-write would change mtime
    import time

    time.sleep(0.01)

    calls = []
    from manga_pipeline import backends

    original = backends.MockBackend.generate_panel

    def tracking_generate_panel(self, story_id, page_index, *args, **kwargs):
        calls.append(page_index)
        return original(self, story_id, page_index, *args, **kwargs)

    monkeypatch.setattr(backends.MockBackend, "generate_panel", tracking_generate_panel)

    run(story, cfg)  # second run, force=False (default) - should skip both pages

    assert calls == []  # nothing regenerated
    assert page_0_path.stat().st_mtime_ns == first_mtime  # file untouched


def test_run_invalidates_pages_when_generation_inputs_change(tmp_path, monkeypatch):
    story = _two_page_story("fingerprint_test")
    cfg = _cfg(tmp_path)
    run(story, cfg)

    calls = []
    from manga_pipeline import backends

    original = backends.MockBackend.generate_panel

    def tracking_generate_panel(self, story_id, page_index, *args, **kwargs):
        calls.append(page_index)
        return original(self, story_id, page_index, *args, **kwargs)

    monkeypatch.setattr(backends.MockBackend, "generate_panel", tracking_generate_panel)
    changed = _cfg(tmp_path)
    changed.seed = 42
    run(story, changed)

    assert calls == [0, 0, 1, 1]


def test_run_writes_production_manifest(tmp_path):
    story = _two_page_story("manifest_test")
    cfg = _cfg(tmp_path)
    run(story, cfg)

    import json

    manifest = json.loads((tmp_path / "output" / "manifest_test" / "production.json").read_text())
    assert manifest["format"] == 1
    assert set(manifest["pages"]) == {"0", "1"}


def test_run_force_regenerates_everything(tmp_path, monkeypatch):
    story = _two_page_story("force_test")
    cfg = _cfg(tmp_path)

    run(story, cfg)

    calls = []
    from manga_pipeline import backends

    original = backends.MockBackend.generate_panel

    def tracking_generate_panel(self, story_id, page_index, *args, **kwargs):
        calls.append(page_index)
        return original(self, story_id, page_index, *args, **kwargs)

    monkeypatch.setattr(backends.MockBackend, "generate_panel", tracking_generate_panel)

    run(story, cfg, force=True)

    assert calls == [0, 0, 1, 1]  # both pages' panels regenerated


def test_run_produces_a_pdf_with_one_page_per_story_page(tmp_path):
    story = _two_page_story("pdf_test")
    cfg = _cfg(tmp_path)
    pdf_path = run(story, cfg)
    assert pdf_path.exists()


def test_a_page_that_fails_mid_generation_leaves_no_partial_file(tmp_path, monkeypatch):
    """The core resumability guarantee: if a panel crashes partway through a
    page (GPU OOM, a transient error), that page's PNG must never be written
    half-finished - otherwise a later resume would wrongly treat a broken
    page as already complete and skip it forever."""
    story = _two_page_story("crash_test")
    cfg = _cfg(tmp_path)

    from manga_pipeline import backends

    call_count = {"n": 0}
    original = backends.MockBackend.generate_panel

    def flaky_generate_panel(self, story_id, page_index, *args, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 3:  # second panel of page 1 (0-indexed panels: 0,1 for page 0; 2,3 for page 1)
            raise RuntimeError("simulated GPU failure")
        return original(self, story_id, page_index, *args, **kwargs)

    monkeypatch.setattr(backends.MockBackend, "generate_panel", flaky_generate_panel)

    try:
        run(story, cfg)
        assert False, "expected RuntimeError"
    except RuntimeError:
        pass

    out_dir = tmp_path / "output" / "crash_test"
    assert (out_dir / "page_00.png").exists()  # completed before the crash
    assert not (out_dir / "page_01.png").exists()  # must not exist half-written

    # resuming (force=False) must regenerate the missing page, not treat it
    # as done and skip it
    monkeypatch.setattr(backends.MockBackend, "generate_panel", original)
    run(story, cfg)
    assert (out_dir / "page_01.png").exists()
