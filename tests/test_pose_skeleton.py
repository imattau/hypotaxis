"""Regression tests for manga_pipeline/pose_skeleton.py - the synthetic
OpenPose skeleton generator used to make multi-character panel headcount a
structural guarantee (see PipelineConfig.use_pose_controlnet). Needs
controlnet_aux/numpy/PIL (real project dependencies, see
requirements-generation.txt) but no GPU or model download - this is plain
deterministic image drawing.
"""

from __future__ import annotations

from manga_pipeline.pose_skeleton import multi_person_skeleton


def test_multi_person_skeleton_returns_requested_canvas_size():
    img = multi_person_skeleton(640, 320, 2)
    assert img.size == (640, 320)
    assert img.mode == "RGB"


def test_multi_person_skeleton_draws_something_for_each_figure_slot():
    # a blank canvas would be all-black; each additional figure should only
    # ever add more non-black pixels, never fewer, since figures are drawn
    # at non-overlapping horizontal slots
    import numpy as np

    one = np.array(multi_person_skeleton(640, 320, 1))
    three = np.array(multi_person_skeleton(640, 320, 3))
    assert (one > 0).sum() > 0
    assert (three > 0).sum() > (one > 0).sum()


def test_multi_person_skeleton_handles_single_figure():
    img = multi_person_skeleton(256, 256, 1)
    assert img.size == (256, 256)


def test_multi_person_skeleton_clamps_nonpositive_count():
    # count=0 (or negative) shouldn't crash - clamped to at least one figure
    img = multi_person_skeleton(256, 256, 0)
    assert img.size == (256, 256)
