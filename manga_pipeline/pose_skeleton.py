from __future__ import annotations


def _standing_pose_keypoints(cx: float, top_y: float, height: float) -> list:
    """One simple frontal standing figure's 18 OpenPose keypoints (nose,
    neck, R/L shoulder/elbow/wrist, R/L hip/knee/ankle, R/L eye/ear - the
    exact order controlnet_aux.open_pose.util.draw_bodypose expects),
    normalized to the page (0-1). cx: horizontal center. top_y: head top.
    height: head-to-foot span. Unconditional full-body pose - no per-figure
    variation, since the point is a reliable, generic placeholder shape for
    ControlNet to anchor headcount/position on, not a specific pose.
    """
    from controlnet_aux.open_pose.body import Keypoint

    head_h = height * 0.12
    neck_y = top_y + head_h
    shoulder_w = height * 0.11
    hip_w = height * 0.08
    shoulder_y = neck_y + height * 0.03
    elbow_y = shoulder_y + height * 0.16
    wrist_y = elbow_y + height * 0.16
    hip_y = top_y + height * 0.5
    knee_y = hip_y + height * 0.25
    ankle_y = top_y + height

    def kp(x, y):
        return Keypoint(x=x, y=y)

    return [
        kp(cx, top_y + head_h * 0.5),  # nose
        kp(cx, neck_y),  # neck
        kp(cx + shoulder_w, shoulder_y),  # Rshoulder
        kp(cx + shoulder_w * 1.15, elbow_y),  # Relbow
        kp(cx + shoulder_w * 1.05, wrist_y),  # Rwrist
        kp(cx - shoulder_w, shoulder_y),  # Lshoulder
        kp(cx - shoulder_w * 1.15, elbow_y),  # Lelbow
        kp(cx - shoulder_w * 1.05, wrist_y),  # Lwrist
        kp(cx + hip_w, hip_y),  # Rhip
        kp(cx + hip_w * 1.05, knee_y),  # Rknee
        kp(cx + hip_w, ankle_y),  # Rankle
        kp(cx - hip_w, hip_y),  # Lhip
        kp(cx - hip_w * 1.05, knee_y),  # Lknee
        kp(cx - hip_w, ankle_y),  # Lankle
        kp(cx + head_h * 0.2, top_y + head_h * 0.4),  # Reye
        kp(cx - head_h * 0.2, top_y + head_h * 0.4),  # Leye
        kp(cx + head_h * 0.35, top_y + head_h * 0.5),  # Rear
        kp(cx - head_h * 0.35, top_y + head_h * 0.5),  # Lear
    ]


def multi_person_skeleton(width: int, height: int, count: int):
    """A PIL Image of `count` evenly-spaced standing-figure OpenPose
    skeletons, in the exact keypoint/color convention an OpenPose-trained
    ControlNet expects (controlnet_aux.open_pose.util.draw_bodypose).

    Not ML pose estimation - there's no source photo to estimate a pose
    from, since panels are generated from prose, not photographed. A
    synthetic, deterministically-placed skeleton instead: found via real
    generation comparisons that no amount of "exactly two people" prompt
    wording reliably fixed SDXL dropping or duplicating figures in a
    multi-character panel, but conditioning on a skeleton with exactly
    `count` figures did, reliably - see
    DiffusersBackend._generate_with_pose_controlnet. Each figure is plain
    and identical (no per-character pose/identity) - this only constrains
    *how many* people and roughly *where*, not *who's who*, which stays an
    open problem the same way it already is for any panel this backend
    can't resolve a single identity for.
    """
    import numpy as np
    from controlnet_aux.open_pose import util
    from PIL import Image

    count = max(1, count)
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    for i in range(count):
        cx = (i + 0.5) / count
        canvas = util.draw_bodypose(canvas, _standing_pose_keypoints(cx, top_y=0.15, height=0.75))
    return Image.fromarray(canvas)
