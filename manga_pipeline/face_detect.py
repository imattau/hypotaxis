from __future__ import annotations

from PIL import Image


class FaceAnchorDetector:
    """Finds approximate face-anchor points in a generated panel image, for
    bubble-tail placement (see bubbles.py's SVG bubble tails). Reuses the
    same OpenPose body detector already validated for pose-ControlNet
    headcount verification (see DiffusersBackend._count_detected_people) -
    real testing on actual generated manga-style panels found it precisely
    accurate on this art style whenever it detects a person at all (every
    detected nose landed right on a real face across two real test images),
    but with the same real recall gap already documented there: it missed
    3 of 5 clearly visible faces in one of those images. It's trained on
    photos, not manga/anime line art. Callers must treat a returned anchor
    list as "these are trustworthy, but faces may be missing," not as a
    complete or guaranteed set - see bubbles.py's fallback behavior for
    panels where nothing (or not enough) gets detected.

    Deliberately anchors on the nose keypoint, not a true mouth/lip
    landmark - OpenPose's 18-point body format doesn't include one, and the
    nose is close enough to "point the bubble tail at this character's
    face" for the purpose here without needing a separate facial-landmark
    model.
    """

    def __init__(self):
        self._detector = None

    def _load(self):
        if self._detector is not None:
            return
        from controlnet_aux import OpenposeDetector

        self._detector = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")

    def find_anchors(self, image: Image.Image) -> list[tuple[float, float]]:
        """Returns (x, y) pixel positions (in `image`'s own coordinate
        space) of each detected person's nose, ordered left-to-right. Left-
        to-right is a simple, deterministic tie-break, not a real match to
        a specific named character - there's no reliable way to tell which
        detected figure is which character from pixels alone, the same
        open problem noted for pose-ControlNet's multi-figure panels."""
        import numpy as np

        self._load()
        width, height = image.size
        poses = self._detector.detect_poses(np.array(image.convert("RGB")))
        anchors = []
        for person in poses:
            keypoints = person.body.keypoints if person.body else None
            nose = keypoints[0] if keypoints else None
            if nose is not None:
                anchors.append((nose.x * width, nose.y * height))
        anchors.sort(key=lambda point: point[0])
        return anchors
