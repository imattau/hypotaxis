"""Page layout templates: template name -> list of (x, y, w, h) fractional boxes.

Naming: H<n> = n panels side by side. V<n> = n panels stacked.
G<r><c> = an r x c grid. H<a><b> = two horizontal bands, top split into
`a` columns and bottom split into `b` columns. V<a><b> = two vertical
bands, left split into `a` rows and right split into `b` rows.
"""

from __future__ import annotations

Box = tuple[float, float, float, float]


def _row(n: int, y: float, h: float) -> list[Box]:
    return [(i / n, y, 1 / n, h) for i in range(n)]


def _col(n: int, x: float, w: float) -> list[Box]:
    return [(x, i / n, w, 1 / n) for i in range(n)]


def _grid(rows: int, cols: int) -> list[Box]:
    boxes = []
    for r in range(rows):
        for c in range(cols):
            boxes.append((c / cols, r / rows, 1 / cols, 1 / rows))
    return boxes


LAYOUTS: dict[str, list[Box]] = {
    "H2": _row(2, 0.0, 1.0),
    "H3": _row(3, 0.0, 1.0),
    "V2": _col(2, 0.0, 1.0),
    "V3": _col(3, 0.0, 1.0),
    "G22": _grid(2, 2),
    "G33": _grid(3, 3),
    "H12": _row(1, 0.0, 0.5) + _row(2, 0.5, 0.5),
    "H21": _row(2, 0.0, 0.5) + _row(1, 0.5, 0.5),
    "H13": _row(1, 0.0, 0.5) + _row(3, 0.5, 0.5),
    "H31": _row(3, 0.0, 0.5) + _row(1, 0.5, 0.5),
    "V12": _col(1, 0.0, 0.5) + _col(2, 0.5, 0.5),
    "V21": _col(2, 0.0, 0.5) + _col(1, 0.5, 0.5),
}


def panel_count(layout: str) -> int:
    return len(LAYOUTS[layout])


def boxes_for(layout: str) -> list[Box]:
    return LAYOUTS[layout]


# Stage A (pack_into_pages in story_adapt.py) picks a layout per page without
# knowing the final render's PipelineConfig - page_width/page_height is a
# Stage C rendering concern, and threading it through would couple two
# currently-independent stages just to cover a case every default
# PipelineConfig already matches. Assumed here instead: the same 1024x1536
# portrait page every default actually renders at.
_ASSUMED_PAGE_ASPECT = 1024 / 1536


def is_wide_box(box: Box) -> bool:
    """True if a panel box's own shape (accounting for the page's overall
    portrait proportions - see _ASSUMED_PAGE_ASPECT) is at least as wide as
    it is tall, i.e. landscape-oriented rather than portrait-oriented.

    Used to keep pack_into_pages from handing a "wide two-shot"/"wide
    establishing shot" panel a box that physically can't hold it - found via
    a real page-generation test where exactly that happened: a wide
    two-shot panel landed in an H3 layout's narrow vertical strip and
    rendered as a single figure standing at a window instead of the two
    people the prompt asked for, even though the prompt text and character
    tags were both correct. The panel shape itself was fighting the
    requested composition.
    """
    _, _, w, h = box
    return (w * _ASSUMED_PAGE_ASPECT) / h >= 1.0
