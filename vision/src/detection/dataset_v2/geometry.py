"""Pure geometry helpers for detection dataset preparation (v2).

Coordinates use continuous pixel-space XYXY with a half-open canvas:
0 <= x <= width and 0 <= y <= height. Source boxes may be out of bounds.
No function decides whether an annotation is suitable for training.
"""

import math
from collections.abc import Sequence
from numbers import Real
from typing import Literal


Box = tuple[float, float, float, float]
BoxLocation = Literal["contained", "partial", "outside"]


def _finite(value: Real, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")

    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _four(values: Sequence[Real], name: str) -> Box:
    if not isinstance(values, (tuple, list)):
        raise TypeError(f"{name} must be a tuple or list")
    if len(values) != 4:
        raise ValueError(f"{name} must contain four coordinates")
    return tuple(
        _finite(value, f"{name}[{i}]")
        for i, value in enumerate(values)
    )


def validate_image_size(width: int, height: int) -> None:
    """Require positive integer canvas dimensions."""
    if type(width) is not int or type(height) is not int:
        raise TypeError("Image dimensions must be integers")
    if width <= 0 or height <= 0:
        raise ValueError("Image dimensions must be positive")


def validate_xyxy(box: Sequence[Real]) -> Box:
    """Return a finite, positive-area box, allowing source overflow."""
    x1, y1, x2, y2 = _four(box, "box")
    if x1 >= x2 or y1 >= y2:
        raise ValueError("Box must have positive width and height")
    return x1, y1, x2, y2


def coco_xywh_to_xyxy(box: Sequence[Real]) -> Box:
    """Convert COCO (left, top, width, height) without clipping."""
    x, y, width, height = _four(box, "COCO box")
    if width <= 0 or height <= 0:
        raise ValueError("COCO box must have positive width and height")
    return validate_xyxy((x, y, x + width, y + height))


def polygons_to_xyxy(
    polygons: Sequence[Sequence[Real]],
) -> Box:
    """Enclose one or more flat COCO polygon coordinate sequences."""
    if not isinstance(polygons, (tuple, list)) or not polygons:
        raise ValueError("Polygons must be a nonempty list or tuple")

    xs: list[float] = []
    ys: list[float] = []

    for index, polygon in enumerate(polygons):
        if not isinstance(polygon, (tuple, list)):
            raise TypeError(f"Polygon {index} must be a list or tuple")
        if len(polygon) < 6 or len(polygon) % 2:
            raise ValueError(
                f"Polygon {index} must have at least 3 points"
            )

        for offset in range(0, len(polygon), 2):
            xs.append(
                _finite(polygon[offset], f"polygon {index} x")
            )
            ys.append(
                _finite(polygon[offset + 1], f"polygon {index} y")
            )

    return validate_xyxy((
        min(xs),
        min(ys),
        max(xs),
        max(ys),
    ))


def classify_box(
    box: Sequence[Real],
    width: int,
    height: int,
) -> BoxLocation:
    """Locate a positive-area box relative to an image canvas."""
    x1, y1, x2, y2 = validate_xyxy(box)
    validate_image_size(width, height)

    if x2 <= 0 or y2 <= 0 or x1 >= width or y1 >= height:
        return "outside"

    if x1 >= 0 and y1 >= 0 and x2 <= width and y2 <= height:
        return "contained"

    return "partial"


def clip_box(
    box: Sequence[Real],
    width: int,
    height: int,
) -> Box | None:
    """Intersect a box with the image; return None for no visible area."""
    x1, y1, x2, y2 = validate_xyxy(box)
    validate_image_size(width, height)

    left = max(0.0, x1)
    top = max(0.0, y1)
    right = min(float(width), x2)
    bottom = min(float(height), y2)

    if left >= right or top >= bottom:
        return None

    return left, top, right, bottom


def xyxy_to_yolo(
    box: Sequence[Real],
    width: int,
    height: int,
) -> Box:
    """Normalize a contained box as YOLO (cx, cy, w, h)."""
    x1, y1, x2, y2 = validate_xyxy(box)

    if classify_box(
        (x1, y1, x2, y2),
        width,
        height,
    ) != "contained":
        raise ValueError(
            "YOLO export requires a box inside the image"
        )

    return (
        (x1 + (x2 - x1) / 2) / width,
        (y1 + (y2 - y1) / 2) / height,
        (x2 - x1) / width,
        (y2 - y1) / height,
    )


def yolo_to_xyxy(
    box: Sequence[Real],
    width: int,
    height: int,
) -> Box:
    """Decode YOLO (cx, cy, w, h) without silently clipping."""
    cx, cy, bw, bh = _four(box, "YOLO box")
    validate_image_size(width, height)

    if bw <= 0 or bh <= 0:
        raise ValueError(
            "YOLO box must have positive width and height"
        )

    return validate_xyxy((
        (cx - bw / 2) * width,
        (cy - bh / 2) * height,
        (cx + bw / 2) * width,
        (cy + bh / 2) * height,
    ))
