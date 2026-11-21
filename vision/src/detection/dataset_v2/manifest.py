"""Immutable records for detection dataset v2 provenance.

Records describe source images, annotations, and their processing
outcomes. They do not parse datasets, assign splits, or write files.
"""

import math
import re
from dataclasses import dataclass
from numbers import Real
from typing import Literal

from vision.src.core.contracts import CLASS_NAMES
from vision.src.detection.dataset_v2.geometry import (
    Box,
    validate_image_size,
    validate_xyxy,
)


Source = Literal["TrashCan", "RiSID", "WPBB"]
OriginalSplit = Literal["train", "val", "unspecified"]
FinalSplit = Literal["train", "val", "test", "quarantine"]
Outcome = Literal[
    "valid",
    "clipped",
    "review_required",
    "rejected",
]
BoxFormat = Literal["coco_xywh", "voc_xyxy"]


SOURCES = ("TrashCan", "RiSID", "WPBB")
ORIGINAL_SPLITS = ("train", "val", "unspecified")
FINAL_SPLITS = ("train", "val", "test", "quarantine")
OUTCOMES = ("valid", "clipped", "review_required", "rejected")
BOX_FORMATS = ("coco_xywh", "voc_xyxy")


def _require_text(value: str, name: str) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if not value.strip():
        raise ValueError(f"{name} must not be empty")


def _require_source(source: Source) -> None:
    if not isinstance(source, str):
        raise TypeError("source must be a string")
    if source not in SOURCES:
        raise ValueError("Unsupported dataset source")


def _require_qualified_id(
    value: str,
    source: Source,
    name: str,
) -> None:
    _require_text(value, name)
    prefix = f"{source}:"
    if not value.startswith(prefix) or not value[len(prefix):].strip():
        raise ValueError(f"{name} must be source-qualified")


def _require_relative_path(value: str) -> None:
    _require_text(value, "relative_path")
    if (
        value.startswith("/")
        or "\\" in value
        or ":" in value
        or any(part in ("", ".", "..") for part in value.split("/"))
    ):
        raise ValueError("relative_path must be a safe POSIX relative path")


def _require_finite(value: Real, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    if not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite")


@dataclass(frozen=True, slots=True)
class ImageRecord:
    """Source image and acquisition-group metadata."""

    source: Source
    image_id: str
    relative_path: str
    width: int
    height: int
    group_id: str
    original_split: OriginalSplit
    image_sha256: str
    frame_index: int | None = None
    final_split: FinalSplit | None = None
    quarantine_reason: str | None = None

    def __post_init__(self) -> None:
        _require_source(self.source)
        _require_qualified_id(self.image_id, self.source, "image_id")
        _require_qualified_id(self.group_id, self.source, "group_id")
        _require_relative_path(self.relative_path)
        validate_image_size(self.width, self.height)

        if self.original_split not in ORIGINAL_SPLITS:
            raise ValueError("Unsupported original split")

        if not isinstance(self.image_sha256, str):
            raise TypeError("image_sha256 must be a string")
        if re.fullmatch(r"[0-9a-f]{64}", self.image_sha256) is None:
            raise ValueError("image_sha256 must be lowercase SHA-256 hex")

        if self.frame_index is not None:
            if type(self.frame_index) is not int:
                raise TypeError("frame_index must be an integer")
            if self.frame_index < 0:
                raise ValueError("frame_index must be non-negative")

        if self.final_split is not None:
            if self.final_split not in FINAL_SPLITS:
                raise ValueError("Unsupported final split")

        if self.final_split == "quarantine":
            _require_text(self.quarantine_reason, "quarantine_reason")
        elif self.quarantine_reason is not None:
            raise ValueError(
                "quarantine_reason requires final_split='quarantine'"
            )


@dataclass(frozen=True, slots=True)
class AnnotationRecord:
    """Original annotation and its reviewed geometry outcome."""

    source: Source
    annotation_id: str
    image_id: str
    source_class_name: str
    class_id: int | None
    raw_annotation: str
    original_box: Box | None
    original_box_format: BoxFormat | None
    original_polygons: tuple[tuple[float, ...], ...] | None
    visible_box: Box | None
    outcome: Outcome
    reason: str | None = None
    boundary_contact: tuple[bool, bool, bool, bool] = (
        False, False, False, False
    )
    detection_eligible: bool = False
    segmentation_eligible: bool = False

    def __post_init__(self) -> None:
        _require_source(self.source)
        _require_qualified_id(
            self.annotation_id, self.source, "annotation_id"
        )
        _require_qualified_id(self.image_id, self.source, "image_id")
        _require_text(self.source_class_name, "source_class_name")
        _require_text(self.raw_annotation, "raw_annotation")

        if self.class_id is not None:
            if type(self.class_id) is not int:
                raise TypeError("class_id must be an integer")
            if not 0 <= self.class_id < len(CLASS_NAMES):
                raise ValueError("class_id is outside the supported range")

        if (self.original_box is None) != (
            self.original_box_format is None
        ):
            raise ValueError(
                "original_box and original_box_format must occur together"
            )

        if self.original_box_format is not None:
            if self.original_box_format not in BOX_FORMATS:
                raise ValueError("Unsupported original box format")

        if self.original_box is not None:
            if not isinstance(self.original_box, tuple):
                raise TypeError("original_box must be a tuple")
            if len(self.original_box) != 4:
                raise ValueError("original_box must contain four values")
            for index, value in enumerate(self.original_box):
                _require_finite(value, f"original_box[{index}]")

        if self.original_polygons is not None:
            if not isinstance(self.original_polygons, tuple):
                raise TypeError("original_polygons must be a tuple")
            if not self.original_polygons:
                raise ValueError("original_polygons must not be empty")
            for index, polygon in enumerate(self.original_polygons):
                if not isinstance(polygon, tuple):
                    raise TypeError("Each polygon must be a tuple")
                if len(polygon) < 6 or len(polygon) % 2:
                    raise ValueError(
                        f"original_polygons[{index}] has invalid length"
                    )
                for offset, value in enumerate(polygon):
                    _require_finite(
                        value,
                        f"original_polygons[{index}][{offset}]",
                    )

        if self.visible_box is not None:
            if not isinstance(self.visible_box, tuple):
                raise TypeError("visible_box must be a tuple")
            validate_xyxy(self.visible_box)

        if self.outcome not in OUTCOMES:
            raise ValueError("Unsupported annotation outcome")

        if self.outcome == "valid":
            if self.reason is not None:
                raise ValueError("Valid annotations cannot have a reason")
        else:
            _require_text(self.reason, "reason")

        if (
            not isinstance(self.boundary_contact, tuple)
            or len(self.boundary_contact) != 4
            or any(type(value) is not bool for value in self.boundary_contact)
        ):
            raise TypeError("boundary_contact must be four booleans")

        if type(self.detection_eligible) is not bool:
            raise TypeError("detection_eligible must be boolean")
        if type(self.segmentation_eligible) is not bool:
            raise TypeError("segmentation_eligible must be boolean")

        accepted = self.outcome in ("valid", "clipped")

        if accepted:
            if self.class_id is None or self.visible_box is None:
                raise ValueError(
                    "Accepted annotations require a class and visible box"
                )

        if self.detection_eligible and not accepted:
            raise ValueError(
                "Unresolved annotations cannot be exported for detection"
            )

        if self.segmentation_eligible:
            if (
                not self.detection_eligible
                or self.original_polygons is None
            ):
                raise ValueError(
                    "Segmentation eligibility requires accepted polygons"
                )
