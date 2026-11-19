
"""Shared data contracts for the HydroTrashVision pipeline.

This module defines the validated data structures exchanged between
image restoration, object detection, and mass estimation.

It does not perform inference, image loading, or file export.
"""

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from numbers import Real
from types import MappingProxyType
from typing import Literal

import numpy as np


CLASS_NAMES = (
    "trash_etc",
    "trash_fabric",
    "trash_fishing_gear",
    "trash_metal",
    "trash_paper",
    "trash_plastic",
    "trash_rubber",
    "trash_wood",
    "rov",
)

MASS_STATUSES = ("unavailable", "uncalibrated", "estimated")

TIMING_KEYS = frozenset(
    {"restoration", "detection", "mass", "total"}
)


def _require_text(value: str, name: str) -> None:
    """Validate a required non-empty string."""
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")

    if not value.strip():
        raise ValueError(f"{name} must not be empty or whitespace")


def _require_real(
    value: Real,
    name: str,
    *,
    allow_zero: bool = True,
) -> float:
    """Validate a finite, non-negative real number."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")

    number = float(value)

    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")

    if number < 0 or (not allow_zero and number == 0):
        raise ValueError(f"{name} must be non-negative or positive")

    return number


def _readonly_bgr_image(image: np.ndarray, name: str) -> np.ndarray:
    """Validate and defensively copy a BGR uint8 image.

    The immutable bytes backing prevents ordinary pixel writes and
    prevents re-enabling write access through NumPy flags.
    """
    if not isinstance(image, np.ndarray):
        raise TypeError(f"{name} must be a NumPy array")

    if image.dtype != np.uint8:
        raise TypeError(f"{name} must have uint8 dtype")

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"{name} must have shape (H, W, 3)")

    if image.shape[0] == 0 or image.shape[1] == 0:
        raise ValueError(f"{name} must have positive dimensions")

    immutable_data = image.tobytes(order="C")

    return np.frombuffer(
        immutable_data,
        dtype=np.uint8,
    ).reshape(image.shape)


@dataclass(frozen=True, eq=False, slots=True)
class FramePacket:
    """An original image frame and its source metadata.

    Images use the external pipeline convention: BGR uint8, HWC.
    """

    image: np.ndarray
    frame_id: str
    source_id: str
    source_type: Literal["image", "video"]
    frame_index: int | None = None
    timestamp_s: float | None = None

    def __post_init__(self) -> None:
        _require_text(self.frame_id, "frame_id")
        _require_text(self.source_id, "source_id")

        if not isinstance(self.source_type, str):
            raise TypeError("source_type must be a string")

        if self.source_type not in ("image", "video"):
            raise ValueError("source_type must be 'image' or 'video'")

        if self.frame_index is not None:
            if type(self.frame_index) is not int:
                raise TypeError("frame_index must be an integer")

            if self.frame_index < 0:
                raise ValueError("frame_index must be non-negative")

        if self.source_type == "image" and self.frame_index is not None:
            raise ValueError("Image sources cannot have a frame_index")

        if self.source_type == "video" and self.frame_index is None:
            raise ValueError("Video sources require a frame_index")

        if self.timestamp_s is not None:
            timestamp = _require_real(self.timestamp_s, "timestamp_s")
            object.__setattr__(self, "timestamp_s", timestamp)

        if self.source_type == "image" and self.timestamp_s is not None:
            raise ValueError("Image sources cannot have a timestamp_s")

        object.__setattr__(
            self,
            "image",
            _readonly_bgr_image(self.image, "image"),
        )

    @property
    def width(self) -> int:
        """Original frame width in pixels."""
        return int(self.image.shape[1])

    @property
    def height(self) -> int:
        """Original frame height in pixels."""
        return int(self.image.shape[0])


@dataclass(frozen=True, eq=False, slots=True)
class Detection:
    """A single detected object in original-frame coordinates."""

    class_id: int
    confidence: float
    bbox_xyxy: tuple[float, float, float, float]

    def __post_init__(self) -> None:
        if type(self.class_id) is not int:
            raise TypeError("class_id must be an integer")

        if not 0 <= self.class_id < len(CLASS_NAMES):
            raise ValueError("class_id is outside the supported class range")

        confidence = _require_real(self.confidence, "confidence")

        if confidence > 1:
            raise ValueError("confidence must be between 0 and 1")

        object.__setattr__(self, "confidence", confidence)

        if not isinstance(self.bbox_xyxy, tuple):
            raise TypeError("bbox_xyxy must be a tuple")

        if len(self.bbox_xyxy) != 4:
            raise ValueError("bbox_xyxy must contain four coordinates")

        x1, y1, x2, y2 = (
            _require_real(value, f"bbox_xyxy[{index}]")
            for index, value in enumerate(self.bbox_xyxy)
        )

        if x1 >= x2 or y1 >= y2:
            raise ValueError("bbox_xyxy must have positive area")

        object.__setattr__(
            self,
            "bbox_xyxy",
            (x1, y1, x2, y2),
        )

    @property
    def class_name(self) -> str:
        """Class name from the project's fixed nine-class scheme."""
        return CLASS_NAMES[self.class_id]


@dataclass(frozen=True, eq=False, slots=True)
class MassEstimate:
    """A mass estimate or an explicit reason for its absence."""

    status: Literal["unavailable", "uncalibrated", "estimated"]
    value_g: float | None = None
    method_id: str | None = None
    calibration_id: str | None = None
    reason: str | None = None
    assumptions: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _require_text(self.status, "status")

        if self.status not in MASS_STATUSES:
            raise ValueError("Unsupported mass estimation status")

        for name in ("method_id", "calibration_id", "reason"):
            value = getattr(self, name)

            if value is not None:
                _require_text(value, name)

        if not isinstance(self.assumptions, tuple):
            raise TypeError("assumptions must be a tuple")

        for index, assumption in enumerate(self.assumptions):
            _require_text(assumption, f"assumptions[{index}]")

        if self.status == "estimated":
            if self.value_g is None:
                raise ValueError("Estimated mass requires value_g")

            mass = _require_real(
                self.value_g,
                "value_g",
                allow_zero=False,
            )

            if self.method_id is None or self.calibration_id is None:
                raise ValueError(
                    "Estimated mass requires method_id and calibration_id"
                )

            if self.reason is not None:
                raise ValueError("Estimated mass cannot have an absence reason")

            object.__setattr__(self, "value_g", mass)

        else:
            if self.value_g is not None:
                raise ValueError(
                    "Unavailable or uncalibrated mass must have value_g=None"
                )

            if self.calibration_id is not None:
                raise ValueError(
                    "Unavailable or uncalibrated mass cannot have calibration_id"
                )

            if self.reason is None:
                raise ValueError(
                    "Unavailable or uncalibrated mass requires a reason"
                )

            if self.status == "uncalibrated" and self.method_id is None:
                raise ValueError(
                    "Uncalibrated mass requires a method_id"
                )


@dataclass(frozen=True, eq=False, slots=True)
class FrameResult:
    """Validated output for one complete pipeline frame."""

    frame: FramePacket
    restoration_method_id: str
    detector_id: str
    mass_estimator_id: str
    restored_image: np.ndarray | None = None
    object_results: tuple[tuple[Detection, MassEstimate], ...] = ()
    timings_ms: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.frame, FramePacket):
            raise TypeError("frame must be a FramePacket")

        _require_text(self.restoration_method_id, "restoration_method_id")
        _require_text(self.detector_id, "detector_id")
        _require_text(self.mass_estimator_id, "mass_estimator_id")

        if self.restoration_method_id == "identity":
            if self.restored_image is not None:
                raise ValueError(
                    "Identity restoration requires restored_image=None"
                )

        else:
            if self.restored_image is None:
                raise ValueError(
                    "A non-identity restoration requires restored_image"
                )

            restored = _readonly_bgr_image(
                self.restored_image,
                "restored_image",
            )

            if restored.shape != self.frame.image.shape:
                raise ValueError(
                    "Restored image must match original frame dimensions"
                )

            object.__setattr__(self, "restored_image", restored)

        if not isinstance(self.object_results, tuple):
            raise TypeError("object_results must be a tuple")

        for index, pair in enumerate(self.object_results):
            if not isinstance(pair, tuple) or len(pair) != 2:
                raise TypeError(
                    f"object_results[{index}] must be a two-item tuple"
                )

            detection, mass_estimate = pair

            if not isinstance(detection, Detection):
                raise TypeError(
                    f"object_results[{index}] requires a Detection"
                )

            if not isinstance(mass_estimate, MassEstimate):
                raise TypeError(
                    f"object_results[{index}] requires a MassEstimate"
                )

            x1, y1, x2, y2 = detection.bbox_xyxy

            if x2 > self.frame.width or y2 > self.frame.height:
                raise ValueError(
                    f"object_results[{index}] has a box outside the frame"
                )

            if detection.class_name == "rov":
                if mass_estimate.status != "unavailable":
                    raise ValueError(
                        "ROV detections cannot have a mass estimate"
                    )

        if not isinstance(self.timings_ms, Mapping):
            raise TypeError("timings_ms must be a mapping")

        validated_timings = {}

        for name, value in self.timings_ms.items():
            _require_text(name, "timing key")

            if name not in TIMING_KEYS:
                raise ValueError(f"Unsupported timing key: {name}")

            validated_timings[name] = _require_real(
                value,
                f"timings_ms[{name}]",
            )

        object.__setattr__(
            self,
            "timings_ms",
            MappingProxyType(validated_timings),
        )