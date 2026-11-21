
"""Inference pipeline coordinator for HydroTrashVision.

Connects restoration, detection, and mass estimation through
lightweight interfaces, without depending on concrete ML models.
"""

from collections.abc import Sequence
from time import perf_counter
from typing import Protocol

import numpy as np

from vision.src.core.contracts import (
    Detection,
    FramePacket,
    FrameResult,
    MassEstimate,
)


class Restorer(Protocol):
    """Interface for an optional image restoration component."""

    method_id: str

    def restore(self, image: np.ndarray) -> np.ndarray:
        """Return a restored BGR uint8 image."""
        ...


class Detector(Protocol):
    """Interface for an object detection component."""

    detector_id: str

    def detect(self, image: np.ndarray) -> Sequence[Detection]:
        """Return detections in original-image coordinates."""
        ...


class MassEstimator(Protocol):
    """Interface for a per-object mass estimation component."""

    mass_estimator_id: str

    def estimate(
        self,
        detection: Detection,
        frame: FramePacket,
    ) -> MassEstimate:
        """Return a physically justified mass state."""
        ...


def _component_id(
    component: object,
    method_name: str,
    identifier_name: str,
) -> str:
    """Validate a component interface and its identifier."""
    method = getattr(component, method_name, None)

    if not callable(method):
        raise TypeError(
            f"Component must provide a callable {method_name} method"
        )

    identifier = getattr(component, identifier_name, None)

    if not isinstance(identifier, str):
        raise TypeError(f"{identifier_name} must be a string")

    if not identifier.strip():
        raise ValueError(f"{identifier_name} must not be empty")

    return identifier


def _validate_restored_image(
    image: np.ndarray,
    frame: FramePacket,
) -> None:
    """Validate restoration output before running detection."""
    if not isinstance(image, np.ndarray):
        raise TypeError("Restorer must return a NumPy array")

    if image.dtype != np.uint8:
        raise TypeError("Restored image must have uint8 dtype")

    if image.shape != frame.image.shape:
        raise ValueError(
            "Restored image must match original frame dimensions"
        )


def _validate_detections(
    detections: Sequence[Detection],
    frame: FramePacket,
) -> tuple[Detection, ...]:
    """Validate all detector outputs before mass estimation."""
    if not isinstance(detections, Sequence):
        raise TypeError("Detector must return a sequence")

    validated = tuple(detections)

    for index, detection in enumerate(validated):
        if not isinstance(detection, Detection):
            raise TypeError(
                f"Detection at index {index} has an invalid type"
            )

        _, _, x2, y2 = detection.bbox_xyxy

        if x2 > frame.width or y2 > frame.height:
            raise ValueError(
                f"Detection at index {index} exceeds frame bounds"
            )

    return validated


class Pipeline:
    """Coordinate restoration, detection, and mass estimation."""

    def __init__(
        self,
        *,
        detector: Detector,
        mass_estimator: MassEstimator,
        restorer: Restorer | None = None,
    ) -> None:
        self._detector_id = _component_id(
            detector, "detect", "detector_id"
        )
        self._mass_estimator_id = _component_id(
            mass_estimator, "estimate", "mass_estimator_id"
        )

        if restorer is None:
            self._restoration_method_id = "identity"
        else:
            self._restoration_method_id = _component_id(
                restorer, "restore", "method_id"
            )

            if self._restoration_method_id == "identity":
                raise ValueError(
                    "A concrete restorer cannot use the identity ID"
                )

        self._restorer = restorer
        self._detector = detector
        self._mass_estimator = mass_estimator

    def run(self, frame: FramePacket) -> FrameResult:
        """Process one frame and return a validated result."""
        if not isinstance(frame, FramePacket):
            raise TypeError("frame must be a FramePacket")

        total_start = perf_counter()

        # Stage A: optional restoration.
        if self._restorer is None:
            restored_image = None
            effective_image = frame.image
            restoration_ms = 0.0
        else:
            stage_start = perf_counter()

            restored_image = self._restorer.restore(
                frame.image.copy()
            )

            _validate_restored_image(restored_image, frame)

            effective_image = restored_image
            restoration_ms = (
                perf_counter() - stage_start
            ) * 1000.0

        # Stage B: detection.
        stage_start = perf_counter()

        raw_detections = self._detector.detect(
            effective_image.copy()
        )

        detections = _validate_detections(
            raw_detections, frame
        )

        detection_ms = (
            perf_counter() - stage_start
        ) * 1000.0

        # Stage C: per-object mass estimation.
        stage_start = perf_counter()

        object_results: list[
            tuple[Detection, MassEstimate]
        ] = []

        for detection in detections:
            if detection.class_name == "rov":
                mass = MassEstimate(
                    status="unavailable",
                    reason="ROV is not waste",
                )
            else:
                mass = self._mass_estimator.estimate(
                    detection, frame
                )

                if not isinstance(mass, MassEstimate):
                    raise TypeError(
                        "Mass estimator must return a MassEstimate"
                    )

            object_results.append((detection, mass))

        mass_ms = (
            perf_counter() - stage_start
        ) * 1000.0

        total_ms = (
            perf_counter() - total_start
        ) * 1000.0

        return FrameResult(
            frame=frame,
            restoration_method_id=self._restoration_method_id,
            detector_id=self._detector_id,
            mass_estimator_id=self._mass_estimator_id,
            restored_image=restored_image,
            object_results=tuple(object_results),
            timings_ms={
                "restoration": restoration_ms,
                "detection": detection_ms,
                "mass": mass_ms,
                "total": total_ms,
            },
        )