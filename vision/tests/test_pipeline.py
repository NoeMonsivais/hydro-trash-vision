
"""Unit tests for the HydroTrashVision pipeline coordinator.

Tests use deterministic doubles instead of real ML models,
checkpoints, datasets, or physical calibration hardware.
"""

import math

import numpy as np
import pytest

from vision.src.core.contracts import (
    Detection,
    FramePacket,
    MassEstimate,
)
from vision.src.pipeline import Pipeline


# ------------------------------------------------------------
# Test helpers
# ------------------------------------------------------------

def make_frame() -> FramePacket:
    """Create a small, valid image frame."""
    image = np.full((12, 20, 3), 10, dtype=np.uint8)

    return FramePacket(
        image=image,
        frame_id="frame_0001",
        source_id="test_source",
        source_type="image",
    )


def make_detection(
    *,
    class_id: int = 5,
    bbox_xyxy: tuple[float, float, float, float] = (
        2.0, 3.0, 10.0, 8.0
    ),
) -> Detection:
    """Create a valid detection with configurable class and box."""
    return Detection(
        class_id=class_id,
        confidence=0.9,
        bbox_xyxy=bbox_xyxy,
    )


# ------------------------------------------------------------
# Test doubles
# ------------------------------------------------------------

class FakeRestorer:
    """Restore images deterministically and record calls."""

    method_id = "fake_restoration"

    def __init__(
        self,
        *,
        events=None,
        result=None,
        mutate_input=False,
    ):
        self.events = events if events is not None else []
        self.result = result
        self.mutate_input = mutate_input
        self.received_images = []

    def restore(self, image: np.ndarray) -> np.ndarray:
        self.events.append("restore")
        self.received_images.append(image.copy())

        if self.mutate_input:
            image[:] = 0

        if self.result is not None:
            return self.result

        return np.full_like(image, 128)


class FakeDetector:
    """Return predetermined detections and record input images."""

    detector_id = "fake_detector"

    def __init__(
        self,
        detections=(),
        *,
        events=None,
        mutate_input=False,
    ):
        self.detections = detections
        self.events = events if events is not None else []
        self.mutate_input = mutate_input
        self.received_images = []

    def detect(self, image: np.ndarray):
        self.events.append("detect")
        self.received_images.append(image.copy())

        if self.mutate_input:
            image[:] = 0

        return self.detections


class FakeMassEstimator:
    """Return uncalibrated mass states and record requests."""

    mass_estimator_id = "area_calibrated"

    def __init__(self, *, events=None):
        self.events = events if events is not None else []
        self.calls = []

    def estimate(
        self,
        detection: Detection,
        frame: FramePacket,
    ) -> MassEstimate:
        self.events.append(f"estimate:{detection.class_id}")
        self.calls.append((detection, frame))

        return MassEstimate(
            status="uncalibrated",
            method_id=self.mass_estimator_id,
            reason=f"No calibration for class {detection.class_id}",
        )


# ------------------------------------------------------------
# Pipeline behavior
# ------------------------------------------------------------

def test_identity_pipeline_uses_original_image():
    """Without restoration, detection uses the original image."""
    frame = make_frame()
    detection = make_detection()
    detector = FakeDetector([detection])
    estimator = FakeMassEstimator()

    pipeline = Pipeline(
        detector=detector,
        mass_estimator=estimator,
    )

    result = pipeline.run(frame)

    assert result.frame is frame
    assert result.restoration_method_id == "identity"
    assert result.restored_image is None
    assert result.detector_id == "fake_detector"
    assert result.mass_estimator_id == "area_calibrated"

    np.testing.assert_array_equal(
        detector.received_images[0],
        frame.image,
    )

    assert result.object_results[0][0] is detection
    assert result.object_results[0][1].status == "uncalibrated"
    assert result.object_results[0][1].value_g is None
    assert estimator.calls[0] == (detection, frame)


def test_stage_order_and_restored_detector_input():
    """Stages run in order and detection sees restored pixels."""
    events = []
    frame = make_frame()

    restorer = FakeRestorer(events=events)
    detector = FakeDetector(
        [make_detection()],
        events=events,
    )
    estimator = FakeMassEstimator(events=events)

    pipeline = Pipeline(
        detector=detector,
        mass_estimator=estimator,
        restorer=restorer,
    )

    result = pipeline.run(frame)

    assert events == ["restore", "detect", "estimate:5"]
    assert result.restoration_method_id == "fake_restoration"

    assert np.all(detector.received_images[0] == 128)
    assert np.all(result.restored_image == 128)
    assert np.all(frame.image == 10)


def test_restorer_cannot_mutate_original_frame():
    """Restorer input mutations must not affect FramePacket."""
    frame = make_frame()
    restorer = FakeRestorer(mutate_input=True)

    pipeline = Pipeline(
        detector=FakeDetector(),
        mass_estimator=FakeMassEstimator(),
        restorer=restorer,
    )

    pipeline.run(frame)

    assert np.all(restorer.received_images[0] == 10)
    assert np.all(frame.image == 10)
    assert not frame.image.flags.writeable


def test_detector_cannot_mutate_restored_result():
    """Detector input mutations must not change stored restoration."""
    frame = make_frame()
    detector = FakeDetector(mutate_input=True)

    pipeline = Pipeline(
        detector=detector,
        mass_estimator=FakeMassEstimator(),
        restorer=FakeRestorer(),
    )

    result = pipeline.run(frame)

    assert np.all(detector.received_images[0] == 128)
    assert np.all(result.restored_image == 128)
    assert np.all(frame.image == 10)
    assert not result.restored_image.flags.writeable


def test_empty_detections_skip_mass_estimation():
    """A frame without detections produces an empty result."""
    estimator = FakeMassEstimator()

    pipeline = Pipeline(
        detector=FakeDetector(),
        mass_estimator=estimator,
    )

    result = pipeline.run(make_frame())

    assert result.object_results == ()
    assert estimator.calls == []


def test_multiple_detections_keep_mass_associations():
    """Each detection must retain its own mass result."""
    frame = make_frame()
    plastic = make_detection(class_id=5)
    metal = make_detection(class_id=3)

    estimator = FakeMassEstimator()
    pipeline = Pipeline(
        detector=FakeDetector([plastic, metal]),
        mass_estimator=estimator,
    )

    result = pipeline.run(frame)

    assert len(result.object_results) == 2
    assert result.object_results[0][0] is plastic
    assert result.object_results[1][0] is metal

    assert result.object_results[0][1].reason == (
        "No calibration for class 5"
    )
    assert result.object_results[1][1].reason == (
        "No calibration for class 3"
    )

    assert estimator.calls == [
        (plastic, frame),
        (metal, frame),
    ]


def test_rov_skips_mass_estimator():
    """ROV is detected but never treated as waste."""
    rov = make_detection(class_id=8)
    plastic = make_detection(class_id=5)
    estimator = FakeMassEstimator()

    pipeline = Pipeline(
        detector=FakeDetector([rov, plastic]),
        mass_estimator=estimator,
    )

    result = pipeline.run(make_frame())

    rov_mass = result.object_results[0][1]
    plastic_mass = result.object_results[1][1]

    assert rov_mass.status == "unavailable"
    assert rov_mass.value_g is None
    assert plastic_mass.status == "uncalibrated"
    assert len(estimator.calls) == 1
    assert estimator.calls[0][0] is plastic


# ------------------------------------------------------------
# Input validation and error propagation
# ------------------------------------------------------------

def test_invalid_frame_is_rejected():
    """Pipeline.run requires a FramePacket."""
    pipeline = Pipeline(
        detector=FakeDetector(),
        mass_estimator=FakeMassEstimator(),
    )

    with pytest.raises(TypeError, match="FramePacket"):
        pipeline.run(None)


def test_out_of_bounds_box_fails_before_mass():
    """Invalid coordinates must fail before any mass estimation."""
    valid = make_detection()
    invalid = make_detection(
        bbox_xyxy=(2.0, 3.0, 21.0, 8.0)
    )
    estimator = FakeMassEstimator()

    pipeline = Pipeline(
        detector=FakeDetector([valid, invalid]),
        mass_estimator=estimator,
    )

    with pytest.raises(ValueError, match="frame bounds"):
        pipeline.run(make_frame())

    assert estimator.calls == []


def test_invalid_detection_object_is_rejected():
    """Detector outputs must contain Detection instances."""
    estimator = FakeMassEstimator()

    pipeline = Pipeline(
        detector=FakeDetector([object()]),
        mass_estimator=estimator,
    )

    with pytest.raises(TypeError, match="invalid type"):
        pipeline.run(make_frame())

    assert estimator.calls == []


def test_detector_must_return_sequence():
    """A generator is not a valid detection sequence."""
    detections = (item for item in [make_detection()])

    pipeline = Pipeline(
        detector=FakeDetector(detections),
        mass_estimator=FakeMassEstimator(),
    )

    with pytest.raises(TypeError, match="sequence"):
        pipeline.run(make_frame())


@pytest.mark.parametrize(
    "invalid_image",
    [
        "not an image",
        np.zeros((12, 20, 3), dtype=np.float32),
        np.zeros((8, 20, 3), dtype=np.uint8),
    ],
)
def test_invalid_restoration_fails_before_detection(invalid_image):
    """Invalid restoration output must not reach the detector."""
    detector = FakeDetector()

    pipeline = Pipeline(
        detector=detector,
        mass_estimator=FakeMassEstimator(),
        restorer=FakeRestorer(result=invalid_image),
    )

    with pytest.raises((TypeError, ValueError)):
        pipeline.run(make_frame())

    assert detector.received_images == []


def test_invalid_mass_output_is_rejected():
    """Mass estimators must return MassEstimate instances."""

    class InvalidMassEstimator:
        mass_estimator_id = "invalid_mass"

        def estimate(self, detection, frame):
            return None

    pipeline = Pipeline(
        detector=FakeDetector([make_detection()]),
        mass_estimator=InvalidMassEstimator(),
    )

    with pytest.raises(TypeError, match="MassEstimate"):
        pipeline.run(make_frame())


def test_detector_exception_propagates():
    """Component failures must not produce successful results."""

    class FailingDetector:
        detector_id = "failing_detector"

        def detect(self, image):
            raise RuntimeError("Simulated detection failure")

    pipeline = Pipeline(
        detector=FailingDetector(),
        mass_estimator=FakeMassEstimator(),
    )

    with pytest.raises(
        RuntimeError,
        match="Simulated detection failure",
    ):
        pipeline.run(make_frame())


def test_component_without_required_method_is_rejected():
    """Dependencies must expose their required callable methods."""

    class InvalidDetector:
        detector_id = "invalid_detector"

    with pytest.raises(TypeError, match="callable detect"):
        Pipeline(
            detector=InvalidDetector(),
            mass_estimator=FakeMassEstimator(),
        )


def test_concrete_restorer_cannot_claim_identity():
    """Identity is reserved for the absence of a restorer."""
    restorer = FakeRestorer()
    restorer.method_id = "identity"

    with pytest.raises(ValueError, match="identity ID"):
        Pipeline(
            detector=FakeDetector(),
            mass_estimator=FakeMassEstimator(),
            restorer=restorer,
        )


# ------------------------------------------------------------
# Timing measurements
# ------------------------------------------------------------

def test_timings_are_valid_and_measured():
    """Pipeline timings must be finite and non-negative."""
    pipeline = Pipeline(
        detector=FakeDetector([make_detection()]),
        mass_estimator=FakeMassEstimator(),
        restorer=FakeRestorer(),
    )

    result = pipeline.run(make_frame())
    timings = result.timings_ms

    assert set(timings) == {
        "restoration",
        "detection",
        "mass",
        "total",
    }

    assert all(
        math.isfinite(value) and value >= 0
        for value in timings.values()
    )

    assert timings["total"] >= (
        timings["restoration"]
        + timings["detection"]
        + timings["mass"]
    )


def test_identity_restoration_has_zero_time():
    """Skipped restoration must report zero milliseconds."""
    pipeline = Pipeline(
        detector=FakeDetector(),
        mass_estimator=FakeMassEstimator(),
    )

    result = pipeline.run(make_frame())

    assert result.timings_ms["restoration"] == 0.0