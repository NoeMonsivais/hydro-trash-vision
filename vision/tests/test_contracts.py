
"""Unit tests for the HydroTrashVision data contracts.

These tests validate data structures and invariants without
loading neural networks, datasets, or physical hardware.
"""

from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np
import pytest
import yaml

from vision.src.core.contracts import (
    CLASS_NAMES,
    Detection,
    FramePacket,
    FrameResult,
    MassEstimate,
)
from vision.src.detection.convert_annotations import FINAL_CLASSES


def make_frame(
    *,
    image=None,
    source_type="image",
    frame_index=None,
    timestamp_s=None,
):
    """Create a valid frame with configurable properties."""
    if image is None:
        image = np.zeros((12, 20, 3), dtype=np.uint8)

    return FramePacket(
        image=image,
        frame_id="frame_0001",
        source_id="test_source",
        source_type=source_type,
        frame_index=frame_index,
        timestamp_s=timestamp_s,
    )


def make_detection(
    *,
    class_id=5,
    confidence=0.9,
    bbox_xyxy=(2.0, 3.0, 10.0, 8.0),
):
    """Create a valid plastic detection."""
    return Detection(
        class_id=class_id,
        confidence=confidence,
        bbox_xyxy=bbox_xyxy,
    )


def make_unavailable_mass():
    """Create a valid unavailable mass result."""
    return MassEstimate(
        status="unavailable",
        reason="Physical mass is not available",
    )


def make_result(
    *,
    frame=None,
    restoration_method_id="identity",
    restored_image=None,
    object_results=(),
    timings_ms=None,
):
    """Create a valid complete frame result."""
    if frame is None:
        frame = make_frame()

    if timings_ms is None:
        timings_ms = {}

    return FrameResult(
        frame=frame,
        restoration_method_id=restoration_method_id,
        detector_id="yolov8s",
        mass_estimator_id="area_calibrated",
        restored_image=restored_image,
        object_results=object_results,
        timings_ms=timings_ms,
    )


# ------------------------------------------------------------
# Class scheme compatibility
# ------------------------------------------------------------

def test_class_names_match_detection_converter():
    """The inference class scheme must match training labels."""
    assert tuple(FINAL_CLASSES) == CLASS_NAMES


def test_class_names_match_yolo_config():
    """The class scheme must match the training YAML."""
    config_path = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "yolo_train_config.yaml"
    )

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    assert config["nc"] == 9
    assert tuple(config["names"]) == CLASS_NAMES


# ------------------------------------------------------------
# FramePacket
# ------------------------------------------------------------

def test_valid_image_frame():
    """Image metadata and dimensions must remain consistent."""
    frame = make_frame()

    assert frame.width == 20
    assert frame.height == 12
    assert frame.image.shape == (12, 20, 3)
    assert frame.image.dtype == np.uint8
    assert frame.source_type == "image"
    assert frame.frame_index is None
    assert frame.timestamp_s is None


def test_valid_video_frame():
    """Video frames require a zero-based frame index."""
    frame = make_frame(
        source_type="video",
        frame_index=0,
        timestamp_s=0.0,
    )

    assert frame.frame_index == 0
    assert frame.timestamp_s == 0.0


@pytest.mark.parametrize(
    "image",
    [
        np.zeros((4, 5), dtype=np.uint8),
        np.zeros((4, 5, 1), dtype=np.uint8),
        np.zeros((4, 5, 4), dtype=np.uint8),
        np.zeros((0, 5, 3), dtype=np.uint8),
        np.zeros((4, 5, 3), dtype=np.float32),
    ],
)
def test_invalid_frame_images(image):
    """Invalid image shapes or dtypes must be rejected."""
    with pytest.raises((TypeError, ValueError)):
        make_frame(image=image)


def test_frame_rejects_none_image():
    """FramePacket must reject a missing image."""
    with pytest.raises(TypeError):
        FramePacket(
            image=None,
            frame_id="frame_0001",
            source_id="test_source",
            source_type="image",
        )


@pytest.mark.parametrize(
    "source_type, frame_index, timestamp_s",
    [
        ("camera", None, None),
        ("image", 0, None),
        ("image", None, 0.0),
        ("video", None, None),
        ("video", -1, None),
        ("video", True, None),
        ("video", 1.5, None),
        ("video", 0, -0.1),
        ("video", 0, float("nan")),
        ("video", 0, float("inf")),
        ("video", 0, False),
    ],
)
def test_invalid_frame_metadata(
    source_type,
    frame_index,
    timestamp_s,
):
    """Invalid source metadata must be rejected."""
    with pytest.raises((TypeError, ValueError)):
        make_frame(
            source_type=source_type,
            frame_index=frame_index,
            timestamp_s=timestamp_s,
        )


@pytest.mark.parametrize(
    "frame_id, source_id",
    [
        ("", "source"),
        ("   ", "source"),
        ("frame", ""),
        ("frame", "   "),
    ],
)
def test_invalid_frame_identifiers(frame_id, source_id):
    """Frame and source identifiers must not be empty."""
    image = np.zeros((4, 5, 3), dtype=np.uint8)

    with pytest.raises(ValueError):
        FramePacket(
            image=image,
            frame_id=frame_id,
            source_id=source_id,
            source_type="image",
        )


def test_frame_image_is_defensively_copied():
    """Changing the input array must not change the packet."""
    original = np.zeros((4, 5, 3), dtype=np.uint8)
    frame = make_frame(image=original)

    original[0, 0, 0] = 255

    assert frame.image[0, 0, 0] == 0


def test_frame_image_is_readonly():
    """Packet pixels must not be writable."""
    frame = make_frame()

    assert not frame.image.flags.writeable

    with pytest.raises(ValueError):
        frame.image[0, 0, 0] = 255

    with pytest.raises(ValueError):
        frame.image.setflags(write=True)


def test_frame_fields_are_frozen():
    """Frame metadata must not be reassigned."""
    frame = make_frame()

    with pytest.raises(FrozenInstanceError):
        frame.frame_id = "another_frame"


# ------------------------------------------------------------
# Detection
# ------------------------------------------------------------

def test_valid_detection():
    """A valid detection must expose its correct class name."""
    detection = make_detection()

    assert detection.class_id == 5
    assert detection.class_name == "trash_plastic"
    assert detection.confidence == 0.9
    assert detection.bbox_xyxy == (2.0, 3.0, 10.0, 8.0)


@pytest.mark.parametrize("class_id", [-1, 9, True, 1.5, "5"])
def test_invalid_detection_class_ids(class_id):
    """Only integer class IDs from 0 to 8 are permitted."""
    with pytest.raises((TypeError, ValueError)):
        make_detection(class_id=class_id)


@pytest.mark.parametrize(
    "confidence",
    [-0.1, 1.1, True, float("nan"), float("inf")],
)
def test_invalid_detection_confidence(confidence):
    """Confidence must be a finite value between 0 and 1."""
    with pytest.raises((TypeError, ValueError)):
        make_detection(confidence=confidence)


@pytest.mark.parametrize(
    "bbox",
    [
        (-1.0, 0.0, 5.0, 5.0),
        (5.0, 1.0, 2.0, 8.0),
        (1.0, 5.0, 8.0, 2.0),
        (1.0, 1.0, 1.0, 5.0),
        (1.0, 1.0, 5.0, 1.0),
        (0.0, 0.0, float("nan"), 5.0),
        (0.0, 0.0, float("inf"), 5.0),
        (0.0, 0.0, 5.0),
    ],
)
def test_invalid_detection_boxes(bbox):
    """Boxes must contain four finite coordinates and positive area."""
    with pytest.raises((TypeError, ValueError)):
        make_detection(bbox_xyxy=bbox)


# ------------------------------------------------------------
# MassEstimate
# ------------------------------------------------------------

def test_valid_unavailable_mass():
    """Unavailable mass must not contain a numeric value."""
    mass = make_unavailable_mass()

    assert mass.status == "unavailable"
    assert mass.value_g is None


def test_valid_uncalibrated_mass():
    """Uncalibrated mass must declare its method and reason."""
    mass = MassEstimate(
        status="uncalibrated",
        method_id="area_calibrated",
        reason="Camera calibration is missing",
    )

    assert mass.value_g is None
    assert mass.calibration_id is None


def test_valid_estimated_mass():
    """A calibrated estimate may contain a positive mass."""
    mass = MassEstimate(
        status="estimated",
        value_g=12.5,
        method_id="area_calibrated",
        calibration_id="calibration_001",
        assumptions=("Known reference area",),
    )

    assert mass.value_g == 12.5
    assert mass.status == "estimated"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"status": "unknown"},
        {"status": "unavailable"},
        {"status": "unavailable", "reason": "Not available", "value_g": 0.0},
        {"status": "uncalibrated", "reason": "No calibration"},
        {
            "status": "uncalibrated",
            "method_id": "area_calibrated",
            "reason": "No calibration",
            "calibration_id": "calibration_001",
        },
        {"status": "estimated", "value_g": 1.0},
        {
            "status": "estimated",
            "value_g": 0.0,
            "method_id": "area_calibrated",
            "calibration_id": "calibration_001",
        },
        {
            "status": "estimated",
            "value_g": float("nan"),
            "method_id": "area_calibrated",
            "calibration_id": "calibration_001",
        },
        {
            "status": "estimated",
            "value_g": 2.0,
            "method_id": "area_calibrated",
            "calibration_id": "calibration_001",
            "reason": "Not available",
        },
    ],
)
def test_invalid_mass_estimates(kwargs):
    """Contradictory mass states must be rejected."""
    with pytest.raises((TypeError, ValueError)):
        MassEstimate(**kwargs)


# ------------------------------------------------------------
# FrameResult
# ------------------------------------------------------------

def test_valid_empty_frame_result():
    """A frame with no detected objects must be valid."""
    result = make_result()

    assert result.object_results == ()
    assert result.restored_image is None


def test_valid_detection_mass_pair():
    """Every detection must remain associated with its mass state."""
    detection = make_detection()
    mass = make_unavailable_mass()

    result = make_result(
        object_results=((detection, mass),),
    )

    assert result.object_results[0][0] is detection
    assert result.object_results[0][1] is mass


def test_box_at_image_boundary_is_valid():
    """A box may reach the right and bottom image borders."""
    detection = make_detection(
        bbox_xyxy=(0.0, 0.0, 20.0, 12.0),
    )

    result = make_result(
        object_results=((detection, make_unavailable_mass()),),
    )

    assert len(result.object_results) == 1


def test_box_outside_frame_is_rejected():
    """A detection must not exceed original frame dimensions."""
    detection = make_detection(
        bbox_xyxy=(0.0, 0.0, 21.0, 12.0),
    )

    with pytest.raises(ValueError):
        make_result(
            object_results=((detection, make_unavailable_mass()),),
        )


def test_rov_requires_unavailable_mass():
    """ROV is not waste and cannot receive estimated mass."""
    rov = make_detection(class_id=8)

    estimated_mass = MassEstimate(
        status="estimated",
        value_g=5.0,
        method_id="area_calibrated",
        calibration_id="calibration_001",
    )

    with pytest.raises(ValueError):
        make_result(
            object_results=((rov, estimated_mass),),
        )


def test_rov_with_unavailable_mass_is_valid():
    """ROV detections are valid when mass is unavailable."""
    rov = make_detection(class_id=8)
    mass = make_unavailable_mass()

    result = make_result(
        object_results=((rov, mass),),
    )

    assert len(result.object_results) == 1
    assert result.object_results[0][0].class_name == "rov"
    assert result.object_results[0][1].status == "unavailable"
    assert result.object_results[0][1].value_g is None


def test_valid_restored_image():
    """Restored output must preserve the original dimensions."""
    restored = np.full((12, 20, 3), 128, dtype=np.uint8)

    result = make_result(
        restoration_method_id="cnn",
        restored_image=restored,
    )

    assert result.restored_image.shape == (12, 20, 3)
    assert not result.restored_image.flags.writeable

    restored[0, 0, 0] = 0
    assert result.restored_image[0, 0, 0] == 128


def test_invalid_restored_image_dimensions():
    """Restoration must not silently change external geometry."""
    restored = np.zeros((8, 8, 3), dtype=np.uint8)

    with pytest.raises(ValueError):
        make_result(
            restoration_method_id="cnn",
            restored_image=restored,
        )


def test_identity_restoration_rejects_extra_image():
    """Identity restoration must reuse the original frame."""
    restored = np.zeros((12, 20, 3), dtype=np.uint8)

    with pytest.raises(ValueError):
        make_result(
            restoration_method_id="identity",
            restored_image=restored,
        )


def test_non_identity_restoration_requires_image():
    """Non-identity methods must provide restored output."""
    with pytest.raises(ValueError):
        make_result(restoration_method_id="cnn")


def test_invalid_object_result_pair():
    """Each object result must contain a detection and mass."""
    with pytest.raises(TypeError):
        make_result(object_results=((make_detection(),),))


def test_timings_are_defensively_copied():
    """Changing the original timing dictionary must not affect results."""
    timings = {"detection": 3.5, "total": 5.0}
    result = make_result(timings_ms=timings)

    timings["detection"] = 100.0

    assert result.timings_ms["detection"] == 3.5

    with pytest.raises(TypeError):
        result.timings_ms["detection"] = 8.0


@pytest.mark.parametrize(
    "timings",
    [
        {"detection": -1.0},
        {"mass": float("nan")},
        {"restoration": float("inf")},
        {"unknown_stage": 1.0},
        {"total": True},
    ],
)
def test_invalid_timings(timings):
    """Timing values and stage names must be valid."""
    with pytest.raises((TypeError, ValueError)):
        make_result(timings_ms=timings)
