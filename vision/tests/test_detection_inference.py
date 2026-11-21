"""Unit tests for the YOLOv8s inference adapter.

Uses deterministic Ultralytics doubles without checkpoints, GPU,
datasets, or network access.
"""

import numpy as np
import pytest

from vision.src.core.contracts import (
    CLASS_NAMES,
    Detection,
    FramePacket,
    MassEstimate,
)
from vision.src.detection import inference
from vision.src.detection.inference import YOLOv8Detector
from vision.src.pipeline import Pipeline


NAMES = dict(enumerate(CLASS_NAMES))


def make_image() -> np.ndarray:
    """Create a valid BGR image with non-square dimensions."""
    return np.full((12, 20, 3), 42, dtype=np.uint8)


class FakeBoxes:
    """Expose the box attributes used by the adapter."""

    def __init__(self, rows=(), confidences=(), classes=()):
        self.xyxy = np.asarray(rows, dtype=float).reshape(-1, 4)
        self.conf = np.asarray(confidences, dtype=float)
        self.cls = np.asarray(classes, dtype=float)

    def cpu(self):
        return self

    def numpy(self):
        return self


class FakeResult:
    """Represent one Ultralytics image result."""

    def __init__(self, boxes=None, *, names=None, shape=(12, 20)):
        self.boxes = boxes if boxes is not None else FakeBoxes()
        self.names = NAMES.copy() if names is None else names
        self.orig_shape = shape


class FakeModel:
    """Return predetermined predictions and record inference arguments."""

    def __init__(self, boxes=None, *, names=None, task="detect"):
        self.names = NAMES.copy() if names is None else names
        self.task = task
        self.results = [
            FakeResult(boxes=boxes, names=self.names)
        ]
        self.calls = []
        self.mutate_input = False

    def predict(self, **kwargs):
        self.calls.append(kwargs)

        if self.mutate_input:
            kwargs["source"][:] = 0

        return self.results


def make_detector(rows=(), confidences=(), classes=()):
    """Return an adapter and its injected model double."""
    boxes = FakeBoxes(rows, confidences, classes)
    model = FakeModel(boxes)
    return YOLOv8Detector(model=model), model


def test_single_detection_preserves_class_and_geometry():
    detector, model = make_detector(
        rows=[(2, 3, 10, 8)],
        confidences=[0.85],
        classes=[5],
    )

    detections = detector.detect(make_image())

    assert len(detections) == 1
    assert isinstance(detections[0], Detection)
    assert detections[0].class_id == 5
    assert detections[0].class_name == "trash_plastic"
    assert detections[0].confidence == pytest.approx(0.85)
    assert detections[0].bbox_xyxy == (2.0, 3.0, 10.0, 8.0)

    kwargs = model.calls[0]
    assert kwargs["conf"] == 0.25
    assert kwargs["imgsz"] == 640
    assert kwargs["stream"] is False
    assert kwargs["save"] is False
    assert kwargs["verbose"] is False


def test_multiple_detections_preserve_order():
    detector, _ = make_detector(
        rows=[(1, 2, 8, 9), (10, 3, 18, 11)],
        confidences=[0.9, 0.7],
        classes=[5, 8],
    )

    detections = detector.detect(make_image())

    assert isinstance(detections, tuple)
    assert [item.class_id for item in detections] == [5, 8]
    assert detections[1].class_name == "rov"


def test_empty_detections_return_empty_tuple():
    detector, _ = make_detector()

    assert detector.detect(make_image()) == ()


def test_inference_cannot_modify_input_image():
    detector, model = make_detector()
    model.mutate_input = True

    image = make_image()
    detector.detect(image)

    np.testing.assert_array_equal(image, make_image())
    assert np.all(model.calls[0]["source"] == 0)


@pytest.mark.parametrize(
    "image",
    [
        None,
        "not an image",
        np.zeros((12, 20, 3), dtype=np.float32),
        np.zeros((12, 20), dtype=np.uint8),
        np.zeros((0, 20, 3), dtype=np.uint8),
        np.zeros((12, 20, 4), dtype=np.uint8),
    ],
)
def test_invalid_images_are_rejected(image):
    detector, _ = make_detector()

    with pytest.raises((TypeError, ValueError)):
        detector.detect(image)


@pytest.mark.parametrize(
    "names",
    [
        {**NAMES, 5: "wrong_class"},
        dict(enumerate(CLASS_NAMES[:-1])),
        {str(i): name for i, name in enumerate(CLASS_NAMES)},
    ],
)
def test_invalid_model_class_mapping_is_rejected(names):
    with pytest.raises(ValueError, match="class names"):
        YOLOv8Detector(model=FakeModel(names=names))


def test_wrong_model_task_is_rejected():
    with pytest.raises(ValueError, match="detection task"):
        YOLOv8Detector(model=FakeModel(task="segment"))


def test_result_names_are_validated_again():
    detector, model = make_detector()
    model.results[0].names = {**NAMES, 5: "incorrect"}

    with pytest.raises(ValueError, match="class names"):
        detector.detect(make_image())


def test_result_dimensions_must_match_input():
    detector, model = make_detector()
    model.results[0].orig_shape = (20, 12)

    with pytest.raises(ValueError, match="dimensions"):
        detector.detect(make_image())


@pytest.mark.parametrize("result_count", [0, 2])
def test_model_must_return_one_image_result(result_count):
    detector, model = make_detector()
    model.results = model.results * result_count

    with pytest.raises(ValueError, match="exactly one"):
        detector.detect(make_image())


def test_missing_boxes_are_rejected():
    detector, model = make_detector()
    model.results[0].boxes = None

    with pytest.raises(TypeError, match="Boxes"):
        detector.detect(make_image())


def test_malformed_box_shape_is_rejected():
    detector, model = make_detector()
    model.results[0].boxes.xyxy = np.zeros((1, 3))

    with pytest.raises(ValueError, match="xyxy"):
        detector.detect(make_image())


def test_inconsistent_array_lengths_are_rejected():
    detector, model = make_detector(
        rows=[(2, 3, 10, 8)],
        confidences=[0.9],
        classes=[5],
    )
    model.results[0].boxes.conf = np.array([0.9, 0.8])

    with pytest.raises(ValueError, match="inconsistent"):
        detector.detect(make_image())


@pytest.mark.parametrize(
    "box",
    [
        (-1, 2, 10, 8),
        (2, 3, 21, 8),
        (2, 3, 10, 13),
        (5, 3, 5, 8),
        (8, 3, 2, 9),
        (2, 3, float("nan"), 8),
        (2, 3, float("inf"), 8),
    ],
)
def test_invalid_box_geometry_is_rejected(box):
    detector, _ = make_detector(
        rows=[box],
        confidences=[0.9],
        classes=[5],
    )

    with pytest.raises(ValueError):
        detector.detect(make_image())


@pytest.mark.parametrize(
    "class_id",
    [-1, 9, 5.5, float("nan"), float("inf")],
)
def test_invalid_class_ids_are_rejected(class_id):
    detector, _ = make_detector(
        rows=[(2, 3, 10, 8)],
        confidences=[0.9],
        classes=[class_id],
    )

    with pytest.raises(ValueError):
        detector.detect(make_image())


@pytest.mark.parametrize(
    "confidence",
    [-0.1, 1.1, float("nan"), float("inf")],
)
def test_invalid_prediction_confidences_are_rejected(confidence):
    detector, _ = make_detector(
        rows=[(2, 3, 10, 8)],
        confidences=[confidence],
        classes=[5],
    )

    with pytest.raises(ValueError):
        detector.detect(make_image())


@pytest.mark.parametrize(
    "confidence",
    [-0.1, 1.1, float("nan"), float("inf"), True],
)
def test_invalid_configuration_confidence_is_rejected(confidence):
    with pytest.raises((TypeError, ValueError)):
        YOLOv8Detector(model=FakeModel(), conf=confidence)


def test_missing_checkpoint_is_rejected(tmp_path):
    path = tmp_path / "missing.pt"

    with pytest.raises(FileNotFoundError):
        YOLOv8Detector(checkpoint_path=path)


def test_checkpoint_loading_uses_injected_loader(tmp_path, monkeypatch):
    path = tmp_path / "best.pt"
    path.write_bytes(b"fake checkpoint placeholder")

    loaded_paths = []

    def fake_loader(checkpoint):
        loaded_paths.append(checkpoint)
        return FakeModel()

    monkeypatch.setattr(inference, "YOLO", fake_loader)

    detector = YOLOv8Detector(checkpoint_path=path)

    assert detector.detector_id == "yolov8s"
    assert loaded_paths == [str(path)]
    assert detector.detect(make_image()) == ()


def test_prediction_errors_propagate():
    detector, model = make_detector()

    def failing_predict(**kwargs):
        raise RuntimeError("Simulated inference failure")

    model.predict = failing_predict

    with pytest.raises(RuntimeError, match="inference failure"):
        detector.detect(make_image())


class FakeMassEstimator:
    """Provide an uncalibrated mass state without physical claims."""

    mass_estimator_id = "area_calibrated"

    def __init__(self):
        self.calls = []

    def estimate(self, detection, frame):
        self.calls.append((detection, frame))
        return MassEstimate(
            status="uncalibrated",
            method_id=self.mass_estimator_id,
            reason="Physical calibration is not available",
        )


def test_detector_integrates_with_pipeline_and_skips_rov_mass():
    detector, _ = make_detector(
        rows=[(2, 3, 10, 8), (11, 2, 18, 9)],
        confidences=[0.9, 0.8],
        classes=[5, 8],
    )
    estimator = FakeMassEstimator()

    pipeline = Pipeline(
        detector=detector,
        mass_estimator=estimator,
    )
    frame = FramePacket(
        image=make_image(),
        frame_id="frame_001",
        source_id="test_image",
        source_type="image",
    )

    result = pipeline.run(frame)

    assert result.detector_id == "yolov8s"
    assert len(result.object_results) == 2
    assert result.object_results[0][1].status == "uncalibrated"
    assert result.object_results[0][1].value_g is None
    assert result.object_results[1][1].status == "unavailable"
    assert len(estimator.calls) == 1
    assert estimator.calls[0][0].class_name == "trash_plastic"
