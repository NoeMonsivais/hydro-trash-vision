
"""Tests for HydroTrashVision restoration inference adapters.

Use deterministic PyTorch doubles to validate preprocessing,
inference, output geometry, and pipeline compatibility.
"""

import numpy as np
import pytest
import torch
from torch import nn

import vision.src.restoration.inference as restoration_module

from vision.src.core.contracts import (
    Detection,
    FramePacket,
    MassEstimate,
)
from vision.src.pipeline import Pipeline
from vision.src.restoration.deep_model import RestorationCNN
from vision.src.restoration.inference import (
    CNNRestorer,
    ClassicalRestorer,
    load_cnn_checkpoint,
)


# ------------------------------------------------------------
# Test helpers and doubles
# ------------------------------------------------------------

def make_image(
    height: int = 11,
    width: int = 19,
) -> np.ndarray:
    """Create a BGR image with distinct channel values."""
    image = np.empty((height, width, 3), dtype=np.uint8)
    image[:] = (17, 91, 203)
    return image


class RecordingCNN(nn.Module):
    """Record model inputs and optionally return invalid outputs."""

    def __init__(self, output_mode: str = "identity"):
        super().__init__()
        self.output_mode = output_mode
        self.last_input = None
        self.training_during_forward = None
        self.grad_enabled = None
        self.inference_enabled = None

    def forward(self, tensor: torch.Tensor):
        self.last_input = tensor.detach().cpu().clone()
        self.training_during_forward = self.training
        self.grad_enabled = torch.is_grad_enabled()
        self.inference_enabled = torch.is_inference_mode_enabled()

        if self.output_mode == "not_tensor":
            return "invalid output"

        if self.output_mode == "wrong_shape":
            return tensor[:, :, :, :-1]

        if self.output_mode == "integer":
            return tensor.to(torch.uint8)

        if self.output_mode == "nan":
            return torch.full_like(tensor, float("nan"))

        if self.output_mode == "inf":
            return torch.full_like(tensor, float("inf"))

        if self.output_mode == "negative":
            return torch.full_like(tensor, -0.1)

        if self.output_mode == "above_one":
            return torch.full_like(tensor, 1.1)

        return tensor


# ------------------------------------------------------------
# Classical restoration
# ------------------------------------------------------------

def test_classical_restorer_delegates_and_copies(monkeypatch):
    """The classical adapter must protect its input image."""
    image = make_image()
    original = image.copy()
    received = []

    def fake_classical(input_image):
        received.append(input_image.copy())
        input_image[:] = 0
        return np.full_like(input_image, 128)

    monkeypatch.setattr(
        restoration_module,
        "restore_classical_pipeline",
        fake_classical,
    )

    restorer = ClassicalRestorer()
    result = restorer.restore(image)

    assert restorer.method_id == "classical"
    assert len(received) == 1

    np.testing.assert_array_equal(received[0], original)
    np.testing.assert_array_equal(image, original)
    assert np.all(result == 128)
    assert result.shape == image.shape
    assert result.dtype == np.uint8


def test_real_classical_restoration_preserves_contract():
    """The existing classical method must return a valid image."""
    rng = np.random.default_rng(42)

    image = rng.integers(
        1,
        255,
        size=(64, 64, 3),
        dtype=np.uint8,
    )

    original = image.copy()
    restored = ClassicalRestorer().restore(image)

    assert restored.shape == image.shape
    assert restored.dtype == np.uint8
    np.testing.assert_array_equal(image, original)


@pytest.mark.parametrize(
    "invalid_image",
    [
        None,
        "not an image",
        np.zeros((8, 8), dtype=np.uint8),
        np.zeros((8, 8, 4), dtype=np.uint8),
        np.zeros((8, 8, 3), dtype=np.float32),
        np.zeros((0, 8, 3), dtype=np.uint8),
    ],
)
def test_classical_rejects_invalid_inputs(invalid_image):
    """Invalid BGR image inputs must fail explicitly."""
    with pytest.raises((TypeError, ValueError)):
        ClassicalRestorer().restore(invalid_image)


@pytest.mark.parametrize(
    "invalid_output",
    [
        "invalid output",
        np.zeros((8, 8, 3), dtype=np.float32),
        np.zeros((8, 7, 3), dtype=np.uint8),
    ],
)
def test_classical_rejects_invalid_outputs(
    monkeypatch,
    invalid_output,
):
    """Invalid classical results must not reach the pipeline."""
    monkeypatch.setattr(
        restoration_module,
        "restore_classical_pipeline",
        lambda image: invalid_output,
    )

    image = make_image(8, 8)

    with pytest.raises((TypeError, ValueError)):
        ClassicalRestorer().restore(image)


# ------------------------------------------------------------
# CNN preprocessing and output geometry
# ------------------------------------------------------------

def test_cnn_color_order_and_normalization():
    """BGR uint8 must become normalized RGB float32 NCHW."""
    model = RecordingCNN()
    restorer = CNNRestorer(model=model, device="cpu")

    image = make_image(8, 8)
    restored = restorer.restore(image)

    tensor = model.last_input

    assert tensor.shape == (1, 3, 8, 8)
    assert tensor.dtype == torch.float32

    expected_rgb = (
        np.array([203, 91, 17], dtype=np.float32) / 255.0
    )

    np.testing.assert_allclose(
        tensor[0, :, 0, 0].numpy(),
        expected_rgb,
        rtol=0,
        atol=1e-7,
    )

    assert restored.shape == image.shape
    assert restored.dtype == np.uint8

    difference = np.abs(
        restored.astype(np.int16)
        - image.astype(np.int16)
    )

    assert np.max(difference) <= 1


@pytest.mark.parametrize(
    "height,width,padded_height,padded_width",
    [
        (8, 16, 8, 16),
        (11, 19, 16, 24),
        (1, 1, 8, 8),
        (479, 641, 480, 648),
    ],
)
def test_cnn_padding_and_crop(
    height,
    width,
    padded_height,
    padded_width,
):
    """Padding must preserve the original output geometry."""
    model = RecordingCNN()
    restorer = CNNRestorer(model=model, device="cpu")

    image = make_image(height, width)
    image[-1, -1] = (23, 157, 211)

    restored = restorer.restore(image)

    assert model.last_input.shape == (
        1, 3, padded_height, padded_width
    )

    assert restored.shape == image.shape

    expected_edge_rgb = (
        np.array([211, 157, 23], dtype=np.float32) / 255.0
    )

    np.testing.assert_allclose(
        model.last_input[
            0, :, padded_height - 1, padded_width - 1
        ].numpy(),
        expected_edge_rgb,
        rtol=0,
        atol=1e-7,
    )

    difference = np.abs(
        restored.astype(np.int16)
        - image.astype(np.int16)
    )

    assert np.max(difference) <= 1


def test_cnn_preserves_original_image():
    """CNN inference must not modify the supplied image."""
    model = RecordingCNN()
    restorer = CNNRestorer(model=model, device="cpu")

    image = make_image()
    original = image.copy()

    image.flags.writeable = False
    restored = restorer.restore(image)

    np.testing.assert_array_equal(image, original)

    assert not image.flags.writeable
    assert restored.shape == image.shape


def test_cnn_uses_evaluation_and_inference_modes():
    """The model must run without gradients in evaluation mode."""
    model = RecordingCNN()
    model.train()

    restorer = CNNRestorer(model=model, device="cpu")
    restorer.restore(make_image())

    assert model.training is False
    assert model.training_during_forward is False
    assert model.grad_enabled is False
    assert model.inference_enabled is True


@pytest.mark.parametrize(
    "invalid_image",
    [
        None,
        "invalid",
        np.zeros((8, 8), dtype=np.uint8),
        np.zeros((8, 8, 4), dtype=np.uint8),
        np.zeros((8, 8, 3), dtype=np.float32),
        np.zeros((0, 8, 3), dtype=np.uint8),
    ],
)
def test_cnn_rejects_invalid_inputs(invalid_image):
    """Invalid inputs must fail before reaching the CNN."""
    model = RecordingCNN()
    restorer = CNNRestorer(model=model, device="cpu")

    with pytest.raises((TypeError, ValueError)):
        restorer.restore(invalid_image)

    assert model.last_input is None


@pytest.mark.parametrize(
    "output_mode,error_type",
    [
        ("not_tensor", TypeError),
        ("wrong_shape", ValueError),
        ("integer", TypeError),
        ("nan", ValueError),
        ("inf", ValueError),
        ("negative", ValueError),
        ("above_one", ValueError),
    ],
)
def test_cnn_rejects_invalid_model_outputs(
    output_mode,
    error_type,
):
    """Invalid model outputs must not be silently corrected."""
    model = RecordingCNN(output_mode=output_mode)
    restorer = CNNRestorer(model=model, device="cpu")

    with pytest.raises(error_type):
        restorer.restore(make_image())


def test_cnn_rejects_non_module_model():
    """Injected models must be torch.nn.Module instances."""
    with pytest.raises(TypeError, match="torch.nn.Module"):
        CNNRestorer(model=object(), device="cpu")


def test_cnn_rejects_model_and_checkpoint(tmp_path):
    """A model and checkpoint cannot be provided together."""
    with pytest.raises(ValueError, match="either"):
        CNNRestorer(
            model=RecordingCNN(),
            checkpoint_path=tmp_path / "weights.pt",
            device="cpu",
        )


def test_cnn_model_exception_propagates():
    """Inference errors must not produce a successful image."""

    class FailingCNN(nn.Module):
        def forward(self, tensor):
            raise RuntimeError("Simulated CNN failure")

    restorer = CNNRestorer(
        model=FailingCNN(),
        device="cpu",
    )

    with pytest.raises(
        RuntimeError,
        match="Simulated CNN failure",
    ):
        restorer.restore(make_image())


# ------------------------------------------------------------
# Checkpoint loading
# ------------------------------------------------------------

def test_missing_cnn_checkpoint_is_rejected(tmp_path):
    """A missing checkpoint must produce an explicit error."""
    missing = tmp_path / "missing.pt"

    with pytest.raises(FileNotFoundError):
        load_cnn_checkpoint(
            missing,
            torch.device("cpu"),
        )


def test_cnn_checkpoint_loads_state_dict(tmp_path):
    """A saved state_dict must recreate the CNN architecture."""
    original_model = RestorationCNN()
    checkpoint = tmp_path / "test_weights.pt"

    torch.save(original_model.state_dict(), checkpoint)

    loaded_model = load_cnn_checkpoint(
        checkpoint,
        torch.device("cpu"),
    )

    assert isinstance(loaded_model, RestorationCNN)
    assert loaded_model.training is False

    original_state = original_model.state_dict()
    loaded_state = loaded_model.state_dict()

    assert original_state.keys() == loaded_state.keys()

    first_key = next(iter(original_state))

    assert torch.equal(
        original_state[first_key],
        loaded_state[first_key],
    )


def test_invalid_cnn_checkpoint_is_rejected(tmp_path):
    """An incompatible state_dict must fail explicitly."""
    checkpoint = tmp_path / "invalid_weights.pt"

    torch.save(
        {"wrong_key": torch.tensor(1.0)},
        checkpoint,
    )

    with pytest.raises(RuntimeError):
        load_cnn_checkpoint(
            checkpoint,
            torch.device("cpu"),
        )


# ------------------------------------------------------------
# Integration with the existing Pipeline
# ------------------------------------------------------------

def test_cnn_restorer_integrates_with_pipeline():
    """The CNN adapter must satisfy the Pipeline Restorer interface."""
    model = RecordingCNN()
    restorer = CNNRestorer(model=model, device="cpu")

    class RecordingDetector:
        detector_id = "fake_detector"

        def __init__(self):
            self.received_image = None

        def detect(self, image):
            self.received_image = image.copy()

            return [
                Detection(
                    class_id=5,
                    confidence=0.9,
                    bbox_xyxy=(1.0, 1.0, 5.0, 5.0),
                )
            ]

    class UncalibratedEstimator:
        mass_estimator_id = "area_calibrated"

        def estimate(self, detection, frame):
            return MassEstimate(
                status="uncalibrated",
                method_id=self.mass_estimator_id,
                reason="Physical calibration is unavailable",
            )

    detector = RecordingDetector()

    pipeline = Pipeline(
        detector=detector,
        mass_estimator=UncalibratedEstimator(),
        restorer=restorer,
    )

    frame = FramePacket(
        image=make_image(),
        frame_id="frame_0001",
        source_id="test_source",
        source_type="image",
    )

    result = pipeline.run(frame)

    assert result.restoration_method_id == "cnn"
    assert result.restored_image.shape == frame.image.shape

    np.testing.assert_array_equal(
        detector.received_image,
        result.restored_image,
    )

    assert len(result.object_results) == 1
    assert result.object_results[0][0].class_name == "trash_plastic"

    mass = result.object_results[0][1]

    assert mass.status == "uncalibrated"
    assert mass.value_g is None
    assert np.all(frame.image[:, :, 0] == 17)