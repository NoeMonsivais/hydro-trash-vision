
"""Restoration inference adapters for the HydroTrashVision pipeline.

Provides classical and CNN restoration through the shared Restorer
interface. Both adapters accept and return BGR uint8 images while
preserving the original spatial dimensions.
"""

from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn

from vision.src.restoration.classical import restore_classical_pipeline
from vision.src.restoration.deep_model import RestorationCNN


VISION_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT_PATH = (
    VISION_ROOT / "checkpoints" / "restoration_cnn" / "best_model.pt"
)

CNN_SIZE_MULTIPLE = 8


def _validate_bgr_image(image: np.ndarray) -> None:
    """Validate the external BGR uint8 image format."""
    if not isinstance(image, np.ndarray):
        raise TypeError("Image must be a NumPy array")

    if image.dtype != np.uint8:
        raise TypeError("Image must have uint8 dtype")

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Image must have shape (H, W, 3)")

    if image.shape[0] == 0 or image.shape[1] == 0:
        raise ValueError("Image dimensions must be positive")


def _validate_restoration_output(
    output: np.ndarray,
    original: np.ndarray,
) -> None:
    """Ensure restoration preserves the external image contract."""
    _validate_bgr_image(output)

    if output.shape != original.shape:
        raise ValueError(
            "Restored image must match original dimensions"
        )


def load_cnn_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
) -> RestorationCNN:
    """Load trained CNN weights without importing evaluation scripts."""
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"CNN checkpoint not found: {checkpoint_path}"
        )

    model = RestorationCNN()

    state_dict = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model


class ClassicalRestorer:
    """Adapter for the existing classical restoration baseline."""

    method_id = "classical"

    def restore(self, image: np.ndarray) -> np.ndarray:
        """Restore a BGR image using the classical pipeline."""
        _validate_bgr_image(image)

        restored = restore_classical_pipeline(image.copy())

        _validate_restoration_output(restored, image)

        return restored


class CNNRestorer:
    """Adapter for the trained CNN restoration model.

    Input images are padded to multiples of eight, converted to
    RGB tensors, restored, and cropped to their original size.
    """

    method_id = "cnn"

    def __init__(
        self,
        *,
        model: nn.Module | None = None,
        checkpoint_path: Path | str | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        if model is not None and checkpoint_path is not None:
            raise ValueError(
                "Provide either a model or a checkpoint, not both"
            )

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self._device = torch.device(device)

        if model is None:
            path = (
                Path(checkpoint_path)
                if checkpoint_path is not None
                else DEFAULT_CHECKPOINT_PATH
            )

            self._model = load_cnn_checkpoint(
                path,
                self._device,
            )
        else:
            if not isinstance(model, nn.Module):
                raise TypeError("model must be a torch.nn.Module")

            self._model = model.to(self._device)
            self._model.eval()

    def restore(self, image: np.ndarray) -> np.ndarray:
        """Restore an image without changing its external geometry."""
        _validate_bgr_image(image)

        original_height, original_width = image.shape[:2]

        pad_height = (
            -original_height
        ) % CNN_SIZE_MULTIPLE

        pad_width = (
            -original_width
        ) % CNN_SIZE_MULTIPLE

        padded = np.pad(
            image,
            (
                (0, pad_height),
                (0, pad_width),
                (0, 0),
            ),
            mode="edge",
        )

        # Convert BGR uint8 HWC to RGB float32 NCHW.
        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        normalized = rgb.astype(np.float32) / 255.0
        chw = np.transpose(normalized, (2, 0, 1))
        chw = np.ascontiguousarray(chw)

        input_tensor = (
            torch.from_numpy(chw)
            .unsqueeze(0)
            .to(self._device)
        )

        with torch.inference_mode():
            output_tensor = self._model(input_tensor)

        expected_shape = (
            1,
            3,
            padded.shape[0],
            padded.shape[1],
        )

        if not isinstance(output_tensor, torch.Tensor):
            raise TypeError("CNN must return a torch.Tensor")

        if tuple(output_tensor.shape) != expected_shape:
            raise ValueError(
                "CNN output must match padded input dimensions"
            )

        if not torch.is_floating_point(output_tensor):
            raise TypeError("CNN output must be floating-point")

        if not torch.isfinite(output_tensor).all().item():
            raise ValueError("CNN output contains non-finite values")

        if (
            (output_tensor < 0).any().item()
            or (output_tensor > 1).any().item()
        ):
            raise ValueError("CNN output must be in [0, 1]")

        # Convert RGB NCHW back to BGR uint8 HWC.
        output_array = (
            output_tensor.squeeze(0)
            .detach()
            .cpu()
            .float()
            .numpy()
        )

        output_rgb = np.transpose(
            output_array, (1, 2, 0)
        )

        output_rgb = (
            output_rgb * 255.0
        ).astype(np.uint8)

        output_bgr = cv2.cvtColor(
            output_rgb,
            cv2.COLOR_RGB2BGR,
        )

        restored = output_bgr[
            :original_height,
            :original_width,
        ].copy()

        _validate_restoration_output(restored, image)

        return restored