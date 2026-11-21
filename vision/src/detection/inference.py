"""YOLOv8s inference adapter for HydroTrashVision.

Converts Ultralytics detection outputs to the project's validated
Detection contract without changing image coordinates or class order.
"""

import math
from collections.abc import Mapping, Sequence
from numbers import Real
from pathlib import Path
from typing import Any

import numpy as np
from ultralytics import YOLO

from vision.src.core.contracts import CLASS_NAMES, Detection


class YOLOv8Detector:
    """Run one-frame YOLOv8s detection using the shared Detector interface."""

    detector_id = "yolov8s"

    def __init__(
        self,
        *,
        model: Any | None = None,
        checkpoint_path: Path | str | None = None,
        conf: float = 0.25,
        imgsz: int = 640,
        device: str | int | None = None,
    ) -> None:
        if model is not None and checkpoint_path is not None:
            raise ValueError("Provide a model or a checkpoint, not both")

        if isinstance(conf, (bool, np.bool_)) or not isinstance(conf, Real):
            raise TypeError("conf must be a real number")

        if not math.isfinite(float(conf)) or not 0.0 <= float(conf) <= 1.0:
            raise ValueError("conf must be finite and in [0, 1]")

        if type(imgsz) is not int:
            raise TypeError("imgsz must be an integer")

        if imgsz <= 0:
            raise ValueError("imgsz must be positive")

        if device is not None:
            if isinstance(device, bool) or not isinstance(device, (str, int)):
                raise TypeError("device must be a string, integer, or None")
            if isinstance(device, str) and not device.strip():
                raise ValueError("device must not be empty")

        if model is None:
            if checkpoint_path is None:
                raise ValueError("A model or checkpoint_path is required")

            path = Path(checkpoint_path)
            if not path.is_file():
                raise FileNotFoundError(f"YOLO checkpoint not found: {path}")
            if path.suffix.lower() != ".pt":
                raise ValueError("The checkpoint must be a .pt file")

            model = YOLO(str(path))

        if not callable(getattr(model, "predict", None)):
            raise TypeError("Model must provide a callable predict method")

        task = getattr(model, "task", None)
        if task is not None and task != "detect":
            raise ValueError("Model must support the detection task")

        self._validate_names(getattr(model, "names", None))

        self._model = model
        self._conf = float(conf)
        self._imgsz = imgsz
        self._device = device

    @staticmethod
    def _validate_names(names: object) -> None:
        """Require an exact index-to-name match with the shared contract."""
        if not isinstance(names, Mapping):
            raise TypeError("Model names must be a class-ID mapping")

        expected_ids = range(len(CLASS_NAMES))
        if (
            len(names) != len(CLASS_NAMES)
            or any(type(key) is not int for key in names)
            or tuple(names.get(class_id) for class_id in expected_ids)
            != CLASS_NAMES
        ):
            raise ValueError("Model class names differ from CLASS_NAMES")

    @staticmethod
    def _validate_image(image: np.ndarray) -> None:
        """Require an image in the pipeline's BGR uint8 HWC format."""
        if not isinstance(image, np.ndarray):
            raise TypeError("Image must be a NumPy array")
        if image.dtype != np.uint8:
            raise TypeError("Image must have uint8 dtype")
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Image must have shape (H, W, 3)")
        if image.shape[0] == 0 or image.shape[1] == 0:
            raise ValueError("Image dimensions must be positive")

    def detect(self, image: np.ndarray) -> tuple[Detection, ...]:
        """Return strict XYXY detections in original-image pixel coordinates."""
        self._validate_image(image)
        height, width = image.shape[:2]

        predict_kwargs: dict[str, Any] = {
            "source": image.copy(),
            "conf": self._conf,
            "imgsz": self._imgsz,
            "verbose": False,
            "save": False,
            "stream": False,
        }
        if self._device is not None:
            predict_kwargs["device"] = self._device

        results = self._model.predict(**predict_kwargs)
        if not isinstance(results, Sequence) or len(results) != 1:
            raise ValueError("Model must return exactly one image result")

        result = results[0]
        if getattr(result, "orig_shape", None) != (height, width):
            raise ValueError("Prediction dimensions differ from input image")
        self._validate_names(getattr(result, "names", None))

        boxes = getattr(result, "boxes", None)
        if boxes is None:
            raise TypeError("Detection result must contain Boxes")

        cpu_boxes = boxes.cpu().numpy()
        xyxy = np.asarray(cpu_boxes.xyxy)
        confidences = np.asarray(cpu_boxes.conf)
        class_ids = np.asarray(cpu_boxes.cls)

        if xyxy.ndim != 2 or xyxy.shape[1] != 4:
            raise ValueError("xyxy must have shape (N, 4)")

        count = xyxy.shape[0]
        if confidences.shape != (count,) or class_ids.shape != (count,):
            raise ValueError("Detection arrays have inconsistent shapes")

        for array in (xyxy, confidences, class_ids):
            if not (
                np.issubdtype(array.dtype, np.integer)
                or np.issubdtype(array.dtype, np.floating)
            ):
                raise TypeError("Detection arrays must contain real numbers")

        detections: list[Detection] = []
        for index in range(count):
            raw_id = float(class_ids[index])
            if not math.isfinite(raw_id) or not raw_id.is_integer():
                raise ValueError(f"Detection {index} has an invalid class ID")

            coordinates = (
                float(xyxy[index, 0]),
                float(xyxy[index, 1]),
                float(xyxy[index, 2]),
                float(xyxy[index, 3]),
            )
            confidence = float(confidences[index])
            if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
                raise ValueError(f"Detection {index} has invalid confidence")
            if not all(math.isfinite(value) for value in coordinates):
                raise ValueError(f"Detection {index} has non-finite coordinates")
            if coordinates[2] > width or coordinates[3] > height:
                raise ValueError(f"Detection {index} exceeds image bounds")

            detections.append(
                Detection(
                    class_id=int(raw_id),
                    confidence=confidence,
                    bbox_xyxy=coordinates,
                )
            )

        return tuple(detections)
