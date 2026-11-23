"""Source readers for detection dataset v2.

Readers preserve original annotations and produce unresolved provenance
records. Geometry policies, split assignment, and dataset export are
handled by separate components.
"""

import hashlib
import json
import math
import re
import xml.etree.ElementTree as ET
from numbers import Real
from pathlib import Path

from vision.src.core.contracts import CLASS_NAMES
from vision.src.detection.dataset_v2.manifest import (
    AnnotationRecord,
    ImageRecord,
)


CLASS_IDS = {
    name: index
    for index, name in enumerate(CLASS_NAMES)
}


def _sha256_file(path: Path) -> str:
    """Hash a source image without modifying its contents."""
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def _finite_numbers(
    values: object,
    length: int | None = None,
) -> tuple[float, ...] | None:
    """Parse numeric source fields while preserving invalid raw input."""
    if not isinstance(values, (list, tuple)):
        return None

    if length is not None and len(values) != length:
        return None

    result = []

    for value in values:
        if isinstance(value, bool) or not isinstance(value, Real):
            return None

        try:
            number = float(value)
        except (OverflowError, ValueError):
            return None

        if not math.isfinite(number):
            return None

        result.append(number)

    return tuple(result)


def _parse_polygons(
    value: object,
) -> tuple[tuple[float, ...], ...] | None:
    """Read COCO polygon lists without correcting their coordinates."""
    if not isinstance(value, list) or not value:
        return None

    polygons = []

    for polygon in value:
        coordinates = _finite_numbers(polygon)

        if (
            coordinates is None
            or len(coordinates) < 6
            or len(coordinates) % 2
        ):
            return None

        polygons.append(coordinates)

    return tuple(polygons)


def _trashcan_group(filename: str) -> tuple[str, int]:
    """Extract the candidate video identifier and frame index."""
    match = re.fullmatch(
        r"(vid_\d+)_frame(\d+)",
        Path(filename).stem,
    )

    if match is None:
        raise ValueError(
            f"Unsupported TrashCan filename: {filename}"
        )

    return match.group(1), int(match.group(2))


def parse_trashcan_split(
    annotation_path: Path,
    images_dir: Path,
    split: str,
) -> tuple[
    tuple[ImageRecord, ...],
    tuple[AnnotationRecord, ...],
]:
    """Parse one TrashCan COCO split without applying correction policies."""
    if split not in ("train", "val"):
        raise ValueError("TrashCan split must be train or val")

    annotation_path = Path(annotation_path)
    images_dir = Path(images_dir)

    data = json.loads(
        annotation_path.read_text(encoding="utf-8")
    )

    categories = {
        category["id"]: category["name"]
        for category in data["categories"]
    }

    images = []
    image_ids = {}
    seen_image_ids = set()

    for item in data["images"]:
        raw_id = item["id"]
        qualified_id = f"TrashCan:{split}:{raw_id}"

        if qualified_id in seen_image_ids:
            raise ValueError(
                f"Duplicate source image ID: {qualified_id}"
            )

        seen_image_ids.add(qualified_id)

        filename = item["file_name"]

        if (
            not isinstance(filename, str)
            or not filename
            or "\\" in filename
            or Path(filename).name != filename
            or filename in (".", "..")
        ):
            raise ValueError("Unsafe TrashCan filename")

        video_id, frame_index = _trashcan_group(filename)

        image_path = images_dir / filename

        if not image_path.is_file():
            raise FileNotFoundError(image_path)

        if not image_path.resolve().is_relative_to(
            images_dir.resolve()
        ):
            raise ValueError("Image path escapes its source directory")

        record = ImageRecord(
            source="TrashCan",
            image_id=qualified_id,
            relative_path=f"{split}/{filename}",
            width=item["width"],
            height=item["height"],
            group_id=f"TrashCan:{video_id}",
            original_split=split,
            image_sha256=_sha256_file(image_path),
            frame_index=frame_index,
        )

        images.append(record)
        image_ids[raw_id] = qualified_id

    annotations = []
    seen_annotation_ids = set()

    for item in data["annotations"]:
        raw_id = item["id"]
        qualified_id = f"TrashCan:{split}:{raw_id}"

        if qualified_id in seen_annotation_ids:
            raise ValueError(
                f"Duplicate source annotation ID: {qualified_id}"
            )

        seen_annotation_ids.add(qualified_id)

        source_image_id = item["image_id"]

        if source_image_id not in image_ids:
            raise ValueError(
                "Annotation references an unknown image"
            )

        source_class_name = categories[item["category_id"]]
        class_id = CLASS_IDS.get(source_class_name)

        original_box = _finite_numbers(
            item.get("bbox"),
            length=4,
        )
        original_polygons = _parse_polygons(
            item.get("segmentation")
        )

        reasons = ["source annotation pending policy review"]

        if class_id is None:
            reasons.append("unmapped source category")

        if original_box is None:
            reasons.append("missing or invalid source box")

        if (
            "segmentation" in item
            and original_polygons is None
        ):
            reasons.append(
                "missing, unsupported, or invalid polygon structure"
            )

        record = AnnotationRecord(
            source="TrashCan",
            annotation_id=qualified_id,
            image_id=image_ids[source_image_id],
            source_class_name=source_class_name,
            class_id=class_id,
            raw_annotation=json.dumps(
                item,
                sort_keys=True,
                ensure_ascii=False,
            ),
            original_box=original_box,
            original_box_format=(
                "coco_xywh" if original_box is not None else None
            ),
            original_polygons=original_polygons,
            visible_box=None,
            outcome="review_required",
            reason="; ".join(reasons),
            detection_eligible=False,
            segmentation_eligible=False,
        )

        annotations.append(record)

    return tuple(images), tuple(annotations)


def _risid_filename(value: object) -> str:
    """Extract a safe filename from RiSID's COCO image metadata."""
    if isinstance(value, list):
        if len(value) != 1:
            raise ValueError("RiSID file_name must contain one name")
        value = value[0]

    if (
        not isinstance(value, str)
        or not value
        or "\\" in value
        or Path(value).name != value
        or value in (".", "..")
    ):
        raise ValueError("Unsafe RiSID filename")

    return value


def _risid_group(filename: str) -> tuple[str, int]:
    """Extract a candidate sequence group and frame index."""
    match = re.fullmatch(
        r"(?P<group>.+)_(?P<frame>\d{6})_\d+_\d+",
        Path(filename).stem,
    )

    if match is None:
        raise ValueError(
            f"Unsupported RiSID filename: {filename}"
        )

    return match.group("group"), int(match.group("frame"))


def parse_risid(
    annotation_path: Path,
    images_dir: Path,
) -> tuple[
    tuple[ImageRecord, ...],
    tuple[AnnotationRecord, ...],
]:
    """Read RiSID COCO records without splitting or correcting labels."""
    annotation_path = Path(annotation_path)
    images_dir = Path(images_dir)

    data = json.loads(
        annotation_path.read_text(encoding="utf-8")
    )

    categories = {
        category["id"]: category["name"]
        for category in data["categories"]
    }

    images = []
    image_ids = {}
    seen_image_ids = set()
    seen_filenames = set()

    for item in data["images"]:
        raw_id = item["id"]
        qualified_id = f"RiSID:{raw_id}"

        if qualified_id in seen_image_ids:
            raise ValueError(
                f"Duplicate RiSID image ID: {qualified_id}"
            )

        seen_image_ids.add(qualified_id)

        filename = _risid_filename(item["file_name"])

        if filename in seen_filenames:
            raise ValueError(
                f"Duplicate RiSID filename: {filename}"
            )

        seen_filenames.add(filename)

        group, frame_index = _risid_group(filename)
        image_path = images_dir / filename

        if not image_path.is_file():
            raise FileNotFoundError(image_path)

        if not image_path.resolve().is_relative_to(
            images_dir.resolve()
        ):
            raise ValueError(
                "RiSID image path escapes its source directory"
            )

        record = ImageRecord(
            source="RiSID",
            image_id=qualified_id,
            relative_path=f"images/{filename}",
            width=item["width"],
            height=item["height"],
            group_id=f"RiSID:{group}",
            original_split="unspecified",
            image_sha256=_sha256_file(image_path),
            frame_index=frame_index,
        )

        images.append(record)
        image_ids[raw_id] = qualified_id

    annotations = []
    seen_annotation_ids = set()

    for item in data["annotations"]:
        raw_id = item["id"]
        qualified_id = f"RiSID:{raw_id}"

        if qualified_id in seen_annotation_ids:
            raise ValueError(
                f"Duplicate RiSID annotation ID: {qualified_id}"
            )

        seen_annotation_ids.add(qualified_id)

        source_image_id = item["image_id"]

        if source_image_id not in image_ids:
            raise ValueError(
                "RiSID annotation references an unknown image"
            )

        category_id = item["category_id"]

        if category_id not in categories:
            raise ValueError(
                "RiSID annotation references an unknown category"
            )

        source_class_name = categories[category_id]

        class_id = (
            CLASS_IDS["trash_plastic"]
            if source_class_name == "pla"
            else None
        )

        original_box = _finite_numbers(
            item.get("bbox"),
            length=4,
        )

        original_polygons = _parse_polygons(
            item.get("segmentation")
        )

        reasons = ["source annotation pending policy review"]

        if source_class_name != "pla":
            reasons.append("unmapped source category")

        if original_box is None:
            reasons.append("missing or invalid source box")

        if original_polygons is None:
            reasons.append(
                "missing, unsupported, or invalid polygon structure"
            )

        record = AnnotationRecord(
            source="RiSID",
            annotation_id=qualified_id,
            image_id=image_ids[source_image_id],
            source_class_name=source_class_name,
            class_id=class_id,
            raw_annotation=json.dumps(
                item,
                sort_keys=True,
                ensure_ascii=False,
            ),
            original_box=original_box,
            original_box_format=(
                "coco_xywh" if original_box is not None else None
            ),
            original_polygons=original_polygons,
            visible_box=None,
            outcome="review_required",
            reason="; ".join(reasons),
            detection_eligible=False,
            segmentation_eligible=False,
        )

        annotations.append(record)

    return tuple(images), tuple(annotations)


def _wpbb_text(element: ET.Element, path: str) -> str:
    """Read required XML text with a descriptive validation error."""
    value = element.findtext(path)

    if value is None or not value.strip():
        raise ValueError(f"Missing or empty WPBB XML field: {path}")

    return value.strip()


def _wpbb_group(filename: str) -> tuple[str, int]:
    """Extract a conservative object group and a frame index."""
    match = re.fullmatch(
        r"(?P<object>(?:Bag|Bottle)\d+)_Trim\d+\s+(?P<frame>\d+)",
        Path(filename).stem,
    )

    if match is None:
        raise ValueError(
            f"Unsupported WPBB filename: {filename}"
        )

    return match.group("object"), int(match.group("frame"))


def _wpbb_box(obj: ET.Element) -> tuple[float, ...] | None:
    """Preserve finite VOC corners, including inverted or unbounded ones."""
    box = obj.find("bndbox")

    if box is None:
        return None

    coordinates = []

    for name in ("xmin", "ymin", "xmax", "ymax"):
        value = box.findtext(name)

        if value is None:
            return None

        try:
            number = float(value.strip())
        except (ValueError, OverflowError):
            return None

        if not math.isfinite(number):
            return None

        coordinates.append(number)

    return tuple(coordinates)


def parse_wpbb(
    annotations_dir: Path,
    images_dir: Path,
) -> tuple[
    tuple[ImageRecord, ...],
    tuple[AnnotationRecord, ...],
]:
    """Read WPBB VOC records without correcting or exporting labels."""
    annotations_dir = Path(annotations_dir)
    images_dir = Path(images_dir)

    xml_paths = sorted(annotations_dir.glob("*.xml"))

    if not xml_paths:
        raise ValueError("No WPBB XML annotation files found")

    images = []
    annotations = []
    seen_filenames = set()

    for xml_path in xml_paths:
        if not xml_path.is_file():
            raise FileNotFoundError(xml_path)

        if not xml_path.resolve().is_relative_to(
            annotations_dir.resolve()
        ):
            raise ValueError(
                "WPBB annotation path escapes its source directory"
            )

        root = ET.parse(xml_path).getroot()

        if root.tag != "annotation":
            raise ValueError("Invalid WPBB XML root element")

        filename = _wpbb_text(root, "filename")

        if (
            "\\" in filename
            or ":" in filename
            or Path(filename).name != filename
            or filename in (".", "..")
        ):
            raise ValueError("Unsafe WPBB image filename")

        if filename in seen_filenames:
            raise ValueError(
                f"Duplicate WPBB image filename: {filename}"
            )

        seen_filenames.add(filename)

        if Path(filename).stem != xml_path.stem:
            raise ValueError(
                "WPBB XML and image filename stems do not match"
            )

        object_id, frame_index = _wpbb_group(filename)
        image_path = images_dir / filename

        if not image_path.is_file():
            raise FileNotFoundError(image_path)

        if not image_path.resolve().is_relative_to(
            images_dir.resolve()
        ):
            raise ValueError(
                "WPBB image path escapes its source directory"
            )

        image_id = f"WPBB:{xml_path.stem}"

        image_record = ImageRecord(
            source="WPBB",
            image_id=image_id,
            relative_path=f"images/{filename}",
            width=int(_wpbb_text(root, "size/width")),
            height=int(_wpbb_text(root, "size/height")),
            group_id=f"WPBB:{object_id}",
            original_split="unspecified",
            image_sha256=_sha256_file(image_path),
            frame_index=frame_index,
        )

        images.append(image_record)

        for index, obj in enumerate(root.findall("object")):
            source_class_name = obj.findtext("name")

            if (
                source_class_name is None
                or not source_class_name.strip()
            ):
                source_class_name = "__missing__"
            else:
                source_class_name = source_class_name.strip()

            class_id = (
                CLASS_IDS["trash_plastic"]
                if source_class_name in (
                    "plastic bag",
                    "plastic bottle",
                )
                else None
            )

            original_box = _wpbb_box(obj)

            reasons = [
                "source annotation pending policy review"
            ]

            if class_id is None:
                reasons.append(
                    "missing or unmapped source category"
                )

            if original_box is None:
                reasons.append(
                    "missing or invalid source box"
                )

            annotation_record = AnnotationRecord(
                source="WPBB",
                annotation_id=f"{image_id}:{index}",
                image_id=image_id,
                source_class_name=source_class_name,
                class_id=class_id,
                raw_annotation=ET.tostring(
                    obj,
                    encoding="unicode",
                ),
                original_box=original_box,
                original_box_format=(
                    "voc_xyxy"
                    if original_box is not None
                    else None
                ),
                original_polygons=None,
                visible_box=None,
                outcome="review_required",
                reason="; ".join(reasons),
                detection_eligible=False,
                segmentation_eligible=False,
            )

            annotations.append(annotation_record)

    return tuple(images), tuple(annotations)
