"""Synthetic tests for TrashCan, RiSID, and WPBB readers (dataset v2, P1-P3).

These tests do not require public datasets, checkpoints, or GPU access.
The image files contain fixture bytes because this parser hashes images
without decoding them; pixel validation belongs to a later build stage.
"""

import hashlib
import json
import xml.etree.ElementTree as ET
from dataclasses import FrozenInstanceError

import pytest

from vision.src.detection.dataset_v2.manifest import (
    AnnotationRecord,
    ImageRecord,
)
from vision.src.detection.dataset_v2.parsers import (
    _finite_numbers,
    _parse_polygons,
    _trashcan_group,
    _risid_group,
    _wpbb_text,
    _wpbb_group,
    _wpbb_box,
    parse_trashcan_split,
    parse_risid,
    parse_wpbb,
)


FILENAME = "vid_000025_frame0000009.jpg"
IMAGE_BYTES = b"synthetic-image-content-for-hashing"


@pytest.fixture
def trashcan_source(tmp_path):
    """Create a minimal COCO dataset and return its editable fixture data."""
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    (images_dir / FILENAME).write_bytes(IMAGE_BYTES)

    annotation_path = tmp_path / "instances_train_trashcan.json"
    data = {
        "categories": [
            {"id": 5, "name": "trash_plastic"},
            {"id": 6, "name": "bio_fish"},
            {"id": 7, "name": "rov"},
        ],
        "images": [
            {"id": 11, "file_name": FILENAME, "width": 480, "height": 270},
        ],
        "annotations": [
            {
                "id": 21,
                "image_id": 11,
                "category_id": 5,
                "bbox": [12, 20, 30, 40],
                "segmentation": [[12, 20, 42, 20, 42, 60]],
            },
        ],
    }
    _write(annotation_path, data)
    return data, annotation_path, images_dir


def _write(annotation_path, data):
    annotation_path.write_text(
        json.dumps(data, ensure_ascii=False), encoding="utf-8"
    )


def _parse(source, split="train"):
    _, annotation_path, images_dir = source
    return parse_trashcan_split(annotation_path, images_dir, split)


def test_reads_image_provenance_and_annotation(trashcan_source):
    images, annotations = _parse(trashcan_source)
    assert len(images) == len(annotations) == 1
    image, annotation = images[0], annotations[0]
    assert isinstance(image, ImageRecord)
    assert isinstance(annotation, AnnotationRecord)
    assert image.source == annotation.source == "TrashCan"
    assert image.image_id == annotation.image_id == "TrashCan:train:11"
    assert annotation.annotation_id == "TrashCan:train:21"
    assert image.relative_path == f"train/{FILENAME}"
    assert (image.width, image.height) == (480, 270)
    assert image.group_id == "TrashCan:vid_000025"
    assert image.frame_index == 9
    assert image.original_split == "train"
    assert image.final_split is None
    assert image.quarantine_reason is None
    assert image.image_sha256 == hashlib.sha256(IMAGE_BYTES).hexdigest()
    assert annotation.source_class_name == "trash_plastic"
    assert annotation.class_id == 5
    assert annotation.original_box == (12.0, 20.0, 30.0, 40.0)
    assert annotation.original_box_format == "coco_xywh"
    assert annotation.original_polygons == (
        (12.0, 20.0, 42.0, 20.0, 42.0, 60.0),
    )
    assert json.loads(annotation.raw_annotation) == trashcan_source[0]["annotations"][0]


def test_reader_keeps_all_annotations_unresolved(trashcan_source):
    _, annotations = _parse(trashcan_source)
    annotation = annotations[0]
    assert annotation.outcome == "review_required"
    assert "pending policy review" in annotation.reason
    assert annotation.visible_box is None
    assert not annotation.detection_eligible
    assert not annotation.segmentation_eligible
    assert annotation.boundary_contact == (False, False, False, False)


def test_val_split_changes_original_provenance(trashcan_source):
    images, annotations = _parse(trashcan_source, split="val")
    assert images[0].image_id == "TrashCan:val:11"
    assert images[0].original_split == "val"
    assert images[0].relative_path == f"val/{FILENAME}"
    assert annotations[0].image_id == "TrashCan:val:11"
    assert annotations[0].annotation_id == "TrashCan:val:21"


def test_keeps_unmapped_annotations_instead_of_dropping(trashcan_source):
    data, annotation_path, _ = trashcan_source
    data["annotations"][0]["category_id"] = 6
    _write(annotation_path, data)
    _, annotations = _parse(trashcan_source)
    assert len(annotations) == 1
    assert annotations[0].source_class_name == "bio_fish"
    assert annotations[0].class_id is None
    assert "unmapped source category" in annotations[0].reason


def test_maps_rov_class_without_mass_assumptions(trashcan_source):
    data, annotation_path, _ = trashcan_source
    data["annotations"][0]["category_id"] = 7
    _write(annotation_path, data)
    _, annotations = _parse(trashcan_source)
    assert annotations[0].source_class_name == "rov"
    assert annotations[0].class_id == 8


def test_preserves_outside_and_inverted_source_boxes(trashcan_source):
    data, annotation_path, _ = trashcan_source
    data["annotations"][0]["bbox"] = [-20, 250, -2, 40]
    _write(annotation_path, data)
    _, annotations = _parse(trashcan_source)
    assert annotations[0].original_box == (-20.0, 250.0, -2.0, 40.0)
    assert annotations[0].visible_box is None
    assert not annotations[0].detection_eligible


@pytest.mark.parametrize(
    "bbox",
    [None, "not-a-box", [1, 2, 3], [1, 2, 3, True],
     [1, 2, 3, float("inf")], [1, 2, 3, 10**400]],
)
def test_invalid_numeric_boxes_are_retained_only_as_raw(trashcan_source, bbox):
    data, annotation_path, _ = trashcan_source
    data["annotations"][0]["bbox"] = bbox
    _write(annotation_path, data)
    _, annotations = _parse(trashcan_source)
    record = annotations[0]
    assert record.original_box is None
    assert record.original_box_format is None
    assert "missing or invalid source box" in record.reason
    assert record.outcome == "review_required"
    assert "bbox" in json.loads(record.raw_annotation)


@pytest.mark.parametrize(
    "segmentation",
    [[], "invalid", [[0, 0, 1, 1]], [[0, 0, 1, 0, 1]],
     [[0, 0, 1, 0, True, 1]], [[0, 0, 1, 0, 10**400, 1]]],
)
def test_invalid_polygons_are_preserved_as_raw(trashcan_source, segmentation):
    data, annotation_path, _ = trashcan_source
    data["annotations"][0]["segmentation"] = segmentation
    _write(annotation_path, data)
    _, annotations = _parse(trashcan_source)
    record = annotations[0]
    assert record.original_polygons is None
    assert "invalid polygon structure" in record.reason
    assert "segmentation" in json.loads(record.raw_annotation)


def test_missing_segmentation_is_not_called_malformed(trashcan_source):
    data, annotation_path, _ = trashcan_source
    data["annotations"][0].pop("segmentation")
    _write(annotation_path, data)
    _, annotations = _parse(trashcan_source)
    assert annotations[0].original_polygons is None
    assert "invalid polygon structure" not in annotations[0].reason


def test_preserves_valid_multiple_polygons(trashcan_source):
    data, annotation_path, _ = trashcan_source
    data["annotations"][0]["segmentation"] = [
        [0, 0, 10, 0, 10, 10], [20, 20, 30, 20, 30, 30]
    ]
    _write(annotation_path, data)
    _, annotations = _parse(trashcan_source)
    assert annotations[0].original_polygons == (
        (0.0, 0.0, 10.0, 0.0, 10.0, 10.0),
        (20.0, 20.0, 30.0, 20.0, 30.0, 30.0),
    )


def test_keeps_images_with_no_annotations(trashcan_source):
    data, annotation_path, images_dir = trashcan_source
    other_name = "vid_000026_frame0000010.jpg"
    (images_dir / other_name).write_bytes(b"another-image")
    data["images"].append(
        {"id": 12, "file_name": other_name, "width": 100, "height": 90}
    )
    _write(annotation_path, data)
    images, annotations = _parse(trashcan_source)
    assert len(images) == 2
    assert len(annotations) == 1
    assert images[1].image_id == "TrashCan:train:12"
    assert images[1].group_id == "TrashCan:vid_000026"
    assert images[1].frame_index == 10


def test_empty_annotation_collection_is_allowed(trashcan_source):
    data, annotation_path, _ = trashcan_source
    data["annotations"] = []
    _write(annotation_path, data)
    images, annotations = _parse(trashcan_source)
    assert len(images) == 1
    assert annotations == ()


def test_source_files_remain_unchanged(trashcan_source):
    _, annotation_path, images_dir = trashcan_source
    source_bytes = annotation_path.read_bytes()
    image_bytes = (images_dir / FILENAME).read_bytes()
    _parse(trashcan_source)
    assert annotation_path.read_bytes() == source_bytes
    assert (images_dir / FILENAME).read_bytes() == image_bytes


def test_records_are_frozen(trashcan_source):
    image, annotation = [records[0] for records in _parse(trashcan_source)]
    with pytest.raises(FrozenInstanceError):
        image.width = 1
    with pytest.raises(FrozenInstanceError):
        annotation.class_id = 0


@pytest.mark.parametrize("split", ["test", "invalid", "", None])
def test_rejects_unrecognized_trashcan_split(trashcan_source, split):
    with pytest.raises(ValueError):
        _parse(trashcan_source, split=split)


@pytest.mark.parametrize(
    "filename",
    ["../vid_000025_frame0000009.jpg", "nested/vid_000025_frame0000009.jpg",
     r"nested\vid_000025_frame0000009.jpg", "other.jpg", "", ".", ".."],
)
def test_rejects_unsafe_or_unknown_filenames(trashcan_source, filename):
    data, annotation_path, _ = trashcan_source
    data["images"][0]["file_name"] = filename
    _write(annotation_path, data)
    with pytest.raises(ValueError):
        _parse(trashcan_source)


def test_rejects_missing_image(trashcan_source):
    _, _, images_dir = trashcan_source
    (images_dir / FILENAME).unlink()
    with pytest.raises(FileNotFoundError):
        _parse(trashcan_source)


def test_rejects_image_symlink_escaping_source_directory(trashcan_source, tmp_path):
    _, _, images_dir = trashcan_source
    external_file = tmp_path / "outside.jpg"
    external_file.write_bytes(IMAGE_BYTES)
    (images_dir / FILENAME).unlink()
    (images_dir / FILENAME).symlink_to(external_file)
    with pytest.raises(ValueError, match="escapes"):
        _parse(trashcan_source)


def test_rejects_duplicate_image_ids(trashcan_source):
    data, annotation_path, _ = trashcan_source
    data["images"].append(dict(data["images"][0]))
    _write(annotation_path, data)
    with pytest.raises(ValueError, match="Duplicate source image ID"):
        _parse(trashcan_source)


def test_rejects_duplicate_annotation_ids(trashcan_source):
    data, annotation_path, _ = trashcan_source
    data["annotations"].append(dict(data["annotations"][0]))
    _write(annotation_path, data)
    with pytest.raises(ValueError, match="Duplicate source annotation ID"):
        _parse(trashcan_source)


def test_rejects_annotation_for_unknown_image(trashcan_source):
    data, annotation_path, _ = trashcan_source
    data["annotations"][0]["image_id"] = 999
    _write(annotation_path, data)
    with pytest.raises(ValueError, match="unknown image"):
        _parse(trashcan_source)


def test_rejects_annotation_with_unknown_category_id(trashcan_source):
    data, annotation_path, _ = trashcan_source
    data["annotations"][0]["category_id"] = 999
    _write(annotation_path, data)
    with pytest.raises(KeyError):
        _parse(trashcan_source)


@pytest.mark.parametrize("width,height", [(0, 270), (480, -1), (480.0, 270)])
def test_rejects_invalid_declared_image_dimensions(trashcan_source, width, height):
    data, annotation_path, _ = trashcan_source
    data["images"][0].update(width=width, height=height)
    _write(annotation_path, data)
    with pytest.raises((TypeError, ValueError)):
        _parse(trashcan_source)


def test_finite_number_parser_rejects_overflow():
    assert _finite_numbers([1, 2, 10**400, 4], length=4) is None


def test_finite_number_parser_rejects_boolean():
    assert _finite_numbers([1, 2, True, 4], length=4) is None


def test_polygon_parser_rejects_empty():
    assert _parse_polygons([]) is None


def test_trashcan_group_parser_extracts_sequence_and_index():
    assert _trashcan_group("vid_000025_frame0000009.jpg") == ("vid_000025", 9)

# ---------------------------------------------------------------------------
# RiSID COCO reader (dataset v2, phase P2)
# ---------------------------------------------------------------------------

RISID_FILENAME = "river_scene_000123_1_2.png"
RISID_BYTES = b"synthetic-risid-image"


@pytest.fixture
def risid_source(tmp_path):
    """Create a RiSID-like COCO fixture without public dataset files."""
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    (images_dir / RISID_FILENAME).write_bytes(RISID_BYTES)

    annotation_path = tmp_path / "annotations_2cat.json"
    data = {
        "categories": [
            {"id": 1, "name": "pla"},
            {"id": 2, "name": "non_pla"},
        ],
        "images": [
            {
                "id": 101,
                "file_name": [RISID_FILENAME],
                "width": 1024,
                "height": 768,
            },
        ],
        "annotations": [
            {
                "id": 201,
                "image_id": 101,
                "category_id": 1,
                "bbox": [10, 20, 30, 40],
                "segmentation": [[10, 20, 40, 20, 40, 60]],
            },
        ],
    }
    _write(annotation_path, data)
    return data, annotation_path, images_dir


def _parse_risid(source):
    _, annotation_path, images_dir = source
    return parse_risid(annotation_path, images_dir)


def test_risid_reads_original_image_and_annotation(risid_source):
    images, annotations = _parse_risid(risid_source)
    assert len(images) == len(annotations) == 1
    image, annotation = images[0], annotations[0]
    assert isinstance(image, ImageRecord)
    assert isinstance(annotation, AnnotationRecord)
    assert image.source == annotation.source == "RiSID"
    assert image.image_id == annotation.image_id == "RiSID:101"
    assert annotation.annotation_id == "RiSID:201"
    assert image.relative_path == f"images/{RISID_FILENAME}"
    assert (image.width, image.height) == (1024, 768)
    assert image.group_id == "RiSID:river_scene"
    assert image.frame_index == 123
    assert image.original_split == "unspecified"
    assert image.final_split is None
    assert image.image_sha256 == hashlib.sha256(RISID_BYTES).hexdigest()
    assert annotation.source_class_name == "pla"
    assert annotation.class_id == 5
    assert annotation.original_box == (10.0, 20.0, 30.0, 40.0)
    assert annotation.original_box_format == "coco_xywh"
    assert annotation.original_polygons == (
        (10.0, 20.0, 40.0, 20.0, 40.0, 60.0),
    )
    assert json.loads(annotation.raw_annotation) == risid_source[0]["annotations"][0]


def test_risid_keeps_annotations_unresolved(risid_source):
    _, annotations = _parse_risid(risid_source)
    record = annotations[0]
    assert record.outcome == "review_required"
    assert "pending policy review" in record.reason
    assert record.visible_box is None
    assert record.boundary_contact == (False, False, False, False)
    assert not record.detection_eligible
    assert not record.segmentation_eligible


def test_risid_accepts_conventional_string_filename(risid_source):
    data, annotation_path, _ = risid_source
    data["images"][0]["file_name"] = RISID_FILENAME
    _write(annotation_path, data)
    images, _ = _parse_risid(risid_source)
    assert images[0].group_id == "RiSID:river_scene"


def test_risid_preserves_non_pla_annotations(risid_source):
    data, annotation_path, _ = risid_source
    data["annotations"][0]["category_id"] = 2
    _write(annotation_path, data)
    _, annotations = _parse_risid(risid_source)
    record = annotations[0]
    assert record.source_class_name == "non_pla"
    assert record.class_id is None
    assert "unmapped source category" in record.reason
    assert not record.detection_eligible


def test_risid_preserves_disagreeing_source_box_and_polygons(risid_source):
    data, annotation_path, _ = risid_source
    data["annotations"][0]["bbox"] = [10, 20, 20, 20]
    data["annotations"][0]["segmentation"] = [[10, 20, 40, 20, 40, 60]]
    _write(annotation_path, data)
    _, annotations = _parse_risid(risid_source)
    assert annotations[0].original_box == (10.0, 20.0, 20.0, 20.0)
    assert annotations[0].original_polygons == (
        (10.0, 20.0, 40.0, 20.0, 40.0, 60.0),
    )
    assert annotations[0].visible_box is None


@pytest.mark.parametrize(
    "bbox",
    [None, [1, 2, 3], [0, 0, True, 5], [1, 2, 3, float("inf")],
     [1, 2, 3, 10**400]],
)
def test_risid_invalid_boxes_remain_in_raw_annotation(risid_source, bbox):
    data, annotation_path, _ = risid_source
    data["annotations"][0]["bbox"] = bbox
    _write(annotation_path, data)
    _, annotations = _parse_risid(risid_source)
    record = annotations[0]
    assert record.original_box is None
    assert record.original_box_format is None
    assert "missing or invalid source box" in record.reason
    assert "bbox" in json.loads(record.raw_annotation)


@pytest.mark.parametrize(
    "segmentation",
    [None, [], [[0, 0, 1, 1]], [[0, 0, True, 0, 1, 1]],
     [[0, 0, 1, 0, 10**400, 1]]],
)
def test_risid_invalid_polygons_remain_in_raw_annotation(
    risid_source, segmentation
):
    data, annotation_path, _ = risid_source
    data["annotations"][0]["segmentation"] = segmentation
    _write(annotation_path, data)
    _, annotations = _parse_risid(risid_source)
    record = annotations[0]
    assert record.original_polygons is None
    assert "invalid polygon structure" in record.reason
    assert "segmentation" in json.loads(record.raw_annotation)


@pytest.mark.parametrize(
    "filename",
    [[], ["a.png", "b.png"], 123, "", ".", "..",
     "../river_scene_000123_1_2.png",
     "folder/river_scene_000123_1_2.png",
     r"folder\\river_scene_000123_1_2.png"],
)
def test_risid_rejects_unsafe_filename_metadata(
    risid_source, filename
):
    data, annotation_path, _ = risid_source
    data["images"][0]["file_name"] = filename
    _write(annotation_path, data)
    with pytest.raises(ValueError):
        _parse_risid(risid_source)


@pytest.mark.parametrize(
    "filename",
    ["river_scene_123_1_2.png", "river_scene_000123_1.png", "other.png"],
)
def test_risid_rejects_invalid_group_pattern(risid_source, filename):
    data, annotation_path, images_dir = risid_source
    (images_dir / filename).write_bytes(b"additional-image")
    data["images"][0]["file_name"] = [filename]
    _write(annotation_path, data)
    with pytest.raises(ValueError, match="Unsupported RiSID filename"):
        _parse_risid(risid_source)


def test_risid_rejects_missing_image(risid_source):
    _, _, images_dir = risid_source
    (images_dir / RISID_FILENAME).unlink()
    with pytest.raises(FileNotFoundError):
        _parse_risid(risid_source)


def test_risid_rejects_symlink_outside_images_dir(risid_source, tmp_path):
    _, _, images_dir = risid_source
    external_file = tmp_path / "outside.png"
    external_file.write_bytes(RISID_BYTES)
    (images_dir / RISID_FILENAME).unlink()
    (images_dir / RISID_FILENAME).symlink_to(external_file)
    with pytest.raises(ValueError, match="escapes"):
        _parse_risid(risid_source)


def test_risid_rejects_duplicate_image_id(risid_source):
    data, annotation_path, _ = risid_source
    data["images"].append(dict(data["images"][0]))
    _write(annotation_path, data)
    with pytest.raises(ValueError, match="Duplicate RiSID image ID"):
        _parse_risid(risid_source)


def test_risid_rejects_duplicate_filename(risid_source):
    data, annotation_path, _ = risid_source
    duplicate = dict(data["images"][0])
    duplicate["id"] = 102
    data["images"].append(duplicate)
    _write(annotation_path, data)
    with pytest.raises(ValueError, match="Duplicate RiSID filename"):
        _parse_risid(risid_source)


def test_risid_rejects_duplicate_annotation_id(risid_source):
    data, annotation_path, _ = risid_source
    data["annotations"].append(dict(data["annotations"][0]))
    _write(annotation_path, data)
    with pytest.raises(ValueError, match="Duplicate RiSID annotation ID"):
        _parse_risid(risid_source)


def test_risid_rejects_unknown_image_reference(risid_source):
    data, annotation_path, _ = risid_source
    data["annotations"][0]["image_id"] = 999
    _write(annotation_path, data)
    with pytest.raises(ValueError, match="unknown image"):
        _parse_risid(risid_source)


def test_risid_rejects_unknown_category_reference(risid_source):
    data, annotation_path, _ = risid_source
    data["annotations"][0]["category_id"] = 999
    _write(annotation_path, data)
    with pytest.raises(ValueError, match="unknown category"):
        _parse_risid(risid_source)


def test_risid_allows_images_without_annotations(risid_source):
    data, annotation_path, _ = risid_source
    data["annotations"] = []
    _write(annotation_path, data)
    images, annotations = _parse_risid(risid_source)
    assert len(images) == 1
    assert annotations == ()


def test_risid_does_not_modify_source_files(risid_source):
    _, annotation_path, images_dir = risid_source
    before_json = annotation_path.read_bytes()
    before_image = (images_dir / RISID_FILENAME).read_bytes()
    _parse_risid(risid_source)
    assert annotation_path.read_bytes() == before_json
    assert (images_dir / RISID_FILENAME).read_bytes() == before_image


def test_risid_group_extracts_candidate_and_frame():
    assert _risid_group(RISID_FILENAME) == ("river_scene", 123)


# ---------------------------------------------------------------------------
# WPBB PASCAL VOC reader (dataset v2, phase P3)
# ---------------------------------------------------------------------------

WPBB_STEM = "Bag1_Trim3 041"
WPBB_FILENAME = WPBB_STEM + ".jpg"
WPBB_BYTES = b"synthetic-wpbb-image"


def _wpbb_object(root, *, name="plastic bag", coords=(10, 20, 40, 60)):
    """Add a VOC object to an editable XML tree."""
    obj = ET.SubElement(root, "object")
    ET.SubElement(obj, "name").text = name
    if coords is not None:
        box = ET.SubElement(obj, "bndbox")
        for tag, value in zip(("xmin", "ymin", "xmax", "ymax"), coords):
            ET.SubElement(box, tag).text = str(value)
    return obj


def _write_wpbb_xml(path, root):
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


@pytest.fixture
def wpbb_source(tmp_path):
    """Build minimal PASCAL VOC source data without real datasets."""
    annotations_dir = tmp_path / "annotations"
    images_dir = tmp_path / "images"
    annotations_dir.mkdir()
    images_dir.mkdir()
    (images_dir / WPBB_FILENAME).write_bytes(WPBB_BYTES)

    xml_path = annotations_dir / (WPBB_STEM + ".xml")
    root = ET.Element("annotation")
    ET.SubElement(root, "filename").text = WPBB_FILENAME
    size = ET.SubElement(root, "size")
    ET.SubElement(size, "width").text = "1920"
    ET.SubElement(size, "height").text = "1080"
    _wpbb_object(root)
    _write_wpbb_xml(xml_path, root)
    return root, annotations_dir, images_dir, xml_path


def _parse_wpbb(source):
    _, annotations_dir, images_dir, _ = source
    return parse_wpbb(annotations_dir, images_dir)


def test_wpbb_reads_image_and_original_voc_box(wpbb_source):
    images, annotations = _parse_wpbb(wpbb_source)
    assert len(images) == len(annotations) == 1
    image, annotation = images[0], annotations[0]
    assert isinstance(image, ImageRecord)
    assert isinstance(annotation, AnnotationRecord)
    assert image.source == annotation.source == "WPBB"
    assert image.image_id == annotation.image_id == f"WPBB:{WPBB_STEM}"
    assert image.relative_path == f"images/{WPBB_FILENAME}"
    assert (image.width, image.height) == (1920, 1080)
    assert image.group_id == "WPBB:Bag1"
    assert image.frame_index == 41
    assert image.image_sha256 == hashlib.sha256(WPBB_BYTES).hexdigest()
    assert image.original_split == "unspecified"
    assert image.final_split is None
    assert annotation.annotation_id == f"WPBB:{WPBB_STEM}:0"
    assert annotation.source_class_name == "plastic bag"
    assert annotation.class_id == 5
    assert annotation.original_box == (10.0, 20.0, 40.0, 60.0)
    assert annotation.original_box_format == "voc_xyxy"
    assert annotation.original_polygons is None
    assert ET.fromstring(annotation.raw_annotation).findtext("name") == "plastic bag"


def test_wpbb_records_remain_unresolved_and_ineligible(wpbb_source):
    _, annotations = _parse_wpbb(wpbb_source)
    record = annotations[0]
    assert record.outcome == "review_required"
    assert "pending policy review" in record.reason
    assert record.visible_box is None
    assert record.boundary_contact == (False, False, False, False)
    assert not record.detection_eligible
    assert not record.segmentation_eligible


@pytest.mark.parametrize("name", ["plastic bag", "plastic bottle"])
def test_wpbb_maps_both_target_categories_to_plastic(wpbb_source, name):
    root, _, _, xml_path = wpbb_source
    root.find("object/name").text = name
    _write_wpbb_xml(xml_path, root)
    _, annotations = _parse_wpbb(wpbb_source)
    assert annotations[0].source_class_name == name
    assert annotations[0].class_id == 5


def test_wpbb_keeps_unknown_category_without_exporting(wpbb_source):
    root, _, _, xml_path = wpbb_source
    root.find("object/name").text = "seaweed"
    _write_wpbb_xml(xml_path, root)
    _, annotations = _parse_wpbb(wpbb_source)
    assert annotations[0].source_class_name == "seaweed"
    assert annotations[0].class_id is None
    assert "unmapped source category" in annotations[0].reason
    assert not annotations[0].detection_eligible


@pytest.mark.parametrize("name", [None, "   "])
def test_wpbb_keeps_object_with_missing_name(wpbb_source, name):
    root, _, _, xml_path = wpbb_source
    obj = root.find("object")
    if name is None:
        obj.remove(obj.find("name"))
    else:
        obj.find("name").text = name
    _write_wpbb_xml(xml_path, root)
    _, annotations = _parse_wpbb(wpbb_source)
    assert annotations[0].source_class_name == "__missing__"
    assert annotations[0].class_id is None
    assert "missing or unmapped" in annotations[0].reason


def test_wpbb_multiple_objects_have_distinct_source_qualified_ids(wpbb_source):
    root, _, _, xml_path = wpbb_source
    _wpbb_object(root, name="plastic bottle", coords=(100, 200, 300, 400))
    _write_wpbb_xml(xml_path, root)
    images, annotations = _parse_wpbb(wpbb_source)
    assert len(images) == 1
    assert [ann.annotation_id for ann in annotations] == [
        f"WPBB:{WPBB_STEM}:0", f"WPBB:{WPBB_STEM}:1",
    ]
    assert [ann.source_class_name for ann in annotations] == [
        "plastic bag", "plastic bottle",
    ]
    assert annotations[1].original_box == (100, 200, 300, 400)


@pytest.mark.parametrize(
    "coords",
    [(-10, 20, 40, 60), (40, 60, 10, 20), (0, 0, 0, 0),
     (0, 0, 1921, 1081)],
)
def test_wpbb_preserves_unbounded_or_inverted_voc_corners(wpbb_source, coords):
    root, _, _, xml_path = wpbb_source
    obj = root.find("object")
    obj.remove(obj.find("bndbox"))
    box = ET.SubElement(obj, "bndbox")
    for key, value in zip(("xmin", "ymin", "xmax", "ymax"), coords):
        ET.SubElement(box, key).text = str(value)
    _write_wpbb_xml(xml_path, root)
    _, annotations = _parse_wpbb(wpbb_source)
    assert annotations[0].original_box == tuple(float(value) for value in coords)
    assert annotations[0].visible_box is None


@pytest.mark.parametrize("field", ["xmin", "ymin", "xmax", "ymax"])
def test_wpbb_missing_box_coordinate_is_preserved_in_xml(wpbb_source, field):
    root, _, _, xml_path = wpbb_source
    box = root.find("object/bndbox")
    box.remove(box.find(field))
    _write_wpbb_xml(xml_path, root)
    _, annotations = _parse_wpbb(wpbb_source)
    assert annotations[0].original_box is None
    assert annotations[0].original_box_format is None
    assert "missing or invalid source box" in annotations[0].reason
    assert ET.fromstring(annotations[0].raw_annotation).find("bndbox") is not None


@pytest.mark.parametrize("value", ["", "oops", "nan", "inf", "-inf", "1e10000"])
def test_wpbb_rejects_invalid_numeric_box_corner_without_losing_object(wpbb_source, value):
    root, _, _, xml_path = wpbb_source
    root.find("object/bndbox/xmin").text = value
    _write_wpbb_xml(xml_path, root)
    _, annotations = _parse_wpbb(wpbb_source)
    assert annotations[0].original_box is None
    assert annotations[0].original_box_format is None
    assert "missing or invalid source box" in annotations[0].reason


def test_wpbb_missing_box_preserves_the_object(wpbb_source):
    root, _, _, xml_path = wpbb_source
    obj = root.find("object")
    obj.remove(obj.find("bndbox"))
    _write_wpbb_xml(xml_path, root)
    _, annotations = _parse_wpbb(wpbb_source)
    assert len(annotations) == 1
    assert annotations[0].original_box is None
    assert "missing or invalid source box" in annotations[0].reason


def test_wpbb_accepts_image_without_objects(wpbb_source):
    root, _, _, xml_path = wpbb_source
    root.remove(root.find("object"))
    _write_wpbb_xml(xml_path, root)
    images, annotations = _parse_wpbb(wpbb_source)
    assert len(images) == 1
    assert annotations == ()


def test_wpbb_no_annotation_files_is_an_error(tmp_path):
    annotations_dir = tmp_path / "annotations"
    images_dir = tmp_path / "images"
    annotations_dir.mkdir()
    images_dir.mkdir()
    with pytest.raises(ValueError, match="No WPBB XML annotation files"):
        parse_wpbb(annotations_dir, images_dir)


@pytest.mark.parametrize("missing", ["filename", "size/width", "size/height"])
def test_wpbb_rejects_missing_required_xml_text(wpbb_source, missing):
    root, _, _, xml_path = wpbb_source
    if missing == "filename":
        root.remove(root.find("filename"))
    else:
        size = root.find("size")
        size.remove(root.find(missing))
    _write_wpbb_xml(xml_path, root)
    with pytest.raises(ValueError, match="Missing or empty WPBB XML field"):
        _parse_wpbb(wpbb_source)


@pytest.mark.parametrize("field", ["size/width", "size/height"])
def test_wpbb_rejects_nonpositive_image_dimensions(wpbb_source, field):
    root, _, _, xml_path = wpbb_source
    root.find(field).text = "0"
    _write_wpbb_xml(xml_path, root)
    with pytest.raises(ValueError):
        _parse_wpbb(wpbb_source)


def test_wpbb_rejects_invalid_root(wpbb_source):
    root, _, _, xml_path = wpbb_source
    root.tag = "dataset"
    _write_wpbb_xml(xml_path, root)
    with pytest.raises(ValueError, match="Invalid WPBB XML root"):
        _parse_wpbb(wpbb_source)


def test_wpbb_rejects_malformed_xml(wpbb_source):
    _, _, _, xml_path = wpbb_source
    xml_path.write_text("<annotation><object></annotation>", encoding="utf-8")
    with pytest.raises(ET.ParseError):
        _parse_wpbb(wpbb_source)


@pytest.mark.parametrize(
    "filename",
    ["../Bag1_Trim3 041.jpg", "dir/Bag1_Trim3 041.jpg",
     r"dir\Bag1_Trim3 041.jpg", "C:Bag1_Trim3 041.jpg",
     "Bag1_Trim3 041/other.jpg", ".", ".."],
)
def test_wpbb_rejects_unsafe_image_filename(wpbb_source, filename):
    root, _, _, xml_path = wpbb_source
    root.find("filename").text = filename
    _write_wpbb_xml(xml_path, root)
    with pytest.raises(ValueError, match="Unsafe WPBB image filename"):
        _parse_wpbb(wpbb_source)


def test_wpbb_rejects_xml_image_stem_mismatch(wpbb_source):
    root, _, images_dir, xml_path = wpbb_source
    other = "Bag1_Trim4 042.jpg"
    (images_dir / other).write_bytes(b"other")
    root.find("filename").text = other
    _write_wpbb_xml(xml_path, root)
    with pytest.raises(ValueError, match="stems do not match"):
        _parse_wpbb(wpbb_source)


@pytest.mark.parametrize("stem", ["Bag_Trim3 041", "Bag1Trim3 041", "Bottle1_Trim2", "other"])
def test_wpbb_rejects_unknown_filename_pattern(wpbb_source, stem):
    root, _, images_dir, xml_path = wpbb_source
    new_path = xml_path.with_name(stem + ".xml")
    root.find("filename").text = stem + ".jpg"
    (images_dir / (stem + ".jpg")).write_bytes(b"other")
    xml_path.unlink()
    _write_wpbb_xml(new_path, root)
    with pytest.raises(ValueError, match="Unsupported WPBB filename"):
        _parse_wpbb(wpbb_source)


def test_wpbb_rejects_missing_image_file(wpbb_source):
    _, _, images_dir, _ = wpbb_source
    (images_dir / WPBB_FILENAME).unlink()
    with pytest.raises(FileNotFoundError):
        _parse_wpbb(wpbb_source)


def test_wpbb_rejects_image_symlink_outside_directory(wpbb_source, tmp_path):
    _, _, images_dir, _ = wpbb_source
    external = tmp_path / "outside.jpg"
    external.write_bytes(b"outside")
    (images_dir / WPBB_FILENAME).unlink()
    (images_dir / WPBB_FILENAME).symlink_to(external)
    with pytest.raises(ValueError, match="escapes"):
        _parse_wpbb(wpbb_source)


def test_wpbb_rejects_annotation_symlink_outside_directory(wpbb_source, tmp_path):
    _, _, _, xml_path = wpbb_source
    external = tmp_path / "outside.xml"
    external.write_bytes(xml_path.read_bytes())
    xml_path.unlink()
    xml_path.symlink_to(external)
    with pytest.raises(ValueError, match="escapes"):
        _parse_wpbb(wpbb_source)


def test_wpbb_rejects_duplicate_image_filename(wpbb_source):
    root, _, _, xml_path = wpbb_source
    second_xml = xml_path.with_name("Bag1_Trim3 042.xml")
    _write_wpbb_xml(second_xml, root)  # Same referenced image in both XMLs.
    with pytest.raises(ValueError, match="Duplicate WPBB image filename"):
        _parse_wpbb(wpbb_source)


def test_wpbb_conservative_group_spans_multiple_trims(wpbb_source):
    root, _, images_dir, xml_path = wpbb_source
    second = "Bag1_Trim9 002"
    root.find("filename").text = second + ".jpg"
    (images_dir / (second + ".jpg")).write_bytes(b"second")
    _write_wpbb_xml(xml_path.with_name(second + ".xml"), root)
    images, _ = _parse_wpbb(wpbb_source)
    assert len(images) == 2
    assert {image.group_id for image in images} == {"WPBB:Bag1"}
    assert {image.frame_index for image in images} == {41, 2}


def test_wpbb_groups_different_object_ids_separately(wpbb_source):
    root, _, images_dir, xml_path = wpbb_source
    second = "Bottle2_Trim1 015"
    root.find("filename").text = second + ".jpg"
    (images_dir / (second + ".jpg")).write_bytes(b"second")
    _write_wpbb_xml(xml_path.with_name(second + ".xml"), root)
    images, _ = _parse_wpbb(wpbb_source)
    assert {im.group_id for im in images} == {"WPBB:Bag1", "WPBB:Bottle2"}


def test_wpbb_deterministic_order_independent_of_file_creation(wpbb_source):
    root, _, images_dir, xml_path = wpbb_source
    second = "Bag1_Trim1 001"
    root.find("filename").text = second + ".jpg"
    (images_dir / (second + ".jpg")).write_bytes(b"second")
    _write_wpbb_xml(xml_path.with_name(second + ".xml"), root)
    images, _ = _parse_wpbb(wpbb_source)
    assert [im.image_id for im in images] == [
        f"WPBB:{second}", f"WPBB:{WPBB_STEM}",
    ]


def test_wpbb_reading_does_not_modify_source_xml_or_images(wpbb_source):
    _, _, images_dir, xml_path = wpbb_source
    before_xml = xml_path.read_bytes()
    before_image = (images_dir / WPBB_FILENAME).read_bytes()
    _parse_wpbb(wpbb_source)
    assert xml_path.read_bytes() == before_xml
    assert (images_dir / WPBB_FILENAME).read_bytes() == before_image


def test_wpbb_group_extracts_object_and_frame():
    assert _wpbb_group("Bag1_Trim3 041.jpg") == ("Bag1", 41)
    assert _wpbb_group("Bottle2_Trim1 015.jpg") == ("Bottle2", 15)


def test_wpbb_required_text_helper_rejects_missing_fields():
    root = ET.Element("annotation")
    with pytest.raises(ValueError, match="Missing or empty WPBB XML field"):
        _wpbb_text(root, "filename")


def test_wpbb_box_helper_rejects_absent_box():
    assert _wpbb_box(ET.Element("object")) is None
