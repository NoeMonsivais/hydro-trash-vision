"""Unit tests for immutable dataset-v2 provenance records.

These tests use synthetic metadata only: no datasets, GPU, or checkpoints.
"""

import math
from dataclasses import FrozenInstanceError, replace

import pytest

from vision.src.detection.dataset_v2.manifest import (
    AnnotationRecord,
    ImageRecord,
)


def make_image(**overrides):
    """Build a valid image record with overridable fields."""
    fields = {
        "source": "TrashCan",
        "image_id": "TrashCan:img_001",
        "relative_path": "train/vid_000001_frame0000001.jpg",
        "width": 480,
        "height": 270,
        "group_id": "TrashCan:vid_000001",
        "original_split": "train",
        "image_sha256": "a" * 64,
        "frame_index": 1,
        "final_split": None,
        "quarantine_reason": None,
    }
    fields.update(overrides)
    return ImageRecord(**fields)


def make_annotation(**overrides):
    """Build an accepted detection record with sample source geometry."""
    fields = {
        "source": "TrashCan",
        "annotation_id": "TrashCan:ann_001",
        "image_id": "TrashCan:img_001",
        "source_class_name": "trash_plastic",
        "class_id": 5,
        "raw_annotation": '{"id": 1}',
        "original_box": (10.0, 20.0, 30.0, 40.0),
        "original_box_format": "coco_xywh",
        "original_polygons": ((10.0, 20.0, 40.0, 20.0, 40.0, 60.0),),
        "visible_box": (10.0, 20.0, 40.0, 60.0),
        "outcome": "valid",
        "reason": None,
        "boundary_contact": (False, False, False, False),
        "detection_eligible": True,
        "segmentation_eligible": False,
    }
    fields.update(overrides)
    return AnnotationRecord(**fields)


@pytest.mark.parametrize("source", ["TrashCan", "RiSID", "WPBB"])
def test_image_accepts_supported_sources(source):
    record = make_image(
        source=source,
        image_id=f"{source}:image1",
        group_id=f"{source}:group1",
    )
    assert record.source == source


def test_image_is_frozen_and_uses_slots():
    record = make_image()
    with pytest.raises(FrozenInstanceError):
        record.width = 100
    assert not hasattr(record, "__dict__")


@pytest.mark.parametrize(
    "source,error", [("Other", ValueError), (None, TypeError), (123, TypeError)]
)
def test_image_rejects_invalid_source(source, error):
    with pytest.raises(error):
        make_image(source=source)


@pytest.mark.parametrize(
    "field,value",
    [
        ("image_id", "RiSID:img_001"),
        ("image_id", "TrashCan:"),
        ("image_id", " "),
        ("group_id", "WPBB:video1"),
        ("group_id", "TrashCan:  "),
    ],
)
def test_image_rejects_invalid_qualified_ids(field, value):
    with pytest.raises(ValueError):
        make_image(**{field: value})


@pytest.mark.parametrize("field", ["image_id", "group_id"])
def test_image_rejects_nonstring_identifiers(field):
    with pytest.raises(TypeError):
        make_image(**{field: 5})


@pytest.mark.parametrize(
    "path",
    [
        "",
        " ",
        "/root/image.jpg",
        "../image.jpg",
        "train/../image.jpg",
        "train//image.jpg",
        "./image.jpg",
        "train/./image.jpg",
        "train/",
        r"train\image.jpg",
        "C:/image.jpg",
        "train:image.jpg",
    ],
)
def test_image_rejects_unsafe_relative_paths(path):
    with pytest.raises(ValueError):
        make_image(relative_path=path)


def test_image_rejects_nonstring_relative_path():
    with pytest.raises(TypeError):
        make_image(relative_path=None)


def test_image_allows_relative_path_with_spaces():
    record = make_image(relative_path="images/Bag1_Trim1 01.jpg")
    assert record.relative_path == "images/Bag1_Trim1 01.jpg"


@pytest.mark.parametrize(
    "width,height,error",
    [
        (0, 270, ValueError),
        (480, -1, ValueError),
        (True, 270, TypeError),
        (480.0, 270, TypeError),
    ],
)
def test_image_rejects_invalid_dimensions(width, height, error):
    with pytest.raises(error):
        make_image(width=width, height=height)


@pytest.mark.parametrize("split", ["train", "val", "unspecified"])
def test_image_accepts_original_splits(split):
    assert make_image(original_split=split).original_split == split


@pytest.mark.parametrize("split", ["test", "quarantine", None, 1])
def test_image_rejects_invalid_original_splits(split):
    with pytest.raises(ValueError):
        make_image(original_split=split)


@pytest.mark.parametrize("digest", ["abc", "A" * 64, "z" * 64, " " * 64])
def test_image_rejects_malformed_sha256(digest):
    with pytest.raises(ValueError):
        make_image(image_sha256=digest)


def test_image_rejects_nonstring_sha256():
    with pytest.raises(TypeError):
        make_image(image_sha256=123)


@pytest.mark.parametrize("index", [None, 0, 12])
def test_image_accepts_optional_nonnegative_frame_index(index):
    assert make_image(frame_index=index).frame_index == index


@pytest.mark.parametrize("index", [True, 2.5, "12"])
def test_image_rejects_noninteger_frame_index(index):
    with pytest.raises(TypeError):
        make_image(frame_index=index)


def test_image_rejects_negative_frame_index():
    with pytest.raises(ValueError):
        make_image(frame_index=-1)


@pytest.mark.parametrize("split", [None, "train", "val", "test"])
def test_image_accepts_nonquarantine_final_split(split):
    assert make_image(final_split=split).final_split == split


def test_image_accepts_quarantine_with_reason():
    record = make_image(final_split="quarantine", quarantine_reason="missing label")
    assert record.quarantine_reason == "missing label"


@pytest.mark.parametrize("reason", [None, "  ", 123])
def test_image_quarantine_requires_text_reason(reason):
    expected = TypeError if reason == 123 else (TypeError if reason is None else ValueError)
    with pytest.raises(expected):
        make_image(final_split="quarantine", quarantine_reason=reason)


def test_image_rejects_quarantine_reason_without_quarantine():
    with pytest.raises(ValueError):
        make_image(final_split="train", quarantine_reason="unreviewed")


def test_image_rejects_invalid_final_split():
    with pytest.raises(ValueError):
        make_image(final_split="evaluation")


def test_annotation_is_frozen_and_uses_slots():
    record = make_annotation()
    with pytest.raises(FrozenInstanceError):
        record.class_id = 0
    assert not hasattr(record, "__dict__")


@pytest.mark.parametrize("source", ["TrashCan", "RiSID", "WPBB"])
def test_annotation_accepts_supported_sources(source):
    record = make_annotation(
        source=source,
        annotation_id=f"{source}:ann1",
        image_id=f"{source}:img1",
    )
    assert record.source == source


@pytest.mark.parametrize(
    "field,value",
    [
        ("annotation_id", "RiSID:ann1"),
        ("image_id", "WPBB:img1"),
        ("annotation_id", "TrashCan:"),
    ],
)
def test_annotation_rejects_invalid_qualified_ids(field, value):
    with pytest.raises(ValueError):
        make_annotation(**{field: value})


@pytest.mark.parametrize("field", ["source_class_name", "raw_annotation"])
def test_annotation_requires_nonempty_text(field):
    with pytest.raises(ValueError):
        make_annotation(**{field: "  "})


@pytest.mark.parametrize("field", ["source_class_name", "raw_annotation"])
def test_annotation_rejects_nonstring_text(field):
    with pytest.raises(TypeError):
        make_annotation(**{field: None})


@pytest.mark.parametrize("class_id", [0, 5, 8])
def test_annotation_accepts_supported_class_ids(class_id):
    assert make_annotation(class_id=class_id).class_id == class_id


@pytest.mark.parametrize("class_id", [-1, 9])
def test_annotation_rejects_class_id_out_of_range(class_id):
    with pytest.raises(ValueError):
        make_annotation(class_id=class_id)


@pytest.mark.parametrize("class_id", [True, 5.0, "5"])
def test_annotation_rejects_noninteger_class_id(class_id):
    with pytest.raises(TypeError):
        make_annotation(class_id=class_id)


def test_annotation_allows_unmapped_class_when_unresolved():
    record = make_annotation(
        class_id=None,
        visible_box=None,
        outcome="review_required",
        reason="unmapped source category",
        detection_eligible=False,
    )
    assert record.class_id is None


@pytest.mark.parametrize(
    "original_box,original_box_format",
    [
        (None, "coco_xywh"),
        ((1.0, 2.0, 3.0, 4.0), None),
    ],
)
def test_annotation_requires_original_box_format_pair(original_box, original_box_format):
    with pytest.raises(ValueError):
        make_annotation(
            original_box=original_box,
            original_box_format=original_box_format,
        )


def test_annotation_allows_no_original_box_when_unavailable():
    record = make_annotation(original_box=None, original_box_format=None)
    assert record.original_box is None


def test_annotation_accepts_voc_box_format():
    record = make_annotation(original_box_format="voc_xyxy")
    assert record.original_box_format == "voc_xyxy"


def test_annotation_rejects_invalid_box_format():
    with pytest.raises(ValueError):
        make_annotation(original_box_format="yolo_cxcywh")


@pytest.mark.parametrize(
    "box,error",
    [
        ([1, 2, 3, 4], TypeError),
        ((1, 2, 3), ValueError),
        ((1, 2, math.inf, 4), ValueError),
        ((1, float("nan"), 3, 4), ValueError),
        ((False, 1, 3, 4), TypeError),
    ],
)
def test_annotation_rejects_invalid_original_box(box, error):
    with pytest.raises(error):
        make_annotation(original_box=box)


def test_annotation_preserves_unbounded_raw_geometry():
    record = make_annotation(original_box=(-10, 270, -3, 40))
    assert record.original_box == (-10, 270, -3, 40)


@pytest.mark.parametrize(
    "polygons,error",
    [
        ((), ValueError),
        ([], TypeError),
        (([0, 0, 1, 0, 1, 1],), TypeError),
        (((0, 0, 1, 1),), ValueError),
        (((0, 0, 1, 0, 1),), ValueError),
        (((0, 0, 1, 0, math.inf, 1),), ValueError),
        (((0, False, 1, 0, 1, 1),), TypeError),
    ],
)
def test_annotation_rejects_invalid_polygons(polygons, error):
    with pytest.raises(error):
        make_annotation(original_polygons=polygons)


def test_annotation_accepts_multiple_polygons():
    polygons = (
        (0, 0, 10, 0, 10, 10),
        (20, 20, 30, 20, 30, 30),
    )
    record = make_annotation(original_polygons=polygons)
    assert record.original_polygons == polygons


def test_annotation_allows_missing_polygons_for_detection():
    record = make_annotation(original_polygons=None)
    assert record.detection_eligible and record.original_polygons is None


@pytest.mark.parametrize(
    "box,error",
    [
        ([0, 0, 10, 10], TypeError),
        ((0, 0, 0, 10), ValueError),
        ((0, 0, 10, math.inf), ValueError),
    ],
)
def test_annotation_rejects_invalid_visible_box(box, error):
    with pytest.raises(error):
        make_annotation(visible_box=box)


def test_annotation_stores_visible_box_without_canvas_crosscheck():
    record = make_annotation(visible_box=(-2, 10, 481, 270))
    assert record.visible_box == (-2, 10, 481, 270)


@pytest.mark.parametrize("outcome", ["valid", "clipped", "review_required", "rejected"])
def test_annotation_accepts_documented_outcomes(outcome):
    record = make_annotation(
        outcome=outcome,
        reason=None if outcome == "valid" else "documented decision",
        detection_eligible=outcome in ("valid", "clipped"),
    )
    assert record.outcome == outcome


def test_annotation_rejects_unknown_outcome():
    with pytest.raises(ValueError):
        make_annotation(outcome="corrected")


def test_annotation_valid_outcome_cannot_have_reason():
    with pytest.raises(ValueError):
        make_annotation(reason="unneeded")


@pytest.mark.parametrize("outcome", ["clipped", "review_required", "rejected"])
def test_annotation_nonvalid_outcome_requires_reason(outcome):
    with pytest.raises((ValueError, TypeError)):
        make_annotation(outcome=outcome, reason=None, detection_eligible=False)


@pytest.mark.parametrize("class_id,visible_box", [(None, (0, 0, 1, 1)), (5, None)])
def test_annotation_accepted_outcome_requires_class_and_visible_box(class_id, visible_box):
    with pytest.raises(ValueError):
        make_annotation(class_id=class_id, visible_box=visible_box)


def test_annotation_accepted_outcome_may_not_be_export_eligible():
    record = make_annotation(detection_eligible=False)
    assert record.outcome == "valid"
    assert not record.detection_eligible


@pytest.mark.parametrize("outcome", ["review_required", "rejected"])
def test_annotation_unresolved_cannot_be_detection_eligible(outcome):
    with pytest.raises(ValueError):
        make_annotation(outcome=outcome, reason="pending", detection_eligible=True)


@pytest.mark.parametrize(
    "boundary_contact",
    [
        [],
        [False] * 4,
        (False, False),
        (False, False, False, 0),
    ],
)
def test_annotation_rejects_invalid_boundary_contact(boundary_contact):
    with pytest.raises(TypeError):
        make_annotation(boundary_contact=boundary_contact)


@pytest.mark.parametrize("field", ["detection_eligible", "segmentation_eligible"])
def test_annotation_requires_boolean_export_flags(field):
    with pytest.raises(TypeError):
        make_annotation(**{field: 1})


def test_annotation_segmentation_requires_detection_eligibility():
    with pytest.raises(ValueError):
        make_annotation(segmentation_eligible=True, detection_eligible=False)


def test_annotation_segmentation_requires_polygons():
    with pytest.raises(ValueError):
        make_annotation(segmentation_eligible=True, original_polygons=None)


def test_annotation_allows_segmentation_when_all_declared_requirements_hold():
    record = make_annotation(segmentation_eligible=True)
    assert record.segmentation_eligible


def test_dataclasses_replace_supports_immutable_state_transition():
    record = make_annotation()
    updated = replace(
        record,
        outcome="review_required",
        reason="semantic ambiguity",
        detection_eligible=False,
    )
    assert updated.outcome == "review_required"
    assert record.outcome == "valid"
    assert record.detection_eligible is True
