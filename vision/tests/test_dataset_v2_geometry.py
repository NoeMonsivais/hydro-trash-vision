"""Unit tests for pure dataset-v2 geometry helpers.

These tests require no datasets, checkpoints, or GPU.
"""

import math

import pytest

from vision.src.detection.dataset_v2.geometry import (
    classify_box,
    clip_box,
    coco_xywh_to_xyxy,
    polygons_to_xyxy,
    validate_image_size,
    validate_xyxy,
    xyxy_to_yolo,
    yolo_to_xyxy,
)


@pytest.mark.parametrize("width,height", [(1, 1), (480, 270)])
def test_validate_image_size_accepts_positive_integers(width, height):
    assert validate_image_size(width, height) is None


@pytest.mark.parametrize("width,height", [(True, 200), (200, 2.5), ("480", 270)])
def test_validate_image_size_rejects_noninteger_dimensions(width, height):
    with pytest.raises(TypeError):
        validate_image_size(width, height)


@pytest.mark.parametrize("width,height", [(0, 200), (480, 0), (-1, 270)])
def test_validate_image_size_rejects_nonpositive_dimensions(width, height):
    with pytest.raises(ValueError):
        validate_image_size(width, height)


def test_validate_xyxy_accepts_source_coordinates_outside_canvas():
    assert validate_xyxy([-20, 10, 50, 70]) == (-20.0, 10.0, 50.0, 70.0)


@pytest.mark.parametrize(
    "box,error",
    [
        ((0, 0, 0, 10), ValueError),
        ((5, 1, 4, 10), ValueError),
        ((0, 0, 10), ValueError),
        ((0, 0, float("nan"), 10), ValueError),
        ((0, 0, math.inf, 10), ValueError),
        ((False, 0, 10, 10), TypeError),
        ("0 0 10 10", TypeError),
    ],
)
def test_validate_xyxy_rejects_invalid_boxes(box, error):
    with pytest.raises(error):
        validate_xyxy(box)


@pytest.mark.parametrize(
    "source,expected",
    [
        ((10, 20, 5, 7), (10, 20, 15, 27)),
        ((-20, 4, 10, 8), (-20, 4, -10, 12)),
    ],
)
def test_coco_conversion_preserves_source_geometry(source, expected):
    assert coco_xywh_to_xyxy(source) == expected


@pytest.mark.parametrize("box", [(0, 0, 0, 4), (0, 0, 3, -1)])
def test_coco_conversion_rejects_nonpositive_dimensions(box):
    with pytest.raises(ValueError):
        coco_xywh_to_xyxy(box)


def test_polygons_to_xyxy_encloses_multiple_parts():
    polygons = [
        [0, 0, 10, 0, 10, 4],
        [20, 7, 25, 9, 21, 4],
    ]
    assert polygons_to_xyxy(polygons) == (0.0, 0.0, 25.0, 9.0)


@pytest.mark.parametrize(
    "polygons,error",
    [
        ([], ValueError),
        ([[0, 0, 1, 1]], ValueError),
        ([[0, 0, 1, 1, 2]], ValueError),
        ([[2, 0, 2, 4, 2, 8]], ValueError),
        ([[0, 0, 2, 0, math.inf, 4]], ValueError),
        (["0,0,1,0,1,1"], TypeError),
    ],
)
def test_polygons_to_xyxy_rejects_invalid_input(polygons, error):
    with pytest.raises(error):
        polygons_to_xyxy(polygons)


@pytest.mark.parametrize(
    "box,expected",
    [
        ((0, 0, 480, 270), "contained"),
        ((20, 30, 100, 90), "contained"),
        ((0, 0, 480.998, 270.998), "partial"),
        ((-20, 30, 100, 90), "partial"),
        ((-50, 30, -10, 90), "outside"),
        ((480, 10, 490, 40), "outside"),
        ((10, 270, 20, 300), "outside"),
    ],
)
def test_classify_box_distinguishes_canvas_relations(box, expected):
    assert classify_box(box, 480, 270) == expected


@pytest.mark.parametrize(
    "box,expected",
    [
        ((20, 30, 100, 90), (20, 30, 100, 90)),
        ((0, 0, 480.998, 270.998), (0, 0, 480, 270)),
        ((-20, 30, 100, 90), (0, 30, 100, 90)),
        ((-50, 30, -10, 90), None),
        ((480, 10, 490, 40), None),
    ],
)
def test_clip_box_intersects_without_inventing_geometry(box, expected):
    assert clip_box(box, 480, 270) == expected


def test_xyxy_to_yolo_normalizes_a_contained_box():
    assert xyxy_to_yolo((10, 20, 110, 120), 200, 200) == pytest.approx(
        (0.3, 0.35, 0.5, 0.5)
    )


@pytest.mark.parametrize("box", [(-1, 0, 10, 10), (0, 0, 481, 270)])
def test_xyxy_to_yolo_rejects_out_of_canvas_boxes(box):
    with pytest.raises(ValueError):
        xyxy_to_yolo(box, 480, 270)


def test_yolo_to_xyxy_decodes_valid_labels():
    assert yolo_to_xyxy((0.3, 0.35, 0.5, 0.5), 200, 200) == pytest.approx(
        (10, 20, 110, 120)
    )


def test_yolo_to_xyxy_does_not_silently_clip_source_labels():
    box = yolo_to_xyxy((1.01, 0.5, 0.1, 0.4), 100, 100)
    assert classify_box(box, 100, 100) == "partial"
    assert box == pytest.approx((96, 30, 106, 70))


@pytest.mark.parametrize("label", [(0.5, 0.5, 0, 0.2), (0.5, 0.5, -0.1, 0.2)])
def test_yolo_to_xyxy_rejects_nonpositive_extents(label):
    with pytest.raises(ValueError):
        yolo_to_xyxy(label, 100, 100)


@pytest.mark.parametrize(
    "box,width,height",
    [
        ((0, 0, 480, 270), 480, 270),
        ((10.25, 22.5, 80.75, 100.125), 480, 270),
        ((0.007, 219.007, 478.997, 270), 480, 270),
    ],
)
def test_yolo_roundtrip_preserves_contained_boxes(box, width, height):
    encoded = xyxy_to_yolo(box, width, height)
    decoded = yolo_to_xyxy(encoded, width, height)
    assert decoded == pytest.approx(box, abs=1e-9)


def test_trashcan_source_box_is_clipped_to_visible_canvas():
    source = (0.007246376811594203, 219.0076923076923,
              478.9978632478632, 270.9980392156863)
    assert classify_box(source, 480, 270) == "partial"
    assert clip_box(source, 480, 270) == pytest.approx(
        (source[0], source[1], source[2], 270)
    )


def test_risid_source_box_without_visible_intersection():
    source = (-272.37927274935396, 973.370970989029,
              -197.81647458038788, 1025.0)
    assert classify_box(source, 1024, 1024) == "outside"
    assert clip_box(source, 1024, 1024) is None
