"""
Converts TrashCan (COCO), RiSID (COCO), and WPBB (PASCAL VOC XML) annotations
into the flat YOLO format expected by ultralytics: one .txt per image
(<class_id> <x_center> <y_center> <width> <height>, normalized 0-1), plus
images/labels folders and a data.yaml.

Design choices (see session notes for reasoning):
- Single CLASS_MAP + FINAL_CLASSES drive all three parsers -- no per-dataset
  exclusion logic duplicated.
- Images are symlinked into the unified structure, never copied (disk budget).
- TrashCan's own train/val split is respected as-is. RiSID and WPBB get a
  new 80/20 split, generated once and saved to JSON -- same pattern as
  Block A's dataset.py (never recomputed by seed alone).
"""
import json
import random
import xml.etree.ElementTree as ET
from pathlib import Path

VISION_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = VISION_ROOT / "data" / "public_datasets"
OUTPUT_ROOT = DATA_ROOT / "detection_yolo"
SPLITS_PATH = VISION_ROOT / "configs" / "detection_splits.json"

# Maps every source category name we keep to its final material-scheme name.
# Anything NOT in this dict (bio classes, non_pla) is silently dropped.
CLASS_MAP = {
    "trash_etc": "trash_etc",
    "trash_fabric": "trash_fabric",
    "trash_fishing_gear": "trash_fishing_gear",
    "trash_metal": "trash_metal",
    "trash_paper": "trash_paper",
    "trash_plastic": "trash_plastic",
    "trash_rubber": "trash_rubber",
    "trash_wood": "trash_wood",
    "rov": "rov",
    "pla": "trash_plastic",             # RiSID
    "plastic bag": "trash_plastic",     # WPBB
    "plastic bottle": "trash_plastic",  # WPBB
}

# Order here fixes the numeric class_id YOLO will train on (0-indexed).
FINAL_CLASSES = [
    "trash_etc", "trash_fabric", "trash_fishing_gear", "trash_metal",
    "trash_paper", "trash_plastic", "trash_rubber", "trash_wood", "rov",
]


def _yolo_line_from_coco_bbox(class_id: int, bbox: list, img_w: int, img_h: int) -> str:
    """Convert a COCO bbox [x_min, y_min, w, h] to a normalized YOLO line."""
    x_min, y_min, w, h = bbox
    x_center = (x_min + w / 2) / img_w
    y_center = (y_min + h / 2) / img_h
    return f"{class_id} {x_center:.6f} {y_center:.6f} {w / img_w:.6f} {h / img_h:.6f}"


def _yolo_line_from_corners(class_id: int, xmin: float, ymin: float, xmax: float, ymax: float,
                             img_w: int, img_h: int) -> str:
    """Convert PASCAL VOC corner coordinates to a normalized YOLO line."""
    x_center = ((xmin + xmax) / 2) / img_w
    y_center = ((ymin + ymax) / 2) / img_h
    width = (xmax - xmin) / img_w
    height = (ymax - ymin) / img_h
    return f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"


def _yolo_line_from_segmentation(class_id: int, segmentation: list, img_w: int, img_h: int) -> str:
    """Derive a normalized YOLO line from a COCO segmentation polygon's
    min/max extent, instead of trusting the bbox field directly.

    RiSID's bbox field is unreliable for ~85% of its annotations (confirmed
    via cross-check against segmentation extent: a subset of images -- all
    sharing a distinct filename pattern, likely a different source video
    batch -- has bbox consistently scaled to ~0.667x the true segmentation
    extent, while the rest match exactly). Deriving from segmentation
    sidesteps the issue entirely: correct for the unaffected subset too.
    """
    all_points = []
    for polygon in segmentation:
        points = [(polygon[i], polygon[i + 1]) for i in range(0, len(polygon), 2)]
        all_points.extend(points)

    xs = [p[0] for p in all_points]
    ys = [p[1] for p in all_points]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    # NOTE: coordinates are intentionally NOT clipped to [0,1] here.
    # ~133 RiSID images (111 train + 22 val, <1% of the dataset) contain
    # objects truncated at the frame edge, whose segmentation polygon extends
    # beyond the actual canvas -> negative or >1.0 coordinates after
    # normalization. Verified programmatically: the bbox derived from
    # segmentation matches exactly (326/331 annotations, remainder is only
    # floating-point rounding) what ultralytics flags as "corrupt" and
    # automatically discards during training -- confirming this is a real
    # limitation of the source data, not a bug in this converter.
    #
    # Decision: leave these images out of training (do not clip). Clipping
    # the box to the frame edge would train the model to systematically
    # underestimate the area of edge-truncated objects -- relevant to
    # Block C, where bounding box area directly feeds the mass estimation
    # calculation (calibrated-area method). An underestimation bias here
    # would silently propagate into the final mass estimate.
    x_center = ((x_min + x_max) / 2) / img_w
    y_center = ((y_min + y_max) / 2) / img_h
    width = (x_max - x_min) / img_w
    height = (y_max - y_min) / img_h
    return f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"


def _write_split_files(image_label_pairs: list, split: str) -> None:
    """Symlink each image and write its YOLO label .txt into the unified
    images/<split>/ and labels/<split>/ folders ultralytics expects.

    Args:
        image_label_pairs: list of (source_image_path, filename_stem, [yolo_lines]).
        split: "train" or "val".
    """
    images_dir = OUTPUT_ROOT / "images" / split
    labels_dir = OUTPUT_ROOT / "labels" / split
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    for source_image_path, stem, yolo_lines in image_label_pairs:
        link_path = images_dir / source_image_path.name
        if not link_path.exists():
            link_path.symlink_to(source_image_path.resolve())

        label_path = labels_dir / f"{stem}.txt"
        label_path.write_text("\n".join(yolo_lines), encoding="utf-8")


def convert_trashcan() -> None:
    """Parse TrashCan-Material COCO annotations, respecting its own
    pre-defined train/val split (two separate JSON files).
    """
    for split, json_name, images_subdir in [
        ("train", "instances_train_trashcan.json", "train"),
        ("val", "instances_val_trashcan.json", "val"),
    ]:
        data = json.load(open(DATA_ROOT / "TrashCan" / json_name))
        cat_names = {c["id"]: c["name"] for c in data["categories"]}
        images_by_id = {img["id"]: img for img in data["images"]}

        annotations_by_image: dict[int, list[str]] = {}
        for ann in data["annotations"]:
            source_name = cat_names[ann["category_id"]]
            if source_name not in CLASS_MAP:
                continue  # bio classes -- dropped
            final_name = CLASS_MAP[source_name]
            class_id = FINAL_CLASSES.index(final_name)

            img_info = images_by_id[ann["image_id"]]
            line = _yolo_line_from_coco_bbox(
                class_id, ann["bbox"], img_info["width"], img_info["height"]
            )
            annotations_by_image.setdefault(ann["image_id"], []).append(line)

        # Iterate over ALL images, not just ones with kept annotations --
        # images left with zero boxes after filtering (e.g. only bio-class
        # originally) are kept as negative/background examples, with an
        # empty label file. Dropping them entirely would silently remove
        # background signal ultralytics can use to reduce false positives.
        pairs = []
        for image_id, img_info in images_by_id.items():
            lines = annotations_by_image.get(image_id, [])
            file_name = img_info["file_name"]
            source_path = DATA_ROOT / "TrashCan" / images_subdir / file_name
            pairs.append((source_path, Path(file_name).stem, lines))

        _write_split_files(pairs, split)
        n_negative = sum(1 for _, _, lines in pairs if not lines)
        print(f"TrashCan {split}: {len(pairs)} images "
              f"({n_negative} negative/background)")


def _find_text(element: ET.Element, path: str) -> str:
    """Get required text content from a required child element.

    Thin wrapper around ElementTree's .find(), which can return None if
    an element is missing -- structurally possible for malformed XML,
    even though every WPBB annotation is expected to have it. The assert
    makes that assumption explicit and gives a clear failure message
    pointing at the actual problem, instead of a generic AttributeError.
    """
    found = element.find(path)
    assert found is not None and found.text is not None, (
        f"Expected required element <{path}> not found or empty"
    )
    return found.text


def convert_risid() -> None:
    """Parse RiSID COCO annotations. pla -> trash_plastic, non_pla dropped
    entirely (images with only non_pla become negative/background examples
    with an empty label file). No official split exists -- generate one
    (80/20), save to JSON.
    """
    data = json.load(open(DATA_ROOT / "RiSID" / "annotations_2cat.json"))
    cat_names = {c["id"]: c["name"] for c in data["categories"]}
    images_by_id = {img["id"]: img for img in data["images"]}

    annotations_by_image: dict[int, list[str]] = {}
    for ann in data["annotations"]:
        source_name = cat_names[ann["category_id"]]
        if source_name not in CLASS_MAP:
            continue  # non_pla -- dropped
        final_name = CLASS_MAP[source_name]
        class_id = FINAL_CLASSES.index(final_name)

        img_info = images_by_id[ann["image_id"]]
        line = _yolo_line_from_segmentation(
            class_id, ann["segmentation"], img_info["width"], img_info["height"]
        )
        annotations_by_image.setdefault(ann["image_id"], []).append(line)

    # Split over ALL images (not just ones with kept annotations) --
    # same reasoning as convert_trashcan().
    image_ids = list(images_by_id.keys())
    random.seed(42)
    random.shuffle(image_ids)
    n_train = int(len(image_ids) * 0.8)
    split_map = {"train": image_ids[:n_train], "val": image_ids[n_train:]}
    _save_split_json("RiSID", split_map, images_by_id)

    for split, ids in split_map.items():
        pairs = []
        for image_id in ids:
            lines = annotations_by_image.get(image_id, [])
            file_name = images_by_id[image_id]["file_name"][0]
            source_path = DATA_ROOT / "RiSID" / "images" / file_name
            pairs.append((source_path, Path(file_name).stem, lines))
        _write_split_files(pairs, split)
        n_negative = sum(1 for _, _, lines in pairs if not lines)
        print(f"RiSID {split}: {len(pairs)} images "
              f"({n_negative} negative/background)")


def convert_wpbb() -> None:
    """Parse WPBB PASCAL VOC XML annotations. Both classes -> trash_plastic.
    No official split exists -- generate one (80/20), save to JSON.
    """
    annotations_dir = DATA_ROOT / "WPBB" / "annotations"
    xml_paths = sorted(annotations_dir.glob("*.xml"))

    random.seed(42)
    shuffled = xml_paths.copy()
    random.shuffle(shuffled)
    n_train = int(len(shuffled) * 0.8)
    split_map = {"train": shuffled[:n_train], "val": shuffled[n_train:]}
    _save_split_json_wpbb("WPBB", split_map)

    for split, paths in split_map.items():
        pairs = []
        for xml_path in paths:
            tree = ET.parse(xml_path)
            root = tree.getroot()

            img_w = int(_find_text(root, "size/width"))
            img_h = int(_find_text(root, "size/height"))
            file_name = _find_text(root, "filename")

            lines = []
            for obj in root.findall("object"):
                source_name = _find_text(obj, "name")
                final_name = CLASS_MAP[source_name]
                class_id = FINAL_CLASSES.index(final_name)

                box = obj.find("bndbox")
                assert box is not None, f"object missing <bndbox> in {xml_path}"
                line = _yolo_line_from_corners(
                    class_id,
                    float(_find_text(box, "xmin")), float(_find_text(box, "ymin")),
                    float(_find_text(box, "xmax")), float(_find_text(box, "ymax")),
                    img_w, img_h,
                )
                lines.append(line)

            # pathlib handles the spaces in WPBB filenames correctly.
            source_path = DATA_ROOT / "WPBB" / "images" / file_name
            pairs.append((source_path, Path(file_name).stem, lines))

        _write_split_files(pairs, split)
        print(f"WPBB {split}: {len(pairs)} images with kept annotations")


def _save_split_json(dataset_label: str, split_map: dict, images_by_id: dict) -> None:
    """Persist which image filenames went to train/val, keyed by dataset."""
    all_splits = {}
    if SPLITS_PATH.exists():
        all_splits = json.loads(SPLITS_PATH.read_text(encoding="utf-8"))

    all_splits[dataset_label] = {
        split: [images_by_id[i]["file_name"][0] for i in ids]
        for split, ids in split_map.items()
    }
    SPLITS_PATH.parent.mkdir(parents=True, exist_ok=True)
    SPLITS_PATH.write_text(json.dumps(all_splits, indent=2), encoding="utf-8")


def _save_split_json_wpbb(dataset_label: str, split_map: dict) -> None:
    """Same as _save_split_json, but for WPBB's XML-path-based split."""
    all_splits = {}
    if SPLITS_PATH.exists():
        all_splits = json.loads(SPLITS_PATH.read_text(encoding="utf-8"))

    all_splits[dataset_label] = {
        split: [p.stem for p in paths] for split, paths in split_map.items()
    }
    SPLITS_PATH.parent.mkdir(parents=True, exist_ok=True)
    SPLITS_PATH.write_text(json.dumps(all_splits, indent=2), encoding="utf-8")


def write_data_yaml() -> None:
    """Write the dataset config file ultralytics needs to train."""
    yaml_content = (
        f"path: {OUTPUT_ROOT}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"nc: {len(FINAL_CLASSES)}\n"
        f"names: {FINAL_CLASSES}\n"
    )
    yaml_path = VISION_ROOT / "configs" / "yolo_train_config.yaml"
    yaml_path.write_text(yaml_content, encoding="utf-8")
    print(f"data.yaml written to {yaml_path}")


def main() -> None:
    convert_trashcan()
    convert_risid()
    convert_wpbb()
    write_data_yaml()


if __name__ == "__main__":
    main()