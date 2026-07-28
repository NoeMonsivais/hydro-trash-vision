"""
train_yolo.py

Trains and compares YOLO26 and YOLOv8 detection models on the HydroTrashVision
Block B dataset (TrashCan + RiSID + WPBB, unified 9-class scheme).

Class imbalance is addressed via weighted oversampling: images are sampled
with probability inversely proportional to the rarest class they contain,
applied only to the TRAIN dataloader (validation stays unweighted so
per-class AP reflects the real class distribution).
"""

import json
import os
from pathlib import Path

import torch
from torch.utils.data import WeightedRandomSampler
from ultralytics import YOLO
from ultralytics.data.build import InfiniteDataLoader, seed_worker
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils import RANK


VISION_ROOT = Path(__file__).resolve().parents[2]  # detection/ -> src/ -> vision/
DATA_YAML = VISION_ROOT / "configs" / "yolo_train_config.yaml"
CHECKPOINT_DIR = VISION_ROOT / "checkpoints" / "detection"
NUM_CLASSES = 9

SMOKE_TEST = True  # safe local default; flipped to False only in the
                   # Kaggle environment for the actual training run

# "s" (small): Kataoka et al. (2024) found medium-sized YOLOv8 generalized
# better than larger sizes on RiSID specifically; "s" balances that finding
# against GPU budget (two architectures trained back-to-back on one T4).
MODEL_SIZES = ["n"] if SMOKE_TEST else ["s"]

# 70 epochs chosen to fit both architectures (yolov8s ~3.9min/epoch,
# yolo26s ~4.7min/epoch measured via a 3-epoch timing test on the full
# dataset) within Kaggle's 12h session limit: ~232s*70 + 284s*70 =~
# 9.6h combined, leaving margin. If results look promising, a longer
# follow-up run (more epochs) is a separate, later decision.
EPOCHS = 2 if SMOKE_TEST else 70
IMG_SIZE = 640
BATCH = 4 if SMOKE_TEST else 16

# Per-architecture overrides -- intentionally NOT identical.
ARCH_CONFIGS = {
    "yolov8": {
        "checkpoint_fmt": "yolov8{size}.pt",
        "train_overrides": {},
    },
    "yolo26": {
        "checkpoint_fmt": "yolo26{size}.pt",
        "train_overrides": {
            # Floating/submerged debris appears at arbitrary rotations,
            # unlike COCO's canonical object orientations. Per the official
            # YOLO26 Training Recipe guide ("Domain-specific imagery"
            # section): increase degrees if objects appear at arbitrary
            # rotations. The doc says "increase", not by how much -- 15.0
            # is a documented, defensible starting point, not an official
            # value; adjust based on real results.
            "degrees": 15.0,
        },
    },
}


def load_class_counts_from_dataset(dataset, num_classes: int) -> dict[int, int]:
    """Counts total instances per class, reading directly from dataset.labels
    (already filtered for corrupt entries by ultralytics itself)."""
    counts = {c: 0 for c in range(num_classes)}
    for label_entry in dataset.labels:
        for class_id in label_entry["cls"].reshape(-1).astype(int):
            counts[int(class_id)] += 1
    return counts


def compute_image_weights_from_dataset(dataset, class_counts: dict[int, int]) -> list[float]:
    """
    One sampling weight per entry in dataset.labels, SAME ORDER as the dataset
    itself -- guaranteed aligned since both come from the same list, unlike
    a separate filesystem scan which can drift out of sync with whatever
    ultralytics silently filtered during its own scan.
    """
    weights = []
    for label_entry in dataset.labels:
        classes_in_image = set(label_entry["cls"].reshape(-1).astype(int).tolist())

        if not classes_in_image:
            weights.append(0.1)
            continue

        rarest_count = min(class_counts[c] for c in classes_in_image)
        weights.append(1.0 / rarest_count)

    return weights


class WeightedDetectionTrainer(DetectionTrainer):
    """
    Same as DetectionTrainer, except the TRAIN dataloader uses a
    WeightedRandomSampler instead of the default shuffle. Validation falls
    back to the normal, unweighted path so per-class AP stays honest.

    The dataloader construction below mirrors ultralytics.data.build.build_dataloader
    (InfiniteDataLoader, same pin_memory/worker_init_fn/generator seed), since
    build_dataloader() itself doesn't accept an external sampler.
    """

    def get_dataloader(self, dataset_path: str, batch_size: int = 16, rank: int = 0, mode: str = "train"):
        if mode != "train":
            return super().get_dataloader(dataset_path, batch_size, rank, mode)

        dataset = self.build_dataset(dataset_path, mode, batch_size)

        class_counts = load_class_counts_from_dataset(dataset, NUM_CLASSES)
        weights = compute_image_weights_from_dataset(dataset, class_counts)
        sampler = WeightedRandomSampler(
            weights=weights, num_samples=len(weights), replacement=True
        )

        nd = torch.cuda.device_count()
        nw = min((os.cpu_count() or 1) // max(nd, 1), self.args.workers)

        generator = torch.Generator()
        generator.manual_seed(6148914691236517205 + RANK)

        return InfiniteDataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=False,  # mutually exclusive with sampler
            num_workers=nw,
            sampler=sampler,
            prefetch_factor=4 if nw > 0 else None,
            pin_memory=nd > 0,
            collate_fn=getattr(dataset, "collate_fn", None),
            worker_init_fn=seed_worker,
            generator=generator,
        )


def train_one_config(arch_name: str, size: str) -> dict:
    config = ARCH_CONFIGS[arch_name]
    checkpoint_name = config["checkpoint_fmt"].format(size=size)
    model = YOLO(checkpoint_name)

    run_name = f"{arch_name}_{size}"
    train_kwargs = dict(
        data=str(DATA_YAML),
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH,
        project=str(CHECKPOINT_DIR),
        name=run_name,
        trainer=WeightedDetectionTrainer,
        fraction=0.01 if SMOKE_TEST else 1.0,
    )
    train_kwargs.update(config["train_overrides"])

    model.train(**train_kwargs)
    val_results = model.val(data=str(DATA_YAML))

    return {
        "arch": arch_name,
        "size": size,
        "map50": float(val_results.box.map50),
        "map50_95": float(val_results.box.map),
        "ap_per_class": val_results.box.maps.tolist(),
        "class_names": val_results.names,
    }


def main():
    all_results = []
    for arch_name in ARCH_CONFIGS:
        for size in MODEL_SIZES:
            print(f"Training {arch_name} size={size} ...")
            result = train_one_config(arch_name, size)
            all_results.append(result)
            print(f"  mAP50={result['map50']:.4f}  mAP50-95={result['map50_95']:.4f}")

    results_path = CHECKPOINT_DIR / "yolo_comparison_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Saved comparison results to {results_path}")


if __name__ == "__main__":
    main()