"""
Dataset splitting and PyTorch Dataset for paired underwater restoration data.

Two responsibilities live here on purpose: generating a fixed, auditable
train/val/test split (once, saved to JSON) and serving individual examples
from that split to a PyTorch DataLoader during training.
"""

import json
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

VISION_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = VISION_ROOT / "data" / "public_datasets"
SPLITS_PATH = VISION_ROOT / "configs" / "datasets_splits.json"

DATASET_SOURCES = {
    "UIEB": (DATA_ROOT / "UIEB" / "raw-890", DATA_ROOT / "UIEB" / "reference-890"),
    "EUVP-underwater_imagenet": (
        DATA_ROOT / "EUVP" / "Paired" / "underwater_imagenet" / "trainA",
        DATA_ROOT / "EUVP" / "Paired" / "underwater_imagenet" / "trainB",
    ),
    "EUVP-underwater_scenes": (
        DATA_ROOT / "EUVP" / "Paired" / "underwater_scenes" / "trainA",
        DATA_ROOT / "EUVP" / "Paired" / "underwater_scenes" / "trainB",
    ),
    "EUVP-underwater_dark": (
        DATA_ROOT / "EUVP" / "Paired" / "underwater_dark" / "trainA",
        DATA_ROOT / "EUVP" / "Paired" / "underwater_dark" / "trainB",
    ),
}


def generate_splits(
    output_path: Path = SPLITS_PATH,
    seed: int = 42,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
) -> None:
    """Generate a fixed train/val/test split and saves it to JSON
    
    We save this to disk instead of doing it dynamically on every run to
    prevent data leakage across training sessions and ensure reproducibility.
    """
    # Fixing the seed ensures that if we ever need to regenerate the splits 
    # (e.g., adding a new dataset later), the existing files stay in their 
    # original buckets, preventing train data from leaking into test.
    random.seed(seed)
    all_splits = {}

    for label, (degraded_dir, reference_dir) in DATASET_SOURCES.items():
        # We strictly enforce that the file exists in BOTH folders. 
        # EUVP sometimes has missing pairs, which would silently crash 
        # the PyTorch DataLoader hours into training if not caught here.
        matched_filenames = [
            p.name for p in sorted(degraded_dir.glob("*"))
            if (reference_dir / p.name).exists()
        ]

        random.shuffle(matched_filenames)

        n = len(matched_filenames)
        n_train = int(n * train_frac)
        n_val = int(n * val_frac)

        all_splits[label] = {
            "train": matched_filenames[:n_train],
            "val": matched_filenames[n_train:n_train + n_val],
            "test": matched_filenames[n_train + n_val:],
        }

        print(f"{label}: {n} pairs -> train={len(all_splits[label]['train'])}, "
              f"val={len(all_splits[label]['val'])}, test={len(all_splits[label]['test'])}")
        
    # parents=True and exist_ok=True prevent crashes if the 'configs' folder 
    # doesn't exist yet or was already created by another script.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(all_splits, indent=2), encoding="utf-8")
    print(f"Splits written to {output_path}")


class UnderwaterRestorationDataset(Dataset):
    """Serves (degraded, reference) image pairs for a specific split.
    
    Instead of scannig directories dynamically, this relies on the fixed
    JSON split file. This guarantees that 'train' and 'test' images never
    mix, even if the underlying folder structures change later.
    """

    def __init__(self, split: str, splits_path: Path = SPLITS_PATH, image_size: int = 256):
        self.image_size = image_size

        with open(splits_path, "r", encoding="utf-8") as f:
            all_splits = json.load(f)

        # Flatten the nested dictionary into a single list of tuples:
        # [("UIEB", "image1.png"), ("EUVP-underwater_dark", "image2.png"), ...]
        # This allows __getitem__ to do an extremely fast O(1) index lookup.
        self.entries = []
        for label, split_dict in all_splits.items():
            for filename in split_dict[split]:
                self.entries.append((label, filename))

    def __len__(self) -> int:
        return len(self.entries)
    
    def _load_image(self, path: Path) -> np.ndarray:
        """Reads, resizes, normalizes, and reorders channels for PyTorch."""
        img = cv2.imread(str(path))
        
        # Safety net: OpenCV silently returns None if a file is corrupt/missing
        if img is None:
            raise FileNotFoundError(f"OpenCV could not read image: {path}")

        img = cv2.resize(img, (self.image_size, self.image_size))
        
        # OpenCV reads in BGR by default. We convert to RGB because standard 
        # deep learning models (and our visualizers) expect RGB order.
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Normalize from [0, 255] integer range to [0.0, 1.0] float32 range
        img = img.astype(np.float32) / 255.0
        
        # PyTorch expects tensors in (Channels, Height, Width) format, 
        # but OpenCV provides arrays in (Height, Width, Channels).
        img = np.transpose(img, (2, 0, 1))
        
        return img
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        label, filename = self.entries[idx]
        degraded_dir, reference_dir = DATASET_SOURCES[label]

        degraded = self._load_image(degraded_dir / filename)
        reference = self._load_image(reference_dir / filename)

        # Convert numpy arrays to PyTorch tensors right at the exit boundary
        return torch.from_numpy(degraded), torch.from_numpy(reference)
    

if __name__ == "__main__":
    if not SPLITS_PATH.exists():
        print("No splits file found. Generating a new one...")
        generate_splits()

    train_dataset = UnderwaterRestorationDataset(split="train")
    print(f"Train dataset size: {len(train_dataset)} pairs")

    degraded, reference = train_dataset[0]
    print(f"Degraded image -> shape: {degraded.shape}, dtype: {degraded.dtype}, "
          f"range: [{degraded.min():.3f}, {degraded.max():.3f}]")
    print(f"Reference image -> shape: {reference.shape}, dtype: {reference.dtype}, "
          f"range: [{reference.min():.3f}, {reference.max():.3f}]")