"""
Evaluation script for the trained CNN restoration model.

Unlike evaluate_restoration.py (which measures the classical method over
100% of the paired datasets, since it doesn't "learn"), this script must
evaluate the CNN strictly on the `test` split saved in
datasets_splits.json -- never train/val. Measuring on images the model
saw during training would inflate PSNR/SSIM and make any comparison
against the classical baseline invalid.
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

sys.path.append(str(Path(__file__).parent))
from deep_model import RestorationCNN
from dataset import UnderwaterRestorationDataset
from evaluate_restoration import summarize, _group_by_dataset


VISION_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_ROOT = VISION_ROOT / "data" / "processed" / "cnn_simple"
CHECKPOINT_PATH = VISION_ROOT / "checkpoints" / "restoration_cnn" / "best_model.pt"


def get_device() -> torch.device:
    """Hardware-agnostic device selection, same logic as train_restoration.py."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(checkpoint_path: Path, device: torch.device) -> RestorationCNN:
    """Reconstruct the architecture, then load trained weights into it.

    The checkpoint only stores the learned weights (state_dict), not the
    class definition -- we must instantiate the exact same architecture
    before the weights can be loaded.
    """
    model = RestorationCNN()
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def tensor_to_image(tensor: torch.Tensor) -> np.ndarray:
    """Convert a (C, H, W) float32 [0,1] RGB tensor back into an
    (H, W, C) uint8 BGR image -- the inverse of _load_image() in dataset.py.
    """
    img = tensor.detach().cpu().numpy()
    img = np.transpose(img, (1, 2, 0))           # CHW -> HWC
    img = np.clip(img, 0, 1) * 255
    img = img.astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def evaluate_test_split(
    model: RestorationCNN,
    device: torch.device,
    output_dir: Path,
) -> list[dict]:
    """Run inference on every pair in the `test` split and measure
    PSNR/SSIM against the real reference image.
    """
    dataset = UnderwaterRestorationDataset(split="test")
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []

    with torch.no_grad():
        for idx in range(len(dataset)):
            label, filename = dataset.entries[idx]
            degraded_tensor, reference_tensor = dataset[idx]

            degraded_batch = degraded_tensor.unsqueeze(0).to(device)
            restored_batch = model(degraded_batch)
            restored_tensor = restored_batch.squeeze(0)

            restored_img = tensor_to_image(restored_tensor)
            reference_img = tensor_to_image(reference_tensor)

            out_path = output_dir / f"{label}__{filename}"
            cv2.imwrite(str(out_path), restored_img)

            psnr_val = peak_signal_noise_ratio(reference_img, restored_img)
            ssim_val = structural_similarity(reference_img, restored_img, channel_axis=-1)

            results.append({
                "dataset": label,
                "filename": filename,
                "psnr": psnr_val,
                "ssim": ssim_val,
            })

    return results


def main() -> None:
    device = get_device()
    print(f"Using device: {device}")

    model = load_model(CHECKPOINT_PATH, device)
    results = evaluate_test_split(model, device, PROCESSED_ROOT)

    grouped = _group_by_dataset(results)
    for label, items in grouped.items():
        s = summarize(items, ["psnr", "ssim"])
        print(f"{label}: n={s['n']}  PSNR={s['psnr_mean']:.2f}±{s['psnr_std']:.2f}  "
              f"SSIM={s['ssim_mean']:.4f}±{s['ssim_std']:.4f}")

    s_all = summarize(results, ["psnr", "ssim"])
    print(f"Overall (CNN, test split): n={s_all['n']}  "
          f"PSNR={s_all['psnr_mean']:.2f}±{s_all['psnr_std']:.2f}  "
          f"SSIM={s_all['ssim_mean']:.4f}±{s_all['ssim_std']:.4f}")


if __name__ == "__main__":
    main()