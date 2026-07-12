import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

sys.path.append(str(Path(__file__).parent))
from classical import restore_classical_pipeline
from metrics import calculate_uiqm, calculate_uciqe


def evaluate_paired_folder(
    degraded_dir: Path, 
    reference_dir: Path,
    output_dir: Path, 
    dataset_label: str,
    ) -> list[dict]:
    """
    Restore every image in degraded_dir, compare it against its matching
    file (by filename) in reference_dir, and compute full-reference
    metrics (PSNR, SSIM). Images without a reference match are skipped.
    """
    results = []
    output_dir.mkdir(parents=True, exist_ok=True)

    image_files = sorted(degraded_dir.glob("*"))

    for degraded_path in image_files:
        reference_path = reference_dir / degraded_path.name

        if not reference_path.exists():
            print(f"  [skip] no reference match for {degraded_path.name}")
            continue

        degraded_img = cv2.imread(str(degraded_path))
        reference_img = cv2.imread(str(reference_path))

        if degraded_img is None or reference_img is None:
            print(f"  [skip] could not read {degraded_path.name}")
            continue

        restored_img = restore_classical_pipeline(degraded_img)

        out_path = output_dir / degraded_path.name
        cv2.imwrite(str(out_path), restored_img)

        if restored_img.shape != reference_img.shape:
            reference_img = cv2.resize(
                reference_img,
                (restored_img.shape[1], restored_img.shape[0])
            )
            print(f"  [note] resized reference to match {degraded_path.name}")

        psnr_val = peak_signal_noise_ratio(reference_img, restored_img)
        ssim_val = structural_similarity(
            reference_img, restored_img, channel_axis=-1
        )

        results.append({
            "dataset": dataset_label,
            "filename": degraded_path.name,
            "psnr": psnr_val,
            "ssim": ssim_val,
        })

    return results


def evaluate_unpaired_folder(
    images_dir: Path, 
    output_dir: Path,
    dataset_label: str,
    ) -> list[dict]:
    """
    Restore every image in images_dir, compute no-reference metrics
    (UIQM, UCIQE) on the restored result. No ground truth involved.
    """
    results = []
    output_dir.mkdir(parents=True, exist_ok=True)

    image_files = sorted(images_dir.glob("*"))

    for img_path in image_files:
        img = cv2.imread(str(img_path))

        if img is None:
            print(f"  [skip] could not read {img_path.name}")
            continue

        restored_img = restore_classical_pipeline(img)

        out_path = output_dir / img_path.name
        cv2.imwrite(str(out_path), restored_img)

        uiqm_val = calculate_uiqm(restored_img)
        uciqe_val = calculate_uciqe(restored_img)

        results.append({
            "dataset": dataset_label,
            "filename": img_path.name,
            "uiqm": uiqm_val,
            "uciqe": uciqe_val,
        })

    return results


def summarize(results: list[dict], metric_keys: list[str]) -> dict:
    """
    Given a list of per-image result dicts, compute mean and std
    for each requested metric key.
    """
    summary: dict[str, Optional[float]] = {"n": len(results)}

    for key in metric_keys:
        values = [r[key] for r in results if key in r]
        summary[f"{key}_mean"] = float(np.mean(values)) if values else None
        summary[f"{key}_std"] = float(np.std(values)) if values else None

    return summary


# vision/ is parents[2] relative to this file — keeps paths correct
# regardless of the working directory the script is run from.
VISION_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = VISION_ROOT / "data" / "public_datasets"
PROCESSED_ROOT = VISION_ROOT / "data" / "processed"
DOCS_ROOT = VISION_ROOT / "docs"


def _group_by_dataset(results: list[dict]) -> dict[str, list[dict]]:
    """Split a flat list of per-image results into groups by dataset.

    Args:
        results: Per-image result dicts, each containing a "dataset" key.

    Returns:
        A dict mapping each dataset label to its own list of results.
    """
    grouped: dict[str, list[dict]] = {}
    for r in results:
        grouped.setdefault(r["dataset"], []).append(r)
    return grouped


def _format_paired_table(paired_results: list[dict]) -> str:
    """Build a markdown table of PSNR/SSIM results, one row per dataset.

    Args:
        paired_results: Combined results from all evaluate_paired_folder()
            calls (UIEB + every EUVP scene).

    Returns:
        A markdown table, with a final row summarizing all paired
        datasets combined.
    """
    grouped = _group_by_dataset(paired_results)

    lines = [
        "| Dataset | n | PSNR (mean ± std) | SSIM (mean ± std) |",
        "|---|---|---|---|",
    ]

    for label, items in grouped.items():
        s = summarize(items, ["psnr", "ssim"])
        lines.append(
            f"| {label} | {s['n']} | "
            f"{s['psnr_mean']:.2f} ± {s['psnr_std']:.2f} | "
            f"{s['ssim_mean']:.4f} ± {s['ssim_std']:.4f} |"
        )

    if paired_results:
        s_all = summarize(paired_results, ["psnr", "ssim"])
        lines.append(
            f"| **Overall (paired)** | {s_all['n']} | "
            f"{s_all['psnr_mean']:.2f} ± {s_all['psnr_std']:.2f} | "
            f"{s_all['ssim_mean']:.4f} ± {s_all['ssim_std']:.4f} |"
        )

    return "\n".join(lines)


def _format_unpaired_table(unpaired_results: list[dict]) -> str:
    """Build a markdown table of UIQM/UCIQE results, one row per dataset.

    Args:
        unpaired_results: Combined results from all
            evaluate_unpaired_folder() calls (RUIE/UIQS folders A-E).

    Returns:
        A markdown table as a single string.
    """
    grouped = _group_by_dataset(unpaired_results)

    lines = [
        "| Dataset | n | UIQM (mean ± std) | UCIQE (mean ± std) |",
        "|---|---|---|---|",
    ]

    for label, items in grouped.items():
        s = summarize(items, ["uiqm", "uciqe"])
        lines.append(
            f"| {label} | {s['n']} | "
            f"{s['uiqm_mean']:.4f} ± {s['uiqm_std']:.4f} | "
            f"{s['uciqe_mean']:.2f} ± {s['uciqe_std']:.2f} |"
        )

    return "\n".join(lines)


def write_markdown_report(
    paired_results: list[dict],
    unpaired_results: list[dict],
    output_path: Path,
) -> None:
    """Assemble and write the full classical-baseline metrics report.

    Args:
        paired_results: Combined PSNR/SSIM results (UIEB + EUVP scenes).
        unpaired_results: Combined UIQM/UCIQE results (RUIE folders A-E).
        output_path: Destination path for the generated report, e.g.
            vision/docs/metrics_log.md.
    """
    # Findings below are hardcoded: qualitative conclusions from prior
    # diagnostic sessions, not values derived from `results` at runtime.
    report = f"""# HydroTrashVision — Classical Baseline Metrics Log

## Dataset structure (reference, not included in repo)

```
UIEB/
  raw-890/<file>.png          <- degraded
  reference-890/<file>.png    <- reference (same filename)

EUVP/Paired/<scene>/
  trainA/<file>.jpg           <- degraded
  trainB/<file>.jpg           <- reference (same filename)
  validation/                 <- excluded (flat structure, no A/B split)

RUIE/UIQS/<A|B|C|D|E>/<file>  <- no reference
```

## Full-reference metrics (PSNR / SSIM)

{_format_paired_table(paired_results)}

## No-reference metrics (UIQM / UCIQE)

{_format_unpaired_table(unpaired_results)}

## Known findings

### UIQM order inversion

UCIQE was validated against RUIE's own UIQS A-E quality labels and
tracks them correctly. UIQM showed inverted ordering on the same
comparison, attributable to its UICM and UISM sub-metrics being
sensitive to color cast and noise respectively -- consistent with
documented literature limitations (Li & Cavallaro, ICIP 2022,
arXiv:2207.05470).

### Extended RUIE UIQS validation (A-E)

The Session 3 preliminary test only compared the two extremes (UIQS
A/E, n=15 each). Running the full evaluation across all five UIQS
folders (A-E, n=726 each) confirms:

- **UIQM** order inversion is monotonic across all five folders
  (3.70 -> 3.79 -> 3.98 -> 4.56 -> 4.63, A to E), reinforcing the
  Session 3 finding with full-dataset granularity rather than just
  the extremes.
- **UCIQE** tracks the expected quality order (A best, E worst) for
  A-B-C but is not perfectly monotonic at the low-quality end: folder
  D (49.13) scores slightly lower than folder E (50.55) -- a small
  but measurable deviation from the dataset's own A-E quality
  labeling.

This closes the extended-validation item left open in Session 3.

### Restoration artifacts in classical UDCP baseline

The classical method (UDCP) shows two independent, evidence-confirmed
sources of restoration artifacts:

1. Strong color imbalance (very dim red channel, gain_r > ~3) combined
   with low-amplitude JPEG compression in the source image, amplified
   by white balance gain and the division in the dehazing reconstruction
   formula. Moderate correlation observed between gain_r and PSNR
   (Pearson r=-0.356, n=890) -- present but not the dominant driver.

2. Large, uniformly-textured bright regions (e.g. sand, open water)
   violate the Dark Channel Prior's core assumption that some color
   channel is near-zero in every local patch -- a known limitation of
   DCP-based methods in general (historically documented for sky
   regions in aerial dehazing). Confirmed via visual inspection of a
   near-color-neutral image (gain_r=1.43) still producing severe
   restoration noise, isolating this from cause (1).

Neither is an implementation defect. Both are documented, expected
limitations of the classical baseline, whose purpose in this project
is to serve as a measured comsparison point -- not a final restoration
method. Deep learning architectures (next phase) are expected to be
more robust to both failure modes.
"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    print(f"Report written to {output_path}")


def main() -> None:
    """Run the full classical-baseline evaluation across all datasets."""
    paired_results: list[dict] = []
    unpaired_results: list[dict] = []

    uieb_results = evaluate_paired_folder(
        degraded_dir=DATA_ROOT / "UIEB" / "raw-890",
        reference_dir=DATA_ROOT / "UIEB" / "reference-890",
        output_dir=PROCESSED_ROOT / "UIEB",
        dataset_label="UIEB",
    )
    paired_results.extend(uieb_results)

    euvp_scenes = ["underwater_imagenet", "underwater_scenes", "underwater_dark"]
    for scene in euvp_scenes:
        scene_dir = DATA_ROOT / "EUVP" / "Paired" / scene
        euvp_results = evaluate_paired_folder(
            degraded_dir=scene_dir / "trainA",
            reference_dir=scene_dir / "trainB",
            output_dir=PROCESSED_ROOT / "EUVP" / scene,
            dataset_label=f"EUVP-{scene}",
        )
        paired_results.extend(euvp_results)

    for letter in ["A", "B", "C", "D", "E"]:
        ruie_results = evaluate_unpaired_folder(
            images_dir=DATA_ROOT / "RUIE" / "UIQS" / letter,
            output_dir=PROCESSED_ROOT / "RUIE" / letter,
            dataset_label=f"RUIE-{letter}",
        )
        unpaired_results.extend(ruie_results)

    write_markdown_report(
        paired_results=paired_results,
        unpaired_results=unpaired_results,
        output_path=DOCS_ROOT / "metrics_log.md",
    )


if __name__ == "__main__":
    main()