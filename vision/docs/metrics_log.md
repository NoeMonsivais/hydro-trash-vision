# HydroTrashVision — Classical Baseline Metrics Log

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

| Dataset | n | PSNR (mean ± std) | SSIM (mean ± std) |
|---|---|---|---|
| UIEB | 890 | 12.74 ± 2.90 | 0.6372 ± 0.1461 |
| EUVP-underwater_imagenet | 3700 | 15.35 ± 2.23 | 0.6110 ± 0.0795 |
| EUVP-underwater_scenes | 2185 | 14.45 ± 2.56 | 0.5905 ± 0.1031 |
| EUVP-underwater_dark | 5550 | 14.93 ± 2.12 | 0.6768 ± 0.1042 |
| **Overall (paired)** | 12325 | 14.81 ± 2.39 | 0.6389 ± 0.1073 |

## No-reference metrics (UIQM / UCIQE)

| Dataset | n | UIQM (mean ± std) | UCIQE (mean ± std) |
|---|---|---|---|
| RUIE-A | 726 | 3.7043 ± 0.3683 | 65.19 ± 7.00 |
| RUIE-B | 726 | 3.7892 ± 0.4659 | 56.95 ± 6.35 |
| RUIE-C | 726 | 3.9786 ± 0.5844 | 54.30 ± 7.41 |
| RUIE-D | 726 | 4.5576 ± 0.7842 | 49.13 ± 7.61 |
| RUIE-E | 726 | 4.6304 ± 0.8704 | 50.55 ± 7.90 |

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
is to serve as a measured comparison point -- not a final restoration
method. Deep learning architectures (next phase) are expected to be
more robust to both failure modes.
