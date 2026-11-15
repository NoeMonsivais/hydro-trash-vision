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

## Deep learning restoration (CNN simple) — Full-reference metrics (PSNR / SSIM)

Evaluated on the `test` split only (never train/val), from a
`RestorationCNN` checkpoint trained for 50 epochs on a cloud GPU
(Kaggle, Tesla T4), batch size 16, L1 loss, Adam (lr=1e-4).

| Dataset | n | PSNR (mean ± std) | SSIM (mean ± std) |
|---|---|---|---|
| UIEB | 89 | 18.95 ± 3.41 | 0.8271 ± 0.1065 |
| EUVP-underwater_imagenet | 370 | 23.96 ± 2.31 | 0.8316 ± 0.0566 |
| EUVP-underwater_scenes | 219 | 26.82 ± 2.36 | 0.8746 ± 0.0497 |
| EUVP-underwater_dark | 555 | 22.57 ± 3.84 | 0.9004 ± 0.0724 |
| **Overall (CNN, test split)** | 1233 | 23.48 ± 3.74 | 0.8699 ± 0.0746 |

### Classical vs. CNN comparison

| Method | n | PSNR (mean ± std) | SSIM (mean ± std) |
|---|---|---|---|
| Classical (UDCP) | 12325 | 14.81 ± 2.39 | 0.6389 ± 0.1073 |
| CNN simple (test split only) | 1233 | 23.48 ± 3.74 | 0.8699 ± 0.0746 |

Note: the classical baseline was measured on 100% of the paired
datasets (it does not learn, so there is no train/test distinction).
The CNN is measured strictly on its held-out test split, never seen
during training -- the comparison is fair, since the CNN's test split
is a subset of the exact same images the classical method was
measured on.

The CNN shows a substantial and consistent improvement over the
classical baseline across all four sub-datasets, with no dataset
lagging behind -- consistent with the CNN not being subject to the
two structural limitations documented for UDCP (block artifacts from
the transmission-map division, and Dark Channel Prior failure on
large uniform regions).

### Block A closure decision

Per the project's documented decision to escalate to a second deep
learning architecture (UWCNN/WaterNet) only if the simple CNN proves
insufficient: given the magnitude and consistency of this improvement,
Block A restoration is considered closed with the classical baseline
and this single CNN architecture. No second architecture is planned
unless future evidence (e.g. evaluation on real river footage)
indicates otherwise.

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

## Block B — Object Detection (YOLO)

### Dataset composition and class scheme

Three public datasets were combined for object detection:
TrashCan-Material, RiSID, and WPBB.

The unified detection dataset contains nine classes:

- `trash_etc`
- `trash_fabric`
- `trash_fishing_gear`
- `trash_metal`
- `trash_paper`
- `trash_plastic`
- `trash_rubber`
- `trash_wood`
- `rov`

Biological categories from TrashCan were treated as background.
RiSID's `non_pla` annotations were also excluded from the target
classes, while its `pla` annotations were mapped to `trash_plastic`.
WPBB's plastic bags and bottles were mapped to `trash_plastic`.

The combined dataset contains 18,555 annotated instances.
The most frequent class is `trash_plastic`, with 10,888 instances
(approximately 58.7% of all annotations).

### Architecture comparison

Two object detection architectures were trained and evaluated:
YOLOv8s and YOLO26s.

Both models were trained for 70 epochs using the unified
nine-class dataset. Training was performed on Kaggle using
an NVIDIA Tesla T4 GPU.

Weighted oversampling was implemented through a custom
`WeightedDetectionTrainer` using `WeightedRandomSampler`.

YOLO26 used a domain-specific augmentation override of
`degrees=15.0`. Other training settings followed the
documented experiment configuration.

| Model | Epochs | mAP50 | mAP50-95 |
|---|---:|---:|---:|
| YOLOv8s | 70 | 0.4782 | 0.3363 |
| YOLO26s | 70 | 0.4106 | 0.2210 |

YOLOv8s achieved the highest performance on both reported
metrics and was selected as the preferred architecture
for subsequent pipeline development.

YOLO26s showed a performance plateau around epoch 62,
whereas YOLOv8s continued improving through epoch 70.

Additional training epochs were not considered necessary
for the initial architecture selection.

### Known limitation — Low recall for trash_plastic

Despite being the most represented class in the combined
dataset, `trash_plastic` achieved an AP of 0.2513 in the
documented evaluation.

A subsequent diagnostic investigation identified:

- Approximately 71% of plastic instances were predicted
  as background in the analyzed confusion matrix.
- The primary failure mode was missed detections rather
  than confusion with other material classes.
- Plastic objects had the smallest median bounding-box
  area, approximately ten times smaller than those of
  other classes.
- Approximately 70% of plastic annotations originated
  from RiSID, which contains river-surface images often
  captured from elevated viewpoints.

The results suggest that object scale and dataset-source
distribution contribute to the low detection performance.

This limitation is particularly relevant because the
intended robotic application involves closer-range
underwater imagery, which differs substantially from
RiSID's elevated river-surface perspective.

However, improved performance in the target environment
has not yet been experimentally demonstrated.

The finding is retained as a documented limitation
rather than treated as evidence of an implementation
defect.

### RiSID annotation limitations

Two annotation-related issues were identified during
dataset preparation and training.

First, approximately 85% of RiSID's original bounding-box
annotations were found to have dimensions scaled to
approximately 0.667 times their corresponding segmentation
extent.

This issue was addressed by deriving bounding boxes
directly from segmentation polygon coordinates.

Second, 133 RiSID images contained annotation coordinates
outside the valid normalized image range:

- 111 training images.
- 22 validation images.

These cases originated from segmentation polygons extending
beyond the image boundaries.

The annotation converter was verified to reproduce the
source polygons correctly.

The affected images were intentionally excluded by
Ultralytics rather than clipping their bounding boxes,
preserving the documented treatment of truncated objects
for future area-based mass estimation.

### Block B architecture selection and remaining validation

YOLOv8s is selected as the detection architecture based
on its superior mAP50 and mAP50-95 results.

The architecture comparison and the investigation of
the `trash_plastic` anomaly are considered complete
for the current experimental stage.

The following validation remains pending:

- Evaluation on representative close-range river or
  underwater imagery.
- Assessment of detection performance under the
  intended robotic operating conditions.
- Investigation of how missed detections and bounding-box
  errors propagate into area-based mass estimation.

No additional architecture training is planned unless
new experimental evidence demonstrates a need for it.
