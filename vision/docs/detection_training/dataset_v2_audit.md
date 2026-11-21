# Detection Dataset v1 Audit

Date: 2026-11-21

Status: Initial diagnostic audit completed; corrective implementation pending.

## 1. Scope and historical reference

This audit examines the combined YOLO detection dataset generated from TrashCan-Material, RiSID and In-Water Plastic Bags and Bottles (WPBB). It investigates geometry, source annotations, class support, group overlap and exact file duplication. Counts below are observed from local, read-only diagnostics of the existing dataset; they are not results from a corrected dataset.

The historical inference implementation is preserved at commit `a8659a0`. The selected YOLOv8s checkpoint is `vision/checkpoints/detection/yolov8_s/weights/best.pt` (SHA-256: `fbafd6fb49100668be4c76aaa6009a76ad2b6523af2177d63caaffcb1198ef78`). The historical architecture comparison is recorded in [yolo_comparison_results.json](yolo_comparison_results.json).

| Model | mAP50 | mAP50-95 |
| --- | ---: | ---: |
| YOLOv8s | 0.4782 | 0.3363 |
| YOLO26s | 0.4106 | 0.2210 |

YOLOv8s was selected under the original protocol. Its plastic AP was 0.2513. The historical metrics must not be interpreted as performance on independent unseen videos. They remain useful for reproducing the original comparison.

The detector adapter passed 46 unit tests and the joint regression suite passed 180 tests. Real-checkpoint diagnostics processed empty and nonempty predictions successfully. The nine-image control produced detections in eight images (16 predictions total); the target class appeared in six images. This was a deliberately selected control, not an accuracy estimate. The real pipeline processed one fishing-gear prediction and three ROV predictions, called the diagnostic mass estimator once and produced no numeric mass. These checks establish functional integration for the tested cases, not detection accuracy or physical mass validity.

## 2. Dataset inventory

| Source | Train images | Val images | Train annotations | Val annotations |
| --- | ---: | ---: | ---: | ---: |
| TrashCan | 6,008 | 1,204 | 7,586 | 1,945 |
| RiSID | 5,884 | 1,472 | 6,206 | 1,561 |
| WPBB | 720 | 180 | 996 | 261 |
| Total | 12,612 | 2,856 | 14,788 | 3,767 |

The audit covered 15,468 images and 18,555 retained annotations. Empty YOLO label files were observed for 753 train and 98 val TrashCan images, and 160 train and 44 val RiSID images. No missing labels or images were reported by the initial audit. An empty label file denotes absence of retained labels, not verified absence of all debris.

## 3. Geometry audit

Normalized YOLO boxes were checked for finite coordinates, positive dimensions, class IDs and canvas bounds. The normalized bounds tolerance was 2e-6. Severity was measured as maximum overflow along any image edge, in pixels, using actual image dimensions. The pixel thresholds included a 0.01 px margin. A non-intersecting box was counted separately, regardless of overflow distance.

| Source | Total boxes | Outside boxes | Up to 1 px | 1-2 px | 2-5 px | Over 5 px, intersects | No intersection |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| TrashCan train | 7,586 | 2,478 | 2,478 | 0 | 0 | 0 | 0 |
| TrashCan val | 1,945 | 596 | 596 | 0 | 0 | 0 | 0 |
| RiSID train | 6,206 | 352 | 94 | 10 | 15 | 109 | 124 |
| RiSID val | 1,561 | 77 | 18 | 1 | 5 | 23 | 30 |
| WPBB train | 996 | 0 | 0 | 0 | 0 | 0 | 0 |
| WPBB val | 261 | 0 | 0 | 0 | 0 | 0 | 0 |
| Total | 18,555 | 3,503 | 3,186 | 11 | 20 | 132 | 154 |

Overall, 18.88% of retained boxes exceeded the audit bounds. This is not the percentage of unusable annotations or the rejection rate of Ultralytics. The original training loader's acceptance and rejection counts have not been reconstructed in this audit.

### TrashCan

All 3,074 outside boxes exceeded a canvas edge by approximately one pixel or less. The original COCO audit recovered the same counts; each outside box matched its polygon extent to the reported 0.001 px precision. Repeated boundary values included 480.998, 270.998 and 360.998 px. All 7,212 image dimensions matched their declared sizes. All 9,531 retained annotations passed basic polygon-structure and finite-coordinate checks.

The overflow exists in the source annotations, not solely in the YOLO conversion. A systematic coordinate-boundary convention is a plausible explanation, but its exact origin is not established. A bounded clipping policy is proposed; polygon validity, visible mask fidelity and semantic labeling remain to be assessed. There is no evidence here that all outside boxes should be discarded.

### RiSID

All 7,356 images were checked without a reported declared-size mismatch. All 7,767 retained plastic annotations passed basic polygon checks. Original boxes and polygon extents matched within 0.01 px in 1,166 cases and differed in 6,601 cases (84.99%).

| Original box status | Polygon-derived inside | Polygon-derived partial | Polygon-derived no intersection |
| --- | ---: | ---: | ---: |
| Inside | 7,338 | 146 | 0 |
| Partial | 0 | 124 | 0 |
| No intersection | 0 | 5 | 154 |
| Total | 7,338 | 275 | 154 |

For annotation 943, the original box width and height were approximately two-thirds of the polygon extent, with the same top-left coordinate. This supports a previously observed scale discrepancy, but does not establish the correct visual geometry of every annotation. Some entirely outside boxes have both representations outside the image. The largest overflow among the printed examples was approximately 586 px (not a dataset-wide maximum).

The polygon-derived boxes are candidates, not automatically correct ground truth. Partial intersections require a documented treatment; entirely outside annotations require review or exclusion. Do not silently turn uncertain positive images into background samples.

### WPBB

All 900 images measured 1920 x 1080 px. No geometry violations were reported for its 1,257 retained VOC boxes. The local README describes 500 samples with bags and 400 with bottles, and PASCAL VOC 2012 annotations. The annotations used here do not supply segmentation masks. Semantic quality is not established by the geometry check.

## 4. Group overlap and duplicate audit

| Source / grouping rule | Train groups | Val groups | Shared | Distinct total | Val groups in train |
| --- | ---: | ---: | ---: | ---: | ---: |
| TrashCan video identifier | 306 | 133 | 127 | 312 | 95.49% |
| RiSID filename sequence candidate | 300 | 280 | 279 | 301 | 99.64% |
| WPBB sequence candidate | 6 | 6 | 6 | 6 | 100% |
| WPBB object identifier candidate | 4 | 4 | 4 | 4 | 100% |

All filenames matched the corresponding grouping patterns. TrashCan uses the `vid_<digits>` prefix; RiSID uses the prefix preceding the final frame and two numeric fields; WPBB uses the sequence prefix or the more conservative Bag/Bottle-number prefix. Filename groups are proxies for acquisition provenance and do not prove distinct physical objects or campaigns. The same original scene may span multiple identifiers.

For RiSID, the nearest train-frame index for each validation image within a shared group was examined:

| Index distance | Validation images |
| --- | ---: |
| 0 | 16 |
| 1-5 | 1,071 |
| 6-30 | 361 |
| 31-100 | 8 |
| Over 100 | 15 |
| Exclusive validation group | 1 |

1,448 of 1,472 validation images (98.37%) had a matching group and an index distance of at most 30. Index differences are not confirmed elapsed time; equal indices may correspond to different crops. Visual duplicate identity was not verified.

WPBB object candidates contain 200 (Bag1), 300 (Bag4), 43 (Bottle1) and 357 (Bottle2) images. The last two largest groups, Bag4 and Bottle2, contribute 73% of all WPBB samples. The README does not establish the number of physical objects.

SHA-256 hashing of all 15,468 combined images yielded 15,468 distinct hashes, with zero exact duplicate groups across splits or sources. This excludes exact binary duplicates in the scanned set, not re-encoded images, crops or near-duplicate frames.

Group overlap compromises an interpretation of the original validation metrics as generalization to independent sequences. Its quantitative effect on model accuracy has not been measured.

## 5. Class and domain support

| TrashCan class | Annotations | Images | Videos | Top five videos share |
| --- | ---: | ---: | ---: | ---: |
| trash_etc | 2,039 | 1,740 | 154 | 17.02% |
| trash_fabric | 314 | 269 | 21 | 51.59% |
| trash_fishing_gear | 196 | 167 | 14 | 65.82% |
| trash_metal | 1,126 | 1,093 | 62 | 26.02% |
| trash_paper | 193 | 167 | 10 | 84.97% |
| trash_plastic | 1,864 | 1,753 | 97 | 21.73% |
| trash_rubber | 142 | 142 | 7 | 93.66% |
| trash_wood | 340 | 324 | 20 | 49.41% |
| rov | 3,317 | 2,721 | 145 | 22.79% |

Thirty-one TrashCan videos had no retained annotations. Video counts are not counts of unique physical objects. Rubber is particularly concentrated: one video contains 88 of 142 annotations (61.97%). More repeated frames or augmentation cannot replace additional independent object diversity.

RiSID contributes 6,206 of 8,692 plastic annotations in the original training labels (71.40%), before any loader filtering or sampling. Its river-surface imagery differs from the intended close-range underwater setting. Domain shift and small objects are plausible contributors to low plastic recall, not established causal effects. WPBB also contributes only plastic under the retained class scheme.

## 6. Conclusions and limitations

- Retain the v1 dataset, historical checkpoint and results unchanged.
- Prioritize group-independent evaluation and source-aware data quality over architecture escalation.
- Use TrashCan as the primary candidate domain for v2; treat RiSID and WPBB as auxiliary experimental sources.
- Preserve available source polygons for future segmentation, without treating structural checks as mask validation.
- Separate detection visibility from suitability for mass estimation. Never infer physical mass from an uncalibrated box or mask.
- Evaluate class support, label semantics, near duplicates and external underwater performance before making real-world generalization claims.

These findings come from local terminal diagnostics and their reported outputs. The temporary diagnostic scripts have not been consolidated into a versioned audit runner. Reproducibility of the next dataset build requires committed validation code, input fingerprints, split manifests and machine-readable reports. The corrective design is specified separately in [dataset_v2_design.md](dataset_v2_design.md).
