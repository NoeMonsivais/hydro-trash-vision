# Detection Dataset v2 Design

Date: 2026-11-21

Status: Proposed design; implementation and acceptance tests pending.

## 1. Purpose and scope

Build a reproducible, source-aware dataset for underwater debris detection with independent acquisition groups and traceable geometry corrections. Preserve source polygons for future instance segmentation and calibrated-area mass estimation. The current inference adapter and shared pipeline contracts remain unchanged. The audit rationale is recorded in [dataset_v2_audit.md](dataset_v2_audit.md).

The v1 dataset, source annotations, checkpoints and historical results must remain untouched. The next training experiment must not begin until the v2 dataset passes its acceptance checks.

## 2. Class and source policy

Use `vision.src.core.contracts.CLASS_NAMES` as the canonical class order:

| ID | Class |
| ---: | --- |
| 0 | trash_etc |
| 1 | trash_fabric |
| 2 | trash_fishing_gear |
| 3 | trash_metal |
| 4 | trash_paper |
| 5 | trash_plastic |
| 6 | trash_rubber |
| 7 | trash_wood |
| 8 | rov |

TrashCan is the primary evaluation domain and supplies all nine classes. RiSID `pla` and WPBB `plastic bag` / `plastic bottle` map to `trash_plastic`. Excluded biological and non-target labels must be recorded rather than silently confused with missing annotations. Review the semantics of excluded RiSID `non_pla` objects before treating their images as negative. ROV remains a detection class, excluded from physical waste mass estimation.

RiSID and WPBB are optional auxiliary training sources. Their value is to be measured using a fixed underwater validation partition. WPBB is box-only for the current source format; no synthetic rectangular masks are accepted as true instance masks.

## 3. Processing architecture

Source parsers -> geometry validation -> unified manifest -> grouped partition assignment -> task-specific export -> acceptance report.

Keep parsing and geometry independent of model training libraries. Use Python 3.10 and existing dependencies. Introduce additional dependencies only when justified and verified. Preserve source provenance and original geometry; all corrections are derived records.

### Proposed module boundaries

| Module | Responsibility |
| --- | --- |
| `geometry.py` | Pure coordinate conversion, validation, intersection and bounded clipping |
| `manifest.py` | Typed image and annotation records; invariants and serialization |
| `parsers.py` | COCO/VOC ingestion, source-specific mapping and group extraction |
| `splits.py` | Deterministic group assignment with class-support constraints |
| `export_yolo.py` | Detection label export, safe names, image links and dataset configuration |
| `build.py` | Build coordination, input/output checks and reports |

Implement in that order, using focused unit tests before integration. Segmentation export and tracking are separate future features; preserving polygons and sequence metadata does not implement either task.

## 4. Geometry contract

Use finite absolute `xyxy` coordinates in continuous image space, with positive image dimensions W and H. A valid visible rectangle satisfies `0 <= x1 < x2 <= W` and `0 <= y1 < y2 <= H`. Width and height must be strictly positive. Edge coordinates equal to W or H are allowed. COCO `xywh` converts by adding width and height to the top-left coordinate; YOLO normalization is performed after validation and any approved clipping.

The geometry module must distinguish invalid input, zero-area intersection, partial intersection and contained boxes. It must not reorder inverted coordinates, silently coerce non-finite values or decide whether a label is semantically correct. The clipping primitive intersects a rectangle with the canvas; approval of a correction is a separate source policy. Numerical tolerances and export precision must be explicit and tested, including round-trip boundary behavior.

| Source | Candidate geometry | Proposed policy |
| --- | --- | --- |
| TrashCan | Original COCO box | Clip boundary excess of at most 1.01 px after confirming source/image dimensions. Larger excess or invalid geometry requires review. Preserve original polygon. |
| RiSID | Polygon extent | Accept geometrically valid interiors as candidates. Review partial or absent intersections and unresolved scale/semantic discrepancies before export. |
| WPBB | Original VOC corners | Verify source convention and actual dimensions; retain valid boxes. Review any unexpected violation. |

Use explicit annotation outcomes: `valid`, `clipped`, `review_required`, `rejected`. Each outcome records its reason and original geometry. A geometry-valid status is not a guarantee of correct class or mask. Do not export unresolved or rejected annotations as positives. Quarantine images with uncertain visible target labels rather than silently adding them as negative images. A genuinely out-of-canvas label may be removed only under a documented policy confirming the remaining labels are suitable.

For masks, preserve original polygons and separately validate polygon topology, visible area, rasterization and semantic fidelity before training. Rectangle clipping is not polygon clipping. Boundary contact and truncation must be recorded as cues, not assumed to prove completeness or physical suitability for mass estimation.

## 5. Unified manifest

The manifest should contain schema and policy versions, source-relative paths, original split, stable source-qualified image and annotation identifiers, image dimensions, source file fingerprints, class mapping, original box and polygon (when available), derived visible box, correction outcome/reason, boundary contact and acquisition group. Record frame index when available, without treating it as time or track identity. Include final split or quarantine state and separate export eligibility for detection and future segmentation.

Keep group provenance and any manual overrides auditable. Do not depend on machine-specific absolute paths for identity. Record source annotation fingerprints, split algorithm version, seed, configuration and report totals. Identical inputs and configuration must generate identical manifests and split assignments.

## 6. Partition protocol

Use source-qualified acquisition groups. Initial rules:

- TrashCan: `vid_<digits>` prefix; investigate shared parent acquisition if metadata permits.
- RiSID: prefix before frame index and two numeric suffixes; validate naming assumptions and merge related groups when justified.
- WPBB: conservative Bag/Bottle-number group, keeping all associated trims together.

Never assign one acquisition group to multiple splits. Co-assign or quarantine confirmed duplicate/related groups, including any cross-source matches. A SHA-256 check alone does not establish full visual independence.

For TrashCan, start with an approximate 70/15/15 train/val/test target. Balance image counts, annotation counts and independent class-support groups. Require all nine classes in each split where feasible; explicitly report infeasibility rather than silently duplicating or splitting groups. Aim for at least two groups per minority class in val and test when compatible with sufficient training support. Rubber has only seven candidate videos, so test precision and uncertainty are limited even with a valid split. Do not promise that a target ratio or group support is feasible until the assignment has been verified.

Freeze the TrashCan split manifest before comparative training. Auxiliaries can be train-only for the main experiments; no auxiliary samples enter the shared primary validation or test sets. If evaluating an auxiliary domain separately, define its own independent held-out groups first. The previously trained v1 checkpoint is not a valid unseen-test comparator on a reshuffled version of previously used data.

Record split results per class and source, including group counts, co-occurrence and negative images. Preserve the test set for final evaluation; use validation for model and hyperparameter selection. Prefer a separate, newly collected underwater test set for stronger external validity.

## 7. Controlled detection experiments

| Experiment | Training sources |
| --- | --- |
| E1 | TrashCan |
| E2 | TrashCan + WPBB |
| E3 | TrashCan + RiSID |
| E4 | TrashCan + WPBB + RiSID |

Use YOLOv8s as the initial reference architecture, starting from the same initial pretrained weights rather than the v1 fine-tuned checkpoint. Keep the primary TrashCan training subset and validation partition fixed, with comparable image size, augmentation, optimizer, initialization and seed. Record the number of samples, effective class sampling and optimization steps; a fixed epoch count alone does not equalize training budgets. Stage experiments according to resource availability and report whether comparisons use matched steps or another explicit budget. Repeat promising configurations across seeds when feasible.

Measure mAP50, mAP50-95, per-class AP, precision and recall, localization, performance by object size and source where applicable, and inference cost. Select on validation; evaluate the selected configuration on the reserved test set. Report group-level support and uncertainty where meaningful; do not treat correlated frames as independent replicates. The final real-world pipeline must also be evaluated on representative underwater footage and physical mass-calibration experiments.

## 8. Output and acceptance criteria

Keep the v1 output `vision/data/public_datasets/detection_yolo/` unchanged. Write v2 to a separate directory (proposed `detection_yolo_v2/`) with explicit experiment configurations or export subdirectories. Persist configuration and split assignment in `vision/configs/`; store machine-readable audit summaries and documentation in `vision/docs/detection_training/`. Generated datasets and model weights remain outside version control according to project policy; verify ignore rules before building.

Before training, verify:

- Source inputs and v1 output are unchanged; no output path can overwrite them.
- Image identities, class mapping, paths and dimensions are valid; no dangling links, missing labels, filename collisions or silent drops.
- All exported boxes are finite, positive and inside the image within the documented numerical tolerance; parsing/export round trips preserve the contract.
- Every input image and annotation has a recorded outcome; quarantine and exclusions are counted and justified.
- No acquisition group or exact duplicate crosses the primary partitions; related-group and near-duplicate limitations are documented.
- All required classes and minority support are reported; any infeasible constraints block or explicitly qualify the experiment.
- Identical builds reproduce manifests and split assignments; all training configurations refer to the frozen primary evaluation split.
- Unit tests, integration tests and existing pipeline regression tests pass; a local read-only validation and training-loader smoke test confirm the exported dataset is usable.

## 9. Implementation sequence

1. Review and commit the audit and design together.
2. Implement and test pure geometry primitives, with no dataset mutation.
3. Implement and test manifest records and source parsers using synthetic fixtures.
4. Implement and verify grouped partition assignment.
5. Implement safe export and build coordination, including dry-run/report behavior.
6. Build v2 locally, inspect accepted/quarantined samples and validate the frozen splits.
7. Run controlled detection experiments. Advance to segmentation and tracking based on evidence while preserving the physical mass objective.

The v2 design is an incremental change to data preparation, not a replacement for the existing inference pipeline. It provides a path to more defensible evaluation without claiming that improved data alone will solve all detection or mass-estimation limitations.
