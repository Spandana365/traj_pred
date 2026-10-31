# Post-Run Research Audit: MOT17 YOLO + BoT-SORT Experiment

**Scope.** Verification of the saved outputs in `runs/mot17_tracking_analysis/` and the code in `src/trackvision/tracking/mot17/`. Nothing was rerun: no YOLO inference, BoT-SORT tracking, TrackEval or training. Figures were recomputed from saved files by a separate read-only script. This report is the only file the audit created.

**Status labels.**
- **VERIFIED:** confirmed directly from files, source code or recalculation. A verified problem is marked **VERIFIED (defect)**.
- **UNCERTAIN:** the evidence is partial or depends on an assumption.
- **UNRESOLVED:** the saved evidence cannot answer the question.

---

## Summary of findings

| # | Area | Status | Finding |
|---|---|---|---|
| 1 | HOTA decomposition | VERIFIED | Detection and association limit HOTA about equally (DetA 44.27, AssA 44.74). |
| 2 | Aggregation | VERIFIED | Combined metrics are TrackEval's own combined output; MOTA, IDF1 and all counts recompute exactly. |
| 3 | Detection thresholds | VERIFIED | 59.5% is recall at conf >= 0.25. The tracker's input reached 66.3%; all saved detections reached 70.1%. |
| 4 | Camera-motion compensation | **VERIFIED (defect)** | **GMC failed on all 5,316 frames. The saved results are effectively BoT-SORT without GMC.** |
| 5 | Error-profile counts | VERIFIED | TP, FP, FN, IDSW and Frag match TrackEval on all seven sequences. |
| 6 | Identity windows | VERIFIED | 43.1% and 20.8% recompute exactly, but they exclude short tracks and are dominated by MOT17-04. |
| 7 | Trajectory outputs | VERIFIED, with limitations | Format is correct. FP and distractor rows are not separated, and tracks carry no camera-motion correction. |
| 8 | Dataset and separation | VERIFIED | Seven sequences, 5,316 frames, FRCNN folders, no ETH/UCY dependency. |
| 9 | Prediction readiness | **Not ready** | Tracking must be rerun with GMC working first. The protocol is in section 9. |

### Effect on earlier claims

All tracker-derived outputs describe a tracker with GMC disabled: tracker files, TrackEval metrics, error profiles, trajectories and tracker figures. They are accurate measurements of what ran, but they must not be described as BoT-SORT with GMC. YOLO detections and detector recall are unaffected.

These statements in the previous run report need correcting:

| Previous statement | Correction |
|---|---|
| Motion compensation "stayed on" for all sequences | **False.** It was configured, but it failed on every frame. |
| Low scores are "mainly" due to the detector, "not" the tracker | True for MOTA only. HOTA and IDF1 show a comparable association problem. |
| "Detector limit": YOLO recall 59.5%, tracker about the same | Misleading. 59.5% covers conf >= 0.25 only; the tracker received boxes with 66.3% recall. |
| BoT-SORT created 1,383 track IDs | 1,434 raw IDs. 1,383 is the count after TrackEval preprocessing. |
| "About 100" pedestrians almost never tracked | 101 with coverage below 5%, of whom 87 have zero coverage. The term was never formally defined. |
| One identity covers 20.8% of 8 s windows | Arithmetic correct, but 372 of 546 pedestrians are outside that denominator. |
| Localization is good once matched | Conditional. It covers only boxes already matched at IoU >= 0.5. |

---

## 1. HOTA decomposition

**Status: VERIFIED.** Source: `trackeval/botsort/pedestrian_detailed.csv` (AUC columns) and `pedestrian_summary.txt`. `metrics.json` does not store DetRe, DetPr, AssRe or AssPr.

| Sequence | HOTA | DetA | AssA | LocA | DetRe | DetPr | AssRe | AssPr | IDF1 | Lower factor |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| MOT17-02 | 32.87 | 30.18 | 36.58 | 83.12 | 33.43 | 68.95 | 38.75 | 82.52 | 39.23 | Detection |
| MOT17-04 | 49.61 | 49.35 | 50.48 | 82.64 | 57.74 | 69.80 | 54.02 | 82.34 | 56.79 | Detection (marginal) |
| MOT17-05 | 37.36 | 39.58 | 35.82 | 76.30 | 48.63 | 57.19 | 43.01 | 61.52 | 46.89 | Association |
| MOT17-09 | 44.63 | 54.22 | 37.02 | 83.01 | 63.50 | 70.98 | 41.62 | 74.94 | 51.48 | Association |
| MOT17-10 | 36.14 | 45.67 | 28.99 | 78.95 | 50.94 | 69.25 | 32.79 | 69.52 | 43.43 | Association |
| MOT17-11 | 49.75 | 48.72 | 51.15 | 83.67 | 60.65 | 65.56 | 56.10 | 81.28 | 54.30 | Detection (marginal) |
| MOT17-13 | 40.60 | 38.02 | 43.83 | 80.21 | 40.34 | 74.37 | 47.81 | 74.89 | 53.43 | Detection |
| **Combined** | **44.16** | **44.27** | **44.74** | **81.80** | **51.09** | **68.73** | **48.70** | **79.15** | **51.34** | **About equal** |

### Interpretation

- **Neither factor dominates overall.** HOTA is the geometric mean of DetA and AssA at each threshold. On the combined averages, the two factors contribute almost equally.
- **The picture depends on the metric.** MOTA is dominated by false negatives: they cost 40.9 points, against 15.3 for FP and 0.7 for IDSW. MOTA barely penalises identity errors, so it understates the association problem.
- **Association problems are real.** IDTP is 50,253 against CLR_TP of 66,351, so 24.3% of matched boxes are not on the identity-consistent assignment. IDR is 44.75 against CLEAR recall of 59.09.
- **Tracks split more than they merge.** AssRe (48.70) is far below AssPr (79.15).
- **Detection misses more than it invents.** DetRe (51.09) is below DetPr (68.73).
- **The limiting factor changes with the IoU threshold.** AssA is the lower factor up to alpha 0.50 (DetA 0.506, AssA 0.494). DetA is lower from alpha 0.55 (0.474 against 0.486), and the gap widens: at alpha 0.80 DetA is 0.247 against AssA 0.386. Tighter box agreement is therefore a detection and localization limit.

### Uncertain

Association results come from a run without GMC, so they may understate what working GMC would achieve. The AssA pattern alone cannot show this. Among moving-camera sequences, MOT17-10 is lowest (28.99) but MOT17-11 is highest of all (51.15).

---

## 2. Aggregation

**Status: VERIFIED.**

- **Source.** `metrics.json` takes `COMBINED_SEQ` directly from TrackEval's results. Its HOTA, DetA, AssA, LocA, MOTA and IDF1 are identical to the `COMBINED` row in `pedestrian_detailed.csv`.
- **Method.** In the installed code, `eval.py` calls `metric.combine_sequences`. HOTA sums its per-threshold counts and weights AssA, AssRe, AssPr and LocA by `HOTA_TP`.
- **DetA check.** DetA recomputed from the summed `HOTA_TP`, `HOTA_FN` and `HOTA_FP` columns is 0.44267, matching the reported value.

### MOTA recalculation

MOTA = (TP - FP - IDSW) / (TP + FN), from `clear.py` line 178.

(66,351 - 17,134 - 799) / (66,351 + 45,946) = 48,418 / 112,297 = **0.43116**, which matches the reported value.

IDF1 = 2 x 50,253 / (2 x 50,253 + 62,044 + 33,232) = **0.51336**, which also matches.

### Count aggregation

Each combined value equals the sum of the seven sequences:

| Field | Sum | Field | Sum |
|---|---:|---|---:|
| CLR_TP | 66,351 | MT | 149 |
| CLR_FN | 45,946 | PT | 251 |
| CLR_FP | 17,134 | ML | 146 |
| IDSW | 799 | GT_Dets | 112,297 |
| Frag | 1,341 | Dets | 83,485 |
| IDTP | 50,253 | IDs | 1,383 |
| IDFN | 62,044 | GT_IDs | 546 |
| IDFP | 33,232 | | |

The combined values are not sequence averages: the unweighted means are MOTA 42.71 and HOTA 41.57.

`Dets` and `IDs` are after preprocessing. The raw tracker files hold 89,658 boxes and 1,434 IDs. The 6,173-box difference equals the boxes removed for matching distractor ground truth.

---

## 3. Detection threshold interpretation

**Status: VERIFIED.** The independent recalculation reproduced the reported 0.25 and 0.10 values exactly. Saved confidences are rounded to five decimals, and no row equals exactly 0.25.

### Thresholds in the code

| Stage | Threshold | Source |
|---|---|---|
| YOLO inference, saved to file | conf 0.05, NMS IoU 0.7 | `run_config.json`, `detection.py` |
| Tracker first stage | conf >= 0.25 | `track_high_thresh`, `byte_tracker.py` line 337 |
| Tracker second stage | 0.10 < conf < 0.25 | `track_low_thresh`, `byte_tracker.py` line 338 |
| Boxes ignored by tracker | conf <= 0.10 | Follows from the two rows above |
| New-track threshold | 0.25 | `new_track_thresh` |

The 178,500 saved detections divide into 92,038 first-stage, 39,334 second-stage and 47,128 unused boxes.

### Recall for each detection set

All rows use evaluated ground truth, distractor removal and one-to-one IoU >= 0.5 matching.

| Set | TP | FP | FN | Recall | Precision |
|---|---:|---:|---:|---:|---:|
| All saved YOLO boxes, conf >= 0.05 | 78,665 | 90,851 | 33,632 | 70.05% | 46.41% |
| Reported YOLO set, conf >= 0.10 | 74,479 | 49,046 | 37,818 | 66.32% | 60.29% |
| Actual tracker input, conf > 0.10 | 74,477 | 49,041 | 37,820 | 66.32% | 60.30% |
| **Reported YOLO set, conf >= 0.25** | **66,799** | **19,607** | **45,498** | **59.48%** | **77.31%** |
| BoT-SORT output | 66,351 | 17,134 | 45,946 | 59.09% | 79.48% |

### Clarification

- **59.5% is YOLO recall at conf >= 0.25.** It is not the detector's maximum recall, and it is not the recall of the tracker's full input.
- **The tracker output is not limited to 59.5%.** The tracker received boxes with 66.32% recall and output 59.09%, so 8,126 potential true positives were not kept.
- **The reported 0.10 row uses >= 0.10**, while the tracker uses > 0.10. The effect is 2 TP.
- **BoT-SORT recall is measured on Kalman-state boxes**, not on raw detections, so the two figures are not strictly the same kind of box.

### Occlusion and size

Tracker recall from `trajectories/*_ground_truth.csv`:

| GT visibility | Boxes | Recall | | GT box height | Boxes | Recall |
|---|---:|---:|---|---|---:|---:|
| >= 0.75 | 48,295 | 88.7% | | >= 200 px | 32,835 | 77.6% |
| 0.50-0.75 | 18,098 | 70.2% | | 100-200 px | 55,889 | 60.0% |
| 0.25-0.50 | 19,435 | 35.7% | | 50-100 px | 19,857 | 34.0% |
| < 0.25 | 26,469 | 14.7% | | < 50 px | 3,716 | 16.3% |

### Uncertain

About 73% of tracker false negatives may be boxes YOLO never found even at conf 0.05 (33,632 of 45,946). This assumes the missed sets are nested, which was not checked box by box.

---

## 4. Camera-motion compensation

**Status: VERIFIED (defect). GMC did not run on any frame.**

### Code path

1. `tracker.py` line 91 calls `tracker.update(batch, read_frame(frame))`. The original sequence frame **is** supplied to the tracker.
2. `byte_tracker.py` line 305 calls `_pre_first_associate(strack_pool, unconfirmed, img, results_high)`.
3. `byte_tracker.py` line 381 evaluates `self.gmc.apply(img, results_high.xyxy)`.
4. `DetectionFrame` in `tracker.py` defines `xywh`, `conf` and `cls`, but not `xyxy`. A direct `hasattr` check returned False.
5. The `AttributeError` is raised while the argument is evaluated, so **`gmc.apply` is never entered**.
6. Lines 382-384 catch the exception, log a warning and use an identity transform.

### Runtime evidence

`config/full_run_output.txt` contains exactly **5,316** copies of one message, equal to the total frame count:

```text
WARNING GMC failed, falling back to identity: 'DetectionFrame' object has no attribute 'xyxy'
```

### Conclusions

| Question | Status | Result |
|---|---|---|
| Were original frames passed to `tracker.update`? | VERIFIED | Yes |
| Did frames reach sparse optical flow in the tracker? | **VERIFIED (defect)** | **No, on any frame** |
| Is `gmc_method: sparseOptFlow` what ran? | **VERIFIED (defect)** | **No, it was requested only** |
| Does GMC work on these frames when called directly? | VERIFIED | Yes: `camera_motion.json` holds measurements from a separate instance |
| How much did the failure change the metrics? | **UNRESOLVED** | Needs a tracking rerun |

### Why it was missed

- The exception was caught and did not stop the run.
- The earlier report relied on the configuration and on the final lines of the output.
- `tests/unit/test_mot17_tracking.py` has no test that GMC is applied. The tracker tests use blank frames and only check identities and boxes.

### Misleading records

- `run_config.json` lists `sparseOptFlow` with no note of the failure.
- `camera_motion.json` says "Global motion compensation stays on for all sequences."
- `run_status.json` does not record the failure.

These files were left unchanged, as this audit required.

### Camera-motion evidence

`camera_motion.json` samples only the first 120 frames of each sequence. Median frame-to-frame translation:

| Static-looking | px | Moving | px |
|---|---:|---|---:|
| MOT17-02 | 0.12 | MOT17-05 | 1.57 |
| MOT17-04 | 0.14 | MOT17-10 | 6.58 |
| MOT17-09 | 0.26 | MOT17-11 | 4.91 |
| | | MOT17-13 | 7.68 |

**Uncertain:** these labels describe only those 120 frames.

### Recommended repair (not applied)

Add an `xyxy` property to `DetectionFrame`. Add a test using real image motion that fails if GMC falls back to identity, and fail the run on any fallback. Then rerun tracking, evaluation, analysis and visualization. Saved YOLO detections can be reused. Keep the current outputs as a labelled no-GMC run, and also save each frame's warp matrix.

---

## 5. Error-profile calculations

**Status: VERIFIED.**

### Agreement with TrackEval

`error_profile.json` shows zero difference from TrackEval for TP, FP, FN, IDSW and Frag on all seven sequences. A third, independent evaluator written for this audit also reproduced TP, FP, FN and IDSW for all seven. Frag was not re-derived in that evaluator.

| Sequence | TP | FP | FN | IDSW | Frag |
|---|---:|---:|---:|---:|---:|
| MOT17-02 | 7,321 | 1,688 | 11,260 | 106 | 188 |
| MOT17-04 | 30,839 | 8,502 | 16,718 | 143 | 306 |
| MOT17-05 | 3,950 | 1,932 | 2,967 | 152 | 220 |
| MOT17-09 | 3,943 | 821 | 1,382 | 56 | 65 |
| MOT17-10 | 8,083 | 1,362 | 4,756 | 223 | 362 |
| MOT17-11 | 6,319 | 2,411 | 3,117 | 28 | 48 |
| MOT17-13 | 5,896 | 418 | 5,746 | 91 | 152 |

The saved profile is from the final `analyze visualize` run in `commands.log`, made after a matching error was corrected: identity priority now lasts only one frame, as in TrackEval. The first analysis run did not match TrackEval exactly.

### Matching rules

- **Evaluated ground truth:** `conf == 1` and `class == 1`.
- **Distractors:** predictions matched at IoU >= 0.5 to classes 2, 7, 8 or 12 are removed first. This removed 6,173 tracker boxes.
- **Assignment:** per-frame one-to-one Hungarian matching, IoU >= 0.5.
- **ID handling:** a pair gets a +1000 bonus only if matched in the immediately preceding frame.
- **ID switch:** the matched tracker ID differs from the last ID matched to that pedestrian, even across a gap.
- **Fragmentation:** a matched, then unmatched, then matched pattern inside a ground-truth track.

### Localization

Calculated from matched pairs only, over 66,351 boxes, in image pixels:

| Measure | Median | P95 |
|---|---:|---:|
| IoU | 0.831 | P5 = 0.556 |
| Centre error (px) | 4.89 | 33.38 |
| Centre error / GT height | 0.027 | 0.155 |

Limitations:

- IoU is truncated at 0.5 by construction.
- Missed and false boxes contribute nothing.
- Tracker boxes are Kalman-state estimates.
- Pixel errors are not comparable across resolutions: MOT17-05 is 640x480.

### Track availability

Availability is the share of ground-truth windows covered by one unbroken tracker identity. Section 6 gives the exact calculation.

### Almost-never-tracked pedestrians

**Status: the term was not formally defined before this audit.** The earlier "about 100" was read from the first bin of a histogram. Coverage is matched frames divided by evaluated ground-truth frames.

| Definition | Pedestrians (of 546) |
|---|---:|
| Coverage = 0 | 87 |
| Coverage < 5% (proposed definition) | **101** |
| Coverage < 10% | 115 |
| Coverage < 20% (TrackEval ML) | 146 |
| Coverage > 80% (TrackEval MT) | 149 |

The 101 pedestrians hold 6.5% of ground-truth boxes, and their median track length is 43 frames. They are mostly short-lived, so this should not be read as 18.5% of pedestrian-time being lost. Median coverage is 56.7%; frame-weighted coverage is 59.09%, which equals recall.

---

## 6. Identity continuity

**Status: VERIFIED.** Reported values recompute exactly, both by run length and by direct window enumeration.

### Calculation

Source: `error_profile.window_availability`.

For each evaluated pedestrian with T ground-truth frames and a window of W = `round(seconds x frame rate)` frames:

- **Denominator:** `sum of max(0, T - W + 1)`, meaning every window position at a stride of one frame.
- **Numerator:** for each unbroken run of length R with one tracker ID, `sum of max(0, R - W + 1)`.
- **Pooled result:** total numerator divided by total denominator, not a mean of sequence percentages.

A run ends at any frame with no match or with a different ID. Reacquiring the same ID after a one-frame miss starts a new run.

### Results

| Sequence | W (1 s) | Available / possible | 1 s | W (8 s) | Available / possible | 8 s | IDs too short for 8 s |
|---|---:|---:|---:|---:|---:|---:|---:|
| MOT17-02 | 30 | 3,896 / 16,913 | 23.0% | 240 | 241 / 7,409 | 3.3% | 23 / 62 |
| MOT17-04 | 30 | 24,341 / 45,159 | 53.9% | 240 | 8,228 / 30,102 | 27.3% | 21 / 83 |
| MOT17-05 | 14 | 1,617 / 5,383 | 30.0% | 112 | 97 / 1,648 | 5.9% | 117 / 133 |
| MOT17-09 | 30 | 2,284 / 4,595 | 49.7% | 240 | 0 / 971 | 0.0% | 14 / 26 |
| MOT17-10 | 30 | 3,684 / 11,216 | 32.8% | 240 | 444 / 3,820 | 11.6% | 33 / 57 |
| MOT17-11 | 30 | 4,427 / 7,492 | 59.1% | 240 | 795 / 2,245 | 35.4% | 66 / 75 |
| MOT17-13 | 25 | 2,835 / 9,178 | 30.9% | 200 | 0 / 974 | 0.0% | 98 / 110 |
| **Pooled** | | **43,084 / 99,936** | **43.1%** | | **9,805 / 47,169** | **20.8%** | **372 / 546** |

### Do the windows include all ground-truth-eligible windows?

**It depends on what "eligible" means.**

- **Included:** every contiguous window that fits inside an evaluated pedestrian's track. No visibility, size or tracker-based filter is applied. Evaluated tracks had no internal frame gaps, so a strict contiguity check gave identical numbers.
- **Excluded:** pedestrians shorter than the window. That is 77 of 546 at 1 s and **372 of 546 at 8 s**.
- **Excluded:** all non-evaluated ground truth, such as class 7 static persons and distractors.

### Interpretation cautions

- Windows overlap heavily and are not independent.
- Long tracks count more. MOT17-04 supplies 63.8% of the 8 s denominator and 83.9% of the numerator.
- The unweighted sequence means are 39.9% at 1 s and 11.9% at 8 s, against pooled 43.1% and 20.8%.
- The measure is strict about tracker continuity. A prediction method that tolerates short gaps could use more windows than this implies.
- The 8 s window is 240 frames at 30 fps. Twenty samples at 2.5 Hz span 7.6 s, so it is close to but not the same as an 8 + 12 prediction window.
- All figures are from the no-GMC run and may change.

---

## 7. Trajectory outputs

**Status: VERIFIED, with the limitations listed below.** Files: `trajectories/<sequence>_ground_truth.csv`, `<sequence>_botsort.csv` and `sequences.json`.

### Formats

**Ground truth:** `sequence, frame, time_s, gt_id, left, top, width, height, center_x, center_y, foot_x, foot_y, visibility, botsort_track_id, tracked`

**BoT-SORT:** `sequence, frame, time_s, track_id, left, top, width, height, confidence, center_x, center_y, foot_x, foot_y, missing_frames_before, matched_gt_id, matched_iou`

### Checks

| Item | Status | Result |
|---|---|---|
| Coordinates | VERIFIED | MOTChallenge 1-based image pixels, stated in `sequences.json` |
| Units | VERIFIED | Pixels. No ground-plane, metric or camera-motion correction |
| Frame rates | VERIFIED | 30, 30, 14, 30, 30, 30 and 25 fps for 02, 04, 05, 09, 10, 11 and 13 |
| Frame IDs | VERIFIED | Original MOT17 frame numbers; all within 1 to sequence length |
| Time | VERIFIED | `time_s = frame / fps`, so frame 1 is 1/fps, not 0 |
| Foot point | VERIFIED | `foot_x = left + width / 2`, `foot_y = top + height`; maximum error 0.0 on every row |
| Row counts | VERIFIED | GT rows equal evaluated boxes; BoT-SORT rows equal tracker-file rows |
| `tracked` flag | VERIFIED | Sum equals TP in each sequence; zero conflicts with `botsort_track_id` |
| `missing_frames_before` | VERIFIED | Recomputed from each track's frames with no conflicts |
| `sequences.json` | VERIFIED | Lengths, sizes and frame rates agree with `seqinfo.ini` |

### Missing frames

- **Ground truth:** every evaluated frame has a row. An unmatched frame has `tracked = 0` and a blank `botsort_track_id`.
- **BoT-SORT:** a missing frame has no row. The next row of that track states the gap in `missing_frames_before`.
- **Limit:** the end of a track is not flagged. A track that stops simply has no more rows.

### ID semantics

- `gt_id` comes from `gt.txt` and is unique only within a sequence.
- `track_id` is unique only within a sequence and restarts for each one. **Use `(sequence, id)` as the key.**
- Track IDs are not contiguous: MOT17-10 has 260 IDs but a maximum of 1,718. **Uncertain:** this probably reflects tentative tracks that were numbered but never output.
- `botsort_track_id` and `matched_gt_id` are frame-level evaluation matches. They are not permanent identity links.

### Limitations

- **Unmatched BoT-SORT rows combine two cases.** A blank `matched_gt_id` covers 23,307 rows: 17,134 false positives and 6,173 distractor-matched boxes. Rows do not say which is which.
- **GT and tracker boxes differ in kind.** GT boxes are annotations; BoT-SORT boxes are Kalman-state estimates.
- **Moving-camera trajectories contain camera motion.** Sequences 05, 10, 11 and 13 are not stabilised. No warp matrices were saved.
- **Non-evaluated ground truth is not in these CSVs.**
- **UNRESOLVED:** whether any foot points fall outside the image was not measured.

---

## 8. Dataset and separation

**Status: VERIFIED.**

| Sequence | seqLength | Images | Numbered | Frames detected | Frame rate | Size | GT copy matches |
|---|---:|---:|---|---:|---:|---|---|
| MOT17-02 | 600 | 600 | 000001-000600 | 600 | 30 | 1920x1080 | Yes |
| MOT17-04 | 1,050 | 1,050 | 000001-001050 | 1,050 | 30 | 1920x1080 | Yes |
| MOT17-05 | 837 | 837 | 000001-000837 | 837 | 14 | 640x480 | Yes |
| MOT17-09 | 525 | 525 | 000001-000525 | 525 | 30 | 1920x1080 | Yes |
| MOT17-10 | 654 | 654 | 000001-000654 | 654 | 30 | 1920x1080 | Yes |
| MOT17-11 | 900 | 900 | 000001-000900 | 900 | 30 | 1920x1080 | Yes |
| MOT17-13 | 750 | 750 | 000001-000750 | 750 | 25 | 1920x1080 | Yes |
| **Total** | **5,316** | **5,316** | | **5,316** | | | |

- **Canonical folders:** `load_sequence` always uses `train/<sequence>-FRCNN`, and `dataset_inventory.json` records FRCNN.
- **Variants:** the inventory reports `variants_agree: true` for all seven, using `gt.txt` and three sample frames per sequence. It does not compare every image.
- **Public detections:** `det/det.txt` is not read. All detections come from YOLO.
- **Test split:** no code reads `data/MOT17/test`.
- **Staged ground truth:** the `gt.txt` copies in `trackeval/gt_staging/` match the FRCNN originals by SHA-256.
- **Code separation:** the package imports only `trackvision.tracking.mot17` from within TrackVision.
- **Path guards:** `ensure_dataset_root` and `ensure_output_root` reject ETH, UCY, prediction and error-injection path segments.
- **Output separation:** MOT17 outputs are under `runs/mot17_tracking_analysis/`. `runs/` has only that folder plus the existing `audits`, `error_injection` and `prediction`.
- **Config references:** no JSON file in `config/` mentions `data/eth`, `data/train`, `data/test` or `error_injection`.
- **ETH integrity:** size and modification time of files under the three ETH result folders were unchanged during this audit, and earlier checks found no changes to them or to `data/eth` since before the run.

### Uncertain

- ETH integrity rests on modification times. No SHA-256 baseline of those files existed before the experiment.
- `data/train` and `data/test` were not inspected, in keeping with the earlier instruction not to access them. The MOT17 code does not reference them.
- `DetectorConfig.weights` is a fixed relative path and is not checked by the path guards. It resolves correctly when the pipeline is started from the project root.

---

## 9. Prediction readiness

**Status: not ready.** Fix GMC and rerun tracking first, then freeze the outputs and this protocol. The split below uses camera type and ground-truth counts only. Decide it before any prediction results exist.

Do not use the ETH-trained LSTM, and do not feed MOT17 coordinates into it.

### Window availability

Counted on saved ground truth, for 8 observed + 12 future samples at 2.5 Hz. These counts come from the no-GMC run and use an anchor stride of one sample.

| Sequence | Camera | Clean windows | Pedestrians | Tracked at anchor | Same ID for all 8 history samples |
|---|---|---:|---:|---:|---:|
| MOT17-02 | Static-looking | 669 | 40 | 258 | 124 |
| MOT17-04 | Static-looking | 2,595 | 63 | 1,828 | 1,251 |
| MOT17-05 | Moving | 295 | 17 | 178 | 58 |
| MOT17-09 | Static-looking | 96 | 12 | 62 | 13 |
| MOT17-10 | Moving | 352 | 24 | 233 | 64 |
| MOT17-11 | Moving | 199 | 9 | 167 | 127 |
| MOT17-13 | Moving | 115 | 13 | 71 | 19 |
| **Total** | | **4,321** | **178** | **2,797 (64.7%)** | **1,656 (38.3%)** |

Keeping only the clean last column would silently discard 61.7% of windows, and that subset would be the easiest cases.

### Recommended split

**Primary evaluation: leave-one-sequence-out, seven folds.** With only seven videos, one fixed split depends heavily on which sequences are held out. Pick validation from the training sequences in each fold, and report each held-out sequence separately.

**Fixed split, for development and a single headline result:**

| Part | Sequences | Clean windows | Camera mix | Frame rates |
|---|---|---:|---|---|
| Train | 04, 05, 13 | 3,005 (69.5%) | Static and moving | 30, 14, 25 |
| Validation | 09, 11 | 295 (6.8%) | Static and moving | 30 |
| Test | 02, 10 | 1,021 (23.6%) | Static and moving | 30 |

Rules:

- Never split one sequence across parts.
- Use balanced sampling, because MOT17-04 supplies 60% of all windows.
- Fit normalisation on training sequences only.
- Do not tune the tracker on these sequences.

### Paired protocol

**1. Define windows from ground truth alone.** A window is `(sequence, gt_id, anchor frame)`. It is eligible when every required evaluated ground-truth sample exists. Tracker results must not influence eligibility.

**2. Evaluate two inputs on the same windows, using the same frozen model weights.**

| | Clean arm | BoT-SORT arm |
|---|---|---|
| History | Ground truth for `gt_id` | Rows of the track matched to `gt_id` at the anchor |
| Selected by | `gt_id` | `track_id` at frames <= anchor |
| **Target** | **GT future of `gt_id`** | **Same GT future of `gt_id`** |

**3. Protect the targets.** Future positions come only from `*_ground_truth.csv`. Tracker positions are never targets. Record the SHA-256 of the target table before evaluation. GT matching selects the window but is never given to the model.

**4. Stay causal.** BoT-SORT input may use only frames up to the anchor.

**5. Include a constant-velocity baseline in both arms.** It shows how much of any change comes from the input alone.

### Every window gets one outcome

| Category | Condition | Handling |
|---|---|---|
| A. Complete | All 8 samples present, one ID, all matched to `gt_id` | Scored normally |
| B. Gaps | Track at anchor, some history samples missing | Scored, with an observation mask and time deltas |
| C. Identity contamination | Any history row matched to another pedestrian | Scored without repair |
| D. Short history | Too few samples, usually a new track or fragment | Scored by a declared short-history rule |
| E. No track at anchor | Pedestrian missed at the anchor | No model input exists; counted, never dropped |

A window with both gaps and contamination goes in category C. A false-positive track has no target, so report its count separately.

### Handling each failure

- **Missing tracks.** Report coverage as (windows with a prediction) / (all clean windows). Report the category E rate. Give two all-window summaries: coverage plus metrics on covered windows, and a sensitivity result that assigns a declared penalty, such as a capped error or the last-known-position fallback.
- **ID switches.** Do not repair them using ground truth, because a deployed system could not. The model receives the contaminated history. Record `history_purity`, the share of history matched to `gt_id`.
- **Fragmented trajectories.** Use only the current `track_id` in the main analysis. Do not stitch fragments using ground truth. Automatic stitching may be tested as a separate labelled arm.
- **Gaps.** Do not interpolate silently. Pass a mask and time deltas, or declare the imputation and apply it the same way to all gap windows.

### Metrics

Report ADE and FDE in pixels and normalised by anchor box height, for both arms, in these groups:

- all covered windows;
- each category A to E;
- each sequence;
- static-looking and moving cameras.

The main effect is the paired difference, BoT-SORT error minus clean error, on the same windows.

Windows overlap, so do not treat them as independent. Use a bootstrap clustered by pedestrian within sequence, and give sequence-level results. With seven sequences, sequence-level significance is limited, so state effect sizes.

### Preparation before training

- **Camera motion.** Moving-camera sequences need history and future expressed in a consistent frame. Save warp matrices in the tracking rerun, as none exist now.
- **Time base.** MOT17-05 needs 5.6 frames per 2.5 Hz sample, so it requires a declared rounding or interpolation rule. Report the timing error it adds.
- **Scale.** MOT17-05 is 640x480, while the rest are 1920x1080. Normalise by image height or box height.
- **Sparse sequences.** Sequences 09, 11 and 13 have few long windows. Consider also running a shorter horizon, declared in advance.

---

## Issue register

| ID | Issue | Status | Action |
|---|---|---|---|
| A1 | GMC failed on all 5,316 frames | **VERIFIED (defect)** | Fix adapter, add test, rerun from tracking |
| A2 | `run_config.json` and `camera_motion.json` imply GMC ran | VERIFIED (defect) | Correct on rerun; label current run as no-GMC |
| A3 | Previous report's claims in the corrections table | VERIFIED (defect) | Use this audit's wording |
| A4 | Effect of GMC on HOTA, IDF1 and IDSW | **UNRESOLVED** | Needs tracking rerun |
| A5 | No test detects GMC fallback | VERIFIED (defect) | Add test |
| A6 | False positives and distractor rows are not separated in trajectories | VERIFIED (limitation) | Add a label on rerun |
| A7 | Warp matrices not saved | VERIFIED (limitation) | Save on rerun |
| A8 | Almost-never-tracked was undefined | VERIFIED, resolved here | Use coverage < 5%, count 101 |
| A9 | Availability excludes short tracks and is MOT17-04 heavy | VERIFIED (limitation) | Report denominators and per-sequence values |
| A10 | Share of tracker FN that YOLO never detected | UNCERTAIN | Compare box by box |
| A11 | Static or moving label beyond the first 120 frames | UNCERTAIN | Measure full sequences |
| A12 | Reason for non-contiguous track IDs | UNCERTAIN | Confirm in Ultralytics source |
| A13 | Foot points outside the image | UNRESOLVED | Measure |
| A14 | ETH integrity has no pre-run hash baseline | UNCERTAIN | Keep a hash manifest in future |
| A15 | Frames 1-10 of MOT17-02 used an earlier code version that passed `half=False` explicitly | UNCERTAIN, low risk | `half` was off in both versions; no action needed |
