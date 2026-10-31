# GMC Ablation Comparison: MOT17 BoT-SORT

**Experiments compared**

| Label | Location | Meaning |
|---|---|---|
| `BoT-SORT_no_GMC` | `runs/mot17_tracking_analysis/` | Archived run. GMC failed on all 5,316 frames and fell back to identity. |
| `BoT-SORT_GMC_enabled` | `runs/mot17_tracking_analysis/gmc_fixed/` | Corrected run. GMC operated on all 5,316 frames. |

The metrics of the two experiments are kept separate and are never pooled.

**Scope.** Both experiments use the same saved YOLO detections, the same seven MOT17 training sequences, the same ground truth and the same BoT-SORT settings. No YOLO inference, training or fine-tuning was performed. Prediction training has not started.

---

## 1. Main results

| Combined TrackEval metric | `BoT-SORT_no_GMC` | `BoT-SORT_GMC_enabled` | Change |
|---|---:|---:|---:|
| HOTA | 44.16 | **45.35** | +1.19 |
| DetA | 44.27 | **44.76** | +0.49 |
| AssA | 44.74 | **46.60** | +1.86 |
| LocA | 81.80 | 81.76 | -0.04 |
| MOTA | 43.12 | 43.20 | +0.08 |
| IDF1 | 51.34 | **53.36** | +2.02 |
| ID switches | 799 | **643** | -156 (-19.5%) |
| Fragmentation | 1,341 | 1,383 | +42 (+3.1%) |
| TP | 66,351 | 67,617 | +1,266 |
| FP | 17,134 | 18,458 | +1,324 |
| FN | 45,946 | 44,680 | -1,266 |

**Findings**

- GMC improved identity handling, mainly on moving-camera sequences. The largest gain is MOT17-10: HOTA +7.82, AssA +11.35, ID switches halved.
- **The combined benefit is modest and uneven.** MOTA is essentially flat because MOT17-11 fell by 6.69 points, offsetting gains elsewhere.
- **MOT17-11 got worse on every headline metric.** 662 extra false positives account for most of the change.
- **False positives and fragmentation both rose.** GMC is not a pure improvement.
- **Longer identity windows did not improve.** 8 s availability fell slightly, from 20.79% to 20.69%.
- **On static cameras, GMC adds small false motion.** It barely changes tracking there, but stabilized coordinates are worse than raw pixels on those sequences.

---

## 2. Attribution control

**Result: the differences between the two experiments come from GMC, not from other code changes.**

With GMC explicitly switched off, the corrected code reproduced all seven archived tracker files **byte for byte** (`gmc_fixed/control_no_gmc/control_result.json`). Saved detections and tracker settings are also unchanged, and tracking is deterministic: the pilot and full runs gave identical tracker and warp files for MOT17-04 and MOT17-10.

So each metric change below is a real, reproducible effect of enabling GMC for this input.

**That does not mean each change is a general property of GMC.** Association is sensitive to small perturbations, there are only seven sequences, and no significance test is meaningful at that sample size.

---

## 3. Per-sequence changes

Values are `BoT-SORT_no_GMC` -> `BoT-SORT_GMC_enabled`, with the change in brackets.

### Static-looking cameras

| Metric | MOT17-02 | MOT17-04 | MOT17-09 |
|---|---:|---:|---:|
| HOTA | 32.87 -> 32.84 (-0.03) | 49.61 -> 49.73 (+0.12) | 44.63 -> 45.50 (+0.87) |
| DetA | 30.18 -> 30.07 (-0.11) | 49.35 -> 49.38 (+0.03) | 54.22 -> 54.23 (+0.01) |
| AssA | 36.58 -> 36.65 (+0.07) | 50.48 -> 50.71 (+0.23) | 37.02 -> 38.46 (+1.44) |
| LocA | 83.12 -> 83.14 | 82.64 -> 82.64 | 83.01 -> 83.01 |
| MOTA | 29.75 -> 29.48 (-0.27) | 46.67 -> 46.60 (-0.07) | 57.58 -> 57.67 (+0.09) |
| IDF1 | 39.23 -> 39.19 (-0.04) | 56.79 -> 57.56 (+0.77) | 51.48 -> 53.71 (+2.23) |
| ID switches | 106 -> 104 | 143 -> 144 | 56 -> 54 |
| Fragmentation | 188 -> 188 | 306 -> 305 | 65 -> 66 |
| TP | 7,321 -> 7,300 | 30,839 -> 30,837 | 3,943 -> 3,944 |
| FP | 1,688 -> 1,718 | 8,502 -> 8,530 | 821 -> 819 |
| FN | 11,260 -> 11,281 | 16,718 -> 16,720 | 1,382 -> 1,381 |

### Moving cameras

| Metric | MOT17-05 | MOT17-10 | MOT17-11 | MOT17-13 |
|---|---:|---:|---:|---:|
| HOTA | 37.36 -> 39.31 (+1.95) | 36.14 -> 43.96 (+7.82) | 49.75 -> 48.54 (**-1.21**) | 40.60 -> 43.78 (+3.18) |
| DetA | 39.58 -> 40.11 (+0.53) | 45.67 -> 48.24 (+2.57) | 48.72 -> 46.93 (**-1.79**) | 38.02 -> 41.62 (+3.60) |
| AssA | 35.82 -> 39.41 (+3.59) | 28.99 -> 40.34 (+11.35) | 51.15 -> 50.57 (**-0.58**) | 43.83 -> 46.55 (+2.72) |
| LocA | 76.30 -> 76.41 | 78.95 -> 78.85 | 83.67 -> 83.71 | 80.21 -> 80.30 |
| MOTA | 26.98 -> 27.38 (+0.40) | 50.61 -> 53.47 (+2.86) | 41.12 -> 34.43 (**-6.69**) | 46.27 -> 49.79 (+3.52) |
| IDF1 | 46.89 -> 47.86 (+0.97) | 43.43 -> 55.88 (+12.45) | 54.30 -> 51.14 (**-3.16**) | 53.43 -> 57.53 (+4.10) |
| ID switches | 152 -> 131 | 223 -> 112 | 28 -> 32 | 91 -> 66 |
| Fragmentation | 220 -> 231 | 362 -> 365 | 48 -> 51 | 152 -> 177 |
| TP | 3,950 -> 4,067 | 8,083 -> 8,638 | 6,319 -> 6,354 | 5,896 -> 6,477 |
| FP | 1,932 -> 2,042 | 1,362 -> 1,661 | 2,411 -> 3,073 | 418 -> 615 |
| FN | 2,967 -> 2,850 | 4,756 -> 4,201 | 3,117 -> 3,082 | 5,746 -> 5,165 |

### Grouped by camera type

These are descriptive sums for one experiment at a time; they are not TrackEval output.

| Group | ID switches | Fragmentation | Mean HOTA change |
|---|---:|---:|---:|
| Static-looking (3 sequences) | 305 -> 302 | 559 -> 559 | +0.32 |
| Moving (4 sequences) | 494 -> 341 (-31.0%) | 782 -> 824 | +2.94 |

---

## 4. Detection and association changes

| Combined | `BoT-SORT_no_GMC` | `BoT-SORT_GMC_enabled` | Change |
|---|---:|---:|---:|
| DetA | 44.27 | 44.76 | +0.49 |
| DetRe | 51.09 | 52.14 | +1.05 |
| DetPr | 68.72 | 68.02 | -0.70 |
| AssA | 44.74 | 46.60 | +1.86 |
| AssRe | 48.70 | 50.79 | +2.09 |
| AssPr | 79.15 | 79.08 | -0.07 |

- **Association gained more than detection.** AssA rose +1.86, against +0.49 for DetA.
- **The gain is in AssRe.** Fewer tracks were split, while AssPr barely moved.
- **Detection trades precision for recall.** The detections did not change, so these DetRe and DetPr shifts come from which boxes the tracker kept. More tracks survive camera motion, which adds both true and false boxes.
- **Detection and association still limit the score about equally.** DetA 44.76 and AssA 46.60 remain close, so GMC does not change the main conclusion of the earlier audit.

### Why MOTA hardly moved

MOTA gives little weight to identity. The 156 fewer ID switches are worth only 0.14 points of 112,297 ground-truth boxes, and the 1,324 extra FP nearly cancel the 1,266 fewer FN. IDF1 and HOTA are better measures of what GMC changed.

---

## 5. MOT17-11: where GMC made results worse

| Item | Value |
|---|---|
| Tracker output boxes | 8,954 -> 9,658 (+704) |
| TP | +35 |
| FP | +662 |
| Median track length | 11 -> 15 frames |
| Track IDs after TrackEval preprocessing | 230 -> 220 |

Nearly all the extra output boxes are false positives. The warps themselves look sound on this sequence: phase-correlation agreement is 0.985 and 0.972, and photometric error improved on 99.9% of frame pairs.

**Likely explanation (not proven).** With GMC, tracks survive camera motion better, including tracks following boxes that do not match evaluated pedestrians. Some false tracks hold still once stabilized: track 186 is false on all 114 of its frames and travels 265 px raw but 23 px stabilized. That suits a fixed object that YOLO repeatedly detects.

**Unresolved.**

- Whether these boxes are real non-pedestrian objects or unannotated people was not checked by hand.
- The share of the 662 extra FP caused by this mechanism was not measured.

MOT17-13 and MOT17-10 show similar FP rises (+197, +299), but there TP rose far more (+581, +555). MOT17-11 was already the strongest sequence for ID switches (28), so it had little to gain.

---

## 6. ID switches and fragmentation

- **ID switches fell 19.5%**, and the reduction is almost entirely on moving cameras: 494 -> 341, with static sequences 305 -> 302.
- **MOT17-10 alone accounts for 111 of the 156 fewer switches.**
- **Fragmentation rose 3.1%.** It increased on MOT17-05 (+11) and MOT17-13 (+25) and was nearly flat elsewhere.
- **Rising fragmentation does not by itself mean tracking got worse.** TrackEval counts a fragment only for a track that has been matched, and more pedestrians are now matched. The share of the rise due to that effect was not measured.
- **Fewer track IDs overall:** 1,434 -> 1,325 raw, and 1,383 -> 1,281 after preprocessing.

---

## 7. Localization changes

Calculated on matched boxes only, so an IoU below 0.5 cannot appear.

| Combined | `BoT-SORT_no_GMC` | `BoT-SORT_GMC_enabled` |
|---|---:|---:|
| Median IoU | 0.8310 | 0.8307 |
| Median centre error (px) | 4.89 | 4.79 |
| P95 centre error (px) | 33.38 | 33.07 |
| Median centre error / height | 0.0268 | 0.0268 |

Localization is essentially unchanged, as LocA also shows. Small centre-error gains appear on moving sequences (for example MOT17-13: 4.17 -> 3.97 px). This is expected: output boxes are Kalman estimates updated with the same detections. Because matched sets differ between runs, these are not paired box-by-box comparisons.

---

## 8. Track availability and continuity

| Combined | `BoT-SORT_no_GMC` | `BoT-SORT_GMC_enabled` |
|---|---:|---:|
| 1 s unbroken-identity availability | 43.11% (43,084 / 99,936) | 43.94% (43,913 / 99,936) |
| 3 s | 31.82% (25,299 / 79,516) | 32.42% (25,783 / 79,516) |
| 8 s | 20.79% (9,805 / 47,169) | **20.69%** (9,757 / 47,169) |
| Median per-pedestrian coverage | 56.7% | 60.7% |
| Pedestrians with zero coverage | 87 | 80 |
| Pedestrians with coverage < 5% | 101 | 92 |
| Mostly tracked / mostly lost (TrackEval) | 149 / 146 | 161 / 138 |
| Interior gaps | 1,341 | 1,383 |
| Median interior gap (s) | 0.200 | 0.167 |
| P95 interior gap (s) | 2.17 | 2.13 |

Denominators are identical between experiments because they depend only on ground truth.

### Availability by sequence

| Sequence | 1 s | 3 s | 8 s |
|---|---:|---:|---:|
| MOT17-02 | 23.04 -> 23.02 | 11.85 -> 11.86 | 3.25 -> 3.25 |
| MOT17-04 | 53.90 -> 53.85 | 43.17 -> 43.04 | 27.33 -> 27.34 |
| MOT17-05 | 30.04 -> 31.53 | 16.53 -> 17.35 | 5.89 -> 5.10 |
| MOT17-09 | 49.71 -> 50.29 | 25.36 -> 27.75 | 0.00 -> 0.00 |
| MOT17-10 | 32.85 -> 37.26 | 21.98 -> 25.23 | 11.62 -> 11.18 |
| MOT17-11 | 59.09 -> 58.56 | 46.45 -> 46.04 | 35.41 -> 34.57 |
| MOT17-13 | 30.89 -> 34.06 | 12.80 -> 16.22 | 0.00 -> 0.00 |

**Interpretation.** GMC mainly helps short continuity, for example MOT17-10's median track span of 0.6 s -> 1.4 s. It did not produce longer unbroken identities: 8 s availability fell on MOT17-05, 10 and 11. This matters for prediction, because long windows remain scarce.

The 8 s result still excludes pedestrians too short for the window. It is pooled across heavily overlapping windows and dominated by MOT17-04.

---

## 9. GMC diagnostics and call counts

Source: `gmc_fixed/gmc/diagnostics.json`, `gmc/<sequence>_warps.csv` and `logs/full_run.txt`.

| Sequence | Frames | Calls | Successful | Failed | Estimated warps | First-frame identity | Other identity | Median translation (px) | Max translation (px) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MOT17-02 | 600 | 600 | 600 | 0 | 599 | 1 | 0 | 0.16 | 0.82 |
| MOT17-04 | 1,050 | 1,050 | 1,050 | 0 | 1,049 | 1 | 0 | 0.20 | 0.67 |
| MOT17-05 | 837 | 837 | 837 | 0 | 836 | 1 | 0 | 4.19 | 41.94 |
| MOT17-09 | 525 | 525 | 525 | 0 | 524 | 1 | 0 | 0.29 | 3.08 |
| MOT17-10 | 654 | 654 | 654 | 0 | 653 | 1 | 0 | 4.12 | 31.42 |
| MOT17-11 | 900 | 900 | 900 | 0 | 899 | 1 | 0 | 4.30 | 17.14 |
| MOT17-13 | 750 | 750 | 750 | 0 | 749 | 1 | 0 | 7.73 | 53.59 |
| **Total** | **5,316** | **5,316** | **5,316** | **0** | **5,309** | **7** | **0** | | |

- **All warps are finite** and well-formed, and each warp file has one row per frame in order.
- **The seven identity warps are expected.** A sequence's first frame has no previous frame.
- **There were no hidden fallbacks.** Ultralytics logged zero "GMC failed", "not enough matching points" or "affine estimation failed" messages, by both the in-process counter and a log search.
- **The GMC code also exposes three silent identity paths.** They were counted, not assumed absent, and none occurred.
- **Boxes reach GMC, but sparse optical flow does not use them.** Pedestrians are therefore not masked out of motion estimation.
- **Real data had no empty frames at conf >= 0.25.** The empty-frame path is covered by unit tests only.

### Independent checks on warp quality

"Agreement" compares warp motion at the image centre with phase correlation, which does not use GMC. "Improved" is the share of frame pairs where the warp reduced photometric error.

| Sequence | Warp motion (px) | Phase motion (px) | Correlation x / y | Improved | Static objects, raw -> stabilized spread (px) |
|---|---:|---:|---:|---:|---|
| MOT17-02 | 0.13 | 0.45 | 0.48 / 0.29 | 16.2% | 0.8 -> 4.6 (13 objects) |
| MOT17-04 | 0.10 | 0.21 | 0.57 / 0.36 | 29.6% | 0.0 -> 22.2 (45 objects) |
| MOT17-09 | 0.30 | 1.48 | 0.48 / 0.06 | 75.0% | 0.0 -> 50.6 (3 objects) |
| MOT17-05 | 3.31 | 3.30 | 0.85 / 0.94 | 91.5% | 38.5 -> 44.4 (1 object) |
| MOT17-10 | 3.29 | 3.24 | 0.99 / 0.97 | 99.7% | 158.3 -> 29.5 (12 objects) |
| MOT17-11 | 3.71 | 3.60 | 0.99 / 0.97 | 99.9% | 72.4 -> 41.7 (3 objects) |
| MOT17-13 | 4.66 | 4.44 | 1.00 / 0.92 | 100.0% | 106.6 -> 6.9 (27 objects) |

Also, for the first 120 frames of MOT17-04 and MOT17-10, warp translation statistics match the separate earlier measurements in `camera_motion.json` to four decimals.

**Conclusions**

- **Moving cameras:** the warps track real frame movement. Agreement is strong on 10, 11 and 13, and good on 05.
- **Static cameras:** low correlation is expected, because there is almost no true motion to correlate with. Warps there are small noise, not useful compensation.
- **MOT17-05 and 09 static-object evidence is weak** because only one and three objects qualify.

### Static-object residual by time window

Each later position is carried back to its anchor frame using only warps inside the window. "Raw" is image motion of a static object; "compensated" is what remains. Pixels; median.

| Sequence | 1 s raw -> compensated | 3 s | 8 s |
|---|---:|---:|---:|
| MOT17-02 | 0.0 -> 1.1 | 0.0 -> 2.3 | 1.0 -> 2.8 |
| MOT17-04 | 0.0 -> 2.8 | 0.0 -> 7.4 | 0.0 -> 17.0 |
| MOT17-09 | 0.0 -> 9.9 | 0.0 -> 30.7 | 0.0 -> 79.0 |
| MOT17-05 | 2.0 -> 7.7 | 4.5 -> 11.6 | 7.3 -> 28.1 |
| MOT17-10 | 79.1 -> 16.3 | 157.9 -> 32.3 | 227.1 -> 73.8 |
| MOT17-11 | 69.3 -> 68.3 | 144.8 -> 105.4 | 213.4 -> 149.2 |
| MOT17-13 | 85.2 -> 19.9 | 231.3 -> 51.3 | 681.1 -> 226.2 |

**Interpretation**

- **Clear benefit on MOT17-10 and 13,** where residual motion falls by 67-80%.
- **False motion on static cameras.** Compensation adds error where there was none. MOT17-09 is the worst at 79 px over 8 s, though only 3 objects support that figure.
- **MOT17-11 is unclear.** Frame-level checks are excellent, yet the 3 static objects barely improve at 1 s. **Uncertain:** nearby objects move differently from the background as the camera walks forward, which a single 2D transform cannot model (parallax), but this was not tested. The sample is small.
- **8 s figures rest on few pairs** for 05, 11 and 13 (3 each) and 09 (9).

### Drift in first-frame coordinates

Accumulating warps across a whole sequence is unreliable. By the last frame, implied zoom is 83.8x on MOT17-05, 50.2x on MOT17-13, 5.8x on MOT17-10 and 4.9x on MOT17-11. MOT17-04, a static camera, shows 6.0 degrees of false rotation.

---

## 10. Runtime and resource use

| Sequence | No-GMC tracking (s, approx.) | GMC tracking (s) | ms per frame |
|---|---:|---:|---:|
| MOT17-02 | 8 | 16.4 | 27 |
| MOT17-04 | 19 | 35.7 | 34 |
| MOT17-05 | 4 | 9.3 | 11 |
| MOT17-09 | 7 | 16.5 | 32 |
| MOT17-10 | 9 | 19.9 | 31 |
| MOT17-11 | 12 | 28.6 | 32 |
| MOT17-13 | 10 | 24.6 | 33 |
| **Total** | **69** | **151.2** | **28** |

- **GMC roughly doubles tracking time**, which remains small.
- **No-GMC times are estimates.** They come from file timestamps, because that run did not record tracking time.
- **Peak working set was 314 MB** for the GMC tracking process, read from the operating system. No equivalent figure exists for the no-GMC run.
- **`gmc_fixed/` uses 70 MB**, across 126 files before this report.

---

## 11. Pilot

The pilot used MOT17-04 and MOT17-10, with criteria declared in code before it ran (`gmc_pipeline.py`). All seven passed (`gmc_fixed/pilot/gmc/pilot_gate.json`).

| Check | Result |
|---|---|
| All 1,704 GMC calls successful | Pass |
| No identity warps after frame 1 | Pass |
| MOT17-04 median motion < 1 px | Pass (0.10) |
| MOT17-10 median motion > 1 px | Pass (3.29) |
| MOT17-10 photometric error improved on >= 80% of pairs | Pass (99.7%) |
| MOT17-10 phase correlation >= 0.7 | Pass (0.99 / 0.97) |
| MOT17-10 static objects steadier | Pass (158.3 -> 29.5 px) |

Frame panels for both sequences were inspected, and the boxes sat on people as expected. MOT17-10 fell from 260 to 163 raw track IDs.

The gate did not test static-camera stabilization. The added drift on MOT17-04 was found by reviewing the pilot output, and the anchored residual check above was added before the full run. It did not change any gate or tracking result.

---

## 12. Failures and warnings

- **Failed sequences:** none.
- **Log warnings:** none. The only matches for GMC warning text are the zero-count field names in the printed diagnostics.
- **TrackEval checks:** ground truth used as predictions scored 100% on HOTA, MOTA and IDF1. Supplementary TP, FP, FN, ID switch and fragmentation counts differ from TrackEval by zero on all seven sequences.
- **Preservation:** all 191 protected files are unchanged. That covers the no-GMC outputs, audit report, detections and weights by SHA-256, and ETH result files and `data/eth` by size and modification time.

### Known limitations

- `sparseOptFlow` does not mask pedestrians, so moving people can influence warps.
- Warps are 2D similarity transforms and cannot model parallax.
- Only seven sequences were compared, so per-sequence results may not generalise.
- Tracker settings were kept at defaults. A fixed 30-frame buffer means about 1 s at 30 fps and 2.1 s at 14 fps.
- Detector settings were not changed, so missed detections remain the same.

---

## 13. What can be attributed to GMC

| Observation | Attribution | Basis |
|---|---|---|
| All metric differences between the two runs | **Caused by enabling GMC, for this input** | Byte-identical GMC-off control; deterministic rerun |
| Fewer ID switches on moving cameras | **Reasonably attributed** | Large on 10, 13 and 05, near zero on static cameras, and warps are validated |
| MOT17-10 and 13 HOTA, AssA and IDF1 gains | **Reasonably attributed** | Strong warp agreement and static-object improvement |
| MOT17-05 gain | **Probably GMC** | Good frame-level agreement, but only one static object |
| MOT17-11 extra false positives | **Caused by GMC here; mechanism uncertain** | Follows from the control, but the cause was not established |
| Static-sequence changes, including MOT17-09 AssA +1.44 and IDF1 +2.23 | **Do not read as benefit** | No real motion to compensate; warps are noise, so this is probably a perturbation effect |
| Rising fragmentation | **Uncertain** | May partly reflect more pedestrians being tracked |
| Localization | **No meaningful effect** | Differences are negligible |
| Combined MOTA, nearly unchanged | **Not evidence of no effect** | Opposite changes offset each other |
| 8 s availability, slightly lower | **No improvement shown** | Small differences; pooled result dominated by MOT17-04 |

---

## 14. Guidance for later stages

These are recommendations only; no prediction work has started.

- **Treat `BoT-SORT_GMC_enabled` as the primary tracker output.** Keep `BoT-SORT_no_GMC` as the labelled baseline.
- **Do not use `stabilized_foot_x` and `stabilized_foot_y` over long spans.** They are image pixels in the first-frame view, not world coordinates or metres, and they drift heavily.
- **Use anchored, window-length compensation on moving-camera sequences,** computed from the saved warps.
- **Prefer raw pixels on static-camera sequences,** where compensation only adds noise.
- **Carry the anchored residuals into prediction as an error floor,** and report them.
- **Decide before prediction whether to test `orb` or `sift` GMC,** which can mask detected boxes. This would be a new experiment, because it changes a tracker setting.
- **Inspect MOT17-11's false tracks by hand** before relying on that sequence.

---

## 15. Output locations

All paths are under `runs/mot17_tracking_analysis/gmc_fixed/`.

| Item | Path |
|---|---|
| Tracker outputs | `trackers/BoT-SORT_GMC_enabled/data/` |
| Warp matrices | `gmc/<sequence>_warps.csv` |
| GMC diagnostics | `gmc/diagnostics.json` |
| Warp verification | `gmc/verification.json` |
| Pilot | `pilot/` |
| GMC-off control | `control_no_gmc/control_result.json` |
| Run configuration and status | `config/run_config.json`, `config/run_status.json` |
| Experiment labels | `config/experiment_labels.json` |
| Logs | `logs/` |
| TrackEval | `trackeval/metrics.csv`, `trackeval/metrics.json`, `trackeval/BoT-SORT_GMC_enabled/` |
| Error profiles | `error_profiles/` |
| Trajectories | `trajectories/` |
| Figures | `visualizations/` |
| Comparison data | `reports/comparison_tables.json` |
