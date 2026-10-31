# Methodology: Deep Learning-Based Multi-Object Tracking and Future Trajectory Prediction in Dynamic Environments

## 1. Project Goal

The goal is to investigate how specific multi-object tracking (MOT) errors affect downstream pedestrian trajectory prediction. The project focuses on four error types:

1. Localization noise
2. Detection dropout
3. Track fragmentation
4. Identity switches

The planned tracking algorithms are YOLO with ByteTrack and BoT-SORT. The planned prediction approaches are Constant Velocity, LSTM, and GRU. Tracking performance is evaluated using metrics such as MOTA, IDF1, HOTA, and identity switches; trajectory prediction is evaluated using Average Displacement Error (ADE) and Final Displacement Error (FDE).

> **Important:** This document describes a proposed methodology. It does not claim that any implementation or experiment has already been completed. The severity levels below are initial experimental suggestions and must be checked against real tracker outputs.

---

## 2. Overall Experimental Design

The work is organized into six phases:

1. Prepare clean pedestrian trajectories.
2. Build and validate clean prediction baselines.
3. Implement the four error injectors.
4. Run controlled error-injection experiments.
5. Evaluate real trackers on MOT17.
6. Relate real tracker error profiles to the controlled prediction experiments.

The core principle is to change one error type at a time, keep the evaluation setup consistent, and compare corrupted-input performance with a clean-input baseline.

---

## 3. Dataset Strategy

### 3.1 ETH/UCY: Trajectory Prediction

Start with ETH/UCY to develop the trajectory preprocessing, prediction, and error-injection pipeline. The commonly used benchmark scenes are ETH, HOTEL, UNIV, ZARA1, and ZARA2.

For the standard 8/12 protocol:

- **Observed history:** 8 positions.
- **Prediction horizon:** the next 12 positions.
- **Input:** the first 8 positions.
- **Evaluation target:** the original ground-truth positions for the following 12 steps.

Under the common 0.4-second sampling interval, 8 observations represent 3.2 seconds and 12 future steps represent 4.8 seconds. Verify the sampling interval in the version of the data being used.

Recommended preprocessing:

1. Load trajectory annotations and inspect the actual file schema.
2. Group rows by scene and pedestrian ID.
3. Sort each track by timestamp.
4. Check for invalid coordinates, duplicate timestamps, and discontinuities.
5. Create valid sliding windows with 8 observed and 12 future positions.
6. Follow a documented train/validation/test protocol. A common ETH/UCY evaluation setup holds out one scene for testing and trains on the remaining scenes, repeating across scenes.
7. Avoid data leakage between training, validation, and test data.

**Initial deliverable:** a loader that reports the number of valid trajectory windows per scene and can visualize sample trajectories.

### 3.2 MOT17: Tracking Evaluation

Use annotated MOT17 training sequences to evaluate YOLO with ByteTrack and BoT-SORT.

Recommended steps:

1. Select and document the MOT17 sequences and split.
2. Run the same YOLO detector and compatible preprocessing with each tracker.
3. Save frame-wise detections, bounding boxes, track IDs, and confidence scores.
4. Evaluate tracker output against ground-truth annotations using a compatible evaluation toolkit such as TrackEval.
5. Record MOTA, IDF1, HOTA, identity switches, missed detections, and localization-related statistics where the evaluation protocol supports them.

The ordinary MOT17 test split does not provide public ground-truth annotations for standard local evaluation. Start with annotated training data and use a clearly documented evaluation split.

### 3.3 How the Datasets Relate

ETH/UCY and MOT17 serve different purposes and use different coordinate systems and annotations. Do not directly combine their coordinates or assume that error rates transfer unchanged between datasets.

**Recommended initial strategy:** use MOT17 to estimate plausible tracking-error characteristics, then use those characteristics to choose realistic synthetic error levels for controlled experiments on ETH/UCY. Describe this explicitly as a cross-dataset calibration approach and state its limitations.

A more integrated alternative is to perform trajectory prediction directly from MOT17 tracker outputs and compare eligible predictions with matched ground-truth tracks. This is more complex and requires consistent coordinate representations, identity matching, and rules for missing or fragmented tracks.

---

## 4. Phase 1 — Prepare Clean Trajectories

For each trajectory, retain at least:

- Scene identifier
- Pedestrian identifier
- Timestamp or frame index
- X coordinate
- Y coordinate

The actual column names and units depend on the downloaded dataset version.

Preprocessing checklist:

1. Load the data and inspect its columns and coordinate ranges.
2. Group by scene and pedestrian ID.
3. Sort by timestamp.
4. Check that coordinates are finite and timestamps are ordered.
5. Identify continuous sequences with enough observations.
6. Create sliding windows containing 20 positions: 8 observed positions and 12 future positions.
7. Verify that the train/validation/test split prevents leakage.
8. Plot several trajectories to confirm that their paths look plausible.

If coordinates are in image pixels rather than ground-plane units, document this and do not label errors in metres without a valid mapping.

**Deliverables:** validated data loader, per-scene window counts, and trajectory visualizations.

---

## 5. Phase 2 — Build Clean Prediction Baselines

Implement and evaluate the following models on uncorrupted trajectories.

### 5.1 Constant Velocity

Use recent observed displacement to estimate velocity and extrapolate it into the future. This is a simple baseline, not a deep-learning model.

### 5.2 LSTM

Use an LSTM recurrent neural network to map an eight-step sequence of 2D positions to twelve future 2D positions.

### 5.3 GRU

Use a GRU recurrent neural network with the same input and output horizons. Keep the evaluation conditions comparable with those used for LSTM and Constant Velocity.

### 5.4 Baseline Evaluation

For every model, record:

- ADE
- FDE
- Number of valid test trajectories
- Dataset scene and split
- Model configuration
- Random seed, where applicable

Plot observed positions, ground-truth future positions, and predictions for sample pedestrians.

Do not proceed until the pipeline produces plausible predictions and the metric implementation has been checked.

---

## 6. Phase 3 — Error-Injection Framework

Each error injector should operate on a copy of the observed input trajectory. Keep the original future ground truth unchanged for evaluation. Test one error type at a time in the primary experiments.

The severity values below are **initial suggested settings**, not values specified in the original project presentation or measured from real tracker output. Adjust them after validating the implementation and examining real tracker statistics.

### 6.1 Localization Noise

**Purpose:** simulate inaccurate observed positions.

For a clean observed position \(p_t=(x_t,y_t)\), create a perturbed position:

\[
\tilde p_t = p_t + \epsilon_t,\qquad
\epsilon_t \sim \mathcal{N}(0,\sigma^2 I)
\]

Here, \(\sigma\) controls the noise scale for each coordinate and \(I\) is the 2D identity matrix.

Initial example levels: \(\sigma = 0.05,\ 0.15,\ 0.30\) metres, only if coordinates genuinely represent metres. Otherwise, choose and label appropriate coordinate units.

Procedure:

1. Copy the clean observed positions.
2. Add independent Gaussian noise to the observed coordinates.
3. Keep future ground truth unchanged.
4. Predict the future trajectory.
5. Calculate ADE and FDE against the clean future ground truth.
6. Repeat with multiple random seeds.

Checks:

- Calculate the actual displacement between clean and perturbed positions.
- Confirm that measured noise scale is consistent with the configured scale.
- Visualize clean and noisy trajectories.
- Verify that no observations were unintentionally removed.

### 6.2 Detection Dropout

**Purpose:** simulate missing observations.

Initial example levels: 10%, 20%, and 30% of observed positions. Also examine consecutive missing runs because the pattern of missingness can matter as much as the overall percentage.

Procedure:

1. Select observed timestamps according to the chosen dropout rate.
2. Mark those positions as missing; do not silently replace them with zero.
3. Apply a documented missing-data policy.
4. Run the predictor and compare against the original future ground truth.

Two possible policies:

- **Imputation:** fill missing positions using a defined method, such as interpolation.
- **Mask-aware model:** provide a missingness mask to a model designed and trained to use it.

Imputation is simpler for an initial experiment, but imputed positions must not be described as real detections.

Checks:

- Calculate the actual dropout percentage and longest consecutive missing run.
- Verify the missing timestamps.
- Confirm that the chosen imputation or masking policy works as intended.
- Report how many samples remain usable.

### 6.3 Track Fragmentation

**Purpose:** simulate one continuous track being split into separate segments.

Example: a pedestrian has one track ID, is lost for a short interval, and then reappears under a new ID.

Possible initial scenarios include breaks associated with 1, 2, or 3 missing time steps. These are proposed settings; define the exact rule before experimentation.

Procedure:

1. Choose a continuous track.
2. Introduce a break at a specified boundary.
3. Split the track into separate segments or IDs.
4. Apply a documented rule for constructing prediction samples.
5. Evaluate only segments that satisfy the input requirements, or use a predeclared missing-history policy.
6. Report prediction accuracy and the proportion of samples that become unusable.

Checks:

- Count segments before and after injection.
- Record segment lengths.
- Confirm that the break occurs at the intended timestamp.
- Report the number of valid and discarded prediction windows.

Fragmentation can reduce sample availability without necessarily worsening ADE for the surviving samples. Report both effects.

### 6.4 Identity Switches

**Purpose:** simulate incorrect associations between pedestrian identities.

For two pedestrians A and B, a synthetic switch can assign B's positions to A's track over a selected interval, with timestamps aligned.

Possible initial scenarios include switching 1, 2, or 3 consecutive observed positions. These are proposed settings, not real tracker measurements.

Procedure:

1. Select two distinct pedestrian tracks with compatible timestamps.
2. Choose the interval in which the incorrect association will occur.
3. Replace the selected observed positions with the other pedestrian's positions, or implement an equivalent explicit ID-assignment corruption.
4. Keep the true future belonging to the original target pedestrian unchanged.
5. Evaluate prediction error and record the switch duration.

Checks:

- Log the original identity and corrupted identity for every affected observation.
- Count switches and their durations.
- Verify that the intended positions, and only those positions, were changed.
- Check for unrealistic jumps caused by the injection.

This is a simplified synthetic model. Real tracker identity switches can be more complex; state the limitation in the report.

---

## 7. Phase 4 — Controlled Experiments

For each prediction model, run these conditions:

| Experiment | Input | Goal |
|---|---|---|
| E0 | Clean trajectories | Establish baseline |
| E1 | Localization noise at each severity | Measure sensitivity to inaccurate positions |
| E2 | Dropout at each severity | Measure sensitivity to missing observations |
| E3 | Fragmentation at each severity | Measure sensitivity to interrupted tracks |
| E4 | Identity switches at each severity | Measure sensitivity to incorrect identity associations |

Repeat the experiments for Constant Velocity, LSTM, and GRU.

For fair comparison:

1. Use the same test samples wherever the input representation permits.
2. Use the same corruption realizations for each model when practical.
3. Keep future ground truth unchanged.
4. Keep model evaluation settings consistent.
5. Repeat randomized experiments with multiple seeds.
6. Report sample counts and variability, not only average scores.

Start with one error type at a time. Combined errors can be explored as a later experiment.

---

## 8. Phase 5 — Validate Every Error Injector

Do not assume an error injector works simply because its code executes. Validate the actual corrupted data.

### 8.1 Visual Validation

Plot clean and corrupted trajectories for each error type. Inspect localization deviations, missing observations, fragment boundaries, and identity changes.

### 8.2 Statistical Validation

| Error type | Statistics to record |
|---|---|
| Localization noise | Mean displacement, noise standard deviation |
| Dropout | Missing percentage, longest missing run |
| Fragmentation | Number of segments, segment lengths, unusable sample rate |
| Identity switches | Number of switched assignments, switch duration, affected tracks |

### 8.3 Isolation Validation

For each experiment, confirm that only the intended error was deliberately introduced. For example, localization noise should not also delete positions, and dropout should not also add noise.

Also verify that the corruption routine does not modify the evaluation ground truth.

---

## 9. Phase 6 — Evaluate Real Trackers on MOT17

After the synthetic error pipeline works:

1. Run YOLO with ByteTrack on selected annotated MOT17 sequences.
2. Run the same detector and compatible preprocessing with BoT-SORT.
3. Save tracker output and configuration details.
4. Evaluate using a compatible tracking toolkit.
5. Record MOTA, IDF1, HOTA, identity switches, missed detections, and localization-related statistics where available.
6. Define how ground-truth and predicted pedestrian positions are matched before computing position errors.
7. Estimate the distributions of real localization errors, dropout durations, fragmentation, and identity switches.
8. Use those distributions to select realistic synthetic severity ranges for ETH/UCY.

Document the exact tracker versions, detector weights, configuration, dataset split, and evaluation protocol.

---

## 10. Evaluation Metrics

### 10.1 Tracking Metrics

- **MOTA:** accounts for false positives, false negatives, and identity switches in an overall tracking score.
- **IDF1:** measures identity-consistent tracking performance.
- **HOTA:** combines detection and association quality.
- **ID switches:** counts identity association errors under the evaluation protocol.

### 10.2 Trajectory Prediction Metrics

Average Displacement Error (ADE) is the mean Euclidean distance between predicted and ground-truth positions across the prediction horizon:

\[
ADE = \frac{1}{T}\sum_{t=1}^{T}\|\hat p_t-p_t\|_2
\]

Final Displacement Error (FDE) is the Euclidean distance between the predicted and ground-truth final positions:

\[
FDE = \|\hat p_T-p_T\|_2
\]

Use consistent coordinate units and evaluation conventions across conditions.

### 10.3 Degradation Relative to Clean Baseline

For a model and a particular error condition:

\[
\Delta ADE = ADE_{\text{corrupt}}-ADE_{\text{clean}}
\]

Relative degradation can be calculated as:

\[
D_{ADE} =
\frac{ADE_{\text{corrupt}}-ADE_{\text{clean}}}
{ADE_{\text{clean}}}\times100\%
\]

Apply the equivalent calculation to FDE. If the clean baseline metric is zero, relative degradation is undefined and should not be calculated using this formula.

Positive degradation indicates worse performance; negative degradation indicates better measured performance under that condition. Interpret small differences in light of experimental variability.

---

## 11. Results to Save

Save a structured experiment log with one record per model, scene, error type, severity, and random seed.

Suggested fields:

- Dataset and scene
- Train/validation/test split
- Prediction model and configuration
- Error type
- Severity parameter and units
- Random seed
- ADE and FDE
- Number of valid test trajectories
- Number of missing or discarded samples
- Relevant error-injection statistics

Suggested results table:

| Model | Error type | Severity | ADE | FDE | Valid samples |
|---|---|---|---|---|---|
| Constant Velocity | Clean | 0 | Measured | Measured | Count |
| Constant Velocity | Localization noise | Low | Measured | Measured | Count |
| LSTM | Clean | 0 | Measured | Measured | Count |
| LSTM | Localization noise | Low | Measured | Measured | Count |
| GRU | Clean | 0 | Measured | Measured | Count |

The entries above are placeholders for actual experiment results.

### Recommended plots

1. **Degradation curves:** error severity versus ADE/FDE, with separate lines for each predictor.
2. **Error-type comparison:** compare degradation caused by different error types at selected severity levels.
3. **Robustness heatmap:** compare models, error types, and severity levels using relative degradation.
4. **Trajectory visualizations:** compare clean, corrupted, predicted, and ground-truth paths.

---

## 12. Interpretation and Research Contribution

The project should answer:

1. Which tracking error types cause the greatest degradation in trajectory prediction?
2. How does prediction accuracy change as each error becomes more severe?
3. Do Constant Velocity, LSTM, and GRU respond differently to the same tracking error?
4. Are the observed trends consistent across different scenes?
5. How do synthetic error experiments relate to error characteristics measured from real trackers?

Do not assume in advance which error is most harmful or which model is most robust. These are empirical questions.

The intended contribution is a controlled analysis of how particular tracking failures propagate into downstream trajectory prediction, including error isolation, severity analysis, model comparisons, and calibration against real tracker behaviour.

A literature review should also verify the closest prior work before making a strong novelty claim.

---

## 13. Recommended Implementation Order

Follow this order to avoid building too many components simultaneously:

- [ ] Inspect the existing project code and result files before starting from scratch.
- [ ] Load one ETH/UCY scene and inspect the data format.
- [ ] Visualize clean pedestrian trajectories.
- [ ] Create and validate the 8-observation/12-prediction windows.
- [ ] Implement Constant Velocity and validate ADE/FDE.
- [ ] Implement and validate localization noise.
- [ ] Add dropout and document the missing-data policy.
- [ ] Add fragmentation and track sample availability reporting.
- [ ] Add identity switches and validate identity corruption.
- [ ] Train and evaluate LSTM and GRU.
- [ ] Run controlled experiments and generate degradation plots.
- [ ] Evaluate ByteTrack and BoT-SORT on annotated MOT17 sequences.
- [ ] Calibrate synthetic error ranges using measured tracker statistics.
- [ ] Write the analysis, limitations, and conclusions from measured results.

**Recommended starting point:** ETH/UCY → clean Constant Velocity baseline → localization noise → ADE/FDE comparison. Once this end-to-end path works, extend it to the remaining errors and models.
