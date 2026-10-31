"""Run the MOT17 YOLO + BoT-SORT analysis stage by stage.

No stage trains or fine-tunes a model. Each stage reads the saved output of the
previous one, so a stage can be repeated without repeating detection.

Example: ``python -m trackvision.tracking.mot17.pipeline detect``
"""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import platform
import shlex
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from trackvision.tracking.mot17.dataset import (
    DEFAULT_DATASET_ROOT,
    DEFAULT_OUTPUT_ROOT,
    SequenceInfo,
    build_inventory,
    discover_sequences,
    ensure_dataset_root,
    ensure_output_root,
    load_ground_truth,
)
from trackvision.tracking.mot17.detection import (
    DetectorConfig,
    describe_weights,
    detect_sequence,
    load_detector,
)
from trackvision.tracking.mot17.error_profile import (
    AVAILABILITY_SECONDS,
    MATCH_FIELDS,
    MATCH_IOU_THRESHOLD,
    SequenceMatches,
    detection_counts,
    distribution,
    ground_truth_continuity,
    match_sequence,
    predicted_track_summary,
    survival_curve,
    window_availability,
)
from trackvision.tracking.mot17.evaluation import (
    COMBINED_KEY,
    TRACKER_NAME,
    run_trackeval,
    stage_ground_truth,
    write_metrics,
)
from trackvision.tracking.mot17.formats import (
    box_centers,
    box_foot_points,
    format_mot_row,
    load_mot_file,
)
from trackvision.tracking.mot17.stabilization import (
    load_warps,
    stabilize_mot_points,
    to_first_frame_transforms,
)
from trackvision.tracking.mot17.tracker import (
    BotSortConfig,
    estimate_camera_motion,
    track_sequence,
)
from trackvision.tracking.mot17.visualization import render_all


STAGES = (
    "inventory",
    "fetch-weights",
    "camera-motion",
    "detect",
    "track",
    "evaluate",
    "analyze",
    "visualize",
)
TRACKED_PACKAGES = (
    "ultralytics",
    "torch",
    "torchvision",
    "numpy",
    "scipy",
    "opencv-python",
    "lap",
    "trackeval",
    "matplotlib",
    "pandas",
)
TRACKEVAL_COMMIT = "12c8791b303e0a0b50f753af204249e622d0281a"
# Measured in seconds because MOT17 frame rates differ between sequences.
SURVIVAL_HORIZON_SECONDS = (0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 20.0)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def _write_csv(path: Path, fields: Sequence[str], rows: Iterable[Sequence[Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(fields)
        writer.writerows(rows)


def _package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for package in TRACKED_PACKAGES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def record_environment(output_root: Path) -> None:
    """Record the interpreter, packages, hardware and invoking command."""
    import torch

    freeze = subprocess.run(
        [sys.executable, "-m", "pip", "freeze", "--disable-pip-version-check"],
        capture_output=True,
        text=True,
        check=False,
    )
    _write_json(
        output_root / "config" / "environment.json",
        {
            "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
            "python": platform.python_version(),
            "executable": sys.executable,
            "platform": platform.platform(),
            "processor": platform.processor(),
            "cuda_available": torch.cuda.is_available(),
            "torch_threads": torch.get_num_threads(),
            "packages": _package_versions(),
            "trackeval_commit": TRACKEVAL_COMMIT,
            "trackeval_note": (
                "NumPy 2 removed np.float/np.int/np.bool; the evaluation module "
                "restores them as builtins at runtime. TrackEval is unmodified."
            ),
        },
    )
    (output_root / "config" / "pip_freeze.txt").write_text(
        freeze.stdout, encoding="utf-8"
    )


def _log_command(output_root: Path, argv: Sequence[str]) -> None:
    path = output_root / "config" / "commands.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    command = " ".join(
        shlex.quote(part)
        for part in (sys.executable, "-m", "trackvision.tracking.mot17.pipeline", *argv)
    )
    with path.open("a", encoding="utf-8") as stream:
        stream.write(f"{datetime.now(timezone.utc).isoformat()} {command}\n")


def write_run_config(
    output_root: Path, detector: DetectorConfig, tracker: BotSortConfig
) -> None:
    _write_json(
        output_root / "config" / "run_config.json",
        {
            "training_performed": False,
            "fine_tuning_performed": False,
            "detector": detector.as_dict(),
            "tracker": {
                "implementation": "ultralytics.trackers.bot_sort.BOTSORT",
                "name": TRACKER_NAME,
                "settings": tracker.as_dict(),
                "settings_source": "Ultralytics botsort.yaml defaults",
                "detections_reused": True,
                "state_reset": "A new tracker is built for every sequence.",
                "omitted_from_official_botsort": (
                    "The official demo's filtering of wide boxes and small areas "
                    "is not part of the Ultralytics implementation."
                ),
            },
            "evaluation": {
                "toolkit": "TrackEval",
                "commit": TRACKEVAL_COMMIT,
                "dataset_class": "MotChallenge2DBox",
                "preprocessing": True,
                "class": "pedestrian",
                "iou_threshold": MATCH_IOU_THRESHOLD,
            },
        },
    )


def stage_inventory(dataset_root: Path, output_root: Path) -> None:
    inventory = build_inventory(dataset_root)
    _write_json(output_root / "config" / "dataset_inventory.json", inventory)
    print(
        f"{inventory['sequence_count']} sequences, "
        f"{inventory['total_frames']} unique frames"
    )


def stage_fetch_weights(detector: DetectorConfig) -> None:
    """Download only the pretrained checkpoint named in the configuration."""
    from ultralytics.utils.downloads import attempt_download_asset

    path = Path(detector.weights)
    path.parent.mkdir(parents=True, exist_ok=True)
    attempt_download_asset(str(path))
    print(json.dumps(describe_weights(detector), indent=2))


def stage_camera_motion(
    sequences: Sequence[SequenceInfo], output_root: Path, tracker: BotSortConfig
) -> None:
    measurements = [
        estimate_camera_motion(sequence, tracker.gmc_method) for sequence in sequences
    ]
    _write_json(
        output_root / "config" / "camera_motion.json",
        {
            "method": tracker.gmc_method,
            "use": "Global motion compensation stays on for all sequences.",
            "reason": (
                "MOT17 mixes static and moving cameras. One setting is applied so "
                "the configuration is not chosen per sequence."
            ),
            "measurements": measurements,
        },
    )
    for row in measurements:
        print(row)


def stage_detect(
    sequences: Sequence[SequenceInfo],
    output_root: Path,
    detector: DetectorConfig,
    limit: int | None,
) -> None:
    before = describe_weights(detector)
    model, person_class = load_detector(detector)
    results = [
        detect_sequence(
            model,
            person_class,
            sequence,
            detector,
            output_root / "detections",
            limit,
        )
        for sequence in sequences
    ]
    after = describe_weights(detector)
    if before["sha256"] != after["sha256"]:
        raise RuntimeError("Detector weights changed during inference.")
    _write_json(
        output_root / "config" / "weights_fingerprint.json",
        {
            "person_class_id": person_class,
            "weights_unchanged_by_run": True,
            **after,
        },
    )
    for result in results:
        print(
            {
                key: result[key]
                for key in (
                    "sequence",
                    "frames_processed",
                    "complete",
                    "detections",
                    "empty_frames",
                    "frames_per_second",
                )
            }
        )


def stage_track(
    sequences: Sequence[SequenceInfo], output_root: Path, tracker: BotSortConfig
) -> None:
    for sequence in sequences:
        print(
            track_sequence(
                sequence,
                output_root / "detections",
                output_root / "trackers" / TRACKER_NAME,
                tracker,
            ),
            flush=True,
        )


def _ground_truth_as_tracker(
    sequences: Sequence[SequenceInfo], trackers_root: Path
) -> None:
    data_dir = trackers_root / "oracle" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    for sequence in sequences:
        rows = load_ground_truth(sequence)
        rows = rows[(rows[:, 6] == 1) & (rows[:, 7] == 1)]
        with (data_dir / f"{sequence.name}.txt").open("w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(format_mot_row(row[0], row[1], row[2:6], 1.0) + "\n")


def stage_evaluate(
    sequences: Sequence[SequenceInfo],
    output_root: Path,
    tracker_name: str = TRACKER_NAME,
    experiment: str = "BoT-SORT_no_GMC",
) -> dict[str, dict[str, float]]:
    """Check TrackEval on ground truth as predictions, then score BoT-SORT."""
    evaluation_dir = output_root / "trackeval"
    gt_root = evaluation_dir / "gt_staging"
    stage_ground_truth(sequences, gt_root)

    sanity_dir = evaluation_dir / "sanity_ground_truth_as_tracker"
    _ground_truth_as_tracker(sequences, sanity_dir / "trackers")
    sanity = run_trackeval(
        sequences, gt_root, sanity_dir / "trackers", sanity_dir / "results", "oracle"
    )
    combined = sanity[COMBINED_KEY]
    # Boxes rounded to 2 decimals overlap imperfectly, so HOTA sits just below 1.
    if min(combined["MOTA"], combined["IDF1"], combined["HOTA"]) < 0.999:
        raise RuntimeError(f"TrackEval sanity check failed: {combined}")
    _write_json(sanity_dir / "summary.json", combined)

    metrics = run_trackeval(
        sequences, gt_root, output_root / "trackers", evaluation_dir, tracker_name
    )
    write_metrics(metrics, evaluation_dir, experiment)
    for name, row in metrics.items():
        print(
            f"{name}: HOTA {row['HOTA']:.4f} MOTA {row['MOTA']:.4f} "
            f"IDF1 {row['IDF1']:.4f} IDSW {row['IDSW']:.0f} "
            f"FP {row['CLR_FP']:.0f} FN {row['CLR_FN']:.0f}"
        )
    return metrics


def _write_trajectories(
    sequence: SequenceInfo,
    predictions: np.ndarray,
    result: SequenceMatches,
    output_dir: Path,
    transforms: dict[int, np.ndarray] | None = None,
) -> None:
    """Preserve pixel trajectories for a later, separately designed MOT17 stage.

    With ``transforms``, both tables also carry foot points in the sequence's
    first-frame view, using one convention. They remain image pixels.
    """
    fps = sequence.frame_rate

    def stabilized_columns(rows: np.ndarray, feet: np.ndarray) -> list[tuple[Any, ...]]:
        if transforms is None:
            return [()] * len(rows)
        return [tuple(point) for point in stabilize_mot_points(rows[:, 0], feet, transforms)]

    stabilized_fields = (
        ("stabilized_foot_x", "stabilized_foot_y") if transforms is not None else ()
    )
    gt = result.evaluated_gt[np.lexsort((result.evaluated_gt[:, 0], result.evaluated_gt[:, 1]))]
    gt_centers = box_centers(gt[:, 2:6])
    gt_feet = box_foot_points(gt[:, 2:6])
    matched_gt = {
        (int(row[0]), int(row[2])): (int(row[1]), row[3]) for row in result.matches
    }
    matched_pred = {
        (int(row[0]), int(row[1])): int(row[2]) for row in result.matches
    }
    false_positives = {
        (int(row[0]), int(row[1])) for row in result.false_positives
    }
    _write_csv(
        output_dir / f"{sequence.name}_ground_truth.csv",
        (
            "sequence", "frame_rate", "frame", "time_s", "gt_id", "left", "top",
            "width", "height", "center_x", "center_y", "foot_x", "foot_y",
            *stabilized_fields, "visibility", "botsort_track_id", "tracked",
            "match_status",
        ),
        (
            (
                sequence.name, fps, int(row[0]), row[0] / fps, int(row[1]), *row[2:6],
                *center, *foot, *stabilized, row[8],
                matched_pred.get((int(row[0]), int(row[1])), ""),
                int((int(row[0]), int(row[1])) in matched_pred),
                "tracked" if (int(row[0]), int(row[1])) in matched_pred else "missed",
            )
            for row, center, foot, stabilized in zip(
                gt, gt_centers, gt_feet, stabilized_columns(gt, gt_feet)
            )
        ),
    )
    ordered = predictions[np.lexsort((predictions[:, 0], predictions[:, 1]))]
    centers = box_centers(ordered[:, 2:6])
    feet = box_foot_points(ordered[:, 2:6])
    previous: dict[int, int] = {}
    rows = []
    for row, center, foot, stabilized in zip(
        ordered, centers, feet, stabilized_columns(ordered, feet)
    ):
        frame, track_id = int(row[0]), int(row[1])
        missing_before = frame - previous[track_id] - 1 if track_id in previous else 0
        previous[track_id] = frame
        match = matched_gt.get((frame, track_id))
        if match:
            status = "matched"
        elif (frame, track_id) in false_positives:
            status = "false_positive"
        else:
            status = "distractor"
        rows.append(
            (
                sequence.name, fps, frame, frame / fps, track_id, *row[2:6], row[6],
                *center, *foot, *stabilized, missing_before, status,
                match[0] if match else "", match[1] if match else "",
            )
        )
    _write_csv(
        output_dir / f"{sequence.name}_botsort.csv",
        (
            "sequence", "frame_rate", "frame", "time_s", "track_id", "left", "top",
            "width", "height", "confidence", "center_x", "center_y", "foot_x",
            "foot_y", *stabilized_fields, "missing_frames_before", "match_status",
            "matched_gt_id", "matched_iou",
        ),
        rows,
    )


def stage_analyze(
    sequences: Sequence[SequenceInfo],
    output_root: Path,
    detections_dir: Path | None = None,
    tracker_name: str = TRACKER_NAME,
    warps_dir: Path | None = None,
    experiment: str = "BoT-SORT_no_GMC",
) -> None:
    """Build the supplementary error profiles and cross-check them with TrackEval."""
    profile_dir = output_root / "error_profiles"
    trajectory_dir = output_root / "trajectories"
    detections_dir = detections_dir or output_root / "detections"
    official = json.loads(
        (output_root / "trackeval" / "metrics.json").read_text(encoding="utf-8")
    )["metrics"]

    matched_rows: list[Sequence[Any]] = []
    continuity_rows: list[Sequence[Any]] = []
    track_rows: list[Sequence[Any]] = []
    gap_rows: list[Sequence[Any]] = []
    availability_rows: list[Sequence[Any]] = []
    survival_rows: list[Sequence[Any]] = []
    summaries: list[dict[str, Any]] = []
    pooled_iou: list[np.ndarray] = []
    pooled_center: list[np.ndarray] = []
    pooled_center_px: list[np.ndarray] = []
    pooled_gaps: list[float] = []
    pooled_gaps_seconds: list[float] = []
    pooled_coverage: list[float] = []
    availability_totals: Counter[float] = Counter()
    availability_possible: Counter[float] = Counter()
    pooled_counts: Counter[str] = Counter()
    detector_pooled: dict[str, Counter[str]] = {"0.25": Counter(), "0.10": Counter()}

    for sequence in sequences:
        fps = float(sequence.frame_rate)
        ground_truth = load_ground_truth(sequence)
        predictions = load_mot_file(
            output_root / "trackers" / tracker_name / "data" / f"{sequence.name}.txt"
        )
        detections = load_mot_file(detections_dir / f"{sequence.name}.txt")
        result = match_sequence(ground_truth, predictions)
        detector_counts = {}
        for label, threshold in (("0.25", 0.25), ("0.10", 0.10)):
            selected = detections[detections[:, 6] >= threshold]
            detector_counts[label] = detection_counts(
                match_sequence(ground_truth, selected)
            )
            detector_pooled[label].update(
                {
                    key: detector_counts[label][key]
                    for key in ("true_positives", "false_positives", "false_negatives")
                }
            )

        matched_rows.extend(
            (sequence.name, *row) for row in result.matches.tolist()
        )
        continuity = ground_truth_continuity(result, fps)
        tracks = predicted_track_summary(predictions, result, fps)
        for record in continuity:
            continuity_rows.append(
                (
                    sequence.name, record["gt_id"], record["gt_frames"],
                    record["gt_seconds"], record["tracked_frames"], record["coverage"],
                    record["missed_frames"], record["fragmentations"],
                    record["identity_switches"], record["distinct_track_ids"],
                    record["longest_continuous_frames"],
                )
            )
            for gap in record["interior_gaps"]:
                gap_rows.append((sequence.name, "gt_interior_gap", gap, gap / fps))
                pooled_gaps.append(gap)
                pooled_gaps_seconds.append(gap / fps)
            pooled_coverage.append(record["coverage"])
        for record in tracks:
            track_rows.append(
                (
                    sequence.name, record["track_id"], record["first_frame"],
                    record["last_frame"], record["observed_frames"],
                    record["span_frames"], record["span_seconds"],
                    record["missing_frames_inside_span"], record["matched_frames"],
                    record["matched_gt_ids"], record["dominant_gt_fraction"],
                )
            )
            for gap in record["gaps"]:
                gap_rows.append((sequence.name, "predicted_track_gap", gap, gap / fps))

        for seconds in AVAILABILITY_SECONDS:
            window = round(seconds * fps)
            availability = window_availability(continuity, window)
            availability_rows.append(
                (
                    sequence.name, seconds, window, availability["possible_windows"],
                    availability["available_windows"], availability["availability"],
                )
            )
            availability_totals[seconds] += availability["available_windows"]
            availability_possible[seconds] += availability["possible_windows"]
        run_seconds_horizons = [round(s * fps) for s in SURVIVAL_HORIZON_SECONDS]
        continuous_runs = [
            run for record in continuity for run in record["continuous_runs"]
        ]
        for seconds, point in zip(
            SURVIVAL_HORIZON_SECONDS,
            survival_curve(continuous_runs, run_seconds_horizons),
        ):
            survival_rows.append(
                (sequence.name, seconds, point["horizon"], point["survival"])
            )

        transforms = (
            to_first_frame_transforms(load_warps(warps_dir / f"{sequence.name}_warps.csv"))
            if warps_dir is not None
            else None
        )
        _write_trajectories(sequence, predictions, result, trajectory_dir, transforms)

        identity_switches = sum(record["identity_switches"] for record in continuity)
        fragmentations = sum(record["fragmentations"] for record in continuity)
        counts = detection_counts(result)
        pooled_counts.update(
            {
                "true_positives": counts["true_positives"],
                "false_positives": counts["false_positives"],
                "false_negatives": counts["false_negatives"],
                "identity_switches": identity_switches,
                "fragmentations": fragmentations,
            }
        )
        pooled_iou.append(result.matches[:, 3])
        pooled_center.append(result.matches[:, 5])
        pooled_center_px.append(result.matches[:, 4])
        row = official[sequence.name]
        summaries.append(
            {
                "sequence": sequence.name,
                "frame_rate": sequence.frame_rate,
                "image_size": [sequence.width, sequence.height],
                "tracker": counts,
                "detector_conf_0.25": detector_counts["0.25"],
                "detector_conf_0.10": detector_counts["0.10"],
                "identity_switches": identity_switches,
                "fragmentations": fragmentations,
                "iou": distribution(result.matches[:, 3]),
                "center_error_px": distribution(result.matches[:, 4]),
                "center_error_over_height": distribution(result.matches[:, 5]),
                "track_span_seconds": distribution(
                    [record["span_seconds"] for record in tracks]
                ),
                "cross_check_against_trackeval": {
                    "true_positives": int(counts["true_positives"] - row["CLR_TP"]),
                    "false_positives": int(counts["false_positives"] - row["CLR_FP"]),
                    "false_negatives": int(counts["false_negatives"] - row["CLR_FN"]),
                    "identity_switches": int(identity_switches - row["IDSW"]),
                    "fragmentations": int(fragmentations - row["Frag"]),
                },
            }
        )

    _write_csv(profile_dir / "matched_localization.csv", ("sequence", *MATCH_FIELDS), matched_rows)
    _write_csv(
        profile_dir / "gt_track_continuity.csv",
        (
            "sequence", "gt_id", "gt_frames", "gt_seconds", "tracked_frames",
            "coverage", "missed_frames", "fragmentations", "identity_switches",
            "distinct_track_ids", "longest_continuous_frames",
        ),
        continuity_rows,
    )
    _write_csv(
        profile_dir / "predicted_tracks.csv",
        (
            "sequence", "track_id", "first_frame", "last_frame", "observed_frames",
            "span_frames", "span_seconds", "missing_frames_inside_span",
            "matched_frames", "matched_gt_ids", "dominant_gt_fraction",
        ),
        track_rows,
    )
    _write_csv(
        profile_dir / "missing_frame_distribution.csv",
        ("sequence", "kind", "length_frames", "length_seconds"),
        gap_rows,
    )
    _write_csv(
        profile_dir / "window_availability.csv",
        ("sequence", "window_seconds", "window_frames", "possible_windows", "available_windows", "availability"),
        availability_rows,
    )
    _write_csv(
        profile_dir / "identity_run_survival.csv",
        ("sequence", "horizon_seconds", "horizon_frames", "survival"),
        survival_rows,
    )

    def pooled_detector(label: str) -> dict[str, Any]:
        totals = detector_pooled[label]
        return {
            **totals,
            "recall": totals["true_positives"]
            / max(1, totals["true_positives"] + totals["false_negatives"]),
            "precision": totals["true_positives"]
            / max(1, totals["true_positives"] + totals["false_positives"]),
        }

    _write_json(
        profile_dir / "error_profile.json",
        {
            "experiment": experiment,
            "kind": "Supplementary analysis, not official TrackEval metrics.",
            "official_metrics": (output_root / "trackeval" / "metrics.json").as_posix(),
            "matching": {
                "iou_threshold": MATCH_IOU_THRESHOLD,
                "evaluated_ground_truth": "conf == 1 and class == 1",
                "distractor_classes_removed": [2, 7, 8, 12],
                "assignment": "per-frame Hungarian; continuing identities take priority",
            },
            "definitions": {
                "iou": "Overlap of matched tracker and ground-truth boxes; only IoU >= 0.5 can match, so this is a truncated distribution.",
                "center_error_px": "Euclidean centre distance of matched boxes in image pixels.",
                "center_error_over_height": "Centre error divided by ground-truth box height.",
                "interior_gap": "Run of unmatched evaluated ground-truth frames between two matched frames.",
                "fragmentation": "An interior gap; equal in meaning to TrackEval's Frag.",
                "window_availability": "Share of ground-truth windows of this duration covered by one unbroken tracker identity.",
                "identity_run_survival": "Share of unbroken identity runs reaching a duration; runs ended by the video are not treated as censored.",
            },
            "coordinate_units": "image pixels; no ground-plane or metric conversion",
            "limitations": [
                "Pixel errors depend on image size, camera distance and motion, so they are not comparable with ETH coordinates.",
                "Missed and unmatched boxes contribute no localization error, so localization statistics describe successful matches only.",
                "Sequences have different frame rates; durations are reported in seconds where they are pooled.",
                "All seven sequences are training-split videos; nothing was tuned on them, but they are not the official test set.",
            ],
            "overall": {
                **pooled_counts,
                "recall": pooled_counts["true_positives"]
                / max(1, pooled_counts["true_positives"] + pooled_counts["false_negatives"]),
                "precision": pooled_counts["true_positives"]
                / max(1, pooled_counts["true_positives"] + pooled_counts["false_positives"]),
                "detector_conf_0.25": pooled_detector("0.25"),
                "detector_conf_0.10": pooled_detector("0.10"),
                "iou": distribution(np.concatenate(pooled_iou)),
                "center_error_px": distribution(np.concatenate(pooled_center_px)),
                "center_error_over_height": distribution(np.concatenate(pooled_center)),
                "interior_gap_frames": distribution(pooled_gaps),
                "interior_gap_seconds": distribution(pooled_gaps_seconds),
                "per_pedestrian_coverage": distribution(pooled_coverage),
                "window_availability": {
                    str(seconds): availability_totals[seconds] / availability_possible[seconds]
                    for seconds in AVAILABILITY_SECONDS
                    if availability_possible[seconds]
                },
            },
            "sequences": summaries,
        },
    )
    _write_json(
        trajectory_dir / "sequences.json",
        {
            "experiment": experiment,
            "note": "Pixel trajectories kept for a future MOT17-specific prediction stage. Do not feed them to the ETH-trained LSTM.",
            "coordinates": "MOTChallenge 1-based image pixels",
            "stabilized_columns": (
                "Foot points mapped to the sequence's first-frame view by accumulated GMC warps. Image pixels, not world coordinates or metres, and subject to drift."
                if warps_dir is not None
                else "None in this experiment."
            ),
            "sequences": [
                {
                    "sequence": sequence.name,
                    "frame_rate": sequence.frame_rate,
                    "frames": sequence.length,
                    "width": sequence.width,
                    "height": sequence.height,
                }
                for sequence in sequences
            ],
        },
    )
    print(json.dumps({"overall_counts": pooled_counts}, indent=2))


def stage_visualize(sequences: Sequence[SequenceInfo], output_root: Path) -> None:
    for path in render_all(sequences, output_root):
        print(path)


def write_status(sequences: Sequence[SequenceInfo], output_root: Path) -> None:
    """Record, per sequence, which stages produced complete outputs."""
    rows = []
    for sequence in sequences:
        detection_meta = output_root / "detections" / f"{sequence.name}.meta.json"
        detection_complete = detection_meta.is_file() and json.loads(
            detection_meta.read_text(encoding="utf-8")
        ).get("complete", False)
        rows.append(
            {
                "sequence": sequence.name,
                "detections_complete": bool(detection_complete),
                "tracks_written": (
                    output_root / "trackers" / TRACKER_NAME / "data" / f"{sequence.name}.txt"
                ).is_file(),
                "trajectories_written": (
                    output_root / "trajectories" / f"{sequence.name}_botsort.csv"
                ).is_file(),
            }
        )
    _write_json(output_root / "config" / "run_status.json", {"sequences": rows})


def reject_archived_overwrite(stages: Sequence[str], output_root: Path) -> None:
    """Protect the BoT-SORT_no_GMC experiment; new runs use gmc_pipeline."""
    archived = output_root / "trackers" / TRACKER_NAME / "data"
    blocked = sorted(set(stages) & {"track", "evaluate", "analyze", "visualize"})
    if blocked and archived.is_dir():
        raise SystemExit(
            f"Refusing {blocked}: {archived} is the archived no-GMC experiment."
        )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stages", nargs="+", choices=(*STAGES, "all"))
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--sequences", nargs="*", default=None)
    parser.add_argument("--limit", type=int, default=None, help="Timing runs only: stop detection at this frame.")
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)

    dataset_root = ensure_dataset_root(args.dataset_root)
    output_root = ensure_output_root(args.output_root)
    sequences = discover_sequences(dataset_root, args.sequences)
    detector = DetectorConfig()
    tracker = BotSortConfig()

    _log_command(output_root, sys.argv[1:] if argv is None else argv)
    record_environment(output_root)
    write_run_config(output_root, detector, tracker)

    stages = [stage for stage in STAGES if stage != "fetch-weights"] if "all" in args.stages else args.stages
    reject_archived_overwrite(stages, output_root)
    for stage in stages:
        print(f"== {stage}", flush=True)
        if stage == "inventory":
            stage_inventory(dataset_root, output_root)
        elif stage == "fetch-weights":
            stage_fetch_weights(detector)
        elif stage == "camera-motion":
            stage_camera_motion(sequences, output_root, tracker)
        elif stage == "detect":
            stage_detect(sequences, output_root, detector, args.limit)
        elif stage == "track":
            stage_track(sequences, output_root, tracker)
        elif stage == "evaluate":
            stage_evaluate(sequences, output_root)
        elif stage == "analyze":
            stage_analyze(sequences, output_root)
        elif stage == "visualize":
            stage_visualize(sequences, output_root)
    write_status(sequences, output_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
