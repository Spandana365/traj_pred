"""Rerun BoT-SORT with working GMC from the saved YOLO detections.

Outputs go to ``runs/mot17_tracking_analysis/gmc_fixed/``. The archived
``BoT-SORT_no_GMC`` experiment, its detections and its reports are only read.
YOLO is never loaded and nothing is trained.

Example: ``python -m trackvision.tracking.mot17.gmc_pipeline pilot``
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import psutil

from trackvision.tracking.mot17.dataset import (
    DEFAULT_DATASET_ROOT,
    SequenceInfo,
    discover_sequences,
    ensure_dataset_root,
    ensure_gmc_output_root,
    file_sha256,
)
from trackvision.tracking.mot17.detection import DetectorConfig
from trackvision.tracking.mot17.evaluation import COMBINED_KEY
from trackvision.tracking.mot17.gmc_verification import verify_sequence
from trackvision.tracking.mot17.pipeline import (
    TRACKER_NAME,
    _write_json,
    record_environment,
    stage_analyze,
    stage_evaluate,
)
from trackvision.tracking.mot17.stabilization import load_warps
from trackvision.tracking.mot17.tracker import (
    BotSortConfig,
    GMCPipelineFailure,
    GMCRecorder,
    track_sequence,
)
from trackvision.tracking.mot17.visualization import render_all, render_frame_panel


DEFAULT_GMC_ROOT = Path("runs/mot17_tracking_analysis/gmc_fixed")
EXPERIMENT_NO_GMC = "BoT-SORT_no_GMC"
EXPERIMENT_GMC = "BoT-SORT_GMC_enabled"
PILOT_SEQUENCES = ("MOT17-04", "MOT17-10")
STATIC_LOOKING = ("MOT17-02", "MOT17-04", "MOT17-09")
EXPECTED_TOTAL_FRAMES = 5316
# Declared before the pilot ran. Pixels are at full image resolution.
PILOT_MAX_STATIC_MEDIAN_MOTION = 1.0
PILOT_MIN_MOVING_MEDIAN_MOTION = 1.0
PILOT_MIN_MOVING_IMPROVED_SHARE = 0.8
PILOT_MIN_MOVING_CORRELATION = 0.7
GMC_WARNING_TEXT = ("GMC failed", "not enough matching points", "affine estimation failed")
COMPARED_METRICS = (
    "HOTA", "DetA", "AssA", "LocA", "DetRe", "DetPr", "AssRe", "AssPr", "MOTA",
    "MOTP", "IDF1", "IDR", "IDP", "IDSW", "CLR_TP", "CLR_FP", "CLR_FN", "Frag",
    "MT", "PT", "ML", "IDs", "Dets",
)


class GMCWarningCounter(logging.Handler):
    """Count Ultralytics GMC warnings, including ones that raise nothing."""

    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.counts = {text: 0 for text in GMC_WARNING_TEXT}

    def emit(self, record: logging.LogRecord) -> None:
        message = record.getMessage()
        for text in GMC_WARNING_TEXT:
            if text in message:
                self.counts[text] += 1


def write_run_config(root: Path, tracker: BotSortConfig, archive_root: Path) -> None:
    _write_json(
        root / "config" / "run_config.json",
        {
            "experiment": EXPERIMENT_GMC,
            "compared_with": EXPERIMENT_NO_GMC,
            "training_performed": False,
            "yolo_inference_performed": False,
            "detections_reused_from": str(archive_root / "detections"),
            "detector_that_produced_detections": DetectorConfig().as_dict(),
            "tracker": {
                "implementation": "ultralytics.trackers.bot_sort.BOTSORT",
                "settings": tracker.as_dict(),
                "settings_changed_from_no_gmc_run": False,
            },
            "gmc": {
                "method": tracker.gmc_method,
                "failure_policy": (
                    "Any exception, malformed warp, skipped frame or run of more "
                    "than 5 identity warps after frame 1 stops the run."
                ),
                "warp_convention": (
                    "2x3 affine mapping 0-based full-resolution pixels of the "
                    "previous frame to the current frame."
                ),
                "boxes_note": (
                    "Boxes are passed to GMC, but the installed sparseOptFlow "
                    "method does not use them, so pedestrians are not masked."
                ),
            },
        },
    )
    _write_json(
        root / "config" / "experiment_labels.json",
        {
            EXPERIMENT_NO_GMC: {
                "tracker_files": str(archive_root / "trackers" / TRACKER_NAME / "data"),
                "note": "Archived run in which GMC failed on all 5,316 frames.",
            },
            EXPERIMENT_GMC: {
                "tracker_files": str(root / "trackers" / EXPERIMENT_GMC / "data"),
            },
        },
    )


def run_tracking(
    sequences: Sequence[SequenceInfo],
    root: Path,
    archive_root: Path,
    tracker: BotSortConfig,
) -> dict[str, Any]:
    """Track every sequence strictly and stop at the first GMC failure."""
    counter = GMCWarningCounter()
    logging.getLogger("ultralytics").addHandler(counter)
    process = psutil.Process()
    rows = []
    diagnostics = []
    failure = None
    try:
        for sequence in sequences:
            recorder = GMCRecorder(sequence.name)
            started = time.perf_counter()
            status = {"sequence": sequence.name, "frames_expected": sequence.length}
            try:
                meta = track_sequence(
                    sequence,
                    archive_root / "detections",
                    root / "trackers" / EXPERIMENT_GMC,
                    tracker,
                    recorder,
                )
                status["status"] = "complete"
                status["output_boxes"] = meta["output_boxes"]
                status["track_ids"] = meta["track_ids"]
            except GMCPipelineFailure as error:
                status["status"] = "failed"
                failure = str(error)
            seconds = time.perf_counter() - started
            recorder.write_warps(root / "gmc" / f"{sequence.name}_warps.csv")
            summary = recorder.summary()
            diagnostics.append(summary)
            status.update(
                frames_with_gmc_result=summary["successful_calls"],
                seconds=seconds,
                seconds_per_frame=seconds / sequence.length,
                peak_working_set_mb=process.memory_info().peak_wset / 2**20,
            )
            rows.append(status)
            print(status, flush=True)
            if failure:
                break
    finally:
        logging.getLogger("ultralytics").removeHandler(counter)
    frames_expected = sum(sequence.length for sequence in sequences)
    totals = {
        "calls_expected": frames_expected,
        "calls": sum(row["calls"] for row in diagnostics),
        "successful_calls": sum(row["successful_calls"] for row in diagnostics),
        "failed_calls": sum(row["failed_calls"] for row in diagnostics),
        "estimated_warps": sum(row["estimated_warps"] for row in diagnostics),
        "first_frame_identity_warps": sum(row["first_frame_identity_warps"] for row in diagnostics),
        "non_first_identity_warps": sum(row["non_first_identity_warps"] for row in diagnostics),
        "ultralytics_gmc_warnings": counter.counts,
    }
    complete = (
        failure is None
        and len(rows) == len(sequences)
        and totals["calls"] == frames_expected
        and totals["failed_calls"] == 0
        and totals["successful_calls"] == frames_expected
        and not any(counter.counts.values())
    )
    _write_json(
        root / "gmc" / "diagnostics.json",
        {
            "experiment": EXPERIMENT_GMC,
            "totals": totals,
            "complete_without_failure": complete,
            "failure": failure,
            "sequences": diagnostics,
        },
    )
    _write_json(
        root / "config" / "run_status.json",
        {
            "experiment": EXPERIMENT_GMC,
            "finished_utc": datetime.now(timezone.utc).isoformat(),
            "all_sequences_complete": complete,
            "failure": failure,
            "sequences": rows,
        },
    )
    if not complete:
        raise SystemExit(f"Tracking did not complete cleanly: {failure or totals}")
    return totals


def stage_verify(sequences: Sequence[SequenceInfo], root: Path) -> list[dict[str, Any]]:
    """Check the saved warps against independent image and annotation evidence."""
    results = []
    for sequence in sequences:
        warps = load_warps(root / "gmc" / f"{sequence.name}_warps.csv")
        result = verify_sequence(sequence, warps)
        results.append(result)
        print(result, flush=True)
    _write_json(
        root / "gmc" / "verification.json",
        {"experiment": EXPERIMENT_GMC, "sequences": results},
    )
    return results


def check_pilot(root: Path, results: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Apply the pilot criteria declared in this module."""
    by_name = {row["sequence"]: row for row in results}
    static = by_name["MOT17-04"]
    moving = by_name["MOT17-10"]
    diagnostics = json.loads((root / "gmc" / "diagnostics.json").read_text(encoding="utf-8"))
    checks = {
        "all_calls_successful": diagnostics["complete_without_failure"],
        "no_non_first_identity": diagnostics["totals"]["non_first_identity_warps"] == 0,
        "static_motion_small": static["median_warp_center_motion_pixels"] < PILOT_MAX_STATIC_MEDIAN_MOTION,
        "moving_motion_present": moving["median_warp_center_motion_pixels"] > PILOT_MIN_MOVING_MEDIAN_MOTION,
        "moving_photometric_improved": moving["share_of_frame_pairs_improved"] >= PILOT_MIN_MOVING_IMPROVED_SHARE,
        "moving_matches_phase_correlation": (
            min(moving["correlation_x"] or 0.0, moving["correlation_y"] or 0.0)
            >= PILOT_MIN_MOVING_CORRELATION
        ),
        "moving_static_objects_steadier": (
            moving["static_objects"]["median_stabilized_spread_pixels"]
            < moving["static_objects"]["median_raw_spread_pixels"]
        ),
    }
    outcome = {"passed": all(checks.values()), "checks": checks}
    _write_json(root / "gmc" / "pilot_gate.json", outcome)
    return outcome


def stage_control(
    sequences: Sequence[SequenceInfo],
    root: Path,
    archive_root: Path,
    tracker: BotSortConfig,
) -> dict[str, Any]:
    """Rerun with GMC off and require byte-identical archived tracker files.

    This shows the code changes alter nothing except GMC. It is a check, not a
    third experiment, and its outputs are not evaluated.
    """
    control_dir = root / "control_no_gmc" / "trackers" / "control"
    control_config = replace(tracker, gmc_method="none")
    rows = []
    for sequence in sequences:
        track_sequence(sequence, archive_root / "detections", control_dir, control_config)
        new = file_sha256(control_dir / "data" / f"{sequence.name}.txt")
        old = file_sha256(
            archive_root / "trackers" / TRACKER_NAME / "data" / f"{sequence.name}.txt"
        )
        rows.append({"sequence": sequence.name, "sha256": new, "identical_to_archive": new == old})
        print(rows[-1], flush=True)
    outcome = {
        "purpose": "GMC-off control; proves other code changes did not alter tracking.",
        "all_identical": all(row["identical_to_archive"] for row in rows),
        "sequences": rows,
    }
    _write_json(root / "control_no_gmc" / "control_result.json", outcome)
    return outcome


def _detailed_metrics(path: Path) -> dict[str, dict[str, float]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    metrics = {}
    for row in rows:
        values = {}
        for key in COMPARED_METRICS:
            source = f"{key}___AUC" if f"{key}___AUC" in row else key
            values[key] = float(row[source])
        metrics[COMBINED_KEY if row["seq"] == "COMBINED" else row["seq"]] = values
    return metrics


def _profile_summary(path: Path) -> dict[str, Any]:
    profile = json.loads(path.read_text(encoding="utf-8"))
    overall = profile["overall"]
    rows = {
        "overall": {
            "iou_median": overall["iou"]["median"],
            "center_error_px_median": overall["center_error_px"]["median"],
            "center_error_px_p95": overall["center_error_px"]["p95"],
            "center_error_over_height_median": overall["center_error_over_height"]["median"],
            "coverage_median": overall["per_pedestrian_coverage"]["median"],
            "interior_gaps": overall["interior_gap_frames"]["count"],
            "interior_gap_seconds_median": overall["interior_gap_seconds"]["median"],
            "interior_gap_seconds_p95": overall["interior_gap_seconds"]["p95"],
            "window_availability": overall["window_availability"],
        }
    }
    for sequence in profile["sequences"]:
        rows[sequence["sequence"]] = {
            "iou_median": sequence["iou"]["median"],
            "center_error_px_median": sequence["center_error_px"]["median"],
            "center_error_px_p95": sequence["center_error_px"]["p95"],
            "center_error_over_height_median": sequence["center_error_over_height"]["median"],
            "track_span_seconds_median": sequence["track_span_seconds"]["median"],
            "cross_check_against_trackeval": sequence["cross_check_against_trackeval"],
        }
    return rows


def _availability(path: Path) -> dict[str, dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return {
            f"{row['sequence']}|{row['window_seconds']}": row
            for row in csv.DictReader(stream)
        }


def stage_compare(root: Path, archive_root: Path) -> dict[str, Any]:
    """Build the paired tables used by the ablation report."""
    old = _detailed_metrics(archive_root / "trackeval" / TRACKER_NAME / "pedestrian_detailed.csv")
    new = _detailed_metrics(root / "trackeval" / EXPERIMENT_GMC / "pedestrian_detailed.csv")
    deltas = {
        name: {
            key: {"no_gmc": old[name][key], "gmc": new[name][key], "delta": new[name][key] - old[name][key]}
            for key in COMPARED_METRICS
        }
        for name in new
    }
    tables = {
        "experiments": {"before": EXPERIMENT_NO_GMC, "after": EXPERIMENT_GMC},
        "metrics": deltas,
        "profile": {
            "no_gmc": _profile_summary(archive_root / "error_profiles" / "error_profile.json"),
            "gmc": _profile_summary(root / "error_profiles" / "error_profile.json"),
        },
        "availability": {
            "no_gmc": _availability(archive_root / "error_profiles" / "window_availability.csv"),
            "gmc": _availability(root / "error_profiles" / "window_availability.csv"),
        },
    }
    _write_json(root / "reports" / "comparison_tables.json", tables)
    return tables


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stages",
        nargs="+",
        choices=("control", "pilot", "track", "verify", "evaluate", "analyze", "visualize", "compare"),
    )
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_GMC_ROOT)
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)

    dataset_root = ensure_dataset_root(args.dataset_root)
    root = ensure_gmc_output_root(args.output_root)
    archive_root = root.parent
    tracker = BotSortConfig()
    sequences = discover_sequences(dataset_root)
    expected = sum(sequence.length for sequence in sequences)
    if expected != EXPECTED_TOTAL_FRAMES:
        raise SystemExit(f"Expected {EXPECTED_TOTAL_FRAMES} frames, found {expected}")

    record_environment(root)
    write_run_config(root, tracker, archive_root)
    with (root / "config" / "commands.log").open("a", encoding="utf-8") as stream:
        stream.write(f"{datetime.now(timezone.utc).isoformat()} {' '.join(sys.argv)}\n")

    tracker_args = {"tracker_name": EXPERIMENT_GMC, "experiment": EXPERIMENT_GMC}
    for stage in args.stages:
        print(f"== {stage}", flush=True)
        if stage == "control":
            outcome = stage_control(sequences, root, archive_root, tracker)
            if not outcome["all_identical"]:
                raise SystemExit("Control differs from the archived no-GMC tracker files.")
        elif stage == "pilot":
            pilot_root = root / "pilot"
            pilot = [s for s in sequences if s.name in PILOT_SEQUENCES]
            print(run_tracking(pilot, pilot_root, archive_root, tracker))
            outcome = check_pilot(pilot_root, stage_verify(pilot, pilot_root))
            print(json.dumps(outcome, indent=2))
            for sequence in pilot:
                print(
                    render_frame_panel(
                        sequence,
                        archive_root / "detections" / f"{sequence.name}.txt",
                        pilot_root / "trackers" / EXPERIMENT_GMC / "data" / f"{sequence.name}.txt",
                        pilot_root / "visualizations" / "frames",
                    )
                )
            if not outcome["passed"]:
                raise SystemExit("Pilot failed; the full run must not proceed.")
        elif stage == "track":
            gate = root / "pilot" / "gmc" / "pilot_gate.json"
            if not gate.is_file() or not json.loads(gate.read_text(encoding="utf-8"))["passed"]:
                raise SystemExit("Full run requires a passed pilot.")
            print(run_tracking(sequences, root, archive_root, tracker))
        elif stage == "verify":
            stage_verify(sequences, root)
        elif stage == "evaluate":
            stage_evaluate(sequences, root, **tracker_args)
        elif stage == "analyze":
            stage_analyze(
                sequences,
                root,
                detections_dir=archive_root / "detections",
                warps_dir=root / "gmc",
                **tracker_args,
            )
        elif stage == "visualize":
            for path in render_all(
                sequences,
                root,
                detections_dir=archive_root / "detections",
                tracker_name=EXPERIMENT_GMC,
            ):
                print(path)
        elif stage == "compare":
            stage_compare(root, archive_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
