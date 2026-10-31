"""Load the allowlisted saved MOT17 tracking reports.

MOT17 is used only for real tracking evaluation, and this module never reads
ETH results. ``BoT-SORT_GMC_enabled`` is the primary experiment and
``BoT-SORT_no_GMC`` is its ablation comparison. Their metrics are never pooled.
"""

from __future__ import annotations

import csv
import logging
import math
import statistics
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from trackvision.api.result_repository import (
    ArtifactInvalid,
    ArtifactNotFound,
    UnknownExperiment,
    _cached_json,
    _finite_number,
)
from trackvision.api.schemas import (
    Mot17AblationResponse,
    Mot17ConfigurationResponse,
    Mot17ErrorProfileResponse,
    Mot17GmcResponse,
    Mot17SequencesResponse,
    Mot17SummaryResponse,
)

logger = logging.getLogger(__name__)

ARCHIVE_ROOT = Path("mot17_tracking_analysis")
GMC_ROOT = ARCHIVE_ROOT / "gmc_fixed"
SEQUENCES = (
    "MOT17-02",
    "MOT17-04",
    "MOT17-05",
    "MOT17-09",
    "MOT17-10",
    "MOT17-11",
    "MOT17-13",
)
COMBINED = "COMBINED"
REGRESSION_SEQUENCE = "MOT17-11"
COORDINATE_UNITS = "image pixels; not metres or world coordinates"
SCOPE_NOTE = (
    "Real tracking evaluation on the seven MOT17 training sequences. This is "
    "not a standard MOT17 test-set result, and no MOT17 prediction experiment "
    "has been run."
)
# Median centre motion per frame below this is reported as static-looking.
STATIC_MOTION_THRESHOLD_PIXELS = 1.0
FRACTION_METRICS = (
    "HOTA", "DetA", "AssA", "LocA", "DetRe", "DetPr", "AssRe", "AssPr",
)
PLAIN_METRICS = (
    "MOTA", "MOTP", "IDF1", "IDR", "IDP", "IDSW", "CLR_TP", "CLR_FP", "CLR_FN",
    "Frag", "MT", "PT", "ML", "Dets", "GT_Dets", "IDs", "GT_IDs",
)
AGREEMENT_TOLERANCE = 1e-9
IOU_BINS = tuple(round(0.5 + index * 0.05, 2) for index in range(11))
CENTER_ERROR_BINS = tuple(round(index * 0.025, 3) for index in range(11))
COVERAGE_BINS = tuple(round(index * 0.1, 1) for index in range(11))
GAP_SECONDS_BINS = (0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
TRACK_SECONDS_BINS = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0)
FIGURES = {
    "localization_errors": "localization_errors.png",
    "per_sequence_metrics": "per_sequence_metrics.png",
    "track_lengths_and_missing_frames": "track_lengths_and_missing_frames.png",
}
FRAME_PANELS = {
    "MOT17-02": "frames/MOT17-02_frame000300.png",
    "MOT17-04": "frames/MOT17-04_frame000525.png",
    "MOT17-05": "frames/MOT17-05_frame000418.png",
    "MOT17-09": "frames/MOT17-09_frame000262.png",
    "MOT17-10": "frames/MOT17-10_frame000327.png",
    "MOT17-11": "frames/MOT17-11_frame000450.png",
    "MOT17-13": "frames/MOT17-13_frame000375.png",
}
CSV_COLUMNS = {
    "gt_track_continuity.csv": ("sequence", "coverage"),
    "matched_localization.csv": ("sequence", "iou", "center_error_over_height"),
    "missing_frame_distribution.csv": ("sequence", "kind", "length_seconds"),
    "predicted_tracks.csv": ("sequence", "observed_frames", "span_seconds"),
    "window_availability.csv": (
        "sequence", "window_seconds", "possible_windows", "available_windows",
    ),
    "identity_run_survival.csv": ("sequence", "horizon_seconds", "survival"),
}


@dataclass(frozen=True)
class Experiment:
    """Where one saved MOT17 experiment lives."""

    label: str
    root: Path
    trackeval_name: str
    tracker_name: str


EXPERIMENTS = {
    "gmc_enabled": Experiment(
        "BoT-SORT_GMC_enabled", GMC_ROOT, "BoT-SORT_GMC_enabled", "BoT-SORT_GMC_enabled"
    ),
    "no_gmc": Experiment("BoT-SORT_no_GMC", ARCHIVE_ROOT, "botsort", "botsort"),
}


@lru_cache(maxsize=32)
def _cached_csv(path: Path, required: tuple[str, ...]) -> tuple[dict[str, str], ...]:
    try:
        with path.open(newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            if not set(required) <= set(reader.fieldnames or ()):
                raise ArtifactInvalid(f"{path.name} is missing required columns.")
            return tuple(reader)
    except FileNotFoundError as error:
        raise ArtifactNotFound from error
    except (OSError, csv.Error, UnicodeDecodeError) as error:
        logger.exception("Could not read a saved MOT17 table.")
        raise ArtifactInvalid from error


def _histogram(values: list[float], edges: tuple[float, ...]) -> list[dict[str, Any]]:
    """Count values into edge-defined bins; values past the last edge overflow."""
    counts = [0] * len(edges)
    for value in values:
        index = len(edges) - 1
        for position in range(len(edges) - 1):
            if value < edges[position + 1]:
                index = position
                break
        counts[index] += 1
    rows = []
    for position, count in enumerate(counts):
        is_last = position == len(edges) - 1
        rows.append(
            {
                "lower": edges[position],
                "upper": None if is_last else edges[position + 1],
                "count": count,
            }
        )
    return rows


def _closed_histogram(
    values: list[float], edges: tuple[float, ...]
) -> list[dict[str, Any]]:
    """Histogram on a closed range: the final bin includes the top edge."""
    rows = _histogram(values, edges)
    last = rows.pop()
    rows[-1]["count"] += last["count"]
    return rows


def _distribution(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "median": None, "p95": None}
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "median": statistics.median(ordered),
        "p95": ordered[min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)],
    }


class Mot17Repository:
    """Read a fixed set of saved MOT17 artifacts beneath the runs root."""

    def __init__(self, results_root: Path) -> None:
        self.results_root = Path(results_root).expanduser().resolve()

    @staticmethod
    def _experiment(key: str) -> Experiment:
        if key not in EXPERIMENTS:
            raise UnknownExperiment
        return EXPERIMENTS[key]

    def _safe_path(self, relative_path: Path) -> Path:
        path = (self.results_root / relative_path).resolve()
        try:
            path.relative_to(self.results_root)
        except ValueError as error:
            raise UnknownExperiment from error
        return path

    def _json(self, relative_path: Path) -> dict[str, Any]:
        return _cached_json(self._safe_path(relative_path))

    def _table(self, experiment: Experiment, name: str) -> tuple[dict[str, str], ...]:
        return _cached_csv(
            self._safe_path(experiment.root / "error_profiles" / name),
            CSV_COLUMNS[name],
        )

    def _verified_primary_run(self) -> tuple[dict[str, Any], dict[str, Any]]:
        """Refuse to present the primary run unless its records show it finished."""
        status = self._json(GMC_ROOT / "config" / "run_status.json")
        diagnostics = self._json(GMC_ROOT / "gmc" / "diagnostics.json")
        if status.get("all_sequences_complete") is not True:
            raise ArtifactInvalid("The GMC run did not complete all sequences.")
        if diagnostics.get("complete_without_failure") is not True:
            raise ArtifactInvalid("The GMC run recorded a failure.")
        return status, diagnostics

    def _metrics(self, key: str) -> dict[str, dict[str, float]]:
        """Return TrackEval metrics by sequence, from the per-threshold table."""
        experiment = self._experiment(key)
        path = self._safe_path(
            experiment.root
            / "trackeval"
            / experiment.trackeval_name
            / "pedestrian_detailed.csv"
        )
        rows = _cached_csv(path, ("seq", "HOTA___AUC", "MOTA", "IDF1", "IDSW"))
        metrics: dict[str, dict[str, float]] = {}
        for row in rows:
            try:
                values = {
                    name: _finite_number(float(row[f"{name}___AUC"]), name)
                    for name in FRACTION_METRICS
                }
                values.update(
                    {
                        name: _finite_number(float(row[name]), name)
                        for name in PLAIN_METRICS
                    }
                )
            except (KeyError, TypeError, ValueError) as error:
                raise ArtifactInvalid(f"{experiment.label} metrics are incomplete.") from error
            metrics[row["seq"]] = values
        if set(metrics) != {*SEQUENCES, COMBINED}:
            raise ArtifactInvalid(f"{experiment.label} metric rows are incomplete.")
        saved = self._json(experiment.root / "trackeval" / "metrics.json")
        try:
            combined = saved["metrics"]["COMBINED_SEQ"]
            for name in ("HOTA", "MOTA", "IDF1"):
                if abs(combined[name] - metrics[COMBINED][name]) > AGREEMENT_TOLERANCE:
                    raise ArtifactInvalid(f"{experiment.label} saved metrics disagree.")
        except (KeyError, TypeError) as error:
            raise ArtifactInvalid(f"{experiment.label} metrics.json is incomplete.") from error
        return metrics

    def _inventory(self) -> dict[str, Any]:
        inventory = self._json(ARCHIVE_ROOT / "config" / "dataset_inventory.json")
        names = [row.get("sequence") for row in inventory.get("sequences", [])]
        if names != list(SEQUENCES) or inventory.get("split") != "train":
            raise ArtifactInvalid("The MOT17 inventory does not list the expected sequences.")
        return inventory

    def _verification(self) -> dict[str, dict[str, Any]]:
        report = self._json(GMC_ROOT / "gmc" / "verification.json")
        rows = {row["sequence"]: row for row in report.get("sequences", [])}
        if set(rows) != set(SEQUENCES):
            raise ArtifactInvalid("GMC verification does not cover all sequences.")
        return rows

    def _camera_class(self, verification: dict[str, Any]) -> str:
        motion = _finite_number(
            verification["median_warp_center_motion_pixels"], "camera motion"
        )
        return "static-looking" if motion < STATIC_MOTION_THRESHOLD_PIXELS else "moving"

    def summary(self) -> Mot17SummaryResponse:
        status, diagnostics = self._verified_primary_run()
        inventory = self._inventory()
        return Mot17SummaryResponse(
            dataset="mot17",
            split="train",
            role="Real multi-object tracking evaluation",
            primary_experiment=EXPERIMENTS["gmc_enabled"].label,
            comparison_experiment=EXPERIMENTS["no_gmc"].label,
            sequence_count=inventory["sequence_count"],
            total_frames=inventory["total_frames"],
            metrics=self._metrics("gmc_enabled")[COMBINED],
            comparison_metrics=self._metrics("no_gmc")[COMBINED],
            all_sequences_complete=status["all_sequences_complete"],
            gmc_failed_calls=diagnostics["totals"]["failed_calls"],
            gmc_calls=diagnostics["totals"]["calls"],
            coordinate_units=COORDINATE_UNITS,
            scope_note=SCOPE_NOTE,
            provenance=[
                (GMC_ROOT / "trackeval" / "metrics.json").as_posix(),
                (GMC_ROOT / "config" / "run_status.json").as_posix(),
            ],
        )

    def sequences(self) -> Mot17SequencesResponse:
        inventory = self._inventory()
        primary = self._metrics("gmc_enabled")
        comparison = self._metrics("no_gmc")
        verification = self._verification()
        rows = []
        for entry in inventory["sequences"]:
            name = entry["sequence"]
            rows.append(
                {
                    "sequence": name,
                    "canonical_directory": entry["canonical_directory"],
                    "frames": entry["frames"],
                    "frame_rate": float(entry["frame_rate"]),
                    "width": entry["width"],
                    "height": entry["height"],
                    "gt_rows": entry["gt_rows"],
                    "evaluated_gt_boxes": entry["evaluated_gt_boxes"],
                    "evaluated_gt_identities": entry["evaluated_gt_identities"],
                    "variants_agree": entry["variants_agree"],
                    "camera_class": self._camera_class(verification[name]),
                    "metrics": primary[name],
                    "comparison_metrics": comparison[name],
                }
            )
        return Mot17SequencesResponse(
            dataset="mot17",
            split="train",
            primary_experiment=EXPERIMENTS["gmc_enabled"].label,
            comparison_experiment=EXPERIMENTS["no_gmc"].label,
            canonical_variant=inventory["canonical_variant"],
            variant_policy=inventory["variant_policy"],
            total_frames=inventory["total_frames"],
            camera_class_rule=(
                "Static-looking means median warp motion below "
                f"{STATIC_MOTION_THRESHOLD_PIXELS:g} pixel per frame."
            ),
            sequences=rows,
            provenance=[
                (ARCHIVE_ROOT / "config" / "dataset_inventory.json").as_posix(),
                (GMC_ROOT / "gmc" / "verification.json").as_posix(),
            ],
        )

    def configuration(self) -> Mot17ConfigurationResponse:
        config = self._json(GMC_ROOT / "config" / "run_config.json")
        archive_config = self._json(ARCHIVE_ROOT / "config" / "run_config.json")
        weights = self._json(ARCHIVE_ROOT / "config" / "weights_fingerprint.json")
        try:
            return Mot17ConfigurationResponse(
                experiment=config["experiment"],
                training_performed=config["training_performed"],
                detector=config["detector_that_produced_detections"],
                weights_sha256=weights["sha256"],
                person_class_id=weights["person_class_id"],
                tracker_implementation=config["tracker"]["implementation"],
                tracker_settings=config["tracker"]["settings"],
                settings_changed_from_comparison=config["tracker"][
                    "settings_changed_from_no_gmc_run"
                ],
                gmc=config["gmc"],
                evaluation=archive_config["evaluation"],
                provenance=[
                    (GMC_ROOT / "config" / "run_config.json").as_posix(),
                    (ARCHIVE_ROOT / "config" / "weights_fingerprint.json").as_posix(),
                ],
            )
        except KeyError as error:
            raise ArtifactInvalid("The MOT17 run configuration is incomplete.") from error

    def gmc(self) -> Mot17GmcResponse:
        _, diagnostics = self._verified_primary_run()
        verification = self._verification()
        control = self._json(GMC_ROOT / "control_no_gmc" / "control_result.json")
        rows = []
        for row in diagnostics["sequences"]:
            checks = verification[row["sequence"]]
            rows.append(
                {
                    **row,
                    "camera_class": self._camera_class(checks),
                    "median_warp_center_motion_pixels": checks[
                        "median_warp_center_motion_pixels"
                    ],
                    "median_phase_motion_pixels": checks["median_phase_motion_pixels"],
                    "correlation_x": checks["correlation_x"],
                    "correlation_y": checks["correlation_y"],
                    "share_of_frame_pairs_improved": checks[
                        "share_of_frame_pairs_improved"
                    ],
                    "static_objects": checks["static_objects"],
                }
            )
        return Mot17GmcResponse(
            experiment=diagnostics["experiment"],
            method=self._json(GMC_ROOT / "config" / "run_config.json")["gmc"]["method"],
            totals=diagnostics["totals"],
            complete_without_failure=diagnostics["complete_without_failure"],
            control_identical_to_no_gmc=control["all_identical"],
            sequences=rows,
            provenance=[
                (GMC_ROOT / "gmc" / "diagnostics.json").as_posix(),
                (GMC_ROOT / "gmc" / "verification.json").as_posix(),
                (GMC_ROOT / "control_no_gmc" / "control_result.json").as_posix(),
            ],
        )

    def error_profile(self, key: str) -> Mot17ErrorProfileResponse:
        experiment = self._experiment(key)
        profile = self._json(experiment.root / "error_profiles" / "error_profile.json")
        overall = profile.get("overall")
        if not isinstance(overall, dict):
            raise ArtifactInvalid(f"{experiment.label} error profile is incomplete.")

        localization = self._table(experiment, "matched_localization.csv")
        continuity = self._table(experiment, "gt_track_continuity.csv")
        gaps = self._table(experiment, "missing_frame_distribution.csv")
        tracks = self._table(experiment, "predicted_tracks.csv")
        availability = self._table(experiment, "window_availability.csv")
        survival = self._table(experiment, "identity_run_survival.csv")

        coverage = [float(row["coverage"]) for row in continuity]
        interior_gaps = [
            float(row["length_seconds"])
            for row in gaps
            if row["kind"] == "gt_interior_gap"
        ]
        track_seconds = [float(row["span_seconds"]) for row in tracks]
        pooled: dict[float, dict[str, int]] = {}
        for row in availability:
            totals = pooled.setdefault(
                float(row["window_seconds"]), {"possible": 0, "available": 0}
            )
            totals["possible"] += int(row["possible_windows"])
            totals["available"] += int(row["available_windows"])

        return Mot17ErrorProfileResponse(
            dataset="mot17",
            experiment=experiment.label,
            kind=profile["kind"],
            coordinate_units=COORDINATE_UNITS,
            matching=profile["matching"],
            definitions=profile["definitions"],
            limitations=profile["limitations"],
            overall={
                name: overall[name]
                for name in (
                    "true_positives", "false_positives", "false_negatives",
                    "identity_switches", "fragmentations", "recall", "precision",
                    "detector_conf_0.25", "detector_conf_0.10", "iou",
                    "center_error_px", "center_error_over_height",
                    "interior_gap_frames", "interior_gap_seconds",
                    "per_pedestrian_coverage",
                )
            },
            coverage_counts={
                "pedestrians": len(coverage),
                "zero": sum(value == 0 for value in coverage),
                "below_5_percent": sum(value < 0.05 for value in coverage),
                "below_20_percent": sum(value < 0.2 for value in coverage),
                "above_80_percent": sum(value > 0.8 for value in coverage),
            },
            sequences=[
                {
                    "sequence": row["sequence"],
                    "recall": row["tracker"]["recall"],
                    "precision": row["tracker"]["precision"],
                    "median_iou": row["iou"]["median"],
                    "median_center_error_px": row["center_error_px"]["median"],
                    "p95_center_error_px": row["center_error_px"]["p95"],
                    "median_center_error_over_height": row["center_error_over_height"]["median"],
                    "median_track_span_seconds": row["track_span_seconds"]["median"],
                }
                for row in profile["sequences"]
            ],
            iou_histogram=_closed_histogram(
                [float(row["iou"]) for row in localization], IOU_BINS
            ),
            center_error_histogram=_histogram(
                [float(row["center_error_over_height"]) for row in localization],
                CENTER_ERROR_BINS,
            ),
            coverage_histogram=_closed_histogram(coverage, COVERAGE_BINS),
            gap_histogram=_histogram(interior_gaps, GAP_SECONDS_BINS),
            track_duration_histogram=_histogram(track_seconds, TRACK_SECONDS_BINS),
            track_duration_seconds=_distribution(track_seconds),
            availability=[
                {
                    "window_seconds": seconds,
                    "possible_windows": totals["possible"],
                    "available_windows": totals["available"],
                    "availability": (
                        totals["available"] / totals["possible"]
                        if totals["possible"]
                        else None
                    ),
                }
                for seconds, totals in sorted(pooled.items())
            ],
            availability_by_sequence=[
                {
                    "sequence": row["sequence"],
                    "window_seconds": float(row["window_seconds"]),
                    "possible_windows": int(row["possible_windows"]),
                    "available_windows": int(row["available_windows"]),
                    "availability": (
                        float(row["availability"]) if row["availability"] else None
                    ),
                }
                for row in availability
            ],
            survival=[
                {
                    "sequence": row["sequence"],
                    "horizon_seconds": float(row["horizon_seconds"]),
                    "survival": float(row["survival"]),
                }
                for row in survival
            ],
            provenance=[
                (experiment.root / "error_profiles" / "error_profile.json").as_posix(),
                (experiment.root / "error_profiles").as_posix(),
            ],
        )

    def _regression(
        self,
        primary: dict[str, dict[str, float]],
        comparison: dict[str, dict[str, float]],
    ) -> dict[str, Any]:
        """Report the saved evidence for the sequence that got worse with GMC."""
        name = REGRESSION_SEQUENCE
        boxes = {}
        lengths = {}
        for key in ("no_gmc", "gmc_enabled"):
            experiment = EXPERIMENTS[key]
            meta = self._json(
                experiment.root / "trackers" / experiment.tracker_name / f"{name}.meta.json"
            )
            boxes[key] = meta["output_boxes"]
            lengths[key] = statistics.median(
                int(row["observed_frames"])
                for row in self._table(experiment, "predicted_tracks.csv")
                if row["sequence"] == name
            )
        checks = self._verification()[name]
        return {
            "sequence": name,
            "output_boxes": boxes,
            "median_track_length_frames": lengths,
            "delta": {
                metric: primary[name][metric] - comparison[name][metric]
                for metric in ("HOTA", "MOTA", "IDF1", "CLR_TP", "CLR_FP", "CLR_FN", "IDSW")
            },
            "warp_correlation_x": checks["correlation_x"],
            "warp_correlation_y": checks["correlation_y"],
            "share_of_frame_pairs_improved": checks["share_of_frame_pairs_improved"],
            "false_positive_tracks": self._false_positive_tracks(name),
        }

    def _false_positive_tracks(self, name: str, limit: int = 6) -> list[dict[str, Any]]:
        rows = _cached_csv(
            self._safe_path(GMC_ROOT / "trajectories" / f"{name}_botsort.csv"),
            ("frame", "track_id", "foot_x", "foot_y", "stabilized_foot_x",
             "stabilized_foot_y", "match_status", "confidence"),
        )
        tracks: dict[str, list[dict[str, str]]] = {}
        for row in rows:
            tracks.setdefault(row["track_id"], []).append(row)
        summaries = []
        for track_id, track_rows in tracks.items():
            false_positive_rows = sum(
                row["match_status"] == "false_positive" for row in track_rows
            )
            if not false_positive_rows:
                continue
            first, last = track_rows[0], track_rows[-1]
            summaries.append(
                {
                    "track_id": int(track_id),
                    "false_positive_frames": false_positive_rows,
                    "total_frames": len(track_rows),
                    "median_confidence": statistics.median(
                        float(row["confidence"]) for row in track_rows
                    ),
                    "raw_travel_pixels": math.dist(
                        (float(first["foot_x"]), float(first["foot_y"])),
                        (float(last["foot_x"]), float(last["foot_y"])),
                    ),
                    "stabilized_travel_pixels": math.dist(
                        (float(first["stabilized_foot_x"]), float(first["stabilized_foot_y"])),
                        (float(last["stabilized_foot_x"]), float(last["stabilized_foot_y"])),
                    ),
                }
            )
        summaries.sort(key=lambda item: item["false_positive_frames"], reverse=True)
        return summaries[:limit]

    def ablation(self) -> Mot17AblationResponse:
        self._verified_primary_run()
        primary = self._metrics("gmc_enabled")
        comparison = self._metrics("no_gmc")
        verification = self._verification()
        control = self._json(GMC_ROOT / "control_no_gmc" / "control_result.json")
        rows = []
        for name in (*SEQUENCES, COMBINED):
            rows.append(
                {
                    "sequence": name,
                    "camera_class": (
                        None if name == COMBINED else self._camera_class(verification[name])
                    ),
                    "metrics": {
                        metric: {
                            "no_gmc": comparison[name][metric],
                            "gmc_enabled": primary[name][metric],
                            "delta": primary[name][metric] - comparison[name][metric],
                        }
                        for metric in primary[name]
                    },
                }
            )
        return Mot17AblationResponse(
            primary_experiment=EXPERIMENTS["gmc_enabled"].label,
            comparison_experiment=EXPERIMENTS["no_gmc"].label,
            control_identical_to_no_gmc=control["all_identical"],
            rows=rows,
            regression=self._regression(primary, comparison),
            provenance=[
                (GMC_ROOT / "trackeval" / EXPERIMENTS["gmc_enabled"].trackeval_name).as_posix(),
                (ARCHIVE_ROOT / "trackeval" / "botsort").as_posix(),
                (GMC_ROOT / "reports" / "gmc_ablation_comparison.md").as_posix(),
            ],
        )

    def visualization_path(self, key: str, figure: str) -> Path:
        experiment = self._experiment(key)
        relative = FIGURES.get(figure) or FRAME_PANELS.get(figure)
        if relative is None:
            raise UnknownExperiment
        path = self._safe_path(experiment.root / "visualizations" / relative)
        if not path.is_file():
            raise ArtifactNotFound
        return path
