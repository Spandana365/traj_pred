"""Load and normalize only the allowlisted saved validation reports."""

from __future__ import annotations

import json
import logging
import math
import os
import statistics
from functools import lru_cache
from pathlib import Path
from typing import Any

from trackvision.api.schemas import (
    BaselineResponse,
    DashboardSummaryResponse,
    ErrorExperimentSummary,
    ErrorExperimentsResponse,
    EthDatasetAuditResponse,
    EthFigure,
    EthFiguresResponse,
    ExperimentDetailResponse,
    ExperimentMetadata,
    MetricPair,
    MetricStats,
    SceneComparison,
    SceneMetrics,
    SeveritySummary,
    TimestepComparison,
    WindowStatistics,
)

logger = logging.getLogger(__name__)

SEEDS = (42, 43, 44)
SEVERITIES = ("low", "medium", "high")
BASELINE_RELATIVE_PATH = Path(
    "prediction/eth/lstm-seed42-h64-l2-scheduler/validation_results.json"
)
BASELINE_EXPERIMENT_RELATIVE_PATH = Path(
    "prediction/eth/lstm-seed42-h64-l2-scheduler/experiment.json"
)
EXPERIMENT_IDS = (
    "localization_noise",
    "detection_dropout",
    "track_fragmentation",
    "identity_switch",
)

VISUALIZATION_ALLOWLIST: dict[str, dict[str, str]] = {
    "localization_noise": {
        severity: "trajectory_noise_examples.png" for severity in SEVERITIES
    },
    "detection_dropout": {
        "low": "imputation-example-low.png",
        "medium": "imputation-example-medium.png",
        "high": "imputation-example-high.png",
    },
    "track_fragmentation": {
        "low": "fragmentation-example-low.png",
        "medium": "fragmentation-example-medium.png",
        "high": "fragmentation-example-high.png",
    },
    "identity_switch": {
        "low": "identity-switch-example-low.png",
        "medium": "identity-switch-example-medium.png",
        "high": "identity-switch-example-high.png",
    },
}

DATASET_AUDIT_RELATIVE_PATH = Path("audits/eth/dataset-audit.json")
# Fixed saved ETH summary figures: id -> (path below error_injection, title, group).
ETH_FIGURES: dict[str, tuple[str, str, str]] = {
    "lstm-localization-noise": (
        "lstm_comparison_plots/lstm_localization_noise_comparison.png",
        "LSTM clean vs localization noise", "LSTM comparison",
    ),
    "lstm-detection-dropout": (
        "lstm_comparison_plots/lstm_detection_dropout_comparison.png",
        "LSTM clean vs detection dropout", "LSTM comparison",
    ),
    "lstm-track-fragmentation": (
        "lstm_comparison_plots/lstm_track_fragmentation_comparison.png",
        "LSTM clean vs track fragmentation", "LSTM comparison",
    ),
    "lstm-identity-switch": (
        "lstm_comparison_plots/lstm_identity_switch_comparison.png",
        "LSTM clean vs identity switches", "LSTM comparison",
    ),
    "dropout-severity-metrics": (
        "detection_dropout/severity-vs-metrics.png",
        "Dropout severity vs metrics", "Detection dropout",
    ),
    "fragmentation-matched-clean": (
        "track_fragmentation/matched-clean-vs-fragmented-metrics.png",
        "Matched clean vs fragmented metrics", "Track fragmentation",
    ),
    "fragmentation-scene-availability": (
        "track_fragmentation/per-scene-window-availability.png",
        "Per-scene window availability", "Track fragmentation",
    ),
    "fragmentation-window-retention": (
        "track_fragmentation/severity-vs-window-retention.png",
        "Severity vs window retention", "Track fragmentation",
    ),
    "identity-affected-windows": (
        "identity_switch/affected-window-percentage.png",
        "Affected window percentage", "Identity switches",
    ),
    "identity-clean-vs-corrupted": (
        "identity_switch/clean-vs-corrupted-trajectory.png",
        "Clean vs corrupted trajectory", "Identity switches",
    ),
    "identity-scene-degradation": (
        "identity_switch/per-scene-prediction-degradation.png",
        "Per-scene prediction degradation", "Identity switches",
    ),
    "identity-severity-degradation": (
        "identity_switch/severity-vs-ade-fde-degradation.png",
        "Severity vs ADE and FDE degradation", "Identity switches",
    ),
}

_AFFECTED_FIELDS: dict[str, tuple[str, ...]] = {
    "localization_noise": (),
    "detection_dropout": (
        "actual_missing_percentage_unique",
        "actual_missing_percentage_window_occurrences",
        "dropped_unique_observation_count",
        "longest_consecutive_missing_run",
    ),
    "track_fragmentation": (
        "missing_sampled_observation_count",
        "future_horizon_crossing_fragment_boundary_count",
        "resulting_continuous_fragment_count",
    ),
    "identity_switch": (
        "affected_window_count",
        "affected_window_percentage",
        "unaffected_window_count",
        "unique_affected_pedestrian_count",
        "event_count",
        "selected_pair_count",
        "switched_observation_assignment_count",
    ),
}


class ArtifactNotFound(Exception):
    """A fixed, required report or allowlisted visualization is unavailable."""


class ArtifactInvalid(Exception):
    """A saved report is malformed or internally inconsistent."""


class UnknownExperiment(Exception):
    """The requested experiment or severity is not on the allowlist."""


@lru_cache(maxsize=32)
def _cached_json(path: Path) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as file:
            data = json.load(file)
    except FileNotFoundError as error:
        raise ArtifactNotFound from error
    except (OSError, json.JSONDecodeError) as error:
        logger.exception("Could not read a saved results report.")
        raise ArtifactInvalid from error
    if not isinstance(data, dict):
        raise ArtifactInvalid("Report root must be a JSON object.")
    return data


def _finite_number(value: Any, description: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ArtifactInvalid(f"{description} must be numeric.")
    result = float(value)
    if not math.isfinite(result):
        raise ArtifactInvalid(f"{description} must be finite.")
    return result


def _nonnegative_count(value: Any, description: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ArtifactInvalid(f"{description} must be a non-negative integer.")
    return value


def _stats(values: list[float] | list[int]) -> MetricStats:
    if not values:
        raise ArtifactInvalid("Cannot summarize an empty set of observations.")
    numeric = [float(value) for value in values]
    return MetricStats(
        mean=statistics.mean(numeric),
        population_std=statistics.pstdev(numeric) if len(numeric) > 1 else None,
    )


def _metric_pair(values: list[dict[str, float]]) -> MetricPair:
    return MetricPair(
        ade=_stats([value["ade"] for value in values]),
        fde=_stats([value["fde"] for value in values]),
    )


def _metric_container(
    report: dict[str, Any], experiment: str, severity: str, kind: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return a model's LSTM metrics and their containing signature object."""
    if experiment == "localization_noise":
        severity_result = report["severities"][severity]
        if kind == "matched_clean":
            parent = report["clean"]
        else:
            parent = severity_result
            parent = parent["metrics"]
    else:
        severity_result = report["severity_results"][severity]
        if experiment == "detection_dropout":
            parent = (
                severity_result["clean_baseline_on_retained_windows"]
                if kind == "matched_clean"
                else severity_result["metrics"]
            )
        elif experiment == "track_fragmentation":
            parent = (
                severity_result["clean_baseline_on_retained_windows"]
                if kind == "matched_clean"
                else severity_result["fragmented_input_metrics"]
            )
        else:
            parent = (
                severity_result["clean_metrics"]
                if kind == "matched_clean"
                else severity_result["corrupted_metrics"]
            )
    metrics = parent.get("lstm")
    if not isinstance(metrics, dict):
        raise ArtifactInvalid(f"{experiment} {severity} report has no LSTM metrics.")
    return metrics, parent


def _seed_observation(
    report: dict[str, Any], experiment: str, severity: str, seed: int
) -> dict[str, Any]:
    if report.get("seed") != seed or report.get("split") != "val":
        raise ArtifactInvalid(f"{experiment} report does not match validation seed {seed}.")

    clean, clean_parent = _metric_container(report, experiment, severity, "matched_clean")
    corrupted, corrupted_parent = _metric_container(
        report, experiment, severity, "corrupted"
    )
    clean_signature = clean.get(
        "window_key_sha256", clean_parent.get("window_key_sha256")
    )
    corrupted_signature = corrupted.get(
        "window_key_sha256", corrupted_parent.get("window_key_sha256")
    )
    if not isinstance(clean_signature, str) or not clean_signature:
        raise ArtifactInvalid(f"{experiment} matched-clean window signature is missing.")
    if clean_signature != corrupted_signature:
        raise ArtifactInvalid(
            f"{experiment} {severity} clean and corrupted windows do not match."
        )

    if experiment == "localization_noise":
        candidate = _nonnegative_count(
            report["clean"]["window_count"], "localization candidate count"
        )
        evaluated = _nonnegative_count(
            report["severities"][severity]["evaluated_windows"],
            "localization evaluated count",
        )
        unusable = candidate - evaluated
    elif experiment == "detection_dropout":
        candidate = _nonnegative_count(
            report["candidate_window_count"], "dropout candidate count"
        )
        severity_result = report["severity_results"][severity]
        evaluated = _nonnegative_count(
            severity_result["evaluated_window_count"], "dropout evaluated count"
        )
        unusable = _nonnegative_count(
            severity_result["unusable_window_count"], "dropout unusable count"
        )
    elif experiment == "track_fragmentation":
        severity_result = report["severity_results"][severity]
        candidate = _nonnegative_count(
            severity_result["candidate_window_count"], "fragmentation candidate count"
        )
        evaluated = _nonnegative_count(
            severity_result["retained_window_count"], "fragmentation retained count"
        )
        unusable = _nonnegative_count(
            severity_result["unusable_window_count"], "fragmentation unusable count"
        )
    else:
        candidate = _nonnegative_count(
            report["candidate_window_count"], "identity-switch candidate count"
        )
        evaluated = _nonnegative_count(
            corrupted["window_count"], "identity-switch evaluated count"
        )
        unusable = candidate - evaluated

    if candidate != evaluated + unusable or evaluated > candidate:
        raise ArtifactInvalid(f"{experiment} window counts are inconsistent.")
    for label, metrics in (("matched-clean", clean), ("corrupted", corrupted)):
        if "window_count" in metrics and metrics["window_count"] != evaluated:
            raise ArtifactInvalid(
                f"{experiment} {label} metric count does not match evaluated windows."
            )

    clean_values = {
        metric: _finite_number(clean[metric], f"{experiment} clean {metric}")
        for metric in ("ade", "fde")
    }
    corrupted_values = {
        metric: _finite_number(corrupted[metric], f"{experiment} corrupted {metric}")
        for metric in ("ade", "fde")
    }
    if experiment == "localization_noise":
        severity_result = report["severities"][severity]
    else:
        severity_result = report["severity_results"][severity]
    affected = {
        key: _finite_number(severity_result[key], f"{experiment} {key}")
        for key in _AFFECTED_FIELDS[experiment]
        if key in severity_result
        and isinstance(severity_result[key], (int, float))
        and not isinstance(severity_result[key], bool)
    }
    return {
        "seed": seed,
        "candidate_count": candidate,
        "evaluated_count": evaluated,
        "unusable_count": unusable,
        "clean": clean_values,
        "corrupted": corrupted_values,
        "clean_container": clean,
        "corrupted_container": corrupted,
        "affected": affected,
    }


def _scene_metrics(
    metric_containers: list[dict[str, Any]], scene: str
) -> SceneMetrics:
    scene_values = []
    scene_window_counts = []
    for container in metric_containers:
        per_scene = container.get("per_scene")
        if not isinstance(per_scene, dict) or scene not in per_scene:
            raise ArtifactInvalid(f"Per-scene metrics are incomplete for {scene}.")
        values = per_scene[scene]
        scene_values.append(
            {
                "ade": _finite_number(values["ade"], f"{scene} ADE"),
                "fde": _finite_number(values["fde"], f"{scene} FDE"),
            }
        )
        if "window_count" in values:
            scene_window_counts.append(
                _nonnegative_count(values["window_count"], f"{scene} window count")
            )
    if scene_window_counts and len(scene_window_counts) != len(scene_values):
        raise ArtifactInvalid(f"Per-scene window counts are incomplete for {scene}.")
    return SceneMetrics(
        ade=_stats([value["ade"] for value in scene_values]),
        fde=_stats([value["fde"] for value in scene_values]),
        window_count=_stats(scene_window_counts) if scene_window_counts else None,
    )


def _timestep_comparisons(
    matched_clean: list[dict[str, Any]] | None,
    corrupted: list[dict[str, Any]],
) -> list[TimestepComparison]:
    corrupted_rows = [item.get("ade_by_timestep") for item in corrupted]
    if any(not isinstance(rows, list) for rows in corrupted_rows):
        return []
    if len({len(rows) for rows in corrupted_rows}) != 1:
        raise ArtifactInvalid("Per-timestep ADE lengths differ across seeds.")
    clean_rows = (
        [item.get("ade_by_timestep") for item in matched_clean]
        if matched_clean is not None
        else None
    )
    if clean_rows is not None and (
        any(not isinstance(rows, list) for rows in clean_rows)
        or len({len(rows) for rows in clean_rows}) != 1
        or len(clean_rows[0]) != len(corrupted_rows[0])
    ):
        raise ArtifactInvalid("Matched-clean and corrupted timestep data do not align.")

    output = []
    for index, row in enumerate(corrupted_rows[0]):
        timestep = row.get("timestep")
        if not isinstance(timestep, int) or isinstance(timestep, bool):
            raise ArtifactInvalid("Timestep index must be an integer.")
        if any(rows[index].get("timestep") != timestep for rows in corrupted_rows):
            raise ArtifactInvalid("Timestep order differs across seeds.")
        corrupt_values = [
            _finite_number(rows[index]["ade"], "per-timestep corrupted ADE")
            for rows in corrupted_rows
        ]
        clean_values = (
            [
                _finite_number(rows[index]["ade"], "per-timestep matched-clean ADE")
                for rows in clean_rows
            ]
            if clean_rows is not None
            else None
        )
        output.append(
            TimestepComparison(
                timestep=timestep,
                matched_clean_ade=_stats(clean_values) if clean_values else None,
                corrupted_ade=_stats(corrupt_values),
                degradation_ade=(
                    _stats(
                        [
                            corrupt - clean
                            for corrupt, clean in zip(corrupt_values, clean_values)
                        ]
                    )
                    if clean_values
                    else None
                ),
            )
        )
    return output


class ResultRepository:
    """Read a fixed set of saved artifacts beneath the configured runs root."""

    def __init__(self, results_root: Path | None = None) -> None:
        project_root = Path(__file__).resolve().parents[3]
        configured_root = results_root or Path(
            os.environ.get("TRACKVISION_RESULTS_ROOT", project_root / "runs")
        )
        self.results_root = configured_root.expanduser().resolve()

    @staticmethod
    def _experiment_id(experiment: str) -> str:
        if experiment not in EXPERIMENT_IDS:
            raise UnknownExperiment
        return experiment

    @staticmethod
    def _severity_id(severity: str) -> str:
        if severity not in SEVERITIES:
            raise UnknownExperiment
        return severity

    def _read_json(self, relative_path: Path) -> dict[str, Any]:
        allowed = {
            BASELINE_RELATIVE_PATH,
            BASELINE_EXPERIMENT_RELATIVE_PATH,
            DATASET_AUDIT_RELATIVE_PATH,
        }
        for experiment in EXPERIMENT_IDS:
            allowed.add(Path("error_injection") / experiment / "aggregated_results.json")
            allowed.add(Path("error_injection") / experiment / "experiment_config.json")
            allowed.update(
                Path("error_injection")
                / experiment
                / "per_seed"
                / f"seed-{seed}.json"
                for seed in SEEDS
            )
        if relative_path not in allowed:
            raise UnknownExperiment
        return _cached_json(self.results_root / relative_path)

    def baseline_report(self) -> dict[str, Any]:
        report = self._read_json(BASELINE_RELATIVE_PATH)
        results = report.get("results")
        if report.get("split") != "val" or not isinstance(results, dict):
            raise ArtifactInvalid("The baseline report is not a validation results object.")
        return report

    def _baseline_experiment(self) -> dict[str, Any]:
        return self._read_json(BASELINE_EXPERIMENT_RELATIVE_PATH)

    def _aggregate(self, experiment: str) -> dict[str, Any]:
        experiment = self._experiment_id(experiment)
        report = self._read_json(
            Path("error_injection") / experiment / "aggregated_results.json"
        )
        if report.get("split") != "val":
            raise ArtifactInvalid(f"{experiment} report is not validation-only.")
        return report

    def _config(self, experiment: str) -> dict[str, Any]:
        experiment = self._experiment_id(experiment)
        return self._read_json(
            Path("error_injection") / experiment / "experiment_config.json"
        )

    def _seed_report(self, experiment: str, seed: int) -> dict[str, Any]:
        experiment = self._experiment_id(experiment)
        if seed not in SEEDS:
            raise UnknownExperiment
        return self._read_json(
            Path("error_injection")
            / experiment
            / "per_seed"
            / f"seed-{seed}.json"
        )

    def _seed_observations(
        self, experiment: str, severity: str, seeds: tuple[int, ...] = SEEDS
    ) -> list[dict[str, Any]]:
        return [
            _seed_observation(
                self._seed_report(experiment, seed), experiment, severity, seed
            )
            for seed in seeds
        ]

    def _global_clean(self, model: str) -> dict[str, float]:
        report = self.baseline_report()
        try:
            metrics = report["results"][model]
            return {
                metric: _finite_number(metrics[metric], f"baseline {model} {metric}")
                for metric in ("ade", "fde")
            }
        except (KeyError, TypeError) as error:
            raise ArtifactInvalid(f"Baseline model {model} is incomplete.") from error

    def _severity_parameters(self, experiment: str, severity: str) -> dict[str, Any]:
        config = self._config(experiment)
        if experiment == "localization_noise":
            parameters = config["noise_model"]["sigma_by_severity"][severity]
            return {
                key: parameters[key]
                for key in ("multiplier", "sigma", "units", "interpretation")
                if key in parameters
            }
        if experiment == "detection_dropout":
            probability = config["dropout"]["rates"][severity]
            return {"configured_dropout_probability": probability}
        if experiment == "track_fragmentation":
            count = config["fragmentation"]["severities"][severity]
            return {"sampled_observations_removed": count}
        count = config["identity_switch"]["severities"][severity]
        return {"swapped_observations": count}

    @staticmethod
    def _window_statistics(
        observations: list[dict[str, Any]],
        affected_statistics: dict[str, MetricStats],
    ) -> WindowStatistics:
        candidates = {item["candidate_count"] for item in observations}
        if len(candidates) != 1:
            raise ArtifactInvalid("Candidate window counts differ across seeds.")
        candidate_count = candidates.pop()
        evaluated = [item["evaluated_count"] for item in observations]
        unusable = [item["unusable_count"] for item in observations]
        retention = [count / candidate_count * 100 for count in evaluated]

        return WindowStatistics(
            candidate_count=candidate_count,
            evaluated_count=_stats(evaluated),
            retained_count=_stats(evaluated),
            unusable_count=_stats(unusable),
            retention_percentage=_stats(retention),
            affected_statistics=affected_statistics,
        )

    def _severity_summary(
        self,
        experiment: str,
        severity: str,
        observations: list[dict[str, Any]],
    ) -> SeveritySummary:
        clean_global = self._global_clean("lstm")
        clean_samples = [item["clean"] for item in observations]
        corrupted_samples = [item["corrupted"] for item in observations]
        degradation = [
            {
                metric: item["corrupted"][metric] - item["clean"][metric]
                for metric in ("ade", "fde")
            }
            for item in observations
        ]
        percentage_degradation = {
            metric: [
                (item["corrupted"][metric] - item["clean"][metric])
                / item["clean"][metric]
                * 100
                for item in observations
            ]
            for metric in ("ade", "fde")
        }
        aggregate = self._aggregate(experiment)
        units = aggregate.get("distance_units")
        if not isinstance(units, str) or not units:
            raise ArtifactInvalid(f"{experiment} coordinate units are missing.")
        severity_aggregate = aggregate["severity_results"][severity]
        affected_statistics: dict[str, MetricStats] = {}
        for name in _AFFECTED_FIELDS[experiment]:
            value = severity_aggregate.get(name)
            if isinstance(value, dict) and {"mean", "std"} <= set(value):
                affected_statistics[name] = MetricStats(
                    mean=_finite_number(value["mean"], f"{experiment} {name} mean"),
                    population_std=_finite_number(
                        value["std"], f"{experiment} {name} standard deviation"
                    ),
                )
            elif isinstance(value, (int, float)) and not isinstance(value, bool):
                affected_statistics[name] = MetricStats(
                    mean=_finite_number(value, f"{experiment} {name}")
                )
            elif value is None:
                seed_values = [
                    item["affected"][name]
                    for item in observations
                    if name in item["affected"]
                ]
                if seed_values:
                    affected_statistics[name] = _stats(seed_values)
        return SeveritySummary(
            severity=severity,
            parameters=self._severity_parameters(experiment, severity),
            global_clean_metrics=MetricPair(
                ade=MetricStats(mean=clean_global["ade"]),
                fde=MetricStats(mean=clean_global["fde"]),
            ),
            matched_clean_metrics=_metric_pair(clean_samples),
            corrupted_metrics=_metric_pair(corrupted_samples),
            degradation=_metric_pair(degradation),
            percentage_degradation=MetricPair(
                ade=_stats(percentage_degradation["ade"]),
                fde=_stats(percentage_degradation["fde"]),
            ),
            windows=self._window_statistics(observations, affected_statistics),
            seed_ids=[item["seed"] for item in observations],
            coordinate_units=units,
        )

    def baseline(self, model: str) -> BaselineResponse:
        model_key = {"constant_velocity": "constant_velocity", "lstm": "lstm"}.get(
            model
        )
        if model_key is None:
            raise UnknownExperiment
        report = self.baseline_report()
        try:
            model_report = report["results"][model_key]
            window_count = _nonnegative_count(
                report["results"]["window_count"], "baseline window count"
            )
            per_scene_data = model_report["per_scene"]
            ade_by_timestep = model_report["ade_by_timestep"]
            units = model_report["distance_units"]
        except (KeyError, TypeError) as error:
            raise ArtifactInvalid(f"Baseline model {model_key} is incomplete.") from error
        if not isinstance(per_scene_data, dict) or not isinstance(ade_by_timestep, list):
            raise ArtifactInvalid("Baseline detail fields have an invalid structure.")
        metrics = {
            metric: _finite_number(model_report[metric], f"baseline {metric}")
            for metric in ("ade", "fde")
        }
        per_scene = {
            scene: SceneMetrics(
                ade=MetricStats(mean=_finite_number(values["ade"], "scene ADE")),
                fde=MetricStats(mean=_finite_number(values["fde"], "scene FDE")),
                window_count=MetricStats(
                    mean=_nonnegative_count(values["window_count"], "scene window count")
                ),
            )
            for scene, values in per_scene_data.items()
        }
        return BaselineResponse(
            dataset="eth",
            split="val",
            model=model_key,
            metrics=metrics,
            window_count=window_count,
            per_scene=per_scene,
            ade_by_timestep=ade_by_timestep,
            coordinate_units=units,
        )

    def dashboard_summary(self, dataset: str) -> DashboardSummaryResponse:
        if dataset != "eth":
            raise UnknownExperiment
        baseline = self.baseline_report()
        results = baseline["results"]
        lstm = results["lstm"]
        config = self._baseline_experiment()["config"]
        scene_names = list(lstm["per_scene"])
        clean_baseline = {
            model: {
                metric: _finite_number(results[model][metric], f"{model} {metric}")
                for metric in ("ade", "fde")
            }
            for model in ("constant_velocity", "lstm")
        }
        return DashboardSummaryResponse(
            dataset="eth",
            split="val",
            candidate_window_count=_nonnegative_count(
                results["window_count"], "baseline candidate count"
            ),
            validation_scene_count=len(scene_names),
            validation_scenes=scene_names,
            clean_baseline=clean_baseline,
            available_error_experiments=list(EXPERIMENT_IDS),
            lstm_model_configuration={
                key: config[key]
                for key in (
                    "hidden_size",
                    "num_layers",
                    "dropout",
                    "observation_len",
                    "horizon_len",
                    "seed",
                )
                if key in config
            },
            coordinate_units=lstm["distance_units"],
        )

    def all_error_summaries(self, dataset: str) -> ErrorExperimentsResponse:
        if dataset != "eth":
            raise UnknownExperiment
        output = []
        for experiment in EXPERIMENT_IDS:
            severity_summaries = []
            for severity in SEVERITIES:
                observations = self._seed_observations(experiment, severity)
                severity_summaries.append(
                    self._severity_summary(experiment, severity, observations)
                )
            output.append(
                ErrorExperimentSummary(
                    experiment_id=experiment,
                    severities=severity_summaries,
                )
            )
        return ErrorExperimentsResponse(
            dataset="eth", split="val", experiments=output
        )

    def experiment_detail(
        self, experiment: str, severity: str, seed: int | None = None
    ) -> ExperimentDetailResponse:
        experiment = self._experiment_id(experiment)
        severity = self._severity_id(severity)
        seeds = (seed,) if seed is not None else SEEDS
        observations = self._seed_observations(experiment, severity, seeds)
        summary = self._severity_summary(experiment, severity, observations)
        aggregate = self._aggregate(experiment)
        severity_result = aggregate["severity_results"][severity]
        corrupted_key = {
            "localization_noise": "models",
            "detection_dropout": "models",
            "track_fragmentation": "fragmented_input_metrics",
            "identity_switch": "corrupted_metrics",
        }[experiment]
        corrupted_aggregate = severity_result[corrupted_key]["lstm"]
        matched_key = {
            "localization_noise": None,
            "detection_dropout": "clean_baseline_on_retained_windows",
            "track_fragmentation": "clean_baseline_on_retained_windows",
            "identity_switch": "clean_metrics",
        }[experiment]
        matched_aggregate = (
            severity_result[matched_key]["lstm"] if matched_key else None
        )

        # Use the per-seed saved rows so all comparisons remain paired to the
        # exact evaluated windows, including fragmentation survivors.
        corrupted_containers = [item["corrupted_container"] for item in observations]
        clean_containers = [item["clean_container"] for item in observations]
        scenes = set(corrupted_containers[0].get("per_scene", {}))
        if any(set(item.get("per_scene", {})) != scenes for item in corrupted_containers):
            raise ArtifactInvalid("Per-scene coverage differs across corruption seeds.")
        if any(set(item.get("per_scene", {})) != scenes for item in clean_containers):
            raise ArtifactInvalid("Matched-clean per-scene coverage differs.")
        per_scene = {
            scene: SceneComparison(
                matched_clean=(
                    _scene_metrics(clean_containers, scene)
                    if matched_aggregate is not None
                    or experiment == "localization_noise"
                    else None
                ),
                corrupted=_scene_metrics(corrupted_containers, scene),
            )
            for scene in sorted(scenes)
        }
        per_timestep = _timestep_comparisons(
            clean_containers, corrupted_containers
        )
        config = self._config(experiment)
        protocol = config.get("protocol", {})
        description = config.get("description")

        result = ExperimentDetailResponse(
            **summary.model_dump(),
            experiment_id=experiment,
            metadata=ExperimentMetadata(
                protocol=protocol if isinstance(protocol, dict) else {},
                description=description if isinstance(description, str) else None,
            ),
            per_scene=per_scene,
            per_timestep=per_timestep,
            selected_seed=seed,
        )
        if seed is None:
            # Aggregates include additional documented metric details; ensure
            # required LSTM ADE/FDE are sourced from the seed-level reports.
            del corrupted_aggregate
            del matched_aggregate
        else:
            result.seed_ids = [seed]
        return result

    def eth_dataset_audit(self) -> EthDatasetAuditResponse:
        audit = self._read_json(DATASET_AUDIT_RELATIVE_PATH)
        try:
            return EthDatasetAuditResponse(
                dataset="eth",
                configuration=audit["configuration"],
                split_summaries=audit["split_summaries"],
                issue_count=len(audit["issues"]),
                scene_overlap_checks=len(audit["overlap_summaries"]),
                identical_observation_rows_across_splits=sum(
                    _nonnegative_count(
                        row["identical_full_observation_rows"], "overlap row count"
                    )
                    for row in audit["overlap_summaries"]
                ),
                source=DATASET_AUDIT_RELATIVE_PATH.as_posix(),
            )
        except (KeyError, TypeError) as error:
            raise ArtifactInvalid("The ETH dataset audit is incomplete.") from error

    @staticmethod
    def eth_figures() -> EthFiguresResponse:
        return EthFiguresResponse(
            dataset="eth",
            figures=[
                EthFigure(figure_id=figure_id, title=title, group=group)
                for figure_id, (_, title, group) in ETH_FIGURES.items()
            ],
        )

    def eth_figure_path(self, figure_id: str) -> Path:
        if figure_id not in ETH_FIGURES:
            raise UnknownExperiment
        relative_path = ETH_FIGURES[figure_id][0]
        path = (self.results_root / "error_injection" / relative_path).resolve()
        try:
            path.relative_to(self.results_root)
        except ValueError as error:
            raise UnknownExperiment from error
        if not path.is_file():
            raise ArtifactNotFound
        return path

    def visualization_path(self, experiment: str, severity: str) -> Path:
        experiment = self._experiment_id(experiment)
        severity = self._severity_id(severity)
        filename = VISUALIZATION_ALLOWLIST[experiment][severity]
        path = (self.results_root / "error_injection" / experiment / filename).resolve()
        try:
            path.relative_to(self.results_root)
        except ValueError as error:
            raise UnknownExperiment from error
        if not path.is_file():
            raise ArtifactNotFound
        return path
