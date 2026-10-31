"""Publication-style comparisons of saved LSTM error-injection results."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SEEDS = (42, 43, 44)
SEVERITIES = ("low", "medium", "high")
ERRORS: dict[str, dict[str, Any]] = {
    "localization_noise": {
        "title": "Localization Noise",
        "labels": {
            "clean": "Clean",
            "low": "Low\n(0.10x)",
            "medium": "Medium\n(0.25x)",
            "high": "High\n(0.50x)",
        },
        "directory": "localization_noise",
    },
    "detection_dropout": {
        "title": "Detection Dropout",
        "labels": {
            "clean": "Clean",
            "low": "Low\n(10%)",
            "medium": "Medium\n(20%)",
            "high": "High\n(30%)",
        },
        "directory": "detection_dropout",
    },
    "track_fragmentation": {
        "title": "Track Fragmentation",
        "labels": {
            "clean": "Clean",
            "low": "Low\n(1 observation\nremoved)",
            "medium": "Medium\n(2 observations\nremoved)",
            "high": "High\n(3 observations\nremoved)",
        },
        "directory": "track_fragmentation",
    },
    "identity_switch": {
        "title": "Identity Switches",
        "labels": {
            "clean": "Clean",
            "low": "Low\n(1 swapped\nobservation)",
            "medium": "Medium\n(2 swapped\nobservations)",
            "high": "High\n(3 swapped\nobservations)",
        },
        "directory": "identity_switch",
    },
}


@dataclass(frozen=True)
class SeedMetrics:
    seed: int
    candidate_windows: int
    evaluated_windows: int
    matched_clean_ade: float
    matched_clean_fde: float
    corrupted_ade: float
    corrupted_fde: float
    clean_window_signature: str
    corrupted_window_signature: str


def _number(value: Any, description: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{description} must be a numeric value")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{description} must be finite")
    return result


def _count(value: Any, description: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{description} must be a non-negative integer")
    return value


def _signature(metrics: dict[str, Any], result: dict[str, Any], label: str) -> str:
    value = metrics.get("window_key_sha256", result.get("window_key_sha256"))
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} is missing its window signature")
    return value


def _extract_seed_metrics(
    error_type: str,
    report: dict[str, Any],
    severity: str,
    expected_seed: int,
) -> SeedMetrics:
    """Read one seed's LSTM metrics and validate its clean/corrupted window match."""
    if report.get("seed") != expected_seed or report.get("split") != "val":
        raise ValueError(f"{error_type} seed report must identify validation seed {expected_seed}")

    if error_type == "localization_noise":
        severity_result = report["severities"][severity]
        clean_metrics = report["clean"]["lstm"]
        corrupted_metrics = severity_result["metrics"]["lstm"]
        candidate_windows = _count(
            report["clean"]["window_count"], "localization-noise clean window count"
        )
        evaluated_windows = _count(
            severity_result["evaluated_windows"], "localization-noise evaluated windows"
        )
        clean_parent = report["clean"]
        corrupted_parent = severity_result
    elif error_type == "detection_dropout":
        severity_result = report["severity_results"][severity]
        clean_metrics = severity_result["clean_baseline_on_retained_windows"]["lstm"]
        corrupted_metrics = severity_result["metrics"]["lstm"]
        candidate_windows = _count(
            report["candidate_window_count"], "dropout candidate window count"
        )
        evaluated_windows = _count(
            severity_result["evaluated_window_count"], "dropout evaluated windows"
        )
        if evaluated_windows + _count(
            severity_result["unusable_window_count"], "dropout unusable windows"
        ) != candidate_windows:
            raise ValueError("dropout evaluated and unusable counts do not account for candidates")
        clean_parent = severity_result["clean_baseline_on_retained_windows"]
        corrupted_parent = severity_result
    elif error_type == "track_fragmentation":
        severity_result = report["severity_results"][severity]
        clean_metrics = severity_result["clean_baseline_on_retained_windows"]["lstm"]
        corrupted_metrics = severity_result["fragmented_input_metrics"]["lstm"]
        candidate_windows = _count(
            severity_result["candidate_window_count"], "fragmentation candidate window count"
        )
        evaluated_windows = _count(
            severity_result["retained_window_count"], "fragmentation retained windows"
        )
        if evaluated_windows + _count(
            severity_result["unusable_window_count"], "fragmentation unusable windows"
        ) != candidate_windows:
            raise ValueError(
                "fragmentation retained and unusable counts do not account for candidates"
            )
        clean_parent = severity_result["clean_baseline_on_retained_windows"]
        corrupted_parent = severity_result["fragmented_input_metrics"]
    elif error_type == "identity_switch":
        severity_result = report["severity_results"][severity]
        clean_metrics = severity_result["clean_metrics"]["lstm"]
        corrupted_metrics = severity_result["corrupted_metrics"]["lstm"]
        candidate_windows = _count(
            report["candidate_window_count"], "identity-switch candidate window count"
        )
        evaluated_windows = _count(
            corrupted_metrics["window_count"], "identity-switch evaluated windows"
        )
        clean_parent = severity_result["clean_metrics"]
        corrupted_parent = severity_result["corrupted_metrics"]
    else:
        raise ValueError(f"unknown error type: {error_type}")

    clean_signature = _signature(clean_metrics, clean_parent, "matched-clean metrics")
    corrupted_signature = _signature(
        corrupted_metrics, corrupted_parent, "corrupted metrics"
    )
    if clean_signature != corrupted_signature:
        raise ValueError(
            f"{error_type} {severity} seed {expected_seed} compares different window sets"
        )
    if evaluated_windows > candidate_windows:
        raise ValueError(f"{error_type} evaluated more windows than its candidate set")
    for label, metrics in (
        ("matched-clean", clean_metrics),
        ("corrupted", corrupted_metrics),
    ):
        if "window_count" in metrics and metrics["window_count"] != evaluated_windows:
            raise ValueError(f"{error_type} {label} metric count does not match evaluated windows")

    return SeedMetrics(
        seed=expected_seed,
        candidate_windows=candidate_windows,
        evaluated_windows=evaluated_windows,
        matched_clean_ade=_number(clean_metrics["ade"], "matched-clean ADE"),
        matched_clean_fde=_number(clean_metrics["fde"], "matched-clean FDE"),
        corrupted_ade=_number(corrupted_metrics["ade"], "corrupted ADE"),
        corrupted_fde=_number(corrupted_metrics["fde"], "corrupted FDE"),
        clean_window_signature=clean_signature,
        corrupted_window_signature=corrupted_signature,
    )


def _summary(values: list[float] | tuple[float, ...]) -> tuple[float, float]:
    if not values:
        raise ValueError("cannot summarize an empty sequence")
    return statistics.mean(values), statistics.pstdev(values)


def _close(actual: float, expected: float, label: str) -> None:
    if not math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-11):
        raise ValueError(f"saved aggregate {label} disagrees with per-seed reports")


def _verify_aggregate(
    error_type: str,
    aggregate: dict[str, Any],
    severity: str,
    records: list[SeedMetrics],
) -> None:
    if aggregate.get("split") != "val":
        raise ValueError(f"{error_type} aggregate is not validation-only")
    result = aggregate["severity_results"][severity]
    if error_type in ("localization_noise", "detection_dropout"):
        corrupted = result["models"]["lstm"]
    elif error_type == "track_fragmentation":
        corrupted = result["fragmented_input_metrics"]["lstm"]
    else:
        corrupted = result["corrupted_metrics"]["lstm"]
    for metric, field in (("ade", "corrupted_ade"), ("fde", "corrupted_fde")):
        mean, std = _summary([getattr(record, field) for record in records])
        _close(float(corrupted[metric]["mean"]), mean, f"{error_type} {severity} {metric} mean")
        _close(float(corrupted[metric]["std"]), std, f"{error_type} {severity} {metric} SD")

    if error_type in ("detection_dropout", "track_fragmentation"):
        clean = result["clean_baseline_on_retained_windows"]["lstm"]
    elif error_type == "identity_switch":
        clean = result["clean_metrics"]["lstm"]
    else:
        clean = None
    if clean is not None:
        for metric, field in (("ade", "matched_clean_ade"), ("fde", "matched_clean_fde")):
            mean, std = _summary([getattr(record, field) for record in records])
            _close(float(clean[metric]["mean"]), mean, f"{error_type} {severity} clean {metric} mean")
            _close(float(clean[metric]["std"]), std, f"{error_type} {severity} clean {metric} SD")


def _global_clean_metrics(
    identity_reports: dict[int, dict[str, Any]],
) -> dict[str, list[float]]:
    result = {"ade": [], "fde": []}
    for seed in SEEDS:
        report = identity_reports[seed]
        clean = report["severity_results"]["low"]["clean_metrics"]
        model = clean["lstm"]
        if report.get("split") != "val" or model.get("window_count") != report.get(
            "candidate_window_count"
        ):
            raise ValueError("identity-switch clean metrics are not the full validation candidate set")
        for metric in result:
            result[metric].append(_number(model[metric], f"global clean {metric}"))
    return result


def _validate_global_clean(
    experiment: str,
    aggregate: dict[str, Any],
    global_clean: dict[str, list[float]],
    candidate_window_count: int,
) -> None:
    clean = aggregate.get("clean_baseline", aggregate.get("clean_baseline_all_candidate_windows"))
    if clean is None:
        raise ValueError(f"{experiment} aggregate has no full-candidate clean baseline")
    if clean["lstm"].get("window_count") != candidate_window_count:
        raise ValueError(f"{experiment} global clean baseline has an unexpected window count")
    for metric, values in global_clean.items():
        mean, _ = _summary(values)
        _close(float(clean["lstm"][metric]), mean, f"{experiment} global clean {metric}")


def _load_comparisons(project_root: Path) -> tuple[dict[str, Any], list[str]]:
    base = project_root / "runs" / "error_injection"
    reports: dict[str, dict[int, dict[str, Any]]] = {}
    aggregates: dict[str, dict[str, Any]] = {}
    sources: list[str] = []
    for error_type, settings in ERRORS.items():
        directory = base / settings["directory"]
        aggregate_path = directory / "aggregated_results.json"
        aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
        if aggregate.get("split") != "val":
            raise ValueError(f"{aggregate_path} is not validation-only")
        aggregates[error_type] = aggregate
        sources.append(aggregate_path.relative_to(project_root).as_posix())
        reports[error_type] = {}
        for seed in SEEDS:
            seed_path = directory / "per_seed" / f"seed-{seed}.json"
            report = json.loads(seed_path.read_text(encoding="utf-8"))
            if report.get("seed") != seed or report.get("split") != "val":
                raise ValueError(f"{seed_path} does not match its validation seed")
            reports[error_type][seed] = report
            sources.append(seed_path.relative_to(project_root).as_posix())

    global_clean = _global_clean_metrics(reports["identity_switch"])
    candidate_window_counts = {
        report["candidate_window_count"]
        for report in reports["identity_switch"].values()
    }
    if len(candidate_window_counts) != 1:
        raise ValueError("identity-switch reports have inconsistent candidate-window counts")
    candidate_window_count = candidate_window_counts.pop()
    for error_type, aggregate in aggregates.items():
        _validate_global_clean(
            error_type, aggregate, global_clean, candidate_window_count
        )

    comparisons: dict[str, Any] = {}
    for error_type in ERRORS:
        per_severity: dict[str, list[SeedMetrics]] = {}
        for severity in SEVERITIES:
            records = [
                _extract_seed_metrics(error_type, reports[error_type][seed], severity, seed)
                for seed in SEEDS
            ]
            if any(record.candidate_windows != candidate_window_count for record in records):
                raise ValueError(
                    f"{error_type} {severity} candidate count differs from the clean baseline"
                )
            _verify_aggregate(error_type, aggregates[error_type], severity, records)
            per_severity[severity] = records
        comparisons[error_type] = {
            "global_clean": global_clean,
            "severities": per_severity,
        }
    return comparisons, sources


def _fmt_count(mean: float, std: float) -> str:
    if std < 0.05:
        return f"n={mean:,.0f}"
    return f"n={mean:,.1f} ± {std:,.1f}"


def _plot_figure(
    output_path: Path,
    error_type: str,
    comparison: dict[str, Any],
) -> None:
    settings = ERRORS[error_type]
    categories = ("clean", *SEVERITIES)
    positions = list(range(len(categories)))
    clean_color = "#374151"
    matched_color = "#64748b"
    corrupted_color = "#1769aa"
    retention_color = "#23856d"

    with plt.rc_context({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 11,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.titlesize": 15,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
    }):
        figure, axes = plt.subplots(1, 3, figsize=(15, 5.3), constrained_layout=False)
        for axis, metric, clean_color_value in (
            (axes[0], "ade", clean_color),
            (axes[1], "fde", clean_color),
        ):
            clean_values = comparison["global_clean"][metric]
            mean, std = _summary(clean_values)
            axis.errorbar(
                positions[0], mean, yerr=std, marker="o", markersize=6,
                capsize=4, elinewidth=1.3, linewidth=1.5, color=clean_color_value,
                label="Clean baseline (all candidates)", zorder=4,
            )
            for index, severity in enumerate(SEVERITIES, start=1):
                records: list[SeedMetrics] = comparison["severities"][severity]
                matched = [getattr(record, f"matched_clean_{metric}") for record in records]
                corrupted = [getattr(record, f"corrupted_{metric}") for record in records]
                matched_mean, matched_std = _summary(matched)
                corrupted_mean, corrupted_std = _summary(corrupted)
                axis.errorbar(
                    positions[index] - 0.11, matched_mean, yerr=matched_std,
                    marker="s", markersize=5, capsize=3, elinewidth=1.2,
                    linestyle="none", color=matched_color,
                    label="Matched clean" if index == 1 else None, zorder=3,
                )
                axis.errorbar(
                    positions[index] + 0.11, corrupted_mean, yerr=corrupted_std,
                    marker="o", markersize=5.5, capsize=3, elinewidth=1.2,
                    linestyle="none", color=corrupted_color,
                    label="After injection" if index == 1 else None, zorder=3,
                )
            axis.set_xticks(positions, [settings["labels"][key] for key in categories])
            axis.set_xlim(-0.45, 3.45)
            axis.grid(axis="y", color="#d9dee7", linewidth=0.7, alpha=0.9)
            axis.set_axisbelow(True)
            if metric == "ade":
                axis.set_title("A  |  Average Displacement Error")
                axis.set_ylabel("ADE (raw dataset coordinate units)")
            else:
                axis.set_title("B  |  Final Displacement Error")
                axis.set_ylabel("FDE (raw dataset coordinate units)")

        retention_means: list[float] = [100.0]
        retention_stds: list[float] = [0.0]
        count_summaries: list[tuple[float, float]] = []
        clean_count_mean, clean_count_std = _summary([
            float(record.candidate_windows)
            for record in comparison["severities"]["low"]
        ])
        count_summaries.append((clean_count_mean, clean_count_std))
        for severity in SEVERITIES:
            records = comparison["severities"][severity]
            candidate_counts = [record.candidate_windows for record in records]
            evaluated_counts = [record.evaluated_windows for record in records]
            if len(set(candidate_counts)) != 1:
                raise ValueError("candidate-window counts vary across seeds")
            ratios = [
                100.0 * evaluated / candidate
                for evaluated, candidate in zip(evaluated_counts, candidate_counts, strict=True)
            ]
            mean, std = _summary(ratios)
            retention_means.append(mean)
            retention_stds.append(std)
            count_summaries.append(_summary([float(count) for count in evaluated_counts]))

        axes[2].errorbar(
            positions, retention_means, yerr=retention_stds, marker="D",
            markersize=5.5, capsize=3, elinewidth=1.2, linewidth=1.4,
            color=retention_color, zorder=3,
        )
        axes[2].set_xticks(positions, [settings["labels"][key] for key in categories])
        axes[2].set_xlim(-0.45, 3.45)
        axes[2].set_ylim(0, 112)
        axes[2].set_title("C  |  Candidate-Window Retention")
        axes[2].set_ylabel("Evaluated candidates (%)")
        axes[2].grid(axis="y", color="#d9dee7", linewidth=0.7, alpha=0.9)
        axes[2].set_axisbelow(True)
        for position, mean, count in zip(
            positions, retention_means, count_summaries, strict=True
        ):
            axes[2].annotate(
                _fmt_count(*count),
                (position, mean),
                xytext=(0, -17 if mean > 88 else 8),
                textcoords="offset points",
                ha="center",
                va="top" if mean > 88 else "bottom",
                fontsize=7,
                color="#374151",
            )

        handles, labels = axes[0].get_legend_handles_labels()
        figure.legend(
            handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.91),
            ncol=3, frameon=False,
        )
        figure.suptitle(
            f"LSTM Trajectory Prediction Under {settings['title']}",
            y=0.99, fontweight="semibold",
        )
        definitions = (
            "ADE: mean Euclidean prediction error over 12 future steps. "
            "FDE: mean Euclidean error at step 12. "
            "Raw coordinate units; points and bars show mean ± population SD (seeds 42–44)."
        )
        if error_type == "track_fragmentation":
            definitions += (
                " Fragmentation prediction metrics use retained windows; "
                "each corrupted result is paired with clean metrics on those same windows."
            )
        figure.text(0.5, 0.025, definitions, ha="center", va="bottom", fontsize=8)
        figure.tight_layout(rect=(0.02, 0.11, 0.98, 0.84))
        figure.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(figure)


def _write_csv(path: Path, comparisons: dict[str, Any]) -> None:
    columns = [
        "error_type", "severity", "severity_label", "seed_ids",
        "global_clean_ade_mean", "global_clean_ade_population_sd",
        "global_clean_fde_mean", "global_clean_fde_population_sd",
        "matched_clean_ade_mean", "matched_clean_ade_population_sd",
        "matched_clean_fde_mean", "matched_clean_fde_population_sd",
        "corrupted_ade_mean", "corrupted_ade_population_sd",
        "corrupted_fde_mean", "corrupted_fde_population_sd",
        "candidate_window_count_mean", "candidate_window_count_population_sd",
        "evaluated_window_count_mean", "evaluated_window_count_population_sd",
        "retention_percentage_mean", "retention_percentage_population_sd",
    ]
    with path.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=columns)
        writer.writeheader()
        for error_type, comparison in comparisons.items():
            global_summaries = {
                metric: _summary(comparison["global_clean"][metric])
                for metric in ("ade", "fde")
            }
            clean_count = _summary([
                float(record.candidate_windows)
                for record in comparison["severities"]["low"]
            ])
            candidate_count = clean_count[0]
            clean_row: dict[str, Any] = {
                "error_type": error_type,
                "severity": "clean",
                "severity_label": "Clean",
                "seed_ids": ";".join(str(seed) for seed in SEEDS),
                "global_clean_ade_mean": global_summaries["ade"][0],
                "global_clean_ade_population_sd": global_summaries["ade"][1],
                "global_clean_fde_mean": global_summaries["fde"][0],
                "global_clean_fde_population_sd": global_summaries["fde"][1],
                "candidate_window_count_mean": candidate_count,
                "candidate_window_count_population_sd": clean_count[1],
                "evaluated_window_count_mean": candidate_count,
                "evaluated_window_count_population_sd": clean_count[1],
                "retention_percentage_mean": 100.0,
                "retention_percentage_population_sd": 0.0,
            }
            writer.writerow(clean_row)
            for severity in SEVERITIES:
                records: list[SeedMetrics] = comparison["severities"][severity]
                matched_ade = _summary([record.matched_clean_ade for record in records])
                matched_fde = _summary([record.matched_clean_fde for record in records])
                corrupted_ade = _summary([record.corrupted_ade for record in records])
                corrupted_fde = _summary([record.corrupted_fde for record in records])
                candidates = _summary([float(record.candidate_windows) for record in records])
                evaluated = _summary([float(record.evaluated_windows) for record in records])
                retention = _summary([
                    100.0 * record.evaluated_windows / record.candidate_windows
                    for record in records
                ])
                writer.writerow({
                    **clean_row,
                    "severity": severity,
                    "severity_label": ERRORS[error_type]["labels"][severity].replace("\n", " "),
                    "matched_clean_ade_mean": matched_ade[0],
                    "matched_clean_ade_population_sd": matched_ade[1],
                    "matched_clean_fde_mean": matched_fde[0],
                    "matched_clean_fde_population_sd": matched_fde[1],
                    "corrupted_ade_mean": corrupted_ade[0],
                    "corrupted_ade_population_sd": corrupted_ade[1],
                    "corrupted_fde_mean": corrupted_fde[0],
                    "corrupted_fde_population_sd": corrupted_fde[1],
                    "candidate_window_count_mean": candidates[0],
                    "candidate_window_count_population_sd": candidates[1],
                    "evaluated_window_count_mean": evaluated[0],
                    "evaluated_window_count_population_sd": evaluated[1],
                    "retention_percentage_mean": retention[0],
                    "retention_percentage_population_sd": retention[1],
                })


def generate_comparison_plots(
    project_root: Path,
    output_dir: Path | None = None,
) -> list[Path]:
    project_root = Path(project_root).resolve()
    destination = (
        Path(output_dir) if output_dir is not None
        else project_root / "runs" / "error_injection" / "lstm_comparison_plots"
    )
    if not destination.is_absolute():
        destination = project_root / destination
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(
            f"refusing to overwrite non-empty output directory: {destination}"
        )

    comparisons, source_paths = _load_comparisons(project_root)
    destination.mkdir(parents=True, exist_ok=True)
    figure_names = {
        "localization_noise": "lstm_localization_noise_comparison.png",
        "detection_dropout": "lstm_detection_dropout_comparison.png",
        "track_fragmentation": "lstm_track_fragmentation_comparison.png",
        "identity_switch": "lstm_identity_switch_comparison.png",
    }
    for error_type, filename in figure_names.items():
        _plot_figure(destination / filename, error_type, comparisons[error_type])
    csv_path = destination / "lstm_four_error_comparison.csv"
    _write_csv(csv_path, comparisons)
    manifest_path = destination / "plot_manifest.json"
    manifest = {
        "predictor": "frozen two-layer LSTM; saved prediction metrics only",
        "split": "validation",
        "seed_ids": list(SEEDS),
        "source_artifacts": source_paths,
        "clean_baseline_source": [
            f"runs/error_injection/identity_switch/per_seed/seed-{seed}.json"
            for seed in SEEDS
        ],
        "metric_definitions": {
            "ADE": "Mean Euclidean prediction error over the 12 future timesteps, averaged over evaluated windows.",
            "FDE": "Mean Euclidean prediction error at the final (12th) future timestep, averaged over evaluated windows.",
            "units": "Raw dataset coordinate units; physical calibration is unverified.",
            "uncertainty": "Arithmetic mean and population standard deviation across corruption seeds 42, 43, and 44.",
        },
        "comparison_policy": {
            "severity_order": ["clean", *SEVERITIES],
            "severity_labels": {
                error_type: settings["labels"] for error_type, settings in ERRORS.items()
            },
            "matched_clean": (
                "For each severity, show clean metrics computed on the same retained/evaluated "
                "window signatures as the corrupted predictions."
            ),
            "fragmentation": (
                "Prediction metrics are restricted to surviving windows; candidate-window "
                "retention is plotted separately."
            ),
        },
        "generated_files": [
            *figure_names.values(),
            csv_path.name,
            manifest_path.name,
        ],
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return [destination / name for name in manifest["generated_files"]]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot saved validation LSTM metrics for four error-injection experiments."
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
        help="TrackVision project root (defaults to the current directory)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (defaults to runs/error_injection/lstm_comparison_plots)",
    )
    arguments = parser.parse_args()
    outputs = generate_comparison_plots(arguments.project_root, arguments.output_dir)
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
