"""Validation-only detection-dropout experiments for frozen predictors."""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
from dataclasses import replace
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from trackvision.io.eth import ETHObservation, ETHWindow
from trackvision.prediction.experiment import (
    DEFAULT_DATASET_ROOT,
    ExperimentConfig,
    PositionNormalization,
    _file_sha256,
    _split_windows,
    _window_key_digest,
    evaluate_models,
    load_prediction_split,
)
from trackvision.prediction.localization_noise import (
    _add_degradation,
    _atomic_json,
    _degradation,
)
from trackvision.prediction.lstm import LSTMTrajectoryPredictor


DROPOUT_RATES = {"low": 0.10, "medium": 0.20, "high": 0.30}
CORRUPTION_SEEDS = (42, 43, 44)
DEFAULT_EXPERIMENT_DIR = Path("runs/prediction/eth/lstm-seed42-h64-l2-scheduler")
DEFAULT_OUTPUT_DIR = Path("runs/error_injection/detection_dropout")


def _observation_key(scene: str, observation: ETHObservation) -> tuple[str, int, int]:
    return scene, observation.pedestrian_id, observation.frame


def generate_dropout_mask(
    windows: tuple[ETHWindow, ...],
    rate: float,
    seed: int,
) -> tuple[dict[tuple[str, int, int], bool], dict[str, int | float]]:
    """Draw one Bernoulli missingness decision per unique observed position."""
    if not windows:
        raise ValueError("windows must not be empty")
    if any(window.split != "val" for window in windows):
        raise ValueError("detection-dropout evaluation accepts validation windows only")
    if not math.isfinite(rate) or not 0.0 <= rate <= 1.0:
        raise ValueError("dropout rate must be finite and in [0, 1]")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")

    observations: dict[tuple[str, int, int], ETHObservation] = {}
    for window in windows:
        for observation in window.observed:
            key = _observation_key(window.scene, observation)
            previous = observations.get(key)
            if previous is not None and (previous.x, previous.y) != (observation.x, observation.y):
                raise ValueError(f"conflicting coordinates for repeated observation {key}")
            observations[key] = observation

    keys = sorted(observations)
    if rate == 0:
        decisions = np.zeros(len(keys), dtype=np.bool_)
    elif rate == 1:
        decisions = np.ones(len(keys), dtype=np.bool_)
    else:
        decisions = np.random.default_rng(seed).random(len(keys)) < rate
    mask = {key: bool(decisions[index]) for index, key in enumerate(keys)}
    dropped_count = int(decisions.sum())
    return mask, {
        "seed": seed,
        "configured_rate": float(rate),
        "unique_observation_count": len(keys),
        "dropped_unique_observation_count": dropped_count,
        "actual_missing_percentage_unique": (
            100.0 * dropped_count / len(keys) if keys else 0.0
        ),
        "window_observation_count": sum(len(window.observed) for window in windows),
    }


def _longest_missing_run(dropped: np.ndarray) -> int:
    longest = current = 0
    for missing in dropped:
        if missing:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def _window_dropout_mask(
    window: ETHWindow,
    mask: dict[tuple[str, int, int], bool],
) -> np.ndarray:
    return np.asarray(
        [
            mask[_observation_key(window.scene, observation)]
            for observation in window.observed
        ],
        dtype=np.bool_,
    )


def impute_dropout_windows(
    windows: tuple[ETHWindow, ...],
    mask: dict[tuple[str, int, int], bool],
) -> tuple[tuple[ETHWindow, ...], dict[str, Any]]:
    """Linearly impute observed gaps and explicitly report all-missing windows."""
    if not windows:
        raise ValueError("windows must not be empty")
    retained: list[ETHWindow] = []
    unusable: list[dict[str, Any]] = []
    dropped_window_observations = 0
    missing_percentages: list[float] = []
    longest_run = 0

    for window in windows:
        absent = _window_dropout_mask(window, mask)
        dropped_window_observations += int(absent.sum())
        missing_percentages.append(100.0 * float(absent.mean()))
        longest_run = max(longest_run, _longest_missing_run(absent))
        valid = np.flatnonzero(~absent)
        if valid.size == 0:
            unusable.append({
                "scene": window.scene,
                "pedestrian_id": window.pedestrian_id,
                "start_frame": window.start_frame,
                "reason": "all_8_observed_positions_missing",
                "missing_run_length": len(window.observed),
            })
            continue

        coordinates = np.asarray(
            [(point.x, point.y) for point in window.observed], dtype=np.float64
        )
        positions = np.arange(len(window.observed))
        imputed = coordinates.copy()
        for axis in range(2):
            imputed[:, axis] = np.interp(
                positions,
                valid,
                coordinates[valid, axis],
            )
        observed = tuple(
            replace(point, x=float(imputed[index, 0]), y=float(imputed[index, 1]))
            for index, point in enumerate(window.observed)
        )
        retained.append(replace(window, observed=observed))

    candidate_observations = sum(len(window.observed) for window in windows)
    return tuple(retained), {
        "candidate_window_count": len(windows),
        "evaluated_window_count": len(retained),
        "unusable_window_count": len(unusable),
        "unusable_windows": unusable,
        "dropped_window_observation_count": dropped_window_observations,
        "actual_missing_percentage_window_occurrences": (
            100.0 * dropped_window_observations / candidate_observations
            if candidate_observations else 0.0
        ),
        "longest_consecutive_missing_run": longest_run,
        "imputation_policy": (
            "linear interpolation between nearest valid observed positions; leading/trailing "
            "gaps use nearest valid observed position; all-missing windows are unusable"
        ),
    }


def _mean_std(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {"mean": float(array.mean()), "std": float(array.std())}


def _aggregate_model(records: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {
        metric: _mean_std([float(record[metric]) for record in records])
        for metric in ("ade", "fde")
    }
    for metric in ("ade", "fde"):
        result.setdefault("degradation_vs_clean", {})[metric] = {
            field: _mean_std([
                float(record["degradation_vs_clean"][metric][field])
                for record in records
                if record["degradation_vs_clean"][metric][field] is not None
            ])
            for field in ("absolute", "percentage")
        }

    scenes = sorted(records[0]["per_scene"])
    result["per_scene"] = {
        scene: {
            metric: _mean_std([
                float(record["per_scene"][scene][metric]) for record in records
            ])
            for metric in ("ade", "fde")
        }
        for scene in scenes
    }
    result["per_scene_degradation_vs_clean"] = {
        scene: {
            metric: {
                field: _mean_std([
                    float(record["per_scene_degradation_vs_clean"][scene][metric][field])
                    for record in records
                ])
                for field in ("absolute", "percentage")
            }
            for metric in ("ade", "fde")
        }
        for scene in scenes
    }
    result["ade_by_timestep"] = [
        {
            "timestep": timestep + 1,
            "ade": _mean_std([
                float(record["ade_by_timestep"][timestep]["ade"])
                for record in records
            ]),
            "degradation_vs_clean": {
                field: _mean_std([
                    float(record["per_timestep_degradation_vs_clean"][timestep][field])
                    for record in records
                ])
                for field in ("absolute", "percentage")
            },
        }
        for timestep in range(len(records[0]["ade_by_timestep"]))
    ]
    return result


def _aggregate_clean_model(records: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {
        metric: _mean_std([float(record[metric]) for record in records])
        for metric in ("ade", "fde")
    }
    scenes = sorted(records[0]["per_scene"])
    result["per_scene"] = {
        scene: {
            metric: _mean_std([
                float(record["per_scene"][scene][metric]) for record in records
            ])
            for metric in ("ade", "fde")
        }
        for scene in scenes
    }
    result["ade_by_timestep"] = [
        {
            "timestep": timestep + 1,
            "ade": _mean_std([
                float(record["ade_by_timestep"][timestep]["ade"])
                for record in records
            ]),
        }
        for timestep in range(len(records[0]["ade_by_timestep"]))
    ]
    return result


def _aggregate_severity(seed_results: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "seeds": [result["seed"] for result in seed_results],
        "candidate_window_count": seed_results[0]["candidate_window_count"],
        "evaluated_window_count": {
            "mean": float(np.mean([r["evaluated_window_count"] for r in seed_results])),
            "std": float(np.std([r["evaluated_window_count"] for r in seed_results])),
        },
        "unusable_window_count": {
            "mean": float(np.mean([r["unusable_window_count"] for r in seed_results])),
            "std": float(np.std([r["unusable_window_count"] for r in seed_results])),
        },
        "window_key_sha256_by_seed": {
            str(r["seed"]): r["window_key_sha256"] for r in seed_results
        },
        "actual_missing_percentage_unique": _mean_std([
            float(r["dropout_statistics"]["actual_missing_percentage_unique"])
            for r in seed_results
        ]),
        "actual_missing_percentage_window_occurrences": _mean_std([
            float(r["imputation_statistics"]["actual_missing_percentage_window_occurrences"])
            for r in seed_results
        ]),
        "dropped_unique_observation_count": _mean_std([
            float(r["dropout_statistics"]["dropped_unique_observation_count"])
            for r in seed_results
        ]),
        "longest_consecutive_missing_run": _mean_std([
            float(r["imputation_statistics"]["longest_consecutive_missing_run"])
            for r in seed_results
        ]),
        "clean_baseline_on_retained_windows": {
            model: _aggregate_clean_model([
                r["clean_baseline_on_retained_windows"][model]
                for r in seed_results
            ])
            for model in ("constant_velocity", "lstm")
        },
        "models": {
            model: _aggregate_model([r["metrics"][model] for r in seed_results])
            for model in ("constant_velocity", "lstm")
        },
    }


def _write_summary_csv(path: Path, aggregate: dict[str, Any]) -> None:
    columns = (
        "severity", "model", "candidate_windows", "evaluated_windows_mean",
        "unusable_windows_mean", "actual_missing_percentage_unique_mean",
        "actual_missing_percentage_window_occurrences_mean", "longest_missing_run_mean",
        "clean_ade_mean", "clean_fde_mean", "ade_mean", "ade_std", "fde_mean", "fde_std",
        "ade_degradation_mean", "ade_degradation_std", "ade_degradation_percentage_mean",
        "ade_degradation_percentage_std", "fde_degradation_mean", "fde_degradation_std",
        "fde_degradation_percentage_mean", "fde_degradation_percentage_std",
    )
    with path.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=columns)
        writer.writeheader()
        for severity, result in aggregate.items():
            for model in ("constant_velocity", "lstm"):
                metrics = result["models"][model]
                clean_metrics = result["clean_baseline_on_retained_windows"][model]
                ade = metrics["degradation_vs_clean"]["ade"]
                fde = metrics["degradation_vs_clean"]["fde"]
                writer.writerow({
                    "severity": severity,
                    "model": model,
                    "candidate_windows": result["candidate_window_count"],
                    "evaluated_windows_mean": result["evaluated_window_count"]["mean"],
                    "unusable_windows_mean": result["unusable_window_count"]["mean"],
                    "actual_missing_percentage_unique_mean": result["actual_missing_percentage_unique"]["mean"],
                    "actual_missing_percentage_window_occurrences_mean": result["actual_missing_percentage_window_occurrences"]["mean"],
                    "longest_missing_run_mean": result["longest_consecutive_missing_run"]["mean"],
                    "clean_ade_mean": clean_metrics["ade"]["mean"],
                    "clean_fde_mean": clean_metrics["fde"]["mean"],
                    "ade_mean": metrics["ade"]["mean"],
                    "ade_std": metrics["ade"]["std"],
                    "fde_mean": metrics["fde"]["mean"],
                    "fde_std": metrics["fde"]["std"],
                    "ade_degradation_mean": ade["absolute"]["mean"],
                    "ade_degradation_std": ade["absolute"]["std"],
                    "ade_degradation_percentage_mean": ade["percentage"]["mean"],
                    "ade_degradation_percentage_std": ade["percentage"]["std"],
                    "fde_degradation_mean": fde["absolute"]["mean"],
                    "fde_degradation_std": fde["absolute"]["std"],
                    "fde_degradation_percentage_mean": fde["percentage"]["mean"],
                    "fde_degradation_percentage_std": fde["percentage"]["std"],
                })


def _plot_imputation_examples(
    path: Path,
    window: ETHWindow,
    mask: dict[tuple[str, int, int], bool],
    rate: float,
) -> None:
    dropped = _window_dropout_mask(window, mask)
    valid = np.flatnonzero(~dropped)
    positions = np.arange(len(window.observed))
    coordinates = np.asarray([(point.x, point.y) for point in window.observed])
    imputed = coordinates.copy()
    for axis in range(2):
        imputed[:, axis] = np.interp(positions, valid, coordinates[valid, axis])

    detected = ~dropped
    figure, axis = plt.subplots(figsize=(8, 6))
    axis.plot(
        coordinates[:, 0], coordinates[:, 1], ":",
        color="0.5", label="Original clean observed",
    )
    axis.scatter(
        coordinates[detected, 0], coordinates[detected, 1],
        marker="o", facecolors="none", edgecolors="tab:blue", s=100,
        label="Detected positions",
    )
    axis.scatter(
        coordinates[dropped, 0], coordinates[dropped, 1],
        marker="x", color="tab:red", s=80, label="Dropped clean positions",
    )
    axis.scatter(
        imputed[dropped, 0], imputed[dropped, 1],
        marker="D", color="tab:orange", s=55, label="Imputed positions",
    )
    future = np.asarray([(point.x, point.y) for point in window.future])
    axis.plot(future[:, 0], future[:, 1], "s-", color="tab:green", label="Future ground truth")
    axis.set_title(f"Detection dropout example ({rate:.0%}), {window.scene}")
    axis.set_xlabel("x (raw coordinate units)")
    axis.set_ylabel("y (raw coordinate units)")
    axis.set_aspect("equal", adjustable="box")
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _plot_severity_comparison(path: Path, aggregates: dict[str, Any]) -> None:
    levels = list(DROPOUT_RATES)
    x = np.asarray([DROPOUT_RATES[level] * 100 for level in levels])
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    for model, label, color in (
        ("constant_velocity", "Constant Velocity", "tab:blue"),
        ("lstm", "Two-layer LSTM", "tab:orange"),
    ):
        for axis, metric in zip(axes, ("ade", "fde"), strict=True):
            y = np.asarray([
                aggregates[level]["models"][model][metric]["mean"]
                for level in levels
            ])
            error = np.asarray([
                aggregates[level]["models"][model][metric]["std"]
                for level in levels
            ])
            axis.errorbar(x, y, yerr=error, marker="o", capsize=3, label=label, color=color)
            clean_y = np.asarray([
                aggregates[level]["clean_baseline_on_retained_windows"][model][metric]["mean"]
                for level in levels
            ])
            clean_error = np.asarray([
                aggregates[level]["clean_baseline_on_retained_windows"][model][metric]["std"]
                for level in levels
            ])
            axis.errorbar(
                x, clean_y, yerr=clean_error, marker="x", linestyle="--",
                alpha=0.65, capsize=3, color=color,
                label=f"{label} clean (matched windows)",
            )
            axis.set_title(metric.upper())
            axis.set_xlabel("Configured observation dropout probability (%)")
            axis.set_ylabel("Error (raw dataset coordinate units)")
            axis.grid(True, alpha=0.3)
    axes[0].legend()
    figure.suptitle("Detection dropout severity versus validation prediction error")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def run_detection_dropout_experiment(
    dataset_root: Path = DEFAULT_DATASET_ROOT,
    experiment_dir: Path = DEFAULT_EXPERIMENT_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    dataset_root = Path(dataset_root)
    experiment_dir = Path(experiment_dir)
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output directory: {output_dir}")

    manifest_path = experiment_dir / "experiment.json"
    checkpoint_path = experiment_dir / "best_lstm.pt"
    if not manifest_path.is_file() or not checkpoint_path.is_file():
        raise FileNotFoundError("selected experiment must contain experiment.json and best_lstm.pt")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "selected" or manifest.get("smoke_test_subset"):
        raise ValueError("detection-dropout evaluation requires a full selected checkpoint")
    if manifest.get("selection", {}).get("test_used_for_selection") is not False:
        raise ValueError("selected model must be certified as test-independent")
    config = ExperimentConfig(**manifest["config"])
    if config.device != "cpu":
        raise ValueError("detection-dropout experiment requires the selected CPU checkpoint")

    checkpoint_hash = _file_sha256(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("selected_epoch") != manifest["selection"]["selected_epoch"]:
        raise ValueError("checkpoint selected epoch does not match the manifest")
    if checkpoint.get("selected_validation_ade") != manifest["selection"]["selected_validation_ade"]:
        raise ValueError("checkpoint validation ADE does not match the manifest")
    if checkpoint.get("model_config") != config.lstm_config().as_dict():
        raise ValueError("checkpoint model configuration does not match the manifest")
    model = LSTMTrajectoryPredictor(config.lstm_config())
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    normalization = PositionNormalization.from_dict(checkpoint["normalization"])

    validation_datasets = load_prediction_split(dataset_root, "val")
    candidate_windows = _split_windows(validation_datasets, config)
    clean_all = evaluate_models(
        candidate_windows, model, normalization, torch.device("cpu"), config.batch_size
    )
    per_seed: dict[int, dict[str, Any]] = {}
    aggregate_inputs: dict[str, list[dict[str, Any]]] = {
        severity: [] for severity in DROPOUT_RATES
    }
    representative_masks: dict[str, dict[tuple[str, int, int], bool]] = {}
    representative_windows: dict[str, ETHWindow] = {}

    for seed in CORRUPTION_SEEDS:
        seed_report: dict[str, Any] = {
            "seed": seed,
            "split": "val",
            "candidate_window_count": len(candidate_windows),
            "candidate_window_key_sha256": _window_key_digest(candidate_windows),
            "severity_results": {},
        }
        for severity, rate in DROPOUT_RATES.items():
            mask, dropout_stats = generate_dropout_mask(candidate_windows, rate, seed)
            corrupted_windows, imputation_stats = impute_dropout_windows(candidate_windows, mask)
            if not corrupted_windows:
                raise RuntimeError(f"all validation windows became unusable at {severity} severity")
            retained_indices = {
                (window.scene, window.split, window.pedestrian_id, window.start_frame)
                for window in corrupted_windows
            }
            clean_windows = tuple(
                window for window in candidate_windows
                if (window.scene, window.split, window.pedestrian_id, window.start_frame)
                in retained_indices
            )
            clean_results = evaluate_models(
                clean_windows, model, normalization, torch.device("cpu"), config.batch_size
            )
            noisy_results = evaluate_models(
                corrupted_windows, model, normalization, torch.device("cpu"), config.batch_size
            )
            signature = _window_key_digest(corrupted_windows)
            if signature != _window_key_digest(clean_windows):
                raise RuntimeError("dropout and matching clean baselines use different windows")
            if noisy_results["window_key_sha256"] != signature:
                raise RuntimeError("model evaluation changed retained window identity or ordering")
            seed_result = {
                "seed": seed,
                "candidate_window_count": len(candidate_windows),
                "evaluated_window_count": len(corrupted_windows),
                "unusable_window_count": imputation_stats["unusable_window_count"],
                "unusable_windows": imputation_stats["unusable_windows"],
                "window_key_sha256": signature,
                "dropout_statistics": dropout_stats,
                "imputation_statistics": imputation_stats,
                "clean_baseline_on_retained_windows": clean_results,
                "metrics": _add_degradation(noisy_results, clean_results),
            }
            seed_report["severity_results"][severity] = seed_result
            aggregate_inputs[severity].append(seed_result)
            if severity not in representative_masks:
                representative_masks[severity] = mask
                representative_windows[severity] = next(
                    window for window in candidate_windows
                    if any(mask[_observation_key(window.scene, point)] for point in window.observed)
                    and not all(mask[_observation_key(window.scene, point)] for point in window.observed)
                )
        per_seed[seed] = seed_report

    aggregates: dict[str, Any] = {}
    for severity, rate in DROPOUT_RATES.items():
        aggregates[severity] = {
            "configured_dropout_probability": rate,
            **_aggregate_severity(aggregate_inputs[severity]),
        }

    checkpoint_hash_after = _file_sha256(checkpoint_path)
    if checkpoint_hash_after != checkpoint_hash:
        raise RuntimeError("frozen checkpoint changed during detection-dropout evaluation")

    output_dir.mkdir(parents=True, exist_ok=True)
    per_seed_dir = output_dir / "per_seed"
    for seed, result in per_seed.items():
        _atomic_json(per_seed_dir / f"seed-{seed}.json", result)

    for severity, rate in DROPOUT_RATES.items():
        _plot_imputation_examples(
            output_dir / f"imputation-example-{severity}.png",
            representative_windows[severity],
            representative_masks[severity],
            rate,
        )
    _plot_severity_comparison(output_dir / "severity-vs-metrics.png", aggregates)

    source_files = [
        {
            "scene": dataset.scene,
            "path": dataset.source_file.as_posix(),
            "sha256": _file_sha256(dataset.source_file),
        }
        for dataset in validation_datasets
    ]
    experiment_config = {
        "protocol": {
            "split_evaluated": "val",
            "candidate_window_count": len(candidate_windows),
            "candidate_window_key_sha256": _window_key_digest(candidate_windows),
            "observation_len": config.observation_len,
            "prediction_horizon": config.horizon_len,
            "frame_step": config.frame_step,
            "stride": config.stride,
            "test_data_accessed": False,
            "training_data_used": False,
        },
        "dropout": {
            "mask": "Bernoulli independently sampled per unique observed position",
            "observation_identity": "(scene, pedestrian_id, frame_id)",
            "rates": DROPOUT_RATES,
            "seeds": list(CORRUPTION_SEEDS),
            "imputation_policy": (
                "linear interpolation between nearest valid observed positions; leading/trailing "
                "gaps use nearest valid observed position; all-missing windows are unusable"
            ),
            "future_ground_truth_modified": False,
            "clean_windows_modified": False,
            "only_observed_positions_corrupted": True,
        },
        "models": {
            "constant_velocity": "existing predictor; no training",
            "lstm_checkpoint": {
                "path": checkpoint_path.as_posix(),
                "sha256": checkpoint_hash,
                "selected_epoch": checkpoint["selected_epoch"],
                "selected_validation_ade": checkpoint["selected_validation_ade"],
                "config": checkpoint["model_config"],
            },
            "normalization": checkpoint["normalization"],
        },
        "clean_baseline_all_candidate_windows": clean_all,
        "dataset_source_files": {"validation": source_files, "train": "not opened", "test": "not opened"},
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "matplotlib": matplotlib.__version__,
            "device": "cpu",
        },
    }
    aggregated_report = {
        "split": "val",
        "distance_units": "raw dataset coordinate units; physical units unverified",
        "aggregation": (
            "Existing per-window ADE/FDE definitions are unchanged. Mean and population standard "
            "deviation are across three corruption seeds; each seed's clean baseline uses the "
            "same retained window subset as its corrupted evaluation."
        ),
        "candidate_window_count": len(candidate_windows),
        "candidate_window_key_sha256": _window_key_digest(candidate_windows),
        "clean_baseline_all_candidate_windows": clean_all,
        "severity_results": aggregates,
    }
    _atomic_json(output_dir / "experiment_config.json", experiment_config)
    _atomic_json(output_dir / "aggregated_results.json", aggregated_report)
    _write_summary_csv(output_dir / "summary_comparison.csv", aggregates)
    return aggregated_report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--experiment-dir", type=Path, default=DEFAULT_EXPERIMENT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = run_detection_dropout_experiment(
        args.dataset_root,
        args.experiment_dir,
        args.output_dir,
    )
    summary = {
        "output_dir": args.output_dir.as_posix(),
        "candidate_window_count": report["candidate_window_count"],
        "severity_results": {
            severity: {
                "rate": result["configured_dropout_probability"],
                "models": {
                    model: {
                        "ade": metrics["ade"],
                        "fde": metrics["fde"],
                    }
                    for model, metrics in result["models"].items()
                },
                "evaluated_windows": result["evaluated_window_count"],
                "unusable_windows": result["unusable_window_count"],
            }
            for severity, result in report["severity_results"].items()
        },
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
