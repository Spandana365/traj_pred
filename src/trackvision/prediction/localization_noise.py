"""Validation-only localization-noise experiments for frozen predictors."""

from __future__ import annotations

import argparse
import csv
import hashlib
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

from trackvision.io.eth import ETHObservation, ETHSplit, ETHWindow
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
from trackvision.prediction.lstm import LSTMTrajectoryPredictor


SEVERITY_MULTIPLIERS = {"low": 0.10, "medium": 0.25, "high": 0.50}
CORRUPTION_SEEDS = (42, 43, 44)
DEFAULT_EXPERIMENT_DIR = Path("runs/prediction/eth/lstm-seed42-h64-l2-scheduler")
DEFAULT_OUTPUT_DIR = Path("runs/error_injection/localization_noise")


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


def _observation_key(scene: str, observation: ETHObservation) -> tuple[str, int, int]:
    return scene, observation.pedestrian_id, observation.frame


def estimate_median_consecutive_displacement(
    training_datasets: tuple[ETHSplit, ...],
    frame_step: int,
) -> tuple[float, int]:
    """Measure median displacement between exact-cadence adjacent train observations."""
    if not training_datasets or any(dataset.split != "train" for dataset in training_datasets):
        raise ValueError("displacement scale must use non-empty training datasets only")
    if frame_step <= 0:
        raise ValueError("frame_step must be positive")

    distances: list[float] = []
    for dataset in training_datasets:
        tracks: dict[int, list[ETHObservation]] = {}
        for observation in dataset.observations:
            tracks.setdefault(observation.pedestrian_id, []).append(observation)
        for track in tracks.values():
            track.sort(key=lambda observation: observation.frame)
            for first, second in zip(track, track[1:]):
                if second.frame - first.frame != frame_step:
                    continue
                distances.append(
                    math.hypot(second.x - first.x, second.y - first.y)
                )
    if not distances:
        raise ValueError("training data has no consecutive observations at frame_step")
    return float(np.median(np.asarray(distances, dtype=np.float64))), len(distances)


def localization_noise_severities(
    median_displacement: float,
) -> dict[str, dict[str, float | str]]:
    if not math.isfinite(median_displacement) or median_displacement < 0:
        raise ValueError("median displacement must be finite and non-negative")
    return {
        name: {
            "multiplier": multiplier,
            "sigma": multiplier * median_displacement,
            "units": "raw dataset coordinate units",
            "interpretation": "exploratory relative severity, not a measured tracker-error distribution",
        }
        for name, multiplier in SEVERITY_MULTIPLIERS.items()
    }


def inject_localization_noise(
    windows: tuple[ETHWindow, ...],
    sigma: float,
    seed: int,
) -> tuple[tuple[ETHWindow, ...], dict[str, float | int]]:
    """Return copied windows with deterministic per-observation Gaussian noise."""
    if not windows:
        raise ValueError("windows must not be empty")
    if any(window.split != "val" for window in windows):
        raise ValueError("localization-noise evaluation accepts validation windows only")
    if not math.isfinite(sigma) or sigma < 0:
        raise ValueError("sigma must be finite and non-negative")
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
    if sigma == 0:
        noise = np.zeros((len(keys), 2), dtype=np.float64)
    else:
        noise = np.random.default_rng(seed).normal(
            loc=0.0, scale=sigma, size=(len(keys), 2)
        )
    noise_by_key = {key: noise[index] for index, key in enumerate(keys)}

    corrupted: list[ETHWindow] = []
    for window in windows:
        corrupted_observed = tuple(
            replace(
                observation,
                x=observation.x + float(noise_by_key[_observation_key(window.scene, observation)][0]),
                y=observation.y + float(noise_by_key[_observation_key(window.scene, observation)][1]),
            )
            for observation in window.observed
        )
        corrupted.append(replace(window, observed=corrupted_observed))

    radial = np.linalg.norm(noise, axis=1)
    statistics: dict[str, float | int] = {
        "unique_observation_count": len(keys),
        "window_observation_count": sum(len(window.observed) for window in windows),
        "configured_sigma": float(sigma),
        "mean_dx": float(noise[:, 0].mean()),
        "mean_dy": float(noise[:, 1].mean()),
        "std_dx": float(noise[:, 0].std()),
        "std_dy": float(noise[:, 1].std()),
        "mean_radial_displacement": float(radial.mean()),
        "std_radial_displacement": float(radial.std()),
        "rms_radial_displacement": float(np.sqrt(np.mean(np.square(radial)))),
    }
    return tuple(corrupted), statistics


def _degradation(noisy_value: float, clean_value: float) -> dict[str, float | None]:
    absolute = noisy_value - clean_value
    percentage = None if clean_value == 0 else 100.0 * absolute / abs(clean_value)
    return {"absolute": absolute, "percentage": percentage}


def _add_degradation(
    noisy_results: dict[str, Any],
    clean_results: dict[str, Any],
) -> dict[str, Any]:
    enriched: dict[str, Any] = {}
    for model_name in ("constant_velocity", "lstm"):
        noisy = noisy_results[model_name]
        clean = clean_results[model_name]
        enriched[model_name] = {
            **noisy,
            "degradation_vs_clean": {
                metric: _degradation(float(noisy[metric]), float(clean[metric]))
                for metric in ("ade", "fde")
            },
            "per_scene_degradation_vs_clean": {
                scene: {
                    metric: _degradation(
                        float(noisy["per_scene"][scene][metric]),
                        float(clean["per_scene"][scene][metric]),
                    )
                    for metric in ("ade", "fde")
                }
                for scene in noisy["per_scene"]
            },
            "per_timestep_degradation_vs_clean": [
                {
                    "timestep": noisy_step["timestep"],
                    **_degradation(
                        float(noisy_step["ade"]),
                        float(clean_step["ade"]),
                    ),
                }
                for noisy_step, clean_step in zip(
                    noisy["ade_by_timestep"], clean["ade_by_timestep"], strict=True
                )
            ],
        }
    return enriched


def _mean_std(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {"mean": float(array.mean()), "std": float(array.std())}


def _aggregate_severity(seed_results: list[dict[str, Any]]) -> dict[str, Any]:
    model_aggregates: dict[str, Any] = {}
    for model_name in ("constant_velocity", "lstm"):
        records = [item["metrics"][model_name] for item in seed_results]
        model_aggregates[model_name] = {
            metric: _mean_std([float(record[metric]) for record in records])
            for metric in ("ade", "fde")
        }
        model_aggregates[model_name]["degradation_vs_clean"] = {
            metric: {
                field: _mean_std(
                    [
                        float(record["degradation_vs_clean"][metric][field])
                        for record in records
                        if record["degradation_vs_clean"][metric][field] is not None
                    ]
                )
                for field in ("absolute", "percentage")
            }
            for metric in ("ade", "fde")
        }
        scenes = sorted(records[0]["per_scene"])
        model_aggregates[model_name]["per_scene"] = {
            scene: {
                metric: _mean_std(
                    [float(record["per_scene"][scene][metric]) for record in records]
                )
                for metric in ("ade", "fde")
            }
            for scene in scenes
        }
        model_aggregates[model_name]["per_scene_degradation_vs_clean"] = {
            scene: {
                metric: {
                    field: _mean_std(
                        [
                            float(record["per_scene_degradation_vs_clean"][scene][metric][field])
                            for record in records
                        ]
                    )
                    for field in ("absolute", "percentage")
                }
                for metric in ("ade", "fde")
            }
            for scene in scenes
        }
        timestep_count = len(records[0]["ade_by_timestep"])
        model_aggregates[model_name]["ade_by_timestep"] = [
            {
                "timestep": timestep + 1,
                "ade": _mean_std(
                    [
                        float(record["ade_by_timestep"][timestep]["ade"])
                        for record in records
                    ]
                ),
                "degradation_vs_clean": {
                    field: _mean_std(
                        [
                            float(record["per_timestep_degradation_vs_clean"][timestep][field])
                            for record in records
                        ]
                    )
                    for field in ("absolute", "percentage")
                },
            }
            for timestep in range(timestep_count)
        ]

    noise_fields = (
        "mean_dx",
        "mean_dy",
        "std_dx",
        "std_dy",
        "mean_radial_displacement",
        "std_radial_displacement",
        "rms_radial_displacement",
    )
    return {
        "seed_count": len(seed_results),
        "seeds": [item["seed"] for item in seed_results],
        "evaluated_windows": seed_results[0]["evaluated_windows"],
        "window_key_sha256": seed_results[0]["window_key_sha256"],
        "noise_statistics": {
            field: _mean_std(
                [float(item["noise_statistics"][field]) for item in seed_results]
            )
            for field in noise_fields
        },
        "models": model_aggregates,
    }


def _write_summary_csv(path: Path, clean: dict[str, Any], aggregate: dict[str, Any]) -> None:
    columns = (
        "severity",
        "sigma",
        "model",
        "ade_mean",
        "ade_std",
        "fde_mean",
        "fde_std",
        "ade_degradation_mean",
        "ade_degradation_std",
        "ade_degradation_percentage_mean",
        "ade_degradation_percentage_std",
        "fde_degradation_mean",
        "fde_degradation_std",
        "fde_degradation_percentage_mean",
        "fde_degradation_percentage_std",
    )
    with path.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=columns)
        writer.writeheader()
        for model_name in ("constant_velocity", "lstm"):
            writer.writerow({
                "severity": "clean",
                "sigma": 0.0,
                "model": model_name,
                "ade_mean": clean[model_name]["ade"],
                "ade_std": 0.0,
                "fde_mean": clean[model_name]["fde"],
                "fde_std": 0.0,
                "ade_degradation_mean": 0.0,
                "ade_degradation_std": 0.0,
                "ade_degradation_percentage_mean": 0.0,
                "ade_degradation_percentage_std": 0.0,
                "fde_degradation_mean": 0.0,
                "fde_degradation_std": 0.0,
                "fde_degradation_percentage_mean": 0.0,
                "fde_degradation_percentage_std": 0.0,
            })
        for severity, result in aggregate.items():
            for model_name, model in result["models"].items():
                ade_deg = model["degradation_vs_clean"]["ade"]
                fde_deg = model["degradation_vs_clean"]["fde"]
                writer.writerow({
                    "severity": severity,
                    "sigma": result["sigma"],
                    "model": model_name,
                    "ade_mean": model["ade"]["mean"],
                    "ade_std": model["ade"]["std"],
                    "fde_mean": model["fde"]["mean"],
                    "fde_std": model["fde"]["std"],
                    "ade_degradation_mean": ade_deg["absolute"]["mean"],
                    "ade_degradation_std": ade_deg["absolute"]["std"],
                    "ade_degradation_percentage_mean": ade_deg["percentage"]["mean"],
                    "ade_degradation_percentage_std": ade_deg["percentage"]["std"],
                    "fde_degradation_mean": fde_deg["absolute"]["mean"],
                    "fde_degradation_std": fde_deg["absolute"]["std"],
                    "fde_degradation_percentage_mean": fde_deg["percentage"]["mean"],
                    "fde_degradation_percentage_std": fde_deg["percentage"]["std"],
                })


def _write_visualization(
    path: Path,
    sample_window: ETHWindow,
    severities: dict[str, dict[str, float | str]],
    seed: int,
) -> None:
    clean_x = np.asarray([point.x for point in sample_window.observed])
    clean_y = np.asarray([point.y for point in sample_window.observed])
    future_x = np.asarray([point.x for point in sample_window.future])
    future_y = np.asarray([point.y for point in sample_window.future])
    figure, axes = plt.subplots(1, len(severities), figsize=(15, 5), sharex=True, sharey=True)
    for axis, (severity, settings) in zip(axes, severities.items(), strict=True):
        noisy_windows, _ = inject_localization_noise(
            (sample_window,), float(settings["sigma"]), seed
        )
        noisy = noisy_windows[0].observed
        axis.plot(clean_x, clean_y, "o-", label="Clean observed")
        axis.plot(
            [point.x for point in noisy],
            [point.y for point in noisy],
            "x--",
            label="Noisy observed",
        )
        axis.plot(future_x, future_y, "s-", label="Future ground truth")
        axis.set_title(f"{severity.title()} (sigma={settings['sigma']:.5g})")
        axis.set_xlabel("x (raw coordinate units)")
        axis.set_ylabel("y (raw coordinate units)")
        axis.grid(True, alpha=0.3)
        axis.set_aspect("equal", adjustable="box")
    axes[0].legend()
    figure.suptitle(
        f"Localization noise example: {sample_window.scene}, pedestrian "
        f"{sample_window.pedestrian_id}, seed {seed}"
    )
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def run_localization_noise_experiment(
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
        raise ValueError("localization-noise evaluation requires a full selected checkpoint")
    if manifest.get("selection", {}).get("test_used_for_selection") is not False:
        raise ValueError("selected model must be certified as test-independent")
    config = ExperimentConfig(**manifest["config"])
    if config.device != "cpu":
        raise ValueError("this localization-noise experiment is configured for CPU evaluation")

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

    training_datasets = load_prediction_split(dataset_root, "train")
    validation_datasets = load_prediction_split(dataset_root, "val")
    median_displacement, displacement_pair_count = estimate_median_consecutive_displacement(
        training_datasets, config.frame_step
    )
    severities = localization_noise_severities(median_displacement)
    validation_windows = _split_windows(validation_datasets, config)
    clean_results = evaluate_models(
        validation_windows, model, normalization, torch.device("cpu"), config.batch_size
    )

    by_seed: dict[int, dict[str, Any]] = {}
    severity_seed_results: dict[str, list[dict[str, Any]]] = {
        name: [] for name in severities
    }
    for seed in CORRUPTION_SEEDS:
        seed_result: dict[str, Any] = {
            "seed": seed,
            "split": "val",
            "evaluated_windows": len(validation_windows),
            "window_key_sha256": _window_key_digest(validation_windows),
            "clean": clean_results,
            "severities": {},
        }
        for severity, settings in severities.items():
            corrupted_windows, noise_statistics = inject_localization_noise(
                validation_windows, float(settings["sigma"]), seed
            )
            results = evaluate_models(
                corrupted_windows, model, normalization, torch.device("cpu"), config.batch_size
            )
            if results["window_key_sha256"] != clean_results["window_key_sha256"]:
                raise RuntimeError("noise injection changed validation-window identity or ordering")
            enriched = _add_degradation(results, clean_results)
            result = {
                "severity": severity,
                "multiplier": settings["multiplier"],
                "sigma": settings["sigma"],
                "units": settings["units"],
                "interpretation": settings["interpretation"],
                "noise_statistics": noise_statistics,
                "evaluated_windows": len(corrupted_windows),
                "window_key_sha256": results["window_key_sha256"],
                "metrics": enriched,
            }
            seed_result["severities"][severity] = result
            severity_seed_results[severity].append({
                "seed": seed,
                "clean": clean_results,
                "evaluated_windows": len(corrupted_windows),
                "window_key_sha256": results["window_key_sha256"],
                "noise_statistics": noise_statistics,
                "metrics": enriched,
            })
        by_seed[seed] = seed_result

    aggregates: dict[str, Any] = {}
    for severity, seed_results in severity_seed_results.items():
        aggregates[severity] = {
            "multiplier": severities[severity]["multiplier"],
            "sigma": severities[severity]["sigma"],
            "units": severities[severity]["units"],
            "interpretation": severities[severity]["interpretation"],
            **_aggregate_severity(seed_results),
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_visualization(
        output_dir / "trajectory_noise_examples.png",
        validation_windows[0],
        severities,
        CORRUPTION_SEEDS[0],
    )
    per_seed_dir = output_dir / "per_seed"
    for seed, result in by_seed.items():
        _atomic_json(per_seed_dir / f"seed-{seed}.json", result)

    training_source_files = [
        {
            "scene": dataset.scene,
            "path": dataset.source_file.as_posix(),
            "sha256": _file_sha256(dataset.source_file),
        }
        for dataset in training_datasets
    ]
    validation_source_files = [
        {
            "scene": dataset.scene,
            "path": dataset.source_file.as_posix(),
            "sha256": _file_sha256(dataset.source_file),
        }
        for dataset in validation_datasets
    ]
    experiment_config: dict[str, Any] = {
        "protocol": {
            "split_evaluated": "val",
            "observation_len": config.observation_len,
            "prediction_horizon": config.horizon_len,
            "frame_step": config.frame_step,
            "stride": config.stride,
            "training_split_used_only_for_noise_scale": "train",
            "test_data_accessed": False,
        },
        "noise_model": {
            "distribution": "independent isotropic Gaussian per observed position: N(0, sigma^2 I)",
            "consistent_overlap_identity": "(scene, pedestrian_id, frame_id)",
            "future_ground_truth_modified": False,
            "clean_windows_modified": False,
            "random_seeds": list(CORRUPTION_SEEDS),
            "training_median_consecutive_displacement": median_displacement,
            "training_displacement_pair_count": displacement_pair_count,
            "displacement_definition": (
                "median Euclidean displacement over unique adjacent training observations "
                f"whose frame difference is exactly {config.frame_step}"
            ),
            "sigma_by_severity": severities,
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
        "validation": {
            "evaluated_windows": len(validation_windows),
            "window_key_sha256": _window_key_digest(validation_windows),
            "clean_baselines": clean_results,
        },
        "dataset_source_files": {
            "train_for_noise_scale_only": training_source_files,
            "validation": validation_source_files,
            "test": "not opened",
        },
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "matplotlib": matplotlib.__version__,
            "device": "cpu",
        },
    }
    aggregate_report = {
        "split": "val",
        "distance_units": "raw dataset coordinate units; physical units unverified",
        "aggregation": (
            "ADE is averaged per window across 12 positions, then equally across windows; "
            "FDE is the final-timestep Euclidean distance averaged equally across windows. "
            "Mean and population standard deviation are across the three corruption seeds."
        ),
        "clean_baseline": clean_results,
        "severity_results": aggregates,
    }
    _atomic_json(output_dir / "experiment_config.json", experiment_config)
    _atomic_json(output_dir / "aggregated_results.json", aggregate_report)
    _write_summary_csv(output_dir / "summary_comparison.csv", clean_results, aggregates)
    if _file_sha256(checkpoint_path) != checkpoint_hash:
        raise RuntimeError("frozen checkpoint changed during localization-noise evaluation")
    return aggregate_report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--experiment-dir", type=Path, default=DEFAULT_EXPERIMENT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = run_localization_noise_experiment(
        args.dataset_root,
        args.experiment_dir,
        args.output_dir,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
