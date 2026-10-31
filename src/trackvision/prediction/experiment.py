"""Reproducible ETH Constant Velocity and LSTM training/evaluation commands."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import random
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from trackvision.evaluation.metrics import ade_fde
from trackvision.io.eth import (
    ETHSplit,
    ETHWindow,
    SOCIAL_STGCNN_SCENES,
    build_eth_windows,
    read_social_stgcnn_file,
)
from trackvision.io.eth_audit import EXPECTED_SOCIAL_STGCNN_SCENES_BY_SPLIT
from trackvision.prediction.constant_velocity import predict_constant_velocity
from trackvision.prediction.lstm import LSTMConfig, LSTMTrajectoryPredictor


DEFAULT_DATASET_ROOT = Path("data/eth/Social-STGCNN/datasets/eth")
_SCENE_STEMS = {scene: stem for stem, scene in SOCIAL_STGCNN_SCENES.items()}
_TRAIN_SCENES = EXPECTED_SOCIAL_STGCNN_SCENES_BY_SPLIT["train"]
_TEST_SCENES = EXPECTED_SOCIAL_STGCNN_SCENES_BY_SPLIT["test"]


@dataclass(frozen=True)
class ExperimentConfig:
    """Frozen protocol and training settings recorded with each selected checkpoint."""

    frame_step: int = 10
    observation_len: int = 8
    horizon_len: int = 12
    stride: int = 1
    hidden_size: int = 64
    num_layers: int = 1
    dropout: float = 0.0
    learning_rate: float = 1e-3
    batch_size: int = 256
    max_epochs: int = 100
    patience: int = 15
    min_delta: float = 0.0
    seed: int = 42
    device: str = "cpu"

    def __post_init__(self) -> None:
        if self.frame_step <= 0 or self.observation_len != 8 or self.horizon_len != 12:
            raise ValueError("protocol requires positive frame_step and fixed 8/12 lengths")
        if self.stride <= 0 or self.batch_size <= 0 or self.max_epochs <= 0 or self.patience <= 0:
            raise ValueError("stride, batch_size, max_epochs, and patience must be positive")
        if self.hidden_size <= 0 or self.num_layers <= 0:
            raise ValueError("hidden_size and num_layers must be positive")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive and finite")
        if not math.isfinite(self.dropout) or not 0 <= self.dropout < 1:
            raise ValueError("dropout must be finite and in [0, 1)")
        if not math.isfinite(self.min_delta) or self.min_delta < 0:
            raise ValueError("min_delta must be finite and non-negative")
        if self.device != "cpu" and not self.device.startswith("cuda"):
            raise ValueError("device must be 'cpu' or a CUDA device name")

    def lstm_config(self) -> LSTMConfig:
        return LSTMConfig(
            observation_len=self.observation_len,
            horizon_len=self.horizon_len,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
        )


@dataclass(frozen=True)
class PositionNormalization:
    """Per-coordinate training-only standardization statistics."""

    mean: tuple[float, float]
    scale: tuple[float, float]
    fit_split: str = "train"
    fit_observation_count: int = 0

    def transform(self, values: NDArray[np.float64]) -> NDArray[np.float64]:
        return (values - np.asarray(self.mean)) / np.asarray(self.scale)

    def inverse_transform(self, values: NDArray[np.float64]) -> NDArray[np.float64]:
        return values * np.asarray(self.scale) + np.asarray(self.mean)

    def as_dict(self) -> dict[str, object]:
        return {
            "mean": list(self.mean),
            "scale": list(self.scale),
            "fit_split": self.fit_split,
            "fit_observation_count": self.fit_observation_count,
            "method": "per-coordinate population mean/std over unique training-split rows",
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> PositionNormalization:
        return cls(
            mean=(float(value["mean"][0]), float(value["mean"][1])),
            scale=(float(value["scale"][0]), float(value["scale"][1])),
            fit_split=str(value["fit_split"]),
            fit_observation_count=int(value["fit_observation_count"]),
        )


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_path(dataset_root: Path, split: str, scene: str) -> Path:
    stem = _SCENE_STEMS[scene]
    filename = f"{stem}_{split}.txt" if split != "test" else f"{stem}.txt"
    return dataset_root / split / filename


def load_prediction_split(dataset_root: Path, split: str) -> tuple[ETHSplit, ...]:
    """Load exactly one expected split; never opens files from the other split directories."""
    expected_scenes = {
        "train": _TRAIN_SCENES,
        "val": EXPECTED_SOCIAL_STGCNN_SCENES_BY_SPLIT["val"],
        "test": _TEST_SCENES,
    }
    if split not in expected_scenes:
        raise ValueError(f"split must be one of {tuple(expected_scenes)}")
    result: list[ETHSplit] = []
    split_directory = dataset_root / split
    if not split_directory.is_dir():
        raise FileNotFoundError(f"required split directory does not exist: {split_directory}")
    expected_names = {
        _source_path(dataset_root, split, scene).name
        for scene in expected_scenes[split]
    }
    for path in sorted(split_directory.iterdir(), key=lambda item: item.name):
        if path.is_dir() or path.name not in expected_names:
            raise ValueError(f"unexpected entry in {split} split: {path}")
    for scene in expected_scenes[split]:
        path = _source_path(dataset_root, split, scene)
        if not path.is_file():
            raise FileNotFoundError(f"required {split} file does not exist: {path}")
        result.append(read_social_stgcnn_file(path, scene, split))
    return tuple(result)


def _split_windows(
    datasets: tuple[ETHSplit, ...],
    config: ExperimentConfig,
) -> tuple[ETHWindow, ...]:
    windows = tuple(
        window
        for dataset in datasets
        for window in build_eth_windows(
            dataset,
            frame_step=config.frame_step,
            history_len=config.observation_len,
            horizon_len=config.horizon_len,
            stride=config.stride,
        )
    )
    if not windows:
        raise ValueError(f"no valid 8/12 windows found in {datasets[0].split} split")
    return windows


def _window_arrays(
    windows: tuple[ETHWindow, ...],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    observed = np.asarray(
        [[(point.x, point.y) for point in window.observed] for window in windows],
        dtype=np.float64,
    )
    future = np.asarray(
        [[(point.x, point.y) for point in window.future] for window in windows],
        dtype=np.float64,
    )
    return observed, future


def fit_training_normalization(
    training_datasets: tuple[ETHSplit, ...],
) -> PositionNormalization:
    """Fit coordinate statistics on unique validated rows from train files only."""
    if not training_datasets or any(dataset.split != "train" for dataset in training_datasets):
        raise ValueError("normalization can only be fit from non-empty train datasets")
    positions = np.asarray(
        [(item.x, item.y) for dataset in training_datasets for item in dataset.observations],
        dtype=np.float64,
    )
    if positions.size == 0 or not np.isfinite(positions).all():
        raise ValueError("training observations must be non-empty and finite")
    mean = positions.mean(axis=0)
    scale = positions.std(axis=0)
    scale[scale == 0.0] = 1.0
    return PositionNormalization(
        mean=(float(mean[0]), float(mean[1])),
        scale=(float(scale[0]), float(scale[1])),
        fit_observation_count=len(positions),
    )


def _set_determinism(seed: int, device: str) -> torch.device:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but CUDA is unavailable: {device}")
    return torch.device(device)


def _window_key_digest(windows: tuple[ETHWindow, ...]) -> str:
    keys = [
        (window.scene, window.split, window.pedestrian_id, window.start_frame)
        for window in windows
    ]
    encoded = json.dumps(keys, separators=(",", ":"), ensure_ascii=True).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _metric_record(
    windows: tuple[ETHWindow, ...],
    predicted: NDArray[np.float64],
    target: NDArray[np.float64],
) -> dict[str, object]:
    per_scene: dict[str, dict[str, object]] = {}
    scene_names = sorted({window.scene for window in windows})
    for scene in scene_names:
        indices = [index for index, window in enumerate(windows) if window.scene == scene]
        scene_metrics = ade_fde(predicted[indices], target[indices])
        per_scene[scene] = {
            "window_count": len(indices),
            "ade": scene_metrics.ade,
            "fde": scene_metrics.fde,
        }
    overall = ade_fde(predicted, target)
    distances = np.linalg.norm(predicted - target, axis=-1)
    return {
        "window_count": len(windows),
        "window_key_sha256": _window_key_digest(windows),
        "aggregation": (
            "Compute ADE per window as the mean Euclidean distance across its 12 positions; "
            "compute FDE per window at position 12; average each metric equally over windows."
        ),
        "distance_units": "raw dataset coordinate units; physical units unverified",
        "ade": overall.ade,
        "fde": overall.fde,
        "ade_by_timestep": [
            {"timestep": timestep + 1, "ade": float(distances[:, timestep].mean())}
            for timestep in range(distances.shape[1])
        ],
        "per_scene": per_scene,
    }


@torch.no_grad()
def _predict_lstm(
    windows: tuple[ETHWindow, ...],
    model: LSTMTrajectoryPredictor,
    normalization: PositionNormalization,
    device: torch.device,
    batch_size: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    if not windows:
        raise ValueError("prediction windows must not be empty")
    observed, future = _window_arrays(windows)
    model.eval()
    normalized_observed = normalization.transform(observed)
    predictions: list[NDArray[np.float64]] = []
    for start in range(0, len(windows), batch_size):
        batch = torch.as_tensor(
            normalized_observed[start:start + batch_size],
            dtype=torch.float32,
            device=device,
        )
        normalized_prediction = model(batch).cpu().numpy().astype(np.float64)
        predictions.append(normalization.inverse_transform(normalized_prediction))
    return np.concatenate(predictions, axis=0), future


@torch.no_grad()
def evaluate_models(
    windows: tuple[ETHWindow, ...],
    model: LSTMTrajectoryPredictor,
    normalization: PositionNormalization,
    device: torch.device,
    batch_size: int,
) -> dict[str, object]:
    """Evaluate CV and LSTM over precisely the same ordered windows."""
    if not windows:
        raise ValueError("evaluation windows must not be empty")
    observed, future = _window_arrays(windows)
    cv_predictions = predict_constant_velocity(observed)

    lstm_predictions, _ = _predict_lstm(
        windows, model, normalization, device, batch_size
    )
    return {
        "window_count": len(windows),
        "window_key_sha256": _window_key_digest(windows),
        "constant_velocity": _metric_record(windows, cv_predictions, future),
        "lstm": _metric_record(windows, lstm_predictions, future),
    }


def _save_best_checkpoint(
    checkpoint_path: Path,
    model: LSTMTrajectoryPredictor,
    config: ExperimentConfig,
    normalization: PositionNormalization,
    epoch: int,
    validation_ade: float,
) -> None:
    torch.save(
        {
            "format_version": 1,
            "model_config": config.lstm_config().as_dict(),
            "state_dict": model.state_dict(),
            "selected_epoch": epoch,
            "selected_validation_ade": validation_ade,
            "normalization": normalization.as_dict(),
        },
        checkpoint_path,
    )


def train_experiment(
    dataset_root: Path,
    output_dir: Path,
    config: ExperimentConfig,
    max_train_windows: int | None = None,
    max_val_windows: int | None = None,
) -> dict[str, object]:
    """Train on train only, select by validation ADE, save a frozen checkpoint and reports."""
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output_dir}")
    if max_train_windows is not None and max_train_windows <= 0:
        raise ValueError("max_train_windows must be positive")
    if max_val_windows is not None and max_val_windows <= 0:
        raise ValueError("max_val_windows must be positive")

    train_datasets = load_prediction_split(Path(dataset_root), "train")
    validation_datasets = load_prediction_split(Path(dataset_root), "val")
    all_train_windows = _split_windows(train_datasets, config)
    all_validation_windows = _split_windows(validation_datasets, config)
    smoke_test = max_train_windows is not None or max_val_windows is not None
    train_windows = all_train_windows[:max_train_windows] if max_train_windows else all_train_windows
    validation_windows = (
        all_validation_windows[:max_val_windows] if max_val_windows else all_validation_windows
    )
    normalization = fit_training_normalization(train_datasets)
    device = _set_determinism(config.seed, config.device)
    model = LSTMTrajectoryPredictor(config.lstm_config()).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=4,
    )
    criterion = nn.MSELoss()

    observed, future = _window_arrays(train_windows)
    normalized_observed = normalization.transform(observed)
    normalized_future = normalization.transform(future)
    dataset = TensorDataset(
        torch.as_tensor(normalized_observed, dtype=torch.float32),
        torch.as_tensor(normalized_future, dtype=torch.float32),
    )
    generator = torch.Generator()
    generator.manual_seed(config.seed)
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        generator=generator,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "best_lstm.pt"
    history: list[dict[str, float | int]] = []
    best_validation_ade = math.inf
    best_epoch = 0
    epochs_without_improvement = 0
    for epoch in range(1, config.max_epochs + 1):
        learning_rate_used = float(optimizer.param_groups[0]["lr"])
        model.train()
        loss_sum = 0.0
        example_count = 0
        for observed_batch, future_batch in loader:
            observed_batch = observed_batch.to(device)
            future_batch = future_batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(observed_batch)
            loss = criterion(prediction, future_batch)
            loss.backward()
            optimizer.step()
            batch_count = observed_batch.shape[0]
            loss_sum += float(loss.item()) * batch_count
            example_count += batch_count

        validation = evaluate_models(
            validation_windows, model, normalization, device, config.batch_size
        )
        validation_lstm_ade = float(validation["lstm"]["ade"])
        train_predictions, train_targets = _predict_lstm(
            train_windows, model, normalization, device, config.batch_size
        )
        training_ade = ade_fde(train_predictions, train_targets).ade
        validation_lstm = validation["lstm"]
        validation_scenes = validation_lstm["per_scene"]
        history.append({
            "epoch": epoch,
            "learning_rate_used": learning_rate_used,
            "train_mse_normalized_coordinates": loss_sum / example_count,
            "train_ade_raw_coordinate_units": training_ade,
            "validation_lstm_ade_raw_coordinate_units": validation_lstm_ade,
            "validation_lstm_fde_raw_coordinate_units": float(validation_lstm["fde"]),
            "validation_lstm_ade_by_scene": {
                scene: float(metrics["ade"])
                for scene, metrics in validation_scenes.items()
            },
            "validation_lstm_fde_by_scene": {
                scene: float(metrics["fde"])
                for scene, metrics in validation_scenes.items()
            },
            "validation_lstm_ade_by_timestep": validation_lstm["ade_by_timestep"],
        })
        scheduler.step(validation_lstm_ade)
        if validation_lstm_ade < best_validation_ade - config.min_delta:
            best_validation_ade = validation_lstm_ade
            best_epoch = epoch
            epochs_without_improvement = 0
            _save_best_checkpoint(
                checkpoint_path,
                model,
                config,
                normalization,
                epoch,
                validation_lstm_ade,
            )
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= config.patience:
            break

    if best_epoch == 0:
        raise RuntimeError("training completed without selecting a validation checkpoint")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    validation_results = evaluate_models(
        validation_windows, model, normalization, device, config.batch_size
    )
    source_files = [
        {
            "split": dataset_item.split,
            "scene": dataset_item.scene,
            "path": dataset_item.source_file.as_posix(),
            "sha256": _file_sha256(dataset_item.source_file),
        }
        for dataset_item in (*train_datasets, *validation_datasets)
    ]
    manifest: dict[str, object] = {
        "format_version": 1,
        "status": "selected",
        "smoke_test_subset": smoke_test,
        "config": asdict(config),
        "model": {
            "name": "LSTM",
            "architecture": (
                "batch-first LSTM encoder; final hidden sequence output projected directly "
                "to 12 two-dimensional future positions"
            ),
            "config": config.lstm_config().as_dict(),
            "loss": "mean squared error over normalized absolute coordinates",
            "optimizer": "Adam",
        },
        "scheduler": {
            "name": "ReduceLROnPlateau",
            "monitor": "validation_lstm_ade_raw_coordinate_units",
            "mode": "min",
            "factor": 0.5,
            "patience_epochs": 4,
        },
        "normalization": normalization.as_dict(),
        "selection": {
            "criterion": "minimum validation mean per-window ADE in raw coordinate units",
            "min_delta": config.min_delta,
            "patience_epochs": config.patience,
            "selected_epoch": best_epoch,
            "selected_validation_ade": best_validation_ade,
            "test_used_for_selection": False,
        },
        "splits": {
            "training": "train",
            "validation": "val",
            "test": "test; not opened by training or selection",
            "training_candidate_windows": len(all_train_windows),
            "training_windows_used": len(train_windows),
            "validation_candidate_windows": len(all_validation_windows),
            "validation_windows_used": len(validation_windows),
            "test_candidate_windows": {
                "last_audited_reference": 364,
                "verified_during_model_selection": False,
            },
            "training_windows_sha256": _window_key_digest(train_windows),
            "validation_windows_sha256": _window_key_digest(validation_windows),
        },
        "exclusions": {
            "invalid_rows": 0,
            "windows_excluded_after_validation": 0,
            "smoke_subset_applied": smoke_test,
            "note": "Malformed files fail loading; no rows are silently dropped.",
        },
        "dataset_source_files": source_files,
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "device": str(device),
            "cuda_available": torch.cuda.is_available(),
        },
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    _atomic_json(output_dir / "experiment.json", manifest)
    _atomic_json(output_dir / "training_history.json", {
        "epochs_completed": len(history),
        "history": history,
        "early_stopped": len(history) < config.max_epochs,
    })
    _atomic_json(output_dir / "validation_results.json", {
        "split": "val",
        "selection_source": "validation",
        "model_selection": "best LSTM selected by validation ADE",
        "results": validation_results,
    })
    return {
        "experiment_dir": output_dir.as_posix(),
        "checkpoint": checkpoint_path.as_posix(),
        "selected_epoch": best_epoch,
        "best_validation_ade": best_validation_ade,
        "validation_results": validation_results,
        "smoke_test_subset": smoke_test,
    }


def evaluate_frozen_checkpoint(
    experiment_dir: Path,
    dataset_root: Path,
    output_path: Path,
) -> dict[str, object]:
    """Evaluate CV and a previously selected LSTM checkpoint on test only."""
    experiment_dir = Path(experiment_dir)
    output_path = Path(output_path)
    manifest_path = experiment_dir / "experiment.json"
    checkpoint_path = experiment_dir / "best_lstm.pt"
    if not manifest_path.is_file() or not checkpoint_path.is_file():
        raise FileNotFoundError("experiment must contain experiment.json and best_lstm.pt")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "selected" or manifest.get("smoke_test_subset"):
        raise ValueError("final test evaluation requires a full, selected training experiment")
    if manifest.get("selection", {}).get("test_used_for_selection") is not False:
        raise ValueError("experiment manifest does not certify test-independent selection")
    config = ExperimentConfig(**manifest["config"])
    device = _set_determinism(config.seed, config.device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if checkpoint.get("format_version") != 1:
        raise ValueError("unsupported checkpoint format")
    if checkpoint.get("selected_epoch") != manifest["selection"]["selected_epoch"]:
        raise ValueError("checkpoint selected epoch does not match experiment manifest")
    normalization = PositionNormalization.from_dict(checkpoint["normalization"])
    model = LSTMTrajectoryPredictor(config.lstm_config()).to(device)
    model.load_state_dict(checkpoint["state_dict"])

    test_datasets = load_prediction_split(Path(dataset_root), "test")
    test_windows = _split_windows(test_datasets, config)
    results = evaluate_models(test_windows, model, normalization, device, config.batch_size)
    report: dict[str, object] = {
        "split": "test",
        "evaluation_type": "limited held-out-scene evaluation",
        "scene_scope": [dataset_item.scene for dataset_item in test_datasets],
        "checkpoint": {
            "path": checkpoint_path.as_posix(),
            "sha256": _file_sha256(checkpoint_path),
            "selected_epoch": checkpoint["selected_epoch"],
            "selected_validation_ade": checkpoint["selected_validation_ade"],
        },
        "config": manifest["config"],
        "normalization": manifest["normalization"],
        "configuration_sha256": hashlib.sha256(
            json.dumps(manifest["config"], sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "test_used_for_training_or_selection": False,
        "candidate_windows": len(test_windows),
        "evaluated_windows": len(test_windows),
        "exclusions": {
            "invalid_rows": 0,
            "windows_excluded_after_validation": 0,
        },
        "dataset_source_files": [
            {
                "split": item.split,
                "scene": item.scene,
                "path": item.source_file.as_posix(),
                "sha256": _file_sha256(item.source_file),
            }
            for item in test_datasets
        ],
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "device": str(device),
        },
        "results": results,
    }
    _atomic_json(output_path, report)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train", help="train and select an LSTM using train/val only")
    train.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    train.add_argument("--output-dir", type=Path, required=True)
    train.add_argument("--frame-step", type=int, default=10)
    train.add_argument("--hidden-size", type=int, default=64)
    train.add_argument("--num-layers", type=int, default=1)
    train.add_argument("--dropout", type=float, default=0.0)
    train.add_argument("--learning-rate", type=float, default=1e-3)
    train.add_argument("--batch-size", type=int, default=256)
    train.add_argument("--max-epochs", type=int, default=100)
    train.add_argument("--patience", type=int, default=15)
    train.add_argument("--min-delta", type=float, default=0.0)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--device", default="cpu")
    train.add_argument("--smoke", action="store_true", help="train on small train/val subsets; cannot test")
    evaluate = commands.add_parser(
        "evaluate", help="evaluate a frozen full-training checkpoint on test only"
    )
    evaluate.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    evaluate.add_argument("--experiment-dir", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "train":
        config = ExperimentConfig(
            frame_step=args.frame_step,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            dropout=args.dropout,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            max_epochs=args.max_epochs,
            patience=args.patience,
            min_delta=args.min_delta,
            seed=args.seed,
            device=args.device,
        )
        result = train_experiment(
            args.dataset_root,
            args.output_dir,
            config,
            max_train_windows=128 if args.smoke else None,
            max_val_windows=64 if args.smoke else None,
        )
    else:
        result = evaluate_frozen_checkpoint(
            args.experiment_dir,
            args.dataset_root,
            args.output,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
