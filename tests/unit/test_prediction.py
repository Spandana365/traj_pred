import json
from pathlib import Path

import numpy as np
import pytest
import torch

import trackvision.prediction.experiment as experiment_module
from trackvision.evaluation.metrics import ade_fde
from trackvision.io.eth import ETHObservation, ETHSplit
from trackvision.prediction.constant_velocity import predict_constant_velocity
from trackvision.prediction.experiment import (
    ExperimentConfig,
    PositionNormalization,
    evaluate_frozen_checkpoint,
    evaluate_models,
    fit_training_normalization,
    train_experiment,
)
from trackvision.prediction.lstm import LSTMConfig, LSTMTrajectoryPredictor


def test_constant_velocity_stationary_horizontal_and_diagonal():
    stationary = np.zeros((8, 2), dtype=np.float64)
    horizontal = np.asarray([(float(index), 0.0) for index in range(8)])
    diagonal = np.asarray([(2.0 * index, -3.0 * index) for index in range(8)])

    assert predict_constant_velocity(stationary).shape == (12, 2)
    np.testing.assert_array_equal(predict_constant_velocity(stationary), np.zeros((12, 2)))
    np.testing.assert_array_equal(
        predict_constant_velocity(horizontal)[-1],
        np.asarray([19.0, 0.0]),
    )
    np.testing.assert_array_equal(
        predict_constant_velocity(diagonal)[-1],
        np.asarray([38.0, -57.0]),
    )


@pytest.mark.parametrize(
    "values, message",
    [
        (np.zeros((7, 2)), "shape"),
        (np.zeros((8, 3)), "shape"),
        (np.empty((0, 8, 2)), "empty"),
        (np.full((8, 2), np.nan), "finite"),
    ],
)
def test_constant_velocity_rejects_invalid_inputs(values, message):
    with pytest.raises(ValueError, match=message):
        predict_constant_velocity(values)


def test_constant_velocity_supports_batched_predictions():
    observed = np.stack((np.zeros((8, 2)), np.ones((8, 2))))

    predicted = predict_constant_velocity(observed)

    assert predicted.shape == (2, 12, 2)
    np.testing.assert_array_equal(predicted[0], np.zeros((12, 2)))
    np.testing.assert_array_equal(predicted[1], np.ones((12, 2)))


def test_ade_fde_match_hand_calculated_values_and_equal_window_weighting():
    target = np.zeros((2, 12, 2), dtype=np.float64)
    prediction = target.copy()
    prediction[0, :, 0] = 1.0
    prediction[1, :, 1] = 2.0
    prediction[1, -1, 0] = 3.0

    metrics = ade_fde(prediction, target)

    expected_ade = (1.0 + (11.0 * 2.0 + 1.0 * np.sqrt(13.0)) / 12.0) / 2.0
    expected_fde = (1.0 + np.sqrt(13.0)) / 2.0
    assert metrics.ade == pytest.approx(expected_ade)
    assert metrics.fde == pytest.approx(expected_fde)


@pytest.mark.parametrize(
    "prediction, target, message",
    [
        (np.zeros((12, 2)), np.zeros((11, 2)), "identical shapes"),
        (np.zeros((0, 2)), np.zeros((0, 2)), "horizon"),
        (np.empty((0, 12, 2)), np.empty((0, 12, 2)), "batch"),
        (np.full((12, 2), np.inf), np.zeros((12, 2)), "finite"),
        (np.zeros((12, 3)), np.zeros((12, 3)), "shape"),
    ],
)
def test_ade_fde_rejects_invalid_inputs(prediction, target, message):
    with pytest.raises(ValueError, match=message):
        ade_fde(prediction, target)


def test_lstm_shape_and_invalid_inputs():
    model = LSTMTrajectoryPredictor(LSTMConfig(hidden_size=8, num_layers=2, dropout=0.1))
    result = model(torch.zeros((3, 8, 2)))

    assert result.shape == (3, 12, 2)
    with pytest.raises(ValueError, match="shape"):
        model(torch.zeros((3, 7, 2)))
    with pytest.raises(ValueError, match="finite"):
        model(torch.full((1, 8, 2), torch.nan))


def _split(split: str, scene: str, offset: float = 0.0) -> ETHSplit:
    observations = tuple(
        ETHObservation(
            frame=frame * 10,
            pedestrian_id=1,
            x=offset + frame * 0.25,
            y=offset - frame * 0.5,
        )
        for frame in range(20)
    )
    return ETHSplit(scene, split, Path(f"{split}-{scene}.txt"), observations)


def test_normalization_uses_training_rows_only_and_handles_constant_axis():
    training = (_split("train", "HOTEL"),)
    validation = (_split("val", "HOTEL", offset=10000),)

    fitted = fit_training_normalization(training)

    expected_x = np.asarray([item.x for item in training[0].observations])
    assert fitted.fit_split == "train"
    assert fitted.fit_observation_count == 20
    assert fitted.mean[0] == pytest.approx(expected_x.mean())
    assert fitted.mean[0] < 100
    constant = ETHSplit(
        "HOTEL",
        "train",
        Path("constant.txt"),
        tuple(ETHObservation(frame, 1, 2.0, 3.0) for frame in range(20)),
    )
    constant_norm = fit_training_normalization((constant,))
    assert constant_norm.scale == (1.0, 1.0)
    assert validation[0].split == "val"


def test_experiment_defaults_keep_requested_training_configuration():
    config = ExperimentConfig()

    assert config.learning_rate == pytest.approx(0.001)
    assert config.batch_size == 256
    assert config.max_epochs == 100
    assert config.patience == 15
    assert config.seed == 42
    assert config.hidden_size == 64
    assert config.num_layers == 1
    assert config.dropout == 0.0


def _write_split(root: Path, split: str, scenes: tuple[str, ...], offset: float) -> None:
    directory = root / split
    directory.mkdir(parents=True, exist_ok=True)
    for scene_index, scene in enumerate(scenes):
        stem = {
            "HOTEL": "biwi_hotel",
            "ZARA1": "crowds_zara01",
            "ZARA2": "crowds_zara02",
            "ZARA3": "crowds_zara03",
            "STUDENTS001": "students001",
            "UNIV": "students003",
            "UNI_EXAMPLES": "uni_examples",
            "ETH": "biwi_eth",
        }[scene]
        filename = f"{stem}_{split}.txt" if split != "test" else f"{stem}.txt"
        lines = [
            f"{frame} 1 {offset + scene_index + frame / 10} "
            f"{offset - scene_index - frame / 20}"
            for frame in range(0, 200, 10)
        ]
        (directory / filename).write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_end_to_end_training_uses_train_val_only_and_frozen_test_checkpoint(tmp_path):
    dataset_root = tmp_path / "eth"
    train_scenes = (
        "HOTEL", "ZARA1", "ZARA2", "ZARA3", "STUDENTS001", "UNIV", "UNI_EXAMPLES",
    )
    _write_split(dataset_root, "train", train_scenes, 0.0)
    _write_split(dataset_root, "val", train_scenes, 10000.0)
    test_directory = dataset_root / "test"
    test_directory.mkdir()
    (test_directory / "biwi_eth.txt").write_text("deliberately malformed\n", encoding="utf-8")

    config = ExperimentConfig(
        hidden_size=8,
        batch_size=4,
        max_epochs=2,
        patience=2,
        seed=17,
    )
    experiment_dir = tmp_path / "selected"
    selected = train_experiment(dataset_root, experiment_dir, config)

    assert selected["smoke_test_subset"] is False
    assert (experiment_dir / "best_lstm.pt").is_file()
    manifest = json.loads((experiment_dir / "experiment.json").read_text(encoding="utf-8"))
    history = json.loads((experiment_dir / "training_history.json").read_text(encoding="utf-8"))
    validation_report = json.loads(
        (experiment_dir / "validation_results.json").read_text(encoding="utf-8")
    )
    checkpoint = torch.load(
        experiment_dir / "best_lstm.pt", map_location="cpu", weights_only=True
    )
    assert manifest["normalization"]["fit_split"] == "train"
    assert manifest["selection"]["test_used_for_selection"] is False
    assert manifest["scheduler"] == {
        "name": "ReduceLROnPlateau",
        "monitor": "validation_lstm_ade_raw_coordinate_units",
        "mode": "min",
        "factor": 0.5,
        "patience_epochs": 4,
    }
    assert manifest["splits"]["training_candidate_windows"] == 7
    assert manifest["splits"]["validation_candidate_windows"] == 7
    assert manifest["splits"]["test_candidate_windows"]["verified_during_model_selection"] is False
    assert len(history["history"]) >= 1
    assert all(
        {
            "learning_rate_used",
            "train_mse_normalized_coordinates",
            "train_ade_raw_coordinate_units",
            "validation_lstm_ade_raw_coordinate_units",
            "validation_lstm_fde_raw_coordinate_units",
            "validation_lstm_ade_by_scene",
            "validation_lstm_fde_by_scene",
            "validation_lstm_ade_by_timestep",
        }.issubset(epoch)
        for epoch in history["history"]
    )
    best_epoch = min(
        history["history"],
        key=lambda epoch: epoch["validation_lstm_ade_raw_coordinate_units"],
    )
    assert checkpoint["selected_epoch"] == best_epoch["epoch"]
    assert checkpoint["selected_validation_ade"] == pytest.approx(
        best_epoch["validation_lstm_ade_raw_coordinate_units"]
    )
    assert selected["validation_results"]["lstm"]["ade"] == pytest.approx(
        checkpoint["selected_validation_ade"]
    )
    lstm_metrics = validation_report["results"]["lstm"]
    assert all({"ade", "fde", "window_count"}.issubset(scene) for scene in lstm_metrics["per_scene"].values())
    assert [row["timestep"] for row in lstm_metrics["ade_by_timestep"]] == list(range(1, 13))
    assert len(lstm_metrics["ade_by_timestep"]) == 12

    test_path = dataset_root / "test" / "biwi_eth.txt"
    test_path.write_text(
        "\n".join(
            f"{frame} 1 {frame / 10} {-frame / 20}"
            for frame in range(0, 200, 10)
        ) + "\n",
        encoding="utf-8",
    )
    result = evaluate_frozen_checkpoint(
        experiment_dir,
        dataset_root,
        tmp_path / "test-results.json",
    )

    assert result["split"] == "test"
    assert result["evaluated_windows"] == 1
    assert result["test_used_for_training_or_selection"] is False
    assert (
        result["results"]["constant_velocity"]["window_key_sha256"]
        == result["results"]["lstm"]["window_key_sha256"]
    )
    assert (tmp_path / "test-results.json").is_file()


def test_scheduler_reduces_lr_and_early_stopping_uses_fifteen_bad_epochs(
    tmp_path, monkeypatch
):
    dataset_root = tmp_path / "eth"
    scenes = (
        "HOTEL", "ZARA1", "ZARA2", "ZARA3", "STUDENTS001", "UNIV", "UNI_EXAMPLES",
    )
    _write_split(dataset_root, "train", scenes, 0.0)
    _write_split(dataset_root, "val", scenes, 100.0)
    epoch_scores = [1.0, 0.8] + [0.81] * 30
    calls = 0

    def fake_evaluate_models(windows, model, normalization, device, batch_size):
        nonlocal calls
        score = epoch_scores[calls] if calls < len(epoch_scores) else 0.8
        calls += 1
        metrics = {
            "window_count": len(windows),
            "window_key_sha256": "fixture",
            "ade": score,
            "fde": score + 0.1,
            "per_scene": {"HOTEL": {"window_count": len(windows), "ade": score, "fde": score + 0.1}},
            "ade_by_timestep": [
                {"timestep": timestep, "ade": score}
                for timestep in range(1, 13)
            ],
        }
        return {
            "window_count": len(windows),
            "window_key_sha256": "fixture",
            "constant_velocity": metrics,
            "lstm": metrics,
        }

    monkeypatch.setattr(experiment_module, "evaluate_models", fake_evaluate_models)
    output_dir = tmp_path / "scheduler"
    train_experiment(
        dataset_root,
        output_dir,
        ExperimentConfig(hidden_size=4, batch_size=8, max_epochs=100, patience=15),
    )

    history = json.loads((output_dir / "training_history.json").read_text(encoding="utf-8"))
    epochs = history["history"]
    assert history["epochs_completed"] == 17
    assert history["early_stopped"] is True
    assert [epoch["learning_rate_used"] for epoch in epochs[:7]] == pytest.approx([0.001] * 7)
    assert epochs[7]["learning_rate_used"] == pytest.approx(0.0005)
    assert epochs[12]["learning_rate_used"] == pytest.approx(0.00025)
    checkpoint = torch.load(
        output_dir / "best_lstm.pt", map_location="cpu", weights_only=True
    )
    assert checkpoint["selected_epoch"] == 2
    assert checkpoint["selected_validation_ade"] == pytest.approx(0.8)


def test_final_test_command_rejects_smoke_subset_checkpoints(tmp_path):
    dataset_root = tmp_path / "eth"
    scenes = (
        "HOTEL", "ZARA1", "ZARA2", "ZARA3", "STUDENTS001", "UNIV", "UNI_EXAMPLES",
    )
    _write_split(dataset_root, "train", scenes, 0.0)
    _write_split(dataset_root, "val", scenes, 0.0)
    experiment_dir = tmp_path / "smoke"
    train_experiment(
        dataset_root,
        experiment_dir,
        ExperimentConfig(hidden_size=4, max_epochs=1, batch_size=8),
        max_train_windows=2,
        max_val_windows=2,
    )

    with pytest.raises(ValueError, match="full, selected"):
        evaluate_frozen_checkpoint(
            experiment_dir, dataset_root, tmp_path / "unused.json"
        )


def test_cv_and_lstm_evaluation_share_window_selection_and_serialize_deterministically():
    from trackvision.io.eth import build_eth_windows

    windows = build_eth_windows(_split("val", "HOTEL"), 10)
    normalization = PositionNormalization((0.0, 0.0), (1.0, 1.0), "train", 20)
    model = LSTMTrajectoryPredictor(LSTMConfig(hidden_size=4))
    result_one = evaluate_models(windows, model, normalization, torch.device("cpu"), 8)
    result_two = evaluate_models(windows, model, normalization, torch.device("cpu"), 8)

    assert result_one["window_key_sha256"] == result_one["constant_velocity"]["window_key_sha256"]
    assert result_one["window_key_sha256"] == result_one["lstm"]["window_key_sha256"]
    assert [row["timestep"] for row in result_one["lstm"]["ade_by_timestep"]] == list(range(1, 13))
    assert result_one["lstm"]["per_scene"]["HOTEL"]["window_count"] > 0
    assert json.dumps(result_one, sort_keys=True) == json.dumps(result_two, sort_keys=True)
