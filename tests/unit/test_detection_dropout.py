import numpy as np
import pytest
import torch

from trackvision.io.eth import ETHObservation, ETHWindow
from trackvision.prediction.detection_dropout import (
    _window_dropout_mask,
    generate_dropout_mask,
    impute_dropout_windows,
)
from trackvision.prediction.experiment import PositionNormalization, evaluate_models
from trackvision.prediction.lstm import LSTMConfig, LSTMTrajectoryPredictor


def _window(start: int = 0, pedestrian_id: int = 1) -> ETHWindow:
    points = tuple(
        ETHObservation(
            frame=index * 10,
            pedestrian_id=pedestrian_id,
            x=float(index),
            y=float(index * index),
        )
        for index in range(start, start + 20)
    )
    return ETHWindow(
        scene="HOTEL",
        split="val",
        pedestrian_id=pedestrian_id,
        start_frame=points[0].frame,
        observed=points[:8],
        future=points[8:],
    )


def test_dropout_mask_is_reproducible_and_seeded():
    windows = tuple(_window(pedestrian_id=index) for index in range(500))

    mask_a, stats_a = generate_dropout_mask(windows, rate=0.2, seed=42)
    mask_b, stats_b = generate_dropout_mask(windows, rate=0.2, seed=42)
    mask_c, _ = generate_dropout_mask(windows, rate=0.2, seed=43)

    assert mask_a == mask_b
    assert stats_a == stats_b
    assert mask_a != mask_c
    assert stats_a["unique_observation_count"] == 4000
    assert stats_a["actual_missing_percentage_unique"] == pytest.approx(
        100.0 * stats_a["dropped_unique_observation_count"] / 4000
    )


def test_overlapping_windows_share_decisions_for_same_observation_identity():
    windows = (_window(start=0), _window(start=1))

    mask, _ = generate_dropout_mask(windows, rate=0.3, seed=42)
    first_frames = {point.frame for point in windows[0].observed}
    second_frames = {point.frame for point in windows[1].observed}
    shared = first_frames & second_frames
    first_decisions = {
        point.frame: bool(decision)
        for point, decision in zip(
            windows[0].observed, _window_dropout_mask(windows[0], mask), strict=True
        )
    }
    second_decisions = {
        point.frame: bool(decision)
        for point, decision in zip(
            windows[1].observed, _window_dropout_mask(windows[1], mask), strict=True
        )
    }

    assert shared
    assert all(first_decisions[frame] == second_decisions[frame] for frame in shared)
    assert all(
        first_decisions[frame] == mask[("HOTEL", 1, frame)]
        for frame in first_frames
    )


def test_internal_and_boundary_missing_positions_are_interpolated():
    window = _window()
    mask = {
        ("HOTEL", 1, point.frame): index in {0, 3, 4, 7}
        for index, point in enumerate(window.observed)
    }

    imputed_windows, stats = impute_dropout_windows((window,), mask)

    assert stats["unusable_window_count"] == 0
    imputed = imputed_windows[0].observed
    # Leading and trailing omissions copy the nearest available point.
    assert (imputed[0].x, imputed[0].y) == (imputed[1].x, imputed[1].y)
    assert (imputed[7].x, imputed[7].y) == (imputed[6].x, imputed[6].y)
    # An internal two-position gap is interpolated between indices 2 and 5.
    assert (imputed[3].x, imputed[3].y) == pytest.approx((3.0, 11.0))
    assert (imputed[4].x, imputed[4].y) == pytest.approx((4.0, 18.0))
    assert stats["longest_consecutive_missing_run"] == 2
    assert imputed_windows[0].future == window.future


def test_all_missing_window_is_explicitly_unusable_not_zero_filled():
    window = _window()
    mask = {
        ("HOTEL", 1, point.frame): True
        for point in window.observed
    }

    retained, stats = impute_dropout_windows((window,), mask)

    assert retained == ()
    assert stats["candidate_window_count"] == 1
    assert stats["evaluated_window_count"] == 0
    assert stats["unusable_window_count"] == 1
    assert stats["unusable_windows"][0]["reason"] == "all_8_observed_positions_missing"
    assert stats["longest_consecutive_missing_run"] == 8


def test_zero_dropout_is_exact_and_inputs_and_future_remain_unchanged():
    windows = (_window(), _window(pedestrian_id=2))
    original_observed = tuple(window.observed for window in windows)
    original_future = tuple(window.future for window in windows)

    mask, dropout_stats = generate_dropout_mask(windows, rate=0, seed=42)
    imputed, imputation_stats = impute_dropout_windows(windows, mask)

    assert not any(mask.values())
    assert tuple(window.observed for window in imputed) == original_observed
    assert tuple(window.future for window in imputed) == original_future
    assert tuple(window.observed for window in windows) == original_observed
    assert tuple(window.future for window in windows) == original_future
    assert dropout_stats["actual_missing_percentage_unique"] == 0
    assert imputation_stats["actual_missing_percentage_window_occurrences"] == 0


def test_both_predictors_evaluate_identical_retained_windows():
    windows = (_window(), _window(start=1), _window(pedestrian_id=2))
    mask, _ = generate_dropout_mask(windows, rate=0.25, seed=44)
    corrupted, stats = impute_dropout_windows(windows, mask)
    assert corrupted

    model = LSTMTrajectoryPredictor(LSTMConfig(hidden_size=4))
    normalization = PositionNormalization((0.0, 0.0), (1.0, 1.0), "train", 1)
    result = evaluate_models(
        corrupted, model, normalization, torch.device("cpu"), batch_size=8
    )

    assert result["window_count"] == stats["evaluated_window_count"]
    assert result["window_key_sha256"] == result["constant_velocity"]["window_key_sha256"]
    assert result["window_key_sha256"] == result["lstm"]["window_key_sha256"]
    assert result["window_key_sha256"] == result["constant_velocity"]["window_key_sha256"]


def test_mask_rejects_invalid_rate_and_non_validation_windows():
    with pytest.raises(ValueError, match="rate"):
        generate_dropout_mask((_window(),), rate=1.1, seed=42)
    train_window = _window()
    train_window = ETHWindow(
        train_window.scene,
        "train",
        train_window.pedestrian_id,
        train_window.start_frame,
        train_window.observed,
        train_window.future,
    )
    with pytest.raises(ValueError, match="validation"):
        generate_dropout_mask((train_window,), rate=0.1, seed=42)
