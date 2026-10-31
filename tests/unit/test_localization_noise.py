from pathlib import Path

import numpy as np
import pytest

from trackvision.io.eth import ETHObservation, ETHSplit, ETHWindow
from trackvision.prediction.localization_noise import (
    _write_visualization,
    estimate_median_consecutive_displacement,
    inject_localization_noise,
    localization_noise_severities,
)


def _window(
    pedestrian_id: int,
    start_index: int = 0,
    split: str = "val",
) -> ETHWindow:
    observations = tuple(
        ETHObservation(
            frame=index * 10,
            pedestrian_id=pedestrian_id,
            x=float(index),
            y=float(index * 2),
        )
        for index in range(start_index, start_index + 20)
    )
    return ETHWindow(
        scene="HOTEL",
        split=split,
        pedestrian_id=pedestrian_id,
        start_frame=observations[0].frame,
        observed=observations[:8],
        future=observations[8:],
    )


def test_noise_is_reproducible_isotropic_and_has_configured_sigma():
    windows = tuple(_window(pedestrian_id) for pedestrian_id in range(1000))

    corrupted_a, stats_a = inject_localization_noise(windows, sigma=0.7, seed=42)
    corrupted_b, stats_b = inject_localization_noise(windows, sigma=0.7, seed=42)
    delta_x = np.asarray([
        point.x - original.x
        for noisy_window, window in zip(corrupted_a, windows, strict=True)
        for point, original in zip(noisy_window.observed, window.observed, strict=True)
    ])
    delta_y = np.asarray([
        point.y - original.y
        for noisy_window, window in zip(corrupted_a, windows, strict=True)
        for point, original in zip(noisy_window.observed, window.observed, strict=True)
    ])

    assert stats_a == stats_b
    assert stats_a["unique_observation_count"] == 8000
    assert stats_a["std_dx"] == pytest.approx(0.7, rel=0.04)
    assert stats_a["std_dy"] == pytest.approx(0.7, rel=0.04)
    assert abs(stats_a["mean_dx"]) < 0.02
    assert abs(stats_a["mean_dy"]) < 0.02
    assert abs(float(np.corrcoef(delta_x, delta_y)[0, 1])) < 0.03
    assert stats_a["mean_radial_displacement"] == pytest.approx(
        0.7 * np.sqrt(np.pi / 2), rel=0.04
    )
    np.testing.assert_array_equal(delta_x, np.asarray([
        point.x - original.x
        for noisy_window, window in zip(corrupted_b, windows, strict=True)
        for point, original in zip(noisy_window.observed, window.observed, strict=True)
    ]))
    np.testing.assert_array_equal(delta_y, np.asarray([
        point.y - original.y
        for noisy_window, window in zip(corrupted_b, windows, strict=True)
        for point, original in zip(noisy_window.observed, window.observed, strict=True)
    ]))


def test_overlapping_windows_share_noise_and_preserve_identifiers_and_future():
    first = _window(7, start_index=0)
    second = _window(7, start_index=1)
    corrupted, _ = inject_localization_noise((first, second), sigma=0.4, seed=43)
    first_noisy, second_noisy = corrupted

    shared_first = {
        point.frame: point for point in first_noisy.observed
    }
    shared_second = {
        point.frame: point for point in second_noisy.observed
    }
    shared_frames = shared_first.keys() & shared_second.keys()
    assert shared_frames
    for frame in shared_frames:
        assert shared_first[frame].x == shared_second[frame].x
        assert shared_first[frame].y == shared_second[frame].y
    assert (first_noisy.scene, first_noisy.pedestrian_id, first_noisy.start_frame) == (
        first.scene, first.pedestrian_id, first.start_frame
    )
    assert (second_noisy.scene, second_noisy.pedestrian_id, second_noisy.start_frame) == (
        second.scene, second.pedestrian_id, second.start_frame
    )
    assert first_noisy.future == first.future
    assert second_noisy.future == second.future
    assert first_noisy.observed != first.observed


def test_zero_sigma_preserves_inputs_and_different_seeds_change_noise():
    windows = (_window(1), _window(2))
    original_observed = tuple(window.observed for window in windows)
    original_future = tuple(window.future for window in windows)

    unchanged, zero_stats = inject_localization_noise(windows, sigma=0, seed=42)
    first, _ = inject_localization_noise(windows, sigma=0.3, seed=42)
    second, _ = inject_localization_noise(windows, sigma=0.3, seed=44)

    assert tuple(window.observed for window in unchanged) == original_observed
    assert tuple(window.future for window in unchanged) == original_future
    assert zero_stats["std_dx"] == zero_stats["std_dy"] == 0.0
    assert first[0].observed != second[0].observed
    assert tuple(window.observed for window in windows) == original_observed
    assert tuple(window.future for window in windows) == original_future


def test_noise_rejects_non_validation_and_invalid_sigma():
    with pytest.raises(ValueError, match="validation"):
        inject_localization_noise((_window(1, split="train"),), sigma=0.1, seed=42)
    with pytest.raises(ValueError, match="sigma"):
        inject_localization_noise((_window(1),), sigma=-0.1, seed=42)


def test_training_only_median_displacement_uses_exact_frame_step_pairs():
    train = ETHSplit(
        scene="HOTEL",
        split="train",
        source_file=Path("train.txt"),
        observations=tuple(
            ETHObservation(frame, 1, frame / 10, 0.0)
            for frame in (0, 10, 20, 40, 50)
        ),
    )

    median, pair_count = estimate_median_consecutive_displacement((train,), frame_step=10)

    assert median == pytest.approx(1.0)
    assert pair_count == 3
    with pytest.raises(ValueError, match="training"):
        estimate_median_consecutive_displacement(
            (ETHSplit("HOTEL", "val", Path("val.txt"), train.observations),),
            frame_step=10,
        )


def test_exploratory_severity_sigmas_are_relative_to_training_displacement():
    severities = localization_noise_severities(2.0)

    assert severities["low"]["sigma"] == pytest.approx(0.2)
    assert severities["medium"]["sigma"] == pytest.approx(0.5)
    assert severities["high"]["sigma"] == pytest.approx(1.0)
    assert all("exploratory" in str(item["interpretation"]) for item in severities.values())


def test_visualization_writes_clean_noisy_and_future_trajectory_comparison(tmp_path):
    output = tmp_path / "noise.png"
    severities = localization_noise_severities(1.0)

    _write_visualization(output, _window(1), severities, seed=42)

    assert output.is_file()
    assert output.stat().st_size > 0
