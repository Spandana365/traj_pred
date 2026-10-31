import math

import pytest

from trackvision.prediction.lstm_error_comparison import (
    _extract_seed_metrics,
    _summary,
    generate_comparison_plots,
)


def test_summary_uses_population_standard_deviation_for_three_seeds():
    mean, std = _summary([1.0, 2.0, 3.0])

    assert mean == pytest.approx(2.0)
    assert std == pytest.approx(math.sqrt(2.0 / 3.0))


def test_fragmentation_metrics_use_matched_clean_surviving_windows():
    signature = "same-window-signature"
    report = {
        "seed": 42,
        "split": "val",
        "severity_results": {
            "low": {
                "candidate_window_count": 100,
                "retained_window_count": 80,
                "unusable_window_count": 20,
                "window_retention_percentage": 80.0,
                "clean_baseline_on_retained_windows": {
                    "lstm": {
                        "ade": 0.4,
                        "fde": 0.8,
                        "window_count": 80,
                        "window_key_sha256": signature,
                    }
                },
                "fragmented_input_metrics": {
                    "lstm": {
                        "ade": 0.4,
                        "fde": 0.8,
                        "window_count": 80,
                        "window_key_sha256": signature,
                    }
                },
            }
        },
    }

    result = _extract_seed_metrics("track_fragmentation", report, "low", 42)

    assert result.candidate_windows == 100
    assert result.evaluated_windows == 80
    assert result.matched_clean_ade == result.corrupted_ade == 0.4
    assert result.matched_clean_fde == result.corrupted_fde == 0.8


def test_fragmentation_rejects_mismatched_clean_and_corrupted_windows():
    report = {
        "seed": 42,
        "split": "val",
        "severity_results": {
            "low": {
                "candidate_window_count": 100,
                "retained_window_count": 80,
                "unusable_window_count": 20,
                "clean_baseline_on_retained_windows": {
                    "lstm": {
                        "ade": 0.4,
                        "fde": 0.8,
                        "window_count": 80,
                        "window_key_sha256": "clean",
                    }
                },
                "fragmented_input_metrics": {
                    "lstm": {
                        "ade": 0.5,
                        "fde": 0.9,
                        "window_count": 80,
                        "window_key_sha256": "corrupted",
                    }
                },
            }
        },
    }

    with pytest.raises(ValueError, match="different window sets"):
        _extract_seed_metrics("track_fragmentation", report, "low", 42)


def test_dropout_uses_matched_clean_metrics_and_actual_evaluated_count():
    signature = "same-window-signature"
    report = {
        "seed": 43,
        "split": "val",
        "candidate_window_count": 100,
        "severity_results": {
            "high": {
                "candidate_window_count": 100,
                "evaluated_window_count": 98,
                "unusable_window_count": 2,
                "window_key_sha256": signature,
                "clean_baseline_on_retained_windows": {
                    "lstm": {
                        "ade": 0.4,
                        "fde": 0.8,
                        "window_count": 98,
                        "window_key_sha256": signature,
                    }
                },
                "metrics": {
                    "lstm": {
                        "ade": 0.6,
                        "fde": 1.0,
                        "window_count": 98,
                        "window_key_sha256": signature,
                    }
                },
            }
        },
    }

    result = _extract_seed_metrics("detection_dropout", report, "high", 43)

    assert result.candidate_windows == 100
    assert result.evaluated_windows == 98
    assert result.matched_clean_ade == 0.4
    assert result.corrupted_ade == 0.6


def test_plotting_refuses_to_overwrite_a_nonempty_output_directory(tmp_path):
    destination = tmp_path / "plots"
    destination.mkdir()
    existing = destination / "keep.png"
    existing.write_bytes(b"existing")

    with pytest.raises(FileExistsError, match="non-empty output directory"):
        generate_comparison_plots(tmp_path, destination)

    assert existing.read_bytes() == b"existing"
