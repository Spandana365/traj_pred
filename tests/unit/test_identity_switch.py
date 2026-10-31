from pathlib import Path

import numpy as np
import pytest
import torch

from trackvision.io.eth import (
    ETHObservation,
    ETHSplit,
    ETHWindow,
    build_eth_windows,
)
from trackvision.prediction.experiment import (
    PositionNormalization,
    _window_key_digest,
    evaluate_models,
)
from trackvision.prediction.identity_switch import (
    IDENTITY_SWITCH_SEVERITIES,
    apply_identity_switches,
    select_identity_switch_events,
)
from trackvision.prediction.lstm import LSTMConfig, LSTMTrajectoryPredictor


def _dataset(track_count: int = 2, length: int = 30) -> ETHSplit:
    observations = tuple(
        ETHObservation(
            frame=index * 10,
            pedestrian_id=pedestrian_id,
            x=float(pedestrian_id * 2 + index * 0.1),
            y=float(pedestrian_id - index * 0.03),
        )
        for pedestrian_id in range(1, track_count + 1)
        for index in range(length)
    )
    return ETHSplit("HOTEL", "val", Path("val.txt"), observations)


def _windows(dataset: ETHSplit) -> tuple[ETHWindow, ...]:
    return build_eth_windows(dataset, frame_step=10, history_len=8, horizon_len=12)


def _selection(dataset: ETHSplit, seed: int = 42) -> dict:
    return select_identity_switch_events(
        (dataset,), _windows(dataset), seed, frame_step=10
    )


def test_eligible_pair_has_distinct_ids_and_three_timestamp_overlap():
    dataset = _dataset()
    selection = _selection(dataset)

    scene = selection["scene_statistics"]["HOTEL"]
    assert scene["eligible_pair_count"] == 1
    assert scene["selected_pair_count"] == 1
    decision = scene["pair_eligibility_decisions"][0]
    assert decision["eligible"]
    assert decision["distinct_original_pedestrian_ids"]
    assert decision["longest_consecutive_shared_timestamp_run"] == 30
    event = scene["selected_event"]
    assert event["pedestrian_id_a"] != event["pedestrian_id_b"]
    assert len(event["high_severity_switch_frame_ids"]) == 3


@pytest.mark.parametrize(
    ("severity", "switch_steps", "frame_gap"),
    (("low", 1, 20), ("medium", 2, 30), ("high", 3, 40)),
)
def test_symmetric_switch_changes_only_aligned_observed_positions(
    severity: str, switch_steps: int, frame_gap: int
):
    del severity
    dataset = _dataset()
    clean = _windows(dataset)
    original_rows = dataset.observations
    selection = _selection(dataset)
    corrupted, statistics = apply_identity_switches(
        clean, (dataset,), selection, switch_steps
    )
    event = statistics["events"][0]
    first_id, second_id = event["pedestrian_id_a"], event["pedestrian_id_b"]
    switched_frames = set(event["switched_frame_ids"])
    source_by_key = {
        (item.pedestrian_id, item.frame): item for item in dataset.observations
    }

    assert len(switched_frames) == switch_steps
    assert event["switched_assignment_count"] == 2 * switch_steps
    assert event["switched_frame_ids"] == event["high_severity_switch_frame_ids"][:switch_steps]
    assert event["hypothetical_boundary_frame_id_separation_units"] == frame_gap
    assert event["switch_interval_frame_id_span_units"] == (switch_steps - 1) * 10
    assert event["pedestrian_id_a"] != event["pedestrian_id_b"]
    for clean_window, corrupted_window in zip(clean, corrupted, strict=True):
        assert clean_window.future == corrupted_window.future
        assert clean_window.pedestrian_id == corrupted_window.pedestrian_id
        for original, changed in zip(clean_window.observed, corrupted_window.observed, strict=True):
            assert original.frame == changed.frame
            assert original.pedestrian_id == changed.pedestrian_id
            if original.frame in switched_frames and original.pedestrian_id in {
                first_id, second_id
            }:
                other_id = second_id if original.pedestrian_id == first_id else first_id
                source = source_by_key[(other_id, original.frame)]
                assert (changed.x, changed.y) == (source.x, source.y)
            else:
                assert (changed.x, changed.y) == (original.x, original.y)
    assert statistics["switched_observation_assignment_count"] == 2 * switch_steps
    assert statistics["unique_affected_pedestrian_count"] == 2
    assert statistics["event_count"] == 1
    assert statistics["events"][0]["mean_spatial_separation"] == pytest.approx(
        float(np.mean([
            np.hypot(
                source_by_key[(first_id, frame)].x - source_by_key[(second_id, frame)].x,
                source_by_key[(first_id, frame)].y - source_by_key[(second_id, frame)].y,
            )
            for frame in event["switched_frame_ids"]
        ]))
    )
    assert dataset.observations == original_rows


def test_seed_reproducibility_and_seed_dependent_pair_selection():
    dataset = _dataset(track_count=6)
    first = _selection(dataset, seed=42)
    repeat = _selection(dataset, seed=42)
    other = _selection(dataset, seed=43)

    assert first == repeat
    first_pair = first["selected_events"][0]
    other_pair = other["selected_events"][0]
    assert (
        first_pair["pedestrian_id_a"], first_pair["pedestrian_id_b"],
        first_pair["start_frame_id"],
    ) != (
        other_pair["pedestrian_id_a"], other_pair["pedestrian_id_b"],
        other_pair["start_frame_id"],
    )


def test_same_seed_reuses_pairs_and_start_across_all_severities():
    dataset = _dataset(track_count=5)
    selection = _selection(dataset)
    results = {
        severity: apply_identity_switches(
            _windows(dataset), (dataset,), selection, steps
        )[1]
        for severity, steps in IDENTITY_SWITCH_SEVERITIES.items()
    }

    identities = {
        severity: [
            (
                event["pedestrian_id_a"],
                event["pedestrian_id_b"],
                event["start_frame_id"],
            )
            for event in result["events"]
        ]
        for severity, result in results.items()
    }
    assert identities["low"] == identities["medium"] == identities["high"]
    assert [
        len(result["events"][0]["switched_frame_ids"])
        for result in results.values()
    ] == [1, 2, 3]


def test_overlapping_windows_have_consistent_source_identity_assignments():
    dataset = _dataset(length=34)
    windows = _windows(dataset)
    selection = _selection(dataset)
    corrupted, statistics = apply_identity_switches(
        windows, (dataset,), selection, 3
    )
    maps = []
    for window in corrupted:
        maps.append({
            (window.scene, window.pedestrian_id, point.frame): (point.x, point.y)
            for point in window.observed
        })
    for first in range(len(maps)):
        for second in range(first + 1, len(maps)):
            shared = maps[first].keys() & maps[second].keys()
            assert all(maps[first][key] == maps[second][key] for key in shared)
    switched = [
        item for item in statistics["source_identity_assignments_by_unique_observation"]
        if item["switched"]
    ]
    assert len(switched) == 6
    assert all(item["source_original_pedestrian_id"] != item["target_original_pedestrian_id"] for item in switched)


def test_candidate_pair_records_ineligible_pairs_and_unpaired_tracks():
    dataset = _dataset(track_count=3)
    observations = tuple(
        item for item in dataset.observations
        if not (item.pedestrian_id == 3 and item.frame >= 100)
    )
    incomplete = ETHSplit("HOTEL", "val", dataset.source_file, observations)
    selection = _selection(incomplete)
    scene = selection["scene_statistics"]["HOTEL"]

    assert scene["possible_distinct_pairs"] == 3
    assert scene["eligible_pair_count"] <= 3
    assert scene["ineligible_pair_count"] == 3 - scene["eligible_pair_count"]
    assert len(scene["pair_eligibility_decisions"]) == 3
    assert len(scene["track_eligibility_decisions"]) == 3
    assert all(
        not track["eligible"] and track["ineligibility_reason"]
        for track in scene["track_eligibility_decisions"]
        if track["original_pedestrian_id"] == 3
    )


def test_zero_switch_preserves_windows_and_evaluation_exactly():
    dataset = _dataset()
    windows = _windows(dataset)
    corrupted, statistics = apply_identity_switches(
        windows, (dataset,), _selection(dataset), 0
    )
    model = LSTMTrajectoryPredictor(LSTMConfig(hidden_size=4))
    normalization = PositionNormalization((0.0, 0.0), (1.0, 1.0), "train", 1)
    clean_metrics = evaluate_models(
        windows, model, normalization, torch.device("cpu"), batch_size=16
    )
    result = evaluate_models(
        corrupted, model, normalization, torch.device("cpu"), batch_size=16
    )

    assert corrupted == windows
    assert statistics["switched_observation_assignment_count"] == 0
    assert statistics["affected_window_count"] == 0
    assert _window_key_digest(corrupted) == _window_key_digest(windows)
    assert result["constant_velocity"] == clean_metrics["constant_velocity"]
    assert result["lstm"] == clean_metrics["lstm"]


def test_cv_lstm_and_clean_corrupted_share_same_window_signature():
    dataset = _dataset()
    windows = _windows(dataset)
    corrupted, _ = apply_identity_switches(
        windows, (dataset,), _selection(dataset), 2
    )
    model = LSTMTrajectoryPredictor(LSTMConfig(hidden_size=4))
    normalization = PositionNormalization((0.0, 0.0), (1.0, 1.0), "train", 1)
    clean_metrics = evaluate_models(
        windows, model, normalization, torch.device("cpu"), batch_size=16
    )
    corrupted_metrics = evaluate_models(
        corrupted, model, normalization, torch.device("cpu"), batch_size=16
    )

    signature = _window_key_digest(windows)
    assert _window_key_digest(corrupted) == signature
    assert clean_metrics["constant_velocity"]["window_key_sha256"] == signature
    assert clean_metrics["lstm"]["window_key_sha256"] == signature
    assert corrupted_metrics["constant_velocity"]["window_key_sha256"] == signature
    assert corrupted_metrics["lstm"]["window_key_sha256"] == signature
