from pathlib import Path

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
from trackvision.prediction.lstm import LSTMConfig, LSTMTrajectoryPredictor
from trackvision.prediction.track_fragmentation import (
    FRAGMENTATION_SEVERITIES,
    _matched_degradation,
    _unusable_window_records,
    construct_fragment_windows,
    fragment_validation_tracks,
)


def _dataset(track_lengths: tuple[int, ...] = (48,)) -> ETHSplit:
    observations = tuple(
        ETHObservation(
            frame=index * 10,
            pedestrian_id=pedestrian_id,
            x=float(index),
            y=float(index * index) / 10,
        )
        for pedestrian_id, length in enumerate(track_lengths, start=1)
        for index in range(length)
    )
    return ETHSplit("HOTEL", "val", Path("val.txt"), observations)


def _candidate_windows(dataset: ETHSplit) -> tuple[ETHWindow, ...]:
    return build_eth_windows(dataset, frame_step=10, history_len=8, horizon_len=12)


@pytest.mark.parametrize(
    ("severity", "sampled_steps", "frame_gap"),
    (("low", 1, 20), ("medium", 2, 30), ("high", 3, 40)),
)
def test_fragmentation_gap_length_and_subsequent_id(
    severity: str, sampled_steps: int, frame_gap: int
):
    del severity
    dataset = _dataset()
    candidates = _candidate_windows(dataset)
    fragments, audit = fragment_validation_tracks(
        (dataset,), candidates, sampled_steps, 42, frame_step=10
    )

    assert audit["fragmented_pedestrian_track_count"] == 1
    event = audit["selected_breaks"][0]
    assert event["missing_sampled_observation_count"] == sampled_steps
    assert event["boundary_frame_id_gap_units"] == frame_gap
    assert event["synthetic_track_id_before"] != event["synthetic_track_id_after"]
    assert len(event["removed_frame_ids"]) == sampled_steps
    assert len(event["fragment_lengths_sampled_observations"]) == 2
    fragment_ids = {
        fragment.synthetic_track_id
        for fragment in fragments
        if fragment.original_pedestrian_id == event["original_pedestrian_id"]
    }
    fragmented_frames = {
        point.frame
        for fragment in fragments
        if fragment.original_pedestrian_id == event["original_pedestrian_id"]
        for point in fragment.observations
    }
    assert event["synthetic_track_id_before"] in fragment_ids
    assert event["synthetic_track_id_after"] in fragment_ids
    assert not fragmented_frames.intersection(event["removed_frame_ids"])


def test_fragmentation_is_reproducible_and_seed_changes_break_location():
    dataset = _dataset()
    candidates = _candidate_windows(dataset)
    first, first_audit = fragment_validation_tracks(
        (dataset,), candidates, 2, 42, frame_step=10
    )
    repeat, repeat_audit = fragment_validation_tracks(
        (dataset,), candidates, 2, 42, frame_step=10
    )
    changed, changed_audit = fragment_validation_tracks(
        (dataset,), candidates, 2, 43, frame_step=10
    )

    assert first == repeat
    assert first_audit["selected_breaks"] == repeat_audit["selected_breaks"]
    assert first_audit["selected_breaks"][0]["removed_frame_ids"] != (
        changed_audit["selected_breaks"][0]["removed_frame_ids"]
    )
    assert first != changed


def test_each_seed_uses_same_break_location_across_severities():
    dataset = _dataset()
    candidates = _candidate_windows(dataset)
    removed_frames_by_severity = {}
    for missing_steps in FRAGMENTATION_SEVERITIES.values():
        _, audit = fragment_validation_tracks(
            (dataset,), candidates, missing_steps, 42, frame_step=10
        )
        removed_frames_by_severity[missing_steps] = {
            (
                event["scene"],
                event["original_pedestrian_id"],
            ): event["first_removed_frame_id"]
            for event in audit["selected_breaks"]
        }

    assert (
        removed_frames_by_severity[1]
        == removed_frames_by_severity[2]
        == removed_frames_by_severity[3]
    )


def test_no_fragment_stitching_or_interpolation_and_original_data_unchanged():
    dataset = _dataset()
    candidates = _candidate_windows(dataset)
    original_rows = dataset.observations
    fragments, audit = fragment_validation_tracks(
        (dataset,), candidates, 3, 42, frame_step=10
    )
    event = audit["selected_breaks"][0]
    removed = set(event["removed_frame_ids"])
    frames_by_id = {
        fragment.synthetic_track_id: {
            point.frame
            for point in fragment.observations
            if fragment.original_pedestrian_id == event["original_pedestrian_id"]
        }
        for fragment in fragments
        if fragment.original_pedestrian_id == event["original_pedestrian_id"]
    }

    assert all(not removed.intersection(frames) for frames in frames_by_id.values())
    assert len(frames_by_id) == 2
    assert dataset.observations == original_rows


def test_zero_fragmentation_reproduces_clean_window_set_exactly():
    dataset = _dataset((48, 23))
    candidates = _candidate_windows(dataset)
    fragments, audit = fragment_validation_tracks(
        (dataset,), candidates, 0, 42, frame_step=10
    )
    retained, fragment_ids = construct_fragment_windows(
        fragments, (dataset,), frame_step=10
    )

    assert audit["fragmented_pedestrian_track_count"] == 0
    assert {fragment.synthetic_track_id for fragment in fragments} == {1, 2}
    assert len(retained) == len(candidates)
    assert _window_key_digest(retained) == _window_key_digest(candidates)
    assert all(retained_window == clean for retained_window, clean in zip(
        retained, candidates, strict=True
    ))
    assert set(fragment_ids) == {
        (window.scene, window.split, window.pedestrian_id, window.start_frame)
        for window in candidates
    }


@pytest.mark.parametrize("missing_steps", FRAGMENTATION_SEVERITIES.values())
def test_survivors_are_8_by_12_single_fragment_windows_and_keep_clean_future(
    missing_steps: int,
):
    dataset = _dataset()
    candidates = _candidate_windows(dataset)
    fragments, _ = fragment_validation_tracks(
        (dataset,), candidates, missing_steps, 42, frame_step=10
    )
    retained, fragment_ids = construct_fragment_windows(
        fragments, (dataset,), frame_step=10
    )
    fragment_id_by_observation = {
        (
            fragment.scene,
            fragment.original_pedestrian_id,
            point.frame,
        ): fragment.synthetic_track_id
        for fragment in fragments
        for point in fragment.observations
    }
    clean_by_key = {
        (window.scene, window.split, window.pedestrian_id, window.start_frame): window
        for window in candidates
    }

    assert retained
    for window in retained:
        key = (window.scene, window.split, window.pedestrian_id, window.start_frame)
        assert len(window.observed) == 8
        assert len(window.future) == 12
        assert len({point.pedestrian_id for point in window.observed}) == 1
        assert fragment_ids[key] is not None
        observed_fragment_ids = {
            fragment_id_by_observation[
                (window.scene, window.pedestrian_id, point.frame)
            ]
            for point in window.observed
        }
        assert observed_fragment_ids == {fragment_ids[key]}
        assert window.future == clean_by_key[key].future
        assert window.observed == clean_by_key[key].observed


def test_cv_lstm_and_matched_clean_use_identical_windows():
    dataset = _dataset()
    candidates = _candidate_windows(dataset)
    fragments, _ = fragment_validation_tracks(
        (dataset,), candidates, 2, 42, frame_step=10
    )
    retained, _ = construct_fragment_windows(
        fragments, (dataset,), frame_step=10
    )
    clean_by_key = {
        (window.scene, window.split, window.pedestrian_id, window.start_frame): window
        for window in candidates
    }
    clean_matched = tuple(
        clean_by_key[(w.scene, w.split, w.pedestrian_id, w.start_frame)]
        for w in retained
    )
    model = LSTMTrajectoryPredictor(LSTMConfig(hidden_size=4))
    normalization = PositionNormalization((0.0, 0.0), (1.0, 1.0), "train", 1)

    fragmented_metrics = evaluate_models(
        retained, model, normalization, torch.device("cpu"), batch_size=16
    )
    clean_metrics = evaluate_models(
        clean_matched, model, normalization, torch.device("cpu"), batch_size=16
    )

    signature = _window_key_digest(retained)
    assert signature == _window_key_digest(clean_matched)
    assert fragmented_metrics["constant_velocity"]["window_key_sha256"] == signature
    assert fragmented_metrics["lstm"]["window_key_sha256"] == signature
    assert fragmented_metrics["constant_velocity"]["ade"] == pytest.approx(
        clean_metrics["constant_velocity"]["ade"]
    )
    assert fragmented_metrics["lstm"]["fde"] == pytest.approx(
        clean_metrics["lstm"]["fde"]
    )
    degradation = _matched_degradation(fragmented_metrics, clean_metrics)
    for model_name in ("constant_velocity", "lstm"):
        assert degradation[model_name]["ade"]["absolute"] == pytest.approx(0.0)
        assert degradation[model_name]["fde"]["absolute"] == pytest.approx(0.0)


def test_unusable_candidates_are_explicitly_accounted_for():
    dataset = _dataset()
    candidates = _candidate_windows(dataset)
    fragments, audit = fragment_validation_tracks(
        (dataset,), candidates, 3, 42, frame_step=10
    )
    retained, _ = construct_fragment_windows(
        fragments, (dataset,), frame_step=10
    )
    retained_keys = {
        (window.scene, window.split, window.pedestrian_id, window.start_frame)
        for window in retained
    }
    unusable = [
        window for window in candidates
        if (window.scene, window.split, window.pedestrian_id, window.start_frame)
        not in retained_keys
    ]
    unusable_records = _unusable_window_records(
        candidates, retained_keys, audit, frame_step=10
    )

    assert len(retained) + len(unusable) == len(candidates)
    assert unusable
    assert len(unusable_records) == len(unusable)
    assert all(
        record["reason"] == "fragmentation_gap_in_observed_history"
        for record in unusable_records
    )
    assert audit["track_count"] == 1
