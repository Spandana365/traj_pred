"""Validation-only temporary identity-switch experiments for frozen predictors."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import random
from dataclasses import replace
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from trackvision.io.eth import ETHObservation, ETHSplit, ETHWindow
from trackvision.prediction.detection_dropout import (
    _aggregate_clean_model,
    _aggregate_model,
    _mean_std,
)
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
from trackvision.prediction.localization_noise import _add_degradation, _atomic_json
from trackvision.prediction.lstm import LSTMTrajectoryPredictor


IDENTITY_SWITCH_SEVERITIES = {"low": 1, "medium": 2, "high": 3}
CORRUPTION_SEEDS = (42, 43, 44)
DEFAULT_EXPERIMENT_DIR = Path("runs/prediction/eth/lstm-seed42-h64-l2-scheduler")
DEFAULT_OUTPUT_DIR = Path("runs/error_injection/identity_switch")
ObservationKey = tuple[str, int, int]
WindowKey = tuple[str, str, int, int]
PairKey = tuple[str, int, int]


def _window_key(window: ETHWindow) -> WindowKey:
    return window.scene, window.split, window.pedestrian_id, window.start_frame


def _rng(seed: int, scene: str) -> random.Random:
    digest = hashlib.sha256(f"{seed}:{scene}".encode("utf-8")).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def _weighted_rank_choice(
    items: list[Any],
    distance: Any,
    generator: random.Random,
) -> Any:
    """Choose by inverse distance rank; nearest candidates are preferred, not thresholded."""
    ordered = sorted(items, key=lambda item: (float(distance(item)), repr(item)))
    weights = [1.0 / (rank + 1) for rank in range(len(ordered))]
    return generator.choices(ordered, weights=weights, k=1)[0]


def _mean_std_or_none(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "std": None}
    return _mean_std(values)


def select_identity_switch_events(
    validation_datasets: tuple[ETHSplit, ...],
    candidate_windows: tuple[ETHWindow, ...],
    seed: int,
    frame_step: int,
    observation_len: int = 8,
    max_switch_steps: int = 3,
) -> dict[str, Any]:
    """Select one non-overlapping pedestrian pair and high-severity start per scene."""
    if not validation_datasets or any(item.split != "val" for item in validation_datasets):
        raise ValueError("identity-switch selection accepts validation datasets only")
    if not candidate_windows or any(item.split != "val" for item in candidate_windows):
        raise ValueError("identity-switch selection requires validation windows")
    if frame_step <= 0 or observation_len <= 0:
        raise ValueError("frame_step and observation_len must be positive")
    if max_switch_steps != max(IDENTITY_SWITCH_SEVERITIES.values()):
        raise ValueError("max_switch_steps must equal the configured high severity")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")

    observations_by_track: dict[tuple[str, int], dict[int, ETHObservation]] = {}
    for dataset in validation_datasets:
        for observation in dataset.observations:
            observations_by_track.setdefault(
                (dataset.scene, observation.pedestrian_id), {}
            )[observation.frame] = observation

    windows_by_track: dict[tuple[str, int], list[frozenset[int]]] = {}
    for window in candidate_windows:
        windows_by_track.setdefault((window.scene, window.pedestrian_id), []).append(
            frozenset(point.frame for point in window.observed)
        )

    scenes: dict[str, Any] = {}
    selected_events: list[dict[str, Any]] = []
    for dataset in validation_datasets:
        scene = dataset.scene
        identities = sorted(
            pedestrian_id
            for scene_name, pedestrian_id in observations_by_track
            if scene_name == scene
        )
        pair_decisions: list[dict[str, Any]] = []
        eligible_pair_options: dict[PairKey, list[dict[str, Any]]] = {}
        for first_index, first_id in enumerate(identities):
            first_track = observations_by_track[(scene, first_id)]
            for second_id in identities[first_index + 1:]:
                second_track = observations_by_track[(scene, second_id)]
                shared_frames = sorted(first_track.keys() & second_track.keys())
                consecutive_runs: list[list[int]] = []
                run: list[int] = []
                for frame in shared_frames:
                    if run and frame - run[-1] != frame_step:
                        consecutive_runs.append(run)
                        run = []
                    run.append(frame)
                if run:
                    consecutive_runs.append(run)
                max_run = max((len(item) for item in consecutive_runs), default=0)
                key = (scene, first_id, second_id)
                feasible: list[dict[str, Any]] = []
                for shared_run in consecutive_runs:
                    for start_index in range(0, len(shared_run) - max_switch_steps + 1):
                        interval = shared_run[start_index:start_index + max_switch_steps]
                        first_affected = any(
                            all(frame in window for frame in interval)
                            for window in windows_by_track.get((scene, first_id), [])
                        )
                        second_affected = any(
                            all(frame in window for frame in interval)
                            for window in windows_by_track.get((scene, second_id), [])
                        )
                        if not (first_affected and second_affected):
                            continue
                        separations = [
                            math.hypot(
                                first_track[frame].x - second_track[frame].x,
                                first_track[frame].y - second_track[frame].y,
                            )
                            for frame in interval
                        ]
                        feasible.append({
                            "start_frame_id": interval[0],
                            "switch_frame_ids_high": interval,
                            "mean_separation": float(np.mean(separations)),
                            "max_separation": float(max(separations)),
                            "separation_by_frame": [
                                {"frame_id": frame, "distance": distance}
                                for frame, distance in zip(interval, separations, strict=True)
                            ],
                            "eligible_window_exists_for_first_pedestrian": first_affected,
                            "eligible_window_exists_for_second_pedestrian": second_affected,
                        })
                eligible = bool(feasible)
                if not shared_frames:
                    reason = "no_timestamp_aligned_observations"
                elif max_run < max_switch_steps:
                    reason = "fewer_than_three_consecutive_shared_timestamps"
                elif not feasible:
                    reason = (
                        "no_three_step_shared_interval_is_inside_observed_history_windows "
                        "for_both_pedestrians"
                    )
                else:
                    reason = None
                    eligible_pair_options[key] = feasible
                pair_decisions.append({
                    "scene": scene,
                    "pedestrian_id_a": first_id,
                    "pedestrian_id_b": second_id,
                    "distinct_original_pedestrian_ids": first_id != second_id,
                    "shared_timestamp_count": len(shared_frames),
                    "longest_consecutive_shared_timestamp_run": max_run,
                    "eligible_high_severity_start_count": len(feasible),
                    "eligible_start_spatial_separations": [
                        {
                            "start_frame_id": item["start_frame_id"],
                            "mean_separation": item["mean_separation"],
                            "max_separation": item["max_separation"],
                        }
                        for item in feasible
                    ],
                    "eligible": eligible,
                    "ineligibility_reason": reason,
                })

        if not identities:
            scene_reason = "scene_contains_no_pedestrian_tracks"
        elif not eligible_pair_options:
            scene_reason = "scene_has_no_pair_meeting_overlap_and_window_eligibility"
        else:
            scene_reason = None

        selected: dict[str, Any] | None = None
        if eligible_pair_options:
            generator = _rng(seed, scene)
            pair_options = [
                {
                    "pair": pair,
                    "starts": starts,
                    "best_mean_separation": min(
                        float(item["mean_separation"]) for item in starts
                    ),
                }
                for pair, starts in eligible_pair_options.items()
            ]
            chosen_pair = _weighted_rank_choice(
                pair_options,
                lambda item: item["best_mean_separation"],
                generator,
            )
            chosen_start = _weighted_rank_choice(
                chosen_pair["starts"],
                lambda item: item["mean_separation"],
                generator,
            )
            scene_name, first_id, second_id = chosen_pair["pair"]
            selected = {
                "scene": scene_name,
                "frame_step": frame_step,
                "pedestrian_id_a": first_id,
                "pedestrian_id_b": second_id,
                "start_frame_id": chosen_start["start_frame_id"],
                "high_severity_switch_frame_ids": chosen_start["switch_frame_ids_high"],
                "mean_spatial_separation_high_severity": chosen_start["mean_separation"],
                "max_spatial_separation_high_severity": chosen_start["max_separation"],
                "spatial_separation_by_frame_high_severity": chosen_start["separation_by_frame"],
                "eligible_high_severity_start_count_for_pair": len(chosen_pair["starts"]),
                "selected_by": (
                    "seeded inverse-distance-rank weighted choice over eligible pairs and "
                    "starts; lower mean spatial separation ranks receive greater weight"
                ),
            }
            selected_events.append(selected)

        eligible_tracks = {
            pedestrian_id
            for pair in eligible_pair_options
            for pedestrian_id in pair[1:]
        }
        track_pair_counts = {pedestrian_id: 0 for pedestrian_id in identities}
        for pair in eligible_pair_options:
            track_pair_counts[pair[1]] += 1
            track_pair_counts[pair[2]] += 1
        scenes[scene] = {
            "track_count": len(identities),
            "possible_distinct_pairs": len(identities) * (len(identities) - 1) // 2,
            "eligible_pair_count": len(eligible_pair_options),
            "ineligible_pair_count": (
                len(identities) * (len(identities) - 1) // 2
                - len(eligible_pair_options)
            ),
            "selected_pair_count": int(selected is not None),
            "selected_event_count": int(selected is not None),
            "eligible_pedestrian_track_count": len(eligible_tracks),
            "ineligible_pedestrian_track_count": len(identities) - len(eligible_tracks),
            "scene_ineligibility_reason": scene_reason,
            "track_eligibility_decisions": [
                {
                    "scene": scene,
                    "original_pedestrian_id": pedestrian_id,
                    "eligible_pair_count": track_pair_counts[pedestrian_id],
                    "eligible": track_pair_counts[pedestrian_id] > 0,
                    "ineligibility_reason": (
                        None if track_pair_counts[pedestrian_id] > 0
                        else "no_eligible_timestamp_aligned_partner_with_window_overlap"
                    ),
                }
                for pedestrian_id in identities
            ],
            "pair_eligibility_decisions": pair_decisions,
            "selected_event": selected,
        }

    return {
        "seed": seed,
        "frame_step": frame_step,
        "max_switch_steps_used_for_pair_eligibility": max_switch_steps,
        "one_event_per_eligible_scene": True,
        "selected_pair_count": len(selected_events),
        "selected_event_count": len(selected_events),
        "selected_events": selected_events,
        "scene_statistics": scenes,
        "pair_reuse_check": "no pedestrian is reused; at most one pair is selected per scene",
    }


def apply_identity_switches(
    windows: tuple[ETHWindow, ...],
    validation_datasets: tuple[ETHSplit, ...],
    selection: dict[str, Any],
    switch_steps: int,
) -> tuple[tuple[ETHWindow, ...], dict[str, Any]]:
    """Exchange positions at aligned timestamps while retaining target identity/future."""
    if not windows or any(window.split != "val" for window in windows):
        raise ValueError("identity switching accepts non-empty validation windows only")
    if not validation_datasets or any(item.split != "val" for item in validation_datasets):
        raise ValueError("identity switching accepts validation datasets only")
    if switch_steps not in (0, *IDENTITY_SWITCH_SEVERITIES.values()):
        raise ValueError("switch_steps must be 0, 1, 2, or 3")

    source_observations: dict[ObservationKey, ETHObservation] = {}
    for dataset in validation_datasets:
        for observation in dataset.observations:
            source_observations[(dataset.scene, observation.pedestrian_id, observation.frame)] = observation

    source_identity: dict[ObservationKey, int] = {}
    event_records: list[dict[str, Any]] = []
    switched_keys: set[ObservationKey] = set()
    for selected in selection["selected_events"]:
        frames = selected["high_severity_switch_frame_ids"][:switch_steps]
        first_id = selected["pedestrian_id_a"]
        second_id = selected["pedestrian_id_b"]
        scene = selected["scene"]
        separation_by_frame = {
            int(item["frame_id"]): float(item["distance"])
            for item in selected["spatial_separation_by_frame_high_severity"]
        }
        for frame in frames:
            first_key = (scene, first_id, int(frame))
            second_key = (scene, second_id, int(frame))
            if first_key not in source_observations or second_key not in source_observations:
                raise RuntimeError("selected identity-switch timestamp is not aligned in source data")
            source_identity[first_key] = second_id
            source_identity[second_key] = first_id
            switched_keys.update((first_key, second_key))
        event_records.append({
            **selected,
            "switch_steps": switch_steps,
            "switched_frame_ids": [int(frame) for frame in frames],
            "switch_interval_frame_id_span_units": (
                max(0, switch_steps - 1) * int(selection["frame_step"])
            ),
            "hypothetical_boundary_frame_id_separation_units": (
                switch_steps + 1
            ) * int(selection["frame_step"]),
            "switched_assignment_count": 2 * len(frames),
            "mean_spatial_separation": (
                float(np.mean([separation_by_frame[int(frame)] for frame in frames]))
                if frames else 0.0
            ),
            "max_spatial_separation": (
                float(max(separation_by_frame[int(frame)] for frame in frames))
                if frames else 0.0
            ),
        })

    corrupted: list[ETHWindow] = []
    affected_windows: list[WindowKey] = []
    affected_by_scene: dict[str, int] = {}
    for window in windows:
        updated_observations: list[ETHObservation] = []
        window_affected = False
        for observation in window.observed:
            key = (window.scene, window.pedestrian_id, observation.frame)
            source_id = source_identity.get(key, observation.pedestrian_id)
            source = source_observations.get((window.scene, source_id, observation.frame))
            if source is None:
                raise RuntimeError(
                    f"identity-switch source observation is missing for {key} from {source_id}"
                )
            if source_id != observation.pedestrian_id:
                if source.frame != observation.frame:
                    raise RuntimeError("identity switch attempted to use a non-aligned timestamp")
                if source.pedestrian_id == observation.pedestrian_id:
                    raise RuntimeError("identity switch source provenance is not distinct")
                window_affected = True
            updated_observations.append(
                replace(observation, x=source.x, y=source.y)
            )
        if window_affected:
            affected_windows.append(_window_key(window))
            affected_by_scene[window.scene] = affected_by_scene.get(window.scene, 0) + 1
        corrupted.append(replace(window, observed=tuple(updated_observations)))

    all_observation_keys = {
        (window.scene, window.pedestrian_id, point.frame)
        for window in windows
        for point in window.observed
    }
    all_assignments = [
        {
            "scene": scene,
            "target_original_pedestrian_id": pedestrian_id,
            "frame_id": frame,
            "source_original_pedestrian_id": source_identity.get(
                (scene, pedestrian_id, frame), pedestrian_id
            ),
            "switched": (scene, pedestrian_id, frame) in switched_keys,
        }
        for scene, pedestrian_id, frame in sorted(all_observation_keys)
    ]
    return tuple(corrupted), {
        "switch_steps": switch_steps,
        "selected_pair_count": len(event_records),
        "event_count": len(event_records),
        "switched_observation_assignment_count": len(switched_keys),
        "unique_affected_pedestrian_count": len({
            (event["scene"], pedestrian_id)
            for event in event_records
            for pedestrian_id in (
                event["pedestrian_id_a"], event["pedestrian_id_b"]
            )
        }),
        "unique_affected_observation_count": len(switched_keys),
        "unique_observed_observation_count": len(all_observation_keys),
        "affected_window_count": len(affected_windows),
        "unaffected_window_count": len(windows) - len(affected_windows),
        "affected_window_percentage": 100.0 * len(affected_windows) / len(windows),
        "affected_window_keys": [
            {
                "scene": scene,
                "split": split,
                "pedestrian_id": pedestrian_id,
                "start_frame": start_frame,
            }
            for scene, split, pedestrian_id, start_frame in affected_windows
        ],
        "affected_window_count_by_scene": affected_by_scene,
        "event_count_by_scene": {
            scene: sum(event["scene"] == scene for event in event_records)
            for scene in sorted({window.scene for window in windows})
        },
        "events": event_records,
        "source_identity_assignments_by_unique_observation": all_assignments,
    }


def _write_summary_csv(path: Path, aggregates: dict[str, Any]) -> None:
    fields = (
        "severity", "model", "candidate_windows", "affected_windows_mean",
        "affected_window_percentage_mean", "switch_events_mean",
        "switched_assignments_mean", "clean_ade_mean", "clean_fde_mean",
        "corrupted_ade_mean", "corrupted_ade_std", "corrupted_fde_mean",
        "corrupted_fde_std", "ade_degradation_mean", "ade_degradation_std",
        "ade_degradation_percentage_mean", "fde_degradation_mean",
        "fde_degradation_std", "fde_degradation_percentage_mean",
    )
    with path.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=fields)
        writer.writeheader()
        for severity, result in aggregates.items():
            for model in ("constant_velocity", "lstm"):
                clean = result["clean_metrics"][model]
                corrupt = result["corrupted_metrics"][model]
                degradation = result["degradation_vs_clean"][model]
                writer.writerow({
                    "severity": severity,
                    "model": model,
                    "candidate_windows": result["candidate_window_count"],
                    "affected_windows_mean": result["affected_window_count"]["mean"],
                    "affected_window_percentage_mean": result["affected_window_percentage"]["mean"],
                    "switch_events_mean": result["event_count"]["mean"],
                    "switched_assignments_mean": result["switched_observation_assignment_count"]["mean"],
                    "clean_ade_mean": clean["ade"]["mean"],
                    "clean_fde_mean": clean["fde"]["mean"],
                    "corrupted_ade_mean": corrupt["ade"]["mean"],
                    "corrupted_ade_std": corrupt["ade"]["std"],
                    "corrupted_fde_mean": corrupt["fde"]["mean"],
                    "corrupted_fde_std": corrupt["fde"]["std"],
                    "ade_degradation_mean": degradation["ade"]["absolute"]["mean"],
                    "ade_degradation_std": degradation["ade"]["absolute"]["std"],
                    "ade_degradation_percentage_mean": degradation["ade"]["percentage"]["mean"],
                    "fde_degradation_mean": degradation["fde"]["absolute"]["mean"],
                    "fde_degradation_std": degradation["fde"]["absolute"]["std"],
                    "fde_degradation_percentage_mean": degradation["fde"]["percentage"]["mean"],
                })


def _plot_switch_example(
    path: Path,
    dataset: ETHSplit,
    event: dict[str, Any],
    switch_steps: int,
) -> None:
    first_id = event["pedestrian_id_a"]
    second_id = event["pedestrian_id_b"]
    first = {
        item.frame: item for item in dataset.observations
        if item.pedestrian_id == first_id
    }
    second = {
        item.frame: item for item in dataset.observations
        if item.pedestrian_id == second_id
    }
    frames = event["high_severity_switch_frame_ids"][:switch_steps]
    frame_set = set(frames)
    start = frames[0]
    last = frames[-1]
    context_start = start - 5 * event["frame_step"]
    context_end = last + 13 * event["frame_step"]
    first_context = [item for frame, item in sorted(first.items()) if context_start <= frame <= context_end]
    second_context = [item for frame, item in sorted(second.items()) if context_start <= frame <= context_end]
    first_future = [item for item in first_context if item.frame > last]
    second_future = [item for item in second_context if item.frame > last]
    first_clean_observed = [item for item in first_context if item.frame <= last]
    second_clean_observed = [item for item in second_context if item.frame <= last]
    first_corrupted = [
        second[point.frame] if point.frame in frame_set else point
        for point in first_clean_observed
    ]
    second_corrupted = [
        first[point.frame] if point.frame in frame_set else point
        for point in second_clean_observed
    ]

    figure, axis = plt.subplots(figsize=(8, 6))
    axis.plot(
        [point.x for point in first_context], [point.y for point in first_context],
        color="tab:blue", linestyle=":", alpha=0.65,
        label=f"Original GT pedestrian A (ID {first_id})",
    )
    axis.plot(
        [point.x for point in second_context], [point.y for point in second_context],
        color="tab:orange", linestyle=":", alpha=0.65,
        label=f"Original GT pedestrian B (ID {second_id})",
    )
    axis.plot(
        [point.x for point in first_corrupted], [point.y for point in first_corrupted],
        "o-", color="tab:blue",
        label=f"Input assigned to A (source B during switch)",
    )
    axis.plot(
        [point.x for point in second_corrupted], [point.y for point in second_corrupted],
        "s-", color="tab:orange",
        label=f"Input assigned to B (source A during switch)",
    )
    axis.plot(
        [point.x for point in first_future], [point.y for point in first_future],
        "^-", color="tab:blue", alpha=0.8,
        label=f"Unchanged future GT A (ID {first_id})",
    )
    axis.plot(
        [point.x for point in second_future], [point.y for point in second_future],
        "v-", color="tab:orange", alpha=0.8,
        label=f"Unchanged future GT B (ID {second_id})",
    )
    first_swapped = [second[frame] for frame in frames]
    second_swapped = [first[frame] for frame in frames]
    axis.scatter(
        [point.x for point in first_swapped], [point.y for point in first_swapped],
        marker="x", s=110, color="tab:blue", label="Switched positions assigned to A",
    )
    axis.scatter(
        [point.x for point in second_swapped], [point.y for point in second_swapped],
        marker="+", s=120, color="tab:orange", label="Switched positions assigned to B",
    )
    axis.set_title(
        f"Temporary identity switch: {dataset.scene}, IDs {first_id} and {second_id}, "
        f"{switch_steps} sampled step(s)"
    )
    axis.set_xlabel("x (raw dataset coordinate units)")
    axis.set_ylabel("y (raw dataset coordinate units)")
    axis.set_aspect("equal", adjustable="datalim")
    axis.grid(True, alpha=0.3)
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _plot_severity_degradation(path: Path, aggregates: dict[str, Any]) -> None:
    severities = list(IDENTITY_SWITCH_SEVERITIES)
    x = np.asarray([IDENTITY_SWITCH_SEVERITIES[item] for item in severities])
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    for model, label, color in (
        ("constant_velocity", "Constant Velocity", "tab:blue"),
        ("lstm", "Two-layer LSTM", "tab:orange"),
    ):
        for axis, metric in zip(axes, ("ade", "fde"), strict=True):
            values = [
                aggregates[severity]["degradation_vs_clean"][model][metric]["absolute"]["mean"]
                for severity in severities
            ]
            errors = [
                aggregates[severity]["degradation_vs_clean"][model][metric]["absolute"]["std"]
                for severity in severities
            ]
            axis.errorbar(x, values, yerr=errors, marker="o", capsize=3, label=label, color=color)
            axis.set_title(f"{metric.upper()} absolute degradation")
            axis.set_xticks(x, ["Low (1)", "Medium (2)", "High (3)"])
            axis.set_xlabel("Consecutive switched sampled observations")
            axis.set_ylabel("Error increase (raw dataset coordinate units)")
            axis.grid(True, alpha=0.3)
    axes[0].legend()
    figure.suptitle("Identity-switch severity versus validation prediction degradation")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _plot_affected_windows(path: Path, aggregates: dict[str, Any]) -> None:
    severities = list(IDENTITY_SWITCH_SEVERITIES)
    x = np.asarray([IDENTITY_SWITCH_SEVERITIES[item] for item in severities])
    means = [aggregates[item]["affected_window_percentage"]["mean"] for item in severities]
    stds = [aggregates[item]["affected_window_percentage"]["std"] for item in severities]
    figure, axis = plt.subplots(figsize=(7, 5))
    axis.errorbar(x, means, yerr=stds, marker="o", capsize=3)
    axis.set_xticks(x, ["Low (1)", "Medium (2)", "High (3)"])
    axis.set_xlabel("Consecutive switched sampled observations")
    axis.set_ylabel("Affected validation windows (%)")
    axis.set_ylim(bottom=0)
    axis.grid(True, alpha=0.3)
    axis.set_title("Identity-switch severity versus affected windows")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _plot_scene_degradation(path: Path, aggregates: dict[str, Any]) -> None:
    scenes = sorted(
        next(iter(aggregates.values()))["corrupted_metrics"]["lstm"]["per_scene"]
    )
    x = np.arange(len(scenes))
    figure, axis = plt.subplots(figsize=(12, 5))
    width = 0.24
    for index, severity in enumerate(IDENTITY_SWITCH_SEVERITIES):
        values = [
            aggregates[severity]["corrupted_metrics"]["lstm"][
                "per_scene_degradation_vs_clean"
            ][scene]["ade"]["absolute"]["mean"]
            for scene in scenes
        ]
        axis.bar(x + (index - 1) * width, values, width, label=severity.title())
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set_xticks(x, scenes, rotation=25, ha="right")
    axis.set_ylabel("LSTM ADE degradation (raw dataset coordinate units)")
    axis.set_title("Per-scene LSTM ADE degradation from identity switches")
    axis.legend()
    axis.grid(axis="y", alpha=0.3)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _plot_clean_corrupted_trajectory(
    path: Path,
    dataset: ETHSplit,
    event: dict[str, Any],
    switch_steps: int,
) -> None:
    _plot_switch_example(path, dataset, event, switch_steps)


def run_identity_switch_experiment(
    dataset_root: Path = DEFAULT_DATASET_ROOT,
    experiment_dir: Path = DEFAULT_EXPERIMENT_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    dataset_root = Path(dataset_root)
    experiment_dir = Path(experiment_dir)
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"refusing to overwrite non-empty output directory: {output_dir}"
        )
    manifest_path = experiment_dir / "experiment.json"
    checkpoint_path = experiment_dir / "best_lstm.pt"
    if not manifest_path.is_file() or not checkpoint_path.is_file():
        raise FileNotFoundError("selected experiment must contain experiment.json and best_lstm.pt")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "selected" or manifest.get("smoke_test_subset"):
        raise ValueError("identity-switch evaluation requires a full selected checkpoint")
    if manifest.get("selection", {}).get("test_used_for_selection") is not False:
        raise ValueError("selected model must be certified as test-independent")
    config = ExperimentConfig(**manifest["config"])
    if config.device != "cpu":
        raise ValueError("identity-switch experiment requires the selected CPU checkpoint")

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
    if len(candidate_windows) != 5422:
        raise ValueError(
            f"expected 5,422 candidate validation windows, got {len(candidate_windows)}"
        )
    clean_all = evaluate_models(
        candidate_windows, model, normalization, torch.device("cpu"), config.batch_size
    )
    source_hashes = [
        {
            "scene": dataset.scene,
            "path": dataset.source_file.as_posix(),
            "sha256": _file_sha256(dataset.source_file),
        }
        for dataset in validation_datasets
    ]
    seeds: dict[int, dict[str, Any]] = {}
    aggregate_inputs: dict[str, list[dict[str, Any]]] = {
        severity: [] for severity in IDENTITY_SWITCH_SEVERITIES
    }
    representatives: dict[str, tuple[ETHSplit, dict[str, Any]]] = {}
    for seed in CORRUPTION_SEEDS:
        selection = select_identity_switch_events(
            validation_datasets,
            candidate_windows,
            seed,
            config.frame_step,
            config.observation_len,
            max(IDENTITY_SWITCH_SEVERITIES.values()),
        )
        seed_report: dict[str, Any] = {
            "seed": seed,
            "split": "val",
            "candidate_window_count": len(candidate_windows),
            "candidate_window_key_sha256": _window_key_digest(candidate_windows),
            "selection": selection,
            "severity_results": {},
        }
        for severity, switch_steps in IDENTITY_SWITCH_SEVERITIES.items():
            corrupted_windows, corruption = apply_identity_switches(
                candidate_windows,
                validation_datasets,
                selection,
                switch_steps,
            )
            if _window_key_digest(corrupted_windows) != _window_key_digest(candidate_windows):
                raise RuntimeError("identity switching changed the validation window ordering")
            clean_results = evaluate_models(
                candidate_windows, model, normalization, torch.device("cpu"), config.batch_size
            )
            corrupted_results = evaluate_models(
                corrupted_windows, model, normalization, torch.device("cpu"), config.batch_size
            )
            if corrupted_results["window_key_sha256"] != clean_results["window_key_sha256"]:
                raise RuntimeError("clean and corrupted evaluation windows differ")
            if (
                corrupted_results["constant_velocity"]["window_key_sha256"]
                != corrupted_results["lstm"]["window_key_sha256"]
            ):
                raise RuntimeError("CV and LSTM identity-switch window signatures differ")
            enriched = _add_degradation(corrupted_results, clean_results)
            scene_metrics = {}
            for dataset in validation_datasets:
                scene = dataset.scene
                scene_metrics[scene] = {
                    "event_count": corruption["event_count_by_scene"][scene],
                    "affected_window_count": corruption[
                        "affected_window_count_by_scene"
                    ].get(scene, 0),
                    "candidate_window_count": sum(
                        window.scene == scene for window in candidate_windows
                    ),
                    "models": {
                        model_name: {
                            "clean": clean_results[model_name]["per_scene"][scene],
                            "corrupted": enriched[model_name]["per_scene"][scene],
                            "degradation_vs_clean": enriched[model_name][
                                "per_scene_degradation_vs_clean"
                            ][scene],
                        }
                        for model_name in ("constant_velocity", "lstm")
                    },
                }
            seed_result = {
                "seed": seed,
                "severity": severity,
                "switch_steps": switch_steps,
                "candidate_window_count": len(candidate_windows),
                "window_key_sha256": _window_key_digest(candidate_windows),
                "pair_statistics_by_scene": {
                    scene: {
                        field: stats[field]
                        for field in (
                            "eligible_pair_count",
                            "ineligible_pair_count",
                            "selected_pair_count",
                            "selected_event_count",
                        )
                    }
                    for scene, stats in selection["scene_statistics"].items()
                },
                "clean_metrics": clean_results,
                "corrupted_metrics": enriched,
                "degradation_vs_clean": {
                    model_name: enriched[model_name]["degradation_vs_clean"]
                    for model_name in ("constant_velocity", "lstm")
                },
                "scene_metrics": scene_metrics,
                "corruption_statistics": corruption,
            }
            seed_report["severity_results"][severity] = seed_result
            aggregate_inputs[severity].append(seed_result)
            if severity not in representatives and corruption["events"]:
                event = corruption["events"][0]
                representative_scene = next(
                    item for item in validation_datasets if item.scene == event["scene"]
                )
                representatives[severity] = (representative_scene, event)
        seeds[seed] = seed_report

    aggregates: dict[str, Any] = {}
    for severity, switch_steps in IDENTITY_SWITCH_SEVERITIES.items():
        severity_results = aggregate_inputs[severity]
        clean_models = {
            model_name: _aggregate_clean_model([
                item["clean_metrics"][model_name] for item in severity_results
            ])
            for model_name in ("constant_velocity", "lstm")
        }
        corrupted_models = {
            model_name: _aggregate_model([
                item["corrupted_metrics"][model_name] for item in severity_results
            ])
            for model_name in ("constant_velocity", "lstm")
        }
        degradation = {
            model_name: {
                metric: {
                    field: _mean_std([
                        float(item["degradation_vs_clean"][model_name][metric][field])
                        for item in severity_results
                        if item["degradation_vs_clean"][model_name][metric][field] is not None
                    ])
                    for field in ("absolute", "percentage")
                }
                for metric in ("ade", "fde")
            }
            for model_name in ("constant_velocity", "lstm")
        }
        scene_names = [dataset.scene for dataset in validation_datasets]
        aggregates[severity] = {
            "switch_steps": switch_steps,
            "switch_interval_frame_id_span_units": (
                max(0, switch_steps - 1) * config.frame_step
            ),
            "hypothetical_boundary_frame_id_separation_units": (
                switch_steps + 1
            ) * config.frame_step,
            "seeds": list(CORRUPTION_SEEDS),
            "candidate_window_count": len(candidate_windows),
            "affected_window_count": _mean_std([
                float(item["corruption_statistics"]["affected_window_count"])
                for item in severity_results
            ]),
            "unaffected_window_count": _mean_std([
                float(item["corruption_statistics"]["unaffected_window_count"])
                for item in severity_results
            ]),
            "affected_window_percentage": _mean_std([
                float(item["corruption_statistics"]["affected_window_percentage"])
                for item in severity_results
            ]),
            "selected_pair_count": _mean_std([
                float(item["corruption_statistics"]["selected_pair_count"])
                for item in severity_results
            ]),
            "event_count": _mean_std([
                float(item["corruption_statistics"]["event_count"])
                for item in severity_results
            ]),
            "switched_observation_assignment_count": _mean_std([
                float(item["corruption_statistics"]["switched_observation_assignment_count"])
                for item in severity_results
            ]),
            "unique_affected_pedestrian_count": _mean_std([
                float(item["corruption_statistics"]["unique_affected_pedestrian_count"])
                for item in severity_results
            ]),
            "mean_spatial_separation": _mean_std_or_none([
                float(event["mean_spatial_separation"])
                for item in severity_results
                for event in item["corruption_statistics"]["events"]
            ]),
            "max_spatial_separation": _mean_std_or_none([
                float(event["max_spatial_separation"])
                for item in severity_results
                for event in item["corruption_statistics"]["events"]
            ]),
            "event_count_by_scene": {
                scene: _mean_std([
                    float(item["corruption_statistics"]["event_count_by_scene"][scene])
                    for item in severity_results
                ])
                for scene in scene_names
            },
            "eligible_pair_count_by_scene": {
                scene: _mean_std([
                    float(item["pair_statistics_by_scene"][scene]["eligible_pair_count"])
                    for item in severity_results
                ])
                for scene in scene_names
            },
            "ineligible_pair_count_by_scene": {
                scene: _mean_std([
                    float(item["pair_statistics_by_scene"][scene]["ineligible_pair_count"])
                    for item in severity_results
                ])
                for scene in scene_names
            },
            "selected_pair_count_by_scene": {
                scene: _mean_std([
                    float(item["pair_statistics_by_scene"][scene]["selected_pair_count"])
                    for item in severity_results
                ])
                for scene in scene_names
            },
            "affected_window_count_by_scene": {
                scene: _mean_std([
                    float(item["corruption_statistics"]["affected_window_count_by_scene"].get(scene, 0))
                    for item in severity_results
                ])
                for scene in scene_names
            },
            "clean_metrics": clean_models,
            "corrupted_metrics": corrupted_models,
            "degradation_vs_clean": degradation,
            "window_key_sha256_by_seed": {
                str(item["seed"]): item["window_key_sha256"]
                for item in severity_results
            },
        }

    if _file_sha256(checkpoint_path) != checkpoint_hash:
        raise RuntimeError("frozen checkpoint changed during identity-switch evaluation")

    output_dir.mkdir(parents=True, exist_ok=True)
    for seed, report in seeds.items():
        _atomic_json(output_dir / "per_seed" / f"seed-{seed}.json", report)
    _atomic_json(output_dir / "aggregated_results.json", {
        "split": "val",
        "distance_units": "raw dataset coordinate units; physical units unverified",
        "aggregation": (
            "ADE is mean Euclidean error across the 12 future positions per window, then "
            "mean across windows; FDE is error at prediction step 12, then mean across "
            "windows. Cross-seed standard deviations are population standard deviations."
        ),
        "candidate_window_count": len(candidate_windows),
        "candidate_window_key_sha256": _window_key_digest(candidate_windows),
        "clean_baseline_all_candidate_windows": clean_all,
        "severity_results": aggregates,
    })
    _atomic_json(output_dir / "experiment_config.json", {
        "protocol": {
            "split_evaluated": "val",
            "validation_scenes": [dataset.scene for dataset in validation_datasets],
            "candidate_window_count": len(candidate_windows),
            "candidate_window_key_sha256": _window_key_digest(candidate_windows),
            "observation_len": config.observation_len,
            "prediction_horizon": config.horizon_len,
            "frame_step": config.frame_step,
            "stride": config.stride,
            "training_data_accessed": False,
            "test_data_accessed": False,
        },
        "identity_switch": {
            "severities": IDENTITY_SWITCH_SEVERITIES,
            "seeds": list(CORRUPTION_SEEDS),
            "operation": (
                "Symmetric exchange of timestamp-aligned observed x/y positions between "
                "two distinct original pedestrian identities for a temporary interval; "
                "after the interval each identity returns to its original position."
            ),
            "position_provenance_key": "(scene, original_pedestrian_id, frame_id)",
            "synthetic_source_identity_recorded_for_every_observed_input": True,
            "eligibility": (
                "At least three consecutive shared timestamps at frame_step cadence, with "
                "the selected three-timestamp interval wholly contained in at least one "
                "valid observed history for each pedestrian."
            ),
            "selection": (
                "At most one pair/event per scene and no pedestrian reuse. Select from "
                "eligible pair/start candidates with seeded inverse-distance-rank weighted "
                "choice; lower interval mean spatial separation ranks receive greater weight. "
                "No physical-distance threshold is used. Pair/start is common across severities."
            ),
            "limitation": (
                "Simplified temporary symmetric position exchange; it is not a complete "
                "simulation of real tracker association failures."
            ),
            "future_ground_truth_swapped": False,
            "windows_removed_due_to_switching": False,
            "switch_interval_frame_id_span_units_by_severity": {
                str(steps): max(0, steps - 1) * config.frame_step
                for steps in IDENTITY_SWITCH_SEVERITIES.values()
            },
            "hypothetical_boundary_frame_id_separation_units_by_severity": {
                str(steps): (steps + 1) * config.frame_step
                for steps in IDENTITY_SWITCH_SEVERITIES.values()
            },
        },
        "model": {
            "experiment_dir": experiment_dir.as_posix(),
            "checkpoint": checkpoint_path.as_posix(),
            "checkpoint_sha256": checkpoint_hash,
            "architecture_config": config.lstm_config().as_dict(),
            "selected_epoch": checkpoint["selected_epoch"],
            "selected_validation_ade": checkpoint["selected_validation_ade"],
            "training_performed": False,
            "test_split_opened": False,
        },
        "data_sources": source_hashes,
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "device": "cpu",
        },
    })
    _write_summary_csv(output_dir / "summary_comparison.csv", aggregates)

    for severity in IDENTITY_SWITCH_SEVERITIES:
        if severity not in representatives:
            continue
        dataset, event = representatives[severity]
        _plot_switch_example(
            output_dir / f"identity-switch-example-{severity}.png",
            dataset,
            event,
            IDENTITY_SWITCH_SEVERITIES[severity],
        )
    _plot_severity_degradation(
        output_dir / "severity-vs-ade-fde-degradation.png", aggregates
    )
    _plot_affected_windows(
        output_dir / "affected-window-percentage.png", aggregates
    )
    _plot_scene_degradation(output_dir / "per-scene-prediction-degradation.png", aggregates)
    if "high" in representatives:
        dataset, event = representatives["high"]
        _plot_clean_corrupted_trajectory(
            output_dir / "clean-vs-corrupted-trajectory.png",
            dataset,
            event,
            IDENTITY_SWITCH_SEVERITIES["high"],
        )
    return {
        "output_dir": output_dir.as_posix(),
        "candidate_window_count": len(candidate_windows),
        "clean_baseline_all_candidate_windows": clean_all,
        "severity_results": aggregates,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--experiment-dir", type=Path, default=DEFAULT_EXPERIMENT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    arguments = parser.parse_args(argv)
    result = run_identity_switch_experiment(
        arguments.dataset_root, arguments.experiment_dir, arguments.output_dir
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
