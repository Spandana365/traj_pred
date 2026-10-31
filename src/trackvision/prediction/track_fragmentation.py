"""Validation-only track-fragmentation experiments for frozen predictors."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from trackvision.io.eth import ETHObservation, ETHSplit, ETHWindow
from trackvision.prediction.detection_dropout import _mean_std
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
from trackvision.prediction.localization_noise import _atomic_json
from trackvision.prediction.lstm import LSTMTrajectoryPredictor


FRAGMENTATION_SEVERITIES = {"low": 1, "medium": 2, "high": 3}
CORRUPTION_SEEDS = (42, 43, 44)
DEFAULT_EXPERIMENT_DIR = Path("runs/prediction/eth/lstm-seed42-h64-l2-scheduler")
DEFAULT_OUTPUT_DIR = Path("runs/error_injection/track_fragmentation")
WindowKey = tuple[str, str, int, int]
TrackKey = tuple[str, int]


@dataclass(frozen=True)
class TrackFragment:
    scene: str
    split: str
    original_pedestrian_id: int
    synthetic_track_id: int
    observations: tuple[ETHObservation, ...]


def _window_key(window: ETHWindow) -> WindowKey:
    return window.scene, window.split, window.pedestrian_id, window.start_frame


def _split_continuous_runs(
    observations: tuple[ETHObservation, ...],
    frame_step: int,
) -> tuple[tuple[ETHObservation, ...], ...]:
    runs: list[list[ETHObservation]] = []
    current: list[ETHObservation] = []
    for observation in sorted(observations, key=lambda item: item.frame):
        if current and observation.frame - current[-1].frame != frame_step:
            runs.append(current)
            current = []
        current.append(observation)
    if current:
        runs.append(current)
    return tuple(tuple(run) for run in runs)


def fragment_validation_tracks(
    validation_datasets: tuple[ETHSplit, ...],
    candidate_windows: tuple[ETHWindow, ...],
    missing_steps: int,
    seed: int,
    frame_step: int,
    observation_len: int = 8,
) -> tuple[tuple[TrackFragment, ...], dict[str, Any]]:
    """Remove one interior run of samples per eligible pedestrian track."""
    if not validation_datasets or any(item.split != "val" for item in validation_datasets):
        raise ValueError("track fragmentation accepts validation datasets only")
    if not candidate_windows or any(item.split != "val" for item in candidate_windows):
        raise ValueError("track fragmentation requires validation candidate windows")
    if missing_steps not in (0, *FRAGMENTATION_SEVERITIES.values()):
        raise ValueError("missing_steps must be 0, 1, 2, or 3 sampled observations")
    if frame_step <= 0 or observation_len <= 0:
        raise ValueError("frame_step and observation_len must be positive")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")

    observations_by_track: dict[TrackKey, list[ETHObservation]] = {}
    for dataset in validation_datasets:
        for observation in dataset.observations:
            observations_by_track.setdefault(
                (dataset.scene, observation.pedestrian_id), []
            ).append(observation)

    windows_by_track: dict[TrackKey, list[ETHWindow]] = {}
    for window in candidate_windows:
        windows_by_track.setdefault((window.scene, window.pedestrian_id), []).append(window)

    choices_by_track: dict[TrackKey, list[tuple[int, int, tuple[ETHObservation, ...]]]] = {}
    track_audit: list[dict[str, Any]] = []
    track_runs: dict[TrackKey, tuple[tuple[ETHObservation, ...], ...]] = {}
    for track_key in sorted(observations_by_track):
        runs = _split_continuous_runs(
            tuple(observations_by_track[track_key]), frame_step
        )
        track_runs[track_key] = runs
        starts = {
            run_index: {
                observation.frame: index
                for index, observation in enumerate(run)
            }
            for run_index, run in enumerate(runs)
        }
        candidate_starts = windows_by_track.get(track_key, [])
        choices: list[tuple[int, int, tuple[ETHObservation, ...]]] = []
        for run_index, run in enumerate(runs):
            position_by_frame = starts[run_index]
            history_starts = [
                position_by_frame[window.start_frame]
                for window in candidate_starts
                if window.start_frame in position_by_frame
            ]
            for gap_start in range(1, len(run) - max(FRAGMENTATION_SEVERITIES.values())):
                if missing_steps and any(
                    start <= gap_start < start + observation_len
                    for start in history_starts
                ):
                    choices.append((run_index, gap_start, run))
        choices_by_track[track_key] = choices
        if missing_steps == 0:
            reason = "zero_fragmentation_control"
        elif not candidate_starts:
            reason = "track_has_no_original_8_observation_12_prediction_windows"
        elif not choices:
            reason = "no_interior_break_location_intersects_an_observed_history"
        else:
            reason = None
        track_audit.append({
            "scene": track_key[0],
            "original_pedestrian_id": track_key[1],
            "original_continuous_run_count": len(runs),
            "original_observation_count": len(observations_by_track[track_key]),
            "candidate_window_count": len(candidate_starts),
            "eligible_break_location_count": len(choices),
            "eligible_for_fragmentation": bool(choices),
            "ineligibility_reason": reason,
        })

    scene_max_ids = {
        dataset.scene: max(
            (
                observation.pedestrian_id
                for observation in dataset.observations
            ),
            default=-1,
        )
        for dataset in validation_datasets
    }
    next_synthetic_id = {scene: max_id + 1 for scene, max_id in scene_max_ids.items()}
    chosen: dict[TrackKey, tuple[int, int, tuple[ETHObservation, ...]]] = {}
    for track_key in sorted(choices_by_track):
        choices = choices_by_track[track_key]
        if choices:
            salt = hashlib.sha256(
                f"{track_key[0]}:{track_key[1]}".encode("utf-8")
            ).digest()
            selected_index = (seed + int.from_bytes(salt[:8], "big")) % len(choices)
            chosen[track_key] = choices[selected_index]

    fragments: list[TrackFragment] = []
    removed_frames_by_track: dict[TrackKey, set[int]] = {}
    events: list[dict[str, Any]] = []
    for track_key in sorted(track_runs):
        selected = chosen.get(track_key)
        for run_index, run in enumerate(track_runs[track_key]):
            if selected is None or selected[0] != run_index:
                fragments.append(TrackFragment(
                    track_key[0], "val", track_key[1], track_key[1], run
                ))
                continue
            _, gap_start, _ = selected
            before = run[:gap_start]
            removed = run[gap_start:gap_start + missing_steps]
            after = run[gap_start + missing_steps:]
            if not before or not after or len(removed) != missing_steps:
                raise RuntimeError("selected fragmentation location is not an interior gap")
            synthetic_id = next_synthetic_id[track_key[0]]
            next_synthetic_id[track_key[0]] += 1
            fragments.extend((
                TrackFragment(track_key[0], "val", track_key[1], track_key[1], before),
                TrackFragment(track_key[0], "val", track_key[1], synthetic_id, after),
            ))
            removed_frames_by_track.setdefault(track_key, set()).update(
                observation.frame for observation in removed
            )
            events.append({
                "scene": track_key[0],
                "original_pedestrian_id": track_key[1],
                "original_continuous_run_index": run_index,
                "original_run_first_frame_id": run[0].frame,
                "gap_start_sample_index": gap_start,
                "gap_end_sample_index_exclusive": gap_start + missing_steps,
                "synthetic_track_id_before": track_key[1],
                "synthetic_track_id_after": synthetic_id,
                "missing_sampled_observation_count": len(removed),
                "removed_frame_ids": [observation.frame for observation in removed],
                "first_removed_frame_id": removed[0].frame,
                "last_removed_frame_id": removed[-1].frame,
                "last_frame_id_before_boundary": before[-1].frame,
                "first_frame_id_after_boundary": after[0].frame,
                "boundary_frame_id_gap_units": after[0].frame - before[-1].frame,
                "fragment_lengths_sampled_observations": [
                    len(before), len(after)
                ],
                "fragment_frame_ranges": [
                    [before[0].frame, before[-1].frame],
                    [after[0].frame, after[-1].frame],
                ],
                "eligible_break_location_count": len(choices_by_track[track_key]),
            })

    audit_by_key = {
        (item["scene"], item["original_pedestrian_id"]): item
        for item in track_audit
    }
    for track_key, event in (
        ((event["scene"], event["original_pedestrian_id"]), event)
        for event in events
    ):
        audit_by_key[track_key]["selected_fragmentation"] = event
    for record in track_audit:
        record.setdefault("selected_fragmentation", None)

    return tuple(fragments), {
        "seed": seed,
        "missing_sampled_observation_count": missing_steps,
        "frame_step": frame_step,
        "track_count": len(track_audit),
        "fragmented_pedestrian_track_count": len(events),
        "resulting_continuous_fragment_count": len(fragments),
        "generated_synthetic_track_id_count": len(events),
        "selected_breaks": events,
        "track_eligibility": track_audit,
        "removed_frame_ids_by_track": removed_frames_by_track,
    }


def construct_fragment_windows(
    fragments: tuple[TrackFragment, ...],
    validation_datasets: tuple[ETHSplit, ...],
    frame_step: int,
    observation_len: int = 8,
    horizon_len: int = 12,
) -> tuple[tuple[ETHWindow, ...], dict[WindowKey, int]]:
    """Build histories from fragments and attach unchanged source ground-truth futures."""
    if not validation_datasets or any(item.split != "val" for item in validation_datasets):
        raise ValueError("fragment windows require validation datasets only")
    if frame_step <= 0 or observation_len <= 0 or horizon_len <= 0:
        raise ValueError("window dimensions and frame_step must be positive")
    source_by_track: dict[TrackKey, dict[int, ETHObservation]] = {}
    for dataset in validation_datasets:
        for observation in dataset.observations:
            source_by_track.setdefault(
                (dataset.scene, observation.pedestrian_id), {}
            )[observation.frame] = observation

    windows_by_key: dict[WindowKey, ETHWindow] = {}
    fragment_id_by_key: dict[WindowKey, int] = {}
    for fragment in fragments:
        track_key = (fragment.scene, fragment.original_pedestrian_id)
        original_by_frame = source_by_track[track_key]
        observations = fragment.observations
        for start in range(0, len(observations) - observation_len + 1):
            observed = observations[start:start + observation_len]
            if any(
                right.frame - left.frame != frame_step
                for left, right in zip(observed, observed[1:])
            ):
                continue
            future_frames = tuple(
                observed[-1].frame + frame_step * step
                for step in range(1, horizon_len + 1)
            )
            if any(frame not in original_by_frame for frame in future_frames):
                continue
            future = tuple(original_by_frame[frame] for frame in future_frames)
            window = ETHWindow(
                scene=fragment.scene,
                split="val",
                pedestrian_id=fragment.original_pedestrian_id,
                start_frame=observed[0].frame,
                observed=observed,
                future=future,
            )
            key = _window_key(window)
            if key in windows_by_key:
                raise RuntimeError(f"duplicate synthetic-fragment window: {key}")
            windows_by_key[key] = window
            fragment_id_by_key[key] = fragment.synthetic_track_id
    return tuple(windows_by_key.values()), fragment_id_by_key


def _unusable_window_records(
    candidate_windows: tuple[ETHWindow, ...],
    retained_keys: set[WindowKey],
    fragmentation: dict[str, Any],
    frame_step: int,
) -> list[dict[str, Any]]:
    removed = fragmentation["removed_frame_ids_by_track"]
    event_by_track = {
        (event["scene"], event["original_pedestrian_id"]): event
        for event in fragmentation["selected_breaks"]
    }
    records: list[dict[str, Any]] = []
    for window in candidate_windows:
        key = _window_key(window)
        if key in retained_keys:
            continue
        track_key = (window.scene, window.pedestrian_id)
        missing_frames = sorted(
            frame for frame in removed.get(track_key, set())
            if window.start_frame <= frame < window.start_frame + 8 * frame_step
        )
        event = event_by_track.get(track_key)
        records.append({
            "scene": window.scene,
            "original_pedestrian_id": window.pedestrian_id,
            "start_frame": window.start_frame,
            "reason": (
                "fragmentation_gap_in_observed_history"
                if missing_frames else "no_single_fragment_contains_observed_history"
            ),
            "missing_observed_frame_ids": missing_frames,
            "fragmentation_event_track_id": (
                event["synthetic_track_id_after"] if event else None
            ),
        })
    return records


def _availability_by_scene(
    candidate_windows: tuple[ETHWindow, ...],
    retained_windows: tuple[ETHWindow, ...],
) -> dict[str, dict[str, int | float]]:
    scenes = sorted({window.scene for window in candidate_windows})
    result: dict[str, dict[str, int | float]] = {}
    retained_keys = {_window_key(window) for window in retained_windows}
    for scene in scenes:
        candidates = [window for window in candidate_windows if window.scene == scene]
        retained = sum(_window_key(window) in retained_keys for window in candidates)
        candidate_count = len(candidates)
        result[scene] = {
            "candidate_window_count": candidate_count,
            "retained_window_count": retained,
            "unusable_window_count": candidate_count - retained,
            "window_retention_percentage": 100.0 * retained / candidate_count,
            "window_loss_percentage": 100.0 * (candidate_count - retained) / candidate_count,
        }
    return result


def _cross_boundary_future_count(
    windows: tuple[ETHWindow, ...],
    fragment_id_by_key: dict[WindowKey, int],
    fragments: tuple[TrackFragment, ...],
    removed_frame_ids_by_track: dict[TrackKey, set[int]],
) -> int:
    synthetic_id_by_observation: dict[tuple[str, int, int], int] = {}
    for fragment in fragments:
        for point in fragment.observations:
            synthetic_id_by_observation[
                (fragment.scene, fragment.original_pedestrian_id, point.frame)
            ] = fragment.synthetic_track_id

    crossed = 0
    for window in windows:
        key = _window_key(window)
        scene, _, pedestrian_id, _ = key
        history_id = fragment_id_by_key[key]
        removed = removed_frame_ids_by_track.get((scene, pedestrian_id), set())
        boundary_crossed = False
        for point in window.future:
            if point.frame in removed:
                boundary_crossed = True
                break
            future_id = synthetic_id_by_observation.get(
                (scene, pedestrian_id, point.frame)
            )
            if future_id is not None and future_id != history_id:
                boundary_crossed = True
                break
        crossed += int(boundary_crossed)
    return crossed


def _degradation_value(value: float, clean: float) -> dict[str, float | None]:
    absolute = value - clean
    return {
        "absolute": absolute,
        "percentage": None if clean == 0 else 100.0 * absolute / abs(clean),
    }


def _matched_degradation(
    fragmented: dict[str, Any],
    clean: dict[str, Any],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for model in ("constant_velocity", "lstm"):
        result[model] = {
            metric: _degradation_value(
                float(fragmented[model][metric]), float(clean[model][metric])
            )
            for metric in ("ade", "fde")
        }
        result[model]["per_timestep_ade"] = [
            {
                "timestep": fragment_step["timestep"],
                **_degradation_value(
                    float(fragment_step["ade"]),
                    float(clean_step["ade"]),
                ),
            }
            for fragment_step, clean_step in zip(
                fragmented[model]["ade_by_timestep"],
                clean[model]["ade_by_timestep"],
                strict=True,
            )
        ]
    return result


def _aggregate_metrics(records: list[dict[str, Any]], field: str) -> dict[str, Any]:
    models: dict[str, Any] = {}
    for model in ("constant_velocity", "lstm"):
        model_records = [record[field][model] for record in records]
        aggregate: dict[str, Any] = {
            metric: _mean_std([float(item[metric]) for item in model_records])
            for metric in ("ade", "fde")
        }
        scenes = sorted(model_records[0]["per_scene"])
        aggregate["per_scene"] = {
            scene: {
                metric: _mean_std([
                    float(item["per_scene"][scene][metric]) for item in model_records
                ])
                for metric in ("ade", "fde")
            }
            for scene in scenes
        }
        aggregate["ade_by_timestep"] = [
            {
                "timestep": timestep + 1,
                "ade": _mean_std([
                    float(item["ade_by_timestep"][timestep]["ade"])
                    for item in model_records
                ]),
            }
            for timestep in range(12)
        ]
        models[model] = aggregate
    return models


def aggregate_severity(seed_results: list[dict[str, Any]]) -> dict[str, Any]:
    if not seed_results:
        raise ValueError("cannot aggregate an empty seed list")
    candidate_count = int(seed_results[0]["candidate_window_count"])
    per_scene: dict[str, Any] = {}
    for scene in sorted(seed_results[0]["availability_by_scene"]):
        per_scene[scene] = {
            field: _mean_std([
                float(item["availability_by_scene"][scene][field])
                for item in seed_results
            ])
            for field in (
                "candidate_window_count",
                "retained_window_count",
                "unusable_window_count",
                "window_retention_percentage",
                "window_loss_percentage",
            )
        }
    return {
        "seeds": [int(item["seed"]) for item in seed_results],
        "candidate_window_count": candidate_count,
        "retained_window_count": _mean_std([
            float(item["retained_window_count"]) for item in seed_results
        ]),
        "unusable_window_count": _mean_std([
            float(item["unusable_window_count"]) for item in seed_results
        ]),
        "window_retention_percentage": _mean_std([
            float(item["window_retention_percentage"]) for item in seed_results
        ]),
        "window_loss_percentage": _mean_std([
            float(item["window_loss_percentage"]) for item in seed_results
        ]),
        "fragmented_pedestrian_track_count": _mean_std([
            float(item["fragmentation"]["fragmented_pedestrian_track_count"])
            for item in seed_results
        ]),
        "resulting_continuous_fragment_count": _mean_std([
            float(item["fragmentation"]["resulting_continuous_fragment_count"])
            for item in seed_results
        ]),
        "future_horizon_crossing_fragment_boundary_count": _mean_std([
            float(item["future_horizon_crossing_fragment_boundary_count"])
            for item in seed_results
        ]),
        "availability_by_scene": per_scene,
        "clean_baseline_on_retained_windows": _aggregate_metrics(
            seed_results, "clean_baseline_on_retained_windows"
        ),
        "fragmented_input_metrics": _aggregate_metrics(seed_results, "fragmented_input_metrics"),
        "fragmentation_vs_matched_clean": {
            model: {
                metric: {
                    field: _mean_std([
                        float(item["fragmentation_vs_matched_clean"][model][metric][field])
                        for item in seed_results
                    ])
                    for field in ("absolute", "percentage")
                }
                for metric in ("ade", "fde")
            }
            for model in ("constant_velocity", "lstm")
        },
        "candidate_window_key_sha256": seed_results[0]["candidate_window_key_sha256"],
        "retained_window_key_sha256_by_seed": {
            str(item["seed"]): item["retained_window_key_sha256"]
            for item in seed_results
        },
    }


def _write_summary_csv(path: Path, clean_all: dict[str, Any], aggregate: dict[str, Any]) -> None:
    fields = (
        "severity", "model", "candidate_windows", "retained_windows_mean",
        "unusable_windows_mean", "retention_percentage_mean", "loss_percentage_mean",
        "fragmented_tracks_mean", "clean_ade_mean", "clean_fde_mean",
        "fragmented_ade_mean", "fragmented_ade_std", "fragmented_fde_mean",
        "fragmented_fde_std", "ade_degradation_mean", "ade_degradation_percentage_mean",
        "fde_degradation_mean", "fde_degradation_percentage_mean",
    )
    with path.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=fields)
        writer.writeheader()
        for model in ("constant_velocity", "lstm"):
            writer.writerow({
                "severity": "clean_all_candidates",
                "model": model,
                "candidate_windows": clean_all["window_count"],
                "retained_windows_mean": clean_all["window_count"],
                "unusable_windows_mean": 0,
                "retention_percentage_mean": 100.0,
                "loss_percentage_mean": 0.0,
                "fragmented_tracks_mean": 0,
                "clean_ade_mean": clean_all[model]["ade"],
                "clean_fde_mean": clean_all[model]["fde"],
                "fragmented_ade_mean": clean_all[model]["ade"],
                "fragmented_ade_std": 0.0,
                "fragmented_fde_mean": clean_all[model]["fde"],
                "fragmented_fde_std": 0.0,
                "ade_degradation_mean": 0.0,
                "ade_degradation_percentage_mean": 0.0,
                "fde_degradation_mean": 0.0,
                "fde_degradation_percentage_mean": 0.0,
            })
        for severity, result in aggregate.items():
            for model in ("constant_velocity", "lstm"):
                clean = result["clean_baseline_on_retained_windows"][model]
                fragmented = result["fragmented_input_metrics"][model]
                ade_deg = result["fragmentation_vs_matched_clean"][model]["ade"]
                fde_deg = result["fragmentation_vs_matched_clean"][model]["fde"]
                writer.writerow({
                    "severity": severity,
                    "model": model,
                    "candidate_windows": result["candidate_window_count"],
                    "retained_windows_mean": result["retained_window_count"]["mean"],
                    "unusable_windows_mean": result["unusable_window_count"]["mean"],
                    "retention_percentage_mean": result["window_retention_percentage"]["mean"],
                    "loss_percentage_mean": result["window_loss_percentage"]["mean"],
                    "fragmented_tracks_mean": result["fragmented_pedestrian_track_count"]["mean"],
                    "clean_ade_mean": clean["ade"]["mean"],
                    "clean_fde_mean": clean["fde"]["mean"],
                    "fragmented_ade_mean": fragmented["ade"]["mean"],
                    "fragmented_ade_std": fragmented["ade"]["std"],
                    "fragmented_fde_mean": fragmented["fde"]["mean"],
                    "fragmented_fde_std": fragmented["fde"]["std"],
                    "ade_degradation_mean": ade_deg["absolute"]["mean"],
                    "ade_degradation_percentage_mean": ade_deg["percentage"]["mean"],
                    "fde_degradation_mean": fde_deg["absolute"]["mean"],
                    "fde_degradation_percentage_mean": fde_deg["percentage"]["mean"],
                })


def _plot_trajectory_fragmentation(
    path: Path,
    event: dict[str, Any],
    source_datasets: tuple[ETHSplit, ...],
) -> None:
    dataset = next(item for item in source_datasets if item.scene == event["scene"])
    track = sorted(
        (
            observation for observation in dataset.observations
            if observation.pedestrian_id == event["original_pedestrian_id"]
        ),
        key=lambda item: item.frame,
    )
    removed_frames = set(event["removed_frame_ids"])
    run = [
        observation for observation in track
        if event["fragment_frame_ranges"][0][0]
        <= observation.frame
        <= event["fragment_frame_ranges"][1][1]
    ]
    before = [item for item in run if item.frame < event["first_removed_frame_id"]]
    removed = [item for item in run if item.frame in removed_frames]
    after = [item for item in run if item.frame > event["last_removed_frame_id"]]
    figure, axis = plt.subplots(figsize=(8, 6))
    axis.plot(
        [item.x for item in run], [item.y for item in run],
        color="0.75", linestyle=":", marker=".", label="Original continuous ground truth",
    )
    axis.plot(
        [item.x for item in before], [item.y for item in before],
        "o-", color="tab:blue",
        label=f"First fragment (synthetic ID {event['synthetic_track_id_before']})",
    )
    axis.plot(
        [item.x for item in after], [item.y for item in after],
        "s-", color="tab:green",
        label=f"Second fragment (synthetic ID {event['synthetic_track_id_after']})",
    )
    if removed:
        axis.scatter(
            [item.x for item in removed], [item.y for item in removed],
            marker="x", s=100, color="tab:red", label="Removed interruption samples",
        )
    axis.scatter(
        [before[-1].x, after[0].x], [before[-1].y, after[0].y],
        marker="D", s=65, color="tab:purple", label="Fragment boundary",
    )
    axis.annotate(
        f"missing {len(removed)} sampled step(s)\n"
        f"frame-ID gap {event['boundary_frame_id_gap_units']}",
        xy=(0.5 * (before[-1].x + after[0].x), 0.5 * (before[-1].y + after[0].y)),
    )
    axis.set_title(
        f"{event['scene']} pedestrian {event['original_pedestrian_id']} "
        f"({len(removed)}-step interruption)"
    )
    axis.set_xlabel("x (raw dataset coordinate units)")
    axis.set_ylabel("y (raw dataset coordinate units)")
    axis.set_aspect("equal", adjustable="datalim")
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _plot_window_retention(path: Path, aggregates: dict[str, Any]) -> None:
    levels = list(FRAGMENTATION_SEVERITIES)
    means = [aggregates[level]["window_retention_percentage"]["mean"] for level in levels]
    stds = [aggregates[level]["window_retention_percentage"]["std"] for level in levels]
    figure, axis = plt.subplots(figsize=(7, 5))
    axis.errorbar(
        [FRAGMENTATION_SEVERITIES[level] for level in levels],
        means, yerr=stds, marker="o", capsize=4,
    )
    axis.set_xticks([1, 2, 3], ["Low (1)", "Medium (2)", "High (3)"])
    axis.set_xlabel("Consecutive missing sampled observations")
    axis.set_ylabel("Retained candidate windows (%)")
    axis.set_ylim(bottom=0, top=102)
    axis.grid(True, alpha=0.3)
    axis.set_title("Fragmentation severity versus validation-window availability")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _plot_scene_availability(path: Path, aggregates: dict[str, Any]) -> None:
    scenes = sorted(next(iter(aggregates.values()))["availability_by_scene"])
    x = np.arange(len(scenes))
    figure, axis = plt.subplots(figsize=(11, 5))
    width = 0.24
    for index, severity in enumerate(FRAGMENTATION_SEVERITIES):
        values = [
            aggregates[severity]["availability_by_scene"][scene][
                "window_retention_percentage"
            ]["mean"]
            for scene in scenes
        ]
        axis.bar(x + (index - 1) * width, values, width, label=severity.title())
    axis.set_xticks(x, scenes, rotation=25, ha="right")
    axis.set_ylabel("Retained candidate windows (%)")
    axis.set_ylim(0, 105)
    axis.set_title("Per-scene window availability under track fragmentation")
    axis.legend()
    axis.grid(axis="y", alpha=0.3)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _plot_prediction_metrics(path: Path, aggregates: dict[str, Any]) -> None:
    levels = list(FRAGMENTATION_SEVERITIES)
    x = np.asarray([FRAGMENTATION_SEVERITIES[level] for level in levels])
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    for model, label, color in (
        ("constant_velocity", "Constant Velocity", "tab:blue"),
        ("lstm", "Two-layer LSTM", "tab:orange"),
    ):
        for axis, metric in zip(axes, ("ade", "fde"), strict=True):
            fragmented = [
                aggregates[level]["fragmented_input_metrics"][model][metric]["mean"]
                for level in levels
            ]
            clean = [
                aggregates[level]["clean_baseline_on_retained_windows"][model][metric]["mean"]
                for level in levels
            ]
            axis.plot(x, fragmented, "o-", color=color, label=f"{label}, fragmented")
            axis.plot(
                x, clean, "x--", color=color, alpha=0.7,
                label=f"{label}, matched clean",
            )
            axis.set_title(metric.upper())
            axis.set_xticks(x, ["Low (1)", "Medium (2)", "High (3)"])
            axis.set_xlabel("Missing sampled observation steps")
            axis.set_ylabel("Error (raw dataset coordinate units)")
            axis.grid(True, alpha=0.3)
    axes[0].legend()
    figure.suptitle("Matched-clean versus fragmented-input prediction error")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def run_track_fragmentation_experiment(
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
        raise ValueError("fragmentation evaluation requires a full selected checkpoint")
    if manifest.get("selection", {}).get("test_used_for_selection") is not False:
        raise ValueError("selected model must be certified as test-independent")
    config = ExperimentConfig(**manifest["config"])
    if config.device != "cpu":
        raise ValueError("fragmentation experiment requires the selected CPU checkpoint")

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
    source_files = [
        {
            "scene": dataset.scene,
            "path": dataset.source_file.as_posix(),
            "sha256": _file_sha256(dataset.source_file),
        }
        for dataset in validation_datasets
    ]

    seed_reports: dict[int, dict[str, Any]] = {}
    aggregate_inputs: dict[str, list[dict[str, Any]]] = {
        severity: [] for severity in FRAGMENTATION_SEVERITIES
    }
    representative_events: dict[str, dict[str, Any]] = {}
    for seed in CORRUPTION_SEEDS:
        seed_report: dict[str, Any] = {
            "seed": seed,
            "split": "val",
            "candidate_window_count": len(candidate_windows),
            "candidate_window_key_sha256": _window_key_digest(candidate_windows),
            "severity_results": {},
        }
        for severity, missing_steps in FRAGMENTATION_SEVERITIES.items():
            fragments, fragmentation = fragment_validation_tracks(
                validation_datasets,
                candidate_windows,
                missing_steps,
                seed,
                config.frame_step,
                config.observation_len,
            )
            generated_windows, fragment_id_by_key = construct_fragment_windows(
                fragments,
                validation_datasets,
                config.frame_step,
                config.observation_len,
                config.horizon_len,
            )
            candidate_by_key = {_window_key(window): window for window in candidate_windows}
            generated_by_key = {_window_key(window): window for window in generated_windows}
            if len(candidate_by_key) != len(candidate_windows):
                raise RuntimeError("candidate validation window identities are not unique")
            if not set(generated_by_key).issubset(candidate_by_key):
                raise RuntimeError("fragmentation generated a non-candidate validation window")
            retained_keys = set(generated_by_key)
            fragmented_windows = tuple(
                generated_by_key[key]
                for key in map(_window_key, candidate_windows)
                if key in retained_keys
            )
            clean_matched_windows = tuple(
                window for window in candidate_windows if _window_key(window) in retained_keys
            )
            if not fragmented_windows:
                raise RuntimeError(f"all validation windows became unusable at {severity}")
            for clean_window, fragmented_window in zip(
                clean_matched_windows, fragmented_windows, strict=True
            ):
                if clean_window != fragmented_window:
                    raise RuntimeError(
                        f"surviving coordinates or ground truth changed for "
                        f"{_window_key(clean_window)}"
                    )
                if len(fragmented_window.observed) != 8 or len(fragmented_window.future) != 12:
                    raise RuntimeError("a retained window violates the 8/12 protocol")
                if any(
                    right.frame - left.frame != config.frame_step
                    for left, right in zip(
                        fragmented_window.observed, fragmented_window.observed[1:]
                    )
                ):
                    raise RuntimeError("a retained window has an invalid observation cadence")

            unusable = _unusable_window_records(
                candidate_windows, retained_keys, fragmentation, config.frame_step
            )
            if len(fragmented_windows) + len(unusable) != len(candidate_windows):
                raise RuntimeError("candidate windows were not fully classified")
            clean_results = evaluate_models(
                clean_matched_windows, model, normalization,
                torch.device("cpu"), config.batch_size,
            )
            fragmented_results = evaluate_models(
                fragmented_windows, model, normalization,
                torch.device("cpu"), config.batch_size,
            )
            retained_digest = _window_key_digest(fragmented_windows)
            if retained_digest != _window_key_digest(clean_matched_windows):
                raise RuntimeError("fragmented and clean evaluations use different windows")
            if (
                fragmented_results["window_key_sha256"] != retained_digest
                or clean_results["window_key_sha256"] != retained_digest
                or fragmented_results["constant_velocity"]["window_key_sha256"] != retained_digest
                or fragmented_results["lstm"]["window_key_sha256"] != retained_digest
            ):
                raise RuntimeError("CV, LSTM and clean window signatures differ")

            fragmentation["removed_frame_ids_by_track"] = {
                key: value
                for key, value in fragmentation["removed_frame_ids_by_track"].items()
            }
            availability = _availability_by_scene(
                candidate_windows, fragmented_windows
            )
            retained_count = len(fragmented_windows)
            candidate_count = len(candidate_windows)
            degradation = _matched_degradation(fragmented_results, clean_results)
            seed_result = {
                "seed": seed,
                "severity": severity,
                "missing_sampled_observation_count": missing_steps,
                "candidate_window_count": candidate_count,
                "retained_window_count": retained_count,
                "unusable_window_count": len(unusable),
                "window_retention_percentage": 100.0 * retained_count / candidate_count,
                "window_loss_percentage": 100.0 * len(unusable) / candidate_count,
                "candidate_window_key_sha256": _window_key_digest(candidate_windows),
                "retained_window_key_sha256": retained_digest,
                "retained_window_provenance": [
                    {
                        "scene": window.scene,
                        "ground_truth_pedestrian_id": window.pedestrian_id,
                        "start_frame": window.start_frame,
                        "synthetic_fragment_id": fragment_id_by_key[_window_key(window)],
                    }
                    for window in fragmented_windows
                ],
                "availability_by_scene": availability,
                "unusable_windows": unusable,
                "fragmentation": {
                    key: value for key, value in fragmentation.items()
                    if key != "removed_frame_ids_by_track"
                },
                "future_horizon_crossing_fragment_boundary_count": _cross_boundary_future_count(
                    fragmented_windows,
                    fragment_id_by_key,
                    fragments,
                    fragmentation["removed_frame_ids_by_track"],
                ),
                "clean_baseline_on_retained_windows": clean_results,
                "fragmented_input_metrics": fragmented_results,
                "fragmentation_vs_matched_clean": degradation,
            }
            seed_report["severity_results"][severity] = seed_result
            aggregate_inputs[severity].append(seed_result)
            if seed == CORRUPTION_SEEDS[0] and fragmentation["selected_breaks"]:
                representative_events[severity] = fragmentation["selected_breaks"][0]
        seed_reports[seed] = seed_report

    aggregates = {
        severity: {
            "missing_sampled_observation_count": steps,
            "frame_id_gap_between_surrounding_retained_observations": (
                (steps + 1) * config.frame_step
            ),
            **aggregate_severity(aggregate_inputs[severity]),
        }
        for severity, steps in FRAGMENTATION_SEVERITIES.items()
    }
    checkpoint_hash_after = _file_sha256(checkpoint_path)
    if checkpoint_hash_after != checkpoint_hash:
        raise RuntimeError("frozen checkpoint changed during track-fragmentation evaluation")

    output_dir.mkdir(parents=True, exist_ok=True)
    per_seed_dir = output_dir / "per_seed"
    for seed, report in seed_reports.items():
        _atomic_json(per_seed_dir / f"seed-{seed}.json", report)
    _atomic_json(output_dir / "aggregated_results.json", {
        "split": "val",
        "distance_units": "raw dataset coordinate units; physical units unverified",
        "aggregation": (
            "Per-window ADE is the mean Euclidean error over 12 future positions; FDE is "
            "the error at position 12; windows are equally weighted. Seed summaries use "
            "population standard deviation."
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
            "test_data_accessed": False,
            "training_data_accessed": False,
            "future_ground_truth_used_only_for_scoring": True,
        },
        "fragmentation": {
            "severities": FRAGMENTATION_SEVERITIES,
            "seeds": list(CORRUPTION_SEEDS),
            "selection": (
                "One reproducibly selected interior interruption per eligible original "
                "(scene, pedestrian_id) track; seed-specific deterministic choice over "
                "valid locations that intersect at least one candidate observed history."
            ),
            "location_control": (
                "For a given seed and track, the selected break start is shared across "
                "all severities; eligible locations are chosen to support the three-step "
                "interruption."
            ),
            "track_id_policy": (
                "Preserve (scene, pedestrian_id) as ground-truth provenance. The first "
                "fragment keeps the original integer ID; the subsequent fragment gets a "
                "new scene-local synthetic integer ID recorded separately from scoring IDs."
            ),
            "sampled_step_policy": (
                "Remove the configured consecutive sampled observations from the source "
                "track before window construction; no stitching or interpolation."
            ),
            "one_event_per_eligible_track_assumption": True,
            "gap_frame_id_units": {
                str(steps): (steps + 1) * config.frame_step
                for steps in FRAGMENTATION_SEVERITIES.values()
            },
            "eligibility": (
                "A track is eligible when it has an original candidate prediction window "
                "and at least one interior location in an exact-cadence run where the "
                "removed observations intersect an 8-step candidate history. Every track, "
                "including ineligible tracks and its reason, is listed per seed."
            ),
            "future_crossing_policy": (
                "Windows are retained based only on the observed history fragment. Original "
                "ground-truth futures may span removed samples or a synthetic ID boundary."
            ),
        },
        "model": {
            "experiment_dir": experiment_dir.as_posix(),
            "checkpoint": checkpoint_path.as_posix(),
            "checkpoint_sha256": checkpoint_hash,
            "architecture_config": config.lstm_config().as_dict(),
            "selected_epoch": checkpoint["selected_epoch"],
            "selected_validation_ade": checkpoint["selected_validation_ade"],
            "checkpoint_modified": False,
            "training_performed": False,
            "test_split_opened": False,
        },
        "data_sources": source_files,
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "device": "cpu",
        },
    })
    _write_summary_csv(output_dir / "summary_comparison.csv", clean_all, aggregates)
    for severity in FRAGMENTATION_SEVERITIES:
        event = representative_events.get(severity)
        if event is not None:
            _plot_trajectory_fragmentation(
                output_dir / f"fragmentation-example-{severity}.png",
                event,
                validation_datasets,
            )
    _plot_window_retention(output_dir / "severity-vs-window-retention.png", aggregates)
    _plot_scene_availability(output_dir / "per-scene-window-availability.png", aggregates)
    _plot_prediction_metrics(output_dir / "matched-clean-vs-fragmented-metrics.png", aggregates)
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
    result = run_track_fragmentation_experiment(
        arguments.dataset_root, arguments.experiment_dir, arguments.output_dir
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
