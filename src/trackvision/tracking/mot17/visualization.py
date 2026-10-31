"""Render figures from saved MOT17 frames, detections, tracks and reports."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Sequence

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from trackvision.tracking.mot17.dataset import SequenceInfo
from trackvision.tracking.mot17.formats import load_mot_file, rows_by_frame


PANEL_CONFIDENCE = 0.25
BAR_COLOR = "#2b6cb0"
SECOND_COLOR = "#c05621"


def _track_color(track_id: int) -> tuple[int, int, int]:
    golden = (track_id * 0.61803398875) % 1.0
    hsv = np.uint8([[[int(golden * 179), 220, 255]]])
    return tuple(int(value) for value in cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0])


def _draw_boxes(
    image: np.ndarray,
    rows: np.ndarray,
    labelled: bool,
) -> np.ndarray:
    canvas = image.copy()
    thickness = max(1, round(image.shape[0] / 450))
    scale = image.shape[0] / 1400
    for row in rows:
        left, top = int(row[2] - 1), int(row[3] - 1)
        right, bottom = int(left + row[4]), int(top + row[5])
        color = _track_color(int(row[1])) if labelled else (60, 220, 60)
        cv2.rectangle(canvas, (left, top), (right, bottom), color, thickness)
        if labelled:
            cv2.putText(
                canvas,
                str(int(row[1])),
                (left, max(12, top - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                scale,
                color,
                thickness,
                cv2.LINE_AA,
            )
    return canvas


def render_frame_panel(
    sequence: SequenceInfo,
    detections_path: Path,
    tracks_path: Path,
    output_dir: Path,
) -> Path:
    """Save the original frame beside its YOLO boxes and BoT-SORT identities."""
    frame = max(1, sequence.length // 2)
    image = cv2.imread(str(sequence.frame_path(frame)))
    if image is None:
        raise FileNotFoundError(sequence.frame_path(frame))
    detections = rows_by_frame(load_mot_file(detections_path)).get(frame)
    tracks = rows_by_frame(load_mot_file(tracks_path)).get(frame)
    shown_detections = (
        detections[detections[:, 6] >= PANEL_CONFIDENCE]
        if detections is not None
        else np.empty((0, 10))
    )
    shown_tracks = tracks if tracks is not None else np.empty((0, 10))
    panels = (
        ("Original frame", image),
        (
            f"YOLO detections, conf >= {PANEL_CONFIDENCE} ({len(shown_detections)})",
            _draw_boxes(image, shown_detections, labelled=False),
        ),
        (
            f"BoT-SORT tracks with IDs ({len(shown_tracks)})",
            _draw_boxes(image, shown_tracks, labelled=True),
        ),
    )
    figure, axes = plt.subplots(1, 3, figsize=(18, 4.2))
    for axis, (title, panel) in zip(axes, panels):
        axis.imshow(cv2.cvtColor(panel, cv2.COLOR_BGR2RGB))
        axis.set_title(title, fontsize=11)
        axis.axis("off")
    figure.suptitle(f"{sequence.name}, frame {frame}", fontsize=12)
    figure.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{sequence.name}_frame{frame:06d}.png"
    figure.savefig(path, dpi=110)
    plt.close(figure)
    return path


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def plot_localization(profile_dir: Path, output_dir: Path) -> Path:
    rows = _read_csv(profile_dir / "matched_localization.csv")
    iou = np.asarray([float(row["iou"]) for row in rows])
    center = np.asarray([float(row["center_error_over_height"]) for row in rows])
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].hist(iou, bins=np.linspace(0.5, 1.0, 41), color=BAR_COLOR)
    axes[0].set_xlabel("IoU of matched tracker box and ground truth")
    axes[0].set_title("Bounding-box overlap of matched boxes")
    axes[1].hist(np.clip(center, 0, 0.5), bins=50, color=BAR_COLOR)
    axes[1].set_xlabel("Centre error / ground-truth box height (clipped at 0.5)")
    axes[1].set_title("Centre localization error of matched boxes")
    for axis in axes:
        axis.set_ylabel("Matched boxes")
    figure.tight_layout()
    path = output_dir / "localization_errors.png"
    figure.savefig(path, dpi=130)
    plt.close(figure)
    return path


def plot_track_continuity(profile_dir: Path, output_dir: Path) -> Path:
    tracks = _read_csv(profile_dir / "predicted_tracks.csv")
    gaps = _read_csv(profile_dir / "missing_frame_distribution.csv")
    continuity = _read_csv(profile_dir / "gt_track_continuity.csv")
    figure, axes = plt.subplots(1, 3, figsize=(15, 4))
    spans = np.asarray([float(row["span_seconds"]) for row in tracks])
    axes[0].hist(spans, bins=40, color=BAR_COLOR)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("BoT-SORT track span (seconds)")
    axes[0].set_ylabel("Tracks (log scale)")
    axes[0].set_title("Track lengths")
    interior = [
        int(row["length_frames"]) for row in gaps if row["kind"] == "gt_interior_gap"
    ]
    if interior:
        axes[1].hist(
            np.clip(interior, 1, 60), bins=np.arange(1, 62), color=BAR_COLOR
        )
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Missing frames in a gap (clipped at 60)")
    axes[1].set_ylabel("Gaps (log scale)")
    axes[1].set_title("Gaps inside ground-truth tracks")
    coverage = np.asarray([float(row["coverage"]) for row in continuity])
    axes[2].hist(coverage, bins=20, range=(0, 1), color=BAR_COLOR)
    axes[2].set_xlabel("Share of a pedestrian's frames tracked")
    axes[2].set_ylabel("Ground-truth pedestrians")
    axes[2].set_title("Per-pedestrian coverage")
    figure.tight_layout()
    path = output_dir / "track_lengths_and_missing_frames.png"
    figure.savefig(path, dpi=130)
    plt.close(figure)
    return path


def plot_sequence_metrics(
    metrics: dict[str, dict[str, float]],
    sequence_names: Sequence[str],
    output_dir: Path,
) -> Path:
    labels = [name.removeprefix("MOT17-") for name in sequence_names]
    figure, axes = plt.subplots(1, 3, figsize=(15, 4), sharex=True)
    for axis, metric in zip(axes, ("HOTA", "MOTA", "IDF1")):
        values = [metrics[name][metric] * 100 for name in sequence_names]
        bars = axis.bar(labels, values, color=BAR_COLOR)
        axis.bar_label(bars, fmt="%.1f", fontsize=9)
        axis.axhline(
            metrics["COMBINED_SEQ"][metric] * 100,
            color=SECOND_COLOR,
            linestyle="--",
            label=f"Combined {metrics['COMBINED_SEQ'][metric] * 100:.1f}",
        )
        axis.set_title(metric)
        axis.set_xlabel("MOT17 sequence")
        axis.legend(frameon=False, fontsize=9)
    axes[0].set_ylabel("Score (%)")
    figure.tight_layout()
    path = output_dir / "per_sequence_metrics.png"
    figure.savefig(path, dpi=130)
    plt.close(figure)
    return path


def render_all(
    sequences: Sequence[SequenceInfo],
    output_root: Path,
    detections_dir: Path | None = None,
    tracker_name: str = "botsort",
) -> list[str]:
    """Render every figure from the saved outputs of earlier stages."""
    output_dir = output_root / "visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)
    profile_dir = output_root / "error_profiles"
    detections_dir = detections_dir or output_root / "detections"
    paths: list[Any] = [
        render_frame_panel(
            sequence,
            detections_dir / f"{sequence.name}.txt",
            output_root / "trackers" / tracker_name / "data" / f"{sequence.name}.txt",
            output_dir / "frames",
        )
        for sequence in sequences
    ]
    metrics = json.loads(
        (output_root / "trackeval" / "metrics.json").read_text(encoding="utf-8")
    )["metrics"]
    paths.append(plot_localization(profile_dir, output_dir))
    paths.append(plot_track_continuity(profile_dir, output_dir))
    paths.append(
        plot_sequence_metrics(
            metrics, [sequence.name for sequence in sequences], output_dir
        )
    )
    return [str(path) for path in paths]
