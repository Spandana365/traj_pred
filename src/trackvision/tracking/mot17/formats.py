"""Bounding-box conversion and MOTChallenge text files.

MOTChallenge files use 1-based pixel coordinates, so the top-left pixel of an
image is (1, 1). Detectors and the tracker use 0-based image coordinates.
Conversion happens only here.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


# Columns of a MOTChallenge file: frame, id, left, top, width, height, conf, x, y, z.
MOT_COLUMNS = 10


def xyxy_to_mot_ltwh(boxes: np.ndarray) -> np.ndarray:
    """Convert 0-based corner boxes to 1-based left, top, width, height."""
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    return np.column_stack(
        (
            boxes[:, 0] + 1.0,
            boxes[:, 1] + 1.0,
            boxes[:, 2] - boxes[:, 0],
            boxes[:, 3] - boxes[:, 1],
        )
    )


def mot_ltwh_to_center_xywh(boxes: np.ndarray) -> np.ndarray:
    """Convert 1-based left, top, width, height to 0-based centre boxes."""
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    return np.column_stack(
        (
            boxes[:, 0] - 1.0 + boxes[:, 2] / 2.0,
            boxes[:, 1] - 1.0 + boxes[:, 3] / 2.0,
            boxes[:, 2],
            boxes[:, 3],
        )
    )


def box_centers(ltwh: np.ndarray) -> np.ndarray:
    """Return box centres in the coordinates of the input boxes."""
    ltwh = np.asarray(ltwh, dtype=np.float64).reshape(-1, 4)
    return ltwh[:, :2] + ltwh[:, 2:] / 2.0


def box_foot_points(ltwh: np.ndarray) -> np.ndarray:
    """Return bottom-centre points, the usual image-plane pedestrian position."""
    ltwh = np.asarray(ltwh, dtype=np.float64).reshape(-1, 4)
    return np.column_stack((ltwh[:, 0] + ltwh[:, 2] / 2.0, ltwh[:, 1] + ltwh[:, 3]))


def format_mot_row(
    frame: int, track_id: int, ltwh: np.ndarray, confidence: float
) -> str:
    left, top, width, height = (float(value) for value in ltwh)
    return (
        f"{int(frame)},{int(track_id)},{left:.2f},{top:.2f},{width:.2f},"
        f"{height:.2f},{float(confidence):.5f},-1,-1,-1"
    )


def load_mot_file(path: Path) -> np.ndarray:
    """Load a MOTChallenge file as an (N, 10) array; an empty file gives no rows."""
    path = Path(path)
    if path.stat().st_size == 0:
        return np.empty((0, MOT_COLUMNS), dtype=np.float64)
    rows = np.loadtxt(path, delimiter=",", dtype=np.float64, ndmin=2)
    if rows.shape[1] != MOT_COLUMNS:
        raise ValueError(f"{path}: expected {MOT_COLUMNS} columns, got {rows.shape[1]}")
    return rows


def rows_by_frame(rows: np.ndarray) -> dict[int, np.ndarray]:
    """Group rows by their frame column, preserving their order."""
    if len(rows) == 0:
        return {}
    frames = rows[:, 0].astype(np.int64)
    return {int(frame): rows[frames == frame] for frame in np.unique(frames)}
