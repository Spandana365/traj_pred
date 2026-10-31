"""Express image positions in one sequence's first-frame camera view.

Convention: the warp saved for frame t maps 0-based pixels of frame t-1 to
frame t. Multiplying the warps gives a transform from frame 1 to frame t, and
its inverse carries a frame t position back to frame 1.

The result is still in image pixels. It removes estimated camera motion from
2D positions; it is not a ground plane or world coordinate system, and small
per-frame errors accumulate as drift over long sequences.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


def load_warps(path: Path) -> dict[int, np.ndarray]:
    """Load saved warps keyed by frame number."""
    warps = {}
    with Path(path).open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            warps[int(row["frame"])] = np.array(
                [
                    [float(row["h00"]), float(row["h01"]), float(row["h02"])],
                    [float(row["h10"]), float(row["h11"]), float(row["h12"])],
                ],
                dtype=np.float64,
            )
    return warps


def to_first_frame_transforms(warps: dict[int, np.ndarray]) -> dict[int, np.ndarray]:
    """Return 3x3 transforms from each frame's 0-based pixels to frame 1's."""
    from_first = np.eye(3)
    transforms = {}
    for frame in sorted(warps):
        step = np.vstack((warps[frame], (0.0, 0.0, 1.0)))
        from_first = step @ from_first
        transforms[frame] = np.linalg.inv(from_first)
    return transforms


def stabilize_mot_points(
    frames: np.ndarray, points: np.ndarray, transforms: dict[int, np.ndarray]
) -> np.ndarray:
    """Map 1-based MOT pixel points to 1-based first-frame view pixels."""
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    result = np.empty_like(points)
    for index, (frame, point) in enumerate(zip(frames, points)):
        transform = transforms[int(frame)]
        result[index] = transform[:2, :2] @ (point - 1.0) + transform[:2, 2] + 1.0
    return result
