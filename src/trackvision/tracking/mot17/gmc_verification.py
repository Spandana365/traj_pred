"""Check saved GMC warps against evidence that does not come from GMC itself.

A successful call does not prove useful compensation, so each sequence gets:

* phase correlation, an independent estimate of frame-to-frame image shift;
* photometric error before and after warping the previous frame;
* the spread of ground-truth static objects before and after stabilization.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from trackvision.tracking.mot17.dataset import SequenceInfo, load_ground_truth
from trackvision.tracking.mot17.formats import box_foot_points
from trackvision.tracking.mot17.stabilization import (
    stabilize_mot_points,
    to_first_frame_transforms,
)


# Static persons and occluders; MOT17 annotates these objects as not moving.
STATIC_OBJECT_CLASSES = (7, 9, 10, 11)
MIN_STATIC_OBJECT_FRAMES = 30
ANALYSIS_HEIGHT = 270
ANCHOR_SECONDS = (1.0, 3.0, 8.0)


def warp_center_motion(warp: np.ndarray, width: int, height: int) -> np.ndarray:
    """Return the displacement a warp gives the image centre."""
    center = np.array((width / 2.0, height / 2.0))
    return warp[:2, :2] @ center + warp[:2, 2] - center


def phase_shift(previous: np.ndarray, current: np.ndarray) -> np.ndarray:
    """Estimate the shift of ``current`` relative to ``previous``."""
    window = cv2.createHanningWindow(previous.shape[::-1], cv2.CV_32F)
    (shift_x, shift_y), _ = cv2.phaseCorrelate(previous, current, window)
    return np.array((shift_x, shift_y))


def photometric_errors(
    previous: np.ndarray, current: np.ndarray, warp: np.ndarray
) -> tuple[float, float]:
    """Return mean squared error unwarped and warped, on pixels valid in both."""
    height, width = previous.shape
    warped = cv2.warpAffine(previous, warp, (width, height))
    valid = cv2.warpAffine(np.ones_like(previous), warp, (width, height)) > 0.999
    return (
        float(np.mean((previous[valid] - current[valid]) ** 2)),
        float(np.mean((warped[valid] - current[valid]) ** 2)),
    )


def _load_gray(sequence: SequenceInfo, frame: int, scale: int) -> np.ndarray:
    image = cv2.imread(str(sequence.frame_path(frame)), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(sequence.frame_path(frame))
    size = (image.shape[1] // scale, image.shape[0] // scale)
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA).astype(np.float32)


def _correlation(first: np.ndarray, second: np.ndarray) -> float | None:
    if first.std() == 0 or second.std() == 0:
        return None
    return float(np.corrcoef(first, second)[0, 1])


def anchored_residuals(
    frames: np.ndarray,
    points: np.ndarray,
    transforms: dict[int, np.ndarray],
    horizon: int,
) -> list[tuple[float, float]]:
    """Return (raw, compensated) motion of a static point over ``horizon`` frames.

    The later position is carried back to the anchor frame directly, so only
    warps inside the window contribute. Prediction windows work this way, and it
    avoids the drift that builds up over a whole sequence.
    """
    positions = {int(frame): point - 1.0 for frame, point in zip(frames, points)}
    residuals = []
    for anchor in sorted(positions)[:: max(1, horizon // 2)]:
        later = anchor + horizon
        if later not in positions:
            continue
        transform = np.linalg.inv(transforms[anchor]) @ transforms[later]
        moved = transform[:2, :2] @ positions[later] + transform[:2, 2]
        residuals.append(
            (
                float(np.linalg.norm(positions[later] - positions[anchor])),
                float(np.linalg.norm(moved - positions[anchor])),
            )
        )
    return residuals


def static_object_spread(
    sequence: SequenceInfo, warps: dict[int, np.ndarray]
) -> dict[str, Any]:
    """Compare static-object foot-point spread in raw and stabilized pixels."""
    rows = load_ground_truth(sequence)
    rows = rows[np.isin(rows[:, 7].astype(np.int64), STATIC_OBJECT_CLASSES)]
    transforms = to_first_frame_transforms(warps)
    horizons = {
        seconds: round(seconds * sequence.frame_rate) for seconds in ANCHOR_SECONDS
    }
    anchored: dict[float, list[tuple[float, float]]] = {s: [] for s in horizons}
    raw_spreads = []
    stabilized_spreads = []
    for object_id in np.unique(rows[:, 1]):
        selected = rows[rows[:, 1] == object_id]
        if len(selected) < MIN_STATIC_OBJECT_FRAMES:
            continue
        points = box_foot_points(selected[:, 2:6])
        stabilized = stabilize_mot_points(selected[:, 0], points, transforms)
        raw_spreads.append(float(np.hypot(*points.std(axis=0))))
        stabilized_spreads.append(float(np.hypot(*stabilized.std(axis=0))))
        for seconds, horizon in horizons.items():
            anchored[seconds].extend(
                anchored_residuals(selected[:, 0], points, transforms, horizon)
            )
    return {
        "anchored_motion_pixels": {
            str(seconds): {
                "pairs": len(values),
                "median_raw": float(np.median([v[0] for v in values])) if values else None,
                "median_compensated": float(np.median([v[1] for v in values])) if values else None,
            }
            for seconds, values in anchored.items()
        },
        "static_objects": len(raw_spreads),
        "median_raw_spread_pixels": float(np.median(raw_spreads)) if raw_spreads else None,
        "median_stabilized_spread_pixels": (
            float(np.median(stabilized_spreads)) if stabilized_spreads else None
        ),
        "objects_with_reduced_spread": int(
            sum(after < before for before, after in zip(raw_spreads, stabilized_spreads))
        ),
    }


def verify_sequence(
    sequence: SequenceInfo, warps: dict[int, np.ndarray]
) -> dict[str, Any]:
    """Compare every frame pair's warp with independent image evidence."""
    scale = max(1, round(sequence.height / ANALYSIS_HEIGHT))
    previous = _load_gray(sequence, 1, scale)
    warp_motion = []
    phase_motion = []
    before = []
    after = []
    for frame in range(2, sequence.length + 1):
        current = _load_gray(sequence, frame, scale)
        warp = warps[frame]
        scaled = warp.copy()
        scaled[:, 2] /= scale
        warp_motion.append(warp_center_motion(warp, sequence.width, sequence.height))
        phase_motion.append(phase_shift(previous, current) * scale)
        unwarped_error, warped_error = photometric_errors(previous, current, scaled)
        before.append(unwarped_error)
        after.append(warped_error)
        previous = current

    gmc = np.asarray(warp_motion)
    phase = np.asarray(phase_motion)
    before_errors = np.asarray(before)
    after_errors = np.asarray(after)
    difference = np.hypot(*(gmc - phase).T)
    return {
        "sequence": sequence.name,
        "frame_pairs": len(gmc),
        "analysis_scale": scale,
        "median_warp_center_motion_pixels": float(np.median(np.hypot(*gmc.T))),
        "median_phase_motion_pixels": float(np.median(np.hypot(*phase.T))),
        "median_warp_phase_difference_pixels": float(np.median(difference)),
        "correlation_x": _correlation(gmc[:, 0], phase[:, 0]),
        "correlation_y": _correlation(gmc[:, 1], phase[:, 1]),
        "median_photometric_error_unwarped": float(np.median(before_errors)),
        "median_photometric_error_warped": float(np.median(after_errors)),
        "share_of_frame_pairs_improved": float(np.mean(after_errors < before_errors)),
        "static_objects": static_object_spread(sequence, warps),
    }
