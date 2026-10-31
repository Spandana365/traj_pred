"""Supplementary tracking-error analysis; these are not official TrackEval metrics.

Matching mirrors TrackEval's MOTChallenge preprocessing and CLEAR rules, so the
counts can be cross-checked against the official result:

* Evaluated ground truth is ``conf == 1`` and ``class == 1`` (pedestrian).
* Predictions matched at IoU >= 0.5 to a distractor class are removed first.
* Per frame, one-to-one Hungarian matching at IoU >= 0.5 maximises IoU, and a
  pair matched in the immediately preceding frame always takes priority.

All positions are image pixels. No camera-motion correction, ground-plane
projection or metric scaling is applied.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

from trackvision.tracking.mot17.formats import box_centers, rows_by_frame


MATCH_IOU_THRESHOLD = 0.5
PEDESTRIAN_CLASS = 1
DISTRACTOR_CLASSES = (2, 7, 8, 12)
CONTINUATION_BONUS = 1000.0
EPSILON = np.finfo(np.float64).eps
AVAILABILITY_SECONDS = (0.5, 1.0, 2.0, 3.0, 4.8, 8.0)
MATCH_FIELDS = (
    "frame",
    "gt_id",
    "pred_id",
    "iou",
    "center_error_px",
    "center_error_over_height",
    "width_error_px",
    "height_error_px",
    "gt_height_px",
    "visibility",
)


@dataclass(frozen=True)
class SequenceMatches:
    """Frame-level matching result for one sequence."""

    matches: np.ndarray
    evaluated_gt: np.ndarray
    false_positives: np.ndarray
    distractor_predictions: np.ndarray

    @property
    def removed_as_distractor(self) -> int:
        return len(self.distractor_predictions)

    @property
    def true_positive_count(self) -> int:
        return len(self.matches)

    @property
    def false_positive_count(self) -> int:
        return len(self.false_positives)

    @property
    def false_negative_count(self) -> int:
        return len(self.evaluated_gt) - len(self.matches)


def iou_matrix(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Return pairwise IoU for two (N, 4) arrays of left, top, width, height."""
    first = np.asarray(first, dtype=np.float64).reshape(-1, 4)
    second = np.asarray(second, dtype=np.float64).reshape(-1, 4)
    if len(first) == 0 or len(second) == 0:
        return np.zeros((len(first), len(second)), dtype=np.float64)
    first_max = first[:, :2] + first[:, 2:]
    second_max = second[:, :2] + second[:, 2:]
    overlap_min = np.maximum(first[:, None, :2], second[None, :, :2])
    overlap_max = np.minimum(first_max[:, None, :], second_max[None, :, :])
    overlap = np.clip(overlap_max - overlap_min, 0.0, None)
    intersection = overlap[..., 0] * overlap[..., 1]
    area_first = first[:, 2] * first[:, 3]
    area_second = second[:, 2] * second[:, 3]
    union = area_first[:, None] + area_second[None, :] - intersection
    return np.divide(
        intersection, union, out=np.zeros_like(intersection), where=union > 0
    )


def assign_pairs(
    iou: np.ndarray,
    threshold: float = MATCH_IOU_THRESHOLD,
    bonus: np.ndarray | None = None,
) -> list[tuple[int, int]]:
    """Return one-to-one (row, column) pairs at or above the IoU threshold."""
    if iou.size == 0:
        return []
    eligible = iou >= threshold - EPSILON
    scores = np.where(eligible, iou, 0.0)
    if bonus is not None:
        scores = scores + np.where(eligible, bonus, 0.0)
    rows, columns = linear_sum_assignment(-scores)
    return [
        (int(row), int(column))
        for row, column in zip(rows, columns)
        if scores[row, column] > EPSILON
    ]


def split_distractor_predictions(
    gt_rows: np.ndarray, predictions: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Separate kept predictions from those matched to distractor classes."""
    if len(gt_rows) == 0 or len(predictions) == 0:
        return predictions, predictions[:0]
    pairs = assign_pairs(iou_matrix(gt_rows[:, 2:6], predictions[:, 2:6]))
    removed = [
        column
        for row, column in pairs
        if int(gt_rows[row, 7]) in DISTRACTOR_CLASSES
    ]
    return np.delete(predictions, removed, axis=0), predictions[removed]


def remove_distractor_predictions(
    gt_rows: np.ndarray, predictions: np.ndarray
) -> np.ndarray:
    """Drop predictions matched to distractor classes, as TrackEval preprocessing does."""
    return split_distractor_predictions(gt_rows, predictions)[0]


def match_sequence(
    ground_truth: np.ndarray,
    predictions: np.ndarray,
    threshold: float = MATCH_IOU_THRESHOLD,
) -> SequenceMatches:
    """Match predictions to evaluated ground truth over a whole sequence."""
    gt_frames = rows_by_frame(ground_truth)
    prediction_frames = rows_by_frame(predictions)
    evaluated_rows = []
    match_rows = []
    false_positive_rows = []
    distractor_rows = []
    previous: dict[int, int] = {}
    empty = np.empty((0, predictions.shape[1]), dtype=np.float64)

    for frame in sorted(set(gt_frames) | set(prediction_frames)):
        frame_gt = gt_frames.get(frame, np.empty((0, 9), dtype=np.float64))
        frame_predictions = prediction_frames.get(frame, empty)
        kept, distractors = split_distractor_predictions(frame_gt, frame_predictions)
        distractor_rows.append(distractors)
        evaluated = frame_gt[
            (frame_gt[:, 6] == 1) & (frame_gt[:, 7] == PEDESTRIAN_CLASS)
        ]
        evaluated_rows.append(evaluated)

        iou = iou_matrix(evaluated[:, 2:6], kept[:, 2:6])
        bonus = np.zeros_like(iou)
        for row, gt_id in enumerate(evaluated[:, 1].astype(np.int64)):
            prior = previous.get(int(gt_id))
            if prior is not None:
                bonus[row, kept[:, 1].astype(np.int64) == prior] = CONTINUATION_BONUS
        pairs = assign_pairs(iou, threshold, bonus)

        # As in TrackEval's CLEAR code, priority does not survive a missed frame.
        previous = {}
        matched_columns = set()
        for row, column in pairs:
            matched_columns.add(column)
            gt_id = int(evaluated[row, 1])
            pred_id = int(kept[column, 1])
            if pred_id >= 0:
                previous[gt_id] = pred_id
            gt_box = evaluated[row, 2:6]
            pred_box = kept[column, 2:6]
            error = box_centers(pred_box)[0] - box_centers(gt_box)[0]
            distance = float(np.hypot(error[0], error[1]))
            match_rows.append(
                (
                    frame,
                    gt_id,
                    pred_id,
                    iou[row, column],
                    distance,
                    distance / gt_box[3],
                    pred_box[2] - gt_box[2],
                    pred_box[3] - gt_box[3],
                    gt_box[3],
                    evaluated[row, 8],
                )
            )
        unmatched = [
            column for column in range(len(kept)) if column not in matched_columns
        ]
        if unmatched:
            false_positive_rows.append(kept[unmatched])

    matches = (
        np.asarray(match_rows, dtype=np.float64)
        if match_rows
        else np.empty((0, len(MATCH_FIELDS)), dtype=np.float64)
    )
    return SequenceMatches(
        matches=matches,
        evaluated_gt=(
            np.vstack(evaluated_rows)
            if evaluated_rows
            else np.empty((0, 9), dtype=np.float64)
        ),
        false_positives=(
            np.vstack(false_positive_rows) if false_positive_rows else empty
        ),
        distractor_predictions=(
            np.vstack(distractor_rows) if distractor_rows else empty
        ),
    )


def run_lengths(mask: Sequence[bool]) -> list[int]:
    """Return the lengths of consecutive True runs."""
    lengths = []
    current = 0
    for value in mask:
        if value:
            current += 1
        elif current:
            lengths.append(current)
            current = 0
    if current:
        lengths.append(current)
    return lengths


def frame_gaps(frames: Sequence[int]) -> list[int]:
    """Return the number of missing frames in each gap of a sorted frame list."""
    values = np.asarray(sorted(frames), dtype=np.int64)
    if len(values) < 2:
        return []
    return [int(step) - 1 for step in np.diff(values) if step > 1]


def identity_runs(identities: Sequence[int]) -> list[int]:
    """Return run lengths of one unbroken identity; 0 marks an untracked frame."""
    lengths = []
    current_id = 0
    current = 0
    for identity in identities:
        if identity != 0 and identity == current_id:
            current += 1
            continue
        if current:
            lengths.append(current)
        current_id = identity
        current = 1 if identity != 0 else 0
    if current:
        lengths.append(current)
    return lengths


def ground_truth_continuity(
    result: SequenceMatches, frame_rate: float
) -> list[dict[str, Any]]:
    """Describe tracker coverage over each evaluated ground-truth pedestrian."""
    matched_ids: dict[tuple[int, int], int] = {
        (int(row[1]), int(row[0])): int(row[2]) for row in result.matches
    }
    records = []
    for gt_id in np.unique(result.evaluated_gt[:, 1]).astype(np.int64):
        rows = result.evaluated_gt[result.evaluated_gt[:, 1] == gt_id]
        frames = sorted(int(frame) for frame in rows[:, 0])
        # Tracker ids start at 1, so 0 can mark an unmatched ground-truth frame.
        identities = [matched_ids.get((int(gt_id), frame), 0) for frame in frames]
        present = [identity != 0 for identity in identities]
        tracked = [identity for identity in identities if identity != 0]
        missing_runs = run_lengths([not value for value in present])
        interior = (
            run_lengths([not value for value in present[present.index(True) :]])
            if any(present)
            else []
        )
        if interior and not present[-1]:
            interior = interior[:-1]
        continuous = identity_runs(identities)
        records.append(
            {
                "gt_id": int(gt_id),
                "gt_frames": len(frames),
                "gt_seconds": len(frames) / frame_rate,
                "tracked_frames": len(tracked),
                "coverage": len(tracked) / len(frames),
                "missed_frames": len(frames) - len(tracked),
                "missing_runs": missing_runs,
                "interior_gaps": interior,
                "fragmentations": len(interior),
                "identity_switches": sum(
                    1 for left, right in zip(tracked, tracked[1:]) if left != right
                ),
                "distinct_track_ids": len(set(tracked)),
                "continuous_runs": continuous,
                "longest_continuous_frames": max(continuous, default=0),
            }
        )
    return records


def predicted_track_summary(
    predictions: np.ndarray, result: SequenceMatches, frame_rate: float
) -> list[dict[str, Any]]:
    """Describe the length, gaps and ground-truth purity of each predicted track."""
    records = []
    for track_id in np.unique(predictions[:, 1]).astype(np.int64):
        frames = sorted(
            int(frame) for frame in predictions[predictions[:, 1] == track_id, 0]
        )
        matched = result.matches[result.matches[:, 2] == track_id]
        gaps = frame_gaps(frames)
        _, counts = (
            np.unique(matched[:, 1], return_counts=True)
            if len(matched)
            else (None, np.zeros(1, dtype=np.int64))
        )
        records.append(
            {
                "track_id": int(track_id),
                "first_frame": frames[0],
                "last_frame": frames[-1],
                "observed_frames": len(frames),
                "span_frames": frames[-1] - frames[0] + 1,
                "span_seconds": (frames[-1] - frames[0] + 1) / frame_rate,
                "gaps": gaps,
                "missing_frames_inside_span": sum(gaps),
                "matched_frames": len(matched),
                "matched_gt_ids": int(len(np.unique(matched[:, 1]))),
                "dominant_gt_fraction": int(counts.max()) / len(frames),
            }
        )
    return records


def window_availability(
    continuity: Sequence[dict[str, Any]], window_frames: int
) -> dict[str, int | float | None]:
    """Fraction of ground-truth windows covered by one unbroken tracker identity."""
    possible = sum(
        max(0, record["gt_frames"] - window_frames + 1) for record in continuity
    )
    available = sum(
        max(0, run - window_frames + 1)
        for record in continuity
        for run in record["continuous_runs"]
    )
    return {
        "possible_windows": possible,
        "available_windows": available,
        "availability": available / possible if possible else None,
    }


def survival_curve(
    lengths: Sequence[int], horizons: Sequence[int]
) -> list[dict[str, float]]:
    """Return the share of observed lengths that reach each horizon."""
    values = np.asarray(lengths, dtype=np.float64)
    return [
        {
            "horizon": int(horizon),
            "survival": float((values >= horizon).mean()) if len(values) else 0.0,
        }
        for horizon in horizons
    ]


def distribution(values: Sequence[float]) -> dict[str, float | int | None]:
    """Summarise a numeric sample."""
    array = np.asarray(values, dtype=np.float64)
    if len(array) == 0:
        return {"count": 0, "mean": None, "std": None, "median": None, "p5": None, "p95": None, "max": None}
    return {
        "count": int(len(array)),
        "mean": float(array.mean()),
        "std": float(array.std()),
        "median": float(np.median(array)),
        "p5": float(np.percentile(array, 5)),
        "p95": float(np.percentile(array, 95)),
        "max": float(array.max()),
    }


def detection_counts(result: SequenceMatches) -> dict[str, float | int]:
    """Summarise detector or tracker boxes against ground truth."""
    true_positives = result.true_positive_count
    false_positives = result.false_positive_count
    false_negatives = result.false_negative_count
    return {
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "removed_as_distractor": result.removed_as_distractor,
        "recall": true_positives / max(1, true_positives + false_negatives),
        "precision": true_positives / max(1, true_positives + false_positives),
    }
