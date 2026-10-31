"""Run the Ultralytics BoT-SORT implementation over saved YOLO detections."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np

from trackvision.tracking.mot17.dataset import SequenceInfo
from trackvision.tracking.mot17.formats import (
    format_mot_row,
    load_mot_file,
    mot_ltwh_to_center_xywh,
    rows_by_frame,
    xyxy_to_mot_ltwh,
)


@dataclass(frozen=True)
class BotSortConfig:
    """Ultralytics ``botsort.yaml`` defaults, fixed for all sequences.

    None of these values were tuned against MOT17 ground truth, because the
    training sequences are also the evaluation sequences here.
    """

    track_high_thresh: float = 0.25
    track_low_thresh: float = 0.1
    new_track_thresh: float = 0.25
    track_buffer: int = 30
    match_thresh: float = 0.8
    fuse_score: bool = True
    gmc_method: str = "sparseOptFlow"
    proximity_thresh: float = 0.5
    appearance_thresh: float = 0.8
    with_reid: bool = False
    model: str = "auto"

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


# Identity is expected only on a sequence's first frame, which has no predecessor.
MAX_CONSECUTIVE_IDENTITY_WARPS = 5
WARP_FIELDS = (
    "sequence", "frame", "h00", "h01", "h02", "h10", "h11", "h12", "status",
    "high_confidence_boxes", "frame_width", "frame_height",
)


class GMCPipelineFailure(BaseException):
    """GMC did not operate correctly, so the run must stop.

    Derives from BaseException because Ultralytics wraps GMC in ``except
    Exception`` and substitutes an identity warp; an Exception would be absorbed.
    """


class GMCRecorder:
    """Check and record every GMC result for one sequence.

    Sparse optical flow can also return identity without raising, so identity
    warps are counted and a run of them after the first frame is a failure.
    """

    def __init__(
        self,
        sequence: str = "",
        max_consecutive_identity: int | None = MAX_CONSECUTIVE_IDENTITY_WARPS,
    ):
        self.sequence = sequence
        self.max_consecutive_identity = max_consecutive_identity
        self.records: list[dict[str, Any]] = []
        self.failed_calls = 0
        self.failure_reason: str | None = None
        self._consecutive_identity = 0

    def fail(self, reason: str) -> GMCPipelineFailure:
        self.failed_calls += 1
        self.failure_reason = reason
        return GMCPipelineFailure(f"{self.sequence or 'sequence'}: {reason}")

    def record(self, warp: Any, boxes: Any, image: np.ndarray) -> None:
        frame = len(self.records) + 1
        matrix = np.asarray(warp, dtype=np.float64)
        if matrix.shape != (2, 3):
            raise self.fail(f"frame {frame}: warp has shape {matrix.shape}")
        if not np.isfinite(matrix).all():
            raise self.fail(f"frame {frame}: warp is not finite")
        if np.linalg.det(matrix[:2, :2]) <= 0:
            raise self.fail(f"frame {frame}: warp is not orientation-preserving")
        is_identity = np.array_equal(matrix, np.eye(2, 3))
        if is_identity and frame == 1:
            status = "first_frame_identity"
        elif is_identity:
            status = "identity"
            self._consecutive_identity += 1
        else:
            status = "estimated"
            self._consecutive_identity = 0
        self.records.append(
            {
                "frame": frame,
                "warp": matrix.copy(),
                "status": status,
                "boxes": 0 if boxes is None else len(boxes),
                "width": int(image.shape[1]),
                "height": int(image.shape[0]),
            }
        )
        limit = self.max_consecutive_identity
        if limit is not None and self._consecutive_identity > limit:
            raise self.fail(
                f"frame {frame}: {self._consecutive_identity} consecutive identity warps"
            )

    def confirm_frame(self, frame: int) -> None:
        """Fail if GMC was not reached for this frame, as in the original defect."""
        if len(self.records) != frame:
            raise self.fail(
                f"frame {frame}: {len(self.records)} GMC results recorded, expected {frame}"
            )

    def summary(self) -> dict[str, Any]:
        statuses = [record["status"] for record in self.records]
        translations = [
            float(np.hypot(record["warp"][0, 2], record["warp"][1, 2]))
            for record in self.records
            if record["status"] == "estimated"
        ]
        scales = [
            float(np.sqrt(np.linalg.det(record["warp"][:2, :2])))
            for record in self.records
            if record["status"] == "estimated"
        ]
        return {
            "sequence": self.sequence,
            "calls": len(self.records) + self.failed_calls,
            "successful_calls": len(self.records),
            "failed_calls": self.failed_calls,
            "failure_reason": self.failure_reason,
            "estimated_warps": statuses.count("estimated"),
            "first_frame_identity_warps": statuses.count("first_frame_identity"),
            "non_first_identity_warps": statuses.count("identity"),
            "median_translation_pixels": float(np.median(translations)) if translations else None,
            "max_translation_pixels": max(translations, default=None),
            "min_scale": min(scales, default=None),
            "max_scale": max(scales, default=None),
        }

    def write_warps(self, path: Path) -> None:
        """Save warps at full precision; each maps previous-frame to current-frame pixels."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(WARP_FIELDS)
            for record in self.records:
                writer.writerow(
                    (
                        self.sequence,
                        record["frame"],
                        *(repr(float(value)) for value in record["warp"].ravel()),
                        record["status"],
                        record["boxes"],
                        record["width"],
                        record["height"],
                    )
                )


class RecordingGMC:
    """Wrap the tracker's GMC object so that its failures cannot be hidden."""

    def __init__(self, gmc: Any, recorder: GMCRecorder):
        self._gmc = gmc
        self._recorder = recorder

    @property
    def method(self) -> Any:
        return self._gmc.method

    def reset_params(self) -> None:
        self._gmc.reset_params()

    def apply(self, raw_frame: np.ndarray, detections: Any = None) -> np.ndarray:
        try:
            warp = self._gmc.apply(raw_frame, detections)
        except Exception as error:
            frame = len(self._recorder.records) + 1
            raise self._recorder.fail(f"frame {frame}: GMC raised {error!r}") from error
        self._recorder.record(warp, detections, raw_frame)
        return warp


class DetectionFrame:
    """Saved detections exposing the fields BoT-SORT reads from ``Boxes``."""

    def __init__(self, center_xywh: np.ndarray, conf: np.ndarray, cls: np.ndarray):
        self.xywh = np.asarray(center_xywh, dtype=np.float64).reshape(-1, 4)
        self.conf = np.asarray(conf, dtype=np.float64).reshape(-1)
        self.cls = np.asarray(cls, dtype=np.float64).reshape(-1)

    @classmethod
    def from_mot_rows(cls, rows: np.ndarray | None) -> "DetectionFrame":
        """Build from MOT detection rows; ``None`` is a frame with no detections."""
        if rows is None or len(rows) == 0:
            return cls(np.empty((0, 4)), np.empty(0), np.empty(0))
        return cls(
            mot_ltwh_to_center_xywh(rows[:, 2:6]), rows[:, 6], np.zeros(len(rows))
        )

    @property
    def xyxy(self) -> np.ndarray:
        """Return 0-based (N, 4) corner boxes, as ``Boxes.xyxy`` does."""
        half_size = self.xywh[:, 2:] / 2.0
        return np.hstack((self.xywh[:, :2] - half_size, self.xywh[:, :2] + half_size))

    def __len__(self) -> int:
        return len(self.conf)

    def __getitem__(self, index: Any) -> "DetectionFrame":
        return DetectionFrame(self.xywh[index], self.conf[index], self.cls[index])


def build_tracker(config: BotSortConfig) -> Any:
    """Create a fresh tracker; state never crosses a sequence boundary."""
    from ultralytics.trackers.bot_sort import BOTSORT

    return BOTSORT(SimpleNamespace(**config.as_dict()))


def track_frames(
    detections: dict[int, np.ndarray],
    frame_count: int,
    read_frame: Any,
    config: BotSortConfig,
    recorder: GMCRecorder | None = None,
) -> np.ndarray:
    """Track frames 1..frame_count, returning (frame, id, l, t, w, h, score) rows.

    GMC is strict: any failure raises GMCPipelineFailure. Only an explicit
    ``gmc_method`` of "none" runs without it.
    """
    tracker = build_tracker(config)
    gmc_enabled = tracker.gmc.method is not None
    if gmc_enabled:
        recorder = recorder if recorder is not None else GMCRecorder()
        tracker.gmc = RecordingGMC(tracker.gmc, recorder)
    tracker.reset()
    outputs: list[np.ndarray] = []
    for frame in range(1, frame_count + 1):
        batch = DetectionFrame.from_mot_rows(detections.get(frame))
        tracked = tracker.update(batch, read_frame(frame))
        if gmc_enabled:
            recorder.confirm_frame(frame)
        if len(tracked) == 0:
            continue
        tracked = np.asarray(tracked, dtype=np.float64)
        outputs.append(
            np.column_stack(
                (
                    np.full(len(tracked), frame, dtype=np.float64),
                    tracked[:, 4],
                    xyxy_to_mot_ltwh(tracked[:, :4]),
                    tracked[:, 5],
                )
            )
        )
    return np.vstack(outputs) if outputs else np.empty((0, 7), dtype=np.float64)


def track_sequence(
    sequence: SequenceInfo,
    detections_dir: Path,
    tracker_dir: Path,
    config: BotSortConfig,
    recorder: GMCRecorder | None = None,
) -> dict[str, Any]:
    """Track one sequence from its saved detections and write the MOT result."""
    detection_path = detections_dir / f"{sequence.name}.txt"
    if not detection_path.is_file():
        raise FileNotFoundError(f"Complete detections not found: {detection_path}")
    detection_rows = load_mot_file(detection_path)
    by_frame = rows_by_frame(detection_rows)

    def read_frame(frame: int) -> np.ndarray:
        path = sequence.frame_path(frame)
        image = cv2.imread(str(path))
        if image is None:
            raise FileNotFoundError(f"Could not read frame: {path}")
        return image

    rows = track_frames(by_frame, sequence.length, read_frame, config, recorder)
    data_dir = tracker_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    output_path = data_dir / f"{sequence.name}.txt"
    with output_path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(format_mot_row(row[0], row[1], row[2:6], row[6]) + "\n")
    meta = {
        "sequence": sequence.name,
        "frames": sequence.length,
        "input_detections": int(len(detection_rows)),
        "frames_without_detections": sequence.length - len(by_frame),
        "output_boxes": int(len(rows)),
        "track_ids": int(len(np.unique(rows[:, 1]))) if len(rows) else 0,
        "frames_with_output": int(len(np.unique(rows[:, 0]))) if len(rows) else 0,
    }
    (tracker_dir / f"{sequence.name}.meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    return meta


def estimate_camera_motion(
    sequence: SequenceInfo, method: str, frames: int = 120
) -> dict[str, Any]:
    """Measure frame-to-frame translation with the tracker's own GMC module."""
    from ultralytics.trackers.utils.gmc import GMC

    gmc = GMC(method=method)
    translations = []
    for frame in range(1, min(frames, sequence.length) + 1):
        image = cv2.imread(str(sequence.frame_path(frame)))
        warp = gmc.apply(image)
        if frame > 1:
            translations.append(float(np.hypot(warp[0, 2], warp[1, 2])))
    values = np.asarray(translations, dtype=np.float64)
    return {
        "sequence": sequence.name,
        "frames_sampled": int(len(values) + 1),
        "median_translation_pixels": float(np.median(values)),
        "p95_translation_pixels": float(np.percentile(values, 95)),
    }
