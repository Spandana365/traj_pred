"""Run a pretrained YOLO detector over MOT17 frames, inference only."""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from trackvision.tracking.mot17.dataset import SequenceInfo, file_sha256
from trackvision.tracking.mot17.formats import format_mot_row, xyxy_to_mot_ltwh


PERSON_CLASS_NAME = "person"
PROGRESS_INTERVAL = 25


@dataclass(frozen=True)
class DetectorConfig:
    """One detector configuration, shared by every sequence."""

    model_name: str = "yolov8s"
    weights: str = "runs/mot17_tracking_analysis/weights/yolov8s.pt"
    # Below the tracker's low threshold so the saved set can serve later analyses.
    confidence_threshold: float = 0.05
    iou_threshold: float = 0.7
    image_size: int = 1280
    device: str = "cpu"
    max_detections: int = 1000
    augment: bool = False

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_detector(config: DetectorConfig) -> tuple[Any, int]:
    """Load existing weights in evaluation mode and return the person class id."""
    weights = Path(config.weights)
    if not weights.is_file():
        raise FileNotFoundError(f"Pretrained weights not found: {weights}")
    os.environ.setdefault("YOLO_AUTOINSTALL", "false")
    from ultralytics import YOLO

    model = YOLO(str(weights))
    names = model.names
    person_ids = [index for index, name in names.items() if name == PERSON_CLASS_NAME]
    if len(person_ids) != 1:
        raise ValueError(f"No unique '{PERSON_CLASS_NAME}' class in {weights}")
    return model, int(person_ids[0])


def describe_weights(config: DetectorConfig) -> dict[str, Any]:
    """Fingerprint the checkpoint so an unchanged file can be demonstrated."""
    path = Path(config.weights)
    return {
        "path": str(path),
        "sha256": file_sha256(path),
        "size_bytes": path.stat().st_size,
    }


def _resume_point(partial: Path, progress: Path) -> int:
    """Return the last completed frame, dropping rows written after it."""
    if not (partial.is_file() and progress.is_file()):
        partial.unlink(missing_ok=True)
        return 0
    last_frame = int(json.loads(progress.read_text(encoding="utf-8"))["last_frame"])
    lines = partial.read_text(encoding="utf-8").splitlines()
    kept = [line for line in lines if int(line.split(",", 1)[0]) <= last_frame]
    partial.write_text("".join(f"{line}\n" for line in kept), encoding="utf-8")
    return last_frame


def detect_sequence(
    model: Any,
    person_class: int,
    sequence: SequenceInfo,
    config: DetectorConfig,
    output_dir: Path,
    limit: int | None = None,
) -> dict[str, Any]:
    """Detect pedestrians frame by frame; an interrupted run resumes where it stopped."""
    output_dir.mkdir(parents=True, exist_ok=True)
    final = output_dir / f"{sequence.name}.txt"
    meta_path = output_dir / f"{sequence.name}.meta.json"
    if limit is None and final.is_file() and meta_path.is_file():
        return json.loads(meta_path.read_text(encoding="utf-8"))

    partial = output_dir / f"{sequence.name}.partial"
    progress = output_dir / f"{sequence.name}.progress.json"
    start_frame = _resume_point(partial, progress)
    end_frame = min(sequence.length, limit or sequence.length)
    previous_seconds = (
        float(json.loads(progress.read_text(encoding="utf-8"))["seconds"])
        if start_frame
        else 0.0
    )
    started = time.perf_counter()

    def elapsed() -> float:
        return previous_seconds + time.perf_counter() - started

    with partial.open("a", encoding="utf-8") as stream:
        for frame in range(start_frame + 1, end_frame + 1):
            result = model.predict(
                source=str(sequence.frame_path(frame)),
                conf=config.confidence_threshold,
                iou=config.iou_threshold,
                imgsz=config.image_size,
                device=config.device,
                max_det=config.max_detections,
                classes=[person_class],
                augment=config.augment,
                save=False,
                stream=False,
                verbose=False,
            )[0]
            boxes = result.boxes.cpu().numpy()
            for ltwh, confidence in zip(xyxy_to_mot_ltwh(boxes.xyxy), boxes.conf):
                stream.write(format_mot_row(frame, -1, ltwh, confidence) + "\n")
            if frame % PROGRESS_INTERVAL == 0 or frame == end_frame:
                stream.flush()
                os.fsync(stream.fileno())
                progress.write_text(
                    json.dumps({"last_frame": frame, "seconds": elapsed()}),
                    encoding="utf-8",
                )
                print(
                    f"{sequence.name} frame {frame}/{end_frame} "
                    f"({frame / elapsed():.2f} fps overall)",
                    flush=True,
                )

    seconds = elapsed()
    lines = partial.read_text(encoding="utf-8").splitlines()
    frames_with_detections = {int(line.split(",", 1)[0]) for line in lines}
    meta = {
        "sequence": sequence.name,
        "frames_processed": end_frame,
        "frames_expected": sequence.length,
        "complete": end_frame == sequence.length,
        "detections": len(lines),
        "empty_frames": end_frame - len(frames_with_detections),
        "seconds": seconds,
        "frames_per_second": end_frame / seconds if seconds else None,
        "detector": config.as_dict(),
        "coordinate_convention": "MOTChallenge 1-based pixels: left, top, width, height",
    }
    if limit is None or end_frame == sequence.length:
        partial.replace(final)
        progress.unlink(missing_ok=True)
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta
