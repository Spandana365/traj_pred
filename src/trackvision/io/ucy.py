"""Readers for the UCY crowd-annotation spline format.

The source files contain pixel-space spline control points, not uniformly sampled positions.
This module preserves those control points and their frame indices without interpolation.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path

UCY_SCENE_ANNOTATIONS = {
    "UNIV": "students003.vsp",
    "ZARA1": "crowds_zara01.vsp",
    "ZARA2": "crowds_zara02.vsp",
}

_COUNT_PATTERN = re.compile(r"^\s*(\d+)(?:\s|$)")
_NUMBER_PATTERN = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:\s|$)")


class DatasetFormatError(ValueError):
    """Raised when a dataset file does not match its documented format."""


@dataclass(frozen=True)
class UCYPoint:
    """One pixel-space spline control point, with its source video frame."""

    x: float
    y: float
    frame: int
    gaze_direction_degrees: float


@dataclass(frozen=True)
class UCYTrack:
    """One spline from a VSP file; `spline_id` is its 1-based file order."""

    spline_id: int
    points: tuple[UCYPoint, ...]


@dataclass(frozen=True)
class UCYScene:
    """Parsed UCY annotations; coordinates remain in the source pixel frame."""

    scene: str
    tracks: tuple[UCYTrack, ...]
    trailing_lines: tuple[str, ...]


def _next_content_line(lines: list[str], index: int) -> tuple[int, str, int] | None:
    while index < len(lines):
        line_number = index + 1
        text = lines[index].strip()
        index += 1
        if not text:
            continue
        if text.startswith("-") and not _NUMBER_PATTERN.match(text):
            continue
        return line_number, text, index
    return None


def _parse_count(line_number: int, text: str, path: Path) -> int:
    match = _COUNT_PATTERN.match(text)
    if match is None:
        raise DatasetFormatError(f"{path}:{line_number}: expected a non-negative count")
    return int(match.group(1))


def _parse_point(line_number: int, text: str, path: Path) -> UCYPoint:
    fields = text.split()
    if len(fields) < 4:
        raise DatasetFormatError(f"{path}:{line_number}: expected x, y, frame, and gaze")
    try:
        x, y, raw_frame, gaze = (float(field) for field in fields[:4])
    except ValueError as exc:
        raise DatasetFormatError(f"{path}:{line_number}: invalid numeric control point") from exc
    if not all(math.isfinite(value) for value in (x, y, raw_frame, gaze)):
        raise DatasetFormatError(f"{path}:{line_number}: control point values must be finite")
    if raw_frame < 0 or not raw_frame.is_integer():
        raise DatasetFormatError(f"{path}:{line_number}: frame must be a non-negative integer")
    return UCYPoint(x=x, y=y, frame=int(raw_frame), gaze_direction_degrees=gaze)


def read_ucy_annotations(path: Path, scene: str) -> UCYScene:
    """Read a UCY VSP file without reconstructing or resampling its splines."""
    path = Path(path)
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    first = _next_content_line(lines, 0)
    if first is None:
        raise DatasetFormatError(f"{path}: missing spline count")
    line_number, text, index = first
    spline_count = _parse_count(line_number, text, path)

    tracks: list[UCYTrack] = []
    for spline_id in range(1, spline_count + 1):
        count_line = _next_content_line(lines, index)
        if count_line is None:
            raise DatasetFormatError(f"{path}: missing control-point count for spline {spline_id}")
        line_number, text, index = count_line
        point_count = _parse_count(line_number, text, path)

        points: list[UCYPoint] = []
        for _ in range(point_count):
            point_line = _next_content_line(lines, index)
            if point_line is None:
                raise DatasetFormatError(f"{path}: incomplete spline {spline_id}")
            line_number, text, index = point_line
            point = _parse_point(line_number, text, path)
            if points and point.frame <= points[-1].frame:
                raise DatasetFormatError(
                    f"{path}:{line_number}: frames in spline {spline_id} must increase"
                )
            points.append(point)
        tracks.append(UCYTrack(spline_id=spline_id, points=tuple(points)))

    trailing_lines = tuple(line for line in lines[index:] if line.strip())
    return UCYScene(scene=scene, tracks=tuple(tracks), trailing_lines=trailing_lines)


def load_ucy_scene(annotation_dir: Path, scene: str) -> UCYScene:
    """Load one of the three UCY scenes present in the repository's frozen static protocol."""
    scene = scene.upper()
    try:
        filename = UCY_SCENE_ANNOTATIONS[scene]
    except KeyError as exc:
        supported = ", ".join(UCY_SCENE_ANNOTATIONS)
        raise ValueError(f"unsupported UCY scene {scene!r}; choose one of: {supported}") from exc
    return read_ucy_annotations(Path(annotation_dir) / filename, scene)
