"""Readers and exact-cadence windows for Social-STGCNN ETH/UCY split tables."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


SOCIAL_STGCNN_SCENES = {
    "biwi_eth": "ETH",
    "biwi_hotel": "HOTEL",
    "crowds_zara01": "ZARA1",
    "crowds_zara02": "ZARA2",
    "crowds_zara03": "ZARA3",
    "students001": "STUDENTS001",
    "students003": "UNIV",
    "uni_examples": "UNI_EXAMPLES",
}
SOCIAL_STGCNN_SPLITS = ("train", "val", "test")


class ETHFormatError(ValueError):
    """Raised when a Social-STGCNN trajectory table is malformed."""


@dataclass(frozen=True)
class ETHObservation:
    """One source row: frame index, pedestrian ID, and the two stored coordinates."""

    frame: int
    pedestrian_id: int
    x: float
    y: float


@dataclass(frozen=True)
class ETHSplit:
    """One file from one Social-STGCNN scene and split."""

    scene: str
    split: str
    source_file: Path
    observations: tuple[ETHObservation, ...]


@dataclass(frozen=True)
class ETHWindow:
    """One same-pedestrian, same-file, exact-cadence prediction window."""

    scene: str
    split: str
    pedestrian_id: int
    start_frame: int
    observed: tuple[ETHObservation, ...]
    future: tuple[ETHObservation, ...]


@dataclass(frozen=True)
class ETHValidationIssue:
    """A validation problem tied to a source file and, when applicable, a row."""

    source_file: Path
    code: str
    message: str
    line_number: int | None = None
    value_count: int | None = None

    def format(self) -> str:
        location = str(self.source_file)
        if self.line_number is not None:
            location = f"{location}:{self.line_number}"
        return f"{location}: {self.message}"


@dataclass(frozen=True)
class ETHFileValidation:
    """Parsed observations and all row/file validation problems for one table."""

    source_file: Path
    scene: str
    split: str
    observations: tuple[ETHObservation, ...]
    issues: tuple[ETHValidationIssue, ...]

    @property
    def is_valid(self) -> bool:
        return not self.issues


def validate_social_stgcnn_file(path: Path, scene: str, split: str) -> ETHFileValidation:
    """Parse a table while retaining every malformed-row issue for an audit report."""
    path = Path(path)
    if split not in SOCIAL_STGCNN_SPLITS:
        raise ValueError(f"split must be one of {SOCIAL_STGCNN_SPLITS}, got {split!r}")

    observations: list[ETHObservation] = []
    issues: list[ETHValidationIssue] = []
    seen: dict[tuple[int, int], ETHObservation] = {}
    previous_frame_by_pedestrian: dict[int, int] = {}
    try:
        lines = path.read_text(encoding="utf-8-sig").splitlines()
    except FileNotFoundError:
        return ETHFileValidation(
            path, scene, split, (),
            (ETHValidationIssue(path, "missing_file", "file does not exist"),),
        )
    except OSError as exc:
        return ETHFileValidation(
            path, scene, split, (),
            (ETHValidationIssue(path, "file_read_error", f"cannot read file: {exc}"),),
        )

    for line_number, line in enumerate(lines, start=1):
        fields = line.split()
        if not fields:
            continue
        if len(fields) != 4:
            issues.append(ETHValidationIssue(
                path, "column_count",
                f"expected 4 columns, found {len(fields)}", line_number,
            ))
            continue
        field_names = ("frame_id", "ped_id", "x", "y")
        values_list: list[float] = []
        invalid_column: int | None = None
        for index, field in enumerate(fields):
            try:
                values_list.append(float(field))
            except ValueError:
                invalid_column = index
                break
        if invalid_column is not None:
            issues.append(ETHValidationIssue(
                path, "non_numeric_value",
                f"non-numeric value in column {invalid_column + 1} "
                f"({field_names[invalid_column]})",
                line_number,
            ))
            continue
        values = tuple(values_list)

        non_finite_identifier_count = sum(
            not math.isfinite(value) for value in values[:2]
        )
        if non_finite_identifier_count:
            issues.append(ETHValidationIssue(
                path, "non_finite_identifier",
                "frame_id and ped_id must be finite", line_number,
                non_finite_identifier_count,
            ))
            continue
        non_finite_coordinate_count = sum(
            not math.isfinite(value) for value in values[2:]
        )
        if non_finite_coordinate_count:
            issues.append(ETHValidationIssue(
                path, "non_finite_coordinate",
                "coordinate values must be finite", line_number,
                non_finite_coordinate_count,
            ))
            continue

        if not values[0].is_integer():
            issues.append(ETHValidationIssue(
                path, "invalid_frame_id",
                "frame_id must be an integer-valued field", line_number,
            ))
            continue
        if not values[1].is_integer():
            issues.append(ETHValidationIssue(
                path, "invalid_pedestrian_id",
                "ped_id must be an integer-valued field", line_number,
            ))
            continue
        frame = int(values[0])
        pedestrian_id = int(values[1])
        if frame < 0:
            issues.append(ETHValidationIssue(
                path, "invalid_frame_id", "frame_id must be non-negative", line_number,
            ))
            continue
        if pedestrian_id < 0:
            issues.append(ETHValidationIssue(
                path, "invalid_pedestrian_id", "ped_id must be non-negative", line_number,
            ))
            continue
        identity = (frame, pedestrian_id)
        observation = ETHObservation(frame, pedestrian_id, values[2], values[3])
        if identity in seen:
            previous = seen[identity]
            if (previous.x, previous.y) == (observation.x, observation.y):
                code = "duplicate_observation"
                message = f"duplicate frame/pedestrian pair {identity}"
            else:
                code = "conflicting_observation"
                message = f"conflicting coordinates for duplicate frame/pedestrian pair {identity}"
            issues.append(ETHValidationIssue(path, code, message, line_number))
            continue
        previous_frame = previous_frame_by_pedestrian.get(pedestrian_id)
        if previous_frame is not None and frame < previous_frame:
            issues.append(ETHValidationIssue(
                path, "out_of_order_observation",
                f"frame_id {frame} precedes prior frame_id {previous_frame} "
                f"for pedestrian {pedestrian_id}",
                line_number,
            ))
            continue
        seen[identity] = observation
        previous_frame_by_pedestrian[pedestrian_id] = frame
        observations.append(observation)

    if not lines or not any(line.strip() for line in lines):
        issues.append(ETHValidationIssue(path, "empty_file", "file contains no observations"))
    elif not observations and not issues:
        issues.append(ETHValidationIssue(path, "empty_file", "file contains no observations"))

    return ETHFileValidation(
        path, scene, split, tuple(observations), tuple(issues)
    )


def read_social_stgcnn_file(path: Path, scene: str, split: str) -> ETHSplit:
    """Read whitespace-separated `<frame_id> <ped_id> <x> <y>` rows.

    Coordinates are kept unchanged. The caller supplies scene and split labels explicitly so
    files from different scenes or partitions cannot be silently merged. Invalid files fail
    rather than returning a partial set of observations.
    """
    validation = validate_social_stgcnn_file(path, scene, split)
    if validation.issues:
        raise ETHFormatError("; ".join(issue.format() for issue in validation.issues))
    return ETHSplit(scene, split, Path(path), validation.observations)


def discover_social_stgcnn_files(dataset_root: Path) -> tuple[tuple[Path, str, str], ...]:
    """Discover known Social-STGCNN ETH dataset files without copying them.

    `dataset_root` is the directory containing `train/`, `val/`, and `test/`.
    """
    dataset_root = Path(dataset_root)
    discovered: list[tuple[Path, str, str]] = []
    for split in SOCIAL_STGCNN_SPLITS:
        split_dir = dataset_root / split
        if not split_dir.is_dir():
            continue
        for path in sorted(split_dir.glob("*.txt")):
            scene_stem = path.stem
            split_suffix = f"_{split}"
            if scene_stem.endswith(split_suffix):
                scene_stem = scene_stem[:-len(split_suffix)]
            scene = SOCIAL_STGCNN_SCENES.get(scene_stem)
            if scene is None:
                raise ETHFormatError(f"{path}: unknown Social-STGCNN scene filename")
            discovered.append((path, scene, split))
    return tuple(discovered)


def load_social_stgcnn_splits(dataset_root: Path) -> tuple[ETHSplit, ...]:
    """Load each discovered file independently, preserving its scene/split boundary."""
    return tuple(
        read_social_stgcnn_file(path, scene, split)
        for path, scene, split in discover_social_stgcnn_files(dataset_root)
    )


def build_eth_windows(
    dataset: ETHSplit,
    frame_step: int,
    history_len: int = 8,
    horizon_len: int = 12,
    stride: int = 1,
) -> tuple[ETHWindow, ...]:
    """Build windows only from existing rows at the requested exact frame cadence.

    No interpolation is performed. Gaps, pedestrian changes, files, and split boundaries are
    never crossed. Frame cadence is specified explicitly because these files do not document
    the mapping from frame IDs to seconds.
    """
    if frame_step <= 0 or history_len <= 0 or horizon_len <= 0 or stride <= 0:
        raise ValueError("frame_step, history_len, horizon_len, and stride must be positive")

    tracks: dict[int, list[ETHObservation]] = defaultdict(list)
    seen: set[tuple[int, int]] = set()
    for observation in dataset.observations:
        if (observation.frame < 0 or observation.pedestrian_id < 0
                or not math.isfinite(observation.x) or not math.isfinite(observation.y)):
            raise ETHFormatError(
                f"{dataset.source_file}: invalid observation "
                f"(frame={observation.frame}, pedestrian_id={observation.pedestrian_id})"
            )
        identity = (observation.frame, observation.pedestrian_id)
        if identity in seen:
            raise ETHFormatError(
                f"{dataset.source_file}: duplicate frame/pedestrian pair {identity}"
            )
        seen.add(identity)
        tracks[observation.pedestrian_id].append(observation)

    window_size = history_len + horizon_len
    windows: list[ETHWindow] = []
    for pedestrian_id in sorted(tracks):
        track = sorted(tracks[pedestrian_id], key=lambda observation: observation.frame)
        runs: list[list[ETHObservation]] = []
        run: list[ETHObservation] = []
        for observation in track:
            if run and observation.frame - run[-1].frame != frame_step:
                runs.append(run)
                run = []
            run.append(observation)
        if run:
            runs.append(run)

        for continuous in runs:
            for start in range(0, len(continuous) - window_size + 1, stride):
                samples = tuple(continuous[start:start + window_size])
                windows.append(ETHWindow(
                    scene=dataset.scene,
                    split=dataset.split,
                    pedestrian_id=pedestrian_id,
                    start_frame=samples[0].frame,
                    observed=samples[:history_len],
                    future=samples[history_len:],
                ))
    return tuple(windows)


def count_split_windows(
    splits: tuple[ETHSplit, ...],
    frame_step: int,
    history_len: int = 8,
    horizon_len: int = 12,
    stride: int = 1,
) -> dict[tuple[str, str], int]:
    """Count windows per `(split, scene)` without combining source files."""
    counts: dict[tuple[str, str], int] = {}
    for dataset in splits:
        key = (dataset.split, dataset.scene)
        if key in counts:
            raise ETHFormatError(
                f"duplicate split file for scene {dataset.scene!r}, split {dataset.split!r}"
            )
        counts[key] = len(build_eth_windows(
            dataset, frame_step, history_len, horizon_len, stride
        ))
    return counts
