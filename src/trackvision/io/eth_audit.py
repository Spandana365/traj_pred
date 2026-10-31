"""Deterministic validation and audit reports for Social-STGCNN ETH split tables."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from trackvision.io.eth import (
    ETHObservation,
    ETHSplit,
    ETHValidationIssue,
    SOCIAL_STGCNN_SCENES,
    SOCIAL_STGCNN_SPLITS,
    build_eth_windows,
    validate_social_stgcnn_file,
)


EXPECTED_SOCIAL_STGCNN_SCENES_BY_SPLIT: dict[str, tuple[str, ...]] = {
    "train": (
        "HOTEL", "ZARA1", "ZARA2", "ZARA3", "STUDENTS001", "UNIV", "UNI_EXAMPLES",
    ),
    "val": (
        "HOTEL", "ZARA1", "ZARA2", "ZARA3", "STUDENTS001", "UNIV", "UNI_EXAMPLES",
    ),
    "test": ("ETH",),
}

_SCENE_STEMS = {scene: stem for stem, scene in SOCIAL_STGCNN_SCENES.items()}


@dataclass(frozen=True)
class ETHDatasetAudit:
    """A stable, JSON-serializable snapshot of dataset structure and data quality."""

    dataset_root: str
    configuration: dict[str, int]
    split_scene_summaries: tuple[dict[str, object], ...]
    split_summaries: tuple[dict[str, object], ...]
    overlap_summaries: tuple[dict[str, object], ...]
    issues: tuple[dict[str, object], ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "dataset_root": self.dataset_root,
            "configuration": self.configuration,
            "split_scene_summaries": list(self.split_scene_summaries),
            "split_summaries": list(self.split_summaries),
            "overlap_summaries": list(self.overlap_summaries),
            "issues": list(self.issues),
        }

    def to_json(self) -> str:
        return json.dumps(self.as_dict(), indent=2, sort_keys=True, ensure_ascii=True) + "\n"

    def summary_text(self) -> str:
        present = sum(int(item["files_present"]) for item in self.split_summaries)
        expected = sum(int(item["files_expected"]) for item in self.split_summaries)
        observations = sum(int(item["observations"]) for item in self.split_summaries)
        issue_count = len(self.issues)
        lines = [
            f"ETH dataset audit: {self.dataset_root}",
            f"Files: {present}/{expected} expected; observations: {observations}; "
            f"validation/structure issues: {issue_count}",
        ]
        for item in self.split_summaries:
            windows = item["candidate_window_count"]
            window_text = "incomplete" if windows is None else str(windows)
            lines.append(
                f"  {item['split']}: {item['files_present']}/{item['files_expected']} files, "
                f"{item['observations']} observations, {window_text} candidate windows"
            )
        return "\n".join(lines)


def _relative_path(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _issue_dict(
    issue: ETHValidationIssue, root: Path
) -> dict[str, object]:
    result: dict[str, object] = {
        "path": _relative_path(issue.source_file, root),
        "code": issue.code,
        "message": issue.message,
    }
    if issue.line_number is not None:
        result["line_number"] = issue.line_number
    if issue.value_count is not None:
        result["value_count"] = issue.value_count
    return result


def _structure_issues(
    root: Path,
    expected: Mapping[str, Sequence[str]],
) -> list[dict[str, object]]:
    issues: list[dict[str, object]] = []
    if not root.is_dir():
        return [{
            "path": ".",
            "code": "missing_dataset_root",
            "message": "dataset root does not exist or is not a directory",
        }]

    expected_split_dirs = set(expected)
    for entry in sorted(root.iterdir(), key=lambda item: item.name):
        if entry.is_dir() and entry.name in expected_split_dirs:
            continue
        code = "unexpected_directory" if entry.is_dir() else "unexpected_file"
        issues.append({
            "path": _relative_path(entry, root),
            "code": code,
            "message": f"unexpected dataset-root entry: {entry.name}",
        })

    for split, scenes in expected.items():
        split_dir = root / split
        if not split_dir.is_dir():
            issues.append({
                "path": split,
                "code": "missing_split_directory",
                "message": f"required split directory {split!r} is missing",
            })
            continue
        expected_names = {
            f"{_SCENE_STEMS[scene]}_{split}.txt"
            if split != "test" else f"{_SCENE_STEMS[scene]}.txt"
            for scene in scenes
        }
        for entry in sorted(split_dir.iterdir(), key=lambda item: item.name):
            if entry.is_file() and entry.name in expected_names:
                continue
            code = "unexpected_directory" if entry.is_dir() else "unexpected_file"
            issues.append({
                "path": _relative_path(entry, root),
                "code": code,
                "message": f"unexpected entry in split {split!r}: {entry.name}",
            })
        for name in sorted(expected_names):
            if not (split_dir / name).is_file():
                relative = (Path(split) / name).as_posix()
                issues.append({
                    "path": relative,
                    "code": "missing_expected_file",
                    "message": f"required file for split {split!r} is missing",
                })
    return issues


def _file_statistics(
    observations: tuple[ETHObservation, ...],
) -> tuple[dict[str, int] | None, dict[str, float] | None, dict[str, int]]:
    if not observations:
        return None, None, {}
    frames = [observation.frame for observation in observations]
    xs = [observation.x for observation in observations]
    ys = [observation.y for observation in observations]
    tracks: dict[int, list[int]] = defaultdict(list)
    for observation in observations:
        tracks[observation.pedestrian_id].append(observation.frame)
    gap_values: list[int] = []
    for frames_for_track in tracks.values():
        ordered_frames = sorted(frames_for_track)
        gap_values.extend(
            later - earlier for earlier, later in zip(ordered_frames, ordered_frames[1:])
        )
    gaps = Counter(gap_values)
    return (
        {"min": min(frames), "max": max(frames)},
        {"x_min": min(xs), "x_max": max(xs), "y_min": min(ys), "y_max": max(ys)},
        {str(gap): count for gap, count in sorted(gaps.items())},
    )


def _overlap_summary(
    first_split: str,
    second_split: str,
    scene: str,
    first: tuple[ETHObservation, ...],
    second: tuple[ETHObservation, ...],
) -> dict[str, object]:
    first_by_id: dict[int, list[ETHObservation]] = defaultdict(list)
    second_by_id: dict[int, list[ETHObservation]] = defaultdict(list)
    for observation in first:
        first_by_id[observation.pedestrian_id].append(observation)
    for observation in second:
        second_by_id[observation.pedestrian_id].append(observation)
    shared_ids = sorted(first_by_id.keys() & second_by_id.keys())

    first_keys = {
        (observation.pedestrian_id, observation.frame) for observation in first
    }
    second_keys = {
        (observation.pedestrian_id, observation.frame) for observation in second
    }
    first_rows = {
        (observation.frame, observation.pedestrian_id, observation.x, observation.y)
        for observation in first
    }
    second_rows = {
        (observation.frame, observation.pedestrian_id, observation.x, observation.y)
        for observation in second
    }
    temporal_deltas = [
        min(item.frame for item in second_by_id[pedestrian_id])
        - max(item.frame for item in first_by_id[pedestrian_id])
        for pedestrian_id in shared_ids
    ]
    nearest_frame_distances = [
        min(
            abs(left.frame - right.frame)
            for left in first_by_id[pedestrian_id]
            for right in second_by_id[pedestrian_id]
        )
        for pedestrian_id in shared_ids
    ]
    return {
        "scene": scene,
        "first_split": first_split,
        "second_split": second_split,
        "identity_scope": "within_scene_only",
        "shared_pedestrian_ids": len(shared_ids),
        "shared_pedestrian_id_values": shared_ids,
        "shared_frame_pedestrian_rows": len(first_keys & second_keys),
        "identical_full_observation_rows": len(first_rows & second_rows),
        "minimum_nearest_frame_distance_for_shared_ids": (
            min(nearest_frame_distances) if nearest_frame_distances else None
        ),
        "validation_start_minus_train_end_frame": (
            {
                "min": min(temporal_deltas),
                "max": max(temporal_deltas),
                "validation_after_train_for_all_shared_ids": all(
                    delta > 0 for delta in temporal_deltas
                ),
            }
            if first_split == "train" and second_split == "val" and temporal_deltas
            else None
        ),
    }


def audit_social_stgcnn_dataset(
    dataset_root: Path,
    frame_step: int,
    history_len: int = 8,
    horizon_len: int = 12,
    stride: int = 1,
    expected_scenes_by_split: Mapping[str, Sequence[str]] = (
        EXPECTED_SOCIAL_STGCNN_SCENES_BY_SPLIT
    ),
) -> ETHDatasetAudit:
    """Audit the expected Social-STGCNN ETH layout without changing any source data.

    ``frame_step`` is a frame-ID difference only; this function makes no conversion to time
    units and no inference about the physical coordinate system.
    """
    if frame_step <= 0 or history_len <= 0 or horizon_len <= 0 or stride <= 0:
        raise ValueError("frame_step, history_len, horizon_len, and stride must be positive")
    if set(expected_scenes_by_split) != set(SOCIAL_STGCNN_SPLITS):
        raise ValueError(
            f"expected_scenes_by_split must define exactly {SOCIAL_STGCNN_SPLITS}"
        )
    for split, scenes in expected_scenes_by_split.items():
        unknown = set(scenes) - set(SOCIAL_STGCNN_SCENES.values())
        if unknown:
            raise ValueError(f"unknown scene(s) in {split}: {sorted(unknown)}")

    root = Path(dataset_root)
    expected = {
        split: tuple(sorted(set(scenes)))
        for split, scenes in expected_scenes_by_split.items()
    }
    issues = _structure_issues(root, expected)

    summaries: list[dict[str, object]] = []
    valid_observations: dict[tuple[str, str], tuple[ETHObservation, ...]] = {}
    expected_pairs = {
        (split, scene)
        for split, scenes in expected.items()
        for scene in scenes
    }
    for split, scene in sorted(expected_pairs):
        stem = _SCENE_STEMS[scene]
        filename = f"{stem}_{split}.txt" if split != "test" else f"{stem}.txt"
        path = root / split / filename
        if not path.is_file():
            summaries.append({
                "split": split,
                "scene": scene,
                "path": _relative_path(path, root),
                "file_present": False,
                "observation_count": 0,
                "unique_pedestrian_count": 0,
                "frame_id_range": None,
                "frame_gap_distribution": {},
                "coordinate_ranges": None,
                "non_finite_value_count": 0,
                "rejected_row_count": 0,
                "invalid_or_rejected_rows": [],
                "candidate_window_count": None,
            })
            continue

        validation = validate_social_stgcnn_file(path, scene, split)
        issues.extend(_issue_dict(issue, root) for issue in validation.issues)
        if validation.is_valid:
            valid_observations[(split, scene)] = validation.observations
        frame_range, coordinate_ranges, gap_distribution = _file_statistics(
            validation.observations
        )
        non_finite_count = sum(
            issue.value_count or 0
            for issue in validation.issues
            if issue.code in {"non_finite_coordinate", "non_finite_identifier"}
        )
        candidate_windows = None
        if validation.is_valid:
            candidate_windows = len(build_eth_windows(
                ETHSplit(scene, split, path, validation.observations),
                frame_step,
                history_len,
                horizon_len,
                stride,
            ))
        summaries.append({
            "split": split,
            "scene": scene,
            "path": _relative_path(path, root),
            "file_present": True,
            "observation_count": len(validation.observations),
            "unique_pedestrian_count": len({
                observation.pedestrian_id for observation in validation.observations
            }),
            "frame_id_range": frame_range,
            "frame_gap_distribution": gap_distribution,
            "coordinate_ranges": coordinate_ranges,
            "non_finite_value_count": non_finite_count,
            "rejected_row_count": len({
                issue.line_number for issue in validation.issues
                if issue.line_number is not None
            }),
            "invalid_or_rejected_rows": [
                _issue_dict(issue, root) for issue in validation.issues
            ],
            "candidate_window_count": candidate_windows,
        })

    split_summaries: list[dict[str, object]] = []
    for split in SOCIAL_STGCNN_SPLITS:
        files = [item for item in summaries if item["split"] == split]
        expected_count = len(expected[split])
        present_count = sum(bool(item["file_present"]) for item in files)
        observation_count = sum(int(item["observation_count"]) for item in files)
        available_windows = sum(
            int(item["candidate_window_count"])
            for item in files if item["candidate_window_count"] is not None
        )
        complete = (
            present_count == expected_count
            and all(item["candidate_window_count"] is not None for item in files)
        )
        split_summaries.append({
            "split": split,
            "files_expected": expected_count,
            "files_present": present_count,
            "observations": observation_count,
            "candidate_window_count": available_windows if complete else None,
            "candidate_window_count_from_valid_files": available_windows,
        })

    overlap_summaries: list[dict[str, object]] = []
    ordered_splits = list(SOCIAL_STGCNN_SPLITS)
    for first_index, first_split in enumerate(ordered_splits):
        for second_split in ordered_splits[first_index + 1:]:
            for scene in sorted(
                {scene for split, scene in valid_observations if split == first_split}
                & {scene for split, scene in valid_observations if split == second_split}
            ):
                overlap_summaries.append(_overlap_summary(
                    first_split,
                    second_split,
                    scene,
                    valid_observations[(first_split, scene)],
                    valid_observations[(second_split, scene)],
                ))

    issues.sort(key=lambda issue: (
        str(issue.get("path", "")),
        str(issue.get("code", "")),
        int(issue.get("line_number", 0)),
        str(issue.get("message", "")),
    ))
    return ETHDatasetAudit(
        dataset_root=root.as_posix(),
        configuration={
            "frame_step_in_frame_id_units": frame_step,
            "history_len": history_len,
            "horizon_len": horizon_len,
            "stride": stride,
        },
        split_scene_summaries=tuple(summaries),
        split_summaries=tuple(split_summaries),
        overlap_summaries=tuple(overlap_summaries),
        issues=tuple(issues),
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("data/eth/Social-STGCNN/datasets/eth"),
    )
    parser.add_argument("--frame-step", type=int, default=10)
    parser.add_argument("--history-len", type=int, default=8)
    parser.add_argument("--horizon-len", type=int, default=12)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/audits/eth/dataset-audit.json"),
        help="JSON report destination (defaults to the ignored runs/ directory)",
    )
    args = parser.parse_args(argv)
    report = audit_social_stgcnn_dataset(
        args.dataset_root,
        frame_step=args.frame_step,
        history_len=args.history_len,
        horizon_len=args.horizon_len,
        stride=args.stride,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report.to_json(), encoding="utf-8")
    print(report.summary_text())
    print(f"JSON report: {args.output}")
    return 1 if report.issues else 0


if __name__ == "__main__":
    raise SystemExit(main())
