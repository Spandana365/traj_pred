"""Discover and validate MOT17 training sequences without modifying the dataset."""

from __future__ import annotations

import configparser
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np


DEFAULT_DATASET_ROOT = Path("data/MOT17")
DEFAULT_OUTPUT_ROOT = Path("runs/mot17_tracking_analysis")
OUTPUT_DIR_NAME = "mot17_tracking_analysis"
GMC_OUTPUT_DIR_NAME = "gmc_fixed"
# DPM, FRCNN and SDP share images and ground truth; only det/det.txt differs.
DETECTOR_VARIANTS = ("DPM", "FRCNN", "SDP")
CANONICAL_VARIANT = "FRCNN"
TRAIN_SEQUENCES = (
    "MOT17-02",
    "MOT17-04",
    "MOT17-05",
    "MOT17-09",
    "MOT17-10",
    "MOT17-11",
    "MOT17-13",
)
FORBIDDEN_PATH_SEGMENTS = frozenset(
    {"eth", "ucy", "error_injection", "prediction", "train_eth"}
)


class MOT17PathError(ValueError):
    """A path falls outside the permitted MOT17 data or output locations."""


class MOT17DatasetError(ValueError):
    """The MOT17 dataset does not match its expected structure."""


@dataclass(frozen=True)
class SequenceInfo:
    """One underlying MOT17 video and its canonical representation."""

    name: str
    variant: str
    directory: Path
    length: int
    frame_rate: int
    width: int
    height: int
    image_extension: str

    @property
    def image_dir(self) -> Path:
        return self.directory / "img1"

    @property
    def gt_path(self) -> Path:
        return self.directory / "gt" / "gt.txt"

    @property
    def seqinfo_path(self) -> Path:
        return self.directory / "seqinfo.ini"

    def frame_path(self, frame: int) -> Path:
        return self.image_dir / f"{frame:06d}{self.image_extension}"


def _forbidden_segments(path: Path) -> set[str]:
    return {part.lower() for part in path.parts} & FORBIDDEN_PATH_SEGMENTS


def ensure_dataset_root(path: Path) -> Path:
    """Return the resolved root, rejecting anything other than a MOT17 directory."""
    resolved = Path(path).expanduser().resolve()
    if resolved.name != "MOT17" or _forbidden_segments(resolved):
        raise MOT17PathError(f"Not a permitted MOT17 dataset root: {resolved}")
    return resolved


def ensure_output_root(path: Path) -> Path:
    """Return the resolved output root, rejecting anything outside the MOT17 run."""
    resolved = Path(path).expanduser().resolve()
    if resolved.name != OUTPUT_DIR_NAME or _forbidden_segments(resolved):
        raise MOT17PathError(f"Not a permitted MOT17 output root: {resolved}")
    return resolved


def ensure_gmc_output_root(path: Path) -> Path:
    """Return the resolved GMC root, which must sit inside the MOT17 run."""
    resolved = Path(path).expanduser().resolve()
    if (
        resolved.name != GMC_OUTPUT_DIR_NAME
        or resolved.parent.name != OUTPUT_DIR_NAME
        or _forbidden_segments(resolved)
    ):
        raise MOT17PathError(f"Not a permitted GMC output root: {resolved}")
    return resolved


def _read_seqinfo(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise MOT17DatasetError(f"Missing seqinfo.ini: {path}")
    parser = configparser.ConfigParser()
    parser.read(path, encoding="utf-8")
    if "Sequence" not in parser:
        raise MOT17DatasetError(f"seqinfo.ini has no [Sequence] section: {path}")
    return dict(parser["Sequence"])


def load_sequence(dataset_root: Path, name: str) -> SequenceInfo:
    """Load one sequence's canonical variant from the training split."""
    directory = dataset_root / "train" / f"{name}-{CANONICAL_VARIANT}"
    seqinfo = _read_seqinfo(directory / "seqinfo.ini")
    return SequenceInfo(
        name=name,
        variant=CANONICAL_VARIANT,
        directory=directory,
        length=int(seqinfo["seqlength"]),
        frame_rate=int(seqinfo["framerate"]),
        width=int(seqinfo["imwidth"]),
        height=int(seqinfo["imheight"]),
        image_extension=seqinfo["imext"],
    )


def discover_sequence_names(dataset_root: Path) -> list[str]:
    """Group the training directories into underlying sequences."""
    train = dataset_root / "train"
    if not train.is_dir():
        raise MOT17DatasetError(f"Missing MOT17 training directory: {train}")
    names = set()
    for path in train.iterdir():
        if not path.is_dir():
            continue
        name, _, variant = path.name.rpartition("-")
        if variant not in DETECTOR_VARIANTS or not name.startswith("MOT17-"):
            raise MOT17DatasetError(f"Unexpected training directory: {path.name}")
        names.add(name)
    return sorted(names)


def discover_sequences(
    dataset_root: Path = DEFAULT_DATASET_ROOT,
    names: Sequence[str] | None = None,
) -> list[SequenceInfo]:
    """Return one canonical sequence per underlying MOT17 training video."""
    root = ensure_dataset_root(dataset_root)
    available = discover_sequence_names(root)
    selected = list(names) if names else available
    unknown = sorted(set(selected) - set(available))
    if unknown:
        raise MOT17DatasetError(f"Unknown MOT17 sequences: {unknown}")
    return [load_sequence(root, name) for name in selected]


def frame_numbers(sequence: SequenceInfo) -> list[int]:
    """Return frame numbers parsed from the image file names, in order."""
    pattern = f"*{sequence.image_extension}"
    return sorted(int(path.stem) for path in sequence.image_dir.glob(pattern))


def validate_frame_numbers(sequence: SequenceInfo) -> None:
    """Confirm the images are numbered 1..seqLength with no gaps."""
    frames = frame_numbers(sequence)
    if frames != list(range(1, sequence.length + 1)):
        raise MOT17DatasetError(
            f"{sequence.name}: expected frames 1..{sequence.length}, "
            f"found {len(frames)} images"
        )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_ground_truth(sequence: SequenceInfo) -> np.ndarray:
    """Load gt.txt: frame, id, left, top, width, height, conf, class, visibility."""
    rows = np.loadtxt(sequence.gt_path, delimiter=",", dtype=np.float64, ndmin=2)
    if rows.shape[1] != 9:
        raise MOT17DatasetError(
            f"{sequence.name}: expected 9 gt columns, found {rows.shape[1]}"
        )
    frames = rows[:, 0]
    if frames.min() < 1 or frames.max() > sequence.length:
        raise MOT17DatasetError(f"{sequence.name}: gt frame outside 1..seqLength")
    return rows


def variant_agreement(dataset_root: Path, name: str) -> dict[str, Any]:
    """Compare each detector variant's metadata, ground truth and sample frames."""
    canonical = load_sequence(dataset_root, name)
    sample_frames = (1, (canonical.length + 1) // 2, canonical.length)
    reference: dict[str, Any] | None = None
    agreement = True
    for variant in DETECTOR_VARIANTS:
        directory = dataset_root / "train" / f"{name}-{variant}"
        seqinfo = _read_seqinfo(directory / "seqinfo.ini")
        digest = {
            "seq_length": seqinfo["seqlength"],
            "frame_rate": seqinfo["framerate"],
            "gt_sha256": file_sha256(directory / "gt" / "gt.txt"),
            "sample_frames_sha256": [
                file_sha256(directory / "img1" / f"{frame:06d}{canonical.image_extension}")
                for frame in sample_frames
            ],
        }
        if reference is None:
            reference = digest
        elif digest != reference:
            agreement = False
    assert reference is not None
    return {"variants_agree": agreement, **reference}


def build_inventory(dataset_root: Path = DEFAULT_DATASET_ROOT) -> dict[str, Any]:
    """Validate every training sequence and describe it for the run manifest."""
    root = ensure_dataset_root(dataset_root)
    entries = []
    for sequence in discover_sequences(root):
        validate_frame_numbers(sequence)
        ground_truth = load_ground_truth(sequence)
        agreement = variant_agreement(root, sequence.name)
        if not agreement["variants_agree"]:
            raise MOT17DatasetError(f"{sequence.name}: detector variants disagree")
        evaluated = (ground_truth[:, 6] == 1) & (ground_truth[:, 7] == 1)
        entries.append(
            {
                "sequence": sequence.name,
                "canonical_directory": sequence.directory.name,
                "frames": sequence.length,
                "frame_rate": sequence.frame_rate,
                "width": sequence.width,
                "height": sequence.height,
                "gt_rows": int(len(ground_truth)),
                "evaluated_gt_boxes": int(evaluated.sum()),
                "evaluated_gt_identities": int(
                    len(np.unique(ground_truth[evaluated, 1]))
                ),
                **agreement,
            }
        )
    return {
        "dataset_root": str(root),
        "split": "train",
        "canonical_variant": CANONICAL_VARIANT,
        "variant_policy": (
            "DPM, FRCNN and SDP are one video each; their public detections are "
            "not used. The FRCNN directory is the canonical image and gt source."
        ),
        "sequence_count": len(entries),
        "total_frames": sum(entry["frames"] for entry in entries),
        "sequences": entries,
    }
