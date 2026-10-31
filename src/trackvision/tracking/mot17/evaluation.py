"""Evaluate saved tracker files with the official TrackEval MOTChallenge code."""

from __future__ import annotations

import contextlib
import csv
import json
import shutil
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from trackvision.tracking.mot17.dataset import SequenceInfo, file_sha256


TRACKER_NAME = "botsort"
CLASS_NAME = "pedestrian"
COMBINED_KEY = "COMBINED_SEQ"
METRIC_COLUMNS = (
    "HOTA",
    "DetA",
    "AssA",
    "LocA",
    "MOTA",
    "MOTP",
    "IDF1",
    "IDP",
    "IDR",
    "IDSW",
    "CLR_TP",
    "CLR_FN",
    "CLR_FP",
    "Frag",
    "MT",
    "PT",
    "ML",
    "Dets",
    "GT_Dets",
    "IDs",
    "GT_IDs",
)


def _restore_removed_numpy_aliases() -> None:
    """Provide aliases TrackEval needs; NumPy 2 removed them.

    TrackEval's MOTChallenge code uses ``np.float``, ``np.int`` and ``np.bool``
    only as dtypes, so the builtins are exact replacements. The installed
    package itself is left unmodified.
    """
    for name, builtin in (("float", float), ("int", int), ("bool", bool)):
        if name not in np.__dict__:
            setattr(np, name, builtin)


def stage_ground_truth(sequences: Sequence[SequenceInfo], gt_root: Path) -> None:
    """Copy gt.txt and seqinfo.ini under plain sequence names, verifying each copy."""
    for sequence in sequences:
        target = gt_root / sequence.name
        (target / "gt").mkdir(parents=True, exist_ok=True)
        for source, destination in (
            (sequence.gt_path, target / "gt" / "gt.txt"),
            (sequence.seqinfo_path, target / "seqinfo.ini"),
        ):
            shutil.copyfile(source, destination)
            if file_sha256(source) != file_sha256(destination):
                raise OSError(f"Ground-truth copy does not match its source: {source}")


def _scalar(value: Any) -> float:
    """HOTA fields are arrays over IoU thresholds; report the mean as TrackEval does."""
    array = np.asarray(value, dtype=np.float64)
    return float(array.mean()) if array.ndim else float(array)


def _collect(result: dict[str, Any]) -> dict[str, float]:
    flattened: dict[str, float] = {}
    for metric_group in ("HOTA", "CLEAR", "Identity", "Count"):
        for key, value in result[CLASS_NAME][metric_group].items():
            if key in METRIC_COLUMNS:
                flattened[key] = _scalar(value)
    return flattened


def run_trackeval(
    sequences: Sequence[SequenceInfo],
    gt_root: Path,
    trackers_root: Path,
    output_dir: Path,
    tracker_name: str = TRACKER_NAME,
) -> dict[str, dict[str, float]]:
    """Run TrackEval and return metrics by sequence plus the combined row."""
    _restore_removed_numpy_aliases()
    import trackeval

    evaluator_config = trackeval.Evaluator.get_default_eval_config()
    evaluator_config.update(
        {
            "USE_PARALLEL": False,
            "PRINT_CONFIG": False,
            "TIME_PROGRESS": False,
            "DISPLAY_LESS_PROGRESS": True,
            "LOG_ON_ERROR": str(output_dir / "trackeval_errors.log"),
            "BREAK_ON_ERROR": True,
            "RETURN_ON_ERROR": False,
        }
    )
    dataset_config = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
    dataset_config.update(
        {
            "GT_FOLDER": str(gt_root),
            "TRACKERS_FOLDER": str(trackers_root),
            "OUTPUT_FOLDER": str(output_dir),
            "TRACKERS_TO_EVAL": [tracker_name],
            "CLASSES_TO_EVAL": [CLASS_NAME],
            "BENCHMARK": "MOT17",
            "SPLIT_TO_EVAL": "train",
            "SKIP_SPLIT_FOL": True,
            "SEQ_INFO": {sequence.name: sequence.length for sequence in sequences},
            "DO_PREPROC": True,
            "TRACKER_SUB_FOLDER": "data",
            "OUTPUT_SUB_FOLDER": "",
            "PRINT_CONFIG": False,
        }
    )
    metrics_config = {
        "METRICS": ["HOTA", "CLEAR", "Identity"],
        "THRESHOLD": 0.5,
        "PRINT_CONFIG": False,
    }
    # The evaluator adds its Count metric itself.
    metrics = [
        trackeval.metrics.HOTA(metrics_config),
        trackeval.metrics.CLEAR(metrics_config),
        trackeval.metrics.Identity(metrics_config),
    ]
    dataset = trackeval.datasets.MotChallenge2DBox(dataset_config)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "evaluation_log.txt").open("w", encoding="utf-8") as log:
        with contextlib.redirect_stdout(log):
            results, messages = trackeval.Evaluator(evaluator_config).evaluate(
                [dataset], metrics
            )
    status = messages["MotChallenge2DBox"][tracker_name]
    if status != "Success":
        raise RuntimeError(f"TrackEval failed: {status}")
    by_sequence = results["MotChallenge2DBox"][tracker_name]
    return {name: _collect(result) for name, result in by_sequence.items()}


def write_metrics(
    metrics: dict[str, dict[str, float]],
    output_dir: Path,
    experiment: str = "BoT-SORT_no_GMC",
) -> None:
    """Save the sequence rows and the TrackEval combined row."""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(
        json.dumps(
            {
                "experiment": experiment,
                "source": "official TrackEval, MotChallenge2DBox",
                "split": "MOT17 training ground truth",
                "combined_row": (
                    f"{COMBINED_KEY} pools the underlying counts of all sequences; "
                    "it is not an average of the sequence percentages."
                ),
                "hota_family": "mean over IoU thresholds 0.05 to 0.95",
                "clear_identity_iou_threshold": 0.5,
                "metrics": metrics,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    with (output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("sequence", *METRIC_COLUMNS))
        for name, row in metrics.items():
            writer.writerow((name, *(row.get(column) for column in METRIC_COLUMNS)))
