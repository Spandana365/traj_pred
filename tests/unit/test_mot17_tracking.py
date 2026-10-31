"""Tests for the MOT17 tracking stage, using synthetic fixtures only."""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

from trackvision.tracking.mot17.dataset import (
    MOT17DatasetError,
    MOT17PathError,
    TRAIN_SEQUENCES,
    build_inventory,
    discover_sequences,
    ensure_dataset_root,
    ensure_output_root,
    frame_numbers,
    load_ground_truth,
    validate_frame_numbers,
)
from trackvision.tracking.mot17.detection import _resume_point
from trackvision.tracking.mot17.error_profile import (
    assign_pairs,
    distribution,
    frame_gaps,
    ground_truth_continuity,
    identity_runs,
    iou_matrix,
    match_sequence,
    predicted_track_summary,
    remove_distractor_predictions,
    run_lengths,
    survival_curve,
    window_availability,
)
from trackvision.tracking.mot17.formats import (
    box_centers,
    box_foot_points,
    format_mot_row,
    load_mot_file,
    mot_ltwh_to_center_xywh,
    rows_by_frame,
    xyxy_to_mot_ltwh,
)

MOT17_SOURCE = Path(__file__).parents[2] / "src" / "trackvision" / "tracking" / "mot17"
SEQINFO = (
    "[Sequence]\nname={name}\nimDir=img1\nframeRate={fps}\nseqLength={length}\n"
    "imWidth=64\nimHeight=48\nimExt=.jpg\n"
)


def write_sequence(root: Path, name: str, length: int = 3, fps: int = 30) -> None:
    for variant in ("DPM", "FRCNN", "SDP"):
        directory = root / "train" / f"{name}-{variant}"
        (directory / "img1").mkdir(parents=True)
        (directory / "gt").mkdir()
        (directory / "seqinfo.ini").write_text(
            SEQINFO.format(name=name, fps=fps, length=length), encoding="utf-8"
        )
        (directory / "gt" / "gt.txt").write_text(
            "1,1,10,10,5,10,1,1,1.0\n2,1,11,10,5,10,1,1,1.0\n3,2,20,20,5,10,0,7,1.0\n",
            encoding="utf-8",
        )
        for frame in range(1, length + 1):
            (directory / "img1" / f"{frame:06d}.jpg").write_bytes(b"frame" + bytes([frame]))


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    root = tmp_path / "MOT17"
    write_sequence(root, "MOT17-02")
    write_sequence(root, "MOT17-05", length=4, fps=14)
    return root


def gt_row(frame, identity, left, top=10.0, width=10.0, height=20.0, conf=1, cls=1):
    return [frame, identity, left, top, width, height, conf, cls, 1.0]


def pred_row(frame, identity, left, top=10.0, width=10.0, height=20.0, score=0.9):
    return [frame, identity, left, top, width, height, score, -1, -1, -1]


# Dataset discovery and frame numbering


def test_variants_collapse_to_one_sequence(dataset: Path) -> None:
    sequences = discover_sequences(dataset)
    assert [sequence.name for sequence in sequences] == ["MOT17-02", "MOT17-05"]
    assert {sequence.variant for sequence in sequences} == {"FRCNN"}
    assert sequences[1].frame_rate == 14


def test_inventory_reports_variant_agreement_and_evaluated_rows(dataset: Path) -> None:
    inventory = build_inventory(dataset)
    assert inventory["sequence_count"] == 2
    assert inventory["total_frames"] == 7
    first = inventory["sequences"][0]
    assert first["variants_agree"] is True
    assert first["evaluated_gt_boxes"] == 2
    assert first["evaluated_gt_identities"] == 1


def test_inventory_rejects_disagreeing_variant_ground_truth(dataset: Path) -> None:
    other = dataset / "train" / "MOT17-02-SDP" / "gt" / "gt.txt"
    other.write_text("1,9,0,0,1,1,1,1,1.0\n", encoding="utf-8")
    with pytest.raises(MOT17DatasetError, match="variants disagree"):
        build_inventory(dataset)


def test_unknown_sequence_and_unexpected_directory_are_rejected(dataset: Path) -> None:
    with pytest.raises(MOT17DatasetError, match="Unknown"):
        discover_sequences(dataset, ["MOT17-99"])
    (dataset / "train" / "MOT17-02-OTHER").mkdir()
    with pytest.raises(MOT17DatasetError, match="Unexpected"):
        discover_sequences(dataset)


def test_frame_numbers_come_from_file_names_and_start_at_one(dataset: Path) -> None:
    sequence = discover_sequences(dataset, ["MOT17-02"])[0]
    assert frame_numbers(sequence) == [1, 2, 3]
    assert sequence.frame_path(2).name == "000002.jpg"
    validate_frame_numbers(sequence)


def test_missing_frame_fails_validation(dataset: Path) -> None:
    sequence = discover_sequences(dataset, ["MOT17-02"])[0]
    sequence.frame_path(2).unlink()
    with pytest.raises(MOT17DatasetError, match="expected frames"):
        validate_frame_numbers(sequence)


def test_ground_truth_rejects_frames_beyond_sequence(dataset: Path) -> None:
    sequence = discover_sequences(dataset, ["MOT17-02"])[0]
    sequence.gt_path.write_text("9,1,10,10,5,10,1,1,1.0\n", encoding="utf-8")
    with pytest.raises(MOT17DatasetError, match="outside"):
        load_ground_truth(sequence)


# Bounding-box conversion and output format


def test_xyxy_to_mot_uses_one_based_left_top() -> None:
    converted = xyxy_to_mot_ltwh(np.array([[10.0, 20.0, 40.0, 80.0]]))
    np.testing.assert_allclose(converted, [[11.0, 21.0, 30.0, 60.0]])


def test_mot_to_center_inverts_the_export_conversion() -> None:
    xyxy = np.array([[10.0, 20.0, 40.0, 80.0]])
    center = mot_ltwh_to_center_xywh(xyxy_to_mot_ltwh(xyxy))
    np.testing.assert_allclose(center, [[25.0, 50.0, 30.0, 60.0]])


def test_box_points() -> None:
    box = np.array([[10.0, 20.0, 4.0, 8.0]])
    np.testing.assert_allclose(box_centers(box), [[12.0, 24.0]])
    np.testing.assert_allclose(box_foot_points(box), [[12.0, 28.0]])


def test_empty_conversions_keep_their_shape() -> None:
    assert xyxy_to_mot_ltwh(np.empty((0, 4))).shape == (0, 4)
    assert mot_ltwh_to_center_xywh(np.empty((0, 4))).shape == (0, 4)


def test_mot_row_has_ten_motchallenge_fields(tmp_path: Path) -> None:
    line = format_mot_row(7, 3, np.array([1.5, 2.5, 3.0, 4.0]), 0.87654321)
    assert line == "7,3,1.50,2.50,3.00,4.00,0.87654,-1,-1,-1"
    path = tmp_path / "result.txt"
    path.write_text(line + "\n", encoding="utf-8")
    rows = load_mot_file(path)
    assert rows.shape == (1, 10)
    assert rows[0, 0] == 7 and rows[0, 1] == 3


def test_empty_mot_file_loads_as_no_rows(tmp_path: Path) -> None:
    path = tmp_path / "empty.txt"
    path.write_text("", encoding="utf-8")
    assert load_mot_file(path).shape == (0, 10)
    assert rows_by_frame(load_mot_file(path)) == {}


def test_detection_resume_discards_rows_after_the_last_saved_frame(tmp_path: Path) -> None:
    partial = tmp_path / "s.partial"
    progress = tmp_path / "s.progress.json"
    partial.write_text("1,-1,1,1,1,1,0.5,-1,-1,-1\n3,-1,1,1,1,1,0.5,-1,-1,-1\n", encoding="utf-8")
    progress.write_text('{"last_frame": 2, "seconds": 1.0}', encoding="utf-8")
    assert _resume_point(partial, progress) == 2
    assert partial.read_text(encoding="utf-8").count("\n") == 1


# Tracker behaviour against the installed Ultralytics implementation


def moving_detections(start: float = 100.0) -> dict[int, np.ndarray]:
    return {
        frame: np.array([pred_row(frame, -1, start + frame, 50.0, 30.0, 60.0, 0.9)])
        for frame in range(1, 7)
    }


def blank_frame(_frame: int) -> np.ndarray:
    return np.zeros((240, 320, 3), dtype=np.uint8)


def test_tracker_keeps_identity_and_box_position() -> None:
    pytest.importorskip("ultralytics")
    from trackvision.tracking.mot17.tracker import BotSortConfig, track_frames

    rows = track_frames(moving_detections(), 6, blank_frame, BotSortConfig())
    assert len(rows) >= 4
    assert set(rows[:, 1]) == {1.0}
    np.testing.assert_allclose(rows[-1, 2:6], [106.0, 50.0, 30.0, 60.0], atol=2.0)


def test_tracker_state_resets_between_sequences() -> None:
    pytest.importorskip("ultralytics")
    from trackvision.tracking.mot17.tracker import BotSortConfig, track_frames

    config = BotSortConfig()
    first = track_frames(moving_detections(), 6, blank_frame, config)
    second = track_frames(moving_detections(), 6, blank_frame, config)
    np.testing.assert_array_equal(first, second)


def test_tracker_handles_empty_frames() -> None:
    pytest.importorskip("ultralytics")
    from trackvision.tracking.mot17.tracker import (
        BotSortConfig,
        GMCRecorder,
        track_frames,
    )

    detections = moving_detections()
    del detections[4]
    # Blank frames have no features, so GMC correctly returns identity here.
    recorder = GMCRecorder(max_consecutive_identity=None)
    rows = track_frames(detections, 8, blank_frame, BotSortConfig(), recorder)
    assert 4 not in set(rows[:, 0])
    assert 8 not in set(rows[:, 0])
    assert set(rows[:, 1]) == {1.0}

    assert track_frames({}, 3, blank_frame, BotSortConfig()).shape == (0, 7)


def test_reid_is_disabled_by_default() -> None:
    from trackvision.tracking.mot17.tracker import BotSortConfig

    assert BotSortConfig().with_reid is False


# Error-profile calculations


def test_iou_values() -> None:
    first = np.array([[0.0, 0.0, 10.0, 10.0]])
    second = np.array([[5.0, 0.0, 10.0, 10.0], [50.0, 50.0, 1.0, 1.0]])
    np.testing.assert_allclose(iou_matrix(first, second), [[50 / 150, 0.0]])
    assert iou_matrix(np.empty((0, 4)), second).shape == (0, 2)


def test_assignment_maximises_iou_and_respects_threshold() -> None:
    iou = np.array([[0.9, 0.6], [0.8, 0.4]])
    assert sorted(assign_pairs(iou)) == [(0, 1), (1, 0)]
    assert assign_pairs(np.array([[0.49]])) == []


def test_distractor_matches_are_removed_but_others_stay() -> None:
    gt = np.array([gt_row(1, 1, 0.0, cls=7), gt_row(1, 2, 100.0)])
    predictions = np.array([pred_row(1, 1, 0.0), pred_row(1, 2, 100.0), pred_row(1, 3, 200.0)])
    kept = remove_distractor_predictions(gt, predictions)
    assert kept[:, 1].tolist() == [2.0, 3.0]


def test_sequence_counts_true_false_and_missed_boxes() -> None:
    gt = np.array(
        [gt_row(1, 1, 0.0), gt_row(1, 2, 100.0), gt_row(2, 1, 0.0), gt_row(2, 9, 300.0, conf=0)]
    )
    predictions = np.array([pred_row(1, 1, 0.0), pred_row(2, 1, 0.0), pred_row(2, 2, 200.0)])
    result = match_sequence(gt, predictions)
    assert result.true_positive_count == 2
    assert result.false_positive_count == 1
    assert result.false_negative_count == 1
    assert len(result.evaluated_gt) == 3


def test_continuing_identity_beats_a_higher_iou_alternative() -> None:
    gt = np.array([gt_row(1, 1, 0.0), gt_row(2, 1, 0.0)])
    predictions = np.array(
        [pred_row(1, 5, 0.0), pred_row(2, 5, 2.0), pred_row(2, 6, 0.0)]
    )
    result = match_sequence(gt, predictions)
    assert result.matches[:, 2].tolist() == [5.0, 5.0]


def test_identity_priority_does_not_survive_a_missed_frame() -> None:
    gt = np.array([gt_row(frame, 1, 0.0) for frame in (1, 2, 3)])
    predictions = np.array(
        [pred_row(1, 5, 0.0), pred_row(3, 5, 2.0), pred_row(3, 6, 0.0)]
    )
    result = match_sequence(gt, predictions)
    assert result.matches[:, 2].tolist() == [5.0, 6.0]


def test_localization_errors_are_in_pixels_and_box_heights() -> None:
    gt = np.array([gt_row(1, 1, 0.0, height=20.0)])
    predictions = np.array([pred_row(1, 1, 3.0, 14.0, 10.0, 20.0)])
    # Centre shift of (3, 4) gives a distance of 5 pixels.
    predictions[0, 3] = 14.0
    result = match_sequence(gt, predictions, threshold=0.1)
    assert result.matches[0, 4] == pytest.approx(5.0)
    assert result.matches[0, 5] == pytest.approx(0.25)


def test_run_gap_and_identity_helpers() -> None:
    assert run_lengths([True, True, False, True]) == [2, 1]
    assert run_lengths([]) == []
    assert frame_gaps([1, 2, 5, 9]) == [2, 3]
    assert frame_gaps([4]) == []
    assert identity_runs([3, 3, 0, 3, 4, 4]) == [2, 1, 2]


def test_continuity_counts_fragmentation_switches_and_outer_gaps() -> None:
    gt = np.array([gt_row(frame, 1, 0.0) for frame in range(1, 9)])
    predictions = np.array(
        [pred_row(2, 5, 0.0), pred_row(3, 5, 0.0), pred_row(5, 5, 0.0), pred_row(6, 6, 0.0)]
    )
    record = ground_truth_continuity(match_sequence(gt, predictions), 10.0)[0]
    assert record["tracked_frames"] == 4
    assert record["coverage"] == 0.5
    assert record["missing_runs"] == [1, 1, 2]
    # Gaps before the first and after the last match are not fragmentation.
    assert record["interior_gaps"] == [1]
    assert record["fragmentations"] == 1
    assert record["identity_switches"] == 1
    assert record["distinct_track_ids"] == 2
    assert record["longest_continuous_frames"] == 2


def test_predicted_track_summary_reports_gaps_and_purity() -> None:
    gt = np.array([gt_row(1, 1, 0.0), gt_row(4, 1, 0.0)])
    predictions = np.array([pred_row(1, 5, 0.0), pred_row(2, 5, 200.0), pred_row(4, 5, 0.0)])
    summary = predicted_track_summary(predictions, match_sequence(gt, predictions), 2.0)[0]
    assert summary["span_frames"] == 4
    assert summary["span_seconds"] == 2.0
    assert summary["gaps"] == [1]
    assert summary["missing_frames_inside_span"] == 1
    assert summary["dominant_gt_fraction"] == pytest.approx(2 / 3)


def test_window_availability_and_survival() -> None:
    continuity = [{"gt_frames": 5, "continuous_runs": [3, 2]}]
    # Possible 3-frame windows: 3. Only the 3-frame run provides one.
    result = window_availability(continuity, 3)
    assert result == {
        "possible_windows": 3,
        "available_windows": 1,
        "availability": pytest.approx(1 / 3),
    }
    assert window_availability(continuity, 9)["availability"] is None
    assert survival_curve([1, 2, 4], [2, 5]) == [
        {"horizon": 2, "survival": pytest.approx(2 / 3)},
        {"horizon": 5, "survival": 0.0},
    ]


def test_distribution_of_nothing_is_explicitly_empty() -> None:
    assert distribution([])["count"] == 0
    assert distribution([])["mean"] is None


# TrackEval integration


def test_trackeval_scores_a_perfect_tracker_as_perfect(tmp_path: Path, dataset: Path) -> None:
    pytest.importorskip("trackeval")
    from trackvision.tracking.mot17.evaluation import (
        COMBINED_KEY,
        run_trackeval,
        stage_ground_truth,
    )

    sequence = discover_sequences(dataset, ["MOT17-02"])[0]
    gt_root = tmp_path / "gt"
    stage_ground_truth([sequence], gt_root)
    tracker_data = tmp_path / "trackers" / "perfect" / "data"
    tracker_data.mkdir(parents=True)
    (tracker_data / "MOT17-02.txt").write_text(
        "1,1,10,10,5,10,1,-1,-1,-1\n2,1,11,10,5,10,1,-1,-1,-1\n", encoding="utf-8"
    )
    metrics = run_trackeval(
        [sequence], gt_root, tmp_path / "trackers", tmp_path / "out", "perfect"
    )
    assert metrics[COMBINED_KEY]["MOTA"] == 1.0
    assert metrics[COMBINED_KEY]["IDF1"] == 1.0
    assert metrics[COMBINED_KEY]["CLR_FP"] == 0


# Separation from ETH


def test_expected_training_sequences_are_the_seven_mot17_videos() -> None:
    assert len(TRAIN_SEQUENCES) == 7


@pytest.mark.parametrize(
    "path", ["data/eth", "data/raw/ucy/MOT17", "data/MOT20", "runs/prediction/MOT17"]
)
def test_dataset_root_rejects_non_mot17_paths(path: str) -> None:
    with pytest.raises(MOT17PathError):
        ensure_dataset_root(Path(path))


@pytest.mark.parametrize(
    "path",
    [
        "runs/error_injection",
        "runs/prediction/eth",
        "runs/prediction/mot17_tracking_analysis",
        "runs/mot17",
    ],
)
def test_output_root_rejects_eth_result_paths(path: str) -> None:
    with pytest.raises(MOT17PathError):
        ensure_output_root(Path(path))


def test_default_paths_are_accepted() -> None:
    assert ensure_dataset_root(Path("data/MOT17")).name == "MOT17"
    assert ensure_output_root(Path("runs/mot17_tracking_analysis")).parts[-2:] == (
        "runs",
        "mot17_tracking_analysis",
    )


def test_mot17_code_imports_nothing_from_eth_or_prediction_modules() -> None:
    for path in MOT17_SOURCE.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                modules = [node.module]
            elif isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            else:
                continue
            for module in modules:
                if module.startswith("trackvision"):
                    assert module.startswith("trackvision.tracking.mot17"), (path, module)
