"""Tests for BoT-SORT GMC, written to catch the silent failure in the first run."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

pytest.importorskip("ultralytics")

from ultralytics.trackers.utils.gmc import GMC

from trackvision.tracking.mot17.dataset import (
    MOT17PathError,
    discover_sequences,
    ensure_gmc_output_root,
)
from trackvision.tracking.mot17.error_profile import match_sequence
from trackvision.tracking.mot17.formats import format_mot_row
from trackvision.tracking.mot17.gmc_pipeline import check_pilot, run_tracking
from trackvision.tracking.mot17.gmc_verification import (
    anchored_residuals,
    photometric_errors,
    phase_shift,
    warp_center_motion,
)
from trackvision.tracking.mot17.pipeline import (
    _write_trajectories,
    reject_archived_overwrite,
)
from trackvision.tracking.mot17.stabilization import (
    load_warps,
    stabilize_mot_points,
    to_first_frame_transforms,
)
from trackvision.tracking.mot17.tracker import (
    BotSortConfig,
    DetectionFrame,
    GMCPipelineFailure,
    GMCRecorder,
    track_frames,
)

SEQINFO = (
    "[Sequence]\nname={name}\nimDir=img1\nframeRate=30\nseqLength={length}\n"
    "imWidth=320\nimHeight=240\nimExt=.jpg\n"
)
FRAME_SIZE = (240, 320)


def scene(seed: int = 7) -> np.ndarray:
    random = np.random.default_rng(seed)
    noise = random.random((500, 700)).astype(np.float32)
    texture = cv2.GaussianBlur(noise, (0, 0), 3.0)
    texture = (texture - texture.min()) / (texture.max() - texture.min()) * 255
    return cv2.cvtColor(texture.astype(np.uint8), cv2.COLOR_GRAY2BGR)


def panning_frames(step_x: int, step_y: int):
    """Simulate a camera moving by (step_x, step_y) pixels per frame."""
    base = scene()

    def read(frame: int) -> np.ndarray:
        left = 40 + step_x * frame
        top = 40 + step_y * frame
        return base[top : top + FRAME_SIZE[0], left : left + FRAME_SIZE[1]].copy()

    return read


def detection_row(frame: int, left: float, confidence: float = 0.9) -> list[float]:
    return [frame, -1, left, 60.0, 30.0, 60.0, confidence, -1, -1, -1]


def steady_detections(frames: int = 6) -> dict[int, np.ndarray]:
    return {
        frame: np.array([detection_row(frame, 100.0 + frame)])
        for frame in range(1, frames + 1)
    }


@pytest.fixture
def gmc_spy(monkeypatch: pytest.MonkeyPatch) -> list[tuple[np.ndarray, np.ndarray]]:
    calls: list[tuple[np.ndarray, np.ndarray]] = []
    original = GMC.apply

    def spy(self: GMC, raw_frame: np.ndarray, detections=None) -> np.ndarray:
        calls.append((raw_frame, np.array(detections)))
        return original(self, raw_frame, detections)

    monkeypatch.setattr(GMC, "apply", spy)
    return calls


# DetectionFrame.xyxy


def test_detection_frame_exposes_xyxy_matching_mot_box() -> None:
    frame = DetectionFrame.from_mot_rows(np.array([[1, -1, 11, 21, 30, 60, 0.9, -1, -1, -1.0]]))
    np.testing.assert_allclose(frame.xyxy, [[10.0, 20.0, 40.0, 80.0]])


def test_xyxy_has_the_shape_and_type_of_ultralytics_boxes() -> None:
    import torch
    from ultralytics.engine.results import Boxes

    corners = np.array([[10.0, 20.0, 40.0, 80.0], [1.5, 2.5, 9.5, 12.5]])
    data = np.column_stack((corners, (0.9, 0.8), (0.0, 0.0)))
    boxes = Boxes(torch.tensor(data), (240, 320)).cpu().numpy()
    frame = DetectionFrame(boxes.xywh, boxes.conf, boxes.cls)
    assert frame.xyxy.shape == boxes.xyxy.shape
    assert np.issubdtype(frame.xyxy.dtype, np.floating)
    np.testing.assert_allclose(frame.xyxy, boxes.xyxy)


def test_xyxy_of_empty_and_subset_frames() -> None:
    empty = DetectionFrame.from_mot_rows(None)
    assert empty.xyxy.shape == (0, 4)
    frame = DetectionFrame(
        np.array([[25.0, 50.0, 30.0, 60.0], [5.0, 5.0, 2.0, 2.0]]),
        np.array([0.9, 0.2]),
        np.zeros(2),
    )
    np.testing.assert_allclose(frame[frame.conf > 0.5].xyxy, [[10.0, 20.0, 40.0, 80.0]])
    assert frame[frame.conf > 2].xyxy.shape == (0, 4)


def test_xyxy_does_not_change_saved_detections() -> None:
    rows = np.array([[1, -1, 11, 21, 30, 60, 0.9, -1, -1, -1.0]])
    before = rows.copy()
    frame = DetectionFrame.from_mot_rows(rows)
    xywh_before = frame.xywh.copy()
    frame.xyxy[:] = -1.0
    np.testing.assert_array_equal(rows, before)
    np.testing.assert_array_equal(frame.xywh, xywh_before)


# GMC is really called


def test_gmc_receives_each_frame_and_high_confidence_boxes(gmc_spy) -> None:
    detections = {
        1: np.array([detection_row(1, 100.0), detection_row(1, 200.0, 0.2)]),
        2: np.array([detection_row(2, 101.0), detection_row(2, 201.0, 0.2)]),
    }
    supplied = []

    def read(frame: int) -> np.ndarray:
        supplied.append(panning_frames(3, 0)(frame))
        return supplied[-1]

    track_frames(detections, 2, read, BotSortConfig())
    assert len(gmc_spy) == 2
    for (frame, boxes), source in zip(gmc_spy, supplied):
        assert frame is source
        assert boxes.shape == (1, 4)
    # MOT left 101, top 60 becomes 0-based corners.
    np.testing.assert_allclose(gmc_spy[1][1], [[100.0, 59.0, 130.0, 119.0]])


def test_gmc_runs_once_per_frame_including_empty_frames(gmc_spy) -> None:
    detections = steady_detections(6)
    del detections[3], detections[4]
    recorder = GMCRecorder()
    track_frames(detections, 6, panning_frames(4, 2), BotSortConfig(), recorder)
    assert len(gmc_spy) == 6
    assert len(recorder.records) == 6
    assert [record["boxes"] for record in recorder.records] == [1, 1, 0, 0, 1, 1]


def test_synthetic_camera_motion_gives_the_expected_warp() -> None:
    recorder = GMCRecorder()
    track_frames(steady_detections(6), 6, panning_frames(4, 2), BotSortConfig(), recorder)
    assert recorder.records[0]["status"] == "first_frame_identity"
    for record in recorder.records[1:]:
        assert record["status"] == "estimated"
        # Camera moving right and down shifts image content left and up.
        np.testing.assert_allclose(record["warp"][:, 2], (-4.0, -2.0), atol=0.5)
        np.testing.assert_allclose(record["warp"][:, :2], np.eye(2), atol=0.02)


def test_warp_changes_tracking_under_fast_camera_motion() -> None:
    step = 25
    detections = {
        frame: np.array([detection_row(frame, 250.0 - step * frame)])
        for frame in range(1, 9)
    }
    read = panning_frames(step, 0)
    with_gmc = track_frames(detections, 8, read, BotSortConfig())
    without_gmc = track_frames(detections, 8, read, BotSortConfig(gmc_method="none"))
    assert set(with_gmc[:, 1]) == {1.0}
    assert len(set(with_gmc[:, 0])) >= 7
    assert len(without_gmc) <= 2


# Strict failure behaviour


def test_injected_gmc_exception_fails_the_run(monkeypatch: pytest.MonkeyPatch) -> None:
    def broken(self, raw_frame, detections=None):
        raise RuntimeError("injected")

    monkeypatch.setattr(GMC, "apply", broken)
    recorder = GMCRecorder()
    with pytest.raises(GMCPipelineFailure, match="injected"):
        track_frames(steady_detections(), 6, panning_frames(4, 2), BotSortConfig(), recorder)
    assert recorder.failed_calls == 1


def test_the_original_missing_xyxy_defect_now_fails_the_run(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delattr(DetectionFrame, "xyxy")
    with pytest.raises(GMCPipelineFailure, match="GMC results recorded"):
        track_frames(steady_detections(), 6, panning_frames(4, 2), BotSortConfig())


def test_repeated_identity_warps_fail_the_run(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(GMC, "apply", lambda self, frame, detections=None: np.eye(2, 3))
    with pytest.raises(GMCPipelineFailure, match="consecutive identity"):
        track_frames(steady_detections(9), 9, panning_frames(4, 2), BotSortConfig())


@pytest.mark.parametrize(
    "warp, message",
    [
        (np.full((2, 3), np.nan), "not finite"),
        (np.eye(3), "shape"),
        (np.array([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]), "orientation"),
    ],
)
def test_malformed_warps_fail_the_run(monkeypatch, warp, message) -> None:
    monkeypatch.setattr(GMC, "apply", lambda self, frame, detections=None: warp)
    with pytest.raises(GMCPipelineFailure, match=message):
        track_frames(steady_detections(), 6, panning_frames(4, 2), BotSortConfig())


# Completed runs, using real image files


def write_sequence(root: Path, name: str, frames: int) -> None:
    directory = root / "train" / f"{name}-FRCNN"
    (directory / "img1").mkdir(parents=True)
    (directory / "gt").mkdir()
    (directory / "seqinfo.ini").write_text(
        SEQINFO.format(name=name, length=frames), encoding="utf-8"
    )
    (directory / "gt" / "gt.txt").write_text("1,1,10,10,5,10,1,1,1.0\n", encoding="utf-8")
    read = panning_frames(4, 2)
    for frame in range(1, frames + 1):
        cv2.imwrite(str(directory / "img1" / f"{frame:06d}.jpg"), read(frame))


@pytest.fixture
def small_run(tmp_path: Path):
    dataset = tmp_path / "MOT17"
    write_sequence(dataset, "MOT17-02", 6)
    archive = tmp_path / "archive"
    (archive / "detections").mkdir(parents=True)
    detections = "\n".join(
        format_mot_row(frame, -1, np.array((100.0 + frame, 60.0, 30.0, 60.0)), 0.9)
        for frame in range(1, 7)
    )
    (archive / "detections" / "MOT17-02.txt").write_text(detections + "\n", encoding="utf-8")
    return discover_sequences(dataset), tmp_path / "gmc_fixed", archive


def test_completed_run_accounts_for_every_frame(small_run) -> None:
    sequences, root, archive = small_run
    totals = run_tracking(sequences, root, archive, BotSortConfig())
    assert totals["calls"] == totals["calls_expected"] == 6
    assert totals["successful_calls"] == 6
    assert totals["failed_calls"] == 0
    assert totals["first_frame_identity_warps"] == 1
    assert totals["estimated_warps"] == 5
    status = json.loads((root / "config" / "run_status.json").read_text(encoding="utf-8"))
    assert status["all_sequences_complete"] is True
    warps = load_warps(root / "gmc" / "MOT17-02_warps.csv")
    assert sorted(warps) == [1, 2, 3, 4, 5, 6]
    np.testing.assert_allclose(warps[3][:, 2], (-4.0, -2.0), atol=1.0)


def test_failed_gmc_call_cannot_produce_a_completed_run(small_run, monkeypatch) -> None:
    sequences, root, archive = small_run
    original = GMC.apply
    calls = {"count": 0}

    def fails_on_third(self, raw_frame, detections=None):
        calls["count"] += 1
        if calls["count"] == 3:
            raise RuntimeError("injected")
        return original(self, raw_frame, detections)

    monkeypatch.setattr(GMC, "apply", fails_on_third)
    with pytest.raises(SystemExit, match="did not complete"):
        run_tracking(sequences, root, archive, BotSortConfig())
    status = json.loads((root / "config" / "run_status.json").read_text(encoding="utf-8"))
    assert status["all_sequences_complete"] is False
    assert status["sequences"][0]["status"] == "failed"
    diagnostics = json.loads((root / "gmc" / "diagnostics.json").read_text(encoding="utf-8"))
    assert diagnostics["totals"]["failed_calls"] == 1
    assert diagnostics["complete_without_failure"] is False
    assert not (root / "trackers" / "BoT-SORT_GMC_enabled" / "data" / "MOT17-02.txt").exists()


def test_non_first_identity_warp_is_counted(small_run, monkeypatch) -> None:
    sequences, root, archive = small_run
    original = GMC.apply
    calls = {"count": 0}

    def identity_on_third(self, raw_frame, detections=None):
        calls["count"] += 1
        warp = original(self, raw_frame, detections)
        return np.eye(2, 3) if calls["count"] == 3 else warp

    monkeypatch.setattr(GMC, "apply", identity_on_third)
    totals = run_tracking(sequences, root, archive, BotSortConfig())
    assert totals["non_first_identity_warps"] == 1
    assert totals["estimated_warps"] == 4


def test_pilot_gate_rejects_a_warp_that_does_not_fit_the_images(tmp_path: Path) -> None:
    (tmp_path / "gmc").mkdir()
    (tmp_path / "gmc" / "diagnostics.json").write_text(
        json.dumps(
            {"complete_without_failure": True, "totals": {"non_first_identity_warps": 0}}
        ),
        encoding="utf-8",
    )
    spread = {"median_stabilized_spread_pixels": 4.0, "median_raw_spread_pixels": 40.0}
    static = {
        "sequence": "MOT17-04", "median_warp_center_motion_pixels": 0.2,
        "share_of_frame_pairs_improved": 0.5, "correlation_x": 0.1, "correlation_y": 0.1,
        "static_objects": spread,
    }
    moving = {
        "sequence": "MOT17-10", "median_warp_center_motion_pixels": 6.0,
        "share_of_frame_pairs_improved": 0.95, "correlation_x": 0.9, "correlation_y": 0.9,
        "static_objects": spread,
    }
    assert check_pilot(tmp_path, [static, moving])["passed"] is True
    assert check_pilot(tmp_path, [static, {**moving, "correlation_y": 0.2}])["passed"] is False
    assert check_pilot(tmp_path, [static, {**moving, "share_of_frame_pairs_improved": 0.3}])["passed"] is False


# Independent checks of warp quality


def test_phase_shift_sign_matches_the_warp_convention() -> None:
    gray = cv2.cvtColor(scene(), cv2.COLOR_BGR2GRAY).astype(np.float32)
    previous = gray[100:340, 100:420]
    current = gray[102:342, 104:424]
    warp = np.array([[1.0, 0.0, -4.0], [0.0, 1.0, -2.0]])
    np.testing.assert_allclose(phase_shift(previous, current), (-4.0, -2.0), atol=0.3)
    np.testing.assert_allclose(warp_center_motion(warp, 320, 240), (-4.0, -2.0))


def test_correct_warp_reduces_photometric_error_and_wrong_warp_does_not() -> None:
    gray = cv2.cvtColor(scene(), cv2.COLOR_BGR2GRAY).astype(np.float32)
    previous = gray[100:340, 100:420]
    current = gray[102:342, 104:424]
    right = np.array([[1.0, 0.0, -4.0], [0.0, 1.0, -2.0]])
    wrong = np.array([[1.0, 0.0, 4.0], [0.0, 1.0, 2.0]])
    unwarped, warped = photometric_errors(previous, current, right)
    assert warped < unwarped / 10
    assert photometric_errors(previous, current, wrong)[1] > unwarped


# Stabilization


def test_anchored_residuals_remove_camera_motion_over_a_window() -> None:
    warps = {1: np.eye(2, 3)}
    for frame in range(2, 12):
        warps[frame] = np.array([[1.0, 0.0, -4.0], [0.0, 1.0, -2.0]])
    transforms = to_first_frame_transforms(warps)
    frames = np.arange(1, 12)
    points = np.column_stack((200.0 - 4.0 * (frames - 1), 100.0 - 2.0 * (frames - 1)))
    residuals = anchored_residuals(frames, points, transforms, 4)
    assert residuals
    for raw, compensated in residuals:
        assert raw == pytest.approx(np.hypot(16.0, 8.0))
        assert compensated == pytest.approx(0.0, abs=1e-9)


def test_stabilization_returns_a_static_scene_point_to_its_first_position() -> None:
    warps = {1: np.eye(2, 3)}
    for frame in range(2, 6):
        warps[frame] = np.array([[1.0, 0.0, -4.0], [0.0, 1.0, -2.0]])
    transforms = to_first_frame_transforms(warps)
    frames = np.arange(1, 6)
    points = np.column_stack((100.0 - 4.0 * (frames - 1), 50.0 - 2.0 * (frames - 1)))
    np.testing.assert_allclose(
        stabilize_mot_points(frames, points, transforms), [[100.0, 50.0]] * 5
    )


def test_stabilization_applies_rotation_in_zero_based_pixels() -> None:
    angle = np.deg2rad(5.0)
    warp = np.array([[np.cos(angle), -np.sin(angle), 3.0], [np.sin(angle), np.cos(angle), 1.0]])
    transforms = to_first_frame_transforms({1: np.eye(2, 3), 2: warp})
    original_zero_based = np.array([40.0, 70.0])
    moved = warp[:2, :2] @ original_zero_based + warp[:2, 2]
    recovered = stabilize_mot_points(np.array([2]), (moved + 1.0)[None, :], transforms)
    np.testing.assert_allclose(recovered, (original_zero_based + 1.0)[None, :])


# Trajectory labels and output protection


def test_trajectories_separate_false_positives_from_distractors(tmp_path: Path, monkeypatch) -> None:
    import csv

    gt = np.array(
        [
            [1, 1, 10, 10, 20, 40, 1, 1, 1.0],
            [1, 2, 100, 10, 20, 40, 0, 7, 1.0],
            [1, 3, 200, 10, 20, 40, 1, 1, 0.5],
        ]
    )
    predictions = np.array(
        [
            [1, 5, 10, 10, 20, 40, 0.9, -1, -1, -1],
            [1, 6, 100, 10, 20, 40, 0.9, -1, -1, -1],
            [1, 7, 400, 10, 20, 40, 0.9, -1, -1, -1],
        ],
        dtype=np.float64,
    )
    sequence = type("Sequence", (), {"name": "MOT17-02", "frame_rate": 30})()
    transforms = {1: np.eye(3)}
    _write_trajectories(sequence, predictions, match_sequence(gt, predictions), tmp_path, transforms)
    with (tmp_path / "MOT17-02_botsort.csv").open(encoding="utf-8") as stream:
        rows = {row["track_id"]: row for row in csv.DictReader(stream)}
    assert rows["5"]["match_status"] == "matched"
    assert rows["6"]["match_status"] == "distractor"
    assert rows["7"]["match_status"] == "false_positive"
    assert float(rows["5"]["stabilized_foot_y"]) == float(rows["5"]["foot_y"])
    with (tmp_path / "MOT17-02_ground_truth.csv").open(encoding="utf-8") as stream:
        ground_truth = {row["gt_id"]: row for row in csv.DictReader(stream)}
    assert ground_truth["1"]["match_status"] == "tracked"
    assert ground_truth["3"]["match_status"] == "missed"
    assert ground_truth["1"]["frame_rate"] == "30"
    assert "2" not in ground_truth


def test_archived_no_gmc_stages_refuse_to_overwrite(tmp_path: Path) -> None:
    (tmp_path / "trackers" / "botsort" / "data").mkdir(parents=True)
    for stage in ("track", "evaluate", "analyze", "visualize"):
        with pytest.raises(SystemExit, match="archived"):
            reject_archived_overwrite([stage], tmp_path)
    reject_archived_overwrite(["inventory"], tmp_path)


def test_gmc_output_root_must_be_inside_the_mot17_run() -> None:
    assert ensure_gmc_output_root(
        Path("runs/mot17_tracking_analysis/gmc_fixed")
    ).parts[-2:] == ("mot17_tracking_analysis", "gmc_fixed")
    for path in ("runs/gmc_fixed", "runs/prediction/mot17_tracking_analysis/gmc_fixed"):
        with pytest.raises(MOT17PathError):
            ensure_gmc_output_root(Path(path))
