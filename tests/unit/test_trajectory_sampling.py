import math

import pytest

from trackvision.io.trajectories import (
    SampledPoint,
    UniformTrajectory,
    build_windows,
    resample_ucy_scene,
    sample_ucy_windows,
    sampling_interval_frames,
)
from trackvision.io.ucy import UCYPoint, UCYScene, UCYTrack


def _track(points, spline_id=1):
    return UCYTrack(
        spline_id=spline_id,
        points=tuple(UCYPoint(x, y, frame, 0.0) for frame, x, y in points),
    )


def test_linear_interpolation_hits_known_frames_and_preserves_input_points():
    original = _track([(0, 0.0, 0.0), (10, 10.0, 20.0)])
    scene = UCYScene("UNIV", (original,), ())

    (trajectory,) = resample_ucy_scene(scene, frame_interval=5, max_gap_frames=10)

    assert trajectory.points == (
        SampledPoint(0, 0.0, 0.0),
        SampledPoint(5, 5.0, 10.0),
        SampledPoint(10, 10.0, 20.0),
    )
    assert original.points == (
        UCYPoint(0.0, 0.0, 0, 0.0),
        UCYPoint(10.0, 20.0, 10, 0.0),
    )


def test_irregular_intervals_are_interpolated_at_global_sampling_frames():
    scene = UCYScene("UNIV", (_track([
        (2, 2.0, 4.0),
        (7, 7.0, 14.0),
        (17, 17.0, 34.0),
    ]),), ())

    (trajectory,) = resample_ucy_scene(scene, frame_interval=5, max_gap_frames=10)

    assert [point.frame for point in trajectory.points] == [5, 10, 15]
    assert [(point.x, point.y) for point in trajectory.points] == [
        (5.0, 10.0),
        (10.0, 20.0),
        (15.0, 30.0),
    ]


def test_approved_twenty_frame_gap_can_be_interpolated():
    scene = UCYScene("UNIV", (_track([
        (0, 0.0, 0.0),
        (20, 20.0, 40.0),
    ]),), ())

    (trajectory,) = resample_ucy_scene(scene, frame_interval=10, max_gap_frames=20)

    assert trajectory.points == (
        SampledPoint(0, 0.0, 0.0),
        SampledPoint(10, 10.0, 20.0),
        SampledPoint(20, 20.0, 40.0),
    )


def test_long_gap_splits_trajectory_and_windows_never_cross_it():
    scene = UCYScene("UNIV", (_track([
        (0, 0.0, 0.0),
        (10, 10.0, 0.0),
        (40, 40.0, 0.0),
        (50, 50.0, 0.0),
    ]),), ())

    trajectories = resample_ucy_scene(scene, frame_interval=10, max_gap_frames=10)

    assert [[point.frame for point in item.points] for item in trajectories] == [
        [0, 10],
        [40, 50],
    ]
    windows = build_windows(trajectories, history_len=1, horizon_len=1, stride=1)
    assert [window.start_frame for window in windows] == [0, 40]


def test_splines_remain_separate_at_track_boundaries():
    scene = UCYScene("ZARA1", (
        _track([(0, 0, 0), (10, 10, 0)], spline_id=1),
        _track([(0, 100, 0), (10, 110, 0)], spline_id=2),
    ), ())

    trajectories = resample_ucy_scene(scene, frame_interval=10, max_gap_frames=10)

    assert [(item.scene, item.spline_id) for item in trajectories] == [
        ("ZARA1", 1),
        ("ZARA1", 2),
    ]
    assert trajectories[0].points[-1].x != trajectories[1].points[0].x


def test_window_has_requested_history_and_future_lengths():
    points = tuple(SampledPoint(frame, float(frame), float(-frame))
                   for frame in range(0, 210, 10))
    trajectory = UniformTrajectory("UNIV", 7, points)

    windows = build_windows((trajectory,), history_len=8, horizon_len=12, stride=1)

    assert len(windows) == 2
    assert len(windows[0].observed) == 8
    assert len(windows[0].future) == 12
    assert windows[0].scene == "UNIV"
    assert windows[0].spline_id == 7
    assert windows[0].observed[-1].frame == 70
    assert windows[0].future[0].frame == 80


def test_window_builder_rejects_irregularly_sampled_input():
    trajectory = UniformTrajectory(
        "UNIV",
        7,
        (
            SampledPoint(0, 0.0, 0.0),
            SampledPoint(10, 10.0, 0.0),
            SampledPoint(25, 25.0, 0.0),
        ),
    )

    with pytest.raises(ValueError, match="not uniformly sampled"):
        build_windows((trajectory,), history_len=1, horizon_len=1, stride=1)


def test_sampling_interval_must_map_to_an_integer_frame_count():
    assert sampling_interval_frames(0.4, 25.0) == 10
    with pytest.raises(ValueError, match="integer number"):
        sampling_interval_frames(0.4, 24.0)


@pytest.mark.parametrize(
    "values",
    [
        (0.0, math.nan, 0, 0.0),
        (math.inf, 0.0, 10, 0.0),
    ],
)
def test_non_finite_coordinates_are_rejected(values):
    track = UCYTrack(1, (UCYPoint(*values),))
    scene = UCYScene("UNIV", (track,), ())

    with pytest.raises(ValueError, match="invalid or unordered"):
        resample_ucy_scene(scene, frame_interval=10, max_gap_frames=10)


def test_sample_ucy_windows_uses_explicit_protocol_lengths_and_rate():
    scene = UCYScene("UNIV", (
        _track([(frame, frame, 0) for frame in range(0, 200, 10)]),
    ), ())

    (window,) = sample_ucy_windows(
        scene,
        sampling_interval_s=0.4,
        frame_rate_fps=25.0,
        max_gap_frames=10,
        history_len=8,
        horizon_len=12,
        stride=1,
    )

    assert window.start_frame == 0
    assert len(window.observed) == 8
    assert len(window.future) == 12
    assert window.future[-1].frame == 190


def test_approved_twenty_frame_policy_produces_a_synthetic_window():
    scene = UCYScene("UNIV", (
        _track([(frame, frame, 2 * frame) for frame in range(0, 200, 10)]),
    ), ())

    (window,) = sample_ucy_windows(
        scene,
        sampling_interval_s=0.4,
        frame_rate_fps=25.0,
        max_gap_frames=20,
        history_len=8,
        horizon_len=12,
        stride=1,
    )

    assert len(window.observed) == 8
    assert len(window.future) == 12
    assert window.start_frame == 0
    assert window.future[-1] == SampledPoint(190, 190.0, 380.0)
