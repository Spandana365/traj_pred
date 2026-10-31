"""Uniform sampling and window construction for UCY spline annotations."""

from __future__ import annotations

import math
from dataclasses import dataclass

from trackvision.io.ucy import UCYPoint, UCYScene, UCYTrack


@dataclass(frozen=True)
class SampledPoint:
    """A position at a uniformly sampled source-video frame, in source pixels."""

    frame: int
    x: float
    y: float


@dataclass(frozen=True)
class UniformTrajectory:
    """One uniformly sampled continuous segment from a single source spline."""

    scene: str
    spline_id: int
    points: tuple[SampledPoint, ...]


@dataclass(frozen=True)
class TrajectoryWindow:
    """An 8/12-style prediction window with source scene and spline identity retained."""

    scene: str
    spline_id: int
    start_frame: int
    observed: tuple[SampledPoint, ...]
    future: tuple[SampledPoint, ...]


def sampling_interval_frames(sampling_interval_s: float, frame_rate_fps: float) -> int:
    """Convert a time interval to an exact integer video-frame interval."""
    if (not math.isfinite(sampling_interval_s) or sampling_interval_s <= 0
            or not math.isfinite(frame_rate_fps) or frame_rate_fps <= 0):
        raise ValueError("sampling interval and frame rate must be positive finite values")
    frame_interval = sampling_interval_s * frame_rate_fps
    rounded = round(frame_interval)
    if not math.isclose(frame_interval, rounded, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("sampling interval must correspond to an integer number of video frames")
    return rounded


def _validate_track(track: UCYTrack) -> None:
    previous_frame = -1
    for point in track.points:
        if (not all(math.isfinite(value) for value in
                    (point.x, point.y, point.gaze_direction_degrees))
                or point.frame < 0 or point.frame <= previous_frame):
            raise ValueError(f"invalid or unordered control point in spline {track.spline_id}")
        previous_frame = point.frame


def _sample_segment(
    track: UCYTrack,
    segment: tuple[UCYPoint, ...],
    frame_interval: int,
    frame_origin: int,
) -> tuple[SampledPoint, ...]:
    first_frame, last_frame = segment[0].frame, segment[-1].frame
    start_frame = first_frame + (frame_origin - first_frame) % frame_interval
    if start_frame > last_frame:
        return ()
    end_frame = last_frame - (last_frame - frame_origin) % frame_interval
    sampled: list[SampledPoint] = []
    if len(segment) == 1:
        point = segment[0]
        if (point.frame - frame_origin) % frame_interval == 0:
            return (SampledPoint(frame=point.frame, x=point.x, y=point.y),)
        return ()
    right_index = 1

    for frame in range(start_frame, end_frame + 1, frame_interval):
        while right_index < len(segment) and segment[right_index].frame < frame:
            right_index += 1
        if right_index >= len(segment):
            break
        right = segment[right_index]
        if right.frame == frame:
            sampled.append(SampledPoint(frame=frame, x=right.x, y=right.y))
            continue
        left = segment[right_index - 1]
        if left.frame == frame:
            sampled.append(SampledPoint(frame=frame, x=left.x, y=left.y))
            continue
        gap = right.frame - left.frame
        if gap <= 0:
            raise ValueError(f"invalid frame order in spline {track.spline_id}")
        ratio = (frame - left.frame) / gap
        sampled.append(SampledPoint(
            frame=frame,
            x=left.x + ratio * (right.x - left.x),
            y=left.y + ratio * (right.y - left.y),
        ))
    return tuple(sampled)


def resample_ucy_scene(
    scene: UCYScene,
    frame_interval: int,
    max_gap_frames: int,
    frame_origin: int = 0,
) -> tuple[UniformTrajectory, ...]:
    """Linearly sample within splines, splitting wherever an annotation gap is too large.

    Interpolation is restricted to adjacent control points no farther apart than
    `max_gap_frames`. Splines are never joined; target frames are aligned to
    `frame_origin` and separated by `frame_interval`.
    """
    if frame_interval <= 0:
        raise ValueError("frame_interval must be a positive integer")
    if max_gap_frames < frame_interval:
        raise ValueError("max_gap_frames must be at least frame_interval")

    trajectories: list[UniformTrajectory] = []
    for track in scene.tracks:
        _validate_track(track)
        if not track.points:
            continue
        start = 0
        for index in range(1, len(track.points) + 1):
            at_end = index == len(track.points)
            gap_exceeded = (not at_end and
                            track.points[index].frame - track.points[index - 1].frame
                            > max_gap_frames)
            if not at_end and not gap_exceeded:
                continue
            segment = track.points[start:index]
            samples = _sample_segment(track, segment, frame_interval, frame_origin)
            if samples:
                trajectories.append(UniformTrajectory(
                    scene=scene.scene, spline_id=track.spline_id, points=samples
                ))
            start = index
    return tuple(trajectories)


def build_windows(
    trajectories: tuple[UniformTrajectory, ...],
    history_len: int,
    horizon_len: int,
    stride: int,
) -> tuple[TrajectoryWindow, ...]:
    """Build sliding windows within individual continuous trajectories only."""
    if history_len <= 0 or horizon_len <= 0 or stride <= 0:
        raise ValueError("history_len, horizon_len, and stride must be positive integers")
    window_len = history_len + horizon_len
    windows: list[TrajectoryWindow] = []
    for trajectory in trajectories:
        points = trajectory.points
        if len(points) > 1:
            sample_step = points[1].frame - points[0].frame
            if sample_step <= 0 or any(
                right.frame - left.frame != sample_step
                for left, right in zip(points, points[1:])
            ):
                raise ValueError(
                    f"trajectory for spline {trajectory.spline_id} is not uniformly sampled"
                )
        for start in range(0, len(points) - window_len + 1, stride):
            window = points[start:start + window_len]
            windows.append(TrajectoryWindow(
                scene=trajectory.scene,
                spline_id=trajectory.spline_id,
                start_frame=window[0].frame,
                observed=window[:history_len],
                future=window[history_len:],
            ))
    return tuple(windows)


def sample_ucy_windows(
    scene: UCYScene,
    sampling_interval_s: float,
    frame_rate_fps: float,
    max_gap_frames: int,
    history_len: int,
    horizon_len: int,
    stride: int,
    frame_origin: int = 0,
) -> tuple[TrajectoryWindow, ...]:
    """Create windows using an explicit sampling interval, video rate, and gap policy."""
    frame_interval = sampling_interval_frames(sampling_interval_s, frame_rate_fps)
    trajectories = resample_ucy_scene(
        scene, frame_interval, max_gap_frames, frame_origin=frame_origin
    )
    return build_windows(trajectories, history_len, horizon_len, stride)
