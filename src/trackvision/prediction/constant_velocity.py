"""Constant-velocity trajectory extrapolation in sampled-observation units."""

from __future__ import annotations

from typing import TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray: TypeAlias = NDArray[np.float64]


def predict_constant_velocity(
    observed: ArrayLike,
    horizon_len: int = 12,
) -> FloatArray:
    """Extrapolate from the final two points of one or more 8-step trajectories.

    The displacement is per sampled observation, not per second. Positions remain in the
    coordinate system and units supplied by the caller.
    """
    values = np.asarray(observed, dtype=np.float64)
    if values.ndim not in (2, 3) or values.shape[-2:] != (8, 2):
        raise ValueError("observed must have shape (8, 2) or (batch, 8, 2)")
    if horizon_len <= 0:
        raise ValueError("horizon_len must be positive")
    if values.ndim == 3 and values.shape[0] == 0:
        raise ValueError("observed batch must not be empty")
    if not np.isfinite(values).all():
        raise ValueError("observed positions must be finite")

    displacement = values[..., -1, :] - values[..., -2, :]
    steps = np.arange(1, horizon_len + 1, dtype=np.float64)
    if values.ndim == 3:
        steps = steps[None, :, None]
    else:
        steps = steps[:, None]
    return values[..., -1:, :] + displacement[..., np.newaxis, :] * steps
