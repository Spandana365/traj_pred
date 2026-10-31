"""ADE/FDE for 2D predictions, averaged equally across trajectory windows."""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
from numpy.typing import ArrayLike


class ADEAndFDE(NamedTuple):
    """Mean per-window ADE and FDE, in the coordinates' original units."""

    ade: float
    fde: float


def ade_fde(predicted: ArrayLike, target: ArrayLike) -> ADEAndFDE:
    """Compute Euclidean ADE/FDE and average each metric over equally weighted windows.

    Inputs may be one trajectory ``(horizon, 2)`` or a batch
    ``(windows, horizon, 2)``. ADE is first averaged over each window's horizon; the
    resulting per-window values are then averaged, so windows rather than positions across the whole
    dataset—receive equal weight.
    """
    predictions = np.asarray(predicted, dtype=np.float64)
    targets = np.asarray(target, dtype=np.float64)
    if predictions.shape != targets.shape:
        raise ValueError("predicted and target must have identical shapes")
    if predictions.ndim not in (2, 3) or predictions.shape[-1] != 2:
        raise ValueError("inputs must have shape (horizon, 2) or (windows, horizon, 2)")
    if predictions.shape[-2] == 0:
        raise ValueError("prediction horizon must not be empty")
    if predictions.ndim == 3 and predictions.shape[0] == 0:
        raise ValueError("window batch must not be empty")
    if not np.isfinite(predictions).all() or not np.isfinite(targets).all():
        raise ValueError("predicted and target positions must be finite")

    distances = np.linalg.norm(predictions - targets, axis=-1)
    if distances.ndim == 1:
        per_window_ade = distances.mean(keepdims=True)
        per_window_fde = distances[-1:]
    else:
        per_window_ade = distances.mean(axis=1)
        per_window_fde = distances[:, -1]
    return ADEAndFDE(float(per_window_ade.mean()), float(per_window_fde.mean()))
