"""Bounded pairwise multi-camera association using epipolar geometry.

This module proposes one-to-one candidate matches between two sets of 2-D
observations. It does not triangulate or assign marker identities; accepted
matches should still be checked by multiview reprojection and rigid-pattern
constraints.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass(frozen=True)
class PairMatch:
    left_index: int
    right_index: int
    sampson_error_px: float


def _points_array(points: object, name: str) -> np.ndarray:
    array = np.asarray(points, dtype=np.float64)
    if array.size == 0:
        return np.empty((0, 2), dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError(f"{name} must have shape (N, 2)")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite coordinates")
    return array


def sampson_errors(left_points: object, right_points: object, fundamental: object) -> np.ndarray:
    """Return squared Sampson errors in pixel units for every point pair."""
    left = _points_array(left_points, "left_points")
    right = _points_array(right_points, "right_points")
    F = np.asarray(fundamental, dtype=np.float64)
    if F.shape != (3, 3) or not np.isfinite(F).all():
        raise ValueError("fundamental matrix must be a finite 3x3 matrix")
    if not np.any(F):
        raise ValueError("fundamental matrix must be non-zero")
    # Fundamental matrices are defined only up to a non-zero scale. Normalize
    # before forming quadratic terms so the absolute denominator guard below
    # cannot make otherwise equivalent matrices behave differently.
    F = F / np.max(np.abs(F))
    if left.shape[0] == 0 or right.shape[0] == 0:
        return np.empty((left.shape[0], right.shape[0]), dtype=np.float64)

    x1 = np.column_stack((left, np.ones(left.shape[0])))
    x2 = np.column_stack((right, np.ones(right.shape[0])))
    Fx1 = (F @ x1.T).T
    Ftx2 = (F.T @ x2.T).T
    numerator = (x2 @ F @ x1.T) ** 2
    denominator = (
        Fx1[None, :, 0] ** 2 + Fx1[None, :, 1] ** 2
        + Ftx2[:, None, 0] ** 2 + Ftx2[:, None, 1] ** 2
    )
    result = np.full_like(numerator, np.inf, dtype=np.float64)
    np.divide(numerator, denominator, out=result, where=denominator > 1e-15)
    # The quadratic form above is indexed [right, left]; expose costs as [left, right].
    return result.T


def associate_pairwise(
    left_points: object,
    right_points: object,
    fundamental: object,
    *,
    max_sampson_error_px: float = 2.0,
) -> list[PairMatch]:
    """Find minimum-cost one-to-one matches, leaving detections unmatched.

    max_sampson_error_px is a distance threshold in pixels; the assignment
    cost uses squared Sampson distance internally.
    Dummy columns let any left detection remain unmatched instead of forcing
    a geometrically invalid match.
    """
    if not np.isfinite(max_sampson_error_px) or max_sampson_error_px <= 0:
        raise ValueError("max_sampson_error_px must be finite and positive")
    left = _points_array(left_points, "left_points")
    right = _points_array(right_points, "right_points")
    errors = sampson_errors(left, right, fundamental)
    n_left, n_right = errors.shape
    if n_left == 0 or n_right == 0:
        return []

    threshold_sq = max_sampson_error_px ** 2
    if not np.isfinite(threshold_sq) or threshold_sq == 0.0:
        raise ValueError("max_sampson_error_px squared must be finite and positive")
    # Each left observation gets its own dummy column at the rejection cost.
    # Real candidate costs at/above the threshold are not competitive.
    costs = np.full((n_left, n_right + n_left), threshold_sq, dtype=np.float64)
    costs[:, :n_right] = np.where(errors < threshold_sq, errors, threshold_sq * 4.0)
    costs[:, n_right:] = threshold_sq
    rows, cols = linear_sum_assignment(costs)

    matches = []
    for row, col in zip(rows, cols):
        if col >= n_right or errors[row, col] >= threshold_sq:
            continue
        matches.append(PairMatch(
            left_index=int(row),
            right_index=int(col),
            sampson_error_px=float(np.sqrt(errors[row, col])),
        ))
    return sorted(matches, key=lambda match: match.left_index)
