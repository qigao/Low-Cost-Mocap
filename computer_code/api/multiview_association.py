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
    # Keep homogeneous-coordinate products and their squared terms inside the
    # float64 range. Larger finite coordinates are not safely scoreable by this
    # direct pixel-domain formulation, so reject them before matrix arithmetic.
    max_safe_coordinate = np.sqrt(np.finfo(np.float64).max) / 8.0
    if (
        np.max(np.abs(left), initial=0.0) > max_safe_coordinate
        or np.max(np.abs(right), initial=0.0) > max_safe_coordinate
    ):
        raise ValueError("point coordinates exceed safe float64 Sampson range")
    # Fundamental matrices are defined only up to a non-zero scale. Normalize
    # before forming quadratic terms so the absolute denominator guard below
    # cannot make otherwise equivalent matrices behave differently.
    F = F / np.max(np.abs(F))
    if left.shape[0] == 0 or right.shape[0] == 0:
        return np.empty((left.shape[0], right.shape[0]), dtype=np.float64)

    x1 = np.column_stack((left, np.ones(left.shape[0])))
    x2 = np.column_stack((right, np.ones(right.shape[0])))
    # Finite input coordinates can still overflow intermediate products.
    # Such pairs are not safely scoreable and must be treated as non-matches,
    # never allowed to leak NaN into the assignment cost matrix.
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        Fx1 = (F @ x1.T).T
        Ftx2 = (F.T @ x2.T).T
        numerator = (x2 @ F @ x1.T) ** 2
        denominator = (
            Fx1[None, :, 0] ** 2 + Fx1[None, :, 1] ** 2
            + Ftx2[:, None, 0] ** 2 + Ftx2[:, None, 1] ** 2
        )
        result = np.full_like(numerator, np.inf, dtype=np.float64)
        np.divide(numerator, denominator, out=result, where=denominator > 1e-15)
    result[~np.isfinite(result)] = np.inf
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

    threshold_sq = max_sampson_error_px * max_sampson_error_px
    if (
        not np.isfinite(threshold_sq)
        or threshold_sq == 0.0
        or threshold_sq > np.finfo(np.float64).max / 4.0
    ):
        raise ValueError(
            "max_sampson_error_px squared must be positive and safely representable"
        )
    # Normalize eligible costs to [0, 1). Give each unmatched left point a
    # penalty larger than the maximum possible sum of eligible costs. This
    # makes the assignment maximize the number of valid matches first, then
    # minimize their total Sampson error. Invalid edges cost more than a dummy.
    unmatched_cost = float(n_left + 1)
    invalid_cost = 2.0 * unmatched_cost
    costs = np.full((n_left, n_right + n_left), unmatched_cost, dtype=np.float64)
    normalized = errors / threshold_sq
    costs[:, :n_right] = np.where(errors < threshold_sq, normalized, invalid_cost)
    costs[:, n_right:] = unmatched_cost
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
