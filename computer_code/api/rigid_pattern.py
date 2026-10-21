"""Rigid-layout fitting for validating reconstructed marker candidates.

The model is a set of known 3-D marker coordinates in object-local space.
Each observed 3-D point must be paired with its model marker index. This
module estimates a proper rigid transform (rotation + translation), never
scale, and reports residuals in the same length units as the input.
"""
from __future__ import annotations

from typing import Any
import numpy as np


def _real_point_array(points: object, name: str) -> np.ndarray:
    """Preserve original scalar types before NumPy can coerce mixed inputs."""
    originals = np.asarray(points, dtype=object).reshape(-1)
    if any(
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, float, np.integer, np.floating))
        for value in originals
    ):
        raise ValueError(f"{name} must contain real numeric coordinates")
    try:
        return np.asarray(points, dtype=np.float64)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain finite float64 coordinates") from exc


def fit_rigid_pattern(
    model_points: object,
    observed_points: object,
    marker_indices: object | None = None,
    *,
    min_markers: int = 3,
    max_rms_error: float | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Fit known marker coordinates to 3-D observations using Kabsch.

    Args:
        model_points: (N, 3) fixed marker coordinates in object-local space.
        observed_points: (M, 3) reconstructed world-space marker coordinates.
        marker_indices: (M,) indices into model_points; defaults to 0..M-1.
        min_markers: minimum number of paired markers (at least 3).
        max_rms_error: optional RMS residual acceptance threshold.

    Returns:
        (R, t, metrics), where observed ~= (R @ model.T).T + t.
        R is a proper 3x3 rotation and t is a 3-vector. Metrics include
        per-marker residuals and RMS/max residual. Raises ValueError for
        invalid, duplicate, degenerate, or over-threshold inputs.
    """
    model = _real_point_array(model_points, "model_points")
    observed = _real_point_array(observed_points, "observed_points")
    if model.ndim != 2 or model.shape[1:] != (3,) or len(model) < 3:
        raise ValueError("model_points must have shape (N, 3), N >= 3")
    if observed.ndim != 2 or observed.shape[1:] != (3,):
        raise ValueError("observed_points must have shape (M, 3)")
    if marker_indices is None:
        indices = np.arange(len(observed), dtype=np.int64)
    else:
        # Check elements before NumPy promotion: [0, True, 2] must never
        # become the valid integer IDs [0, 1, 2].
        original_indices = np.asarray(marker_indices, dtype=object)
        if original_indices.ndim != 1 or len(original_indices) != len(observed):
            raise ValueError("marker_indices must have one entry per observation")
        parsed_indices = []
        for value in original_indices:
            if (
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, float, np.integer, np.floating))
            ):
                raise ValueError("marker_indices must be real integers")
            if isinstance(value, (float, np.floating)) and (
                not np.isfinite(value) or value != np.floor(value)
            ):
                raise ValueError("marker_indices must be integers")
            index = int(value)
            if index < 0 or index >= len(model):
                raise ValueError("marker index is outside model_points")
            parsed_indices.append(index)
        indices = np.asarray(parsed_indices, dtype=np.int64)
    if (
        isinstance(min_markers, (bool, np.bool_))
        or not isinstance(min_markers, (int, np.integer))
        or min_markers < 3
    ):
        raise ValueError("min_markers must be an integer >= 3")
    if len(observed) < min_markers:
        raise ValueError("at least min_markers paired markers are required")
    if len(indices) != len(set(indices.tolist())):
        raise ValueError("marker_indices must be unique")
    if np.any(indices < 0) or np.any(indices >= len(model)):
        raise ValueError("marker index is outside model_points")
    if not np.isfinite(model).all() or not np.isfinite(observed).all():
        raise ValueError("point coordinates must be finite")
    max_rms_limit = None
    if max_rms_error is not None:
        if (
            isinstance(max_rms_error, (bool, np.bool_))
            or not isinstance(max_rms_error, (int, float, np.integer, np.floating))
        ):
            raise ValueError("max_rms_error must be a finite non-negative real scalar")
        try:
            max_rms_limit = float(max_rms_error)
        except (OverflowError, TypeError, ValueError) as exc:
            raise ValueError(
                "max_rms_error must be a finite non-negative real scalar"
            ) from exc
        if not np.isfinite(max_rms_limit) or max_rms_limit < 0:
            raise ValueError("max_rms_error must be a finite non-negative real scalar")

    paired_model = model[indices]
    model_center = paired_model.mean(axis=0)
    observed_center = observed.mean(axis=0)
    X = paired_model - model_center
    Y = observed - observed_center
    # Use a relative singular-value threshold so degeneracy detection is
    # invariant to the coordinate units used for the marker layout.
    centered_singular_values = np.linalg.svd(X, compute_uv=False)
    if (
        centered_singular_values[0] == 0.0
        or centered_singular_values[1] <= 1e-10 * centered_singular_values[0]
    ):
        raise ValueError("model marker subset is collinear or degenerate")

    U, singular_values, Vt = np.linalg.svd(X.T @ Y)
    Q = U @ Vt
    if np.linalg.det(Q) < 0:
        U[:, -1] *= -1
        Q = U @ Vt
    # Q maps row-vector model coordinates to row-vector observed coordinates.
    R = Q.T
    t = observed_center - R @ model_center
    predicted = (R @ paired_model.T).T + t
    residuals = np.linalg.norm(predicted - observed, axis=1)
    rms = float(np.sqrt(np.mean(residuals ** 2)))
    maximum = float(np.max(residuals))
    if not np.isfinite(rms) or not np.isfinite(maximum):
        raise ValueError("rigid-pattern residuals must be finite")
    if max_rms_limit is not None and rms > max_rms_limit:
        raise ValueError(f"rigid-pattern RMS {rms:.6g} exceeds limit {max_rms_limit:.6g}")
    return R, t, {
        "marker_indices": indices.tolist(),
        "residuals": residuals.tolist(),
        "rms_error": rms,
        "max_error": maximum,
        "singular_values": singular_values.tolist(),
        "det_rotation": float(np.linalg.det(R)),
    }
