"""Numerically stable, camera-index-preserving multi-view geometry primitives.

Input image points are expected to be in the same undistorted pixel coordinate
system used by each intrinsic matrix. Camera poses use world-to-camera
convention: X_camera = R @ X_world + t.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence
import numpy as np


def _camera_arrays(pose: Mapping[str, Any], intrinsic: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    K = np.asarray(intrinsic, dtype=np.float64)
    R = np.asarray(pose["R"], dtype=np.float64)
    t = np.asarray(pose["t"], dtype=np.float64).reshape(-1)
    if K.shape != (3, 3) or R.shape != (3, 3) or t.shape != (3,):
        raise ValueError("K and R must be 3x3; t must contain 3 values")
    if not (np.isfinite(K).all() and np.isfinite(R).all() and np.isfinite(t).all()):
        raise ValueError("camera parameters must be finite")
    if abs(np.linalg.det(K)) < 1e-12:
        raise ValueError("intrinsic matrix is singular")
    return K, R, t


def triangulate_point(
    observations: Sequence[Sequence[float] | None],
    camera_poses: Sequence[Mapping[str, Any]],
    intrinsics: Sequence[Any],
    *,
    min_views: int = 2,
    max_reprojection_rms_px: float | None = None,
    min_depth: float = 1e-9,
    min_parallax_angle_deg: float = 0.1,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Triangulate one point from sparse camera-indexed observations.

    observations[i] belongs to camera i; use None when that camera has no
    detection. Invalid or weak-baseline solutions raise ValueError. The
    parallax gate is the largest pairwise angle between world-space viewing
    rays; it guards against depth estimates from nearly parallel rays.
    """
    n = len(observations)
    if len(camera_poses) != n or len(intrinsics) != n:
        raise ValueError("observations, camera_poses, and intrinsics must have equal length")
    if min_views < 2:
        raise ValueError("min_views must be at least 2")
    if max_reprojection_rms_px is not None and (
        not np.isfinite(max_reprojection_rms_px) or max_reprojection_rms_px < 0
    ):
        raise ValueError("max_reprojection_rms_px must be finite and non-negative")
    if not np.isfinite(min_depth) or min_depth < 0:
        raise ValueError("min_depth must be finite and non-negative")
    if not np.isfinite(min_parallax_angle_deg) or not 0 <= min_parallax_angle_deg <= 180:
        raise ValueError("min_parallax_angle_deg must be finite and between 0 and 180")

    valid: list[tuple[int, np.ndarray, np.ndarray]] = []
    rays: list[np.ndarray] = []
    for camera_id, xy in enumerate(observations):
        if xy is None:
            continue
        point = np.asarray(xy, dtype=np.float64).reshape(-1)
        if point.shape != (2,) or not np.isfinite(point).all():
            raise ValueError(f"observation for camera {camera_id} must be a finite (x, y) pair")
        K, R, t = _camera_arrays(camera_poses[camera_id], intrinsics[camera_id])
        P = K @ np.column_stack((R, t))
        valid.append((camera_id, point, P))
        ray_camera = np.linalg.solve(K, np.array([point[0], point[1], 1.0]))
        ray_world = R.T @ ray_camera
        ray_norm = float(np.linalg.norm(ray_world))
        if not np.isfinite(ray_norm) or ray_norm <= 1e-15:
            raise ValueError(f"viewing ray for camera {camera_id} is degenerate")
        rays.append(ray_world / ray_norm)

    if len(valid) < min_views:
        raise ValueError(f"need at least {min_views} valid camera observations; got {len(valid)}")

    max_parallax_deg = 0.0
    for i in range(len(rays)):
        for j in range(i + 1, len(rays)):
            cosine = float(np.clip(np.dot(rays[i], rays[j]), -1.0, 1.0))
            angle = float(np.degrees(np.arccos(cosine)))
            max_parallax_deg = max(max_parallax_deg, angle)
    if max_parallax_deg < min_parallax_angle_deg:
        raise ValueError(
            f"triangulation parallax {max_parallax_deg:.6g} deg is below "
            f"minimum {min_parallax_angle_deg:.6g} deg"
        )

    rows = []
    for _, (u, v), P in valid:
        rows.append(u * P[2] - P[0])
        rows.append(v * P[2] - P[1])
    A = np.asarray(rows, dtype=np.float64)
    if not np.isfinite(A).all() or np.linalg.matrix_rank(A) < 3:
        raise ValueError("triangulation system is degenerate")

    # SVD the original design matrix; A.T @ A would square its condition number.
    _, singular_values, vh = np.linalg.svd(A, full_matrices=False)
    homogeneous = vh[-1]
    if abs(homogeneous[3]) <= 1e-12 * max(1.0, np.linalg.norm(homogeneous[:3])):
        raise ValueError("triangulated point is at infinity or numerically unstable")
    xyz = homogeneous[:3] / homogeneous[3]
    if not np.isfinite(xyz).all():
        raise ValueError("triangulation produced a non-finite point")

    residuals: dict[int, float] = {}
    depths: dict[int, float] = {}
    for camera_id, observed, _ in valid:
        K, R, t = _camera_arrays(camera_poses[camera_id], intrinsics[camera_id])
        camera_xyz = R @ xyz + t
        depth = float(camera_xyz[2])
        depths[camera_id] = depth
        if depth <= min_depth:
            raise ValueError(f"triangulated point is behind camera {camera_id} (depth={depth:g})")
        projected_h = K @ camera_xyz
        if abs(projected_h[2]) <= 1e-12:
            raise ValueError(f"projection is singular for camera {camera_id}")
        projected = projected_h[:2] / projected_h[2]
        residuals[camera_id] = float(np.linalg.norm(projected - observed))

    rms = float(np.sqrt(np.mean(np.square(list(residuals.values())))))
    if max_reprojection_rms_px is not None and rms > max_reprojection_rms_px:
        raise ValueError(
            f"reprojection RMS {rms:.4f}px exceeds limit {max_reprojection_rms_px:.4f}px"
        )
    metrics = {
        "camera_ids": [camera_id for camera_id, _, _ in valid],
        "reprojection_errors_px": residuals,
        "reprojection_rms_px": rms,
        "depths": depths,
        "max_parallax_angle_deg": max_parallax_deg,
        "design_matrix_singular_values": singular_values.tolist(),
    }
    return xyz, metrics
