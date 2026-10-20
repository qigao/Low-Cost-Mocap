"""Robust multi-view consensus over one candidate observation per camera.

This is a geometry validator, not a detector-assignment algorithm. Input observations
must already be candidate correspondences indexed by camera; this module rejects
cross-view outliers using reprojection consensus and positive camera-space depth.
"""
from __future__ import annotations

from itertools import combinations
from typing import Any, Mapping, Sequence

import numpy as np

from geometry import _camera_arrays, triangulate_point


def _project(point: np.ndarray, pose: Mapping[str, Any], intrinsic: Any) -> tuple[np.ndarray, float]:
    K, R, t = _camera_arrays(pose, intrinsic)
    camera_point = R @ point + t
    depth = float(camera_point[2])
    if depth <= 0.0:
        return np.array([np.nan, np.nan]), depth
    projected_h = K @ camera_point
    if abs(projected_h[2]) <= 1e-12:
        return np.array([np.nan, np.nan]), depth
    return projected_h[:2] / projected_h[2], depth


def triangulate_consensus(
    observations: Sequence[Sequence[float] | None],
    camera_poses: Sequence[Mapping[str, Any]],
    intrinsics: Sequence[Any],
    *,
    min_views: int = 3,
    inlier_threshold_px: float = 2.0,
    max_reprojection_rms_px: float = 1.0,
    min_depth: float = 1e-9,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Robustly triangulate sparse camera-indexed observations.

    Tries camera subsets to seed a 3-D hypothesis, scores it against every
    supplied observation, and refits using consensus inliers. Returns per-camera
    residuals and explicit inlier/outlier camera IDs. Raises ValueError if no
    hypothesis has enough positive-depth, reprojection-consistent views.
    """
    n = len(observations)
    if len(camera_poses) != n or len(intrinsics) != n:
        raise ValueError("observations, camera_poses, and intrinsics must have equal length")
    if min_views < 3:
        raise ValueError("min_views must be at least 3 for multiview consensus")
    if not np.isfinite(inlier_threshold_px) or inlier_threshold_px <= 0:
        raise ValueError("inlier_threshold_px must be finite and positive")
    if not np.isfinite(max_reprojection_rms_px) or max_reprojection_rms_px < 0:
        raise ValueError("max_reprojection_rms_px must be finite and non-negative")
    if not np.isfinite(min_depth) or min_depth < 0:
        raise ValueError("min_depth must be finite and non-negative")

    points: dict[int, np.ndarray] = {}
    for camera_id, observation in enumerate(observations):
        if observation is None:
            continue
        point = np.asarray(observation, dtype=np.float64).reshape(-1)
        if point.shape != (2,) or not np.isfinite(point).all():
            raise ValueError(f"observation for camera {camera_id} must be a finite (x, y) pair")
        points[camera_id] = point
    if len(points) < min_views:
        raise ValueError(f"need at least {min_views} observations; got {len(points)}")

    camera_ids = tuple(points)
    best = None
    # Prefer hypotheses supported by more cameras; combinations are bounded by
    # camera count, which is small for this low-cost capture rig.
    for subset_size in range(len(camera_ids), min_views - 1, -1):
        for subset in combinations(camera_ids, subset_size):
            sparse = [points.get(i) if i in subset else None for i in range(n)]
            try:
                xyz, _ = triangulate_point(
                    sparse, camera_poses, intrinsics,
                    min_views=min_views, min_depth=min_depth,
                )
            except (ValueError, np.linalg.LinAlgError):
                continue

            residuals: dict[int, float] = {}
            depths: dict[int, float] = {}
            inliers = []
            for camera_id, observed in points.items():
                projected, depth = _project(xyz, camera_poses[camera_id], intrinsics[camera_id])
                depths[camera_id] = depth
                error = float(np.linalg.norm(projected - observed)) if depth > min_depth else float("inf")
                residuals[camera_id] = error
                if depth > min_depth and error <= inlier_threshold_px:
                    inliers.append(camera_id)

            if len(inliers) < min_views:
                continue
            inlier_rms = float(np.sqrt(np.mean([residuals[i] ** 2 for i in inliers])))
            score = (len(inliers), -inlier_rms)
            if best is None or score > best[0]:
                best = (score, xyz, inliers, residuals, depths)

        # Once a hypothesis explains every available observation, smaller
        # subsets cannot improve consensus cardinality.
        if best is not None and best[0][0] == len(points):
            break

    if best is None:
        raise ValueError("no multiview hypothesis satisfies inlier and positive-depth checks")

    _, xyz, inliers, _, _ = best
    # Refit on consensus inliers, then re-evaluate every observation.
    sparse = [points.get(i) if i in inliers else None for i in range(n)]
    xyz, fit_metrics = triangulate_point(
        sparse, camera_poses, intrinsics,
        min_views=min_views, max_reprojection_rms_px=max_reprojection_rms_px,
        min_depth=min_depth,
    )
    residuals: dict[int, float] = {}
    depths: dict[int, float] = {}
    final_inliers = []
    for camera_id, observed in points.items():
        projected, depth = _project(xyz, camera_poses[camera_id], intrinsics[camera_id])
        depths[camera_id] = depth
        error = float(np.linalg.norm(projected - observed)) if depth > min_depth else float("inf")
        residuals[camera_id] = error
        if depth > min_depth and error <= inlier_threshold_px:
            final_inliers.append(camera_id)

    if len(final_inliers) < min_views:
        raise ValueError("consensus became invalid after refitting")
    rms = float(np.sqrt(np.mean([residuals[i] ** 2 for i in final_inliers])))
    if rms > max_reprojection_rms_px:
        raise ValueError(
            f"consensus reprojection RMS {rms:.4f}px exceeds limit {max_reprojection_rms_px:.4f}px"
        )

    return xyz, {
        "inlier_camera_ids": final_inliers,
        "outlier_camera_ids": [i for i in points if i not in final_inliers],
        "reprojection_errors_px": residuals,
        "reprojection_rms_px": rms,
        "depths": depths,
        "fit_metrics": fit_metrics,
    }
