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
    min_parallax_angle_deg: float = 0.1,
    ambiguity_rms_tolerance_px: float = 0.01,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Robustly triangulate sparse camera-indexed observations.

    Tries camera subsets to seed a 3-D hypothesis, scores it against every
    supplied observation, and refits using consensus inliers. The same minimum
    viewing-ray parallax gate is applied during hypothesis generation and refit.
    Returns per-camera residuals and explicit inlier/outlier camera IDs.
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
    if not np.isfinite(min_parallax_angle_deg) or not 0 <= min_parallax_angle_deg <= 180:
        raise ValueError("min_parallax_angle_deg must be finite and between 0 and 180")
    if not np.isfinite(ambiguity_rms_tolerance_px) or ambiguity_rms_tolerance_px < 0:
        raise ValueError("ambiguity_rms_tolerance_px must be finite and non-negative")

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
    competing_hypotheses: list[tuple[np.ndarray, tuple[int, ...], float]] = []
    # Prefer hypotheses supported by more cameras; combinations are bounded by
    # camera count, which is small for this low-cost capture rig. Retain distinct,
    # equally supported hypotheses so a split-consensus case is not silently
    # decided by combination enumeration order.
    for subset_size in range(len(camera_ids), min_views - 1, -1):
        for subset in combinations(camera_ids, subset_size):
            sparse = [points.get(i) if i in subset else None for i in range(n)]
            try:
                xyz, _ = triangulate_point(
                    sparse, camera_poses, intrinsics,
                    min_views=min_views, min_depth=min_depth,
                    min_parallax_angle_deg=min_parallax_angle_deg,
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
            if best is None or score[0] > best[0][0]:
                best = (score, xyz, inliers, residuals, depths)
                competing_hypotheses = [(xyz.copy(), tuple(inliers), inlier_rms)]
            elif score[0] == best[0][0]:
                # Retain all hypotheses at the current maximum support. Apply
                # the RMS tie tolerance only after the final best RMS is known;
                # otherwise a later, much better fit could hide a competitor.
                competing_hypotheses.append((xyz.copy(), tuple(inliers), inlier_rms))
                if score[1] > best[0][1]:
                    best = (score, xyz, inliers, residuals, depths)

        # Once a hypothesis explains every available observation, smaller
        # subsets cannot improve consensus cardinality.
        if best is not None and best[0][0] == len(points):
            break

    if best is None:
        raise ValueError("no multiview hypothesis satisfies parallax, inlier, and positive-depth checks")

    # Do not silently choose one of two spatially distinct, equally supported
    # hypotheses backed by disjoint camera sets. Nearby hypotheses are treated
    # as the same solution; this avoids flagging small numerical differences
    # from alternative subsets of one physical marker.
    best_count = best[0][0]
    best_rms = -best[0][1]
    best_xyz = best[1]
    best_inliers = tuple(best[2])
    scale = max(1.0, float(np.linalg.norm(best_xyz)))
    separation_tolerance = 1e-3 * scale
    for candidate_xyz, candidate_inliers, candidate_rms in competing_hypotheses:
        if len(candidate_inliers) != best_count:
            continue
        if set(candidate_inliers).intersection(best_inliers):
            continue
        if abs(candidate_rms - best_rms) > ambiguity_rms_tolerance_px:
            continue
        if np.linalg.norm(candidate_xyz - best_xyz) > separation_tolerance:
            raise ValueError(
                "ambiguous multiview consensus: distinct hypotheses have equal support"
            )

    _, xyz, inliers, _, _ = best
    # Iterate refit/reclassification until the reported consensus is exactly
    # the set used by the fit. This avoids returning fit metrics for a stale
    # inlier set when thresholding changes membership after refitting.
    seen_inlier_sets: set[tuple[int, ...]] = set()
    fit_metrics: dict[str, Any] = {}
    residuals: dict[int, float] = {}
    depths: dict[int, float] = {}
    final_inliers: list[int] = []
    for _ in range(len(points) + 1):
        inlier_key = tuple(inliers)
        if inlier_key in seen_inlier_sets:
            raise ValueError("consensus inlier set did not stabilize after refitting")
        seen_inlier_sets.add(inlier_key)
        sparse = [points.get(i) if i in inliers else None for i in range(n)]
        xyz, fit_metrics = triangulate_point(
            sparse, camera_poses, intrinsics,
            min_views=min_views, max_reprojection_rms_px=max_reprojection_rms_px,
            min_depth=min_depth, min_parallax_angle_deg=min_parallax_angle_deg,
        )
        residuals = {}
        depths = {}
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
        if final_inliers == inliers:
            break
        inliers = final_inliers
    else:
        raise ValueError("consensus inlier set did not stabilize after refitting")

    # Since the fit and classification now share the same camera set, this RMS
    # is consistent with fit_metrics' residual statistics.
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
        "max_parallax_angle_deg": fit_metrics["max_parallax_angle_deg"],
        "fit_metrics": fit_metrics,
    }
