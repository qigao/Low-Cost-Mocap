"""Pure geometry for calibrated adjacent-camera pose initialization.

All extrinsics use world-to-camera coordinates: x_cam = R @ x_world + t.
Essential-matrix translation has direction, but *no metric length*; callers
must supply the chosen baseline gauge explicitly before composing poses.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from geometry import _camera_arrays, triangulate_point
from multiview_association import _points_array


def _unit_translation(translation: np.ndarray) -> np.ndarray:
    """Normalize even very small/large finite translation vectors safely."""
    scale = float(np.max(np.abs(translation)))
    if not np.isfinite(scale) or scale == 0.0:
        raise ValueError("relative translation direction must be nonzero and finite")
    scaled = translation / scale
    length = float(np.linalg.norm(scaled))
    if not np.isfinite(length) or length == 0.0:
        raise ValueError("relative translation direction is degenerate")
    return scaled / length


def _pose_components(pose: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    _, rotation, translation = _camera_arrays(pose, np.eye(3))
    return rotation, translation


def compose_world_to_camera_pose(
    previous_pose: Mapping[str, Any],
    relative_pose: Mapping[str, Any],
    *,
    baseline_length: float,
) -> dict[str, np.ndarray]:
    """Compose a second camera pose from its transform relative to the first.

    With x_prev = R_prev @ x_world + t_prev and
    x_next = R_rel @ x_prev + baseline_length * unit(t_rel), this yields:
        R_next = R_rel @ R_prev
        t_next = R_rel @ t_prev + baseline_length * unit(t_rel)

    baseline_length MUST be supplied explicitly. With only an essential
    matrix, its value is a gauge choice, not a measured metric baseline.
    """
    if (
        isinstance(baseline_length, (bool, np.bool_))
        or not isinstance(baseline_length, (int, float, np.integer, np.floating))
    ):
        raise ValueError("baseline_length must be a finite positive real scalar")
    try:
        baseline = float(baseline_length)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError("baseline_length must be a finite positive real scalar") from exc
    if not np.isfinite(baseline) or baseline <= 0.0:
        raise ValueError("baseline_length must be a finite positive real scalar")
    previous_rotation, previous_translation = _pose_components(previous_pose)
    relative_rotation, relative_translation = _pose_components(relative_pose)
    unit_translation = _unit_translation(relative_translation)
    rotation = relative_rotation @ previous_rotation
    translation = relative_rotation @ previous_translation + baseline * unit_translation
    if not np.isfinite(translation).all():
        raise ValueError("composed camera translation must be finite")
    return {"R": rotation, "t": translation}


def select_relative_pose_by_cheirality(
    left_points: object,
    right_points: object,
    left_intrinsic: object,
    right_intrinsic: object,
    candidate_rotations: Sequence[object],
    candidate_translations: Sequence[object],
    *,
    min_support: int = 8,
    max_reprojection_rms_px: float = 2.0,
    min_parallax_angle_deg: float = 0.1,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Choose an essential-matrix solution using validated in-front matches.

    Inputs must be matched UNDISTORTED pixel observations, preferably only
    RANSAC inliers. Each hypothesis triangulates with a local identity first
    camera and its own relative second camera (never global extrinsics).
    Ambiguous maximum-support candidates fail closed. Returned translation
    is a unit direction and conveys no metric baseline.
    """
    left = _points_array(left_points, "left_points")
    right = _points_array(right_points, "right_points")
    if left.shape != right.shape:
        raise ValueError("paired observations must have matching (N, 2) shapes")
    if isinstance(min_support, (bool, np.bool_)) or not isinstance(min_support, (int, np.integer)) or min_support < 2:
        raise ValueError("min_support must be an integer >= 2")
    if len(left) < min_support:
        raise ValueError(f"need at least {min_support} matched observations")
    for name, value, lower, upper in (
        ("max_reprojection_rms_px", max_reprojection_rms_px, 0.0, np.inf),
        ("min_parallax_angle_deg", min_parallax_angle_deg, 0.0, 90.0),
    ):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
            raise ValueError(f"{name} must be a finite real scalar in range")
        try:
            number = float(value)
        except (OverflowError, TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be a finite real scalar in range") from exc
        if not np.isfinite(number) or not lower <= number <= upper:
            raise ValueError(f"{name} must be a finite real scalar in range")
    first_pose = {"R": np.eye(3), "t": np.zeros(3)}
    # Validate both camera intrinsics up front. Invalid calibration is not
    # an outlier that the candidate-scoring loop may silently ignore.
    _camera_arrays(first_pose, left_intrinsic)
    _camera_arrays(first_pose, right_intrinsic)
    if len(candidate_rotations) == 0 or len(candidate_rotations) != len(candidate_translations):
        raise ValueError("rotation and translation candidate lists must be nonempty and equal length")

    counts: list[int] = []
    valid_indices: list[list[int]] = []
    candidate_poses: list[dict[str, np.ndarray]] = []
    candidate_errors: list[list[float]] = []
    for rotation, translation in zip(candidate_rotations, candidate_translations):
        raw_pose = {"R": rotation, "t": translation}
        relative_rotation, relative_translation = _pose_components(raw_pose)
        pose = {"R": relative_rotation, "t": _unit_translation(relative_translation)}
        inlier_indices: list[int] = []
        errors: list[float] = []
        for index, (x1, x2) in enumerate(zip(left, right)):
            try:
                _, metrics = triangulate_point(
                    [x1, x2], [first_pose, pose],
                    [left_intrinsic, right_intrinsic],
                    min_views=2,
                    max_reprojection_rms_px=max_reprojection_rms_px,
                    min_parallax_angle_deg=min_parallax_angle_deg,
                )
            except (ValueError, np.linalg.LinAlgError):
                continue
            inlier_indices.append(index)
            errors.append(metrics["reprojection_rms_px"])
        candidate_poses.append(pose)
        valid_indices.append(inlier_indices)
        candidate_errors.append(errors)
        counts.append(len(inlier_indices))

    best_count = max(counts)
    if best_count < min_support:
        raise ValueError("no relative-pose candidate has sufficient positive-depth support")
    if counts.count(best_count) != 1:
        raise ValueError("ambiguous relative-pose candidates have equal positive-depth support")
    selected_index = counts.index(best_count)
    errors = candidate_errors[selected_index]
    return candidate_poses[selected_index], {
        "candidate_support_counts": counts,
        "selected_candidate_index": selected_index,
        "support_count": best_count,
        "inlier_pair_indices": valid_indices[selected_index],
        "median_reprojection_rms_px": float(np.median(errors)),
        "translation_scale_known": False,
    }
