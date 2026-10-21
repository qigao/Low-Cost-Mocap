"""Calibrated bundle adjustment with explicit camera and metric-scale gauges.

Image observations are already undistorted and indexed [track, camera, xy].
The first camera's world-to-camera pose and second camera's full translation
vector are held fixed: essential-matrix initialization supplies only an
arbitrary *gauge*, not a measured metric baseline. Per-camera K is fixed.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from scipy.spatial.transform import Rotation

from geometry import _camera_arrays, _observation_array, triangulate_point


def _observations(data: object, camera_count: int) -> tuple[np.ndarray, np.ndarray]:
    raw = np.asarray(data, dtype=object)
    if raw.ndim != 3 or raw.shape[1:] != (camera_count, 2):
        raise ValueError("image_points must have shape (tracks, cameras, 2)")
    coords = np.zeros(raw.shape, dtype=np.float64)
    visible = np.zeros(raw.shape[:2], dtype=bool)
    for track in range(raw.shape[0]):
        for camera in range(camera_count):
            pair = raw[track, camera]
            if all(value is None for value in pair):
                continue
            if any(value is None for value in pair):
                raise ValueError(f"track {track} camera {camera} has incomplete observation")
            # The outer sparse mask uses object dtype for None sentinels;
            # convert each valid pair back to scalars so strict numeric
            # validation does not mistake the container dtype for data type.
            coords[track, camera] = _observation_array(pair.tolist(), camera)
            visible[track, camera] = True
    return coords, visible


def _camera_state(
    poses: Sequence[Mapping[str, Any]], intrinsics: Sequence[object]
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    if len(poses) != len(intrinsics):
        raise ValueError("camera poses and intrinsics must have equal length")
    if len(poses) < 2:
        raise ValueError("bundle adjustment requires at least two cameras")
    Ks, Rs, ts = [], [], []
    for pose, intrinsic in zip(poses, intrinsics):
        K, R, t = _camera_arrays(pose, intrinsic)
        Ks.append(K)
        Rs.append(R)
        ts.append(t)
    if not np.isfinite(np.linalg.norm(ts[1])) or np.linalg.norm(ts[1]) <= 1e-12:
        raise ValueError("second camera must establish a nonzero translation gauge")
    return Ks, Rs, ts


def _pixel_residuals(
    coords: np.ndarray,
    visible: np.ndarray,
    world: np.ndarray,
    Ks: Sequence[np.ndarray],
    Rs: Sequence[np.ndarray],
    ts: Sequence[np.ndarray],
) -> np.ndarray:
    """Signed [u_projected-u_observed, v_projected-v_observed] in track order.

    Candidate points moving behind the camera receive a fixed, finite
    penalty. The number and order of terms cannot change during optimization.
    """
    residuals = np.empty((int(visible.sum()), 2), dtype=np.float64)
    position = 0
    for track, camera in np.argwhere(visible):
        camera_xyz = Rs[camera] @ world[track] + ts[camera]
        if not np.isfinite(camera_xyz).all() or camera_xyz[2] <= 1e-9:
            residuals[position] = (1e5, 1e5)
        else:
            pixel_h = Ks[camera] @ camera_xyz
            if (
                not np.isfinite(pixel_h).all()
                or not np.isfinite(pixel_h[2])
                or abs(pixel_h[2]) <= np.finfo(np.float64).tiny
            ):
                residuals[position] = (1e5, 1e5)
            else:
                predicted = pixel_h[:2] / pixel_h[2]
                if np.isfinite(predicted).all():
                    residuals[position] = predicted - coords[track, camera]
                else:
                    residuals[position] = (1e5, 1e5)
        position += 1
    return residuals.reshape(-1)


def bundle_reprojection_residuals(
    image_points: object,
    camera_poses: Sequence[Mapping[str, Any]],
    intrinsics: Sequence[object],
    world_points: object,
) -> np.ndarray:
    """Return a fixed-cardinality signed pixel residual vector.

    Every visible (track, camera) contributes exactly two entries, in stable
    track-major/camera-minor order. No focal or camera state is mutated.
    """
    Ks, Rs, ts = _camera_state(camera_poses, intrinsics)
    coords, visible = _observations(image_points, len(Ks))
    world = np.asarray(world_points, dtype=np.float64)
    if world.shape != (len(coords), 3) or not np.isfinite(world).all():
        raise ValueError("world_points must contain one finite 3D point per track")
    return _pixel_residuals(coords, visible, world, Ks, Rs, ts)


def refine_calibrated_bundle(
    image_points: object,
    initial_poses: Sequence[Mapping[str, Any]],
    intrinsics: Sequence[object],
    *,
    min_tracks: int = 8,
    max_nfev: int = 160,
    robust_scale_px: float = 1.0,
) -> tuple[list[dict[str, np.ndarray]], dict[str, Any]]:
    """Jointly refine camera poses and 3D tracks at fixed calibrated K.

    Camera 0's pose and camera 1's translation are held at their initial
    values, removing global rigid-frame and similarity-scale freedoms.
    Tracks with fewer than two observations, or degenerate triangulation,
    are excluded ONCE before fitting and their original IDs are recorded.
    Invalid calibration and disconnected cameras fail closed.
    """
    if (
        isinstance(min_tracks, (bool, np.bool_))
        or not isinstance(min_tracks, (int, np.integer))
        or min_tracks < 3
    ):
        raise ValueError("min_tracks must be an integer >= 3")
    if (
        isinstance(max_nfev, (bool, np.bool_))
        or not isinstance(max_nfev, (int, np.integer))
        or max_nfev < 1
    ):
        raise ValueError("max_nfev must be a positive integer")
    if (
        isinstance(robust_scale_px, (bool, np.bool_))
        or not isinstance(robust_scale_px, (int, float, np.integer, np.floating))
    ):
        raise ValueError("robust_scale_px must be a positive finite real scalar")
    try:
        robust_scale = float(robust_scale_px)
    except (OverflowError, ValueError, TypeError) as exc:
        raise ValueError("robust_scale_px must be a positive finite real scalar") from exc
    if not np.isfinite(robust_scale) or robust_scale <= 0:
        raise ValueError("robust_scale_px must be a positive finite real scalar")

    Ks, Rs0, ts0 = _camera_state(initial_poses, intrinsics)
    coords_all, visible_all = _observations(image_points, len(Ks))
    count = len(Ks)
    selected = []
    excluded = []
    xyz_initial = []
    for track_id in range(len(coords_all)):
        if visible_all[track_id].sum() < 2:
            excluded.append(track_id)
            continue
        track_observations = [
            coords_all[track_id, camera] if visible_all[track_id, camera] else None
            for camera in range(count)
        ]
        try:
            xyz, _ = triangulate_point(
                track_observations, initial_poses, intrinsics,
                min_views=2, min_parallax_angle_deg=0.1,
            )
        except (ValueError, np.linalg.LinAlgError):
            excluded.append(track_id)
            continue
        selected.append(track_id)
        xyz_initial.append(xyz)

    if len(selected) < min_tracks:
        raise ValueError(f"bundle has only {len(selected)} usable tracks; need {min_tracks}")
    coords = coords_all[selected]
    visible = visible_all[selected]
    if np.any(visible.sum(axis=0) < 3):
        raise ValueError("each camera needs at least three admitted observations")
    world0 = np.asarray(xyz_initial, dtype=np.float64)
    num_tracks = len(selected)
    rotation_parameter_count = 3 * (count - 1)
    translation_parameter_count = 3 * (count - 2)
    point_offset = rotation_parameter_count + translation_parameter_count

    x0 = np.concatenate((
        np.concatenate([Rotation.from_matrix(R).as_rotvec() for R in Rs0[1:]]),
        np.concatenate(ts0[2:]) if count > 2 else np.empty(0),
        world0.reshape(-1),
    ))

    def unpack(params: np.ndarray):
        rotations = [Rs0[0]]
        for camera in range(1, count):
            start = 3 * (camera - 1)
            rotations.append(Rotation.from_rotvec(params[start:start + 3]).as_matrix())
        translations = [ts0[0], ts0[1]]
        for camera in range(2, count):
            start = rotation_parameter_count + 3 * (camera - 2)
            translations.append(params[start:start + 3])
        world = params[point_offset:].reshape(num_tracks, 3)
        return rotations, translations, world

    def objective(params: np.ndarray) -> np.ndarray:
        rotations, translations, world = unpack(params)
        return _pixel_residuals(coords, visible, world, Ks, rotations, translations)

    # Each observation depends only on its 3-D track and its own camera pose.
    # Sparse Jacobians avoid O(parameter_count) dense finite differences.
    observation_indices = np.argwhere(visible)
    jac = lil_matrix((2 * len(observation_indices), len(x0)), dtype=np.int8)
    for index, (track, camera) in enumerate(observation_indices):
        rows = slice(2 * index, 2 * index + 2)
        jac[rows, point_offset + 3 * track:point_offset + 3 * track + 3] = 1
        if camera > 0:
            start = 3 * (camera - 1)
            jac[rows, start:start + 3] = 1
        if camera > 1:
            start = rotation_parameter_count + 3 * (camera - 2)
            jac[rows, start:start + 3] = 1

    initial_residual = objective(x0)
    result = least_squares(
        objective, x0, method="trf", jac_sparsity=jac.tocsr(),
        loss="soft_l1", f_scale=robust_scale,
        max_nfev=int(max_nfev), x_scale="jac",
        ftol=1e-7, xtol=1e-7, gtol=1e-7,
    )
    if not result.success or not np.isfinite(result.x).all():
        last_residual = objective(result.x)
        raise ValueError(
            f"bundle adjustment did not converge: {result.message}; "
            f"nfev={result.nfev}, initial_rms="
            f"{np.sqrt(np.mean(initial_residual ** 2)):.6g}, final_rms="
            f"{np.sqrt(np.mean(last_residual ** 2)):.6g}, "
            f"optimality={result.optimality:.6g}"
        )
    optimized_rotations, optimized_translations, optimized_world = unpack(result.x)
    for track, camera in observation_indices:
        depth = float(optimized_rotations[camera][2] @ optimized_world[track] + optimized_translations[camera][2])
        if not np.isfinite(depth) or depth <= 1e-9:
            raise ValueError("bundle adjustment produced a point behind a camera")
    final_residual = objective(result.x)
    if not np.isfinite(final_residual).all():
        raise ValueError("bundle adjustment produced nonfinite residuals")
    refined_poses = [
        {"R": R.copy(), "t": t.copy()}
        for R, t in zip(optimized_rotations, optimized_translations)
    ]
    return refined_poses, {
        "initial_reprojection_rms_px": float(np.sqrt(np.mean(initial_residual ** 2))),
        "reprojection_rms_px": float(np.sqrt(np.mean(final_residual ** 2))),
        "signed_residual_count": len(final_residual),
        "admitted_track_indices": selected,
        "excluded_track_indices": excluded,
        "world_points": optimized_world.copy(),
        "camera_zero_pose_fixed": True,
        "camera_one_translation_fixed": True,
        "baseline_gauge_length": float(np.linalg.norm(ts0[1])),
        "intrinsics_fixed": True,
        "translation_scale_known": False,
        "function_evaluations": int(result.nfev),
    }
