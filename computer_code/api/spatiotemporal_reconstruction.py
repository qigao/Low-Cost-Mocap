"""Physical-camera-ID-safe geometry following synchronized marker admission.

Markers are already decoded and assigned to the exact same-trigger batch.
Use explicit physical camera ID -> pose/intrinsics mappings, never assume
that a sparse set of camera IDs can index dense calibration arrays.
Hardware exposure timing, metric scale and live flight remain unverified.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import numpy as np

from geometry import triangulate_point
from rigid_pattern import fit_rigid_pattern
from temporal_pattern import GroupedMarkers, SynchronizedBatch, _nonnegative_int


@dataclass(frozen=True)
class MarkerGeometry:
    marker_id: int
    xyz: tuple[float, float, float]
    camera_ids: tuple[int, ...]
    reprojection_rms_px: float
    reprojection_errors_px_by_camera: Mapping[int, float]
    depths_by_camera: Mapping[int, float]


@dataclass(frozen=True)
class SynchronizedGeometry:
    trigger_id: int
    camera_ids: tuple[int, ...]
    markers: Mapping[int, MarkerGeometry]
    excluded_marker_ids: tuple[int, ...]


@dataclass(frozen=True)
class SynchronizedRigidPattern:
    trigger_id: int
    rotation: np.ndarray
    translation: np.ndarray
    marker_ids: tuple[int, ...]
    reprojection_max_rms_px: float
    rigid_rms_error: float
    metric_scale_validated: bool = False


def _ordered_camera_calibration(
    values: Mapping[int, Any],
    camera_ids: tuple[int, ...],
    name: str,
) -> list[Any]:
    if not isinstance(values, Mapping):
        raise ValueError(f"{name} must map physical camera IDs to calibration")
    for raw in values:
        _nonnegative_int(raw, f"{name} camera_id")
    if set(values) != set(camera_ids):
        raise ValueError(f"{name} must cover exactly the synchronized physical camera IDs")
    return [values[camera_id] for camera_id in camera_ids]


def reconstruct_synchronized_markers(
    batch: SynchronizedBatch,
    grouped: GroupedMarkers,
    camera_poses_by_id: Mapping[int, Any],
    intrinsics_by_id: Mapping[int, Any],
    *,
    min_views: int = 2,
    max_reprojection_rms_px: float = 1.0,
    min_parallax_angle_deg: float = 0.1,
) -> SynchronizedGeometry:
    """Reconstruct all admitted markers or fail the *entire* frame.

    Observation tuples from group_synchronized_detections follow
    batch.camera_ids. Reprojection error/depth maps use those physical
    camera IDs, not the dense 0..N-1 slots used inside triangulate_point.
    """
    if not isinstance(batch, SynchronizedBatch) or not isinstance(grouped, GroupedMarkers):
        raise ValueError("expected validated synchronized batch and grouped markers")
    cameras = tuple(_nonnegative_int(v, "camera_id") for v in batch.camera_ids)
    if len(cameras) < 2 or len(set(cameras)) != len(cameras):
        raise ValueError("synchronized batch camera IDs must be unique and complete")
    trigger = _nonnegative_int(batch.trigger_id, "trigger_id")
    poses = _ordered_camera_calibration(camera_poses_by_id, cameras, "camera_poses_by_id")
    intrinsics = _ordered_camera_calibration(intrinsics_by_id, cameras, "intrinsics_by_id")
    if not isinstance(min_views, int) or isinstance(min_views, bool) or not 2 <= min_views <= len(cameras):
        raise ValueError("min_views must be an integer within the camera count")
    if not grouped.observations_by_marker_id:
        raise ValueError("no admitted synchronized markers to reconstruct")

    output = {}
    for raw_id, points in grouped.observations_by_marker_id.items():
        marker_id = _nonnegative_int(raw_id, "marker_id")
        if marker_id in output:
            raise ValueError("duplicate admitted marker_id")
        if not isinstance(points, (tuple, list)) or len(points) != len(cameras):
            raise ValueError("marker camera observations do not match synchronized camera_ids")
        xyz, metrics = triangulate_point(
            points, poses, intrinsics,
            min_views=min_views,
            max_reprojection_rms_px=max_reprojection_rms_px,
            min_parallax_angle_deg=min_parallax_angle_deg,
        )
        sources = tuple(cameras[dense_id] for dense_id in metrics["camera_ids"])
        errors = {
            cameras[dense_id]: float(value)
            for dense_id, value in metrics["reprojection_errors_px"].items()
        }
        depths = {
            cameras[dense_id]: float(value)
            for dense_id, value in metrics["depths"].items()
        }
        output[marker_id] = MarkerGeometry(
            marker_id=marker_id,
            xyz=tuple(float(v) for v in xyz),
            camera_ids=sources,
            reprojection_rms_px=float(metrics["reprojection_rms_px"]),
            reprojection_errors_px_by_camera=MappingProxyType(errors),
            depths_by_camera=MappingProxyType(depths),
        )
    excluded = tuple(_nonnegative_int(v, "excluded marker_id") for v in grouped.excluded_marker_ids)
    if set(excluded) & set(output):
        raise ValueError("excluded marker ID also appears in admitted geometry")
    return SynchronizedGeometry(
        trigger_id=trigger,
        camera_ids=cameras,
        markers=MappingProxyType(output),
        excluded_marker_ids=excluded,
    )


def fit_synchronized_rigid_pattern(
    reconstruction: SynchronizedGeometry,
    model_points_by_marker_id: Mapping[int, Any],
    *,
    min_markers: int = 3,
    max_rms_error: float = 0.02,
) -> SynchronizedRigidPattern:
    """Validate the known rigid layout; never infer metric scale certainty.

    Only reconstructed IDs present in the known model may be fitted.
    No similarity scaling is performed. The returned scale flag remains
    false until an external measured-distance/hardware gate is satisfied.
    """
    if not isinstance(reconstruction, SynchronizedGeometry):
        raise ValueError("reconstruction must be SynchronizedGeometry")
    if not isinstance(model_points_by_marker_id, Mapping):
        raise ValueError("model_points_by_marker_id must map marker IDs to 3D points")
    model_ids = sorted(
        _nonnegative_int(v, "model marker_id") for v in model_points_by_marker_id
    )
    observed_ids = sorted(reconstruction.markers)
    if not set(observed_ids).issubset(set(model_ids)):
        raise ValueError("reconstructed marker ID is absent from rigid pattern model")
    model_index = {v: i for i, v in enumerate(model_ids)}
    model = [model_points_by_marker_id[v] for v in model_ids]
    observed = [reconstruction.markers[v].xyz for v in observed_ids]
    R, t, metrics = fit_rigid_pattern(
        model, observed, marker_indices=[model_index[v] for v in observed_ids],
        min_markers=min_markers, max_rms_error=max_rms_error,
    )
    return SynchronizedRigidPattern(
        trigger_id=reconstruction.trigger_id,
        rotation=R.copy(), translation=t.copy(),
        marker_ids=tuple(observed_ids),
        reprojection_max_rms_px=max(
            reconstruction.markers[v].reprojection_rms_px for v in observed_ids
        ),
        rigid_rms_error=float(metrics["rms_error"]),
        metric_scale_validated=False,
    )
