"""Robust multi-tag consensus for a map-frame camera pose.

Each admitted single-tag measurement independently implies map_from_camera:

    map_from_camera = map_from_tag @ tag_from_camera

Consensus happens in pose space, not by averaging image corners across unrelated
planar tags. A minority inconsistent tag may be excluded, but equal-support
competing pose clusters fail closed.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real
from typing import Sequence

import numpy as np
from scipy.spatial.transform import Rotation

from fiducial_contract import FiducialTagMap
from fiducial_pose import FiducialTagPoseMeasurement


def _text(value: object, name: str, limit: int = 256) -> str:
    if not isinstance(value, str) or not value.strip() or len(value.strip()) > limit:
        raise ValueError(f"{name} must be a nonempty string of at most {limit} characters")
    return value.strip()


def _integer(value: object, name: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _real(value: object, name: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real scalar")
    try:
        result = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite real scalar") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite real scalar")
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return result


def _digest(value: object, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _vector(value: object, length: int, name: str) -> tuple[float, ...]:
    if not isinstance(value, (tuple, list)) or len(value) != length:
        raise ValueError(f"{name} must contain exactly {length} values")
    return tuple(_real(item, f"{name}[{index}]") for index, item in enumerate(value))


def _rotation_distance_rad(first: np.ndarray, second: np.ndarray) -> float:
    relative = first.T @ second
    cosine = max(-1.0, min(1.0, (float(np.trace(relative)) - 1.0) * 0.5))
    return math.acos(cosine)


def _canonical_quaternion(rotation: np.ndarray) -> tuple[float, float, float, float]:
    quaternion = Rotation.from_matrix(rotation).as_quat()
    if quaternion[3] < 0.0:
        quaternion = -quaternion
    return tuple(float(value) for value in quaternion)


def _average_rotation(rotations: Sequence[np.ndarray]) -> np.ndarray:
    if not rotations:
        raise ValueError("cannot average an empty rotation set")
    accumulator = np.zeros((4, 4), dtype=np.float64)
    for rotation in rotations:
        quaternion = Rotation.from_matrix(rotation).as_quat()
        accumulator += np.outer(quaternion, quaternion)
    eigenvalues, eigenvectors = np.linalg.eigh(accumulator)
    quaternion = eigenvectors[:, int(np.argmax(eigenvalues))]
    if quaternion[3] < 0.0:
        quaternion = -quaternion
    result = Rotation.from_quat(quaternion).as_matrix()
    if not np.isfinite(result).all():
        raise ValueError("rotation consensus produced non-finite values")
    return result


def _map_from_camera_candidate(
    measurement: FiducialTagPoseMeasurement,
    tag_map: FiducialTagMap,
) -> tuple[np.ndarray, np.ndarray]:
    anchor = tag_map.by_key[(measurement.tag_family, measurement.tag_id)]
    rotation_map_from_tag = Rotation.from_quat(
        anchor.orientation_map_from_tag_xyzw
    ).as_matrix()
    translation_map_from_tag = np.asarray(anchor.position_xyz_m, dtype=np.float64)

    rotation_camera_from_tag = Rotation.from_quat(
        measurement.orientation_camera_from_tag_xyzw
    ).as_matrix()
    translation_camera_from_tag = np.asarray(
        measurement.translation_camera_from_tag_m, dtype=np.float64
    )

    rotation_map_from_camera = (
        rotation_map_from_tag @ rotation_camera_from_tag.T
    )
    translation_map_from_camera = (
        translation_map_from_tag
        - rotation_map_from_camera @ translation_camera_from_tag
    )
    if not (
        np.isfinite(rotation_map_from_camera).all()
        and np.isfinite(translation_map_from_camera).all()
    ):
        raise ValueError("tag/map transform produced non-finite camera pose")
    return rotation_map_from_camera, translation_map_from_camera


def _aggregate(
    candidates: Sequence[tuple[np.ndarray, np.ndarray]],
    support: Sequence[int],
) -> tuple[np.ndarray, np.ndarray]:
    rotations = [candidates[index][0] for index in support]
    translations = np.asarray(
        [candidates[index][1] for index in support], dtype=np.float64
    )
    return _average_rotation(rotations), np.mean(translations, axis=0)


@dataclass(frozen=True)
class MultiTagCameraPose:
    camera_id: int
    camera_serial: str
    frame_id: int
    trigger_id: int
    exposure_timestamp_ns: int
    clock_domain: str
    map_frame_id: str
    position_map_from_camera_m: tuple[float, float, float]
    orientation_map_from_camera_xyzw: tuple[float, float, float, float]
    inlier_tag_keys: tuple[tuple[str, int], ...]
    excluded_tag_keys: tuple[tuple[str, int], ...]
    max_translation_residual_m: float
    max_orientation_residual_rad: float
    mean_single_tag_reprojection_rms_px: float
    calibration_sha256: str
    map_revision_sha256: str

    def __post_init__(self) -> None:
        for name in ("camera_id", "frame_id", "trigger_id", "exposure_timestamp_ns"):
            _integer(getattr(self, name), name)
        object.__setattr__(self, "camera_serial", _text(self.camera_serial, "camera_serial"))
        object.__setattr__(self, "clock_domain", _text(self.clock_domain, "clock_domain"))
        object.__setattr__(self, "map_frame_id", _text(self.map_frame_id, "map_frame_id"))
        object.__setattr__(
            self,
            "position_map_from_camera_m",
            _vector(self.position_map_from_camera_m, 3, "position_map_from_camera_m"),
        )
        quaternion = np.asarray(
            _vector(
                self.orientation_map_from_camera_xyzw,
                4,
                "orientation_map_from_camera_xyzw",
            )
        )
        norm = float(np.linalg.norm(quaternion))
        if not math.isfinite(norm) or norm <= 1e-12:
            raise ValueError("orientation_map_from_camera_xyzw quaternion norm is zero")
        quaternion /= norm
        if quaternion[3] < 0.0:
            quaternion = -quaternion
        object.__setattr__(
            self,
            "orientation_map_from_camera_xyzw",
            tuple(float(value) for value in quaternion),
        )
        for field_name in (
            "max_translation_residual_m",
            "max_orientation_residual_rad",
            "mean_single_tag_reprojection_rms_px",
        ):
            object.__setattr__(
                self,
                field_name,
                _real(getattr(self, field_name), field_name, minimum=0.0),
            )
        if not isinstance(self.inlier_tag_keys, tuple) or not self.inlier_tag_keys:
            raise ValueError("inlier_tag_keys must be a nonempty tuple")
        if not isinstance(self.excluded_tag_keys, tuple):
            raise ValueError("excluded_tag_keys must be a tuple")
        all_keys = self.inlier_tag_keys + self.excluded_tag_keys
        if any(
            not isinstance(key, tuple)
            or len(key) != 2
            or not isinstance(key[0], str)
            or type(key[1]) is not int
            for key in all_keys
        ):
            raise ValueError("tag keys must be (family, integer ID) tuples")
        if len(set(all_keys)) != len(all_keys):
            raise ValueError("tag keys must be unique")
        if tuple(sorted(self.inlier_tag_keys)) != self.inlier_tag_keys:
            raise ValueError("inlier_tag_keys must use canonical ordering")
        if tuple(sorted(self.excluded_tag_keys)) != self.excluded_tag_keys:
            raise ValueError("excluded_tag_keys must use canonical ordering")
        _digest(self.calibration_sha256, "calibration_sha256")
        _digest(self.map_revision_sha256, "map_revision_sha256")


def solve_multi_tag_camera_pose(
    measurements: Sequence[FiducialTagPoseMeasurement],
    tag_map: FiducialTagMap,
    *,
    min_tags: int = 2,
    max_translation_disagreement_m: float = 0.08,
    max_orientation_disagreement_rad: float = math.radians(8.0),
    max_single_tag_reprojection_rms_px: float = 2.0,
) -> MultiTagCameraPose:
    """Robustly combine single-tag measurements into map_from_camera.

    The largest mutually compatible pose-support set wins. Equal-size distinct
    maximum-support sets are treated as an unresolved pose ambiguity.
    """
    if not isinstance(tag_map, FiducialTagMap):
        raise ValueError("tag_map must be FiducialTagMap")
    if (
        not isinstance(measurements, Sequence)
        or isinstance(measurements, (str, bytes))
        or not measurements
    ):
        raise ValueError("measurements must be a nonempty sequence")
    if any(not isinstance(item, FiducialTagPoseMeasurement) for item in measurements):
        raise ValueError("measurements must contain FiducialTagPoseMeasurement records")

    min_tags = _integer(min_tags, "min_tags", minimum=2)
    max_translation_disagreement_m = _real(
        max_translation_disagreement_m,
        "max_translation_disagreement_m",
        minimum=0.0,
    )
    max_orientation_disagreement_rad = _real(
        max_orientation_disagreement_rad,
        "max_orientation_disagreement_rad",
        minimum=0.0,
    )
    if max_orientation_disagreement_rad > math.pi:
        raise ValueError("max_orientation_disagreement_rad must be <= pi")
    max_single_tag_reprojection_rms_px = _real(
        max_single_tag_reprojection_rms_px,
        "max_single_tag_reprojection_rms_px",
        minimum=0.0,
    )

    first = measurements[0]
    metadata = (
        first.camera_id,
        first.camera_serial,
        first.frame_id,
        first.trigger_id,
        first.exposure_timestamp_ns,
        first.clock_domain,
        first.calibration_sha256,
        first.map_revision_sha256,
    )
    if first.map_revision_sha256 != tag_map.revision_sha256:
        raise ValueError("single-tag measurement map revision does not match active map")

    keys = []
    usable_measurements = []
    preexcluded_keys = []
    for measurement in measurements:
        current_metadata = (
            measurement.camera_id,
            measurement.camera_serial,
            measurement.frame_id,
            measurement.trigger_id,
            measurement.exposure_timestamp_ns,
            measurement.clock_domain,
            measurement.calibration_sha256,
            measurement.map_revision_sha256,
        )
        if current_metadata != metadata:
            raise ValueError("multi-tag measurements mix frame/camera/time/revision metadata")
        key = (measurement.tag_family, measurement.tag_id)
        if key not in tag_map.by_key:
            raise ValueError("single-tag measurement references a tag outside the active map")
        if key in keys:
            raise ValueError("multi-tag measurements contain duplicate tag identity")
        keys.append(key)
        if measurement.reprojection_rms_px > max_single_tag_reprojection_rms_px:
            preexcluded_keys.append(key)
        else:
            usable_measurements.append(measurement)

    if len(usable_measurements) < min_tags:
        raise ValueError("insufficient usable single-tag measurements for consensus")

    candidate_keys = [
        (measurement.tag_family, measurement.tag_id)
        for measurement in usable_measurements
    ]
    candidates = [
        _map_from_camera_candidate(measurement, tag_map)
        for measurement in usable_measurements
    ]

    def compatible(first_index: int, second_index: int) -> bool:
        first_rotation, first_translation = candidates[first_index]
        second_rotation, second_translation = candidates[second_index]
        translation_error = float(
            np.linalg.norm(first_translation - second_translation)
        )
        orientation_error = _rotation_distance_rad(
            first_rotation, second_rotation
        )
        return (
            translation_error <= max_translation_disagreement_m
            and orientation_error <= max_orientation_disagreement_rad
        )

    supports = []
    seen_supports = set()
    for seed_index in range(len(candidates)):
        support = tuple(
            index
            for index in range(len(candidates))
            if compatible(seed_index, index)
        )
        if support not in seen_supports:
            supports.append(support)
            seen_supports.add(support)

    maximum_support = max(len(support) for support in supports)
    if maximum_support < min_tags:
        raise ValueError("no multi-tag pose consensus has sufficient support")
    best_supports = [
        support for support in supports if len(support) == maximum_support
    ]
    if len(best_supports) != 1:
        raise ValueError("ambiguous competing multi-tag pose clusters have equal support")

    support = best_supports[0]
    # Refit and re-evaluate against the aggregate pose. This catches a
    # non-transitive pairwise chain where the seed was close to each member
    # but the final average is not actually supported by all of them.
    for _ in range(3):
        rotation, translation = _aggregate(candidates, support)
        updated = tuple(
            index
            for index, (candidate_rotation, candidate_translation) in enumerate(candidates)
            if (
                float(np.linalg.norm(candidate_translation - translation))
                <= max_translation_disagreement_m
                and _rotation_distance_rad(candidate_rotation, rotation)
                <= max_orientation_disagreement_rad
            )
        )
        if len(updated) < min_tags:
            raise ValueError("refitted multi-tag pose lost required consensus support")
        if updated == support:
            break
        support = updated
    else:
        raise ValueError("multi-tag pose consensus did not stabilize")

    rotation, translation = _aggregate(candidates, support)
    translation_residuals = [
        float(np.linalg.norm(candidates[index][1] - translation))
        for index in support
    ]
    orientation_residuals = [
        _rotation_distance_rad(candidates[index][0], rotation)
        for index in support
    ]
    inlier_keys = tuple(sorted(candidate_keys[index] for index in support))
    excluded = set(preexcluded_keys)
    excluded.update(
        candidate_keys[index]
        for index in range(len(candidate_keys))
        if index not in support
    )
    excluded_keys = tuple(sorted(excluded))
    mean_reprojection = float(np.mean([
        usable_measurements[index].reprojection_rms_px
        for index in support
    ]))

    return MultiTagCameraPose(
        camera_id=first.camera_id,
        camera_serial=first.camera_serial,
        frame_id=first.frame_id,
        trigger_id=first.trigger_id,
        exposure_timestamp_ns=first.exposure_timestamp_ns,
        clock_domain=first.clock_domain,
        map_frame_id=tag_map.map_frame_id,
        position_map_from_camera_m=tuple(float(value) for value in translation),
        orientation_map_from_camera_xyzw=_canonical_quaternion(rotation),
        inlier_tag_keys=inlier_keys,
        excluded_tag_keys=excluded_keys,
        max_translation_residual_m=max(translation_residuals),
        max_orientation_residual_rad=max(orientation_residuals),
        mean_single_tag_reprojection_rms_px=mean_reprojection,
        calibration_sha256=first.calibration_sha256,
        map_revision_sha256=first.map_revision_sha256,
    )
