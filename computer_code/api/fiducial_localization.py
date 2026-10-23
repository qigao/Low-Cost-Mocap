"""Compose a robust fiducial camera pose into a qualified body localization observation.

This layer owns frame transforms and the software uncertainty/calibration
contract. It does not perform temporal fusion or claim hardware qualification.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
import json
import math
from numbers import Real

import numpy as np
from scipy.spatial.transform import Rotation

from fiducial_contract import FiducialTagMap
from fiducial_multitag_pose import MultiTagCameraPose
from localization_contract import (
    LocalizationQuality,
    LocalizationTopology,
    MapPoseObservation,
    PoseSource,
    ScaleReference,
)


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


def _vector(value: object, length: int, name: str, *, nonnegative: bool = False) -> tuple[float, ...]:
    if not isinstance(value, (tuple, list)) or len(value) != length:
        raise ValueError(f"{name} must contain exactly {length} values")
    result = tuple(_real(item, f"{name}[{index}]") for index, item in enumerate(value))
    if nonnegative and any(item < 0.0 for item in result):
        raise ValueError(f"{name} values must be nonnegative")
    return result


def _quaternion(value: object, name: str) -> tuple[float, float, float, float]:
    quaternion = np.asarray(_vector(value, 4, name), dtype=np.float64)
    norm = float(np.linalg.norm(quaternion))
    if not math.isfinite(norm) or norm <= 1e-12:
        raise ValueError(f"{name} quaternion norm is zero")
    quaternion /= norm
    if quaternion[3] < 0.0:
        quaternion = -quaternion
    return tuple(float(item) for item in quaternion)


def _canonical_digest(payload: dict[str, object]) -> str:
    raw = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return sha256(raw).hexdigest()


@dataclass(frozen=True)
class CameraBodyExtrinsic:
    """Rigid body-from-camera mounting calibration.

    X_body = R_body_from_camera @ X_camera + t_body_from_camera
    """

    camera_id: int
    camera_serial: str
    camera_frame_id: str
    body_frame_id: str
    translation_body_from_camera_m: tuple[float, float, float]
    orientation_body_from_camera_xyzw: tuple[float, float, float, float]
    reference_method: str
    revision_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        _integer(self.camera_id, "camera_id")
        object.__setattr__(self, "camera_serial", _text(self.camera_serial, "camera_serial"))
        object.__setattr__(self, "camera_frame_id", _text(self.camera_frame_id, "camera_frame_id", 128))
        object.__setattr__(self, "body_frame_id", _text(self.body_frame_id, "body_frame_id", 128))
        if self.camera_frame_id == self.body_frame_id:
            raise ValueError("camera_frame_id and body_frame_id must differ")
        object.__setattr__(
            self,
            "translation_body_from_camera_m",
            _vector(self.translation_body_from_camera_m, 3, "translation_body_from_camera_m"),
        )
        object.__setattr__(
            self,
            "orientation_body_from_camera_xyzw",
            _quaternion(self.orientation_body_from_camera_xyzw, "orientation_body_from_camera_xyzw"),
        )
        method = _text(self.reference_method, "reference_method", 32)
        if method not in {"calibrated", "surveyed"}:
            raise ValueError("reference_method must be calibrated or surveyed")
        object.__setattr__(self, "reference_method", method)
        object.__setattr__(
            self,
            "revision_sha256",
            _canonical_digest({
                "camera_id": self.camera_id,
                "camera_serial": self.camera_serial,
                "camera_frame_id": self.camera_frame_id,
                "body_frame_id": self.body_frame_id,
                "translation_body_from_camera_m": list(self.translation_body_from_camera_m),
                "orientation_body_from_camera_xyzw": list(self.orientation_body_from_camera_xyzw),
                "reference_method": self.reference_method,
            }),
        )


@dataclass(frozen=True)
class FiducialUncertaintyModel:
    """Versioned software model for MapPoseObservation standard deviations.

    The base terms are deployment/calibration inputs and must include the
    accepted single-tag/intrinsic solve uncertainty. Multi-tag consensus
    residuals and anchor-map uncertainty are then added in quadrature.
    This is a software uncertainty model, not independent hardware evidence.
    """

    model_id: str
    base_position_stddev_m: tuple[float, float, float]
    base_orientation_stddev_rad: tuple[float, float, float]
    translation_residual_gain: float = 1.0
    orientation_residual_gain: float = 1.0
    revision_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "model_id", _text(self.model_id, "model_id", 128))
        object.__setattr__(
            self,
            "base_position_stddev_m",
            _vector(self.base_position_stddev_m, 3, "base_position_stddev_m", nonnegative=True),
        )
        object.__setattr__(
            self,
            "base_orientation_stddev_rad",
            _vector(self.base_orientation_stddev_rad, 3, "base_orientation_stddev_rad", nonnegative=True),
        )
        object.__setattr__(
            self,
            "translation_residual_gain",
            _real(self.translation_residual_gain, "translation_residual_gain", minimum=0.0),
        )
        object.__setattr__(
            self,
            "orientation_residual_gain",
            _real(self.orientation_residual_gain, "orientation_residual_gain", minimum=0.0),
        )
        object.__setattr__(
            self,
            "revision_sha256",
            _canonical_digest({
                "model_id": self.model_id,
                "base_position_stddev_m": list(self.base_position_stddev_m),
                "base_orientation_stddev_rad": list(self.base_orientation_stddev_rad),
                "translation_residual_gain": self.translation_residual_gain,
                "orientation_residual_gain": self.orientation_residual_gain,
            }),
        )

    def estimate(
        self,
        camera_pose: MultiTagCameraPose,
        tag_map: FiducialTagMap,
    ) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        if not isinstance(camera_pose, MultiTagCameraPose) or not isinstance(tag_map, FiducialTagMap):
            raise ValueError("camera_pose/tag_map has invalid type")
        anchors = [
            tag_map.by_key[key]
            for key in camera_pose.inlier_tag_keys
        ]
        if not anchors:
            raise ValueError("uncertainty estimate requires inlier tag anchors")
        anchor_position = np.max(
            np.asarray([anchor.position_stddev_m for anchor in anchors], dtype=np.float64),
            axis=0,
        )
        anchor_orientation = np.max(
            np.asarray([anchor.orientation_stddev_rad for anchor in anchors], dtype=np.float64),
            axis=0,
        )
        translation_term = (
            self.translation_residual_gain
            * camera_pose.max_translation_residual_m
        )
        orientation_term = (
            self.orientation_residual_gain
            * camera_pose.max_orientation_residual_rad
        )
        position = np.sqrt(
            np.square(self.base_position_stddev_m)
            + np.square(anchor_position)
            + translation_term * translation_term
        )
        orientation = np.sqrt(
            np.square(self.base_orientation_stddev_rad)
            + np.square(anchor_orientation)
            + orientation_term * orientation_term
        )
        if not (np.isfinite(position).all() and np.isfinite(orientation).all()):
            raise ValueError("uncertainty model produced non-finite values")
        return (
            tuple(float(value) for value in position),
            tuple(float(value) for value in orientation),
        )


def localization_calibration_sha256(
    camera_calibration_sha256: str,
    extrinsic: CameraBodyExtrinsic,
    uncertainty_model: FiducialUncertaintyModel,
) -> str:
    if not isinstance(extrinsic, CameraBodyExtrinsic):
        raise ValueError("extrinsic must be CameraBodyExtrinsic")
    if not isinstance(uncertainty_model, FiducialUncertaintyModel):
        raise ValueError("uncertainty_model must be FiducialUncertaintyModel")
    camera_digest = _digest(camera_calibration_sha256, "camera_calibration_sha256")
    return _canonical_digest({
        "camera_calibration_sha256": camera_digest,
        "camera_body_extrinsic_sha256": extrinsic.revision_sha256,
        "uncertainty_model_sha256": uncertainty_model.revision_sha256,
    })


def map_body_observation_from_fiducials(
    camera_pose: MultiTagCameraPose,
    tag_map: FiducialTagMap,
    extrinsic: CameraBodyExtrinsic,
    uncertainty_model: FiducialUncertaintyModel,
    *,
    relocalized: bool = False,
    tracking_min_tags: int = 3,
    max_tracking_translation_residual_m: float = 0.03,
    max_tracking_orientation_residual_rad: float = math.radians(3.0),
) -> MapPoseObservation:
    """Compose map_from_camera with camera/body mounting into MapPoseObservation."""
    if not isinstance(camera_pose, MultiTagCameraPose):
        raise ValueError("camera_pose must be MultiTagCameraPose")
    if not isinstance(tag_map, FiducialTagMap):
        raise ValueError("tag_map must be FiducialTagMap")
    if not isinstance(extrinsic, CameraBodyExtrinsic):
        raise ValueError("extrinsic must be CameraBodyExtrinsic")
    if not isinstance(uncertainty_model, FiducialUncertaintyModel):
        raise ValueError("uncertainty_model must be FiducialUncertaintyModel")
    if not isinstance(relocalized, bool):
        raise ValueError("relocalized must be boolean")

    tracking_min_tags = _integer(tracking_min_tags, "tracking_min_tags", minimum=2)
    max_tracking_translation_residual_m = _real(
        max_tracking_translation_residual_m,
        "max_tracking_translation_residual_m",
        minimum=0.0,
    )
    max_tracking_orientation_residual_rad = _real(
        max_tracking_orientation_residual_rad,
        "max_tracking_orientation_residual_rad",
        minimum=0.0,
    )
    if max_tracking_orientation_residual_rad > math.pi:
        raise ValueError("max_tracking_orientation_residual_rad must be <= pi")

    if camera_pose.map_frame_id != tag_map.map_frame_id:
        raise ValueError("camera pose map frame does not match active tag map")
    if camera_pose.map_revision_sha256 != tag_map.revision_sha256:
        raise ValueError("camera pose map revision does not match active tag map")
    if (
        camera_pose.camera_id != extrinsic.camera_id
        or camera_pose.camera_serial != extrinsic.camera_serial
    ):
        raise ValueError("camera pose identity does not match camera/body extrinsic")
    if any(key not in tag_map.by_key for key in camera_pose.inlier_tag_keys):
        raise ValueError("camera pose inliers are not present in active tag map")

    rotation_map_from_camera = Rotation.from_quat(
        camera_pose.orientation_map_from_camera_xyzw
    ).as_matrix()
    translation_map_from_camera = np.asarray(
        camera_pose.position_map_from_camera_m, dtype=np.float64
    )
    rotation_body_from_camera = Rotation.from_quat(
        extrinsic.orientation_body_from_camera_xyzw
    ).as_matrix()
    translation_body_from_camera = np.asarray(
        extrinsic.translation_body_from_camera_m, dtype=np.float64
    )

    rotation_map_from_body = (
        rotation_map_from_camera @ rotation_body_from_camera.T
    )
    translation_map_from_body = (
        translation_map_from_camera
        - rotation_map_from_body @ translation_body_from_camera
    )
    if not (
        np.isfinite(rotation_map_from_body).all()
        and np.isfinite(translation_map_from_body).all()
    ):
        raise ValueError("camera/body composition produced non-finite pose")

    position_stddev, orientation_stddev = uncertainty_model.estimate(
        camera_pose, tag_map
    )
    calibration_digest = localization_calibration_sha256(
        camera_pose.calibration_sha256,
        extrinsic,
        uncertainty_model,
    )

    inlier_anchors = [
        tag_map.by_key[key] for key in camera_pose.inlier_tag_keys
    ]
    if all(anchor.reference_method == "surveyed" for anchor in inlier_anchors):
        scale_reference = ScaleReference.SURVEYED_TAGS
    else:
        scale_reference = ScaleReference.CALIBRATED_TAG_MAP

    quality = LocalizationQuality.TRACKING
    if (
        len(camera_pose.inlier_tag_keys) < tracking_min_tags
        or camera_pose.max_translation_residual_m
        > max_tracking_translation_residual_m
        or camera_pose.max_orientation_residual_rad
        > max_tracking_orientation_residual_rad
    ):
        quality = LocalizationQuality.DEGRADED

    quaternion = Rotation.from_matrix(rotation_map_from_body).as_quat()
    if quaternion[3] < 0.0:
        quaternion = -quaternion

    return MapPoseObservation(
        source=PoseSource.FIDUCIAL_LOCALIZATION,
        topology=LocalizationTopology.INSIDE_OUT,
        tracked_frame_id=extrinsic.body_frame_id,
        map_frame_id=tag_map.map_frame_id,
        timestamp_ns=camera_pose.exposure_timestamp_ns,
        clock_domain=camera_pose.clock_domain,
        position_xyz=tuple(float(value) for value in translation_map_from_body),
        position_unit="meters",
        orientation_xyzw=tuple(float(value) for value in quaternion),
        position_stddev=position_stddev,
        orientation_stddev_rad=orientation_stddev,
        map_revision_sha256=tag_map.revision_sha256,
        calibration_sha256=calibration_digest,
        quality=quality,
        inlier_count=len(camera_pose.inlier_tag_keys),
        metric_scale_qualified=True,
        scale_reference=scale_reference,
        relocalized=relocalized,
        reprojection_rms_px=camera_pose.mean_single_tag_reprojection_rms_px,
    )
