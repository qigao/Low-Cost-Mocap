"""Provider-neutral contracts for standardized indoor tags and surveyed outdoor anchors.

This module validates detector observations and map metadata. It does not implement
a detector, solvePnP, or claim physical timing/map provenance.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
import json
import math
from numbers import Real
from types import MappingProxyType
from typing import Mapping


def _int(value: object, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _real(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be finite")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _text(value: object, name: str, limit: int = 256) -> str:
    if not isinstance(value, str) or not value.strip() or len(value.strip()) > limit:
        raise ValueError(f"{name} must be a nonempty string of at most {limit} characters")
    return value.strip()


def _digest(value: object, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _vector(value: object, length: int, name: str, *, nonnegative: bool = False) -> tuple[float, ...]:
    if not isinstance(value, (tuple, list)) or len(value) != length:
        raise ValueError(f"{name} must contain exactly {length} values")
    result = tuple(_real(v, f"{name}[{i}]") for i, v in enumerate(value))
    if nonnegative and any(v < 0.0 for v in result):
        raise ValueError(f"{name} values must be nonnegative")
    return result


def _quaternion(value: object, name: str) -> tuple[float, float, float, float]:
    q = _vector(value, 4, name)
    norm = math.sqrt(sum(v * v for v in q))
    if norm < 1e-12:
        raise ValueError(f"{name} quaternion norm is zero")
    q = tuple(v / norm for v in q)
    # Canonicalize q and -q so the map digest is deterministic.
    flip = q[3] < 0.0 or (q[3] == 0.0 and next((v for v in q[:3] if v), 1.0) < 0.0)
    if flip:
        q = tuple(-v for v in q)
    return q


def _corners(value: object) -> tuple[tuple[float, float], ...]:
    if not isinstance(value, (tuple, list)) or len(value) != 4:
        raise ValueError("image_corners_px must contain four ordered corners")
    points = tuple(_vector(p, 2, f"image_corners_px[{i}]") for i, p in enumerate(value))
    if len(set(points)) != 4:
        raise ValueError("image_corners_px must contain four distinct corners")
    crosses = []
    for i in range(4):
        ax, ay = points[i]
        bx, by = points[(i + 1) % 4]
        cx, cy = points[(i + 2) % 4]
        crosses.append((bx - ax) * (cy - by) - (by - ay) * (cx - bx))
    if any(abs(c) <= 1e-6 for c in crosses):
        raise ValueError("image_corners_px contains a degenerate corner sequence")
    if not (all(c > 0 for c in crosses) or all(c < 0 for c in crosses)):
        raise ValueError("image_corners_px must be a convex ordered quadrilateral")
    area2 = abs(sum(
        points[i][0] * points[(i + 1) % 4][1] - points[(i + 1) % 4][0] * points[i][1]
        for i in range(4)
    ))
    if area2 <= 2.0:
        raise ValueError("image_corners_px quadrilateral area is too small")
    return points


@dataclass(frozen=True)
class TagAnchor:
    tag_family: str
    tag_id: int
    map_frame_id: str
    position_xyz_m: tuple[float, float, float]
    orientation_map_from_tag_xyzw: tuple[float, float, float, float]
    physical_size_m: float
    position_stddev_m: tuple[float, float, float]
    orientation_stddev_rad: tuple[float, float, float]
    reference_method: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "tag_family", _text(self.tag_family, "tag_family", 96))
        _int(self.tag_id, "tag_id")
        object.__setattr__(self, "map_frame_id", _text(self.map_frame_id, "map_frame_id", 128))
        object.__setattr__(self, "position_xyz_m", _vector(self.position_xyz_m, 3, "position_xyz_m"))
        object.__setattr__(self, "orientation_map_from_tag_xyzw", _quaternion(self.orientation_map_from_tag_xyzw, "orientation_map_from_tag_xyzw"))
        size = _real(self.physical_size_m, "physical_size_m")
        if size <= 0:
            raise ValueError("physical_size_m must be positive")
        object.__setattr__(self, "physical_size_m", size)
        object.__setattr__(self, "position_stddev_m", _vector(self.position_stddev_m, 3, "position_stddev_m", nonnegative=True))
        object.__setattr__(self, "orientation_stddev_rad", _vector(self.orientation_stddev_rad, 3, "orientation_stddev_rad", nonnegative=True))
        method = _text(self.reference_method, "reference_method", 32)
        if method not in {"surveyed", "calibrated"}:
            raise ValueError("reference_method must be surveyed or calibrated")
        object.__setattr__(self, "reference_method", method)


@dataclass(frozen=True)
class FiducialTagMap:
    profile: str
    map_frame_id: str
    tags: tuple[TagAnchor, ...]
    revision_sha256: str = field(init=False)
    _index: Mapping[tuple[str, int], TagAnchor] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        profile = _text(self.profile, "profile", 32)
        if profile not in {"indoor_dense", "outdoor_sparse_anchors"}:
            raise ValueError("profile must be indoor_dense or outdoor_sparse_anchors")
        frame = _text(self.map_frame_id, "map_frame_id", 128)
        if not isinstance(self.tags, (tuple, list)) or not self.tags:
            raise ValueError("tag map must contain at least one tag")
        tags = tuple(self.tags)
        if any(not isinstance(tag, TagAnchor) for tag in tags):
            raise ValueError("tag map must contain TagAnchor records")
        if any(tag.map_frame_id != frame for tag in tags):
            raise ValueError("all tags must use the tag map coordinate frame")
        keys = [(tag.tag_family, tag.tag_id) for tag in tags]
        if len(set(keys)) != len(keys):
            raise ValueError("tag family/ID pairs must be unique in a map")
        if profile == "outdoor_sparse_anchors" and any(tag.reference_method != "surveyed" for tag in tags):
            raise ValueError("outdoor anchor tags must have surveyed reference poses")
        canonical = {
            "profile": profile,
            "map_frame_id": frame,
            "tags": [
                {
                    "tag_family": tag.tag_family, "tag_id": tag.tag_id,
                    "map_frame_id": tag.map_frame_id,
                    "position_xyz_m": list(tag.position_xyz_m),
                    "orientation_map_from_tag_xyzw": list(tag.orientation_map_from_tag_xyzw),
                    "physical_size_m": tag.physical_size_m,
                    "position_stddev_m": list(tag.position_stddev_m),
                    "orientation_stddev_rad": list(tag.orientation_stddev_rad),
                    "reference_method": tag.reference_method,
                }
                for tag in sorted(tags, key=lambda item: (item.tag_family, item.tag_id))
            ],
        }
        digest = sha256(json.dumps(canonical, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()
        object.__setattr__(self, "profile", profile)
        object.__setattr__(self, "map_frame_id", frame)
        object.__setattr__(self, "tags", tags)
        object.__setattr__(self, "_index", MappingProxyType({key: tag for key, tag in zip(keys, tags)}))
        object.__setattr__(self, "revision_sha256", digest)

    @property
    def by_key(self) -> Mapping[tuple[str, int], TagAnchor]:
        return self._index


@dataclass(frozen=True)
class FiducialObservation:
    tag_family: str
    tag_id: int
    camera_id: int
    camera_serial: str
    frame_id: int
    trigger_id: int
    exposure_timestamp_ns: int
    clock_domain: str
    image_corners_px: tuple[tuple[float, float], ...]
    corrected_bit_errors: int
    corner_rms_px: float
    calibration_sha256: str
    map_revision_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "tag_family", _text(self.tag_family, "tag_family", 96))
        for name in ("tag_id", "camera_id", "frame_id", "trigger_id", "exposure_timestamp_ns", "corrected_bit_errors"):
            _int(getattr(self, name), name)
        object.__setattr__(self, "camera_serial", _text(self.camera_serial, "camera_serial"))
        object.__setattr__(self, "clock_domain", _text(self.clock_domain, "clock_domain"))
        object.__setattr__(self, "image_corners_px", _corners(self.image_corners_px))
        rms = _real(self.corner_rms_px, "corner_rms_px")
        if rms < 0:
            raise ValueError("corner_rms_px must be nonnegative")
        object.__setattr__(self, "corner_rms_px", rms)
        _digest(self.calibration_sha256, "calibration_sha256")
        _digest(self.map_revision_sha256, "map_revision_sha256")


def validate_fiducial_observation(
    observation: FiducialObservation,
    tag_map: FiducialTagMap,
    *,
    expected_camera_id: int,
    expected_camera_serial: str,
    expected_frame_id: int,
    expected_trigger_id: int,
    expected_exposure_timestamp_ns: int,
    expected_clock_domain: str,
    expected_calibration_sha256: str,
    max_corrected_bit_errors: int = 0,
    max_corner_rms_px: float = 1.5,
) -> FiducialObservation:
    """Validate association and quality; caller still performs pose estimation/fusion."""
    if not isinstance(observation, FiducialObservation) or not isinstance(tag_map, FiducialTagMap):
        raise ValueError("observation/tag_map has invalid type")
    checks = (
        (observation.camera_id == _int(expected_camera_id, "expected_camera_id"), "camera ID mismatch"),
        (observation.camera_serial == _text(expected_camera_serial, "expected_camera_serial"), "camera serial mismatch"),
        (observation.frame_id == _int(expected_frame_id, "expected_frame_id"), "frame ID mismatch"),
        (observation.trigger_id == _int(expected_trigger_id, "expected_trigger_id"), "trigger ID mismatch"),
        (observation.exposure_timestamp_ns == _int(expected_exposure_timestamp_ns, "expected_exposure_timestamp_ns"), "exposure timestamp mismatch"),
        (observation.clock_domain == _text(expected_clock_domain, "expected_clock_domain"), "clock domain mismatch"),
        (observation.calibration_sha256 == _digest(expected_calibration_sha256, "expected_calibration_sha256"), "camera calibration revision mismatch"),
        (observation.map_revision_sha256 == tag_map.revision_sha256, "tag map revision mismatch"),
    )
    for passed, message in checks:
        if not passed:
            raise ValueError(message)
    if (observation.tag_family, observation.tag_id) not in tag_map.by_key:
        raise ValueError("observation references a tag not present in the active map")
    errors = _int(max_corrected_bit_errors, "max_corrected_bit_errors")
    rms_limit = _real(max_corner_rms_px, "max_corner_rms_px")
    if rms_limit < 0:
        raise ValueError("max_corner_rms_px must be nonnegative")
    if observation.corrected_bit_errors > errors:
        raise ValueError("tag detector corrected too many code bits")
    if observation.corner_rms_px > rms_limit:
        raise ValueError("tag corner RMS exceeds configured limit")
    return observation
