"""Validated pose, state and map-alignment contracts for SLAM/Mocap adapters."""
from dataclasses import dataclass
from enum import Enum
import math
from numbers import Real


class LocalizationTopology(str, Enum):
    INSIDE_OUT = "inside_out"
    OUTSIDE_IN = "outside_in"


class PoseSource(str, Enum):
    VISUAL_SLAM = "visual_slam"
    FIDUCIAL_LOCALIZATION = "fiducial_localization"
    OUTSIDE_IN_MOCAP = "outside_in_mocap"


class LocalizationQuality(str, Enum):
    TRACKING = "TRACKING"
    DEGRADED = "DEGRADED"
    LOST = "LOST"
    INVALID = "INVALID"


class ScaleReference(str, Enum):
    NONE = "none"
    STEREO_BASELINE = "stereo_baseline"
    DEPTH_SENSOR = "depth_sensor"
    SURVEYED_TAGS = "surveyed_tags"
    RTK_CONTROL_POINTS = "rtk_control_points"
    CALIBRATED_CAMERA_GEOMETRY = "calibrated_camera_geometry"


def _text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be nonempty")
    return value.strip()


def _int(value, name):
    if type(value) is not int or not 0 <= value < 2**63:
        raise ValueError(f"{name} must be a nonnegative int64")
    return value


def _real(value, name):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite")
    return float(value)


def _digest(value, name):
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _vector(value, n, name, *, nonnegative=False):
    if not isinstance(value, (tuple, list)) or len(value) != n:
        raise ValueError(f"{name} must contain exactly {n} values")
    result = tuple(_real(v, name) for v in value)
    if nonnegative and any(v < 0 for v in result):
        raise ValueError(f"{name} values must be nonnegative")
    return result


def _quaternion(value, name):
    q = _vector(value, 4, name)
    norm = math.sqrt(sum(v * v for v in q))
    if norm < 1e-12:
        raise ValueError(f"{name} quaternion norm is zero")
    return tuple(v / norm for v in q)


def _source_and_topology(source, topology):
    try:
        source = PoseSource(source)
        topology = LocalizationTopology(topology)
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid pose source or topology") from exc
    if source is PoseSource.VISUAL_SLAM and topology is not LocalizationTopology.INSIDE_OUT:
        raise ValueError("visual SLAM camera poses require inside_out topology")
    if source is PoseSource.OUTSIDE_IN_MOCAP and topology is not LocalizationTopology.OUTSIDE_IN:
        raise ValueError("outside-in mocap poses require outside_in topology")
    return source, topology


def _quality(value):
    try:
        return LocalizationQuality(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid localization quality") from exc


def _scale_reference(value):
    try:
        return ScaleReference(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("unsupported scale_reference") from exc


def _validate_scale_qualification(metric_scale_qualified, scale_reference):
    if not isinstance(metric_scale_qualified, bool):
        raise ValueError("metric_scale_qualified must be boolean")
    reference = _scale_reference(scale_reference)
    if metric_scale_qualified != (reference is not ScaleReference.NONE):
        raise ValueError("metric scale qualification and scale reference disagree")
    return reference


@dataclass(frozen=True)
class MapPoseObservation:
    """Current pose of tracked_frame in map_frame at a sensor-acquisition timestamp.

    LOST/INVALID are status states rather than usable pose measurements and must
    be represented by LocalizationStatus, not by carrying a stale/fabricated pose.
    """
    source: PoseSource
    topology: LocalizationTopology
    tracked_frame_id: str
    map_frame_id: str
    timestamp_ns: int
    clock_domain: str
    position_xyz: tuple[float, float, float]
    position_unit: str
    orientation_xyzw: tuple[float, float, float, float]
    position_stddev: tuple[float, float, float]
    orientation_stddev_rad: tuple[float, float, float]
    map_revision_sha256: str
    calibration_sha256: str
    quality: LocalizationQuality
    inlier_count: int
    metric_scale_qualified: bool
    scale_reference: ScaleReference
    relocalized: bool = False
    reprojection_rms_px: float | None = None

    def __post_init__(self):
        source, topology = _source_and_topology(self.source, self.topology)
        quality = _quality(self.quality)
        if quality in (LocalizationQuality.LOST, LocalizationQuality.INVALID):
            raise ValueError("LOST/INVALID localization state cannot carry a pose observation")
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "topology", topology)
        object.__setattr__(self, "quality", quality)
        for name in ("tracked_frame_id", "map_frame_id", "clock_domain"):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        _int(self.timestamp_ns, "timestamp_ns")
        unit = _text(self.position_unit, "position_unit")
        if unit not in {"meters", "map_units"}:
            raise ValueError("position_unit must be meters or map_units")
        object.__setattr__(self, "position_unit", unit)
        reference = _validate_scale_qualification(
            self.metric_scale_qualified, self.scale_reference
        )
        if unit == "meters" and not self.metric_scale_qualified:
            raise ValueError("meters require independently qualified metric scale")
        if unit == "map_units" and self.metric_scale_qualified:
            raise ValueError("map_units cannot claim qualified metric scale")
        object.__setattr__(self, "scale_reference", reference)
        object.__setattr__(self, "position_xyz", _vector(self.position_xyz, 3, "position_xyz"))
        object.__setattr__(self, "orientation_xyzw", _quaternion(self.orientation_xyzw, "orientation_xyzw"))
        object.__setattr__(self, "position_stddev", _vector(self.position_stddev, 3, "position_stddev", nonnegative=True))
        object.__setattr__(self, "orientation_stddev_rad", _vector(self.orientation_stddev_rad, 3, "orientation_stddev_rad", nonnegative=True))
        _digest(self.map_revision_sha256, "map_revision_sha256")
        _digest(self.calibration_sha256, "calibration_sha256")
        if _int(self.inlier_count, "inlier_count") < 1:
            raise ValueError("inlier_count must be positive for an accepted pose")
        if not isinstance(self.relocalized, bool):
            raise ValueError("relocalized must be boolean")
        if self.reprojection_rms_px is not None:
            rms = _real(self.reprojection_rms_px, "reprojection_rms_px")
            if rms < 0:
                raise ValueError("reprojection_rms_px must be nonnegative")
            object.__setattr__(self, "reprojection_rms_px", rms)


@dataclass(frozen=True)
class LocalizationStatus:
    """Pose-free localization state, including DEGRADED/LOST/INVALID transitions."""
    source: PoseSource
    topology: LocalizationTopology
    tracked_frame_id: str
    map_frame_id: str
    timestamp_ns: int
    clock_domain: str
    map_revision_sha256: str
    calibration_sha256: str
    quality: LocalizationQuality
    reason: str
    last_pose_timestamp_ns: int | None = None

    def __post_init__(self):
        source, topology = _source_and_topology(self.source, self.topology)
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "topology", topology)
        object.__setattr__(self, "quality", _quality(self.quality))
        for name in ("tracked_frame_id", "map_frame_id", "clock_domain", "reason"):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        timestamp = _int(self.timestamp_ns, "timestamp_ns")
        _digest(self.map_revision_sha256, "map_revision_sha256")
        _digest(self.calibration_sha256, "calibration_sha256")
        if self.last_pose_timestamp_ns is not None:
            last_pose = _int(self.last_pose_timestamp_ns, "last_pose_timestamp_ns")
            if last_pose > timestamp:
                raise ValueError("last_pose_timestamp_ns cannot be newer than status timestamp")


@dataclass(frozen=True)
class MapAlignment:
    """Similarity transform: target = scale * rotate(source) + translation.

    scale_m_per_source_unit is None until metric scale is independently
    qualified. A numeric placeholder such as 1.0 is deliberately rejected for
    an unqualified map so downstream code cannot silently consume fake scale.
    """
    source_map_frame_id: str
    target_map_frame_id: str
    translation_xyz_m: tuple[float, float, float]
    rotation_xyzw: tuple[float, float, float, float]
    scale_m_per_source_unit: float | None
    metric_scale_qualified: bool
    scale_reference: ScaleReference
    source_map_revision_sha256: str
    target_map_revision_sha256: str
    alignment_revision_sha256: str

    def __post_init__(self):
        object.__setattr__(self, "source_map_frame_id", _text(self.source_map_frame_id, "source_map_frame_id"))
        object.__setattr__(self, "target_map_frame_id", _text(self.target_map_frame_id, "target_map_frame_id"))
        if self.source_map_frame_id == self.target_map_frame_id:
            raise ValueError("source and target map frames must differ")
        object.__setattr__(self, "translation_xyz_m", _vector(self.translation_xyz_m, 3, "translation_xyz_m"))
        object.__setattr__(self, "rotation_xyzw", _quaternion(self.rotation_xyzw, "rotation_xyzw"))
        reference = _validate_scale_qualification(
            self.metric_scale_qualified, self.scale_reference
        )
        object.__setattr__(self, "scale_reference", reference)
        if self.metric_scale_qualified:
            if self.scale_m_per_source_unit is None:
                raise ValueError("qualified metric alignment requires scale_m_per_source_unit")
            scale = _real(self.scale_m_per_source_unit, "scale_m_per_source_unit")
            if scale <= 0:
                raise ValueError("scale_m_per_source_unit must be positive")
            object.__setattr__(self, "scale_m_per_source_unit", scale)
        elif self.scale_m_per_source_unit is not None:
            raise ValueError("unqualified map scale must be None")
        for name in ("source_map_revision_sha256", "target_map_revision_sha256", "alignment_revision_sha256"):
            _digest(getattr(self, name), name)
