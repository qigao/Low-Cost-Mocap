"""Validated pose and map-alignment contracts for SLAM/Mocap adapters."""
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


@dataclass(frozen=True)
class MapPoseObservation:
    """Pose of tracked_frame in map_frame at a sensor-acquisition timestamp."""
    source: PoseSource
    topology: LocalizationTopology
    tracked_frame_id: str
    map_frame_id: str
    timestamp_ns: int
    clock_domain: str
    position_xyz_m: tuple[float, float, float]
    orientation_xyzw: tuple[float, float, float, float]
    position_stddev_m: tuple[float, float, float]
    orientation_stddev_rad: tuple[float, float, float]
    map_revision_sha256: str
    calibration_sha256: str
    quality: LocalizationQuality
    inlier_count: int
    relocalized: bool = False
    reprojection_rms_px: float | None = None

    def __post_init__(self):
        try:
            source, topology, quality = PoseSource(self.source), LocalizationTopology(self.topology), LocalizationQuality(self.quality)
        except (TypeError, ValueError) as exc:
            raise ValueError("invalid pose source, topology or quality") from exc
        if source is PoseSource.VISUAL_SLAM and topology is not LocalizationTopology.INSIDE_OUT:
            raise ValueError("visual SLAM camera poses require inside_out topology")
        if source is PoseSource.OUTSIDE_IN_MOCAP and topology is not LocalizationTopology.OUTSIDE_IN:
            raise ValueError("outside-in mocap poses require outside_in topology")
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "topology", topology)
        object.__setattr__(self, "quality", quality)
        for name in ("tracked_frame_id", "map_frame_id", "clock_domain"):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        _int(self.timestamp_ns, "timestamp_ns")
        object.__setattr__(self, "position_xyz_m", _vector(self.position_xyz_m, 3, "position_xyz_m"))
        object.__setattr__(self, "orientation_xyzw", _quaternion(self.orientation_xyzw, "orientation_xyzw"))
        object.__setattr__(self, "position_stddev_m", _vector(self.position_stddev_m, 3, "position_stddev_m", nonnegative=True))
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
class MapAlignment:
    """Similarity transform: target = scale * rotate(source) + translation."""
    source_map_frame_id: str
    target_map_frame_id: str
    translation_xyz_m: tuple[float, float, float]
    rotation_xyzw: tuple[float, float, float, float]
    scale_m_per_source_unit: float
    metric_scale_qualified: bool
    scale_reference: str
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
        scale = _real(self.scale_m_per_source_unit, "scale_m_per_source_unit")
        if scale <= 0:
            raise ValueError("scale_m_per_source_unit must be positive")
        object.__setattr__(self, "scale_m_per_source_unit", scale)
        if not isinstance(self.metric_scale_qualified, bool):
            raise ValueError("metric_scale_qualified must be boolean")
        reference = _text(self.scale_reference, "scale_reference")
        allowed = {"none", "stereo_baseline", "depth_sensor", "surveyed_tags", "rtk_control_points"}
        if reference not in allowed:
            raise ValueError("unsupported scale_reference")
        if self.metric_scale_qualified != (reference != "none"):
            raise ValueError("metric scale qualification and scale reference disagree")
        object.__setattr__(self, "scale_reference", reference)
        for name in ("source_map_revision_sha256", "target_map_revision_sha256", "alignment_revision_sha256"):
            _digest(getattr(self, name), name)
