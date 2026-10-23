"""Calibrated single-tag pose estimation with fail-closed planar ambiguity gates.

Coordinate convention:
- tag frame is right-handed, +X right, +Y up, +Z out of the printed front.
- detector corners must be ordered by decoded tag orientation as:
  top-left, top-right, bottom-right, bottom-left when viewed from +Z.
- returned pose is camera-from-tag:
      X_camera = R_camera_from_tag @ X_tag + t_camera_from_tag

The solver uses a calibrated planar homography for initialization, then
nonlinear reprojection refinement from two deterministic planar-pose seeds.
It rejects weak image support, back-facing/oblique poses, excessive
reprojection error, and competing planar solutions with similar residuals.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from fiducial_contract import (
    FiducialObservation,
    FiducialTagMap,
    validate_fiducial_observation,
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


def _vector(value: object, length: int, name: str) -> tuple[float, ...]:
    if not isinstance(value, (tuple, list)) or len(value) != length:
        raise ValueError(f"{name} must contain exactly {length} values")
    return tuple(_real(item, f"{name}[{index}]") for index, item in enumerate(value))


def _intrinsic_matrix(value: object) -> np.ndarray:
    raw = np.asarray(value, dtype=object)
    if raw.shape != (3, 3):
        raise ValueError("intrinsic_matrix must be 3x3")
    if any(
        isinstance(item, (bool, np.bool_))
        or not isinstance(item, (int, float, np.integer, np.floating))
        for item in raw.reshape(-1)
    ):
        raise ValueError("intrinsic_matrix must contain real numeric values")
    matrix = np.asarray(raw, dtype=np.float64)
    if not np.isfinite(matrix).all():
        raise ValueError("intrinsic_matrix must contain finite values")
    scale = float(np.max(np.abs(matrix)))
    if scale == 0.0:
        raise ValueError("intrinsic_matrix is singular or ill-conditioned")
    normalized = matrix / scale
    singular_values = np.linalg.svd(normalized, compute_uv=False)
    if singular_values[-1] <= np.finfo(np.float64).eps * singular_values[0] * 3.0:
        raise ValueError("intrinsic_matrix is singular or ill-conditioned")
    return matrix


def _quaternion_xyzw(rotation: np.ndarray) -> tuple[float, float, float, float]:
    quaternion = Rotation.from_matrix(rotation).as_quat()
    if quaternion[3] < 0.0:
        quaternion = -quaternion
    return tuple(float(value) for value in quaternion)


def _polygon_area(points: np.ndarray) -> float:
    return abs(float(sum(
        points[index, 0] * points[(index + 1) % len(points), 1]
        - points[(index + 1) % len(points), 0] * points[index, 1]
        for index in range(len(points))
    ))) * 0.5


def _tag_object_corners(size_m: float) -> np.ndarray:
    half = size_m * 0.5
    return np.asarray([
        [-half, +half, 0.0],
        [+half, +half, 0.0],
        [+half, -half, 0.0],
        [-half, -half, 0.0],
    ], dtype=np.float64)


def _distort_normalized(points: np.ndarray, coefficients: tuple[float, ...]) -> np.ndarray:
    if not coefficients:
        return points
    k1, k2, p1, p2, k3 = coefficients
    x = points[:, 0]
    y = points[:, 1]
    radius2 = x * x + y * y
    radial = 1.0 + k1 * radius2 + k2 * radius2 * radius2 + k3 * radius2 * radius2 * radius2
    distorted_x = x * radial + 2.0 * p1 * x * y + p2 * (radius2 + 2.0 * x * x)
    distorted_y = y * radial + p1 * (radius2 + 2.0 * y * y) + 2.0 * p2 * x * y
    result = np.column_stack((distorted_x, distorted_y))
    if not np.isfinite(result).all():
        raise ValueError("camera distortion produced non-finite coordinates")
    return result


def _undistorted_normalized_pixels(
    pixels: np.ndarray,
    intrinsic: np.ndarray,
    coefficients: tuple[float, ...],
) -> np.ndarray:
    inverse = np.linalg.inv(intrinsic)
    homogeneous = (inverse @ np.column_stack((pixels, np.ones(len(pixels)))).T).T
    if not np.isfinite(homogeneous).all() or np.any(np.abs(homogeneous[:, 2]) <= 1e-15):
        raise ValueError("pixel normalization is singular")
    distorted = homogeneous[:, :2] / homogeneous[:, 2, None]
    if not coefficients:
        return distorted

    k1, k2, p1, p2, k3 = coefficients
    estimate = distorted.copy()
    for _ in range(12):
        x = estimate[:, 0]
        y = estimate[:, 1]
        radius2 = x * x + y * y
        radial = 1.0 + k1 * radius2 + k2 * radius2 * radius2 + k3 * radius2 * radius2 * radius2
        if not np.isfinite(radial).all() or np.any(np.abs(radial) <= 1e-12):
            raise ValueError("camera distortion inversion is singular")
        delta_x = 2.0 * p1 * x * y + p2 * (radius2 + 2.0 * x * x)
        delta_y = p1 * (radius2 + 2.0 * y * y) + 2.0 * p2 * x * y
        estimate = np.column_stack((
            (distorted[:, 0] - delta_x) / radial,
            (distorted[:, 1] - delta_y) / radial,
        ))
        if not np.isfinite(estimate).all():
            raise ValueError("camera distortion inversion produced non-finite coordinates")
    return estimate


def _homography_initial_pose(
    object_xy: np.ndarray,
    normalized_image_xy: np.ndarray,
    object_points: np.ndarray,
    *,
    min_depth_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    rows = []
    for (x_world, y_world), (x_image, y_image) in zip(object_xy, normalized_image_xy):
        rows.append([
            -x_world, -y_world, -1.0,
            0.0, 0.0, 0.0,
            x_image * x_world, x_image * y_world, x_image,
        ])
        rows.append([
            0.0, 0.0, 0.0,
            -x_world, -y_world, -1.0,
            y_image * x_world, y_image * y_world, y_image,
        ])
    design = np.asarray(rows, dtype=np.float64)
    if not np.isfinite(design).all() or np.linalg.matrix_rank(design) < 8:
        raise ValueError("planar tag homography is degenerate")
    _, _, vh = np.linalg.svd(design, full_matrices=True)
    homography = vh[-1].reshape(3, 3)
    first = homography[:, 0]
    second = homography[:, 1]
    third = homography[:, 2]
    denominator = float(np.linalg.norm(first) + np.linalg.norm(second))
    if not math.isfinite(denominator) or denominator <= np.finfo(np.float64).tiny:
        raise ValueError("planar tag homography has invalid scale")
    scale = 2.0 / denominator

    positive_depth = []
    for sign in (1.0, -1.0):
        r1 = sign * scale * first
        r2 = sign * scale * second
        translation = sign * scale * third
        raw_rotation = np.column_stack((r1, r2, np.cross(r1, r2)))
        u, _, vh_rotation = np.linalg.svd(raw_rotation)
        rotation = u @ vh_rotation
        if np.linalg.det(rotation) < 0.0:
            u[:, -1] *= -1.0
            rotation = u @ vh_rotation
        camera_points = (rotation @ object_points.T).T + translation
        if np.isfinite(camera_points).all() and np.all(camera_points[:, 2] > min_depth_m):
            positive_depth.append((rotation, translation))
    if len(positive_depth) != 1:
        raise ValueError("planar homography does not have one positive-depth initialization")
    return positive_depth[0]


def _rotation_between(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    a = first / np.linalg.norm(first)
    b = second / np.linalg.norm(second)
    cross = np.cross(a, b)
    sine = float(np.linalg.norm(cross))
    cosine = float(np.dot(a, b))
    if sine <= 1e-12:
        if cosine > 0.0:
            return np.eye(3)
        reference = np.asarray([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.asarray([0.0, 1.0, 0.0])
        axis = np.cross(a, reference)
        axis /= np.linalg.norm(axis)
        return Rotation.from_rotvec(axis * math.pi).as_matrix()
    axis = cross / sine
    return Rotation.from_rotvec(axis * math.atan2(sine, cosine)).as_matrix()


def _planar_alternative_seed(rotation: np.ndarray) -> np.ndarray:
    normal = rotation[:, 2]
    reflected = np.asarray([-normal[0], -normal[1], normal[2]])
    if np.linalg.norm(reflected) <= np.finfo(np.float64).tiny:
        return rotation.copy()
    return _rotation_between(normal, reflected) @ rotation


def _project_pixels(
    rotation: np.ndarray,
    translation: np.ndarray,
    object_points: np.ndarray,
    intrinsic: np.ndarray,
    distortion: tuple[float, ...],
) -> tuple[np.ndarray, np.ndarray]:
    camera_points = (rotation @ object_points.T).T + translation
    if not np.isfinite(camera_points).all() or np.any(np.abs(camera_points[:, 2]) <= 1e-15):
        raise ValueError("candidate tag pose has singular camera depth")
    normalized = camera_points[:, :2] / camera_points[:, 2, None]
    distorted = _distort_normalized(normalized, distortion)
    projected_h = (intrinsic @ np.column_stack((distorted, np.ones(len(distorted)))).T).T
    if not np.isfinite(projected_h).all() or np.any(np.abs(projected_h[:, 2]) <= 1e-15):
        raise ValueError("candidate tag projection is singular")
    return projected_h[:, :2] / projected_h[:, 2, None], camera_points


def _rotation_distance_rad(first: np.ndarray, second: np.ndarray) -> float:
    relative = first.T @ second
    cosine = max(-1.0, min(1.0, (float(np.trace(relative)) - 1.0) * 0.5))
    return math.acos(cosine)


@dataclass(frozen=True)
class FiducialCameraCalibration:
    camera_id: int
    camera_serial: str
    intrinsic_matrix: tuple[tuple[float, float, float], ...]
    distortion_model: str
    distortion_coefficients: tuple[float, ...]
    calibration_sha256: str

    def __post_init__(self) -> None:
        _integer(self.camera_id, "camera_id")
        object.__setattr__(self, "camera_serial", _text(self.camera_serial, "camera_serial"))
        matrix = _intrinsic_matrix(self.intrinsic_matrix)
        object.__setattr__(
            self,
            "intrinsic_matrix",
            tuple(tuple(float(value) for value in row) for row in matrix),
        )
        model = _text(self.distortion_model, "distortion_model", 32)
        if model not in {"none", "brown5"}:
            raise ValueError("distortion_model must be none or brown5")
        coefficients = tuple(self.distortion_coefficients)
        if model == "none":
            if coefficients:
                raise ValueError("none distortion model requires zero coefficients")
        else:
            if len(coefficients) != 5:
                raise ValueError("brown5 distortion model requires five coefficients")
            coefficients = tuple(_real(value, f"distortion_coefficients[{index}]") for index, value in enumerate(coefficients))
        object.__setattr__(self, "distortion_model", model)
        object.__setattr__(self, "distortion_coefficients", coefficients)
        _digest(self.calibration_sha256, "calibration_sha256")


@dataclass(frozen=True)
class FiducialTagPoseMeasurement:
    tag_family: str
    tag_id: int
    camera_id: int
    camera_serial: str
    frame_id: int
    trigger_id: int
    exposure_timestamp_ns: int
    clock_domain: str
    translation_camera_from_tag_m: tuple[float, float, float]
    orientation_camera_from_tag_xyzw: tuple[float, float, float, float]
    reprojection_rms_px: float
    image_area_px2: float
    viewing_angle_deg: float
    minimum_corner_depth_m: float
    candidate_count: int
    ambiguity_rms_margin_px: float | None
    calibration_sha256: str
    map_revision_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "tag_family", _text(self.tag_family, "tag_family", 96))
        for name in ("tag_id", "camera_id", "frame_id", "trigger_id", "exposure_timestamp_ns"):
            _integer(getattr(self, name), name)
        object.__setattr__(self, "camera_serial", _text(self.camera_serial, "camera_serial"))
        object.__setattr__(self, "clock_domain", _text(self.clock_domain, "clock_domain"))
        object.__setattr__(
            self,
            "translation_camera_from_tag_m",
            _vector(self.translation_camera_from_tag_m, 3, "translation_camera_from_tag_m"),
        )
        quaternion = np.asarray(_vector(self.orientation_camera_from_tag_xyzw, 4, "orientation_camera_from_tag_xyzw"))
        norm = float(np.linalg.norm(quaternion))
        if not math.isfinite(norm) or norm <= 1e-12:
            raise ValueError("orientation_camera_from_tag_xyzw quaternion norm is zero")
        quaternion /= norm
        if quaternion[3] < 0.0:
            quaternion = -quaternion
        object.__setattr__(
            self,
            "orientation_camera_from_tag_xyzw",
            tuple(float(value) for value in quaternion),
        )
        object.__setattr__(self, "reprojection_rms_px", _real(self.reprojection_rms_px, "reprojection_rms_px", minimum=0.0))
        object.__setattr__(self, "image_area_px2", _real(self.image_area_px2, "image_area_px2", minimum=0.0))
        angle = _real(self.viewing_angle_deg, "viewing_angle_deg", minimum=0.0)
        if angle > 90.0:
            raise ValueError("viewing_angle_deg must be <= 90")
        object.__setattr__(self, "viewing_angle_deg", angle)
        object.__setattr__(
            self,
            "minimum_corner_depth_m",
            _real(self.minimum_corner_depth_m, "minimum_corner_depth_m", minimum=0.0),
        )
        _integer(self.candidate_count, "candidate_count", minimum=1)
        if self.ambiguity_rms_margin_px is not None:
            object.__setattr__(
                self,
                "ambiguity_rms_margin_px",
                _real(self.ambiguity_rms_margin_px, "ambiguity_rms_margin_px", minimum=0.0),
            )
        _digest(self.calibration_sha256, "calibration_sha256")
        _digest(self.map_revision_sha256, "map_revision_sha256")


def solve_single_tag_pose(
    observation: FiducialObservation,
    tag_map: FiducialTagMap,
    calibration: FiducialCameraCalibration,
    *,
    max_corrected_bit_errors: int = 0,
    max_corner_rms_px: float = 1.5,
    max_reprojection_rms_px: float = 2.0,
    min_image_area_px2: float = 16.0,
    max_viewing_angle_deg: float = 80.0,
    ambiguity_rms_tolerance_px: float = 0.1,
    min_depth_m: float = 1e-4,
) -> FiducialTagPoseMeasurement:
    """Estimate one camera-from-tag pose and reject planar ambiguity.

    The result is a tag-relative measurement only. Map/world and body transforms
    are deliberately deferred to the next localization layer.
    """
    if not isinstance(calibration, FiducialCameraCalibration):
        raise ValueError("calibration must be FiducialCameraCalibration")
    max_corrected_bit_errors = _integer(max_corrected_bit_errors, "max_corrected_bit_errors")
    max_corner_rms_px = _real(max_corner_rms_px, "max_corner_rms_px", minimum=0.0)
    max_reprojection_rms_px = _real(max_reprojection_rms_px, "max_reprojection_rms_px", minimum=0.0)
    min_image_area_px2 = _real(min_image_area_px2, "min_image_area_px2", minimum=0.0)
    max_viewing_angle_deg = _real(max_viewing_angle_deg, "max_viewing_angle_deg", minimum=0.0)
    if max_viewing_angle_deg > 90.0:
        raise ValueError("max_viewing_angle_deg must be <= 90")
    ambiguity_rms_tolerance_px = _real(
        ambiguity_rms_tolerance_px, "ambiguity_rms_tolerance_px", minimum=0.0
    )
    min_depth_m = _real(min_depth_m, "min_depth_m", minimum=0.0)

    if not isinstance(observation, FiducialObservation) or not isinstance(tag_map, FiducialTagMap):
        raise ValueError("observation/tag_map has invalid type")
    validate_fiducial_observation(
        observation,
        tag_map,
        expected_camera_id=calibration.camera_id,
        expected_camera_serial=calibration.camera_serial,
        expected_frame_id=observation.frame_id,
        expected_trigger_id=observation.trigger_id,
        expected_exposure_timestamp_ns=observation.exposure_timestamp_ns,
        expected_clock_domain=observation.clock_domain,
        expected_calibration_sha256=calibration.calibration_sha256,
        max_corrected_bit_errors=max_corrected_bit_errors,
        max_corner_rms_px=max_corner_rms_px,
    )

    anchor = tag_map.by_key[(observation.tag_family, observation.tag_id)]
    pixels = np.asarray(observation.image_corners_px, dtype=np.float64)
    area = _polygon_area(pixels)
    if area < min_image_area_px2:
        raise ValueError(
            f"tag image area {area:.6g}px^2 is below minimum {min_image_area_px2:.6g}px^2"
        )

    intrinsic = _intrinsic_matrix(calibration.intrinsic_matrix)
    distortion = calibration.distortion_coefficients
    normalized = _undistorted_normalized_pixels(pixels, intrinsic, distortion)
    object_points = _tag_object_corners(anchor.physical_size_m)
    initial_rotation, initial_translation = _homography_initial_pose(
        object_points[:, :2],
        normalized,
        object_points,
        min_depth_m=min_depth_m,
    )
    alternate_rotation = _planar_alternative_seed(initial_rotation)
    seeds = (
        (initial_rotation, initial_translation),
        (alternate_rotation, initial_translation),
    )

    candidates: list[dict[str, object]] = []
    for seed_rotation, seed_translation in seeds:
        initial_parameters = np.concatenate((
            Rotation.from_matrix(seed_rotation).as_rotvec(),
            np.asarray(seed_translation, dtype=np.float64),
        ))

        def residual(parameters: np.ndarray) -> np.ndarray:
            try:
                rotation = Rotation.from_rotvec(parameters[:3]).as_matrix()
                projected, _ = _project_pixels(
                    rotation,
                    parameters[3:],
                    object_points,
                    intrinsic,
                    distortion,
                )
            except (ValueError, np.linalg.LinAlgError):
                return np.full(8, 1e9, dtype=np.float64)
            error = (projected - pixels).reshape(-1)
            if not np.isfinite(error).all():
                return np.full(8, 1e9, dtype=np.float64)
            return error

        refined = least_squares(
            residual,
            initial_parameters,
            method="trf",
            max_nfev=600,
            ftol=1e-12,
            xtol=1e-12,
            gtol=1e-12,
        )
        if not np.isfinite(refined.x).all():
            continue
        rotation = Rotation.from_rotvec(refined.x[:3]).as_matrix()
        translation = refined.x[3:]
        try:
            projected, camera_points = _project_pixels(
                rotation, translation, object_points, intrinsic, distortion
            )
        except (ValueError, np.linalg.LinAlgError):
            continue
        if np.any(camera_points[:, 2] <= min_depth_m):
            continue
        translation_norm = float(np.linalg.norm(translation))
        if not math.isfinite(translation_norm) or translation_norm <= min_depth_m:
            continue
        tag_front_normal_camera = rotation[:, 2]
        tag_to_camera = -translation / translation_norm
        cosine = float(np.dot(tag_front_normal_camera, tag_to_camera))
        viewing_angle = math.degrees(math.acos(max(-1.0, min(1.0, cosine))))
        if viewing_angle > max_viewing_angle_deg:
            continue
        reprojection = float(np.sqrt(np.mean(np.sum((projected - pixels) ** 2, axis=1))))
        if not math.isfinite(reprojection):
            continue

        duplicate = False
        for candidate in candidates:
            old_rotation = candidate["rotation"]
            old_translation = candidate["translation"]
            if (
                _rotation_distance_rad(old_rotation, rotation) <= 1e-4
                and float(np.linalg.norm(old_translation - translation))
                <= 1e-5 * max(1.0, translation_norm)
            ):
                duplicate = True
                if reprojection < candidate["reprojection_rms_px"]:
                    candidate.update(
                        rotation=rotation,
                        translation=translation,
                        reprojection_rms_px=reprojection,
                        viewing_angle_deg=viewing_angle,
                        minimum_corner_depth_m=float(np.min(camera_points[:, 2])),
                    )
                break
        if not duplicate:
            candidates.append({
                "rotation": rotation,
                "translation": translation,
                "reprojection_rms_px": reprojection,
                "viewing_angle_deg": viewing_angle,
                "minimum_corner_depth_m": float(np.min(camera_points[:, 2])),
            })

    if not candidates:
        raise ValueError("no positive-depth front-facing tag pose candidate")

    candidates.sort(key=lambda item: item["reprojection_rms_px"])
    best = candidates[0]
    if best["reprojection_rms_px"] > max_reprojection_rms_px:
        raise ValueError(
            f"tag reprojection RMS {best['reprojection_rms_px']:.6g}px exceeds "
            f"limit {max_reprojection_rms_px:.6g}px"
        )

    admissible = [
        candidate
        for candidate in candidates
        if candidate["reprojection_rms_px"] <= max_reprojection_rms_px
    ]
    ambiguity_margin = None
    if len(admissible) > 1:
        ambiguity_margin = float(
            admissible[1]["reprojection_rms_px"] - admissible[0]["reprojection_rms_px"]
        )
        if ambiguity_margin <= ambiguity_rms_tolerance_px:
            raise ValueError(
                f"planar tag pose is ambiguous: competing candidate RMS margin "
                f"{ambiguity_margin:.6g}px <= {ambiguity_rms_tolerance_px:.6g}px"
            )

    return FiducialTagPoseMeasurement(
        tag_family=observation.tag_family,
        tag_id=observation.tag_id,
        camera_id=observation.camera_id,
        camera_serial=observation.camera_serial,
        frame_id=observation.frame_id,
        trigger_id=observation.trigger_id,
        exposure_timestamp_ns=observation.exposure_timestamp_ns,
        clock_domain=observation.clock_domain,
        translation_camera_from_tag_m=tuple(float(value) for value in best["translation"]),
        orientation_camera_from_tag_xyzw=_quaternion_xyzw(best["rotation"]),
        reprojection_rms_px=float(best["reprojection_rms_px"]),
        image_area_px2=area,
        viewing_angle_deg=float(best["viewing_angle_deg"]),
        minimum_corner_depth_m=float(best["minimum_corner_depth_m"]),
        candidate_count=len(candidates),
        ambiguity_rms_margin_px=ambiguity_margin,
        calibration_sha256=observation.calibration_sha256,
        map_revision_sha256=observation.map_revision_sha256,
    )
