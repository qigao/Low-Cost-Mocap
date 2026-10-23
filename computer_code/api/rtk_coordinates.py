"""Explicit WGS-84 ECEF/ENU conversion and RTK antenna lever-arm helpers."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np

from outdoor_sensor_contract import RTKSample, RTKSolutionStatus


_WGS84_A_M = 6_378_137.0
_WGS84_F = 1.0 / 298.257223563
_WGS84_E2 = _WGS84_F * (2.0 - _WGS84_F)


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be finite")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _vec3(value: Sequence[float], name: str) -> np.ndarray:
    if not isinstance(value, (tuple, list, np.ndarray)) or len(value) != 3:
        raise ValueError(f"{name} must contain exactly three coordinates")
    result = np.asarray([_finite(v, f"{name}[{i}]") for i, v in enumerate(value)], dtype=np.float64)
    if result.shape != (3,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must contain three finite coordinates")
    return result


@dataclass(frozen=True)
class ENUOrigin:
    """Surveyed local tangent-frame origin in WGS-84 geodetic coordinates."""

    latitude_deg: float
    longitude_deg: float
    ellipsoid_height_m: float
    datum: str = "WGS84"

    def __post_init__(self) -> None:
        lat = _finite(self.latitude_deg, "latitude_deg")
        lon = _finite(self.longitude_deg, "longitude_deg")
        height = _finite(self.ellipsoid_height_m, "ellipsoid_height_m")
        if not -90.0 <= lat <= 90.0:
            raise ValueError("latitude_deg must be within [-90, 90]")
        if not -180.0 <= lon <= 180.0:
            raise ValueError("longitude_deg must be within [-180, 180]")
        if self.datum != "WGS84":
            raise ValueError("only explicit WGS84 datum is supported")
        object.__setattr__(self, "latitude_deg", lat)
        object.__setattr__(self, "longitude_deg", lon)
        object.__setattr__(self, "ellipsoid_height_m", height)

    @property
    def ecef_m(self) -> tuple[float, float, float]:
        return geodetic_to_ecef(
            self.latitude_deg, self.longitude_deg, self.ellipsoid_height_m
        )


def geodetic_to_ecef(
    latitude_deg: float, longitude_deg: float, ellipsoid_height_m: float
) -> tuple[float, float, float]:
    """Convert WGS-84 geodetic latitude/longitude and ellipsoid height to ECEF."""
    lat_deg = _finite(latitude_deg, "latitude_deg")
    lon_deg = _finite(longitude_deg, "longitude_deg")
    height = _finite(ellipsoid_height_m, "ellipsoid_height_m")
    if not -90.0 <= lat_deg <= 90.0:
        raise ValueError("latitude_deg must be within [-90, 90]")
    if not -180.0 <= lon_deg <= 180.0:
        raise ValueError("longitude_deg must be within [-180, 180]")
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    sin_lat = math.sin(lat)
    cos_lat = math.cos(lat)
    radius = _WGS84_A_M / math.sqrt(1.0 - _WGS84_E2 * sin_lat * sin_lat)
    x = (radius + height) * cos_lat * math.cos(lon)
    y = (radius + height) * cos_lat * math.sin(lon)
    z = (radius * (1.0 - _WGS84_E2) + height) * sin_lat
    return (x, y, z)


def _ecef_delta_to_enu_rotation(origin: ENUOrigin) -> np.ndarray:
    lat = math.radians(origin.latitude_deg)
    lon = math.radians(origin.longitude_deg)
    sin_lat, cos_lat = math.sin(lat), math.cos(lat)
    sin_lon, cos_lon = math.sin(lon), math.cos(lon)
    return np.asarray([
        [-sin_lon, cos_lon, 0.0],
        [-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat],
        [cos_lat * cos_lon, cos_lat * sin_lon, sin_lat],
    ], dtype=np.float64)


def ecef_to_enu(position_ecef_m: Sequence[float], origin: ENUOrigin) -> tuple[float, float, float]:
    """Convert an ECEF position to metres in the origin's East-North-Up frame."""
    if not isinstance(origin, ENUOrigin):
        raise ValueError("origin must be ENUOrigin")
    position = _vec3(position_ecef_m, "position_ecef_m")
    origin_ecef = np.asarray(origin.ecef_m, dtype=np.float64)
    enu = _ecef_delta_to_enu_rotation(origin) @ (position - origin_ecef)
    return tuple(float(v) for v in enu)


def enu_to_ecef(position_enu_m: Sequence[float], origin: ENUOrigin) -> tuple[float, float, float]:
    """Convert local ENU metres back to WGS-84 ECEF metres."""
    if not isinstance(origin, ENUOrigin):
        raise ValueError("origin must be ENUOrigin")
    enu = _vec3(position_enu_m, "position_enu_m")
    origin_ecef = np.asarray(origin.ecef_m, dtype=np.float64)
    ecef = origin_ecef + _ecef_delta_to_enu_rotation(origin).T @ enu
    return tuple(float(v) for v in ecef)


def qualified_rtk_position_enu(
    sample: RTKSample,
    origin: ENUOrigin,
    *,
    max_correction_age_ms: float,
    max_horizontal_accuracy_m: float,
    max_vertical_accuracy_m: float,
    min_satellites: int = 6,
) -> tuple[float, float, float]:
    """Convert only a caller-qualified FIXED RTK solution into local ENU."""
    if not isinstance(sample, RTKSample):
        raise ValueError("sample must be RTKSample")
    if sample.datum != origin.datum:
        raise ValueError("RTK sample datum does not match ENU origin datum")
    if not sample.usable_as_global_position(
        max_correction_age_ms=max_correction_age_ms,
        max_horizontal_accuracy_m=max_horizontal_accuracy_m,
        max_vertical_accuracy_m=max_vertical_accuracy_m,
        min_satellites=min_satellites,
    ):
        raise ValueError("RTK sample does not meet declared global-position gates")
    return ecef_to_enu(sample.position_ecef_m, origin)


def antenna_enu_to_body_origin_enu(
    antenna_position_enu_m: Sequence[float],
    rotation_enu_from_body: Sequence[Sequence[float]],
    lever_arm_body_m: Sequence[float],
) -> tuple[float, float, float]:
    """Remove the body-frame lever arm from antenna position.

    lever_arm_body_m is the vector from body origin to antenna phase center,
    expressed in body coordinates. rotation_enu_from_body must be a proper
    rotation taking body vectors into ENU coordinates.
    """
    antenna = _vec3(antenna_position_enu_m, "antenna_position_enu_m")
    lever = _vec3(lever_arm_body_m, "lever_arm_body_m")
    rotation = np.asarray(rotation_enu_from_body, dtype=np.float64)
    if rotation.shape != (3, 3) or not np.isfinite(rotation).all():
        raise ValueError("rotation_enu_from_body must be a finite 3x3 matrix")
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6, rtol=0.0):
        raise ValueError("rotation_enu_from_body must be orthonormal")
    if not math.isclose(float(np.linalg.det(rotation)), 1.0, abs_tol=1e-6, rel_tol=0.0):
        raise ValueError("rotation_enu_from_body must be a proper rotation")
    body_origin = antenna - rotation @ lever
    return tuple(float(v) for v in body_origin)
