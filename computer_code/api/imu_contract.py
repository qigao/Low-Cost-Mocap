"""Hardware-neutral IMU sample and clock-mapping contracts.

A fitted mapping is only as trustworthy as its externally measured anchors.
This module deliberately cannot certify that an anchor came from a physical
synchronization pulse; the adapter/evidence package must establish provenance.
Host receive/dequeue timestamps are diagnostics, not sensor acquisition time.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import math
from numbers import Real
from typing import Sequence


_MAX_I64 = (1 << 63) - 1
_MIN_I64 = -(1 << 63)


def _timestamp_ns(value: object, name: str, *, allow_negative: bool = False) -> int:
    if type(value) is not int:
        raise ValueError(f"{name} must be an integer nanosecond timestamp")
    lower = _MIN_I64 if allow_negative else 0
    if not lower <= value <= _MAX_I64:
        raise ValueError(f"{name} is outside the signed 64-bit timestamp range")
    return value


def _nonnegative_int(value: object, name: str) -> int:
    if type(value) is not int or not 0 <= value <= _MAX_I64:
        raise ValueError(f"{name} must be a nonnegative signed-64-bit integer")
    return value


def _nonempty_text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value.strip()


def _finite_real(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    try:
        result = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite real number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite real number")
    return result


def _vector3(value: object, name: str) -> tuple[float, float, float]:
    if not isinstance(value, (tuple, list)) or len(value) != 3:
        raise ValueError(f"{name} must contain exactly three real values")
    result = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, Real):
            raise ValueError(f"{name} must contain only real numeric values")
        try:
            number = float(item)
        except (OverflowError, TypeError, ValueError) as exc:
            raise ValueError(f"{name} must contain finite real values") from exc
        if not math.isfinite(number):
            raise ValueError(f"{name} must contain finite real values")
        result.append(number)
    return tuple(result)


@dataclass(frozen=True)
class IMUSample:
    """One acquisition-time IMU sample in the sensor-native coordinate frame.

    Units are fixed at the boundary: gyro_rad_s and accel_m_s2. The optional
    host_received_timestamp_ns is diagnostic only and is never used for fusion.
    Invalid/saturated samples can be represented for logging but must not be
    passed to estimator propagation.
    """

    imu_id: str
    sequence_number: int
    sensor_timestamp_ns: int
    clock_domain: str
    gyro_rad_s: tuple[float, float, float]
    accel_m_s2: tuple[float, float, float]
    calibration_revision: str
    valid: bool = True
    saturated: bool = False
    overrun: bool = False
    host_received_timestamp_ns: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "imu_id", _nonempty_text(self.imu_id, "imu_id"))
        object.__setattr__(
            self, "sequence_number",
            _nonnegative_int(self.sequence_number, "sequence_number"),
        )
        object.__setattr__(
            self, "sensor_timestamp_ns",
            _timestamp_ns(self.sensor_timestamp_ns, "sensor_timestamp_ns"),
        )
        object.__setattr__(self, "clock_domain", _nonempty_text(self.clock_domain, "clock_domain"))
        object.__setattr__(self, "calibration_revision", _nonempty_text(self.calibration_revision, "calibration_revision"))
        object.__setattr__(self, "gyro_rad_s", _vector3(self.gyro_rad_s, "gyro_rad_s"))
        object.__setattr__(self, "accel_m_s2", _vector3(self.accel_m_s2, "accel_m_s2"))
        for name in ("valid", "saturated", "overrun"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be boolean")
        if self.host_received_timestamp_ns is not None:
            object.__setattr__(
                self,
                "host_received_timestamp_ns",
                _timestamp_ns(
                    self.host_received_timestamp_ns,
                    "host_received_timestamp_ns",
                ),
            )

    def require_usable(self) -> None:
        """Fail closed before integration when the sample is invalid."""
        if not self.valid:
            raise ValueError("IMU sample is marked invalid")
        if self.saturated:
            raise ValueError("IMU sample is saturated")
        if self.overrun:
            raise ValueError("IMU device reports sample overrun")


class IMUSampleStreamGuard:
    """Monotonic per-device sample guard with bounded inter-sample gaps."""

    def __init__(
        self,
        imu_id: str,
        clock_domain: str,
        calibration_revision: str,
        *,
        max_sample_gap_ns: int,
    ) -> None:
        self.imu_id = _nonempty_text(imu_id, "imu_id")
        self.clock_domain = _nonempty_text(clock_domain, "clock_domain")
        self.calibration_revision = _nonempty_text(calibration_revision, "calibration_revision")
        self.max_sample_gap_ns = _nonnegative_int(max_sample_gap_ns, "max_sample_gap_ns")
        if self.max_sample_gap_ns < 1:
            raise ValueError("max_sample_gap_ns must be positive")
        self._last_sequence: int | None = None
        self._last_timestamp_ns: int | None = None

    def accept(self, sample: IMUSample) -> None:
        """Validate ordering and continuity; state advances only on acceptance."""
        if not isinstance(sample, IMUSample):
            raise ValueError("sample must be IMUSample")
        sample.require_usable()
        if sample.imu_id != self.imu_id:
            raise ValueError("IMU device ID mismatch")
        if sample.clock_domain != self.clock_domain:
            raise ValueError("IMU clock-domain mismatch")
        if sample.calibration_revision != self.calibration_revision:
            raise ValueError("IMU calibration revision mismatch")
        if self._last_sequence is not None and sample.sequence_number <= self._last_sequence:
            raise ValueError("IMU sequence is duplicate or out of order")
        if self._last_timestamp_ns is not None:
            if sample.sensor_timestamp_ns <= self._last_timestamp_ns:
                raise ValueError("IMU acquisition timestamp is duplicate or regressed")
            if sample.sensor_timestamp_ns - self._last_timestamp_ns > self.max_sample_gap_ns:
                raise ValueError("IMU sample gap exceeds configured limit")
        self._last_sequence = sample.sequence_number
        self._last_timestamp_ns = sample.sensor_timestamp_ns

    def reset(self) -> None:
        """Explicitly begin a new stream epoch after a separately handled reset."""
        self._last_sequence = None
        self._last_timestamp_ns = None


@dataclass(frozen=True)
class ClockAnchor:
    """One paired source/target timestamp from an externally measured event.

    anchor_kind must be 'hardware_sync_pulse'. That label is a declaration,
    not cryptographic or physical proof; provenance belongs in capture evidence.
    """

    source_timestamp_ns: int
    target_timestamp_ns: int
    anchor_kind: str = "hardware_sync_pulse"
    event_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "source_timestamp_ns",
            _timestamp_ns(self.source_timestamp_ns, "source_timestamp_ns"),
        )
        object.__setattr__(
            self, "target_timestamp_ns",
            _timestamp_ns(self.target_timestamp_ns, "target_timestamp_ns"),
        )
        object.__setattr__(self, "anchor_kind", _nonempty_text(self.anchor_kind, "anchor_kind"))
        if self.event_id:
            object.__setattr__(self, "event_id", _nonempty_text(self.event_id, "event_id"))


@dataclass(frozen=True)
class ClockMapping:
    """Affine source-clock to target-clock mapping with bounded extrapolation."""

    source_clock_domain: str
    target_clock_domain: str
    source_reference_ns: int
    target_reference_ns: int
    offset_from_reference_ns: float
    slope: float
    source_min_ns: int
    source_max_ns: int
    max_extrapolation_ns: int
    max_residual_ns: float
    max_drift_ppm: float
    observed_max_residual_ns: float
    drift_ppm: float
    anchor_count: int
    mapping_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_clock_domain", _nonempty_text(self.source_clock_domain, "source_clock_domain"))
        object.__setattr__(self, "target_clock_domain", _nonempty_text(self.target_clock_domain, "target_clock_domain"))
        if self.source_clock_domain == self.target_clock_domain:
            raise ValueError("clock mapping source and target domains must differ")
        _timestamp_ns(self.source_reference_ns, "source_reference_ns")
        _timestamp_ns(self.target_reference_ns, "target_reference_ns")
        _timestamp_ns(self.source_min_ns, "source_min_ns")
        _timestamp_ns(self.source_max_ns, "source_max_ns")
        _nonnegative_int(self.max_extrapolation_ns, "max_extrapolation_ns")
        _nonnegative_int(self.anchor_count, "anchor_count")
        for name in ("offset_from_reference_ns", "slope", "max_residual_ns", "max_drift_ppm", "observed_max_residual_ns", "drift_ppm"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite")
        if (
            self.slope <= 0
            or self.max_residual_ns < 0
            or self.max_drift_ppm < 0
            or self.observed_max_residual_ns < 0
            or self.observed_max_residual_ns > self.max_residual_ns
            or abs(self.drift_ppm) > self.max_drift_ppm
            or self.anchor_count < 3
            or self.source_min_ns > self.source_reference_ns
            or self.source_reference_ns > self.source_max_ns
            or self.source_min_ns > self.source_max_ns
        ):
            raise ValueError("clock mapping fit or declared bounds are invalid")
        expected_drift_ppm = (self.slope - 1.0) * 1_000_000.0
        if abs(expected_drift_ppm - self.drift_ppm) > 1e-6:
            raise ValueError("clock mapping drift metadata disagrees with slope")
        if not isinstance(self.mapping_sha256, str) or len(self.mapping_sha256) != 64 or any(c not in "0123456789abcdef" for c in self.mapping_sha256):
            raise ValueError("mapping_sha256 must be lowercase SHA-256")

    def map_timestamp(self, source_timestamp_ns: int) -> int:
        """Map a timestamp only within the calibrated range plus bounded margin."""
        timestamp = _timestamp_ns(source_timestamp_ns, "source_timestamp_ns")
        if (
            timestamp < self.source_min_ns - self.max_extrapolation_ns
            or timestamp > self.source_max_ns + self.max_extrapolation_ns
        ):
            raise ValueError("timestamp lies outside qualified clock-mapping horizon")
        relative = timestamp - self.source_reference_ns
        mapped_relative = (
            self.offset_from_reference_ns + self.slope * relative
        )
        mapped = self.target_reference_ns + mapped_relative
        if not math.isfinite(mapped) or mapped < 0 or mapped > _MAX_I64:
            raise ValueError("mapped timestamp exceeds nonnegative signed 64-bit range")
        return int(round(mapped))


def fit_clock_mapping(
    anchors: Sequence[ClockAnchor],
    *,
    source_clock_domain: str,
    target_clock_domain: str,
    max_drift_ppm: float = 500.0,
    max_residual_ns: float = 1_000_000.0,
    max_extrapolation_ns: int = 10_000_000,
    min_anchor_span_ns: int = 10_000_000,
) -> ClockMapping:
    """Fit an affine clock mapping from monotonic external sync-pulse anchors.

    Defaults are engineering guardrails, not hardware-qualified limits. Real
    deployments must choose limits from measured sensor/trigger characteristics.
    """
    source_domain = _nonempty_text(source_clock_domain, "source_clock_domain")
    target_domain = _nonempty_text(target_clock_domain, "target_clock_domain")
    if source_domain == target_domain:
        raise ValueError("clock mapping source and target domains must differ")
    if not isinstance(anchors, (tuple, list)) or len(anchors) < 3:
        raise ValueError("at least three clock anchors are required")
    if any(not isinstance(anchor, ClockAnchor) for anchor in anchors):
        raise ValueError("anchors must contain only ClockAnchor values")
    if any(anchor.anchor_kind != "hardware_sync_pulse" for anchor in anchors):
        raise ValueError("clock anchors must come from declared hardware sync pulses")
    if any(anchor.event_id for anchor in anchors) and len({a.event_id for a in anchors if a.event_id}) != len([a for a in anchors if a.event_id]):
        raise ValueError("clock anchor event IDs must be unique")
    drift_limit = _finite_real(max_drift_ppm, "max_drift_ppm")
    residual_limit = _finite_real(max_residual_ns, "max_residual_ns")
    if drift_limit < 0:
        raise ValueError("max_drift_ppm must be nonnegative")
    if residual_limit < 0:
        raise ValueError("max_residual_ns must be nonnegative")
    extrapolation = _nonnegative_int(max_extrapolation_ns, "max_extrapolation_ns")
    min_span = _nonnegative_int(min_anchor_span_ns, "min_anchor_span_ns")
    if min_span < 1:
        raise ValueError("min_anchor_span_ns must be positive")

    source = [a.source_timestamp_ns for a in anchors]
    target = [a.target_timestamp_ns for a in anchors]
    if any(b <= a for a, b in zip(source, source[1:])):
        raise ValueError("source clock anchors must be strictly increasing")
    if any(b <= a for a, b in zip(target, target[1:])):
        raise ValueError("target clock anchors must be strictly increasing")
    if source[-1] - source[0] < min_span:
        raise ValueError("clock anchors span is too short to estimate drift")

    source_reference = source[0]
    target_reference = target[0]
    x = [float(v - source_reference) for v in source]
    y = [float(v - target_reference) for v in target]
    x_mean = sum(x) / len(x)
    y_mean = sum(y) / len(y)
    denominator = sum((v - x_mean) ** 2 for v in x)
    if not math.isfinite(denominator) or denominator <= 0:
        raise ValueError("clock anchor source variance is degenerate")
    slope = sum((xv - x_mean) * (yv - y_mean) for xv, yv in zip(x, y)) / denominator
    intercept = y_mean - slope * x_mean
    drift_ppm = (slope - 1.0) * 1_000_000.0
    residuals = [abs((intercept + slope * xv) - yv) for xv, yv in zip(x, y)]
    observed_residual = max(residuals)
    if not math.isfinite(slope) or slope <= 0:
        raise ValueError("fitted clock mapping slope must be finite and positive")
    if abs(drift_ppm) > drift_limit:
        raise ValueError("clock drift exceeds configured ppm limit")
    if not math.isfinite(observed_residual) or observed_residual > residual_limit:
        raise ValueError("clock anchor residual exceeds configured limit")

    fingerprint_payload = {
        "source_clock_domain": source_domain,
        "target_clock_domain": target_domain,
        "anchors": [
            [a.source_timestamp_ns, a.target_timestamp_ns, a.anchor_kind, a.event_id]
            for a in anchors
        ],
        "max_drift_ppm": drift_limit,
        "max_residual_ns": residual_limit,
        "max_extrapolation_ns": extrapolation,
        "min_anchor_span_ns": min_span,
        "slope": slope,
        "intercept": intercept,
    }
    fingerprint = sha256(
        json.dumps(fingerprint_payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()
    return ClockMapping(
        source_clock_domain=source_domain,
        target_clock_domain=target_domain,
        source_reference_ns=source_reference,
        target_reference_ns=target_reference,
        offset_from_reference_ns=intercept,
        slope=slope,
        source_min_ns=source[0],
        source_max_ns=source[-1],
        max_extrapolation_ns=extrapolation,
        max_residual_ns=residual_limit,
        max_drift_ppm=drift_limit,
        observed_max_residual_ns=observed_residual,
        drift_ppm=drift_ppm,
        anchor_count=len(anchors),
        mapping_sha256=fingerprint,
    )
