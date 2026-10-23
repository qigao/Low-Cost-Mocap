"""Conditioned 9-state translational error-state estimator.

Nominal/error state:
    x = [position_world(3), velocity_world(3), residual_accel_bias_body(3)]

Attitude is supplied by the existing provider-neutral fused-attitude state and
is not estimated here. The estimator maintains the full 9x9 covariance so a
current-time absolute position update can legitimately correct position,
velocity and residual accelerometer bias through propagated cross-covariance.

This remains short of a full INS/ESKF because attitude error and gyro bias are
outside this covariance, delayed measurements are rejected, and nonzero IMU
lever-arm acceleration is not yet modeled.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import math
from numbers import Real

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from imu_calibration import CorrectedIMUVectors, IMUCalibration
from localization_contract import LocalizationQuality
from localization_imu_fusion import LocalizationIMUAttitudeState
from position_measurement import (
    MetricPositionMeasurement,
    PositionMeasurementSource,
    PositionSourceContract,
)
from translational_propagation import GravityModel, TranslationalSeed


_MAX_I64 = (1 << 63) - 1


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be nonempty")
    return value.strip()


def _i64(value: object, name: str) -> int:
    if type(value) is not int or not 0 <= value <= _MAX_I64:
        raise ValueError(f"{name} must be a nonnegative signed-64-bit integer")
    return value


def _real(value: object, name: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be finite real")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite real")
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return result


def _vec3(value: object, name: str) -> np.ndarray:
    if not isinstance(value, (tuple, list, np.ndarray)) or len(value) != 3:
        raise ValueError(f"{name} must contain exactly three values")
    result = np.asarray([_real(v, f"{name}[{i}]") for i, v in enumerate(value)], dtype=np.float64)
    if result.shape != (3,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must contain three finite values")
    return result


def _digest(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(c not in "0123456789abcdef" for c in value)
    ):
        raise ValueError(f"{name} must be lowercase SHA-256")
    return value


def _canonical_digest(payload: dict[str, object]) -> str:
    return sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _covariance_tuple(value: np.ndarray) -> tuple[tuple[float, ...], ...]:
    return tuple(tuple(float(item) for item in row) for row in value)


def _validated_covariance(value: object) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (9, 9) or not np.isfinite(matrix).all():
        raise ValueError("covariance_p_v_bias must be a finite 9x9 matrix")
    if not np.allclose(matrix, matrix.T, atol=1e-10, rtol=0.0):
        raise ValueError("covariance_p_v_bias must be symmetric")
    eigenvalues = np.linalg.eigvalsh(matrix)
    if float(np.min(eigenvalues)) < -1e-9:
        raise ValueError("covariance_p_v_bias must be positive semidefinite")
    return matrix


@dataclass(frozen=True)
class TranslationalErrorState:
    body_id: str
    imu_id: str
    map_frame_id: str
    timestamp_ns: int
    imu_sequence_number: int
    clock_domain: str
    position_world_from_body_m: tuple[float, float, float] | None
    velocity_world_from_body_m_s: tuple[float, float, float] | None
    residual_accel_bias_body_m_s2: tuple[float, float, float]
    covariance_p_v_bias: tuple[tuple[float, ...], ...] | None
    status: str
    source: str
    last_position_update_timestamp_ns: int
    position_age_ns: int
    last_position_measurement_source: str
    last_position_source_sequence_number: int
    last_position_source_revision_sha256: str
    last_position_calibration_sha256: str
    last_position_clock_mapping_sha256: str
    localization_source_revision_sha256: str
    localization_calibration_sha256: str
    imu_calibration_sha256: str
    gravity_model_sha256: str
    clock_mapping_sha256: str

    def __post_init__(self) -> None:
        for name in ("body_id", "imu_id", "map_frame_id", "clock_domain"):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        for name in (
            "timestamp_ns",
            "imu_sequence_number",
            "last_position_update_timestamp_ns",
            "position_age_ns",
            "last_position_source_sequence_number",
        ):
            _i64(getattr(self, name), name)
        if self.position_age_ns != self.timestamp_ns - self.last_position_update_timestamp_ns:
            raise ValueError("position_age_ns is inconsistent with timestamps")
        if self.last_position_measurement_source not in {
            PositionMeasurementSource.LOCALIZATION.value,
            PositionMeasurementSource.RTK.value,
        }:
            raise ValueError("invalid last position measurement source")
        if self.status not in {"TRACKING", "DEGRADED", "LOST"}:
            raise ValueError("invalid translational error-state status")
        if self.source not in {"POSITION_SEEDED", "POSITION_CORRECTED", "IMU_PROPAGATED", "LOST"}:
            raise ValueError("invalid translational error-state source")
        bias = _vec3(self.residual_accel_bias_body_m_s2, "residual_accel_bias_body_m_s2")
        object.__setattr__(self, "residual_accel_bias_body_m_s2", tuple(float(v) for v in bias))
        if self.status == "LOST":
            if (
                self.position_world_from_body_m is not None
                or self.velocity_world_from_body_m_s is not None
                or self.covariance_p_v_bias is not None
                or self.source != "LOST"
            ):
                raise ValueError("LOST translational error state must hide position/velocity/covariance")
        else:
            position = _vec3(self.position_world_from_body_m, "position_world_from_body_m")
            velocity = _vec3(self.velocity_world_from_body_m_s, "velocity_world_from_body_m_s")
            covariance = _validated_covariance(self.covariance_p_v_bias)
            object.__setattr__(self, "position_world_from_body_m", tuple(float(v) for v in position))
            object.__setattr__(self, "velocity_world_from_body_m_s", tuple(float(v) for v in velocity))
            object.__setattr__(self, "covariance_p_v_bias", _covariance_tuple(covariance))
        for name in (
            "last_position_source_revision_sha256",
            "last_position_calibration_sha256",
            "last_position_clock_mapping_sha256",
            "localization_source_revision_sha256",
            "localization_calibration_sha256",
            "imu_calibration_sha256",
            "gravity_model_sha256",
            "clock_mapping_sha256",
        ):
            _digest(getattr(self, name), name)

    @property
    def position_stddev_m(self) -> tuple[float, float, float] | None:
        if self.covariance_p_v_bias is None:
            return None
        matrix = np.asarray(self.covariance_p_v_bias)
        return tuple(float(math.sqrt(max(0.0, matrix[i, i]))) for i in range(3))

    @property
    def velocity_stddev_m_s(self) -> tuple[float, float, float] | None:
        if self.covariance_p_v_bias is None:
            return None
        matrix = np.asarray(self.covariance_p_v_bias)
        return tuple(float(math.sqrt(max(0.0, matrix[i, i]))) for i in range(3, 6))

    @property
    def residual_accel_bias_stddev_m_s2(self) -> tuple[float, float, float] | None:
        if self.covariance_p_v_bias is None:
            return None
        matrix = np.asarray(self.covariance_p_v_bias)
        return tuple(float(math.sqrt(max(0.0, matrix[i, i]))) for i in range(6, 9))


@dataclass(frozen=True)
class TranslationalESKFSnapshot:
    """Explicit in-memory rewind checkpoint for one estimator configuration."""

    estimator_fingerprint_sha256: str
    position_world_from_body_m: tuple[float, float, float]
    velocity_world_from_body_m_s: tuple[float, float, float]
    residual_accel_bias_body_m_s2: tuple[float, float, float]
    covariance_p_v_bias: tuple[tuple[float, ...], ...]
    last_sample: CorrectedIMUVectors | None
    last_attitude: LocalizationIMUAttitudeState | None
    state: TranslationalErrorState | None
    expired: bool
    last_position_update_timestamp_ns: int
    last_position_sequences: tuple[tuple[str, int], ...]
    last_position_measurement_source: str
    last_position_source_revision_sha256: str
    last_position_calibration_sha256: str
    last_position_clock_mapping_sha256: str

    def __post_init__(self) -> None:
        _digest(self.estimator_fingerprint_sha256, "estimator_fingerprint_sha256")
        for name in (
            "position_world_from_body_m",
            "velocity_world_from_body_m_s",
            "residual_accel_bias_body_m_s2",
        ):
            value = _vec3(getattr(self, name), name)
            object.__setattr__(self, name, tuple(float(v) for v in value))
        object.__setattr__(
            self,
            "covariance_p_v_bias",
            _covariance_tuple(_validated_covariance(self.covariance_p_v_bias)),
        )
        if self.last_sample is not None and not isinstance(
            self.last_sample, CorrectedIMUVectors
        ):
            raise ValueError("last_sample must be CorrectedIMUVectors or None")
        if self.last_attitude is not None and not isinstance(
            self.last_attitude, LocalizationIMUAttitudeState
        ):
            raise ValueError("last_attitude must be LocalizationIMUAttitudeState or None")
        if self.state is not None and not isinstance(self.state, TranslationalErrorState):
            raise ValueError("state must be TranslationalErrorState or None")
        if (
            (self.last_sample is None)
            != (self.last_attitude is None)
            or (self.last_sample is None) != (self.state is None)
        ):
            raise ValueError(
                "snapshot sample, attitude and state must be absent or present together"
            )
        if self.state is not None:
            if (
                self.last_sample.sensor_timestamp_ns != self.state.timestamp_ns
                or self.last_attitude.timestamp_ns != self.state.timestamp_ns
                or self.last_sample.sequence_number != self.state.imu_sequence_number
            ):
                raise ValueError("snapshot sample/attitude/state timestamp or sequence mismatch")
        if type(self.expired) is not bool:
            raise ValueError("expired must be boolean")
        if self.state is not None and self.expired != (self.state.status == "LOST"):
            raise ValueError("snapshot expired flag must agree with LOST state")
        if self.state is None and self.expired:
            raise ValueError("initial snapshot cannot be expired")
        _i64(
            self.last_position_update_timestamp_ns,
            "last_position_update_timestamp_ns",
        )
        if not isinstance(self.last_position_sequences, tuple):
            raise ValueError("last_position_sequences must be a tuple")
        seen = set()
        for source, sequence in self.last_position_sequences:
            if source not in {
                PositionMeasurementSource.LOCALIZATION.value,
                PositionMeasurementSource.RTK.value,
            }:
                raise ValueError("snapshot contains unsupported position source")
            if source in seen:
                raise ValueError("snapshot contains duplicate position source")
            seen.add(source)
            _i64(sequence, "last position source sequence")
        if self.last_position_measurement_source not in seen:
            raise ValueError("snapshot current position source has no sequence")
        for name in (
            "last_position_source_revision_sha256",
            "last_position_calibration_sha256",
            "last_position_clock_mapping_sha256",
        ):
            _digest(getattr(self, name), name)


class ConditionedTranslationalESKF:
    """9-state p/v/residual-accel-bias filter conditioned on fused attitude."""

    def __init__(
        self,
        seed: TranslationalSeed,
        imu_calibration: IMUCalibration,
        gravity_model: GravityModel,
        *,
        max_sample_gap_ns: int = 20_000_000,
        max_step_ns: int = 5_000_000,
        max_substeps: int = 64,
        max_position_dead_reckoning_ns: int = 100_000_000,
        max_attitude_variance_rad2: float = 0.05,
        max_residual_accel_bias_m_s2: float = 2.0,
        body_origin_tolerance_m: float = 1e-9,
        position_source_contracts: Sequence[PositionSourceContract] = (),
    ) -> None:
        if not isinstance(seed, TranslationalSeed):
            raise ValueError("seed must be TranslationalSeed")
        if not isinstance(imu_calibration, IMUCalibration):
            raise ValueError("imu_calibration must be IMUCalibration")
        if not isinstance(gravity_model, GravityModel):
            raise ValueError("gravity_model must be GravityModel")
        if seed.imu_id != imu_calibration.imu_id:
            raise ValueError("translation seed IMU ID does not match calibration")
        if seed.clock_domain != imu_calibration.imu_clock_domain:
            raise ValueError("translation seed clock does not match IMU calibration")
        if seed.map_frame_id != gravity_model.frame_id:
            raise ValueError("gravity frame does not match translation map frame")
        if float(np.linalg.norm(imu_calibration.translation_body_from_imu_m)) > _real(
            body_origin_tolerance_m, "body_origin_tolerance_m", minimum=0.0
        ):
            raise ValueError("nonzero IMU lever arm requires rotational acceleration compensation")

        self.max_gap = _i64(max_sample_gap_ns, "max_sample_gap_ns")
        self.max_step = _i64(max_step_ns, "max_step_ns")
        self.max_substeps = _i64(max_substeps, "max_substeps")
        self.max_dead_reckoning = _i64(max_position_dead_reckoning_ns, "max_position_dead_reckoning_ns")
        if min(self.max_gap, self.max_step, self.max_substeps, self.max_dead_reckoning) < 1:
            raise ValueError("gap/step/substep/dead-reckoning limits must be positive")
        if self.max_step > self.max_gap:
            raise ValueError("max_step_ns must not exceed max_sample_gap_ns")
        self.max_attitude_variance = _real(
            max_attitude_variance_rad2, "max_attitude_variance_rad2", minimum=0.0
        )
        self.max_bias = _real(
            max_residual_accel_bias_m_s2, "max_residual_accel_bias_m_s2", minimum=0.0
        )
        if self.max_attitude_variance <= 0.0 or self.max_bias <= 0.0:
            raise ValueError("attitude variance and residual bias limits must be positive")

        self.seed = seed
        self.calibration = imu_calibration
        self.gravity = gravity_model
        self.body_origin_tolerance = _real(
            body_origin_tolerance_m,
            "body_origin_tolerance_m",
            minimum=0.0,
        )
        if (
            not isinstance(position_source_contracts, Sequence)
            or isinstance(position_source_contracts, (str, bytes))
        ):
            raise ValueError("position_source_contracts must be a sequence")
        contracts = {
            PositionMeasurementSource.LOCALIZATION: PositionSourceContract(
                source=PositionMeasurementSource.LOCALIZATION,
                source_revision_sha256=seed.localization_source_revision_sha256,
                measurement_calibration_sha256=seed.localization_calibration_sha256,
                clock_mapping_sha256=seed.clock_mapping_sha256,
            )
        }
        for contract in position_source_contracts:
            if not isinstance(contract, PositionSourceContract):
                raise ValueError(
                    "position_source_contracts must contain PositionSourceContract"
                )
            if contract.source in contracts:
                raise ValueError(
                    f"{contract.source.value} source contract is already pinned by estimator seed"
                )
            contracts[contract.source] = contract
        self._position_source_contracts = contracts
        self._fingerprint = _canonical_digest({
            "body_id": seed.body_id,
            "imu_id": seed.imu_id,
            "map_frame_id": seed.map_frame_id,
            "seed_timestamp_ns": seed.timestamp_ns,
            "seed_source_sequence_number": seed.source_sequence_number,
            "seed_clock_domain": seed.clock_domain,
            "seed_position_world_from_body_m": list(seed.position_world_from_body_m),
            "seed_velocity_world_from_body_m_s": list(seed.velocity_world_from_body_m_s),
            "seed_residual_accel_bias_body_m_s2": list(seed.residual_accel_bias_body_m_s2),
            "seed_position_stddev_m": list(seed.position_stddev_m),
            "seed_velocity_stddev_m_s": list(seed.velocity_stddev_m_s),
            "seed_residual_accel_bias_stddev_m_s2": list(
                seed.residual_accel_bias_stddev_m_s2
            ),
            "localization_source_revision_sha256": seed.localization_source_revision_sha256,
            "localization_calibration_sha256": seed.localization_calibration_sha256,
            "clock_mapping_sha256": seed.clock_mapping_sha256,
            "imu_calibration_sha256": imu_calibration.calibration_sha256,
            "gravity_model_sha256": gravity_model.revision_sha256,
            "max_sample_gap_ns": self.max_gap,
            "max_step_ns": self.max_step,
            "max_substeps": self.max_substeps,
            "max_position_dead_reckoning_ns": self.max_dead_reckoning,
            "max_attitude_variance_rad2": self.max_attitude_variance,
            "max_residual_accel_bias_m_s2": self.max_bias,
            "body_origin_tolerance_m": self.body_origin_tolerance,
            "position_source_contracts": [
                {
                    "source": contract.source.value,
                    "source_revision_sha256": contract.source_revision_sha256,
                    "measurement_calibration_sha256": contract.measurement_calibration_sha256,
                    "clock_mapping_sha256": contract.clock_mapping_sha256,
                }
                for contract in sorted(
                    self._position_source_contracts.values(),
                    key=lambda item: item.source.value,
                )
            ],
        })
        self._position = _vec3(seed.position_world_from_body_m, "seed position")
        self._velocity = _vec3(seed.velocity_world_from_body_m_s, "seed velocity")
        self._bias = _vec3(seed.residual_accel_bias_body_m_s2, "seed residual accel bias")
        diagonal = np.concatenate((
            np.square(seed.position_stddev_m),
            np.square(seed.velocity_stddev_m_s),
            np.square(seed.residual_accel_bias_stddev_m_s2),
        ))
        self._covariance = np.diag(diagonal)
        self._last_sample: CorrectedIMUVectors | None = None
        self._last_attitude: LocalizationIMUAttitudeState | None = None
        self._state: TranslationalErrorState | None = None
        self._expired = False
        self._last_position_update_timestamp_ns = seed.timestamp_ns
        self._last_position_sequences = {
            PositionMeasurementSource.LOCALIZATION: seed.source_sequence_number
        }
        self._last_position_measurement_source = PositionMeasurementSource.LOCALIZATION
        self._last_position_source_revision_sha256 = seed.localization_source_revision_sha256
        self._last_position_calibration_sha256 = seed.localization_calibration_sha256
        self._last_position_clock_mapping_sha256 = seed.clock_mapping_sha256

    @property
    def state(self) -> TranslationalErrorState | None:
        return self._state

    @property
    def estimator_fingerprint_sha256(self) -> str:
        return self._fingerprint

    def snapshot(self) -> TranslationalESKFSnapshot:
        """Capture an explicit bounded-replay checkpoint without raw object copying."""
        sequences = tuple(sorted(
            (source.value, sequence)
            for source, sequence in self._last_position_sequences.items()
        ))
        return TranslationalESKFSnapshot(
            estimator_fingerprint_sha256=self._fingerprint,
            position_world_from_body_m=tuple(float(v) for v in self._position),
            velocity_world_from_body_m_s=tuple(float(v) for v in self._velocity),
            residual_accel_bias_body_m_s2=tuple(float(v) for v in self._bias),
            covariance_p_v_bias=_covariance_tuple(self._covariance),
            last_sample=self._last_sample,
            last_attitude=self._last_attitude,
            state=self._state,
            expired=self._expired,
            last_position_update_timestamp_ns=self._last_position_update_timestamp_ns,
            last_position_sequences=sequences,
            last_position_measurement_source=self._last_position_measurement_source.value,
            last_position_source_revision_sha256=self._last_position_source_revision_sha256,
            last_position_calibration_sha256=self._last_position_calibration_sha256,
            last_position_clock_mapping_sha256=self._last_position_clock_mapping_sha256,
        )

    def restore(self, snapshot: TranslationalESKFSnapshot) -> TranslationalErrorState | None:
        """Restore only a checkpoint produced by an equivalent estimator configuration."""
        if not isinstance(snapshot, TranslationalESKFSnapshot):
            raise ValueError("snapshot must be TranslationalESKFSnapshot")
        if snapshot.estimator_fingerprint_sha256 != self._fingerprint:
            raise ValueError("snapshot estimator fingerprint mismatch")
        if snapshot.last_sample is not None:
            if (
                snapshot.last_sample.imu_id != self.seed.imu_id
                or snapshot.last_sample.clock_domain != self.seed.clock_domain
                or snapshot.last_sample.calibration_sha256
                != self.calibration.calibration_sha256
            ):
                raise ValueError("snapshot IMU sample identity/revision mismatch")
        if snapshot.last_attitude is not None:
            attitude = snapshot.last_attitude
            if (
                attitude.body_id != self.seed.body_id
                or attitude.imu_id != self.seed.imu_id
                or attitude.map_frame_id != self.seed.map_frame_id
                or attitude.clock_domain != self.seed.clock_domain
                or attitude.imu_calibration_sha256
                != self.calibration.calibration_sha256
                or attitude.localization_source_revision_sha256
                != self.seed.localization_source_revision_sha256
                or attitude.localization_calibration_sha256
                != self.seed.localization_calibration_sha256
                or attitude.clock_mapping_sha256 != self.seed.clock_mapping_sha256
            ):
                raise ValueError("snapshot attitude identity/revision mismatch")
        if snapshot.state is not None:
            state = snapshot.state
            if (
                state.body_id != self.seed.body_id
                or state.imu_id != self.seed.imu_id
                or state.map_frame_id != self.seed.map_frame_id
                or state.clock_domain != self.seed.clock_domain
                or state.imu_calibration_sha256
                != self.calibration.calibration_sha256
                or state.localization_source_revision_sha256
                != self.seed.localization_source_revision_sha256
                or state.localization_calibration_sha256
                != self.seed.localization_calibration_sha256
                or state.clock_mapping_sha256 != self.seed.clock_mapping_sha256
            ):
                raise ValueError("snapshot state identity/revision mismatch")

        position_sequences = {
            PositionMeasurementSource(source): sequence
            for source, sequence in snapshot.last_position_sequences
        }
        source = PositionMeasurementSource(snapshot.last_position_measurement_source)

        self._position = np.asarray(
            snapshot.position_world_from_body_m, dtype=np.float64
        ).copy()
        self._velocity = np.asarray(
            snapshot.velocity_world_from_body_m_s, dtype=np.float64
        ).copy()
        self._bias = np.asarray(
            snapshot.residual_accel_bias_body_m_s2, dtype=np.float64
        ).copy()
        self._covariance = _validated_covariance(
            snapshot.covariance_p_v_bias
        ).copy()
        self._last_sample = snapshot.last_sample
        self._last_attitude = snapshot.last_attitude
        self._state = snapshot.state
        self._expired = snapshot.expired
        self._last_position_update_timestamp_ns = (
            snapshot.last_position_update_timestamp_ns
        )
        self._last_position_sequences = position_sequences
        self._last_position_measurement_source = source
        self._last_position_source_revision_sha256 = (
            snapshot.last_position_source_revision_sha256
        )
        self._last_position_calibration_sha256 = (
            snapshot.last_position_calibration_sha256
        )
        self._last_position_clock_mapping_sha256 = (
            snapshot.last_position_clock_mapping_sha256
        )
        return self._state

    def _validate_pair(
        self,
        sample: CorrectedIMUVectors,
        attitude: LocalizationIMUAttitudeState,
    ) -> None:
        if not isinstance(sample, CorrectedIMUVectors):
            raise ValueError("sample must be CorrectedIMUVectors")
        if not isinstance(attitude, LocalizationIMUAttitudeState):
            raise ValueError("attitude must be LocalizationIMUAttitudeState")
        if sample.imu_id != self.seed.imu_id or attitude.imu_id != self.seed.imu_id:
            raise ValueError("IMU identity mismatch")
        if attitude.body_id != self.seed.body_id:
            raise ValueError("attitude body mismatch")
        if attitude.map_frame_id != self.seed.map_frame_id:
            raise ValueError("attitude map frame mismatch")
        if sample.clock_domain != self.seed.clock_domain or attitude.clock_domain != self.seed.clock_domain:
            raise ValueError("sample/attitude clock mismatch")
        if sample.sensor_timestamp_ns != attitude.timestamp_ns:
            raise ValueError("specific force and attitude timestamps must match")
        if sample.calibration_sha256 != self.calibration.calibration_sha256:
            raise ValueError("corrected IMU calibration mismatch")
        if attitude.imu_calibration_sha256 != self.calibration.calibration_sha256:
            raise ValueError("attitude IMU calibration mismatch")
        if attitude.localization_source_revision_sha256 != self.seed.localization_source_revision_sha256:
            raise ValueError("attitude localization source revision mismatch")
        if attitude.localization_calibration_sha256 != self.seed.localization_calibration_sha256:
            raise ValueError("attitude localization calibration mismatch")
        if attitude.clock_mapping_sha256 != self.seed.clock_mapping_sha256:
            raise ValueError("attitude localization clock mapping mismatch")
        if attitude.status == "LOST" or attitude.quaternion_world_from_body_xyzw is None:
            raise ValueError("translation ESKF requires publishable attitude")
        if (
            attitude.attitude_variance_rad2 is None
            or attitude.attitude_variance_rad2 > self.max_attitude_variance
        ):
            raise ValueError("attitude uncertainty exceeds translation ESKF limit")

    def _stabilize_covariance(self, covariance: np.ndarray) -> np.ndarray:
        covariance = 0.5 * (covariance + covariance.T)
        eigenvalues = np.linalg.eigvalsh(covariance)
        minimum = float(np.min(eigenvalues))
        if minimum < -1e-8:
            raise ValueError("translational covariance lost positive semidefiniteness")
        if minimum < 0.0:
            covariance = covariance + np.eye(9) * (-minimum + 1e-15)
        if not np.isfinite(covariance).all():
            raise ValueError("translational covariance is non-finite")
        return covariance

    def _publish(self, timestamp_ns: int, sequence: int, status: str, source: str) -> TranslationalErrorState:
        visible = status != "LOST"
        state = TranslationalErrorState(
            body_id=self.seed.body_id,
            imu_id=self.seed.imu_id,
            map_frame_id=self.seed.map_frame_id,
            timestamp_ns=timestamp_ns,
            imu_sequence_number=sequence,
            clock_domain=self.seed.clock_domain,
            position_world_from_body_m=tuple(self._position) if visible else None,
            velocity_world_from_body_m_s=tuple(self._velocity) if visible else None,
            residual_accel_bias_body_m_s2=tuple(self._bias),
            covariance_p_v_bias=_covariance_tuple(self._covariance) if visible else None,
            status=status,
            source=source,
            last_position_update_timestamp_ns=self._last_position_update_timestamp_ns,
            position_age_ns=timestamp_ns - self._last_position_update_timestamp_ns,
            last_position_measurement_source=self._last_position_measurement_source.value,
            last_position_source_sequence_number=self._last_position_sequences[
                self._last_position_measurement_source
            ],
            last_position_source_revision_sha256=self._last_position_source_revision_sha256,
            last_position_calibration_sha256=self._last_position_calibration_sha256,
            last_position_clock_mapping_sha256=self._last_position_clock_mapping_sha256,
            localization_source_revision_sha256=self.seed.localization_source_revision_sha256,
            localization_calibration_sha256=self.seed.localization_calibration_sha256,
            imu_calibration_sha256=self.calibration.calibration_sha256,
            gravity_model_sha256=self.gravity.revision_sha256,
            clock_mapping_sha256=self.seed.clock_mapping_sha256,
        )
        self._state = state
        return state

    def advance(
        self,
        sample: CorrectedIMUVectors,
        attitude: LocalizationIMUAttitudeState,
    ) -> TranslationalErrorState:
        if self._expired:
            raise ValueError("translation ESKF expired; explicit full-state reseed required")
        self._validate_pair(sample, attitude)
        previous_sample = self._last_sample
        previous_attitude = self._last_attitude
        if previous_sample is None:
            if sample.sensor_timestamp_ns != self.seed.timestamp_ns:
                raise ValueError("first IMU sample must align exactly with translation seed")
            self._last_sample = sample
            self._last_attitude = attitude
            return self._publish(sample.sensor_timestamp_ns, sample.sequence_number, "TRACKING", "POSITION_SEEDED")
        if sample.sequence_number <= previous_sample.sequence_number:
            raise ValueError("IMU sequence is duplicate or out of order")
        if sample.sensor_timestamp_ns <= previous_sample.sensor_timestamp_ns:
            raise ValueError("IMU timestamp is duplicate or regressed")
        dt_ns = sample.sensor_timestamp_ns - previous_sample.sensor_timestamp_ns
        if dt_ns > self.max_gap:
            raise ValueError("IMU sample gap exceeds translation ESKF limit")
        substeps = (dt_ns + self.max_step - 1) // self.max_step
        if substeps > self.max_substeps:
            raise ValueError("translation ESKF interval exceeds bounded substep capacity")

        previous_force = _vec3(previous_sample.specific_force_body_m_s2, "previous specific force")
        current_force = _vec3(sample.specific_force_body_m_s2, "current specific force")
        previous_rotation = Rotation.from_quat(previous_attitude.quaternion_world_from_body_xyzw)
        current_rotation = Rotation.from_quat(attitude.quaternion_world_from_body_xyzw)
        slerp = Slerp([0.0, 1.0], Rotation.from_quat([
            previous_rotation.as_quat(), current_rotation.as_quat()
        ]))
        previous_attitude_var = previous_attitude.attitude_variance_rad2
        current_attitude_var = attitude.attitude_variance_rad2
        accel_noise_body = np.diag(np.square(
            np.asarray(self.calibration.accel_noise_density_m_s2_sqrt_hz, dtype=np.float64)
        ))
        bias_rw_body = np.diag(np.square(
            np.asarray(self.calibration.accel_bias_random_walk_m_s2_per_sqrt_s, dtype=np.float64)
        ))
        gravity = _vec3(self.gravity.gravity_world_m_s2, "gravity")

        candidate_p = self._position.copy()
        candidate_v = self._velocity.copy()
        candidate_b = self._bias.copy()
        candidate_P = self._covariance.copy()

        base_step_ns, remainder = divmod(dt_ns, substeps)
        elapsed_ns = 0
        total_ns = float(dt_ns)
        for index in range(substeps):
            step_ns = base_step_ns + (1 if index < remainder else 0)
            midpoint_ns = elapsed_ns + 0.5 * step_ns
            fraction = midpoint_ns / total_ns
            dt = step_ns / 1_000_000_000.0
            rotation = slerp([fraction]).as_matrix()[0]
            force = (1.0 - fraction) * previous_force + fraction * current_force
            specific_force = force - candidate_b
            acceleration = rotation @ specific_force + gravity
            if not np.isfinite(acceleration).all():
                raise ValueError("world acceleration is non-finite")

            candidate_p = candidate_p + candidate_v * dt + 0.5 * acceleration * dt * dt
            candidate_v = candidate_v + acceleration * dt

            identity = np.eye(3)
            F = np.eye(9)
            F[0:3, 3:6] = identity * dt
            F[0:3, 6:9] = -0.5 * rotation * (dt ** 2)
            F[3:6, 6:9] = -rotation * dt

            Qa = rotation @ accel_noise_body @ rotation.T
            Qb_world = rotation @ bias_rw_body @ rotation.T
            Q = np.zeros((9, 9), dtype=np.float64)
            Q[0:3, 0:3] += Qa * (dt ** 3) / 3.0
            Q[0:3, 3:6] += Qa * (dt ** 2) / 2.0
            Q[3:6, 0:3] += Qa * (dt ** 2) / 2.0
            Q[3:6, 3:6] += Qa * dt

            # Bias random walk propagates through -R*b into velocity/position.
            Q[0:3, 0:3] += Qb_world * (dt ** 5) / 20.0
            Q[0:3, 3:6] += Qb_world * (dt ** 4) / 8.0
            Q[3:6, 0:3] += Qb_world * (dt ** 4) / 8.0
            Q[3:6, 3:6] += Qb_world * (dt ** 3) / 3.0
            Q[0:3, 6:9] += -rotation @ bias_rw_body * (dt ** 3) / 6.0
            Q[6:9, 0:3] += Q[0:3, 6:9].T
            Q[3:6, 6:9] += -rotation @ bias_rw_body * (dt ** 2) / 2.0
            Q[6:9, 3:6] += Q[3:6, 6:9].T
            Q[6:9, 6:9] += bias_rw_body * dt

            attitude_variance = (
                (1.0 - fraction) * previous_attitude_var
                + fraction * current_attitude_var
            )
            accel_from_attitude_var = (
                float(np.dot(specific_force, specific_force))
                * attitude_variance
            )
            Qa_att = np.eye(3) * accel_from_attitude_var
            Q[0:3, 0:3] += Qa_att * 0.25 * (dt ** 4)
            Q[0:3, 3:6] += Qa_att * 0.5 * (dt ** 3)
            Q[3:6, 0:3] += Qa_att * 0.5 * (dt ** 3)
            Q[3:6, 3:6] += Qa_att * (dt ** 2)

            candidate_P = self._stabilize_covariance(F @ candidate_P @ F.T + Q)
            elapsed_ns += step_ns

        if not (
            np.isfinite(candidate_p).all()
            and np.isfinite(candidate_v).all()
            and np.isfinite(candidate_b).all()
        ):
            raise ValueError("translation ESKF produced non-finite nominal state")

        self._position = candidate_p
        self._velocity = candidate_v
        self._bias = candidate_b
        self._covariance = candidate_P
        self._last_sample = sample
        self._last_attitude = attitude

        age = sample.sensor_timestamp_ns - self._last_position_update_timestamp_ns
        if age > self.max_dead_reckoning:
            self._expired = True
            return self._publish(sample.sensor_timestamp_ns, sample.sequence_number, "LOST", "LOST")
        return self._publish(sample.sensor_timestamp_ns, sample.sequence_number, "DEGRADED", "IMU_PROPAGATED")

    def ingest_position_measurement(
        self,
        measurement: MetricPositionMeasurement,
        *,
        max_position_nis: float = 16.0,
    ) -> TranslationalErrorState:
        if self._expired:
            raise ValueError("translation ESKF expired; explicit full-state reseed required")
        if not isinstance(measurement, MetricPositionMeasurement):
            raise ValueError("measurement must be MetricPositionMeasurement")
        if self._state is None or self._state.status == "LOST":
            raise ValueError("position update requires a current translation ESKF state")
        if measurement.body_id != self.seed.body_id:
            raise ValueError("position measurement body mismatch")
        if measurement.map_frame_id != self.seed.map_frame_id:
            raise ValueError("position measurement map frame mismatch")
        if measurement.clock_domain != self.seed.clock_domain:
            raise ValueError("position measurement clock mismatch")
        if measurement.timestamp_ns != self._state.timestamp_ns:
            raise ValueError("position measurement must align exactly with current ESKF timestamp")
        contract = self._position_source_contracts.get(measurement.source)
        if contract is None:
            raise ValueError(
                f"{measurement.source.value} position source contract is not configured"
            )
        if measurement.source_revision_sha256 != contract.source_revision_sha256:
            raise ValueError(
                f"{measurement.source.value.lower()} position source revision mismatch"
            )
        if (
            measurement.measurement_calibration_sha256
            != contract.measurement_calibration_sha256
        ):
            raise ValueError(
                f"{measurement.source.value.lower()} position calibration mismatch"
            )
        if measurement.clock_mapping_sha256 != contract.clock_mapping_sha256:
            raise ValueError(
                f"{measurement.source.value.lower()} position clock mapping mismatch"
            )
        last_sequence = self._last_position_sequences.get(measurement.source)
        if last_sequence is not None and measurement.source_sequence_number <= last_sequence:
            raise ValueError("position measurement sequence is duplicate or out of order")

        nis_limit = _real(max_position_nis, "max_position_nis", minimum=0.0)
        if nis_limit <= 0.0:
            raise ValueError("max_position_nis must be positive")
        z = _vec3(measurement.position_world_from_body_m, "position measurement")
        Rm = np.diag(np.square(_vec3(measurement.position_stddev_m, "position stddev")))
        residual = z - self._position
        S = self._covariance[0:3, 0:3] + Rm
        try:
            solved_residual = np.linalg.solve(S, residual)
            K = np.linalg.solve(S, self._covariance[:, 0:3].T).T
        except np.linalg.LinAlgError as exc:
            raise ValueError("position innovation covariance is singular") from exc
        nis = float(residual @ solved_residual)
        if not math.isfinite(nis) or nis > nis_limit:
            raise ValueError("position innovation exceeds configured NIS limit")

        dx = K @ residual
        candidate_p = self._position + dx[0:3]
        candidate_v = self._velocity + dx[3:6]
        candidate_b = self._bias + dx[6:9]
        if float(np.linalg.norm(candidate_b)) > self.max_bias:
            raise ValueError("position update would exceed residual accel bias limit")
        H = np.zeros((3, 9), dtype=np.float64)
        H[:, 0:3] = np.eye(3)
        identity = np.eye(9)
        joseph_left = identity - K @ H
        candidate_P = self._stabilize_covariance(
            joseph_left @ self._covariance @ joseph_left.T + K @ Rm @ K.T
        )
        if not (
            np.isfinite(candidate_p).all()
            and np.isfinite(candidate_v).all()
            and np.isfinite(candidate_b).all()
        ):
            raise ValueError("position update produced non-finite nominal state")

        self._position = candidate_p
        self._velocity = candidate_v
        self._bias = candidate_b
        self._covariance = candidate_P
        self._last_position_update_timestamp_ns = measurement.timestamp_ns
        self._last_position_sequences[measurement.source] = measurement.source_sequence_number
        self._last_position_measurement_source = measurement.source
        self._last_position_source_revision_sha256 = measurement.source_revision_sha256
        self._last_position_calibration_sha256 = measurement.measurement_calibration_sha256
        self._last_position_clock_mapping_sha256 = measurement.clock_mapping_sha256
        status = (
            "TRACKING"
            if measurement.quality is LocalizationQuality.TRACKING
            else "DEGRADED"
        )
        return self._publish(
            self._state.timestamp_ns,
            self._state.imu_sequence_number,
            status,
            "POSITION_CORRECTED",
        )
