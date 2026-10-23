"""Bounded deterministic delayed-position rewind/replay for translational ESKF.

Position measurements are never retimestamped to arrival time. A delayed Tag or
RTK correction is accepted only when its mapped timestamp exactly matches a
retained IMU/attitude frame. The owner restores an explicit ESKF checkpoint,
inserts the measurement in canonical order, and deterministically replays all
retained frames to the current timestamp. Replay failure is transactional.

This owner does not replay attitude fusion. It consumes already-qualified,
historical LocalizationIMUAttitudeState records at the same IMU timestamps.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Sequence

from imu_calibration import CorrectedIMUVectors, IMUCalibration
from imu_interval_interpolation import interpolate_imu_observation_epoch
from localization_imu_fusion import LocalizationIMUAttitudeState
from position_measurement import (
    MetricPositionMeasurement,
    PositionMeasurementSource,
    PositionSourceContract,
)
from translational_eskf import (
    ConditionedTranslationalESKF,
    TranslationalErrorState,
    TranslationalESKFSnapshot,
)
from translational_propagation import GravityModel, TranslationalSeed


_MAX_I64 = (1 << 63) - 1


def _positive_int(value: object, name: str, *, maximum: int) -> int:
    if type(value) is not int or not 1 <= value <= maximum:
        raise ValueError(f"{name} must be an integer within 1..{maximum}")
    return value


def _nonnegative_i64(value: object, name: str) -> int:
    if type(value) is not int or not 0 <= value <= _MAX_I64:
        raise ValueError(f"{name} must be a nonnegative signed-64-bit integer")
    return value


@dataclass(frozen=True)
class ReplayPositionUpdate:
    measurement: MetricPositionMeasurement
    max_position_nis: float

    def __post_init__(self) -> None:
        if not isinstance(self.measurement, MetricPositionMeasurement):
            raise ValueError("measurement must be MetricPositionMeasurement")
        if isinstance(self.max_position_nis, bool) or not isinstance(
            self.max_position_nis, (int, float)
        ):
            raise ValueError("max_position_nis must be positive finite numeric")
        value = float(self.max_position_nis)
        if not (value > 0.0 and value < float("inf")):
            raise ValueError("max_position_nis must be positive finite numeric")
        object.__setattr__(self, "max_position_nis", value)


@dataclass(frozen=True)
class TranslationalReplayFrame:
    sample: CorrectedIMUVectors
    attitude: LocalizationIMUAttitudeState
    position_updates: tuple[ReplayPositionUpdate, ...]
    snapshot_after_frame: TranslationalESKFSnapshot
    interior_position_updates: tuple[ReplayPositionUpdate, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.sample, CorrectedIMUVectors):
            raise ValueError("sample must be CorrectedIMUVectors")
        if not isinstance(self.attitude, LocalizationIMUAttitudeState):
            raise ValueError("attitude must be LocalizationIMUAttitudeState")
        if self.sample.sensor_timestamp_ns != self.attitude.timestamp_ns:
            raise ValueError("replay frame sample/attitude timestamps must match")
        if not isinstance(self.position_updates, tuple):
            raise ValueError("position_updates must be a tuple")
        if any(
            not isinstance(update, ReplayPositionUpdate)
            for update in self.position_updates
        ):
            raise ValueError("position_updates must contain ReplayPositionUpdate")
        if any(
            update.measurement.timestamp_ns != self.sample.sensor_timestamp_ns
            for update in self.position_updates
        ):
            raise ValueError("frame position updates must match the real IMU timestamp")
        if not isinstance(self.interior_position_updates, tuple):
            raise ValueError("interior_position_updates must be a tuple")
        if any(
            not isinstance(update, ReplayPositionUpdate)
            for update in self.interior_position_updates
        ):
            raise ValueError("interior_position_updates must contain ReplayPositionUpdate")
        if any(
            update.measurement.timestamp_ns == self.sample.sensor_timestamp_ns
            for update in self.interior_position_updates
        ):
            raise ValueError("interior position updates must not match the right IMU timestamp")
        if not isinstance(self.snapshot_after_frame, TranslationalESKFSnapshot):
            raise ValueError("snapshot_after_frame must be TranslationalESKFSnapshot")

        sources = [update.measurement.source for update in self.position_updates]
        if len(set(sources)) != len(sources):
            raise ValueError("one replay frame may contain at most one update per source")
        canonical = tuple(sorted(
            self.position_updates,
            key=lambda update: (
                update.measurement.source.value,
                update.measurement.source_sequence_number,
            ),
        ))
        if canonical != self.position_updates:
            raise ValueError("position_updates must use canonical source ordering")

        interior_canonical = tuple(sorted(
            self.interior_position_updates,
            key=lambda update: (
                update.measurement.timestamp_ns,
                update.measurement.source.value,
                update.measurement.source_sequence_number,
            ),
        ))
        if interior_canonical != self.interior_position_updates:
            raise ValueError("interior position updates must use canonical timestamp/source ordering")
        interior_keys = [
            (update.measurement.timestamp_ns, update.measurement.source)
            for update in self.interior_position_updates
        ]
        if len(set(interior_keys)) != len(interior_keys):
            raise ValueError("one interior replay timestamp may contain only one update per source")

    @property
    def timestamp_ns(self) -> int:
        return self.sample.sensor_timestamp_ns


class BoundedDelayedPositionReplay:
    """Single-owner bounded history around ConditionedTranslationalESKF."""

    def __init__(
        self,
        seed: TranslationalSeed,
        imu_calibration: IMUCalibration,
        gravity_model: GravityModel,
        *,
        max_history_frames: int = 256,
        max_history_ns: int = 500_000_000,
        max_sample_gap_ns: int = 20_000_000,
        max_step_ns: int = 5_000_000,
        max_substeps: int = 64,
        max_position_dead_reckoning_ns: int = 100_000_000,
        max_attitude_variance_rad2: float = 0.05,
        max_residual_accel_bias_m_s2: float = 2.0,
        body_origin_tolerance_m: float = 1e-9,
        position_source_contracts: Sequence[PositionSourceContract] = (),
    ) -> None:
        self.max_history_frames = _positive_int(
            max_history_frames, "max_history_frames", maximum=100_000
        )
        self.max_history_ns = _positive_int(
            max_history_ns, "max_history_ns", maximum=_MAX_I64
        )
        if (
            not isinstance(position_source_contracts, Sequence)
            or isinstance(position_source_contracts, (str, bytes))
        ):
            raise ValueError("position_source_contracts must be a sequence")
        self._seed = seed
        self._calibration = imu_calibration
        self._gravity = gravity_model
        self._filter_kwargs = dict(
            max_sample_gap_ns=max_sample_gap_ns,
            max_step_ns=max_step_ns,
            max_substeps=max_substeps,
            max_position_dead_reckoning_ns=max_position_dead_reckoning_ns,
            max_attitude_variance_rad2=max_attitude_variance_rad2,
            max_residual_accel_bias_m_s2=max_residual_accel_bias_m_s2,
            body_origin_tolerance_m=body_origin_tolerance_m,
            position_source_contracts=tuple(position_source_contracts),
        )
        self._filter = self._new_filter()
        self._base_snapshot = self._filter.snapshot()
        self._frames: list[TranslationalReplayFrame] = []

    def _new_filter(self) -> ConditionedTranslationalESKF:
        return ConditionedTranslationalESKF(
            self._seed,
            self._calibration,
            self._gravity,
            **self._filter_kwargs,
        )

    @property
    def state(self) -> TranslationalErrorState | None:
        return self._filter.state

    @property
    def retained_frame_count(self) -> int:
        return len(self._frames)

    @property
    def retained_timestamp_range_ns(self) -> tuple[int, int] | None:
        if not self._frames:
            return None
        return (self._frames[0].timestamp_ns, self._frames[-1].timestamp_ns)

    def _prune(self) -> None:
        if not self._frames:
            return
        newest = self._frames[-1].timestamp_ns
        while self._frames and (
            len(self._frames) > self.max_history_frames
            or newest - self._frames[0].timestamp_ns > self.max_history_ns
        ):
            removed = self._frames.pop(0)
            self._base_snapshot = removed.snapshot_after_frame

    def ingest_imu(
        self,
        sample: CorrectedIMUVectors,
        attitude: LocalizationIMUAttitudeState,
    ) -> TranslationalErrorState:
        """Append one current IMU/attitude frame; IMU history itself is never reordered."""
        if not isinstance(sample, CorrectedIMUVectors):
            raise ValueError("sample must be CorrectedIMUVectors")
        if not isinstance(attitude, LocalizationIMUAttitudeState):
            raise ValueError("attitude must be LocalizationIMUAttitudeState")
        if self._frames:
            last = self._frames[-1]
            if sample.sensor_timestamp_ns <= last.timestamp_ns:
                raise ValueError("replay owner IMU timestamp is duplicate or out of order")
            if sample.sequence_number <= last.sample.sequence_number:
                raise ValueError("replay owner IMU sequence is duplicate or out of order")
        state = self._filter.advance(sample, attitude)
        frame = TranslationalReplayFrame(
            sample=sample,
            attitude=attitude,
            position_updates=(),
            snapshot_after_frame=self._filter.snapshot(),
        )
        self._frames.append(frame)
        self._prune()
        return state

    def _validate_position_sequence(
        self,
        measurement: MetricPositionMeasurement,
    ) -> None:
        source = measurement.source
        base_sequences = {
            PositionMeasurementSource(name): sequence
            for name, sequence in self._base_snapshot.last_position_sequences
        }
        base_sequence = base_sequences.get(source)
        if (
            base_sequence is not None
            and measurement.source_sequence_number <= base_sequence
        ):
            raise ValueError(
                "delayed position measurement sequence predates replay checkpoint"
            )

        existing_updates = (
            update
            for frame in self._frames
            for update in frame.position_updates + frame.interior_position_updates
        )
        for update in existing_updates:
            existing = update.measurement
            if existing.source is not source:
                continue
            if existing.timestamp_ns == measurement.timestamp_ns:
                raise ValueError("one replay timestamp may contain only one update per source")
            if (
                existing.timestamp_ns < measurement.timestamp_ns
                and existing.source_sequence_number >= measurement.source_sequence_number
            ):
                raise ValueError(
                    "delayed position source sequence disagrees with timestamp order"
                )
            if (
                existing.timestamp_ns > measurement.timestamp_ns
                and existing.source_sequence_number <= measurement.source_sequence_number
            ):
                raise ValueError(
                    "delayed position source sequence disagrees with timestamp order"
                )

    def _interpolated_pair(
        self,
        left_sample: CorrectedIMUVectors,
        left_attitude: LocalizationIMUAttitudeState,
        right: TranslationalReplayFrame,
        timestamp_ns: int,
    ) -> tuple[CorrectedIMUVectors, LocalizationIMUAttitudeState]:
        epoch = interpolate_imu_observation_epoch(
            left_sample, left_attitude, right.sample, right.attitude, timestamp_ns
        )
        sample = CorrectedIMUVectors(
            imu_id=epoch.imu_id,
            sequence_number=epoch.left_sequence_number,
            sensor_timestamp_ns=epoch.timestamp_ns,
            clock_domain=epoch.clock_domain,
            calibration_revision=left_sample.calibration_revision,
            calibration_sha256=epoch.calibration_sha256,
            gyro_body_rad_s=epoch.gyro_body_rad_s,
            specific_force_body_m_s2=epoch.specific_force_body_m_s2,
        )
        last_localization_timestamp = left_attitude.last_localization_timestamp_ns
        localization_age = (
            None if last_localization_timestamp is None
            else timestamp_ns - last_localization_timestamp
        )
        attitude = replace(
            left_attitude,
            timestamp_ns=timestamp_ns,
            quaternion_world_from_body_xyzw=epoch.quaternion_world_from_body_xyzw,
            attitude_variance_rad2=epoch.attitude_variance_rad2,
            status="DEGRADED",
            source="IMU_PROPAGATED",
            localization_age_ns=localization_age,
        )
        return sample, attitude

    def _replay(
        self,
        frames: Sequence[TranslationalReplayFrame],
    ) -> tuple[
        ConditionedTranslationalESKF,
        list[TranslationalReplayFrame],
    ]:
        candidate = self._new_filter()
        candidate.restore(self._base_snapshot)
        rebuilt: list[TranslationalReplayFrame] = []
        for index, frame in enumerate(frames):
            if frame.interior_position_updates:
                if index == 0:
                    left_sample = self._base_snapshot.last_sample
                    left_attitude = self._base_snapshot.last_attitude
                else:
                    left_sample = frames[index - 1].sample
                    left_attitude = frames[index - 1].attitude
                if left_sample is None or left_attitude is None:
                    raise ValueError("interior position update has no retained left IMU sample")
                grouped: dict[int, list[ReplayPositionUpdate]] = {}
                for update in frame.interior_position_updates:
                    grouped.setdefault(update.measurement.timestamp_ns, []).append(update)
                for timestamp_ns in sorted(grouped):
                    sample, attitude = self._interpolated_pair(
                        left_sample, left_attitude, frame, timestamp_ns
                    )
                    candidate.advance_interpolated_epoch(
                        sample,
                        attitude,
                        right_sample=frame.sample,
                        right_attitude=frame.attitude,
                    )
                    for update in sorted(
                        grouped[timestamp_ns],
                        key=lambda item: (
                            item.measurement.source.value,
                            item.measurement.source_sequence_number,
                        ),
                    ):
                        candidate.ingest_position_measurement(
                            update.measurement,
                            max_position_nis=update.max_position_nis,
                        )
            candidate.advance(frame.sample, frame.attitude)
            for update in frame.position_updates:
                candidate.ingest_position_measurement(
                    update.measurement,
                    max_position_nis=update.max_position_nis,
                )
            rebuilt.append(TranslationalReplayFrame(
                sample=frame.sample,
                attitude=frame.attitude,
                position_updates=frame.position_updates,
                interior_position_updates=frame.interior_position_updates,
                snapshot_after_frame=candidate.snapshot(),
            ))
        return candidate, rebuilt

    def ingest_position_measurement(
        self,
        measurement: MetricPositionMeasurement,
        *,
        max_position_nis: float = 16.0,
    ) -> TranslationalErrorState:
        """Insert a current or delayed position update at its acquisition epoch."""
        update = ReplayPositionUpdate(measurement, max_position_nis)
        exact_index = next(
            (
                index for index, frame in enumerate(self._frames)
                if frame.timestamp_ns == measurement.timestamp_ns
            ),
            None,
        )
        interior_index = None
        if exact_index is None:
            base_sample = self._base_snapshot.last_sample
            if (
                self._frames
                and base_sample is not None
                and base_sample.sensor_timestamp_ns < measurement.timestamp_ns
                < self._frames[0].timestamp_ns
            ):
                interior_index = 0
            else:
                for index in range(1, len(self._frames)):
                    left = self._frames[index - 1]
                    right = self._frames[index]
                    if left.timestamp_ns < measurement.timestamp_ns < right.timestamp_ns:
                        interior_index = index
                        break
            if interior_index is None:
                raise ValueError(
                    "position measurement timestamp is outside retained IMU replay intervals"
                )

        self._validate_position_sequence(measurement)
        candidate_frames = list(self._frames)
        if exact_index is not None:
            target = candidate_frames[exact_index]
            updates = tuple(sorted(
                target.position_updates + (update,),
                key=lambda item: (
                    item.measurement.source.value,
                    item.measurement.source_sequence_number,
                ),
            ))
            candidate_frames[exact_index] = TranslationalReplayFrame(
                sample=target.sample,
                attitude=target.attitude,
                position_updates=updates,
                interior_position_updates=target.interior_position_updates,
                snapshot_after_frame=target.snapshot_after_frame,
            )
        else:
            target = candidate_frames[interior_index]
            interior_updates = tuple(sorted(
                target.interior_position_updates + (update,),
                key=lambda item: (
                    item.measurement.timestamp_ns,
                    item.measurement.source.value,
                    item.measurement.source_sequence_number,
                ),
            ))
            candidate_frames[interior_index] = TranslationalReplayFrame(
                sample=target.sample,
                attitude=target.attitude,
                position_updates=target.position_updates,
                interior_position_updates=interior_updates,
                snapshot_after_frame=target.snapshot_after_frame,
            )

        # Transactional: do not mutate live history/filter until all replayed
        # propagation and corrections succeed.
        candidate_filter, rebuilt = self._replay(candidate_frames)
        self._filter = candidate_filter
        self._frames = rebuilt
        return self._filter.state
