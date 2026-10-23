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

from dataclasses import dataclass
from typing import Sequence

from imu_calibration import CorrectedIMUVectors, IMUCalibration
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
        target_index: int,
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

        for frame_index, frame in enumerate(self._frames):
            for update in frame.position_updates:
                existing = update.measurement
                if existing.source is not source:
                    continue
                if existing.source_sequence_number == measurement.source_sequence_number:
                    raise ValueError("delayed position measurement sequence is duplicate")
                if frame.timestamp_ns == measurement.timestamp_ns:
                    raise ValueError(
                        "one replay timestamp may contain only one update per source"
                    )
                if (
                    frame_index < target_index
                    and existing.source_sequence_number
                    >= measurement.source_sequence_number
                ):
                    raise ValueError(
                        "delayed position source sequence disagrees with timestamp order"
                    )
                if (
                    frame_index > target_index
                    and existing.source_sequence_number
                    <= measurement.source_sequence_number
                ):
                    raise ValueError(
                        "delayed position source sequence disagrees with timestamp order"
                    )

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
        for frame in frames:
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
                snapshot_after_frame=candidate.snapshot(),
            ))
        return candidate, rebuilt

    def ingest_position_measurement(
        self,
        measurement: MetricPositionMeasurement,
        *,
        max_position_nis: float = 16.0,
    ) -> TranslationalErrorState:
        """Insert a current or delayed exact-timestamp position update and replay."""
        update = ReplayPositionUpdate(measurement, max_position_nis)
        target_index = next(
            (
                index
                for index, frame in enumerate(self._frames)
                if frame.timestamp_ns == measurement.timestamp_ns
            ),
            None,
        )
        if target_index is None:
            raise ValueError(
                "position measurement timestamp is outside retained exact IMU replay frames"
            )
        self._validate_position_sequence(measurement, target_index)

        candidate_frames = list(self._frames)
        target = candidate_frames[target_index]
        updates = tuple(sorted(
            target.position_updates + (update,),
            key=lambda item: (
                item.measurement.source.value,
                item.measurement.source_sequence_number,
            ),
        ))
        candidate_frames[target_index] = TranslationalReplayFrame(
            sample=target.sample,
            attitude=target.attitude,
            position_updates=updates,
            snapshot_after_frame=target.snapshot_after_frame,
        )

        # Transactional: replay into a fresh estimator and only replace the
        # live owner after every historical and subsequent event passes again.
        candidate_filter, rebuilt = self._replay(candidate_frames)
        self._filter = candidate_filter
        self._frames = rebuilt
        return self._filter.state
