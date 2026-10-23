"""Provider-neutral orchestration for a wired, externally-triggered camera group.

A real GigE Vision/GenICam adapter and a real trigger generator must implement
these protocols. This module intentionally does not claim that metadata or PTP
status is physically trustworthy; HW-01 still requires independent timing
evidence.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
import threading
from typing import Callable, Mapping, Protocol, Sequence

from outdoor_sensor_contract import (
    CameraFrameMetadata,
    CameraSyncBatch,
    CameraSyncProfile,
    validate_camera_batch,
)


def _nonnegative_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


@dataclass(frozen=True)
class CameraDeviceInfo:
    device_serial: str
    firmware_version: str
    sdk_version: str
    shutter_type: str
    timestamp_semantics: str
    clock_domain: str
    supports_external_trigger: bool

    def __post_init__(self) -> None:
        for name in (
            "device_serial", "firmware_version", "sdk_version", "shutter_type",
            "timestamp_semantics", "clock_domain",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a nonempty string")
            object.__setattr__(self, name, value.strip())
        if not isinstance(self.supports_external_trigger, bool):
            raise ValueError("supports_external_trigger must be boolean")


@dataclass(frozen=True)
class CameraFrameEnvelope:
    metadata: CameraFrameMetadata
    payload: object


@dataclass(frozen=True)
class SynchronizedCameraBatch:
    summary: CameraSyncBatch
    frames: tuple[CameraFrameEnvelope, ...]


class CameraSyncState(str, Enum):
    STOPPED = "STOPPED"
    STARTING = "STARTING"
    CAPTURING = "CAPTURING"
    FAULT = "FAULT"


class CameraSyncFault(RuntimeError):
    """A camera group lost its synchronization contract and must be re-armed."""


class CameraAdapter(Protocol):
    """Vendor adapter boundary. No vendor SDK object may escape the adapter."""

    def open(self) -> CameraDeviceInfo: ...
    def configure(self, profile: CameraSyncProfile) -> None: ...
    def arm(self) -> None: ...
    def start_acquisition(self) -> None: ...
    def read_frame(self, timeout_s: float) -> CameraFrameEnvelope: ...
    def stop_acquisition(self) -> None: ...
    def disarm(self) -> None: ...
    def close(self) -> None: ...


class TriggerController(Protocol):
    """Physical trigger distributor / timing generator interface."""

    def open(self) -> None: ...
    def arm(self) -> None: ...
    def pulse(self, trigger_id: int) -> None: ...
    def disarm(self) -> None: ...
    def close(self) -> None: ...


class CameraSyncDaemon:
    """Fail-closed lifecycle and batch orchestrator for an external-trigger rig.

    Capture calls are serialized. If a trigger/read/metadata gate fails, the
    session transitions to FAULT, stops acquisition best-effort, and requires
    explicit reset before restart. It never emits a partial camera batch.
    """

    def __init__(
        self,
        profiles: Sequence[CameraSyncProfile],
        adapters: Mapping[int, CameraAdapter],
        trigger_controller: TriggerController,
        *,
        first_trigger_id: int = 0,
        frame_timeout_s: float = 1.0,
    ) -> None:
        if not isinstance(profiles, Sequence) or isinstance(profiles, (str, bytes)) or not profiles:
            raise ValueError("profiles must be a nonempty sequence")
        if any(not isinstance(p, CameraSyncProfile) for p in profiles):
            raise ValueError("profiles must contain CameraSyncProfile records")
        profile_ids = [p.camera_id for p in profiles]
        if len(set(profile_ids)) != len(profile_ids):
            raise ValueError("camera profiles must have unique IDs")
        if set(adapters) != set(profile_ids):
            raise ValueError("adapter IDs must exactly match configured camera IDs")
        if any(p.trigger_mode != "external_line" or not p.require_hardware_trigger for p in profiles):
            raise ValueError("first daemon implementation requires hardware external-line triggers")
        if len({p.clock_domain for p in profiles}) != 1:
            raise ValueError("all camera profiles must declare one common clock domain")
        if len({p.timestamp_semantics for p in profiles}) != 1:
            raise ValueError("all camera profiles must use matching exposure timestamp semantics")
        if isinstance(frame_timeout_s, bool) or not isinstance(frame_timeout_s, (int, float)):
            raise ValueError("frame_timeout_s must be finite and positive")
        timeout = float(frame_timeout_s)
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("frame_timeout_s must be finite and positive")
        self._profiles = tuple(sorted(profiles, key=lambda p: p.camera_id))
        self._adapters = {p.camera_id: adapters[p.camera_id] for p in self._profiles}
        self._trigger = trigger_controller
        self._next_trigger_id = _nonnegative_int(first_trigger_id, "first_trigger_id")
        self._timeout = timeout
        self._state = CameraSyncState.STOPPED
        self._lock = threading.RLock()
        self._opened: set[int] = set()
        self._armed: set[int] = set()
        self._started: set[int] = set()
        self._trigger_open = False
        self._trigger_armed = False
        self._previous_frame_counters: dict[int, int] = {}

    @property
    def state(self) -> CameraSyncState:
        with self._lock:
            return self._state

    @property
    def next_trigger_id(self) -> int:
        with self._lock:
            return self._next_trigger_id

    def _cleanup(self) -> list[str]:
        errors: list[str] = []
        if self._trigger_armed:
            try:
                self._trigger.disarm()
            except Exception as exc:
                errors.append(f"trigger disarm: {exc}")
            self._trigger_armed = False
        for camera_id in sorted(self._started, reverse=True):
            try:
                self._adapters[camera_id].stop_acquisition()
            except Exception as exc:
                errors.append(f"camera {camera_id} stop: {exc}")
        self._started.clear()
        for camera_id in sorted(self._armed, reverse=True):
            try:
                self._adapters[camera_id].disarm()
            except Exception as exc:
                errors.append(f"camera {camera_id} disarm: {exc}")
        self._armed.clear()
        for camera_id in sorted(self._opened, reverse=True):
            try:
                self._adapters[camera_id].close()
            except Exception as exc:
                errors.append(f"camera {camera_id} close: {exc}")
        self._opened.clear()
        if self._trigger_open:
            try:
                self._trigger.close()
            except Exception as exc:
                errors.append(f"trigger close: {exc}")
            self._trigger_open = False
        return errors

    def start(self) -> None:
        with self._lock:
            if self._state is not CameraSyncState.STOPPED:
                raise CameraSyncFault(f"cannot start from {self._state.value}")
            self._state = CameraSyncState.STARTING
            try:
                for profile in self._profiles:
                    adapter = self._adapters[profile.camera_id]
                    # Mark the resource before the call so partial-open failures
                    # still trigger best-effort close during rollback.
                    self._opened.add(profile.camera_id)
                    info = adapter.open()
                    if not isinstance(info, CameraDeviceInfo):
                        raise ValueError("camera adapter returned invalid device info")
                    if info.device_serial != profile.device_serial:
                        raise ValueError(f"camera {profile.camera_id} serial does not match profile")
                    if info.shutter_type != profile.shutter_type:
                        raise ValueError(f"camera {profile.camera_id} shutter type does not match profile")
                    if info.timestamp_semantics != profile.timestamp_semantics:
                        raise ValueError(f"camera {profile.camera_id} timestamp semantics do not match profile")
                    if info.clock_domain != profile.clock_domain:
                        raise ValueError(f"camera {profile.camera_id} clock domain does not match profile")
                    if not info.supports_external_trigger:
                        raise ValueError(f"camera {profile.camera_id} lacks external-trigger support")
                    adapter.configure(profile)
                self._trigger_open = True
                self._trigger.open()
                for profile in self._profiles:
                    adapter = self._adapters[profile.camera_id]
                    self._armed.add(profile.camera_id)
                    adapter.arm()
                for profile in self._profiles:
                    adapter = self._adapters[profile.camera_id]
                    self._started.add(profile.camera_id)
                    adapter.start_acquisition()
                self._trigger_armed = True
                self._trigger.arm()
                self._state = CameraSyncState.CAPTURING
            except Exception as exc:
                cleanup_errors = self._cleanup()
                self._state = CameraSyncState.FAULT
                suffix = f"; cleanup errors: {'; '.join(cleanup_errors)}" if cleanup_errors else ""
                raise CameraSyncFault(f"camera group start failed: {exc}{suffix}") from exc

    def capture_batch(self) -> SynchronizedCameraBatch:
        with self._lock:
            if self._state is not CameraSyncState.CAPTURING:
                raise CameraSyncFault(f"cannot capture from {self._state.value}")
            trigger_id = self._next_trigger_id
            # Reserve the ID before firing: even an ambiguous trigger-controller
            # exception must never cause an ID to be reused after recovery.
            self._next_trigger_id += 1
            try:
                self._trigger.pulse(trigger_id)
                envelopes = tuple(
                    self._adapters[profile.camera_id].read_frame(self._timeout)
                    for profile in self._profiles
                )
                if any(not isinstance(frame, CameraFrameEnvelope) for frame in envelopes):
                    raise ValueError("camera adapter returned an invalid frame envelope")
                summary = validate_camera_batch(
                    tuple(frame.metadata for frame in envelopes),
                    self._profiles,
                    expected_trigger_id=trigger_id,
                    previous_frame_counters=self._previous_frame_counters,
                )
                self._previous_frame_counters = {
                    frame.metadata.camera_id: frame.metadata.frame_counter
                    for frame in envelopes
                }
                return SynchronizedCameraBatch(summary, envelopes)
            except Exception as exc:
                cleanup_errors = self._cleanup()
                self._state = CameraSyncState.FAULT
                suffix = f"; cleanup errors: {'; '.join(cleanup_errors)}" if cleanup_errors else ""
                raise CameraSyncFault(
                    f"trigger {trigger_id} rejected; no partial batch emitted: {exc}{suffix}"
                ) from exc

    def stop(self) -> None:
        with self._lock:
            if self._state is CameraSyncState.STOPPED:
                return
            was_faulted = self._state is CameraSyncState.FAULT
            errors = self._cleanup()
            if was_faulted or errors:
                self._state = CameraSyncState.FAULT
            else:
                self._state = CameraSyncState.STOPPED
            if errors:
                raise CameraSyncFault("camera group stop had errors: " + "; ".join(errors))

    def reset_fault(self) -> None:
        with self._lock:
            if self._state is not CameraSyncState.FAULT:
                raise CameraSyncFault("reset_fault is only valid from FAULT")
            errors = self._cleanup()
            if errors:
                raise CameraSyncFault("cannot reset camera group: " + "; ".join(errors))
            self._state = CameraSyncState.STOPPED

    def run_forever(
        self,
        on_batch: Callable[[SynchronizedCameraBatch], None],
        stop_event: threading.Event,
    ) -> None:
        """Run capture synchronously in the caller's service thread until stopped."""
        if not callable(on_batch) or not isinstance(stop_event, threading.Event):
            raise ValueError("on_batch and stop_event are required")
        self.start()
        try:
            while not stop_event.is_set():
                on_batch(self.capture_batch())
        finally:
            if self.state is CameraSyncState.CAPTURING:
                self.stop()
