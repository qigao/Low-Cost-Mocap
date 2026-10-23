"""Provider-neutral outdoor camera-sync and RTK adapter contracts."""
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
import math
from typing import Mapping, Sequence


def _integer(value: object, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _nonnegative(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite nonnegative number")
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a finite nonnegative number")
    return value


def _text(value: object, name: str, limit: int = 256) -> str:
    if not isinstance(value, str) or not value.strip() or len(value.strip()) > limit:
        raise ValueError(f"{name} must be a nonempty string of at most {limit} characters")
    return value.strip()


class RTKSolutionStatus(str, Enum):
    FIXED = "FIXED"
    FLOAT = "FLOAT"
    DGNSS = "DGNSS"
    SINGLE = "SINGLE"
    INVALID = "INVALID"


@dataclass(frozen=True)
class CameraSyncProfile:
    camera_id: int
    device_serial: str
    transport: str
    shutter_type: str
    trigger_mode: str
    timestamp_semantics: str
    clock_domain: str
    max_exposure_skew_ns: int
    require_hardware_trigger: bool = True
    require_ptp_lock: bool = False

    def __post_init__(self) -> None:
        _integer(self.camera_id, "camera_id")
        object.__setattr__(self, "device_serial", _text(self.device_serial, "device_serial"))
        if not isinstance(self.transport, str) or self.transport not in {"gige_vision", "usb3_vision", "other"}:
            raise ValueError("unsupported camera transport")
        if not isinstance(self.shutter_type, str) or self.shutter_type not in {"global", "rolling", "unknown"}:
            raise ValueError("unsupported shutter_type")
        if not isinstance(self.trigger_mode, str) or self.trigger_mode not in {"external_line", "ptp_scheduled", "free_run"}:
            raise ValueError("unsupported trigger_mode")
        if not isinstance(self.timestamp_semantics, str) or self.timestamp_semantics not in {"exposure_start", "exposure_midpoint", "exposure_end"}:
            raise ValueError("timestamp semantics must describe sensor exposure")
        object.__setattr__(self, "clock_domain", _text(self.clock_domain, "clock_domain"))
        _integer(self.max_exposure_skew_ns, "max_exposure_skew_ns")
        if not isinstance(self.require_hardware_trigger, bool) or not isinstance(self.require_ptp_lock, bool):
            raise ValueError("trigger/PTP requirements must be booleans")
        if self.require_hardware_trigger and self.trigger_mode == "free_run":
            raise ValueError("free_run cannot satisfy required hardware trigger association")
        if self.require_ptp_lock and self.trigger_mode != "ptp_scheduled":
            raise ValueError("require_ptp_lock requires ptp_scheduled mode")


@dataclass(frozen=True)
class CameraFrameMetadata:
    camera_id: int
    device_serial: str
    frame_counter: int
    trigger_id: int | None
    acquisition_timestamp_ns: int | None
    clock_domain: str | None
    timestamp_semantics: str
    hardware_triggered: bool
    ptp_locked: bool
    packet_loss_count: int = 0
    host_receive_timestamp_ns: int | None = None

    def __post_init__(self) -> None:
        _integer(self.camera_id, "camera_id")
        object.__setattr__(self, "device_serial", _text(self.device_serial, "device_serial"))
        _integer(self.frame_counter, "frame_counter")
        if self.trigger_id is not None:
            _integer(self.trigger_id, "trigger_id")
        if self.acquisition_timestamp_ns is not None:
            _integer(self.acquisition_timestamp_ns, "acquisition_timestamp_ns")
        if self.clock_domain is not None:
            object.__setattr__(self, "clock_domain", _text(self.clock_domain, "clock_domain"))
        if not isinstance(self.timestamp_semantics, str) or not self.timestamp_semantics:
            raise ValueError("timestamp_semantics must be nonempty")
        if not isinstance(self.hardware_triggered, bool) or not isinstance(self.ptp_locked, bool):
            raise ValueError("hardware_triggered and ptp_locked must be booleans")
        _integer(self.packet_loss_count, "packet_loss_count")
        if self.host_receive_timestamp_ns is not None:
            _integer(self.host_receive_timestamp_ns, "host_receive_timestamp_ns")


def validate_camera_frame(frame: CameraFrameMetadata, profile: CameraSyncProfile, *,
                          expected_trigger_id: int,
                          previous_frame_counter: int | None = None) -> CameraFrameMetadata:
    if not isinstance(frame, CameraFrameMetadata) or not isinstance(profile, CameraSyncProfile):
        raise ValueError("frame/profile has invalid type")
    expected = _integer(expected_trigger_id, "expected_trigger_id")
    if frame.camera_id != profile.camera_id or frame.device_serial != profile.device_serial:
        raise ValueError("camera identity does not match configured profile")
    if frame.trigger_id is None or frame.trigger_id != expected:
        raise ValueError("missing or mismatched hardware-associated trigger_id")
    if frame.acquisition_timestamp_ns is None:
        raise ValueError("sensor exposure timestamp is missing")
    if frame.timestamp_semantics != profile.timestamp_semantics:
        raise ValueError("frame timestamp semantics do not match profile")
    if frame.timestamp_semantics not in {"exposure_start", "exposure_midpoint", "exposure_end"}:
        raise ValueError("host receive/dequeue timestamps are not exposure timestamps")
    if frame.clock_domain != profile.clock_domain:
        raise ValueError("camera clock domain does not match common clock")
    if profile.require_hardware_trigger and not frame.hardware_triggered:
        raise ValueError("frame is not associated with a hardware trigger")
    if profile.require_ptp_lock and not frame.ptp_locked:
        raise ValueError("PTP clock is not locked")
    if previous_frame_counter is not None and frame.frame_counter <= _integer(previous_frame_counter, "previous_frame_counter"):
        raise ValueError("camera frame counter is stale or repeated")
    if frame.packet_loss_count:
        raise ValueError("camera reports packet loss for this frame")
    return frame


@dataclass(frozen=True)
class CameraSyncBatch:
    trigger_id: int
    camera_ids: tuple[int, ...]
    clock_domain: str
    exposure_skew_ns: int


def validate_camera_batch(frames: Sequence[CameraFrameMetadata],
                          profiles: Sequence[CameraSyncProfile], *,
                          expected_trigger_id: int,
                          previous_frame_counters: Mapping[int, int] | None = None) -> CameraSyncBatch:
    if not isinstance(frames, Sequence) or isinstance(frames, (str, bytes)):
        raise ValueError("frames must be a sequence")
    if not isinstance(profiles, Sequence) or isinstance(profiles, (str, bytes)) or not profiles:
        raise ValueError("profiles must be a nonempty sequence")
    if any(not isinstance(profile, CameraSyncProfile) for profile in profiles):
        raise ValueError("profiles must contain CameraSyncProfile records")
    if any(not isinstance(frame, CameraFrameMetadata) for frame in frames):
        raise ValueError("frames must contain CameraFrameMetadata records")
    by_id = {p.camera_id: p for p in profiles}
    if len(by_id) != len(profiles):
        raise ValueError("camera profiles must have unique IDs")
    if len(frames) != len(profiles):
        raise ValueError("synchronized batch has missing or extra frames")
    frame_ids = [f.camera_id for f in frames]
    if len(set(frame_ids)) != len(frame_ids) or set(frame_ids) != set(by_id):
        raise ValueError("batch has missing, duplicate or unexpected camera IDs")
    previous = previous_frame_counters or {}
    for frame in frames:
        validate_camera_frame(frame, by_id[frame.camera_id],
                              expected_trigger_id=expected_trigger_id,
                              previous_frame_counter=previous.get(frame.camera_id))
    domains = {f.clock_domain for f in frames}
    semantics = {f.timestamp_semantics for f in frames}
    if len(domains) != 1 or len(semantics) != 1:
        raise ValueError("batch mixes clock domains or timestamp semantics")
    timestamps = [f.acquisition_timestamp_ns for f in frames]
    if any(t is None for t in timestamps):
        raise ValueError("batch contains a missing exposure timestamp")
    skew = max(timestamps) - min(timestamps)
    if skew > min(p.max_exposure_skew_ns for p in profiles):
        raise ValueError("camera exposure timestamp skew exceeds configured limit")
    return CameraSyncBatch(_integer(expected_trigger_id, "expected_trigger_id"),
                           tuple(sorted(frame_ids)), next(iter(domains)), skew)


@dataclass(frozen=True)
class RTKSample:
    receiver_id: str
    sequence_number: int
    solution_timestamp_ns: int
    clock_domain: str
    status: RTKSolutionStatus
    position_ecef_m: tuple[float, float, float]
    horizontal_accuracy_m: float
    vertical_accuracy_m: float
    correction_age_ms: float
    satellites_used: int
    datum: str = "WGS84"
    antenna_reference_point: str = "antenna_phase_center"
    host_receive_timestamp_ns: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "receiver_id", _text(self.receiver_id, "receiver_id"))
        _integer(self.sequence_number, "sequence_number")
        _integer(self.solution_timestamp_ns, "solution_timestamp_ns")
        object.__setattr__(self, "clock_domain", _text(self.clock_domain, "clock_domain"))
        try:
            object.__setattr__(self, "status", RTKSolutionStatus(self.status))
        except (TypeError, ValueError) as exc:
            raise ValueError("unsupported RTK solution status") from exc
        if not isinstance(self.position_ecef_m, (tuple, list)) or len(self.position_ecef_m) != 3:
            raise ValueError("position_ecef_m must contain exactly three coordinates")
        coordinates = []
        for index, value in enumerate(self.position_ecef_m):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise ValueError(f"position_ecef_m[{index}] must be finite")
            coordinates.append(float(value))
        object.__setattr__(self, "position_ecef_m", tuple(coordinates))
        for name in ("horizontal_accuracy_m", "vertical_accuracy_m", "correction_age_ms"):
            object.__setattr__(self, name, _nonnegative(getattr(self, name), name))
        _integer(self.satellites_used, "satellites_used")
        object.__setattr__(self, "datum", _text(self.datum, "datum"))
        object.__setattr__(self, "antenna_reference_point", _text(self.antenna_reference_point, "antenna_reference_point"))
        if self.host_receive_timestamp_ns is not None:
            _integer(self.host_receive_timestamp_ns, "host_receive_timestamp_ns")

    def usable_as_global_position(self, *, max_correction_age_ms: float,
                                  max_horizontal_accuracy_m: float,
                                  max_vertical_accuracy_m: float,
                                  min_satellites: int = 6) -> bool:
        return (
            self.status is RTKSolutionStatus.FIXED
            and self.correction_age_ms <= _nonnegative(max_correction_age_ms, "max_correction_age_ms")
            and self.horizontal_accuracy_m <= _nonnegative(max_horizontal_accuracy_m, "max_horizontal_accuracy_m")
            and self.vertical_accuracy_m <= _nonnegative(max_vertical_accuracy_m, "max_vertical_accuracy_m")
            and self.satellites_used >= _integer(min_satellites, "min_satellites", minimum=1)
        )
