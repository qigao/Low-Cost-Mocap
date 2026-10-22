"""Bounded provider-neutral PoseStream publisher and independent consumer.

No camera, network server, flight-control, or asynchronous worker is started
here. One serialized owner calls subscribe/publish/flush. An optional caller
can inject a Socket.IO-style emit callback into flush_with_callback; only a
successful callback advances the subscriber queue. Schema/identity and timing
gates are never inferred from host receive/dequeue timestamps.
"""
from __future__ import annotations

from collections import deque
from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass, field
import json
import re
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from board_patterns import BoardPatternRegistry
from identity_locked_board_tracker import LockedBoardFrame
from pose_stream import describe_board_assets, pose_frame_to_dict
from fused_attitude_stream import validate_fused_attitude_payload
from temporal_pattern import _nonnegative_int


_STREAM_PROTOCOL = "low_cost_mocap_stream"
_PAYLOAD_PROTOCOL = "low_cost_mocap_pose"
_SESSION_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}\Z")
_EVENT_NAME = re.compile(r"[A-Za-z][A-Za-z0-9_.:-]{0,63}\Z")
_HEX_64 = re.compile(r"[0-9a-f]{64}\Z")


def _json_size(value: object) -> int:
    return len(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8"))


def _bounded_positive(value: object, name: str, *, maximum: int) -> int:
    result = _nonnegative_int(value, name)
    if not 1 <= result <= maximum:
        raise ValueError(f"{name} must be within 1..{maximum}")
    return result


def _session_name(value: object, name: str) -> str:
    if not isinstance(value, str) or _SESSION_NAME.fullmatch(value) is None:
        raise ValueError(f"{name} must be a safe nonempty identifier")
    return value


def _sha(value: object, label: str) -> str:
    if not isinstance(value, str) or _HEX_64.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase hex SHA-256 digest")
    return value


def _finite_list(data: object, expected: int, name: str) -> tuple[float, ...]:
    if not isinstance(data, (tuple, list)) or len(data) != expected:
        raise ValueError(f"{name} must have {expected} numeric components")
    if any(
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, float, np.integer, np.floating))
        for value in data
    ):
        raise ValueError(f"{name} must contain real numeric components")
    try:
        values = np.asarray(data, dtype=np.float64)
    except (ValueError, OverflowError, TypeError) as exc:
        raise ValueError(f"{name} has an unrepresentable numeric component") from exc
    if not np.isfinite(values).all():
        raise ValueError(f"{name} must be finite")
    return tuple(float(v) for v in values)


def _envelope(
    payload: dict[str, Any], *, session_id: int, sequence: int,
) -> dict[str, Any]:
    return {
        "protocol": _STREAM_PROTOCOL,
        "schema_version": 1,
        "session_id": session_id,
        "sequence": sequence,
        "payload": deepcopy(payload),
    }


@dataclass
class _Subscriber:
    next_sequence: int = 0
    pending_asset: dict[str, Any] | None = None
    pending_frames: deque[dict[str, Any]] = field(default_factory=deque)
    dropped_frames: int = 0


@dataclass(frozen=True)
class SubscriberStatistics:
    subscriber_id: str
    session_id: int
    pending_frames: int
    pending_asset: bool
    dropped_frames: int


class BoundedPoseStreamPublisher:
    """In-memory queue with explicit bounded per-subscriber ownership.

    The publisher rejects stale source triggers, source-clock regressions and
    camera calibration revisions changed mid-session. Bounded queues drop
    only the oldest *frame*, never the pending asset description. A disconnected
    subscriber cannot read a last-good pose from an old session.
    """

    def __init__(
        self,
        registry: BoardPatternRegistry,
        *,
        max_subscribers: int = 16,
        max_pending_frames: int = 8,
        max_message_bytes: int = 1_000_000,
    ) -> None:
        if not isinstance(registry, BoardPatternRegistry):
            raise ValueError("registry must be BoardPatternRegistry")
        self.registry = registry
        self.max_subscribers = _bounded_positive(
            max_subscribers, "max_subscribers", maximum=1000
        )
        self.max_pending_frames = _bounded_positive(
            max_pending_frames, "max_pending_frames", maximum=10000
        )
        self.max_message_bytes = _bounded_positive(
            max_message_bytes, "max_message_bytes", maximum=100_000_000
        )
        self._asset = describe_board_assets(registry)
        if _json_size(_envelope(self._asset, session_id=1, sequence=0)) > self.max_message_bytes:
            raise ValueError("asset description exceeds transport message limit")
        self.session_id = 1
        self._subscribers: dict[str, _Subscriber] = {}
        self._last_frame_id: int | None = None
        self._last_timestamp_ns: int | None = None
        self._clock_domain: str | None = None
        self._calibration_sha: str | None = None

    @property
    def subscriber_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._subscribers))

    def _new_asset(self, sub: _Subscriber) -> dict[str, Any]:
        envelope = _envelope(
            self._asset, session_id=self.session_id,
            sequence=sub.next_sequence,
        )
        if _json_size(envelope) > self.max_message_bytes:
            raise ValueError("asset envelope exceeds transport message limit")
        sub.next_sequence += 1
        return envelope

    def subscribe(self, subscriber_id: str) -> None:
        ident = _session_name(subscriber_id, "subscriber_id")
        if ident in self._subscribers:
            raise ValueError("subscriber already connected; disconnect before resubscribe")
        if len(self._subscribers) >= self.max_subscribers:
            raise ValueError("subscriber capacity exceeded")
        sub = _Subscriber()
        sub.pending_asset = self._new_asset(sub)
        self._subscribers[ident] = sub

    def unsubscribe(self, subscriber_id: str) -> None:
        ident = _session_name(subscriber_id, "subscriber_id")
        if ident not in self._subscribers:
            raise ValueError("subscriber is not connected")
        del self._subscribers[ident]

    def subscriber_statistics(self, subscriber_id: str) -> SubscriberStatistics:
        ident = _session_name(subscriber_id, "subscriber_id")
        try:
            sub = self._subscribers[ident]
        except KeyError as exc:
            raise ValueError("subscriber is not connected") from exc
        return SubscriberStatistics(
            ident, self.session_id,
            len(sub.pending_frames), sub.pending_asset is not None,
            sub.dropped_frames,
        )

    def _new_session(self) -> None:
        if self.session_id >= (1 << 63) - 1:
            raise ValueError("publisher session ID exhausted")
        self.session_id += 1
        self._last_frame_id = None
        self._last_timestamp_ns = None
        self._clock_domain = None
        self._calibration_sha = None
        for sub in self._subscribers.values():
            sub.next_sequence = 0
            sub.pending_frames.clear()
            sub.dropped_frames = 0
            sub.pending_asset = self._new_asset(sub)

    def restart_session(self) -> None:
        """Explicitly reset source trigger/timebase after a capture restart."""
        self._new_session()

    def replace_registry(self, registry: BoardPatternRegistry) -> None:
        """Pattern change forces a new session and asset-before-frame barrier."""
        if not isinstance(registry, BoardPatternRegistry):
            raise ValueError("registry must be BoardPatternRegistry")
        asset = describe_board_assets(registry)
        if _json_size(_envelope(
            asset, session_id=self.session_id + 1, sequence=0
        )) > self.max_message_bytes:
            raise ValueError("asset description exceeds transport message limit")
        if registry.config_sha256 == self.registry.config_sha256:
            raise ValueError("new registry has unchanged pattern revision")
        self.registry = registry
        self._asset = asset
        self._new_session()

    def publish(self, frame: LockedBoardFrame) -> int:
        """Queue an immutable JSON snapshot of a *validated* source frame."""
        payload = pose_frame_to_dict(frame, self.registry)
        next_seq = max(
            (sub.next_sequence for sub in self._subscribers.values()),
            default=0,
        )
        if _json_size(_envelope(
            payload, session_id=self.session_id, sequence=next_seq
        )) > self.max_message_bytes:
            raise ValueError("pose frame exceeds transport message limit")
        frame_id = _nonnegative_int(payload["frame_id"], "frame_id")
        timestamp = _nonnegative_int(
            payload["latest_exposure_timestamp_ns"],
            "latest_exposure_timestamp_ns",
        )
        domain = payload["clock_domain"]
        calibration = _sha(payload["calibration_sha256"], "calibration_sha256")
        if self._last_frame_id is not None and frame_id <= self._last_frame_id:
            raise ValueError("pose stream received stale/repeated source trigger")
        if self._last_timestamp_ns is not None and timestamp <= self._last_timestamp_ns:
            raise ValueError("pose stream exposure source time is not monotonic")
        if self._clock_domain is not None and domain != self._clock_domain:
            raise ValueError("pose stream clock domain changed mid-session")
        if self._calibration_sha is not None and calibration != self._calibration_sha:
            raise ValueError("camera calibration revision changed mid-session")

        # Mutate state only after the entire source contract has been checked.
        for sub in self._subscribers.values():
            packet = _envelope(
                payload, session_id=self.session_id,
                sequence=sub.next_sequence,
            )
            if len(sub.pending_frames) == self.max_pending_frames:
                sub.pending_frames.popleft()
                sub.dropped_frames += 1
            sub.pending_frames.append(packet)
            sub.next_sequence += 1
        self._last_frame_id = frame_id
        self._last_timestamp_ns = timestamp
        self._clock_domain = domain
        self._calibration_sha = calibration
        return frame_id

    def peek_next(self, subscriber_id: str) -> dict[str, Any] | None:
        """Peek without dequeue: callbacks must acknowledge successful send."""
        ident = _session_name(subscriber_id, "subscriber_id")
        try:
            sub = self._subscribers[ident]
        except KeyError as exc:
            raise ValueError("subscriber is not connected") from exc
        if sub.pending_asset is not None:
            return deepcopy(sub.pending_asset)
        if sub.pending_frames:
            return deepcopy(sub.pending_frames[0])
        return None

    def acknowledge(self, subscriber_id: str, sequence: int) -> None:
        ident = _session_name(subscriber_id, "subscriber_id")
        seq = _nonnegative_int(sequence, "sequence")
        try:
            sub = self._subscribers[ident]
        except KeyError as exc:
            raise ValueError("subscriber is not connected") from exc
        packet = sub.pending_asset if sub.pending_asset is not None else (
            sub.pending_frames[0] if sub.pending_frames else None
        )
        if packet is None or packet["sequence"] != seq:
            raise ValueError("acknowledged sequence is not the current queued message")
        if sub.pending_asset is not None:
            sub.pending_asset = None
        else:
            sub.pending_frames.popleft()

    def flush_with_callback(
        self,
        subscriber_id: str,
        emit: Callable[[str, dict[str, Any], str], object],
        *,
        max_messages: int = 20,
        event_name: str = "mocap:pose-stream",
    ) -> int:
        """Send via injected callback(event, envelope, subscriber_id).

        The adapter can be wired to Socket.IO by the application owner.
        Callback errors or a False return keep the message pending for
        explicit retry. We do NOT silently claim transport delivery.
        """
        if not callable(emit):
            raise ValueError("emit callback must be callable")
        if not isinstance(event_name, str) or _EVENT_NAME.fullmatch(event_name) is None:
            raise ValueError("event_name must be a safe nonempty event identifier")
        name = event_name
        count = _bounded_positive(max_messages, "max_messages", maximum=10000)
        sent = 0
        for _ in range(count):
            packet = self.peek_next(subscriber_id)
            if packet is None:
                break
            result = emit(name, packet, subscriber_id)
            if result is False:
                raise ValueError("transport callback did not acknowledge send")
            self.acknowledge(subscriber_id, packet["sequence"])
            sent += 1
        return sent


def _asset_identity(
    payload: object,
) -> tuple[
    str,
    dict[str, set[int]],
    dict[str, int],
    dict[str, tuple[np.ndarray, np.ndarray]],
]:
    if not isinstance(payload, dict):
        raise ValueError("asset description must be a JSON object")
    if (
        payload.get("protocol") != _PAYLOAD_PROTOCOL
        or type(payload.get("schema_version")) is not int
        or payload["schema_version"] != 1
        or payload.get("kind") != "asset_description"
        or payload.get("hardware_provenance_verified") is not False
    ):
        raise ValueError("invalid or hardware-asserting asset description")
    if set(payload) != {
        "protocol", "schema_version", "kind", "pattern_sha256",
        "hardware_provenance_verified",
        "model_coordinates_are_declared_not_metrologically_verified",
        "period_triggers", "max_bit_errors", "pose_phase", "boards",
    } or payload["model_coordinates_are_declared_not_metrologically_verified"] is not True:
        raise ValueError("asset description fields do not match schema version 1")
    revision = _sha(payload.get("pattern_sha256"), "pattern_sha256")
    period = _bounded_positive(payload.get("period_triggers"), "period_triggers", maximum=256)
    errors = _nonnegative_int(payload.get("max_bit_errors"), "max_bit_errors")
    if errors >= period:
        raise ValueError("asset code correction budget is invalid")
    boards = payload.get("boards")
    if not isinstance(boards, list) or not 1 <= len(boards) <= 32:
        raise ValueError("asset description needs a nonempty board list")
    expected = {}
    requirements = {}
    transforms = {}
    global_markers = set()
    for board in boards:
        if not isinstance(board, dict):
            raise ValueError("board description must be an object")
        name = _session_name(board.get("board_id"), "board_id")
        if name in expected:
            raise ValueError("duplicate asset board_id")
        markers = board.get("markers")
        if not isinstance(markers, list) or not markers:
            raise ValueError("board must have marker descriptions")
        local = set()
        for entry in markers:
            if not isinstance(entry, dict):
                raise ValueError("marker description must be an object")
            marker = _nonnegative_int(entry.get("marker_id"), "marker_id")
            code = entry.get("code")
            if (
                marker in global_markers
                or not isinstance(code, str)
                or len(code) != period
                or any(bit not in "01" for bit in code)
            ):
                raise ValueError("duplicate marker ID or invalid temporal code")
            _finite_list(entry.get("xyz_m"), 3, "model marker xyz_m")
            global_markers.add(marker)
            local.add(marker)
        minimum = _bounded_positive(
            board.get("min_visible_markers"),
            "min_visible_markers", maximum=128,
        )
        if minimum < 3 or minimum > len(local):
            raise ValueError("asset minimum visible marker count is inconsistent")
        transform = board.get("body_from_board")
        if not isinstance(transform, dict):
            raise ValueError("board body mounting transform is missing")
        raw_rotation = transform.get("rotation")
        if not isinstance(raw_rotation, list) or len(raw_rotation) != 3:
            raise ValueError("board body mounting rotation must be 3x3")
        R = np.asarray([
            _finite_list(row, 3, "body mounting rotation")
            for row in raw_rotation
        ], dtype=np.float64)
        t = np.asarray(
            _finite_list(transform.get("translation_m"), 3, "body mounting translation"),
            dtype=np.float64,
        )
        if (
            not np.allclose(R.T @ R, np.eye(3), atol=1e-7, rtol=0)
            or abs(float(np.linalg.det(R)) - 1.0) > 1e-7
        ):
            raise ValueError("asset board mounting rotation must be proper")
        expected[name] = local
        requirements[name] = minimum
        transforms[name] = (R, t)

    # The SHA is a fingerprint of *all* semantic board definitions, not
    # merely a caller-supplied label. Reconstruct the active registry from
    # the wire asset and verify its exact canonical configuration revision.
    manifest_boards = {}
    for board in boards:
        name = board["board_id"]
        if set(board) != {
            "board_id", "type", "min_visible_markers",
            "body_from_board", "markers",
        }:
            raise ValueError("unexpected board asset fields")
        if set(board["body_from_board"]) != {"rotation", "translation_m"}:
            raise ValueError("unexpected board mounting asset fields")
        serialized_markers = {}
        for entry in board["markers"]:
            if set(entry) != {"marker_id", "xyz_m", "code"}:
                raise ValueError("unexpected marker asset fields")
            serialized_markers[str(entry["marker_id"])] = {
                "xyz_m": entry["xyz_m"],
                "code": entry["code"],
            }
        manifest_boards[name] = {
            "type": board["type"],
            "min_visible_markers": board["min_visible_markers"],
            "body_from_board": board["body_from_board"],
            "markers": serialized_markers,
        }
    actual = BoardPatternRegistry.from_mapping({
        "version": 1,
        "period_triggers": period,
        "max_bit_errors": errors,
        "pose_phase": payload["pose_phase"],
        "boards": manifest_boards,
    })
    if actual.config_sha256 != revision:
        raise ValueError("asset semantic pattern SHA-256 mismatch")
    return revision, expected, requirements, transforms


class PoseStreamConsumer:
    """Independent deserializer with strict reconnect and freshness gates."""

    def __init__(self, *, max_message_bytes: int = 1_000_000) -> None:
        self.max_message_bytes = _bounded_positive(
            max_message_bytes, "max_message_bytes", maximum=100_000_000
        )
        self.reset_for_reconnect()

    def reset_for_reconnect(self) -> None:
        """Require a fresh asset before consuming a resumed subscriber."""
        self.session_id: int | None = None
        self.last_sequence: int | None = None
        self.pattern_sha256: str | None = None
        self.calibration_sha256: str | None = None
        self.last_frame_id: int | None = None
        self.last_exposure_ns: int | None = None
        self.clock_domain: str | None = None
        self.boards: dict[str, set[int]] = {}
        self.min_visible: dict[str, int] = {}
        self.body_mounts: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self.last_fused_timestamp_ns: int | None = None
        self.fused_imu_id: str | None = None
        self.fused_clock_domain: str | None = None
        self.fused_mapping_sha256: str | None = None
        self.fused_imu_calibration_sha256: str | None = None
        self.fused_optical_calibration_sha256: str | None = None

    def _check_frame(self, payload: dict[str, Any]) -> None:
        if (
            payload.get("protocol") != _PAYLOAD_PROTOCOL
            or type(payload.get("schema_version")) is not int
            or payload["schema_version"] != 1
            or payload.get("kind") != "pose_frame"
            or payload.get("hardware_provenance_verified") is not False
            or payload.get("position_units") != "uncalibrated_world"
        ):
            raise ValueError("invalid or hardware-asserting PoseFrame")
        if _sha(payload.get("pattern_sha256"), "pattern_sha256") != self.pattern_sha256:
            raise ValueError("PoseFrame pattern revision disagrees with assets")
        calibration = _sha(payload.get("calibration_sha256"), "calibration_sha256")
        if self.calibration_sha256 is not None and calibration != self.calibration_sha256:
            raise ValueError("PoseFrame calibration revision changed in same session")
        frame_id = _nonnegative_int(payload.get("frame_id"), "frame_id")
        timestamp = _nonnegative_int(
            payload.get("latest_exposure_timestamp_ns"), "exposure_timestamp_ns"
        )
        if self.last_frame_id is not None and frame_id <= self.last_frame_id:
            raise ValueError("duplicate/stale source PoseFrame")
        if self.last_exposure_ns is not None and timestamp <= self.last_exposure_ns:
            raise ValueError("PoseFrame source exposure time is nonmonotonic")
        domain = payload.get("clock_domain")
        if not isinstance(domain, str) or not domain:
            raise ValueError("PoseFrame clock domain must be declared")
        if self.clock_domain is not None and domain != self.clock_domain:
            raise ValueError("PoseFrame mixed source clock domains")
        camera_ids = payload.get("camera_ids")
        if not isinstance(camera_ids, list) or not camera_ids:
            raise ValueError("PoseFrame must contain physical camera IDs")
        cameras = [_nonnegative_int(x, "camera_id") for x in camera_ids]
        if len(set(cameras)) != len(cameras):
            raise ValueError("duplicate physical camera ID")
        quarantined = payload.get("quarantined_camera_ids")
        if not isinstance(quarantined, list):
            raise ValueError("quarantined_camera_ids must be a list")
        q = [_nonnegative_int(x, "quarantined_camera_id") for x in quarantined]
        if len(set(q)) != len(q) or not set(q).issubset(set(cameras)):
            raise ValueError("unexpected quarantine camera ID")
        _nonnegative_int(payload.get("exposure_skew_ns"), "exposure_skew_ns")

        bodies = payload.get("rigid_bodies")
        if not isinstance(bodies, list) or len(bodies) != len(self.boards):
            raise ValueError("PoseFrame must contain every registered body")
        seen_boards = set()
        for body in bodies:
            if not isinstance(body, dict):
                raise ValueError("rigid body must be a JSON object")
            name = _session_name(body.get("board_id"), "board_id")
            if name not in self.boards or name in seen_boards:
                raise ValueError("unknown or duplicated rigid body")
            seen_boards.add(name)
            marker_ids = body.get("visible_marker_ids")
            excluded_ids = body.get("excluded_marker_ids")
            if not isinstance(marker_ids, list) or not isinstance(excluded_ids, list):
                raise ValueError("marker identity lists missing")
            observed = [_nonnegative_int(v, "visible marker_id") for v in marker_ids]
            excluded = [_nonnegative_int(v, "excluded marker_id") for v in excluded_ids]
            if (
                len(set(observed)) != len(observed)
                or len(set(excluded)) != len(excluded)
                or not (set(observed) | set(excluded)).issubset(self.boards[name])
                or set(observed) & set(excluded)
            ):
                raise ValueError("marker identity is duplicated or owned by a different board")
            source_trigger = body.get("measurement_trigger_id")
            age = body.get("measurement_age_triggers")
            source_time = body.get("last_measurement_exposure_timestamp_ns")
            age_ns = body.get("measurement_age_ns")
            if (source_trigger is None) != (age is None):
                raise ValueError("inconsistent source trigger age")
            if (source_time is None) != (age_ns is None):
                raise ValueError("inconsistent source timestamp age")
            if source_trigger is not None:
                source = _nonnegative_int(source_trigger, "measurement_trigger_id")
                elapsed = _nonnegative_int(age, "measurement_age_triggers")
                if source > frame_id or frame_id - source != elapsed:
                    raise ValueError("inconsistent source trigger/age")
            if source_time is not None:
                t = _nonnegative_int(source_time, "last_measurement_exposure_timestamp_ns")
                elapsed_ns = _nonnegative_int(age_ns, "measurement_age_ns")
                if t > timestamp or timestamp - t != elapsed_ns:
                    raise ValueError("inconsistent source timestamp/age")
            if body.get("metric_scale_validated") is not False:
                raise ValueError("consumer cannot assume verified metric scale")
            pose_columns = (
                "position_world_from_body",
                "quaternion_world_from_body_xyzw",
                "position_world_from_board",
                "quaternion_world_from_board_xyzw",
            )
            if body.get("evidence_kind") == "MEASURED":
                if (
                    body.get("tracking_status") != "TRACKING"
                    or age != 0 or age_ns != 0
                    or source_trigger != frame_id or source_time != timestamp
                ):
                    raise ValueError("MEASURED rigid body is stale/invalid")
                if len(observed) < self.min_visible[name]:
                    raise ValueError("MEASURED body lacks enough observed model markers")
                values = {}
                for label in pose_columns:
                    xyzq = _finite_list(
                        body.get(label), 4 if "quaternion" in label else 3, label
                    )
                    if "quaternion" in label:
                        if abs(sum(x*x for x in xyzq) - 1.0) > 1e-6:
                            raise ValueError("non-unit XYZW quaternion")
                    values[label] = np.asarray(xyzq, dtype=np.float64)
                R_world_board = Rotation.from_quat(
                    values["quaternion_world_from_board_xyzw"]
                ).as_matrix()
                R_world_body = Rotation.from_quat(
                    values["quaternion_world_from_body_xyzw"]
                ).as_matrix()
                R_body_board, t_body_board = self.body_mounts[name]
                expected_R = R_world_board @ R_body_board.T
                expected_t = (
                    values["position_world_from_board"]
                    - expected_R @ t_body_board
                )
                if (
                    not np.allclose(R_world_body, expected_R, atol=1e-6, rtol=0)
                    or not np.allclose(
                        values["position_world_from_body"],
                        expected_t, atol=1e-6, rtol=0
                    )
                ):
                    raise ValueError("PoseFrame board/body transform disagrees with asset")
                quality = body.get("quality")
                if not isinstance(quality, dict):
                    raise ValueError("MEASURED rigid body needs quality metrics")
                for metric in ("rigid_rms_world_units", "reprojection_max_rms_px"):
                    number = quality.get(metric)
                    if isinstance(number, bool) or not isinstance(number, (int, float)):
                        raise ValueError("invalid rigid body quality")
                    if not np.isfinite(float(number)) or number < 0:
                        raise ValueError("nonfinite rigid body quality")
            else:
                if (
                    body.get("evidence_kind") != "INVALID"
                    or body.get("tracking_status") not in ("DISCOVER", "DEGRADED", "LOST")
                    or any(body.get(label) is not None for label in pose_columns)
                    or body.get("quality") is not None
                ):
                    raise ValueError("INVALID rigid body must have null pose and quality")

        self.last_frame_id = frame_id
        self.last_exposure_ns = timestamp
        self.calibration_sha256 = calibration
        self.clock_domain = domain

    def consume(self, packet: object) -> dict[str, Any]:
        """Validate independent wire data, then apply state transactionally."""
        if not isinstance(packet, dict):
            raise ValueError("stream envelope must be a bounded JSON object")
        try:
            cost = _json_size(packet)
        except (TypeError, OverflowError, ValueError) as exc:
            raise ValueError("stream envelope contains non-JSON values") from exc
        if cost > self.max_message_bytes:
            raise ValueError("stream envelope must be a bounded JSON object")
        if (
            packet.get("protocol") != _STREAM_PROTOCOL
            or type(packet.get("schema_version")) is not int
            or packet["schema_version"] != 1
        ):
            raise ValueError("unsupported PoseStream wire protocol")
        session = _nonnegative_int(packet.get("session_id"), "session_id")
        sequence = _nonnegative_int(packet.get("sequence"), "sequence")
        payload = packet.get("payload")
        if not isinstance(payload, dict):
            raise ValueError("PoseStream payload must be an object")
        kind = payload.get("kind")
        new_session = self.session_id is None or session != self.session_id
        if new_session:
            if (
                (self.session_id is not None and session <= self.session_id)
                or sequence != 0 or kind != "asset_description"
            ):
                raise ValueError("new/reconnected stream must start with fresh asset")
            revision, boards, minimum, transforms = _asset_identity(payload)
            self.session_id = session
            self.pattern_sha256 = revision
            self.boards = boards
            self.min_visible = minimum
            self.body_mounts = transforms
            self.last_frame_id = None
            self.last_exposure_ns = None
            self.calibration_sha256 = None
            self.clock_domain = None
            self.last_fused_timestamp_ns = None
            self.fused_imu_id = None
            self.fused_clock_domain = None
            self.fused_mapping_sha256 = None
            self.fused_imu_calibration_sha256 = None
            self.fused_optical_calibration_sha256 = None
            self.last_sequence = sequence
            return deepcopy(payload)

        if self.last_sequence is not None and sequence <= self.last_sequence:
            raise ValueError("repeated or stale PoseStream sequence")
        if self.pattern_sha256 is None:
            raise ValueError("stream has no active asset description")
        if kind == "fused_attitude_state":
            expected_optical_calibration = (
                self.calibration_sha256
                if self.calibration_sha256 is not None
                else self.fused_optical_calibration_sha256
            )
            checked = validate_fused_attitude_payload(
                payload,
                known_body_ids=set(self.boards),
                expected_pattern_sha256=self.pattern_sha256,
                expected_optical_calibration_sha256=expected_optical_calibration,
                previous_timestamp_ns=self.last_fused_timestamp_ns,
            )
            # Revisions and timestamps commit only after the full payload passes.
            if self.fused_imu_id is not None and checked["imu_id"] != self.fused_imu_id:
                raise ValueError("fused attitude IMU identity changed in session")
            if self.fused_clock_domain is not None and checked["clock_domain"] != self.fused_clock_domain:
                raise ValueError("fused attitude clock domain changed in session")
            if self.fused_mapping_sha256 is not None and checked["clock_mapping_sha256"] != self.fused_mapping_sha256:
                raise ValueError("fused attitude clock mapping changed in session")
            if self.fused_imu_calibration_sha256 is not None and checked["imu_calibration_sha256"] != self.fused_imu_calibration_sha256:
                raise ValueError("fused attitude IMU calibration changed in session")
            if self.fused_optical_calibration_sha256 is not None and checked["optical_calibration_sha256"] != self.fused_optical_calibration_sha256:
                raise ValueError("fused attitude optical calibration changed in session")
            if self.calibration_sha256 is not None and checked["optical_calibration_sha256"] != self.calibration_sha256:
                raise ValueError("fused attitude calibration disagrees with optical PoseStream")
            self.last_fused_timestamp_ns = checked["timestamp_ns"]
            self.fused_imu_id = checked["imu_id"]
            self.fused_clock_domain = checked["clock_domain"]
            self.fused_mapping_sha256 = checked["clock_mapping_sha256"]
            self.fused_imu_calibration_sha256 = checked["imu_calibration_sha256"]
            self.fused_optical_calibration_sha256 = checked["optical_calibration_sha256"]
            if self.calibration_sha256 is None:
                self.calibration_sha256 = checked["optical_calibration_sha256"]
            self.last_sequence = sequence
            return deepcopy(checked)
        if kind != "pose_frame":
            raise ValueError("unsupported mid-session payload kind")
        # A frame is validated fully before source clock/triggers advance.
        self._check_frame(payload)
        self.last_sequence = sequence
        return deepcopy(payload)



def flush_to_socketio(
    publisher: BoundedPoseStreamPublisher,
    subscriber_id: str,
    socketio_emit: Callable[..., object],
    *,
    max_messages: int = 20,
    event_name: str = "mocap:pose-stream",
) -> int:
    """Adapt to an injected Socket.IO emit(event,packet,to=client) method.

    Network infrastructure and routing must be provided by the application.
    This operation acknowledges only a successful *local emit call*, NOT
    remote client reception. Subscribers must reconnect/resync from assets.
    No serial, arming or flight-control channel is touched.
    """
    if not isinstance(publisher, BoundedPoseStreamPublisher):
        raise ValueError("publisher must be a BoundedPoseStreamPublisher")
    if not callable(socketio_emit):
        raise ValueError("socketio_emit must be callable")
    def emit(event: str, packet: dict[str, Any], client: str):
        return socketio_emit(event, packet, to=client)

    return publisher.flush_with_callback(
        subscriber_id, emit,
        max_messages=max_messages, event_name=event_name,
    )
