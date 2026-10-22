"""Opt-in read-only Socket.IO observer host for PoseStream v1.

This is deliberately NOT imported by the legacy index.py, which opens a
serial port at import time and exposes drone arming/setpoint handlers.
No camera SDK, serial device or flight-control module is imported here.

Security scope: loopback-only development preview with an explicit shared
viewer secret and allowlisted frontend origins. Socket.IO sends are local
queue dispatch confirmations, NOT independently acknowledged remote delivery.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from hashlib import sha256
import hmac
import threading
import time

from flask import Flask, jsonify, request
from flask_socketio import SocketIO

from board_patterns import BoardPatternRegistry
from identity_locked_board_tracker import LockedBoardFrame
from pose_stream_broker import (
    BoundedPoseStreamPublisher,
    _bounded_positive,
    flush_to_socketio,
)

POSE_OBSERVER_NAMESPACE = "/pose-observe"
POSE_OBSERVER_EVENT = "mocap:pose-stream"


def _viewer_secret(raw: object) -> str:
    if (
        not isinstance(raw, str)
        or len(raw) < 24
        or len(raw) > 256
        or any(ord(ch) < 33 or ord(ch) > 126 for ch in raw)
    ):
        raise ValueError("observer viewer secret must be 24..256 printable ASCII characters")
    return raw


def _origins(values: Sequence[str]) -> tuple[str, ...]:
    if (
        isinstance(values, str)
        or not isinstance(values, (tuple, list))
        or not values
        or len(values) > 12
    ):
        raise ValueError("observer requires a bounded explicit CORS origin allowlist")
    origins = []
    for value in values:
        if (
            not isinstance(value, str)
            or not (value.startswith("http://") or value.startswith("https://"))
            or value in ("*", "http://*", "https://*")
            or "*" in value
            or len(value) > 256
            or "/" in value.split("://", 1)[1]
        ):
            raise ValueError("invalid explicit observer CORS origin")
        origins.append(value)
    if len(origins) != len(set(origins)):
        raise ValueError("duplicate observer CORS origin")
    return tuple(origins)


class PoseObserverHost:
    """Single-owner observation service with a bounded subscriber broker.

    When used with threading Socket.IO, subscription/publish/dispatch share
    one RLock so a reconnect cannot race a publisher flush. The external
    application supplies only already validated LockedBoardFrame instances.
    The host itself never synthesizes hardware trigger IDs or positions.
    """

    def __init__(
        self,
        registry: BoardPatternRegistry,
        *,
        viewer_secret: str,
        allowed_origins: Sequence[str] = (
            "http://127.0.0.1:5173", "http://localhost:5173"
        ),
        max_subscribers: int = 8,
        max_pending_frames: int = 4,
        max_emit_per_dispatch: int = 8,
        min_poll_interval_ms: int = 50,
        monotonic_ns: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        if not isinstance(registry, BoardPatternRegistry):
            raise ValueError("registry must be BoardPatternRegistry")
        self._secret = _viewer_secret(viewer_secret)
        self._origins = _origins(allowed_origins)
        self._min_poll_ns = _bounded_positive(
            min_poll_interval_ms, "min_poll_interval_ms", maximum=60_000
        ) * 1_000_000
        self._max_dispatch = _bounded_positive(
            max_emit_per_dispatch, "max_emit_per_dispatch", maximum=1000
        )
        if not callable(monotonic_ns):
            raise ValueError("monotonic_ns must be a monotonic clock callback")
        self._clock_ns = monotonic_ns
        self._last_poll_ns: dict[str, int] = {}
        self._lock = threading.RLock()
        self.publisher = BoundedPoseStreamPublisher(
            registry,
            max_subscribers=max_subscribers,
            max_pending_frames=max_pending_frames,
        )
        self.app = Flask("low_cost_mocap_pose_observer")
        self.socketio = SocketIO(
            self.app,
            async_mode="threading",
            cors_allowed_origins=list(self._origins),
            logger=False, engineio_logger=False,
        )
        self._install_handlers()

    @staticmethod
    def _subscriber_key(sid: str) -> str:
        """Normalize URL-safe Engine.IO SIDs to the broker's bounded ID syntax.

        A real random Socket.IO SID can begin with '-' or '_', which are
        valid for transport but not the stricter broker subscriber syntax.
        Never mutate the *actual* SID used as the Socket.IO room target.
        """
        if not isinstance(sid, str) or not sid:
            raise ValueError("missing Socket.IO SID")
        return "viewer-" + sha256(sid.encode("utf-8")).hexdigest()[:40]

    @property
    def subscriber_count(self) -> int:
        with self._lock:
            return len(self.publisher.subscriber_ids)

    def _flush(self, sid: str, *, max_messages: int | None = None) -> int:
        limit = self._max_dispatch if max_messages is None else max_messages
        key = self._subscriber_key(sid)
        def local_emit(event: str, packet: dict, *, to: str):
            # Publisher subscription identity is a *normalized hash*;
            # wire delivery must still address the original Socket.IO SID.
            if to != key:
                raise ValueError("observer publisher/subscriber identity mismatch")
            self.socketio.emit(
                event, packet, to=sid, namespace=POSE_OBSERVER_NAMESPACE
            )
        return flush_to_socketio(
            self.publisher, key, local_emit,
            max_messages=limit, event_name=POSE_OBSERVER_EVENT,
        )

    def _install_handlers(self) -> None:
        @self.app.get("/pose-observe/healthz")
        def healthz():
            return jsonify({
                "service": "pose_observer",
                "mode": "read_only",
                "hardware_provenance_verified": False,
                "live_hardware_input_enabled": False,
            })

        @self.socketio.on("connect", namespace=POSE_OBSERVER_NAMESPACE)
        def viewer_connected(auth):
            # Never authenticate by a query parameter, cookie or public
            # frontend build constant; the viewer sends auth.token explicitly.
            if not isinstance(auth, dict) or not isinstance(auth.get("token"), str):
                return False
            token = auth["token"]
            if len(token) > 256 or not hmac.compare_digest(token, self._secret):
                return False
            sid = request.sid
            key = self._subscriber_key(sid)
            with self._lock:
                try:
                    self.publisher.subscribe(key)
                    self._last_poll_ns[sid] = -1
                    # Only static assets at connection time. Dynamic frames
                    # must come from a host-side publish_frame invocation.
                    self._flush(sid, max_messages=1)
                except (ValueError, RuntimeError):
                    if key in self.publisher.subscriber_ids:
                        self.publisher.unsubscribe(key)
                    self._last_poll_ns.pop(sid, None)
                    return False
            return True

        @self.socketio.on("disconnect", namespace=POSE_OBSERVER_NAMESPACE)
        def viewer_disconnected():
            sid = request.sid
            key = self._subscriber_key(sid)
            with self._lock:
                if key in self.publisher.subscriber_ids:
                    self.publisher.unsubscribe(key)
                self._last_poll_ns.pop(sid, None)

        @self.socketio.on("pose:poll", namespace=POSE_OBSERVER_NAMESPACE)
        def viewer_poll(_ignored=None):
            sid = request.sid
            key = self._subscriber_key(sid)
            with self._lock:
                if key not in self.publisher.subscriber_ids:
                    return {"ok": False, "error": "not-subscribed"}
                now = self._clock_ns()
                if type(now) is not int or now < 0:
                    raise ValueError("monotonic clock must return nonnegative nanoseconds")
                last = self._last_poll_ns.get(sid, -1)
                if last >= 0 and now - last < self._min_poll_ns:
                    return {"ok": False, "error": "rate-limited"}
                self._last_poll_ns[sid] = now
                sent = self._flush(sid)
                stats = self.publisher.subscriber_statistics(key)
                return {
                    "ok": True, "sent": sent,
                    "pending_frames": stats.pending_frames,
                    "dropped_frames": stats.dropped_frames,
                    "session_id": stats.session_id,
                }

    def publish_frame(self, frame: LockedBoardFrame) -> int:
        """Validate and enqueue one frame; never push to transport here.

        The subscriber must request a bounded flush via pose:poll. This
        keeps backpressure in the explicitly bounded broker instead of
        transferring an unbounded backlog to the Socket.IO transport queue.
        This is the ONLY ingress for pose data. No client can publish a
        pose, enable cameras, send setpoints or arm drones.
        """
        with self._lock:
            return self.publisher.publish(frame)

    def restart_source_session(self) -> None:
        """Host-only restart after a verified camera source reset."""
        with self._lock:
            self.publisher.restart_session()
            for sid in tuple(self._last_poll_ns):
                self._flush(sid, max_messages=1)

    def replace_pattern_registry(self, registry: BoardPatternRegistry) -> None:
        """Host-only pattern update and mandatory asset resend barrier."""
        with self._lock:
            self.publisher.replace_registry(registry)
            for sid in tuple(self._last_poll_ns):
                self._flush(sid, max_messages=1)
