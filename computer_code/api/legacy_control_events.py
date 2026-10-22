"""Isolated, testable legacy Socket.IO control handlers.

Every event that can directly command the serial port or enable the tracking
loop that emits serial commands is authorized here. Remote control remains
disabled; explicit opt-in still permits loopback clients only.
"""
from __future__ import annotations

import json
import os
import time
from typing import Any, Callable

from flask import request

from legacy_control_security import (
    ArmCounterRegistry,
    legacy_control_peer_allowed,
    validate_arm_payload,
    validate_pid_payload,
    validate_setpoint_payload,
    validate_trim_payload,
)


def register_legacy_control_handlers(
    socketio: Any,
    *,
    serial_port: Any,
    serial_lock: Any,
    get_camera_state: Callable[[], Any],
    is_camera_initialized: Callable[[], bool],
    num_objects: int = 2,
    sleep: Callable[[float], None] = time.sleep,
    enabled_getter: Callable[[], str | None] | None = None,
) -> ArmCounterRegistry:
    """Register protected serial and tracking-control events on a Socket.IO server."""
    if type(num_objects) is not int or num_objects < 1:
        raise ValueError("num_objects must be a positive integer")
    if enabled_getter is None:
        enabled_getter = lambda: os.environ.get("MOCAP_ENABLE_LEGACY_FLIGHT_CONTROL")

    arm_counter_registry = ArmCounterRegistry(max_clients=64)

    def _legacy_control_authorized() -> bool:
        if legacy_control_peer_allowed(request.remote_addr, enabled_getter()):
            return True
        socketio.emit(
            "legacy-control-rejected",
            {"reason": "legacy flight control is disabled or the client is not loopback"},
            to=request.sid,
        )
        return False

    def _validated_control_payload(validator: Callable[[object], Any], data: object):
        try:
            return validator(data)
        except (ValueError, TypeError, KeyError) as exc:
            socketio.emit("legacy-control-rejected", {"reason": str(exc)}, to=request.sid)
            return None

    def _reject(reason: str) -> None:
        socketio.emit("legacy-control-rejected", {"reason": reason}, to=request.sid)

    @socketio.on("disconnect")
    def _discard_legacy_control_session():
        arm_counter_registry.discard(request.sid)

    @socketio.on("arm-drone")
    def arm_drone(data):
        if not _legacy_control_authorized():
            return
        if not is_camera_initialized():
            return
        validated = _validated_control_payload(validate_arm_payload, data)
        if validated is None:
            return
        drone_armed, count, _current_drone_index = validated
        if len(drone_armed) != num_objects:
            _reject("droneArmed count does not match configured object count")
            return
        if not arm_counter_registry.accept(request.sid, count):
            _reject("stale or replayed arm command counter")
            return
        serial_messages = [
            f"{drone_index}{json.dumps({'armed': drone_armed[drone_index]}, allow_nan=False)}".encode("utf-8")
            for drone_index in range(num_objects)
        ]
        if len(serial_messages) != num_objects:
            return
        get_camera_state().drone_armed = list(drone_armed)
        for message in serial_messages:
            with serial_lock:
                serial_port.write(message)
            sleep(0.01)

    @socketio.on("set-drone-pid")
    def set_drone_pid(data):
        if not _legacy_control_authorized():
            return
        validated = _validated_control_payload(validate_pid_payload, data)
        if validated is None:
            return
        drone_index, pid_values = validated
        message = f"{drone_index}{json.dumps({'pid': pid_values}, allow_nan=False)}".encode("utf-8")
        with serial_lock:
            serial_port.write(message)
            sleep(0.01)

    @socketio.on("set-drone-setpoint")
    def set_drone_setpoint(data):
        if not _legacy_control_authorized():
            return
        validated = _validated_control_payload(validate_setpoint_payload, data)
        if validated is None:
            return
        drone_index, coordinates = validated
        message = f"{drone_index}{json.dumps({'setpoint': coordinates}, allow_nan=False)}".encode("utf-8")
        with serial_lock:
            serial_port.write(message)
            sleep(0.01)

    @socketio.on("set-drone-trim")
    def set_drone_trim(data):
        if not _legacy_control_authorized():
            return
        validated = _validated_control_payload(validate_trim_payload, data)
        if validated is None:
            return
        drone_index, trim_values = validated
        message = f"{drone_index}{json.dumps({'trim': trim_values}, allow_nan=False)}".encode("utf-8")
        with serial_lock:
            serial_port.write(message)
            sleep(0.01)

    @socketio.on("locate-objects")
    def start_or_stop_locating_objects(data):
        if not _legacy_control_authorized():
            return
        if not isinstance(data, dict) or set(data) != {"startOrStop"}:
            _reject("locate-objects payload has missing or unknown fields")
            return
        action = data["startOrStop"]
        if action not in ("start", "stop"):
            _reject("locate-objects startOrStop must be start or stop")
            return
        cameras = get_camera_state()
        if action == "start":
            cameras.start_locating_objects()
        else:
            cameras.stop_locating_objects()

    @socketio.on("triangulate-points")
    def live_mocap(data):
        if not _legacy_control_authorized():
            return
        if (
            not isinstance(data, dict)
            or set(data) != {"startOrStop", "cameraPoses", "toWorldCoordsMatrix"}
            or data.get("startOrStop") not in ("start", "stop")
        ):
            _reject("triangulate-points payload has invalid fields or action")
            return
        cameras = get_camera_state()
        cameras.to_world_coords_matrix = data["toWorldCoordsMatrix"]
        if data["startOrStop"] == "start":
            cameras.start_trangulating_points(data["cameraPoses"])
        else:
            cameras.stop_trangulating_points()

    return arm_counter_registry
