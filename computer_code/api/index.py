from helpers import camera_pose_to_serializable, calculate_reprojection_errors, bundle_adjustment, Cameras, triangulate_points
from KalmanFilter import KalmanFilter
from calibration_geometry import compose_world_to_camera_pose, select_relative_pose_by_cheirality

from flask import Flask, Response, request
import cv2 as cv
import numpy as np
import json
from scipy import linalg

from flask_socketio import SocketIO
import copy
import time
import serial
import threading
from ruckig import InputParameter, OutputParameter, Result, Ruckig
from flask_cors import CORS
import json
import os

from legacy_control_security import (
    configured_origins,
    legacy_control_peer_allowed,
    validate_arm_payload,
    validate_pid_payload,
    validate_setpoint_payload,
    validate_trim_payload,
)

ALLOWED_ORIGINS = configured_origins(os.environ.get("MOCAP_ALLOWED_ORIGINS"))

serialLock = threading.Lock()

ser = serial.Serial("/dev/cu.usbserial-02X2K2GE", 1000000, write_timeout=1, )

app = Flask(__name__)
CORS(app, origins=ALLOWED_ORIGINS, supports_credentials=True)
socketio = SocketIO(app, cors_allowed_origins=ALLOWED_ORIGINS)


def _legacy_control_authorized() -> bool:
    """Fail closed for serial commands unless explicitly enabled from loopback."""
    if legacy_control_peer_allowed(request.remote_addr, os.environ.get("MOCAP_ENABLE_LEGACY_FLIGHT_CONTROL")):
        return True
    socketio.emit(
        "legacy-control-rejected",
        {"reason": "legacy flight control is disabled or the client is not loopback"},
        to=request.sid,
    )
    return False


def _validated_control_payload(validator, data):
    """Return a validated payload or emit a fail-closed rejection."""
    try:
        return validator(data)
    except (ValueError, TypeError, KeyError) as exc:
        socketio.emit(
            "legacy-control-rejected",
            {"reason": str(exc)},
            to=request.sid,
        )
        return None

cameras_init = False

num_objects = 2

@app.route("/api/camera-stream")
def camera_stream():
    cameras = Cameras.instance()
    cameras.set_socketio(socketio)
    cameras.set_ser(ser)
    cameras.set_serialLock(serialLock)
    cameras.set_num_objects(num_objects)
    
    def gen(cameras):
        frequency = 150
        loop_interval = 1.0 / frequency
        last_run_time = 0
        i = 0

        while True:
            time_now = time.time()

            i = (i+1)%10
            if i == 0:
                socketio.emit("fps", {"fps": round(1/(time_now - last_run_time))})

            if time_now - last_run_time < loop_interval:
                time.sleep(last_run_time - time_now + loop_interval)
            last_run_time = time.time()
            frames = cameras.get_frames()
            jpeg_frame = cv.imencode('.jpg', frames)[1].tostring()

            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + jpeg_frame + b'\r\n')

    return Response(gen(cameras), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/api/trajectory-planning", methods=["POST"])
def trajectory_planning_api():
    data = json.loads(request.data)

    waypoint_groups = [] # grouped by continuious movement (no stopping)
    for waypoint in data["waypoints"]:
        stop_at_waypoint = waypoint[-1]
        if stop_at_waypoint:
            waypoint_groups.append([waypoint[:3*num_objects]])
        else:
            waypoint_groups[-1].append(waypoint[:3*num_objects])
    
    setpoints = []
    for i in range(0, len(waypoint_groups)-1):
        start_pos = waypoint_groups[i][0]
        end_pos = waypoint_groups[i+1][0]
        waypoints = waypoint_groups[i][1:]
        setpoints += plan_trajectory(start_pos, end_pos, waypoints, data["maxVel"], data["maxAccel"], data["maxJerk"], data["timestep"])

    return json.dumps({
        "setpoints": setpoints
    })

def plan_trajectory(start_pos, end_pos, waypoints, max_vel, max_accel, max_jerk, timestep):
    otg = Ruckig(3*num_objects, timestep, len(waypoints))  # DoFs, timestep, number of waypoints
    inp = InputParameter(3*num_objects)
    out = OutputParameter(3*num_objects, len(waypoints))

    inp.current_position = start_pos
    inp.current_velocity = [0,0,0]*num_objects
    inp.current_acceleration = [0,0,0]*num_objects

    inp.target_position = end_pos
    inp.target_velocity = [0,0,0]*num_objects
    inp.target_acceleration = [0,0,0]*num_objects

    inp.intermediate_positions = waypoints

    inp.max_velocity = max_vel*num_objects
    inp.max_acceleration = max_accel*num_objects
    inp.max_jerk = max_jerk*num_objects

    setpoints = []
    res = Result.Working
    while res == Result.Working:
        res = otg.update(inp, out)
        setpoints.append(copy.copy(out.new_position))
        out.pass_to_input(inp)

    return setpoints

@socketio.on("arm-drone")
def arm_drone(data):
    if not _legacy_control_authorized():
        return
    global cameras_init
    if not cameras_init:
        return
    validated = _validated_control_payload(validate_arm_payload, data)
    if validated is None:
        return
    drone_armed, _count, _current_drone_index = validated

    # Validate the complete command before any serial write or shared-state update.
    serial_messages = [
        f"{drone_index}{json.dumps({'armed': drone_armed[drone_index]})}".encode("utf-8")
        for drone_index in range(num_objects)
    ]
    if len(serial_messages) != num_objects:
        return
    Cameras.instance().drone_armed = list(drone_armed)
    for message in serial_messages:
        with serialLock:
            ser.write(message)
        time.sleep(0.01)

@socketio.on("set-drone-pid")
def arm_drone(data):
    if not _legacy_control_authorized():
        return
    validated = _validated_control_payload(validate_pid_payload, data)
    if validated is None:
        return
    drone_index, pid_values = validated
    serial_data = {"pid": pid_values}
    message = f"{drone_index}{json.dumps(serial_data, allow_nan=False)}".encode("utf-8")
    with serialLock:
        ser.write(message)
        time.sleep(0.01)

@socketio.on("set-drone-setpoint")
def arm_drone(data):
    if not _legacy_control_authorized():
        return
    validated = _validated_control_payload(validate_setpoint_payload, data)
    if validated is None:
        return
    drone_index, coordinates = validated
    serial_data = {"setpoint": coordinates}
    message = f"{drone_index}{json.dumps(serial_data, allow_nan=False)}".encode("utf-8")
    with serialLock:
        ser.write(message)
        time.sleep(0.01)

@socketio.on("set-drone-trim")
def arm_drone(data):
    if not _legacy_control_authorized():
        return
    validated = _validated_control_payload(validate_trim_payload, data)
    if validated is None:
        return
    drone_index, trim_values = validated
    serial_data = {"trim": trim_values}
    message = f"{drone_index}{json.dumps(serial_data, allow_nan=False)}".encode("utf-8")
    with serialLock:
        ser.write(message)
        time.sleep(0.01)


@socketio.on("acquire-floor")
def acquire_floor(data):
    cameras = Cameras.instance()
    object_points = data["objectPoints"]
    object_points = np.array([item for sublist in object_points for item in sublist])

    tmp_A = []
    tmp_b = []
    for i in range(len(object_points)):
        tmp_A.append([object_points[i,0], object_points[i,1], 1])
        tmp_b.append(object_points[i,2])
    b = np.matrix(tmp_b).T
    A = np.matrix(tmp_A)

    fit, residual, rnk, s = linalg.lstsq(A, b)
    fit = fit.T[0]

    plane_normal = np.array([[fit[0]], [fit[1]], [-1]])
    plane_normal = plane_normal / linalg.norm(plane_normal)
    up_normal = np.array([[0],[0],[1]], dtype=np.float32)

    plane = np.array([fit[0], fit[1], -1, fit[2]])

    # https://math.stackexchange.com/a/897677/1012327
    G = np.array([
        [np.dot(plane_normal.T,up_normal)[0][0], -linalg.norm(np.cross(plane_normal.T[0],up_normal.T[0])), 0],
        [linalg.norm(np.cross(plane_normal.T[0],up_normal.T[0])), np.dot(plane_normal.T,up_normal)[0][0], 0],
        [0, 0, 1]
    ])
    F = np.array([plane_normal.T[0], ((up_normal-np.dot(plane_normal.T,up_normal)[0][0]*plane_normal)/linalg.norm((up_normal-np.dot(plane_normal.T,up_normal)[0][0]*plane_normal))).T[0], np.cross(up_normal.T[0],plane_normal.T[0])]).T
    R = F @ G @ linalg.inv(F)

    R = R @ [[1,0,0],[0,-1,0],[0,0,1]] # i dont fucking know why

    cameras.to_world_coords_matrix = np.array(np.vstack((np.c_[R, [0,0,0]], [[0,0,0,1]])))

    socketio.emit("to-world-coords-matrix", {"to_world_coords_matrix": cameras.to_world_coords_matrix.tolist()})


@socketio.on("set-origin")
def set_origin(data):
    cameras = Cameras.instance()
    object_point = np.array(data["objectPoint"])
    to_world_coords_matrix = np.array(data["toWorldCoordsMatrix"])
    transform_matrix = np.eye(4)

    object_point[1], object_point[2] = object_point[2], object_point[1] # i dont fucking know why
    transform_matrix[:3, 3] = -object_point

    to_world_coords_matrix = transform_matrix @ to_world_coords_matrix
    cameras.to_world_coords_matrix = to_world_coords_matrix

    socketio.emit("to-world-coords-matrix", {"to_world_coords_matrix": cameras.to_world_coords_matrix.tolist()})

@socketio.on("update-camera-settings")
def change_camera_settings(data):
    cameras = Cameras.instance()
    
    cameras.edit_settings(data["exposure"], data["gain"])

@socketio.on("capture-points")
def capture_points(data):
    start_or_stop = data["startOrStop"]
    cameras = Cameras.instance()

    if (start_or_stop == "start"):
        cameras.start_capturing_points()
        return
    elif (start_or_stop == "stop"):
        cameras.stop_capturing_points()

@socketio.on("calculate-camera-pose")
def calculate_camera_pose(data):
    """Seed all camera extrinsics from pairwise calibrated, local-frame poses.

    Every essential-matrix translation has unknown metric scale. A unit
    adjacent-camera baseline is explicitly chosen as an initialization gauge;
    later scale calibration must establish actual physical units.
    """
    cameras = Cameras.instance()
    try:
        image_points = np.asarray(data["cameraPoints"], dtype=object)
        if (
            image_points.ndim != 3
            or image_points.shape[1:] != (cameras.num_cameras, 2)
        ):
            raise ValueError("cameraPoints must have shape (frames, cameras, 2)")
        image_points_t = image_points.transpose((1, 0, 2))

        camera_poses = [{"R": np.eye(3), "t": np.zeros(3)}]
        for camera_i in range(cameras.num_cameras - 1):
            left_raw = image_points_t[camera_i]
            right_raw = image_points_t[camera_i + 1]
            valid_mask = np.array([
                all(value is not None for value in p1)
                and all(value is not None for value in p2)
                for p1, p2 in zip(left_raw, right_raw)
            ])
            left = np.asarray(left_raw[valid_mask], dtype=np.float64)
            right = np.asarray(right_raw[valid_mask], dtype=np.float64)
            if len(left) < 8:
                raise ValueError(
                    f"camera pair {camera_i}/{camera_i + 1} needs at least 8 matched points"
                )
            fundamental, ransac_mask = cv.findFundamentalMat(
                left, right, cv.FM_RANSAC, 1.0, 0.99999
            )
            if (
                fundamental is None
                or np.asarray(fundamental).shape != (3, 3)
                or ransac_mask is None
            ):
                raise ValueError(f"camera pair {camera_i}/{camera_i + 1} has no valid fundamental matrix")
            inliers = np.asarray(ransac_mask).reshape(-1)
            if inliers.shape != (len(left),):
                raise ValueError(f"camera pair {camera_i}/{camera_i + 1} has malformed RANSAC inliers")
            left_inliers = left[inliers != 0]
            right_inliers = right[inliers != 0]
            if len(left_inliers) < 8:
                raise ValueError(f"camera pair {camera_i}/{camera_i + 1} has fewer than 8 RANSAC inliers")

            # Use the *current pair's* intrinsics, not K0/K1 for every pair.
            left_intrinsic = cameras.get_camera_params(camera_i)["intrinsic_matrix"]
            right_intrinsic = cameras.get_camera_params(camera_i + 1)["intrinsic_matrix"]
            essential = cv.sfm.essentialFromFundamental(
                fundamental, left_intrinsic, right_intrinsic
            )
            relative_rotations, relative_translations = cv.sfm.motionFromEssential(essential)
            relative_pose, _ = select_relative_pose_by_cheirality(
                left_inliers, right_inliers, left_intrinsic, right_intrinsic,
                relative_rotations, relative_translations,
                min_support=8,
            )
            # Relative camera extrinsics are camera_i -> camera_(i+1):
            # R_next = R_rel @ R_prev and t_next = R_rel @ t_prev + t_rel.
            camera_poses.append(compose_world_to_camera_pose(
                camera_poses[-1], relative_pose, baseline_length=1.0
            ))

        # This is still a projective-gauge initialization. The legacy
        # bundle-adjustment and later measured-distance calibration remain
        # separate acceptance gates; neither establishes hardware sync.
        camera_poses = bundle_adjustment(image_points, camera_poses, socketio)
    except (ValueError, np.linalg.LinAlgError) as exc:
        socketio.emit("camera-pose", {"error": str(exc)})
        return

    socketio.emit("camera-pose", {
        "camera_poses": camera_pose_to_serializable(camera_poses)
    })

@socketio.on("locate-objects")
def start_or_stop_locating_objects(data):
    cameras = Cameras.instance()
    start_or_stop = data["startOrStop"]

    if (start_or_stop == "start"):
        cameras.start_locating_objects()
        return
    elif (start_or_stop == "stop"):
        cameras.stop_locating_objects()

@socketio.on("determine-scale")
def determine_scale(data):
    object_points = data["objectPoints"]
    camera_poses = data["cameraPoses"]
    actual_distance = 0.15
    observed_distances = []

    for object_points_i in object_points:
        if len(object_points_i) != 2:
            continue

        object_points_i = np.array(object_points_i)

        observed_distances.append(np.sqrt(np.sum((object_points_i[0] - object_points_i[1])**2)))

    scale_factor = actual_distance/np.mean(observed_distances)
    for i in range(0, len(camera_poses)):
        camera_poses[i]["t"] = (np.array(camera_poses[i]["t"]) * scale_factor).tolist()

    socketio.emit("camera-pose", {"error": None, "camera_poses": camera_poses})


@socketio.on("triangulate-points")
def live_mocap(data):
    cameras = Cameras.instance()
    start_or_stop = data["startOrStop"]
    camera_poses = data["cameraPoses"]
    cameras.to_world_coords_matrix = data["toWorldCoordsMatrix"]

    if (start_or_stop == "start"):
        cameras.start_trangulating_points(camera_poses)
        return
    elif (start_or_stop == "stop"):
        cameras.stop_trangulating_points()


if __name__ == '__main__':
    bind_host = os.environ.get("MOCAP_BIND_HOST", "127.0.0.1")
    socketio.run(app, host=bind_host, port=3001, debug=False, use_reloader=False)