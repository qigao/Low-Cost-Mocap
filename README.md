# Low-Cost-Mocap

### A general-purpose optical motion-capture engine for fixed external cameras, active LED/IR rigid-body markers, and optional body-mounted IMU fusion

The repository originated as an indoor multi-drone mocap project. Its target architecture is now a reusable Outside-In motion-capture engine: optical measurement, time alignment, state estimation and quality-qualified pose output are core capabilities; drone control is a separate downstream consumer.

- [General optical–IMU architecture and implementation roadmap](docs/GENERAL_MOCAP_ARCHITECTURE.md)
- [Typed IMU sample and clock mapping contract](computer_code/api/imu_contract.py)
- [IMU-to-body calibration contract](computer_code/api/imu_calibration.py)
- [Architecture epic / acceptance tasks](https://github.com/qigao/Low-Cost-Mocap/issues/33)

**Implementation status:** LED identity tracking, synchronized-batch software contracts, multi-view geometry, rigid-body fitting, optical PoseStream v1, bounded recording/replay, typed IMU sample/clock-mapping and calibration contracts, gyro-only attitude propagation, software optical/IMU attitude fusion, versioned fused-attitude messages, deterministic replay and single-series orientation metrics now exist. A paired optical-only-vs-fused report, strict offline CLI, and capture-bundle-bound report path are available in software. Hardware-qualified camera exposure/trigger, hardware clock qualification, measured calibration provenance and independently qualified fusion remain **open**. No current IMU-only module estimates global position/velocity. Hardware evidence gates remain in [Issues #2–#7](docs/HARDWARE_VALIDATION.md).

## YouTube Video
Watch this for information about the project & a demo!
[https://youtu.be/0ql20JKrscQ?si=jkxyOe-iCG7fa5th](https://youtu.be/0ql20JKrscQ?si=jkxyOe-iCG7fa5th)

[<img src="https://github.com/jyjblrd/Mocap-Drones/blob/main/images/thumbnail.png">](https://youtu.be/0ql20JKrscQ?si=jkxyOe-iCG7fa5th)

## Architectural Diagram
![](https://github.com/jyjblrd/Mocap-Drones/blob/main/images/architecture.png?raw=true)

## Dependencies
Install the pseyepy python library: [https://github.com/bensondaled/pseyepy](https://github.com/bensondaled/pseyepy)

This project requires the sfm (structure from motion) OpenCV module, which requires you to compile OpenCV from source[^1]. This is a bit of a pain, but these links should help you get started: [SFM dependencies](https://docs.opencv.org/4.x/db/db8/tutorial_sfm_installation.html) [OpenCV module installation guide](https://github.com/opencv/opencv_contrib/blob/master/README.md)

[^1]: ⚠️ The experimental [`no-cv-sfm`](https://github.com/jyjblrd/Low-Cost-Mocap/tree/no-cv-sfm) branch removes this OpenCV-SFM dependency, however it is *completely* untested. It is recommended to first try use the `main` branch which is tested, however feedback and bug reports on the `no-cv-sfm` branch are greatly appreciated. 

install npm and yarn

## Runing the code

From the computer_code directory Run `yarn install` to install node dependencies 

Then run `yarn run dev` to start the webserver. You will be given a url view the frontend interface.

In another terminal window, run `python3 api/index.py` to start the backend server. This is what receives the camera streams and does motion capture computations.

## Multi-board LED pattern configuration (experimental software)

Multiple LED boards are configured with one globally unique marker ID and
temporal code per LED, plus board-local measured 3D positions. A shared code
epoch and optional common ON pose phase let independent camera observations
join the correct board without distance-only drone-index heuristics.

- [Board pattern registry and recognition guide](computer_code/api/BOARD_PATTERNS.md)
- [Two-board balanced example JSON](computer_code/api/board-patterns.example.json)
- [High-visibility 16-phase candidate](computer_code/api/board-patterns.continuous.example.json)
- [Every-trigger observability analysis](computer_code/api/BOARD_PATTERNS.md#per-phase-rigid-body-visibility-a-separate-design-gate)
- [Identity-locked continuous pose prototype](computer_code/api/CONTINUOUS_TRACKING.md)
- [Atomic multi-camera local track integration and PoseStream v1 contract](computer_code/api/POSE_STREAM.md)
- [Bounded publisher and independent consumer](computer_code/api/pose_stream_broker.py)
- [Bounded JSONL recording/replay](computer_code/api/pose_stream_recording.py)
- [Versioned capture bundle manifest validator](computer_code/api/CAPTURE_BUNDLE.md) — verifies schema, paths, sizes and artifact hashes without certifying hardware
- [Bundle-gated offline replay orchestrator](computer_code/api/bundle_gated_offline_replay.py) — runs statistics only after the full bundle verifies
- [Isolated read-only Socket.IO observer and Vite Pose Monitor](computer_code/api/POSE_OBSERVER.md) — bounded poll acknowledgements, **not connected to hardware or flight control**
- Software-only checker: `cd computer_code/api && python board_pattern_cli.py board-patterns.continuous.example.json --require-every-phase`

**Important:** These are standalone software modules and tests. The legacy
camera capture and serial flight-control code has **not** been switched.
Real sensor-exposure synchronization, physical scale and motion-to-control
safety remain in [hardware validation Issues #2–#7](docs/HARDWARE_VALIDATION.md).

**Legacy controller safety default:** `computer_code/api/index.py` now binds to
`127.0.0.1` with debug/reloader disabled by default. CORS origins are an
explicit local-development allowlist; override with
`MOCAP_ALLOWED_ORIGINS` only when needed. Serial flight-control events are
denied unless `MOCAP_ENABLE_LEGACY_FLIGHT_CONTROL=1` is explicitly set, and
even then only loopback clients are authorized. Setting `MOCAP_BIND_HOST` to
a LAN address does **not** authorize remote flight-control commands. Remote
control requires a separately reviewed authentication design; do not expose
this legacy controller to untrusted networks. Serial command payloads are also
checked for exact vector lengths, valid drone indexes, finite numbers and broad
sanity caps (17 PID values with magnitude <= 1000, 3 setpoint coordinates with
magnitude <= 100 m, and 4 integer trim values with magnitude <= 1000). Arm
heartbeat counters must be monotonic per connection and within JavaScript's
safe-integer range; replayed counters are rejected and disconnected sessions
are removed from a bounded registry. These are input-safety caps, **not**
qualified flight-envelope limits; confirm hardware units/ranges before any live use.

## Documentation
The documentation for this project is admittedly pretty lacking, if anyone would like to put type definitions in the Python code that would be amazing and probably go a long way to helping the readability of the code. Feel free to also use the [discussion](https://github.com/jyjblrd/Mocap-Drones/discussions) tab to ask questions.

My blog post has some more information about the drones & camera: [joshuabird.com/blog/post/mocap-drones](https://joshuabird.com/blog/post/mocap-drones)

[This post](https://github.com/jyjblrd/Low-Cost-Mocap/discussions/11#discussioncomment-9380283) by gumby0q explains how `camera_params.json` can be calculated for your cameras.



## "Inside-Out" Multi-Agent Tracking (SLAM)
This motion capture system is an "outside-in" system, with external cameras tracking objects within a fixed space. There are also "inside-out" systems which use cameras on the drones/robots to determine their locations, not requiring any external infrastructure. 

My undergraduate dissertation presents such a system, which is capable of localizing multiple agents within a world in real time using purely visual data, with state-of-the-art performance. Check it out here: [https://github.com/jyjblrd/distributed_visual_SLAM](https://github.com/jyjblrd/distributed_visual_SLAM)
