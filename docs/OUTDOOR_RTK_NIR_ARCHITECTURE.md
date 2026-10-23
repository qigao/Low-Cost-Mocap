# Outdoor RTK + Active-NIR Motion-Capture Profile

**Status:** proposed architecture and software contracts; no outdoor hardware has been qualified.
**Core topology:** retain fixed-camera Outside-In tracking. Add RTK GNSS as an absolute-position observation and active near-infrared (NIR) markers as the optical identity/pose channel.
**Tracking/flight boundary:** this profile is a general motion-capture capability. It does not authorize vehicle control.

## 1. Short answer

Yes, RTK + active NIR is viable outdoors, but the sensors solve different problems:

- **RTK GNSS:** global/georeferenced position when the receiver has a trustworthy fixed solution and correction data.
- **NIR camera network + active LED markers:** high-rate relative optical observations, marker identity and rigid-body orientation while enough cameras see the marker constellation.
- **IMU:** short-term propagation during optical gaps, with growing uncertainty and a bounded timeout.
- **Wired synchronization daemon:** camera discovery/configuration, trigger/PTP health, frame metadata, dropped-frame detection and capture provenance.

RTK alone does not provide full six-degree-of-freedom pose from one antenna. NIR tracking alone does not create a global coordinate frame unless the camera volume is surveyed/georeferenced. The combined system must explicitly calibrate both.

## 2. Three outdoor operating profiles

### A. Instrumented outdoor capture volume — implement first

Fixed GigE Vision cameras with global shutters observe active NIR LEDs on the moving rigid body. The body carries an RTK rover and optionally an IMU. Survey the camera rig/ground-control targets into a local East-North-Up (ENU) frame, then align the optical world frame to that surveyed frame.

- Optical reconstruction supplies rigid-body pose and high-rate relative motion within camera coverage.
- RTK supplies an independent global position observation and a consistency check.
- IMU propagates orientation/motion only for a bounded interval when optical measurements are missing.
- When RTK becomes FLOAT/invalid, preserve valid local optical pose but mark the georeferenced position DEGRADED.
- When optical visibility is lost, do not pretend RTK position alone supplies the body's full orientation.

This is the closest extension of the existing Outside-In engine.

### B. Open-area operation without a fixed camera network

RTK + IMU can provide global navigation, and a multi-antenna GNSS setup may provide heading when its baseline and solution are qualified. Active NIR can assist only where cameras or beacons have been deployed. This is a navigation/INS architecture, not full optical mocap across an uninstrumented field.

### C. Inside-Out NIR (later, separate topology)

Cameras mounted on the moving body observe fixed active NIR beacons. This can reduce the need for a fixed external camera array, but it requires a different calibration, visibility and association model. Implement it as another adapter/profile; do not silently reinterpret the current Outside-In camera IDs or calibration.

## 3. NIR optical design under sunlight

Do not assume that IR automatically removes sunlight. Solar irradiance includes near-IR, and grass/concrete/metallic backgrounds have different NIR reflectance. Outdoor capture is possible with suitable cameras, powerful active illumination and tuning, but working range is generally shorter and calibration more sensitive to environmental changes. See the [OptiTrack outdoor setup guidance](https://docs.optitrack.com/v3.3/robotics/outdoor-tracking-setup) and the research on [modulated IR beacons for outdoor positioning](https://www.mdpi.com/1424-8220/20/8/2163).

Initial engineering recommendations:

1. Start by bench-testing 850 nm and 940 nm emitters with the actual sensor, lens and bandpass filter. Do not hard-code a wavelength before direct-sun testing.
2. Use a narrow optical bandpass matched to the emitter, global-shutter sensors, short controlled exposure, active pulsing and sufficient dynamic range.
3. Keep the temporal code synchronized to the real trigger epoch. Treat saturated frames, ambiguous brightness and missing observations as erasures—not OFF bits.
4. Test direct sun, overcast, shade, grass, asphalt, concrete, reflective surfaces, distance, motion blur and partial occlusion.
5. Validate LED pulse current, duty cycle, temperature and eye safety for the chosen emitter and mounting geometry.
6. Recalibrate when camera mounts move or temperature/wind changes the rig geometry.

The existing pattern report now exposes per-marker duty cycle, global LED load per phase, Hamming distance and erasure budget. These are codebook/geometry diagnostics, not a measurement of physical LED output or outdoor visibility.

## 4. Wired camera synchronization daemon

Each accepted batch exposes a tuple of the existing `FrameStamp` contract
and an immutable camera-ID-to-image-payload mapping, so downstream optical
detection can consume the batch without reconstructing timestamps from host
arrival time. The daemon validates the full batch before creating these
pipeline inputs.

The provider-neutral lifecycle orchestrator is implemented in
`computer_code/api/camera_sync_daemon.py`. Hardware vendors supply a
`CameraAdapter` for each physical camera and a `TriggerController` for the
shared external trigger line. The orchestrator verifies device serial,
shutter/timestamp semantics and clock-domain metadata before arming; it emits
only complete, validated batches. Any timeout, trigger mismatch, packet loss
or skew violation moves the session to `FAULT`, performs best-effort cleanup,
and requires explicit reset/re-arm. It does not implement a vendor SDK or
spawn its own process; `run_forever` is intended to run in the service thread
managed by the deployment supervisor.

## 4. Wired camera synchronization daemon

Keep device SDK calls outside the tracking estimator. The daemon should own the camera lifecycle:

`DISCOVER \rightarrow CONFIGURE \rightarrow VERIFY_CLOCK \rightarrow ARM
\rightarrow CAPTURE \rightarrow DRAIN \rightarrow STOP`

For each physical camera, record:

- stable camera ID and serial number, firmware and SDK version;
- GigE Vision transport details, global/rolling shutter, exposure settings and trigger mode;
- timestamp semantics (exposure start/midpoint/end), frame counter and trigger ID;
- PTP capability/lock state, clock-domain identity, packet loss/overrun counters;
- host receive time as a separate diagnostic field;
- pattern/calibration revision and the raw-frame checksum.

### Trigger strategy

**Preferred:** distribute one physical hardware trigger to all cameras and the LED controller, with frame/trigger counters and sensor-exposure timestamps recorded by the camera adapter. If timestamps come from separate camera clocks, require PTP lock or a validated clock-mapping/common-clock mechanism before comparing timestamps; the physical trigger and clock alignment solve different problems.

**Alternative where supported:** PTP-scheduled hardware actions with measured clock state and a validated relationship between the scheduled action and sensor exposure. PTP may also align timestamp clocks while a separate external trigger drives the exposure.

PTP lock means clocks are synchronized; it does not by itself prove that sensor exposures began simultaneously. A GigE packet arrival timestamp is not an exposure timestamp. GNSS 1PPS can discipline a timing generator, but a qualified divider/trigger device must generate the actual camera-rate pulses. Do not create frame IDs by counting Python polls or packets received.

The new provider-neutral software contract lives in `computer_code/api/outdoor_sensor_contract.py`. It validates camera identity, trigger association, clock domain, exposure timestamp semantics, frame-counter monotonicity, packet-loss flags and batch skew. It also defines an RTK sample with solution status, solution epoch, ECEF coordinates, accuracy estimates, correction age, satellite count, datum and antenna reference point.

These contracts do **not** implement a vendor camera SDK or prove hardware timing. A real adapter must map the camera's documented chunk metadata into the contract and pass independent timing qualification before its timestamps are trusted.

## 5. RTK-to-optical coordinate integration

1. Preserve GNSS solution status: FIXED, FLOAT, DGNSS, SINGLE or INVALID. Do not collapse these into a generic `valid=true`.
2. Carry the receiver solution epoch and clock domain separately from host receive time.
3. Convert the documented WGS-84 ECEF position to the surveyed ENU/world frame with the helpers in `computer_code/api/rtk_coordinates.py`; ENU axis order is East, North, Up.
4. Calibrate the lever arm from the rigid-body origin to the GNSS antenna phase center in body coordinates. Given body-to-world rotation `R_enu_from_body`, recover the body origin as `p_body_enu = p_antenna_enu - R_enu_from_body * r_antenna_body`. Position of the antenna is not position of the body origin.
5. Treat reported horizontal/vertical accuracy, correction age and satellite count as inputs to predeclared gates; a FIXED flag alone is insufficient.
6. A single antenna is a position observation, not full body orientation. Orientation comes from optical rigid-body geometry, qualified IMU fusion or a qualified multi-antenna heading baseline.
7. Keep optical-only, RTK-only and fused results separately reportable for A/B analysis. Reject stale clock mappings and frame mismatches; never silently jump between local optical and global ENU frames.

## 6. Software acceptance gates

The unit tests for `outdoor_sensor_contract.py` cover:

- rejection of host receive/dequeue time as exposure time;
- trigger, camera serial, clock-domain and frame-counter mismatch;
- missing exposure timestamps, packet loss and unlocked PTP when PTP scheduling is required;
- missing/duplicate camera frames and exposure-skew threshold violations;
- RTK FIXED acceptance under explicit caller-provided limits and rejection of FLOAT, stale corrections, low satellite count or excessive reported error;
- non-finite ECEF coordinates.

The new `rtk_coordinates.py` tests cover WGS-84 geodetic-to-ECEF conversion, ECEF/ENU round trips, explicit RTK quality gating before ENU conversion, and body-origin recovery from antenna position using a proper rotation and calibrated lever arm.

These tests qualify only the software contract. They do not establish real camera support, outdoor range, RTK accuracy, or exposure timing.

## 7. Hardware evidence plan

Keep [HW-01](https://github.com/qigao/Low-Cost-Mocap/issues/2) open. Outdoor implementation is tracked in [issue #74](https://github.com/qigao/Low-Cost-Mocap/issues/74).

Before hardware acceptance, collect a reproducible bundle containing:

- exact camera models, serials, firmware, SDK, trigger wiring and PTP configuration;
- raw images and camera chunk metadata, frame/trigger counters and packet-loss logs;
- independent photodiode/oscilloscope or equivalent reference for trigger-to-exposure timing;
- GNSS correction source, fix-state logs, correction age, antenna placement/lever arm and surveyed reference points;
- calibration and pattern hashes, predeclared limits, rejected-frame report and SHA-256 checksums;
- outdoor runs in sun/overcast/shade, different ranges and speeds, marker crossings and occlusions;
- position/orientation error, exposure-skew p50/p95/p99/max, drop rate, ID-switch rate, reacquisition jump, end-to-end latency and correct DEGRADED/LOST behavior.

Do not mark HW-01..HW-06 passed, enable a live controller, or call the system outdoor-qualified from synthetic tests or software CI alone.
