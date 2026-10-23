import threading
import unittest

from camera_sync_daemon import (
    CameraDeviceInfo,
    CameraFrameEnvelope,
    CameraSyncDaemon,
    CameraSyncFault,
    CameraSyncState,
)
from outdoor_sensor_contract import CameraFrameMetadata, CameraSyncProfile


def profile(camera_id, serial):
    return CameraSyncProfile(
        camera_id=camera_id,
        device_serial=serial,
        transport="gige_vision",
        shutter_type="global",
        trigger_mode="external_line",
        timestamp_semantics="exposure_midpoint",
        clock_domain="rig-clock-A",
        max_exposure_skew_ns=1000,
    )


def envelope(camera_id, serial, counter, trigger, timestamp, payload=None):
    return CameraFrameEnvelope(
        CameraFrameMetadata(
            camera_id=camera_id,
            device_serial=serial,
            frame_counter=counter,
            trigger_id=trigger,
            acquisition_timestamp_ns=timestamp,
            clock_domain="rig-clock-A",
            timestamp_semantics="exposure_midpoint",
            hardware_triggered=True,
            ptp_locked=False,
        ),
        payload if payload is not None else f"image-{camera_id}-{counter}",
    )


class FakeCamera:
    def __init__(self, serial, frames, *, reported_serial=None, supports_trigger=True, fail_arm=False):
        self.serial = serial
        self.reported_serial = reported_serial or serial
        self.frames = list(frames)
        self.supports_trigger = supports_trigger
        self.fail_arm = fail_arm
        self.opened = False
        self.configured = False
        self.armed = False
        self.started = False
        self.closed = False

    def open(self):
        self.opened = True
        return CameraDeviceInfo(
            device_serial=self.reported_serial,
            firmware_version="fw-test",
            sdk_version="sdk-test",
            shutter_type="global",
            timestamp_semantics="exposure_midpoint",
            clock_domain="rig-clock-A",
            supports_external_trigger=self.supports_trigger,
        )

    def configure(self, configured_profile):
        self.configured = configured_profile.device_serial == self.serial

    def arm(self):
        self.armed = True
        if self.fail_arm:
            raise RuntimeError("fake arm failure")

    def start_acquisition(self):
        self.started = True

    def read_frame(self, timeout_s):
        if not self.frames:
            raise TimeoutError("fake frame timeout")
        return self.frames.pop(0)

    def stop_acquisition(self):
        self.started = False

    def disarm(self):
        self.armed = False

    def close(self):
        self.closed = True
        self.opened = False


class FakeTrigger:
    def __init__(self, fail_pulse=False):
        self.opened = False
        self.armed = False
        self.closed = False
        self.pulses = []
        self.fail_pulse = fail_pulse

    def open(self):
        self.opened = True

    def arm(self):
        self.armed = True

    def pulse(self, trigger_id):
        if self.fail_pulse:
            raise RuntimeError("pulse generator fault")
        self.pulses.append(trigger_id)

    def disarm(self):
        self.armed = False

    def close(self):
        self.closed = True
        self.opened = False


class CameraSyncDaemonTests(unittest.TestCase):
    def make_daemon(self, frames_a, frames_b, **kwargs):
        profiles = (profile(2, "CAM-02"), profile(5, "CAM-05"))
        cameras = {
            2: FakeCamera("CAM-02", frames_a, **kwargs.get("camera_a", {})),
            5: FakeCamera("CAM-05", frames_b, **kwargs.get("camera_b", {})),
        }
        trigger = FakeTrigger(fail_pulse=kwargs.get("fail_pulse", False))
        daemon = CameraSyncDaemon(
            profiles, cameras, trigger, first_trigger_id=42, frame_timeout_s=0.1
        )
        return daemon, cameras, trigger

    def test_start_capture_and_stop_emits_only_a_complete_validated_batch(self):
        daemon, cameras, trigger = self.make_daemon(
            [envelope(2, "CAM-02", 1, 42, 8_000_000, b"left")],
            [envelope(5, "CAM-05", 1, 42, 8_000_040, b"right")],
        )
        daemon.start()
        self.assertEqual(daemon.state, CameraSyncState.CAPTURING)
        batch = daemon.capture_batch()
        self.assertEqual(batch.summary.trigger_id, 42)
        self.assertEqual(batch.summary.exposure_skew_ns, 40)
        self.assertEqual([frame.payload for frame in batch.frames], [b"left", b"right"])
        self.assertEqual(trigger.pulses, [42])
        self.assertEqual(daemon.next_trigger_id, 43)
        daemon.stop()
        self.assertEqual(daemon.state, CameraSyncState.STOPPED)
        self.assertTrue(all(camera.closed for camera in cameras.values()))
        self.assertTrue(trigger.closed)

    def test_wrong_physical_serial_fails_start_and_enters_fault(self):
        daemon, cameras, trigger = self.make_daemon(
            [], [], camera_b={"reported_serial": "OTHER-CAMERA"}
        )
        with self.assertRaisesRegex(CameraSyncFault, "serial does not match"):
            daemon.start()
        self.assertEqual(daemon.state, CameraSyncState.FAULT)
        self.assertTrue(cameras[2].closed)
        self.assertTrue(cameras[5].closed)
        self.assertFalse(trigger.opened)
        with self.assertRaises(CameraSyncFault):
            daemon.start()
        daemon.reset_fault()
        self.assertEqual(daemon.state, CameraSyncState.STOPPED)

    def test_partial_camera_arm_failure_rolls_back_all_opened_resources(self):
        profiles = (profile(2, "CAM-02"), profile(5, "CAM-05"))
        cameras = {
            2: FakeCamera("CAM-02", []),
            5: FakeCamera("CAM-05", [], fail_arm=True),
        }
        trigger = FakeTrigger()
        daemon = CameraSyncDaemon(profiles, cameras, trigger)
        with self.assertRaisesRegex(CameraSyncFault, "fake arm failure"):
            daemon.start()
        self.assertEqual(daemon.state, CameraSyncState.FAULT)
        self.assertTrue(all(camera.closed for camera in cameras.values()))
        self.assertTrue(trigger.closed)
        self.assertFalse(trigger.armed)

    def test_trigger_mismatch_fails_closed_without_emitting_partial_batch(self):
        daemon, cameras, trigger = self.make_daemon(
            [envelope(2, "CAM-02", 1, 42, 8_000_000)],
            [envelope(5, "CAM-05", 1, 43, 8_000_040)],
        )
        daemon.start()
        with self.assertRaisesRegex(CameraSyncFault, "no partial batch emitted"):
            daemon.capture_batch()
        self.assertEqual(daemon.state, CameraSyncState.FAULT)
        self.assertEqual(trigger.pulses, [42])
        self.assertTrue(trigger.closed)
        self.assertTrue(all(camera.closed for camera in cameras.values()))

    def test_missing_camera_frame_fails_closed_and_trigger_id_is_never_reused(self):
        daemon, cameras, trigger = self.make_daemon(
            [envelope(2, "CAM-02", 1, 42, 8_000_000)], []
        )
        daemon.start()
        with self.assertRaisesRegex(CameraSyncFault, "frame timeout"):
            daemon.capture_batch()
        self.assertEqual(daemon.state, CameraSyncState.FAULT)
        self.assertEqual(daemon.next_trigger_id, 43)
        self.assertEqual(trigger.pulses, [42])

    def test_trigger_controller_failure_reserves_id_and_faults_session(self):
        daemon, cameras, trigger = self.make_daemon([], [], fail_pulse=True)
        daemon.start()
        with self.assertRaisesRegex(CameraSyncFault, "pulse generator fault"):
            daemon.capture_batch()
        self.assertEqual(daemon.next_trigger_id, 43)
        self.assertEqual(daemon.state, CameraSyncState.FAULT)
        self.assertTrue(all(camera.closed for camera in cameras.values()))

    def test_daemon_rejects_ptp_scheduled_profile_until_supported_by_adapter(self):
        p = profile(2, "CAM-02")
        p = CameraSyncProfile(
            camera_id=p.camera_id, device_serial=p.device_serial,
            transport=p.transport, shutter_type=p.shutter_type,
            trigger_mode="ptp_scheduled", timestamp_semantics=p.timestamp_semantics,
            clock_domain=p.clock_domain, max_exposure_skew_ns=p.max_exposure_skew_ns,
            require_hardware_trigger=True,
        )
        with self.assertRaisesRegex(ValueError, "external-line triggers"):
            CameraSyncDaemon((p,), {2: FakeCamera("CAM-02", [])}, FakeTrigger())


if __name__ == "__main__":
    unittest.main()
