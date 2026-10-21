"""Synthetic raw-frame -> local tracks -> time-code ID -> fixed rigid pose.

The static/motion-bounded scene has perfect shared trigger metadata and an
explicitly valid predicted ROI. Real hardware exposure timing, camera motion,
shutter integration and tracking uncertainty remain independent acceptance.
"""
import unittest

import numpy as np

from geometry import triangulate_point
from light_intensity import (
    BinaryIntensityCalibration, sample_tracked_light_roi, quantize_roi_track,
)
from light_tracking import BoundedLightTracker
from rigid_pattern import fit_rigid_pattern
from temporal_pattern import (
    FrameStamp, IdentifiedDetection, TemporalCodebook,
    decode_temporal_marker, group_decoded_observations,
    validate_trigger_batch,
)


def project(xyz, K, pose):
    camera_xyz = pose["R"] @ xyz + pose["t"]
    pixel_h = K @ camera_xyz
    return pixel_h[:2] / pixel_h[2]


class RealisticSyntheticLightPipelineTests(unittest.TestCase):
    def setUp(self):
        self.model = np.array([
            [-0.12, -0.10, 0.00],
            [0.12, -0.10, 0.00],
            [-0.12, 0.12, 0.00],
            [0.12, 0.12, 0.12],
        ])
        theta = 0.18
        c, s = np.cos(theta), np.sin(theta)
        self.rotation = np.array([
            [c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0],
        ])
        self.translation = np.array([0.01, 0.0, 1.0])
        self.world = (self.rotation @ self.model.T).T + self.translation
        self.K = [
            np.array([[100., 0., 32.], [0., 100., 32.], [0., 0., 1.]]),
            np.array([[100., 0., 32.], [0., 100., 32.], [0., 0., 1.]]),
        ]
        self.poses = [
            {"R": np.eye(3), "t": np.zeros(3)},
            {"R": np.eye(3), "t": np.array([-0.1, 0.0, 0.0])},
        ]
        self.patterns = {
            0: (0, 0, 0, 0, 1, 1, 1, 1),
            1: (0, 0, 1, 1, 0, 0, 1, 1),
            2: (0, 1, 0, 1, 0, 1, 0, 1),
            3: (0, 1, 1, 0, 1, 0, 0, 1),
        }
        self.codes = TemporalCodebook(self.patterns, max_bit_errors=1)

    def test_two_cameras_recover_rigid_pose_despite_led_off_frames(self):
        trackers = [
            BoundedLightTracker(
                camera_id=i, max_tracks=4, max_motion_px_per_trigger=5.0,
                max_missing_triggers=2,
            )
            for i in range(2)
        ]
        calibration = BinaryIntensityCalibration(0.0, 120.0)
        measured = [dict(), dict()]
        latest = [dict(), dict()]
        previous = None
        epoch = 1000
        for phase in range(8):
            trigger = epoch + phase
            frame_t = 10_000_000_000 + phase * 10_000_000
            stamps = [
                FrameStamp(i, trigger, frame_t + i * 50, "shared-trigger")
                for i in range(2)
            ]
            batch = validate_trigger_batch(
                stamps, [0, 1], max_exposure_skew_ns=80,
                previous_trigger_id=previous,
            )
            previous = batch.trigger_id
            for camera, tracker in enumerate(trackers):
                image = np.full((64, 64), 12.0 + (phase % 3) * 5.0)
                visible = []
                for marker in range(4):
                    if not self.patterns[marker][phase]:
                        continue
                    point = project(self.world[marker], self.K[camera],
                                    self.poses[camera])
                    x, y = np.rint(point).astype(int)
                    image[y-2:y+3, x-2:x+3] += 120.0
                    visible.append(point)
                # Intentionally reverse the contour iteration order.
                if phase % 2:
                    visible.reverse()
                evidence = tracker.update(trigger, visible)
                for entry in evidence:
                    # In this controlled static synthetic scene, predicted
                    # positions remain valid during off phases. Real moving
                    # scenes require an independent motion quality gate:
                    # a missing detection cannot automatically mean OFF.
                    if entry.observed_xy is None:
                        self.assertIsNone(sample_tracked_light_roi(image, entry))
                    roi = sample_tracked_light_roi(
                        image, entry, predicted_roi_qualified=True
                    )
                    measured[camera].setdefault(entry.track_id, []).append(
                        (trigger, roi)
                    )
                    latest[camera][entry.track_id] = entry

        detections = []
        for camera, tracker in enumerate(trackers):
            self.assertEqual(len(tracker.active_track_ids), 4)
            decoded_ids = set()
            for local_track_id, raw_samples in measured[camera].items():
                lights = quantize_roi_track(raw_samples, calibration)
                decoded = decode_temporal_marker(
                    lights, self.codes, trigger_epoch=epoch,
                    min_valid_samples=4,
                )
                self.assertNotIn(decoded.marker_id, decoded_ids)
                decoded_ids.add(decoded.marker_id)
                last = latest[camera][local_track_id]
                self.assertIsNotNone(last.observed_xy)
                detections.append(IdentifiedDetection(
                    camera, decoded.marker_id, last.observed_xy,
                ))
            self.assertEqual(decoded_ids, set(range(4)))

        grouped = group_decoded_observations(
            detections, camera_count=2,
            allowed_marker_ids=tuple(self.codes.patterns),
        )
        self.assertEqual(grouped.excluded_marker_ids, ())
        reconstructed = []
        for marker, observations in grouped.observations_by_marker_id.items():
            xyz, metrics = triangulate_point(
                observations, self.poses, self.K,
                min_views=2, max_reprojection_rms_px=1e-6,
            )
            np.testing.assert_allclose(xyz, self.world[marker], atol=1e-8)
            self.assertLess(metrics["reprojection_rms_px"], 1e-7)
            reconstructed.append(xyz)
        R, t, metrics = fit_rigid_pattern(
            self.model, reconstructed, marker_indices=[0, 1, 2, 3],
            max_rms_error=1e-7,
        )
        np.testing.assert_allclose(R, self.rotation, atol=1e-7)
        np.testing.assert_allclose(t, self.translation, atol=1e-7)
        self.assertLess(metrics["rms_error"], 1e-7)


if __name__ == "__main__":
    unittest.main()
