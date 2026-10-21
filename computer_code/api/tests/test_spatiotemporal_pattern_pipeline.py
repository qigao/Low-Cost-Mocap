"""Synthetic shared-trigger -> temporal ID -> geometry -> rigid layout chain.

Algorithm-only conformance. No real hardware clock, exposure, or detector
tracking is simulated as though verified.
"""
import unittest

import numpy as np

from geometry import triangulate_point
from rigid_pattern import fit_rigid_pattern
from temporal_pattern import (
    FrameStamp,
    IdentifiedDetection,
    LightSample,
    TemporalCodebook,
    decode_temporal_marker,
    group_decoded_observations,
    validate_trigger_batch,
)


def project(point, intrinsic, pose):
    camera_xyz = pose["R"] @ point + pose["t"]
    pixel = intrinsic @ camera_xyz
    return pixel[:2] / pixel[2]


class SpatiotemporalPatternPipelineTests(unittest.TestCase):
    def setUp(self):
        self.model = np.array([
            [-0.10, -0.04, 0.00],
            [0.05, -0.04, 0.00],
            [-0.10, 0.055, 0.00],
            [-0.10, -0.04, 0.08],
        ])
        theta = 0.23
        c, s = np.cos(theta), np.sin(theta)
        self.R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        self.t = np.array([0.08, -0.03, 3.2])
        self.world = (self.R @ self.model.T).T + self.t
        self.poses = [
            {"R": np.eye(3), "t": np.zeros(3)},
            {"R": np.eye(3), "t": np.array([-0.8, 0.0, 0.0])},
            {"R": np.eye(3), "t": np.array([0.0, -0.6, 0.0])},
        ]
        self.K = [
            np.array([[800., 0., 320.], [0., 800., 240.], [0., 0., 1.]]),
            np.array([[925., 0., 300.], [0., 875., 230.], [0., 0., 1.]]),
            np.array([[1100., 0., 310.], [0., 1050., 250.], [0., 0., 1.]]),
        ]
        self.patterns = {
            0: (0, 0, 0, 0, 1, 1, 1, 1),
            1: (0, 0, 1, 1, 0, 0, 1, 1),
            2: (0, 1, 0, 1, 0, 1, 0, 1),
            3: (0, 1, 1, 0, 1, 0, 0, 1),
        }
        self.codebook = TemporalCodebook(self.patterns, max_bit_errors=1)
        self.epoch = 1000

    def test_three_camera_full_rig_with_shuffled_detections_and_occlusion(self):
        # Explicit common triggers and clock domain are test inputs, not
        # assertions about the actual camera SDK.
        previous = None
        for phase in range(self.codebook.period):
            trigger = self.epoch + phase
            timestamp = 10_000_000_000 + phase * 10_000_000
            frames = [
                FrameStamp(camera, trigger, timestamp + camera * 80, "rig-trigger")
                for camera in range(3)
            ]
            batch = validate_trigger_batch(
                frames, [0, 1, 2], max_exposure_skew_ns=200,
                previous_trigger_id=previous,
            )
            self.assertEqual(batch.exposure_skew_ns, 160)
            previous = batch.trigger_id

        orders = ((3, 1, 0, 2), (2, 0, 3, 1), (1, 3, 0, 2))
        detections = []
        for camera, order in enumerate(orders):
            for marker in order:
                if camera == 2 and marker == 3:
                    continue  # Entire camera view of this marker is occluded.
                samples = []
                for phase, bit in enumerate(self.patterns[marker]):
                    erased = (
                        (camera == 1 and marker == 2 and phase in (1, 6))
                        or (camera == 2 and marker == 1 and phase == 5)
                    )
                    flipped = camera == 0 and marker == 3 and phase == 0
                    illumination = None if erased else (1 - bit if flipped else bit)
                    samples.append(LightSample(self.epoch + phase, illumination))
                decoded = decode_temporal_marker(
                    samples[::-1], self.codebook, trigger_epoch=self.epoch,
                    min_valid_samples=6,
                )
                self.assertEqual(decoded.marker_id, marker)
                xy = project(self.world[marker], self.K[camera], self.poses[camera])
                detections.append(
                    IdentifiedDetection(camera, decoded.marker_id, xy)
                )

        grouped = group_decoded_observations(
            detections, camera_count=3,
            allowed_marker_ids=tuple(self.codebook.patterns),
        )
        self.assertEqual(grouped.excluded_marker_ids, ())
        self.assertEqual(tuple(grouped.observations_by_marker_id), (0, 1, 2, 3))
        self.assertIsNone(grouped.observations_by_marker_id[3][2])

        reconstructed = []
        for marker, observations in grouped.observations_by_marker_id.items():
            xyz, metrics = triangulate_point(
                observations, self.poses, self.K,
                min_views=2, max_reprojection_rms_px=1e-5,
            )
            reconstructed.append(xyz)
            self.assertLess(metrics["reprojection_rms_px"], 1e-7)
            np.testing.assert_allclose(xyz, self.world[marker], atol=1e-8)
        R, t, metrics = fit_rigid_pattern(
            self.model, np.asarray(reconstructed), marker_indices=[0, 1, 2, 3],
            max_rms_error=1e-6,
        )
        np.testing.assert_allclose(R, self.R, atol=1e-8)
        np.testing.assert_allclose(t, self.t, atol=1e-8)
        self.assertLess(metrics["rms_error"], 1e-8)

    def test_ambiguous_temporal_identity_prevents_spatial_reconstruction(self):
        ambiguous = [LightSample(self.epoch, 0)] + [
            LightSample(self.epoch + i, None) for i in range(1, 8)
        ]
        with self.assertRaisesRegex(ValueError, "ambiguous temporal"):
            decode_temporal_marker(
                ambiguous, self.codebook,
                trigger_epoch=self.epoch, min_valid_samples=1,
            )


if __name__ == "__main__":
    unittest.main()
