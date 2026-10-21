---
name: "[HW-02] Qualify physical multi-camera calibration and metric scale"
about: "Independent real-rig intrinsic/extrinsic accuracy and measured distance scale"
title: "[HW-02] Qualify physical multi-camera calibration and metric scale"
labels: []
assignees: []
---

## Context

Merged PR #1 fixes relative pose composition, per-camera intrinsics, sparse camera indexing, triangulation and gauge-fixed bundle adjustment. Essential-matrix translation alone has no measured metric scale.

## Verification tasks

- [ ] Calibrate actual camera intrinsics and lens distortion for the deployed resolutions, gains and mechanical mounting.
- [ ] Measure at least one independent physical reference baseline/pattern dimension; establish metric scale with traceable units.
- [ ] Validate world-to-camera R/t conventions and consistent undistorted pixel coordinates on every camera, including noncontiguous camera IDs.
- [ ] Use an independently measured calibration target and repeat the full setup across multiple positions, depths and baselines.
- [ ] Report per-camera reprojection distributions, reconstructed 3D errors, relative pose repeatability and sensitivity to calibration perturbations.
- [ ] Include degenerate views, weak parallax and partial occlusion; verify the geometry pipeline refuses invalid outputs.

## Done only when

- [ ] Ground-truth fixtures, procedure, calibration version, raw frames and error reports are attached.
- [ ] Configured reprojection/metric-scale limits were decided before acceptance and are met repeatedly.
- [ ] The physical scale-validity status is supported by measurements, not the initial essential-matrix unit baseline.

## Boundaries

Hardware gate only; merged software CI does not imply actual camera calibration is accepted.
