---
name: "[HW-04] Measure local LED tracking and motion/occlusion ID-switch safety"
about: "Real-motion tracking of blinking rigid LED layouts with fail-closed identity decisions"
title: "[HW-04] Measure local LED tracking and motion/occlusion ID-switch safety"
labels: []
assignees: []
---

## Context

Merged PR #1 has bounded camera-local tracks and enforces independent qualification before interpreting predicted-only dark ROIs. Moving lights can still swap identity or move outside a predicted ROI during the OFF phase.

## Verification tasks

- [ ] Collect independently labeled trajectories containing drone motion, crossings, accelerations, rotations, LED dark phases and temporary camera occlusion.
- [ ] Measure camera-local track ID switches, fragmentation, recovery latency, birth/death behavior and assignment ambiguity rejection.
- [ ] Derive a tested motion-prediction uncertainty gate for sampling predicted dark ROIs, including camera motion and scene-dependent background artifacts.
- [ ] Validate simultaneous drones/overlapping lights, motion blur, false bright points and distorted detections across cameras.
- [ ] Verify all track and temporal-ID joins preserve the originating camera, trigger, clock domain, local track and cycle; no borrowed samples.
- [ ] Report ID-switch/false-accept/rejection rates and confidence limits against independent physical labels.

## Done only when

- [ ] Repeatable real-motion recordings with exact versioned labels and hardware timing are available.
- [ ] Safety-critical ambiguous associations abstain without emitting a wrong physical marker ID.
- [ ] Predeclared real-motion acceptance limits and their measurements are documented.

## Boundaries

A static synthetic ROI test is not proof of a moving drone's dark-phase location.
