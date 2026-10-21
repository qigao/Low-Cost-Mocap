---
name: "[HW-06] Qualify end-to-end pose latency and fail-closed flight-control interface"
about: "Bench validation of tracking-to-control latency, freshness and safety before flight"
title: "[HW-06] Qualify end-to-end pose latency and fail-closed flight-control interface"
labels: []
assignees: []
---

## Context

Merged PR #1 intentionally does not switch the new experimental temporal/synchronization pipeline into live flight control. Runtime behavior must be measured on an instrumented, non-flying bench before any production switch.

## Test plan

- [ ] Define maximum acceptable pose age, timing jitter and error policy before recording results.
- [ ] Trace actual sensor exposure -> frame acquisition -> detection -> timecode window -> marker admission -> 3D/rigid pose -> serial command generation.
- [ ] Measure p50/p95/p99/max latency, sustained throughput, deadline misses, dropped frames and quality-gate rejection-to-actuation delays.
- [ ] Reject stale triggers, cross-frame measurements, missing camera sync, ambiguous marker IDs, partial rigid fits and inconsistent physical scale at the control boundary.
- [ ] Test multi-drone contention, disconnect/cancel/restart, calibration invalidation and safe disarm behavior with motor outputs disabled.
- [ ] Gate any flight test on completion of HW-01 through HW-05 and an independently reviewed safe test plan.

## Done only when

- [ ] Independent instrumented traces and explicit safety/latency acceptance criteria are attached.
- [ ] Bench fail-closed behavior is repeatable and documented; no bad or stale pose can reach armed-control output.
- [ ] A separate explicit approval authorizes any real flight-control implementation switch.

## Boundaries

Do not treat software conformance, local loopback or synthetic trigger metadata as evidence of real-time safety.
