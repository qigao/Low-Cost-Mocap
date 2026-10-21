---
name: "[HW-01] Verify real sensor-exposure synchronization and trigger provenance"
about: "Hardware validation of camera exposure/trigger IDs and common clock timing"
title: "[HW-01] Verify real sensor-exposure synchronization and trigger provenance"
labels: []
assignees: []
---

## Why this needs physical validation

Merged PR #1 provides strict shared-trigger software contracts, but the currently inspected upstream pseyepy/PS3Eye path timestamps frame *dequeue* with gettimeofday rather than sensor exposure. Software timestamps are not a substitute for independently verified hardware synchronization.

## Investigation / implementation

- [ ] Identify deployed PS3Eye firmware, driver, and pseyepy fork/revision and document the actual frame metadata source.
- [ ] Determine whether real exposure-associated trigger IDs exist. If they do not, propose and validate an alternative externally measurable hardware synchronization mechanism before trusting FrameStamp.
- [ ] Establish a physically common clock domain and a trigger epoch shared with the LED controller; never synthesize IDs from Python polling or frame arrival order.
- [ ] Record raw camera frames with independent reference trigger events, including missing, duplicate, out-of-order and stale frames.
- [ ] Measure inter-camera exposure skew, jitter, dropped triggers, duplicated frames and their distributions (p50/p95/p99/max); agree on limits *before* declaring acceptance.

## Done only when

- [ ] Independent test equipment or another trusted hardware reference substantiates the recorded exposure/trigger mapping.
- [ ] A reproducible raw recording + versioned metadata + SHA-256 and a full admission/rejection report are attached.
- [ ] Every violated synchronization gate fails closed; no host dequeue timestamp fallback is introduced.
- [ ] Explicit test evidence and predeclared acceptance limits are linked here.

## Boundaries

Follow-up to merged PR #1. No flight-control activation or production hardware-sync claims from unit tests alone.
