---
name: "[HW-03] Validate physical LED temporal codes, exposure aliasing and code epoch"
about: "LED controller, temporal/frequency code robustness and independent optical verification"
title: "[HW-03] Validate physical LED temporal codes, exposure aliasing and code epoch"
labels: []
assignees: []
---

## Context

The software decoder uses phase-anchored binary codes with bounded Hamming correction, erasures and ambiguity rejection. Its synthetic validation does not prove camera/LED phase or optical signal quality.

## Verification tasks

- [ ] Instrument the actual LED controller; independently measure code epoch, phase, toggling latency and intensity per phase.
- [ ] Characterize frame rate, exposure integration, rolling/global shutter behavior and aliasing under all deployed camera configurations.
- [ ] Verify distinct spatial LED layouts plus time/frequency patterns across all drone identities; reject pattern collisions rather than choosing an arbitrary ID.
- [ ] Calibrate local background-relative ON/OFF contrast per camera/gain and verify the uncertain/erasure region under changing light.
- [ ] Test LED crosstalk, reflection, occlusion, overexposure, low signal, timing slips and adjacent codes with external labeled traces.
- [ ] Record false accepts, correct accepts and abstentions separately; preserve raw frames and labels for offline_capture_acceptance.

## Done only when

- [ ] Independent timing/optical evidence proves epoch and signal alignment rather than inferred phase shifting.
- [ ] A documented codebook/noise budget is supported by measured confusion statistics and confidence intervals.
- [ ] Ambiguous/corrupted patterns never silently become another marker identity within the declared operating envelope.

## Boundaries

Do not use pseyepy host dequeue timestamps as exposure-trigger provenance.
