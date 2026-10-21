---
name: "[HW-05] Build independent labeled hardware replay dataset and acceptance report"
about: "Reproducible captures, independent reference timestamps/labels, confusion and statistics"
title: "[HW-05] Build independent labeled hardware replay dataset and acceptance report"
labels: []
assignees: []
---

## Goal

Provide reproducible, externally grounded real-rig evidence instead of treating passing synthetic unit tests as physical acceptance.

## Dataset and replay tasks

- [ ] Record original frames plus per-camera timing metadata, device configuration, LED codebook, calibration version and trigger references.
- [ ] Obtain independent reference exposure/trigger events and independent physical marker-ID labels; do not derive ground truth from the tested decoder.
- [ ] Include representative normal and stress sessions across depths, lighting, gains, movement, occlusion, synchronization errors and dropped frames.
- [ ] Publish versioned data manifests, raw-file SHA-256 checksums, labeling protocol and known source-clock limitations.
- [ ] Run computer_code/api/offline_acceptance_cli.py on the immutable recordings, retaining the input SHA-256 and full report JSON.
- [ ] Summarize correct/wrong/rejected IDs separately, confidence intervals, trigger gaps, skew distributions and any correlated-trial sampling limitations.

## Done only when

- [ ] Source provenance, independently checked labels and explicit operating conditions are documented.
- [ ] Statistical limits and sample-size assumptions are defined in advance, and failing cases are not removed post hoc.
- [ ] The report states hardware provenance truthfully; the software report's default false flag is not silently flipped without evidence.

## Boundaries

Issue is for real hardware recordings; synthetic fixtures remain useful only as software regression evidence.
