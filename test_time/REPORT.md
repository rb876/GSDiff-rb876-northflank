# GSDiff: test-time constraint guidance

## Goal

Compare pretrained generation, Best-of-N, and SVDD-PM using the same frozen models,
scenario and constraint evaluator. No training or fine-tuning.

## Organization

Code, configs, tests and reports live in `test_time/`.
Each experiment has a short versioned report under `experiments/`; full generated runs live in ignored `outputs/`.
These Markdown files are ready to paste into HackMD.

## Experiments

| Experiment | Question | Report |
|---|---|---|
| 000 — baseline | Can we reproduce released unconstrained inference? | [Report](experiments/000_baseline/REPORT.md) |
| 001 — constraints | Implementation verified; comparison design pending | [Report](experiments/001_constraints/REPORT.md) |

## Measurement policy

Six scored constraints: C033, C035, C036, C076, C077, C078; C034 is auxiliary.
The user-approved scale is **assumed** 0.025 m/drawing pixel, with two occupants and 2.5 m ceilings.
These are conditional geometric/scenario checks, not verified real-world compliance.
Unknown measurements never count as passes. Structural validity and coverage accompany every score.

## Method

Best-of-N ranks complete independent samples. SVDD-PM compares stochastic reverse proposals
using complete graphs reconstructed from predicted clean nodes, then continues the selected noisy state.
Both share one evaluator. Selection prioritizes validity and measurement coverage before energy.

## Current status

Implementation is ready for experiment planning. Only unit tests and one-sample GPU smoke checks
have been run; no eight-seed comparison or larger experiment has been launched.
All 48 tests passed both locally and in the Northflank environment.
Missing local checkpoint files produce setup instructions.

Implementation and exact run commands: [README](README.md).
