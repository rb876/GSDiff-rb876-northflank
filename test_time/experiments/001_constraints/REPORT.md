# Experiment 001 — implementation ready, experiment design pending

## Question

Can inference-time selection improve the selected feasibility checks while retaining valid graphs?

## Setup

Frozen unconstrained RPLAN checkpoint, 1,000 DDPM steps, H100.
Scenario: two occupants, 2.5 m ceiling height, **assumed** 0.025 m per drawing pixel.
Six scored IDs: C033, C035, C036, C076, C077, C078. C034 is an auxiliary bed-footprint check.
Available methods: baseline, Best-of-N, and SVDD-PM. CLI defaults are one trial;
N=4 and K=4 with ten late stochastic steps are smoke-test settings, not an agreed experiment budget.

## Status

Unit tests cover geometry, missing observations, the C076 logical OR, occupancy thresholds,
joint bed placement, coverage protection, sampler equivalence, RNG handling and reporting.
One real H100 smoke test completed for each method and produced graphs, annotated PNG/SVG and reports.
The baseline adapter reproduces the saved original seed-42 raw nodes and edges exactly.
Frozen-parameter checks passed. All 48 tests passed on Kraken and Northflank.
Final CLI checks used one trial each: baseline, Best-of-2, and SVDD-PM with K=2,
two guided steps and temperature 0.1. Earlier smoke checks also exercised Best-of-4
and greedy SVDD-PM with K=4 and ten guided steps.
Full smoke outputs are stored in `test_time/outputs/code_verification/` and `test_time/outputs/stage_smoke/`.

[Verification record](verification.json) · [Annotated baseline smoke output](figures/baseline_smoke.png).

These checks establish that the code runs; they are not evidence of general improvements.
No eight-seed comparison or larger experiment has been launched. We will choose the seed set,
budgets, ablations and evaluation questions together before running experiments.

## Limits

Bedroom/bathroom checks are geometric feasibility proxies. A missed finite bed-placement search
is unknown. RPLAN-relative distribution metrics are unavailable without dataset access.
Validity, measurement coverage and runtime must be reported alongside measurable compliance.
