# Experiment 001 — implementation and initial pilot

## Question

Can inference-time selection improve the selected feasibility checks while retaining valid graphs?

## Setup

Frozen unconstrained RPLAN checkpoint, 1,000 DDPM steps, H100.
Scenario: two occupants, 2.5 m ceiling height, **assumed** 0.025 m per drawing pixel.
Six scored IDs: C033, C035, C036, C076, C077, C078. C034 is an auxiliary bed-footprint check.
Available methods: baseline, Best-of-N, and SVDD-PM. Current CLI defaults are one trial,
N=4, and SVDD-PM with K=10, 999 guided stochastic steps and temperature 0.
The historical smoke tests and pilot below retain their original settings.

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

## Initial pilot

The user ran three paired trials (seeds 1000–1002) per method. Overall validity requires a valid
structure and all applicable scored checks to measurably pass; unknowns cannot establish validity.
C034 is auxiliary and excluded. These metrics cover the six implemented scored checks.

| Method | Overall valid | Structurally valid | Measurable checks passing | Unknown checks |
|---|---:|---:|---:|---:|
| Baseline | 0/3 | 2/3 | 15/18 (83.3%) | 0 |
| Best-of-2 | 0/3 | 3/3 | 14/18 (77.8%) | 0 |
| SVDD-PM, K=4, ten guided steps, temperature=0 | 0/3 | 3/3 | 13/17 (76.5%) | 1 |

All nine selected designs fail C035 (bedroom circulation). High per-check compliance therefore
does not imply an overall-valid design. Best-of-2 and SVDD take about 21.1 and 16.4 seconds per trial,
respectively, so this pilot does not satisfy our 20% compute-matching rule.
Three trials are insufficient to conclude that either method improves constraint satisfaction.

Full outputs: `test_time/outputs/pilot_check/batch_1_seed_1000/` (ignored by Git).
Reports and figure labels were updated offline from saved evaluations; no new generation was needed.
All 52 tests pass locally after this reporting update.
Larger seed sets, compute calibration, diversity metrics and confidence intervals remain to be agreed.

## Next configuration — not yet evaluated

The user selected K=10, guidance at every stochastic reverse step (`t=999,...,1`),
and greedy selection (temperature 0), following [SVDD §§4.2 and 6.1](https://arxiv.org/html/2408.08252v3).
The deterministic final step is not duplicated. No result above measures this new configuration.
The launcher still uses Best-of-2; it must be recalibrated against full-guidance runtime before
claiming a compute-matched comparison.

The runner now supports batched trials and proposal inference: defaults are `batch_size=4` and
`inference_batch_size=10`. Every design keeps its own seed and candidate selection; CPU geometry
checks remain per graph. These settings have no measured H100 results yet. Timing reports use
amortized seconds per design and record both forward calls and the number of states processed.
All 92 local tests pass after batching, including independent seeds, proposal selection, tail batches,
adapter RNG handling and exported timing totals. These tests use tiny CPU networks, not pretrained generation.

```bash
python -m test_time.run --method svdd_pm --k 10 --guided_steps 999 --temperature 0 \
  --batch_size 4 --inference_batch_size 10
```

## Limits

Bedroom/bathroom checks are geometric feasibility proxies. A missed finite bed-placement search
is unknown. RPLAN-relative distribution metrics are unavailable without dataset access.
Validity, measurement coverage and runtime must be reported alongside measurable compliance.
