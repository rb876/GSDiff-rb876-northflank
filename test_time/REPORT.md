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
| 001 — constraints | Implementation verified; three-trial pilot inspected | [Report](experiments/001_constraints/REPORT.md) |
| 002 — GPU backend | Tensor geometry and guidance; offline implementation audit | [Report](experiments/002_gpu_backend/REPORT.md) |

## Measurement policy

Six scored constraints: C033, C035, C036, C076, C077, C078; C034 is auxiliary.
The user-approved scale is **assumed** 0.025 m/drawing pixel, with two occupants and 2.5 m ceilings.
These are conditional geometric/scenario checks, not verified real-world compliance.
The headline overall-valid rate requires structural validity **and all applicable scored checks to pass**.
Unknown measurements never count as passes; unresolved samples without a known failure remain unknown.
No applicable scored checks means not applicable, not success. Every selected design stays in the denominator.
Structural validity, per-check compliance and measurement coverage remain separate diagnostics.

## Method

Best-of-N ranks complete independent samples. SVDD-PM compares stochastic reverse proposals
using complete graphs reconstructed from predicted clean nodes, then continues the selected noisy state.
Both use the selected guidance backend. Final metrics share the CPU reference evaluator.
Selection prioritizes structural validity and measurement coverage before energy.
The new SVDD defaults are K=10, all 999 stochastic reverse steps, and temperature 0,
following the [paper's full-process, greedy configuration](https://arxiv.org/html/2408.08252v3).
These defaults have not yet been evaluated. The launcher's Best-of-2 setting remains provisional
and needs runtime calibration before a compute-matched comparison.

Generation now batches up to four independent trials, with at most ten states per neural-network call.
SVDD proposals and Best-of-N candidates use these batched calls; each design retains its own random
stream and selection. The default `--reward_backend gpu` also performs room extraction, constraint
searches and selection using tensors; `--reward_backend cpu` opts into reference CPU guidance. All six checks
are retained, with finite-search limitations recorded. Final exported designs always receive the CPU
audit. Reports count batch time once and separate forward calls from processed states.
H100 throughput and memory for the new backend have not yet been benchmarked.
The launcher also defaults to `REWARD_BACKEND=gpu`; set it to `cpu` for the reference comparison.

## Current status

The historical CPU-guided pilot has three trials per method (seeds 1000–1002): all three methods have **0/3 overall-valid**
designs. All nine selected designs fail C035. Structural validity is 2/3 for baseline and 3/3 for both guided methods.
This small pilot does not establish an improvement; larger evaluation design remains pending.
The original 48-test suite passed locally and on Northflank; the batching update passed 92 local tests.
The current suite passes **146 tests** locally, with two CUDA checks skipped because CUDA is unavailable.
The GPU backend's offline audit matches room geometry and structural validity on all 12 saved pilot
candidates, and every constraint status on all 10 structurally valid candidates. Invalid graphs skip
GPU searches. This is a small implementation check, not evidence of improved generation or speed.
No new pretrained generation was launched for this update. Pilot reports and figures retain the saved earlier results.
Missing local checkpoint files produce setup instructions.

Implementation and exact run commands: [README](README.md).
