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
Both share one evaluator. Selection prioritizes structural validity and measurement coverage before energy.
The new SVDD defaults are K=10, all 999 stochastic reverse steps, and temperature 0,
following the [paper's full-process, greedy configuration](https://arxiv.org/html/2408.08252v3).
These defaults have not yet been evaluated. The launcher's Best-of-2 setting remains provisional
and needs runtime calibration before a compute-matched comparison.

Generation now batches up to four independent trials, with at most ten states per neural-network call.
SVDD proposals and Best-of-N candidates use these batched calls; each design retains its own random
stream and selection. Geometry checks remain on the CPU. Reports count batch time once and separate
forward calls from processed states. H100 throughput and memory have not yet been benchmarked.

## Current status

The user-run pilot has three trials per method (seeds 1000–1002): all three methods have **0/3 overall-valid**
designs. All nine selected designs fail C035. Structural validity is 2/3 for baseline and 3/3 for both guided methods.
This small pilot does not establish an improvement; larger evaluation design remains pending.
The original 48-test suite passed locally and on Northflank. After the batching update, all **92 tests**
pass locally, including serial/batched equivalence with tiny CPU networks and CLI output accounting.
No pretrained batched generation has been run. Pilot reports and figures retain the saved earlier results.
Missing local checkpoint files produce setup instructions.

Implementation and exact run commands: [README](README.md).
