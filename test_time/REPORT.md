# GSDiff: test-time constraint guidance

The [concise technical report](../docs/technical_report.md) presents the model, constraints,
reward and SVDD comparison, with results pending.
[Technical notes](../docs/technical_notes.md) retain implementation details and assessment alignment.

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
| 003 — guidance window | Pending: full SVDD guidance versus the final 200 stochastic steps | [Report](experiments/003_guidance_window/REPORT.md) |
| 004 — feasible constraints | Six GPU-compatible checks with three supporting measurement rules | [Report](experiments/004_feasible_constraints/REPORT.md) |
| 005 — denoiser budget | Match Best-of-N to full SVDD by evaluated denoiser states | [Report](experiments/005_denoiser_budget/REPORT.md) |
| 006 — sweep analysis | Offline method comparisons, constraint bottlenecks, diversity and sample gallery | [Report](experiments/006_sweep_analysis/REPORT.md) |

## Measurement policy

Six scored constraints: C033, C035, C036, C076, C077, C078. The v2 registry contains those six
checks and three supporting measurement rules: C034, C079 and C080. All nine entries are implemented.
C079 declares the zero-thickness area proxy; C080 centralizes semantic and below-1.80 m exclusions.
These supporting rules define measurements and add no separate reward or pass/fail result. The v1 configuration remains available.
The user-approved scale is **assumed** 0.025 m/drawing pixel, with two occupants and 2.5 m ceilings.
These are conditional geometric/scenario checks, not verified real-world compliance.
The headline overall-valid rate requires structural validity **and all applicable scored checks to pass**.
Unknown measurements never count as passes; unresolved samples without a known failure remain unknown.
No applicable scored checks means not applicable, not success. Every selected design stays in the denominator.
Structural validity, per-check compliance and measurement coverage remain separate diagnostics.
The offline analysis reports passes/applicable checks as constraint success and retains
measurable-only compliance separately. It also reports per-constraint sole failures, recorded
failure reasons, overall-validity intervals and paired trial outcomes. Diversity and distribution
diagnostics show their eligible geometry counts; seed batches are not pooled.

## Method

Best-of-N ranks complete independent samples. SVDD-PM compares stochastic reverse proposals
using complete graphs reconstructed from predicted clean nodes, then continues the selected noisy state.
Both use the selected guidance backend. Final metrics share the CPU reference evaluator.
Selection prioritizes structural validity and measurement coverage before energy.
The new SVDD defaults are K=10, all 999 stochastic reverse steps, and temperature 0,
following the [paper's full-process, greedy configuration](https://arxiv.org/html/2408.08252v3).
The launcher now matches Best-of-N to full SVDD by evaluated denoiser states per selected output.
Full SVDD evaluates 1000 + 10 × 999 = 10,990 states; Best-of-11 evaluates 11,000, a 0.091%
difference. N is derived from the configured full-guidance budget and rounded to the nearest
integer, with ties upward. K=1 or no guidance takes the baseline shortcut and gives N=1.
This replaces the earlier provisional Best-of-2 plan; it does not equalize runtime, edge-model
calls or reward evaluation costs.

Generation now batches up to four independent trials, with at most ten states per neural-network call.
SVDD proposals and Best-of-N candidates use these batched calls; each design retains its own random
stream and selection. The default `--reward_backend gpu` also performs room extraction, constraint
searches and selection using tensors; `--reward_backend cpu` opts into reference CPU guidance. All six checks
are retained, with finite-search limitations recorded. Final exported designs always receive the CPU
audit. Reports count batch time once and separate forward calls from processed states.
H100 throughput and memory for the new backend have not yet been benchmarked.
The launcher also defaults to `REWARD_BACKEND=gpu`; set it to `cpu` for the reference comparison.
The planned evaluation uses one seed batch starting at 10000, with 100 selected designs for each of four variants:
baseline, Best-of-11, SVDD-PM guided only at `t=200,...,1` (approximately the final 20%), then full SVDD-PM.
This gives **100 selected designs per variant, 400 total**, using paired design seeds.
All trajectories denoise for 1,000 steps; both SVDD variants use K=10 and temperature 0 and leave
deterministic `t=0` unguided. Late SVDD remains an unmatched shorter-budget comparison, with
3,000 denoiser states per output. This order is fixed across seed batches.
`compute_budget.json` saves the planned full-SVDD match. Reports use actual `node_samples` per
selected trial and a 1% tolerance for approximate denoiser matches; runtime comparisons remain
separate. The launcher rejects old `BEST_N` overrides so the chosen budget cannot silently change.

## Current status

The historical CPU-guided pilot has three trials per method (seeds 1000–1002): all three methods have **0/3 overall-valid**
designs. All nine selected designs fail C035. Structural validity is 2/3 for baseline and 3/3 for both guided methods.
This small pilot does not establish an improvement; larger evaluation design remains pending.
The original 48-test suite passed locally and on Northflank; the batching update passed 92 local tests.
The prior constraint update passed **186 tests** locally, with three CUDA checks skipped because CUDA is unavailable.
The denoiser-budget update passed **205 tests**, with the same three CUDA checks skipped locally.
The GPU backend's offline audit matches room geometry and structural validity on all 12 saved pilot
candidates, and every constraint status on all 10 structurally valid candidates. Invalid graphs skip
GPU searches. This is a small implementation check, not evidence of improved generation or speed.
No new pretrained generation was launched for this update. Pilot reports and figures retain the saved earlier results.
Missing local checkpoint files produce setup instructions.

The sweep-analysis workflow is prepared and uses saved results only. It produces six PNG/PDF
figures, CSV/JSON tables, a concise Markdown report and a gallery of every selected design.
The current suite passes **221 tests**, with three CUDA tests skipped. Offline verification
rendered nine selected designs from the three-method historical pilot; no full-sweep conclusion is claimed.
Command and metric definitions: [analysis guide](../docs/ANALYSIS.md).

Implementation and exact run commands: [README](README.md).
