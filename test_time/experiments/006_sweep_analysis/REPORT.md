# 006 — prepare offline sweep analysis

## Question

How do baseline, Best-of-N, last-20% SVDD and full SVDD differ in constraint success, structural
validity, diversity and inference cost? Which constraints account for the remaining failures?

## Analysis

```bash
python -m pip install -r test_time/requirements-analysis.txt
python -m test_time.analyze --experiment_dir test_time/outputs/sweep_001
```

The command reads saved final CPU evaluations and selected vector graphs. It does not load
model weights, generate designs or repeat geometry checks. The default output is
`test_time/outputs/sweep_001/analysis/`; the original run files remain unchanged.

Each completed variant/seed batch is a separate population. Best-of-N contributes one selected
output per trial. Running, failed and incomplete runs are excluded and recorded; there is no
cross-batch pooling.

## Evidence produced

| Evidence | Purpose |
|---|---|
| Overall validity with 95% Wilson intervals and structural-valid rate | Compare complete design success and sampling uncertainty. |
| Per-constraint pass/fail/unknown/N/A bars and sole-failure counts | Find bottlenecks such as C035 without assuming which check will dominate. |
| Histograms of passed-check counts and measured severity | Show partial success and violation magnitude. |
| Recorded failure/unknown explanations | Connect aggregate failures with the actual checker decisions. |
| Graph uniqueness, semantic-layout distance and near-duplicate fraction | Describe diversity, conditional on usable structurally valid geometry. |
| Room-count/area distributions, room-type JS divergence and feature Wasserstein distances | Describe changes from a compatible generated baseline. |
| Runtime and denoiser evaluations per output, with baseline multipliers | Keep elapsed cost separate from the selected denoiser budget. |
| Paired validity outcomes | Compare the same trial seeds, using the trial seed rather than a Best-of-N winner's seed. |
| Complete sample gallery | Inspect every selected design with PNG/SVG, graph JSON, exact evaluation and failure explanations. |

The main constraint-success rate divides passes by applicable scored checks. Unknowns remain
in this denominator. Measurable-only compliance and measurement coverage are separate columns.
Supporting definitions do not add successes. Severity statistics omit unknown values rather
than setting them to zero. Overall success still requires all applicable checks and structure
to pass.

## Assessment and limitations

The assessment requests comparison metrics, compute multipliers, rendered vector outputs,
annotated successes/failures, an ablation and limitations (PDF pp. 2–3, §§2.2 and 2.6–2.8).
Full versus last-20% SVDD supplies the guidance-window ablation. Only full SVDD is the intended
Best-of-N denoiser-budget match; their runtime and reward/edge-model costs can differ.

Diversity uses graph canonicalization and a fixed 64×64 semantic raster of the 512-pixel canvas.
Near duplicates have another layout within `1 − IoU ≤ 0.05`. Geometry/raster eligibility counts
are reported, because methods can differ in structural validity. Translation, rotation and
scale are not aligned. Large layout distance alone does not establish quality.

Baseline-relative distribution diagnostics are not RPLAN fidelity; no FID/KID is reported.
Intervals describe individual validity rates, not a significance test of method differences.
The assumed metric scale, occupants, height and finite-search limitations remain in force.

## Status

The full local test suite passes **221 tests**, with three CUDA tests skipped. Offline analysis
of the historical pilot produced all six PNG/PDF figures and a gallery of nine selected designs
across three methods in `test_time/outputs/analysis_preview/`. The figures were visually inspected.
These are the older pilot settings, not the new full sweep. Full-sweep results have not been
inspected locally, and this report does not claim that guidance improves validity or diversity.
No pretrained generation was launched for this analysis work.
Exact commands, denominators and artifact locations are documented in
[the analysis guide](../../../docs/ANALYSIS.md).
