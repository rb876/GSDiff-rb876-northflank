# 005 — match Best-of-N to full SVDD's denoiser budget

## Decision

Match the number of evaluated denoiser states per selected output. Full SVDD is the sole
matching target; late SVDD remains a shorter-budget comparison. This supersedes the provisional
Best-of-2 setting in experiment 003. Model weights, reward checks and sampling behavior are unchanged.

## Budget

For T denoising steps, K proposals and G guided steps:

\[
C_{\mathrm{SVDD}}=T+KG,\qquad C_{\mathrm{BoN}}=NT,
\qquad N=\left\lfloor C_{\mathrm{SVDD}}/T+\tfrac12\right\rfloor.
\]

K=1 or G=0 invokes the baseline shortcut, giving a budget of T and N=1.
The count is evaluated states (`node_samples`), not batched forward calls (`node_calls`).

| Variant, in launch order | Default settings | Denoiser states per selected output |
|---|---|---:|
| Baseline | T=1000 | 1,000 |
| Best-of-N | N=11, derived from full SVDD | 11,000 |
| SVDD last 20% | K=10, G=200 | 3,000 |
| SVDD full | K=10, G=999 | 10,990 |

Best-of-11 exceeds the full-SVDD budget by 10 states, or 0.091%. N is recomputed when the full
SVDD settings change; the launcher rejects `BEST_N`. Integer rounding can produce a larger gap
for other settings, so the report records the actual difference rather than assuming a match.

## Outputs and interpretation

There are still four variants, 100 selected designs each by default, for 400 selected designs.
The fixed launch order produces late-SVDD results before full SVDD. `compute_budget.json` at the
sweep root records planned budgets. Per-run counters and `denoiser_comparison.json` in each
batch record measured states per selected output; differences within 1% are approximately matched.

This budget excludes differences in edge-model work and reward checking. Runtime remains a
separate measured quantity; equal denoiser counts do not establish equal elapsed time or total
compute. Synthetic sampler tests verify the state-count formulas across inference batch sizes
and the no-guidance shortcuts. No pretrained generation was launched for this implementation.

## Validation

The complete local suite passes **205 tests**, with three CUDA tests skipped. Sampler tests verify
actual denoiser counters across inference batch sizes; launcher tests verify automatic N and the
variant order. Reporting tests cover exact/rounded matches, incompatible runs and missing counters.
A stub execution verified the saved budget, four variant logs and overwrite protection without
loading models or generating designs.
