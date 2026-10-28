# Experiment 003 — full versus late SVDD guidance

## Question

How does guiding only the final 200 stochastic reverse steps affect overall validity,
constraint coverage and runtime compared with guiding all 999 stochastic steps?

## Planned setup

Frozen GSDiff checkpoints; the existing assumed .025 m/pixel, two-occupant, 2.5 m-ceiling scenario.
GPU rewards with final CPU reference evaluation. One paired seed batch starting at 10000, with
100 selected designs per variant, gives **100 per variant and 400 total**.

| Variant directory | Settings |
|---|---|
| `baseline` | Unconstrained generation |
| `best_of_n` | N=2, provisional |
| `svdd_pm_full` | K=10, temperature 0, `GUIDED_STEPS=999` |
| `svdd_pm_last20` | K=10, temperature 0, `LATE_GUIDED_STEPS=200` |

Both SVDD variants run `--method svdd_pm`. Full guidance covers `t=999,...,1`; late guidance
covers `t=200,...,1`, approximately the final 20% of stochastic steps. Every trajectory runs
all 1,000 denoising steps, including unguided deterministic `t=0`.
Each variant has separate logs, configs, candidate files and reports in each seed batch;
the batch comparison table includes both guidance windows.

Preview without generation:

```bash
bash test_time/launch_eval.sh --dry-run
```

## Status and interpretation

Pending; no evaluation has been launched and no results are claimed. Compare final CPU-audited
overall validity, per-check pass/unknown rates, structural validity and measured runtime.
These variants have different compute costs. Best-of-2 is provisional; any matched-compute
claim needs runtime calibration. Cross-batch aggregation and confidence intervals remain pending.
