# 004 — six evaluated constraints and three measurement rules

## Scope

Keep the six implemented checks and three supporting measurement rules in the current registry.
These use the generated geometry and the existing experimental assumptions. GPU guidance remains the default;
final metrics retain the CPU reference audit. No new pretrained experiment is launched here.

| Decision | IDs |
|---|---|
| Retain six scored checks | C033, C035, C036, C076, C077, C078 |
| Retain bed-size supporting definition | C034 |
| Add area supporting definitions | C079, C080 |

All nine registry entries are implemented and enabled. None of the original six scored checks
is discarded. Unimplemented entries have been removed from the configuration and current
documentation. Missing door or fixture information is never converted into a passing result.

## Geometry and claims

C033/C035 remain potential bedroom feasibility with a shared bed arrangement. C036 remains
empty-bathroom turning feasibility, excluding unobserved fixtures and door swings.
The metric scale remains assumed .025 m/pixel, with two occupants and uniform 2.5 m ceilings.

C079 declares the existing zero-thickness area proxy rather than inventing actual wall thickness.
C080 supplies a central exclusion table and removes area below 1.80 m under the uniform-height
scenario; equality is included. C076–C078 use this shared area definition on CPU and GPU.
Missing height or an unknown area model leaves these active measurements unknown.
Bed/bath geometry still uses physical room area. Supporting rules create no separate pass or reward.

At the default 2.5 m height, area values retain the previous estimates. Overall validity means
all six applicable scored checks pass. Supporting measurement rules add no separate pass/fail result.

## Configuration and reporting

The runner and launcher default to `configs/experiment_v2.json`. Full/late SVDD, Best-of-N and
baseline use the same selected profile. Group metrics cover the six scored checks and exclude
the three supporting measurement rules. Registry and claim metadata are saved.
The v1 configuration is preserved for reproducing older results.

```bash
python -m test_time.run --method baseline --list-constraints
python -m pytest -q test_time/tests
bash test_time/launch_eval.sh --dry-run
```

These commands inspect configuration, test the implementation and preview the evaluation plan.
Generation still requires an explicit run.

## Validation

Replayed all 12 saved pilot candidates without loading model weights or generating new designs.
The six CPU check statuses and energy match the saved v1 results on all 12. Tensor structural
validity matches on all 12, and tensor check statuses agree with the CPU reference on all 10
structurally valid candidates. Invalid candidates deliberately skip tensor searches as before.
The replay is saved in `outputs/004_feasible_constraints/implementation_audit.json`.

Synthetic tests cover registry counts, unknown-checker rejection, supporting-rule denominators,
missing metadata, the 1.80 m boundary, legacy behavior, CPU/tensor agreement, sampler integration,
group reporting and configuration listing: **186 tests passed, three CUDA tests skipped locally**.
CUDA validation remains pending on the node. Launcher syntax and the default preview were checked.
