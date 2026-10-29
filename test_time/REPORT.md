# GSDiff: test-time constraint guidance

The [technical report](../docs/technical_report.md) is the presentation document: model background, SVDD mathematics, the PDF constraint set, evaluation protocol and algorithmic extensions. This page summarizes the runnable setup.

## Default benchmark

All eleven scored constraints come from Appendix A of `Technical_assessment_TTC.pdf`:

| Family | Scored IDs | What is checked |
|---|---|---|
| Bedroom and bathroom accessibility | C033, C035, C036, C037, C041 | Bed/turning-space placement and bathroom turning space |
| Habitability | C076, C077, C078 | Main-room size and dwelling area/volume |
| Ventilation | C085, C086, C090 | Normal kitchen/bathroom and reduced-mode total/kitchen extraction |

C034, C079 and C080 supply measurements without separate reward or compliance points.

The scenario uses **18/512 = 0.03515625 m/drawing pixel**, two occupants and 2.5 m ceilings. Declared extraction rates are **105 m³/h normal kitchen, 30 m³/h normal bathroom, 75 m³/h reduced dwelling total and 45 m³/h reduced kitchen**. The model supplies labelled room geometry; ventilation rates, height and furniture assumptions are explicit scenario inputs. These checks do not establish complete physical or legal compliance.

## Methods and compute

All methods share frozen unconditional GSDiff checkpoints, scenario and constraints. The default reward runs on GPU; exported outputs receive the CPU audit. Overall validity requires valid structure and every applicable scored check to measurably pass. Unknowns never count as passes.

| Method | Default configuration | Planned denoiser NFE/output |
|---|---|---:|
| Baseline | One ordinary 1,000-step trajectory | 1,000 |
| Best-of-11 | Select among eleven complete candidates | 11,000 |
| Late SVDD | K=10 at the final 200 stochastic steps | 3,000 |
| Full SVDD | K=10 at all 999 stochastic steps | 10,990 |

Best-of-11 approximately matches full SVDD by denoiser NFE. This does not match edge-model calls, reward evaluations or elapsed time. Hard-feasibility MC and repair-aware SVDD use the same PDF profile with their separately recorded budgets.

## Evaluation status

Results for the eleven-check profile are pending. Existing baseline designs and complete Best-of-N banks can be rescored; guided methods need trajectories using this reward. Current saved candidates are a development pilot; hold the scenario fixed and use shared fresh seeds for a final comparison if the pilot informs further choices.

Report overall validity with confidence intervals, per-constraint and group outcomes, structural validity, unknown rates, severity, diversity and NFE/runtime separately. Figures and gallery examples use each run's saved registry and scenario.

[Run and rescore commands](../docs/bathroom_profile.md) · [constraint definitions and reward formulas](../docs/constraints.md) · [implementation guide](README.md).

Earlier experiment reports remain under `experiments/` as historical records. Saved output files are preserved; results from different registries or scenarios are not pooled into this benchmark.
