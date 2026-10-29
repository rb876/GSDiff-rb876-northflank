# Compact three-bedroom design profile

The new default is **ten scored checks**: the six assessment-derived checks plus four user-defined design requirements. Three supporting rules still supply measurements without separate reward. The older six-check profiles remain available for reproduction.

| ID | Design requirement | Penalty |
|---|---|---|
| D001 | At least 3 bedrooms | Normalized bedroom-count shortfall |
| D002 | Every bedroom ≥10 m² | Worst normalized bedroom-area shortfall |
| D003 | At least one living room ≥18 m² | Normalized shortfall of the largest living room |
| D004 | Habitable area ≤100 m² | Normalized excess above the area budget |

These are a design brief, **not four additional Appendix A regulations**. No new occupancy assumption is introduced: the scenario remains two intended occupants, uniform 2.5 m height, paper-derived 18/512 m per drawing pixel and zero-thickness walls. Room counts and areas use generated labels/polygons. D002/D003 use geometric area; D004 shares C079/C080 habitable-area exclusions.

All four guidance rewards are tensor operations on the model GPU. Final selected outputs receive the CPU reference audit, including the same new requirements. The new count/area checks need no placement search. Missing required rooms fail; missing required measurements remain unknown. Full definitions, assumptions and LaTeX are in [constraints.md](constraints.md).

## Reuse saved candidates first

After pushing from Kraken and pulling on Northflank, activate the existing environment:

```bash
cd /workspace/GSDiff
source /workspace/venvs/gsdiff/bin/activate
python -m pytest -q test_time/tests
```

Use the **original** `sweep_002` with all 1,100 Best-of-11 candidates, not its selected-only audit. This performs no diffusion or edge-model inference:

```bash
python -m test_time.recalibrate \
  --experiment_dir test_time/outputs/sweep_002 \
  --output_dir test_time/outputs/design_001/controls \
  --scenario test_time/configs/scenario_rplan18.json \
  --constraints test_time/configs/constraints_design_v1.json \
  --device cuda:0 --inference_batch_size 32
python -m test_time.evaluate --experiment_dir test_time/outputs/design_001/controls
python -m test_time.analyze --experiment_dir test_time/outputs/design_001/controls
```

On Kraken, use `--device cpu` with the local environment. The same tensor code runs on CPU.

The new folders contain re-evaluated baseline designs, reselected Best-of-11 winners, saved candidate-level scores and provenance. New selected CPU metrics determine all-pass rates. Historical generation NFE and separate rescoring time remain distinguished. Existing outputs are preserved.

## Read the plots and examples

Open `test_time/outputs/design_001/controls/analysis/REPORT.md` or `index.html`.

- `figures/constraint_outcomes.*`: all ten scored checks, with the D checks marked as design requirements.
- `figures/passed_constraints.*`: histogram covering zero through ten passes.
- `tables/constraints.csv`: per-check pass/fail/unknown and failure reasons.
- `tables/groups.csv`: assessment groups and `design_brief` results separately.
- `samples/`: labelled rooms and a measured explanation for each passed or failed requirement.

No new design results are inferred from old figures. Re-analysis must use newly evaluated outputs and their saved expanded registry.

## Smoke tests and subsequent guidance

Both launchers now default to the expanded profile. Two outputs per SVDD variant with K=2:

```bash
EVAL_OUTPUT=test_time/outputs/design_svdd_smoke_001 VARIANTS='svdd_pm_last20 svdd_pm_full' TRIALS_PER_RUN=2 SVDD_K=2 bash test_time/launch_eval.sh --run
```

Two attempts per extension, repair-aware first:

```bash
WORKERS=2 EVAL_OUTPUT=test_time/outputs/design_extensions_smoke_001 bash test_time/launch_extensions.sh --smoke --run
```

To run both ordinary SVDD variants on the same pilot seeds after inspecting the controls and smoke tests:

```bash
nohup env GSDIFF_PYTHON=/workspace/venvs/gsdiff/bin/python \
  EVAL_OUTPUT=test_time/outputs/design_001/guided \
  VARIANTS='svdd_pm_last20 svdd_pm_full' EVAL_SEEDS=10000 TRIALS_PER_RUN=100 \
  bash test_time/launch_eval.sh --run > design_svdd_001.log 2>&1 < /dev/null &
```

After it finishes:

```bash
python -m test_time.evaluate --experiment_dir test_time/outputs/design_001
python -m test_time.analyze --experiment_dir test_time/outputs/design_001
```

The saved bank is a difficulty pilot. Freeze requirements before a final comparison on fresh design seeds; baseline and Best-of-N must also use those fresh seeds. Every compared method must use the same registry and scenario. Existing guided trajectories used the earlier reward and cannot establish performance of guidance toward D001–D004 merely by rescoring their final outputs.

The original six-check profile is selected explicitly with `EVAL_CONFIG=test_time/configs/experiment_rplan18.json` or `EXTENSION_CONFIG=test_time/configs/extensions_rplan18.json`. Current configs are [experiment_design_v1.json](../test_time/configs/experiment_design_v1.json) and [extensions_design_v1.json](../test_time/configs/extensions_design_v1.json).
