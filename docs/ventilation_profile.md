# PDF constraint profile: baseline and Best-of-11

The default benchmark scores **nine Appendix A constraints**: C033, C035, C036, C076, C077, C078, C085, C086 and C090. C034/C079/C080 remain supporting definitions.

The three ventilation rules come from [Technical_assessment_TTC.pdf, Appendix A, page 7](../Technical_assessment_TTC.pdf). The scenario declares a system sized for three main rooms:

| Scenario input | Fixed value |
|---|---:|
| Kitchen normal extraction, per kitchen | 105 m³/h |
| Bathroom normal extraction, per bathroom | 30 m³/h |
| Dwelling aggregate reduced extraction | 75 m³/h |
| Kitchen reduced extraction, per kitchen | 45 m³/h |
| Ventilation mode | Occupant-adjustable reduction |

These user-selected capacities are **scenario assumptions**, not measurements or outputs of GSDiff. They stay fixed for all methods and candidates. Main rooms are living-room and bedroom polygons. Label-4 rooms are explicitly treated as full bathrooms sharing the declared bathroom rate; other shower-room roles are unresolved. Normal and reduced operation have different rates. Vent locations, ducts, air inlets and physical airflow are not verified.

The scale remains **18/512 = 0.03515625 m/drawing pixel**, with two intended occupants and uniform 2.5 m ceilings. [Scenario JSON](../test_time/configs/scenario_ventilation_v1.json), [constraint registry](../test_time/configs/constraints_ventilation_v1.json), [definitions and reward formulas](constraints.md).

## How the three checks work

- **C085:** required kitchen flow is 75/90/105/120/135 m³/h for 1/2/3/4/5+ main rooms.
- **C086:** required bathroom flow is 15/15/30/30/30 m³/h for the same counts.
- **C090:** for 1–7 main rooms, required reduced total flow is 35/60/75/90/105/120/135 m³/h and required reduced kitchen flow is 20/30/45/45/45/45/45 m³/h. **Both** must pass; the larger normalized deficit supplies the penalty.

The GPU implementation uses room counts and tensor table lookups; final selected outputs receive the same checks in the CPU audit. Missing rates, unknown bathroom roles, missing service-room mappings and unsupported table counts remain unknown. C090 is not applicable to a known different mode; an unspecified mode is unknown. A reduced total below the sum of declared kitchen reduced flows is inconsistent and unknown.

At the selected rates, C085 and C090 share a three-main-room limit for supported, consistently mapped layouts. C086's 30 m³/h meets every listed flow threshold when the bathroom mapping is available. These are distinct PDF rules with correlated outcomes, not three independent sources of difficulty. Rescoring will establish the benchmark's actual difficulty; no lower Best-of-11 success rate is assumed.

## Stop previous runs

On Northflank, these commands stop experiment launchers and generation workers; they do not remove saved outputs:

```bash
pkill -TERM -f 'bash test_time/launch_(eval|extensions)[.]sh' || true
pkill -TERM -f 'test_time[.]run(_extensions)?( |$)' || true
```

Commit/push the changes from Kraken. On the node:

```bash
cd /workspace/GSDiff
git pull --ff-only
source /workspace/venvs/gsdiff/bin/activate
python -m pytest -q test_time/tests
```

## Re-evaluate baseline and reselect Best-of-11

Use the original `sweep_002`, which contains **100 baseline designs and all 1,100 Best-of-11 candidates**. Do not use a selected-only CPU-audit folder. No model inference is required:

```bash
python -m test_time.recalibrate \
  --experiment_dir test_time/outputs/sweep_002 \
  --output_dir test_time/outputs/ventilation_001/controls \
  --scenario test_time/configs/scenario_ventilation_v1.json \
  --constraints test_time/configs/constraints_ventilation_v1.json \
  --device cuda:0 --inference_batch_size 32
python -m test_time.evaluate --experiment_dir test_time/outputs/ventilation_001/controls
python -m test_time.analyze --experiment_dir test_time/outputs/ventilation_001/controls
cat test_time/outputs/ventilation_001/controls/comparison.md
```

On Kraken, `--device cpu` runs the same tensor scoring code on CPU. The input files must exist on the machine running the command. Use a new output name if the destination already exists.

`calibration.json` records the original scenario/registry, the declared ventilation changes, source hashes and changed winners. Baseline is rechecked; every Best-of-11 candidate is rescored, one winner is reselected per trial and those winners receive the final CPU check. Original files remain unchanged. Historical generation NFE and new rescoring time are separate; generation runtime is not remeasured.

Open `ventilation_001/controls/analysis/REPORT.md` or `index.html` for nine-check plots, the ventilation-group table, labelled rooms and measured explanations.

## Later guided runs

The ordinary and extension launchers select [experiment_ventilation_v1.json](../test_time/configs/experiment_ventilation_v1.json) and [extensions_ventilation_v1.json](../test_time/configs/extensions_ventilation_v1.json). Guided methods need trajectories using this reward and scenario. Runs with different saved constraints cannot supply these guided results.

The first evaluation of the saved bank is a pilot. Keep capacities and thresholds fixed, and use shared fresh seeds across all methods for a final comparison if the pilot informs further scenario choices.
