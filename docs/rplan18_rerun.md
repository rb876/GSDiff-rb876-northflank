# Re-evaluate at RPLAN's documented scale

Use **0.03515625 m/drawing pixel = 18/512**. The RPLAN paper describes an 18 m canvas rasterized to 256 pixels; GSDiff's fixed normalization and 512-pixel decoder double the drawing coordinates. This calibration is paper/code-derived, not a field measurement. See [scenario_rplan18.json](../test_time/configs/scenario_rplan18.json) and [the source paper, §4](https://wutomwu.github.io/publications/2019-DeepLayout/paper.pdf).

The original configurations and outputs remain available. Do not mix old-scale guided outputs into the corrected comparison. The main technical report has corrected results marked **pending**; the [old report](technical_report_assumed_scale_archive.md) is archived.

## 1. Stop the old extension launch on Northflank

These commands stop the extension launcher and its Python workers. They do not delete outputs.

```bash
pkill -TERM -f 'bash test_time/launch_extensions[.]sh'
pkill -TERM -f 'test_time[.]run_extensions'
pgrep -af '[t]est_time\.run_extensions'
```

No output from `pgrep` means no matching workers remain. Commit/push the code changes from Kraken, then on the node:

```bash
cd /workspace/GSDiff
git pull --ff-only
source /workspace/venvs/gsdiff/bin/activate
python -m pytest -q test_time/tests
```

The corrected six-check config files are already included; no manual JSON editing is needed. This runbook explicitly selects the original six-check profile. For the current nine-check PDF profile, use [the ventilation commands](ventilation_profile.md). Use this virtual environment for every command below.

## 2. Reuse baseline and reselect Best-of-11

Use the **original `sweep_002`**, which contains all 1,100 Best-of-11 candidates. The CPU-audit folder contains only the 100 previously selected designs and cannot support reselection.

```bash
python -m test_time.recalibrate \
  --experiment_dir test_time/outputs/sweep_002 \
  --output_dir test_time/outputs/rplan18_001/controls \
  --scenario test_time/configs/scenario_rplan18.json \
  --device cuda:0 --inference_batch_size 32
python -m test_time.evaluate --experiment_dir test_time/outputs/rplan18_001/controls
cat test_time/outputs/rplan18_001/controls/comparison.md
```

No model inference occurs: baseline designs are re-evaluated; all Best-of-11 candidates are rescored using the original tensor ranking policy at the new scale, then selected again within each trial. Selected outputs receive the refined CPU final check. Rejected candidates retain tensor evaluations, so selected CPU results supply the comparison statistics. `calibration.json` records changed selections and separate recalibration time. Original generation counters remain historical; no fresh generation runtime is claimed.

On Kraken, use the same commands with `--device cpu`; this runs the same tensor scoring algorithm on CPU. The input folder must be present on the machine where the command runs.

## 3. Smoke-test guidance before the full rerun

First inspect the corrected control success rates. If they already approach 100%, the constraints leave little room to demonstrate a guidance benefit.

Two outputs per SVDD variant, with fewer candidates for this smoke test:

```bash
EVAL_CONFIG=test_time/configs/experiment_rplan18.json EVAL_OUTPUT=test_time/outputs/rplan18_svdd_smoke_001 \
VARIANTS='svdd_pm_last20 svdd_pm_full' TRIALS_PER_RUN=2 SVDD_K=2 \
bash test_time/launch_eval.sh --run
```

Two attempts per extension (repair-aware first, then hard feasibility):

```bash
EXTENSION_CONFIG=test_time/configs/extensions_rplan18.json WORKERS=2 EVAL_OUTPUT=test_time/outputs/rplan18_extensions_smoke_001 \
bash test_time/launch_extensions.sh --smoke --run
```

Check each smoke folder's `comparison.md`, per-method reports and sample artefacts. Smoke results are kept out of the full comparison. If rerunning any stage, choose a new output name; overwriting/resuming is intentionally unsupported.

## 4. Full reruns, after inspecting the smoke tests

Rerun both SVDD variants because their guidance used the old scale. This gives 100 outputs per variant, K=10, paired seeds 10000–10099; full SVDD remains matched to the saved Best-of-11 bank by denoiser NFE.

```bash
nohup env GSDIFF_PYTHON=/workspace/venvs/gsdiff/bin/python \
  EVAL_CONFIG=test_time/configs/experiment_rplan18.json VARIANTS='svdd_pm_last20 svdd_pm_full' \
  EVAL_OUTPUT=test_time/outputs/rplan18_001/guided \
  EVAL_SEEDS=10000 TRIALS_PER_RUN=100 \
  bash test_time/launch_eval.sh --run > rplan18_svdd_001.log 2>&1 < /dev/null &
tail -f rplan18_svdd_001.log
```

When it finishes, rerun the extensions (100 attempts each, two workers; repair-aware first):

```bash
nohup env EXTENSION_CONFIG=test_time/configs/extensions_rplan18.json PYTHON=/workspace/venvs/gsdiff/bin/python WORKERS=2 \
  EVAL_OUTPUT=test_time/outputs/rplan18_extensions_001 \
  EVAL_SEED=10000 TRIALS_PER_RUN=100 \
  bash test_time/launch_extensions.sh --run > rplan18_extensions_001.log 2>&1 < /dev/null &
tail -f rplan18_extensions_001.log
```

`Ctrl+C` stops `tail`, not the detached job. `nohup` survives terminal disconnection, but not a container restart.

## 5. Analyse only the corrected outputs

The parent folder combines the reused controls and new SVDD runs, excluding smoke tests and old sweeps:

```bash
python -m test_time.evaluate --experiment_dir test_time/outputs/rplan18_001
python -m test_time.analyze --experiment_dir test_time/outputs/rplan18_001
python -m test_time.run_extensions --summarize_root test_time/outputs/rplan18_extensions_001
```

Read `rplan18_001/analysis/REPORT.md` and `rplan18_extensions_001/comparison.md`. Populate the main technical report from these corrected results only; retain the scale, scenario assumptions and separate historical NFE/recalibration timing provenance.
