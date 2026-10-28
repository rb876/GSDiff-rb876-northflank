# GSDiff test-time experiments

Edit on Kraken, push the code, pull on Northflank, and run on its H100.
The original GSDiff networks and pretrained weights are unchanged.

```text
test_time/
  REPORT.md                 # concise project report, ready for HackMD
  experiments/
    TEMPLATE.md             # use this for each new research experiment
    000_baseline/REPORT.md
    001_constraints/REPORT.md
  configs/                  # constraints, scenario and experiment settings
  adapter.py                # original GSDiff sampler/model interface
  geometry.py               # geometric placement witnesses
  constraints.py            # config-driven checks
  policy.py                 # shared energy and structural validity
  sampling.py               # baseline, Best-of-N, SVDD-PM
  run.py                    # one experiment entry point
  launch_eval.sh            # preview/run paired batches; no automatic launch
  evaluate.py               # saved-result summaries and comparisons
  visualize.py              # annotated PNG/SVG
  tests/                    # small synthetic tests; no weights required
  outputs/                  # generated runs; ignored by Git
```

## Run on Northflank

From Kraken:

```bash
northflank exec service --project test-rb876 --service gsdiff-dev
```

Inside the container, after your code has been pushed:

```bash
cd /workspace/GSDiff
git pull --ff-only origin main
source /workspace/venvs/gsdiff/bin/activate
python -m pip install -r test_time/requirements.txt
python -m pytest -q test_time/tests

python -m test_time.run --method baseline
python -m test_time.run --method best_of_n --n 4
python -m test_time.run --method svdd_pm --k 4 --guided_steps 10 --temperature 0
python -m test_time.evaluate --experiment_dir test_time/outputs/001_constraints
```

Defaults are one smoke-test trial, seed 42, and the same scenario/checkpoints for all methods.
Larger experiments and their budgets will be decided separately; `--trials` controls their size.
Use `--config path.json` to change the experiment, or `--output_dir path` for a new explicit run directory.
`--n`, `--k`, `--guided_steps`, `--temperature`, and `--seed` override JSON settings.
Positive temperature samples from stable softmax weights; zero selects greedily.

Each run records every complete candidate in `candidates.jsonl`, selected IDs,
PNG/SVG/JSON files, the config and source hashes, timings, model call counts, and a `report.md`.
SVDD additionally saves every intermediate surrogate graph/evaluation and selection in `traces/`.
Failed or invalid candidates are retained. A partial run is not a completed benchmark.

From Kraken, copy results back:

```bash
northflank download service file \
  --project test-rb876 --service gsdiff-dev \
  --remotePath /workspace/GSDiff/test_time/outputs \
  --localPath /home/rbarbano/home/git/GSDiff-rb876-northflank/test_time/outputs
```

Commit code, configs and concise experiment reports. Keep full generated runs in ignored `outputs/`.
Copy a few useful figures into the relevant experiment's `figures/` folder for a reviewable report.
The Markdown can be pasted into HackMD; no document has been published there automatically.

## Proposed larger evaluation: launcher

The launcher is prepared, but no larger evaluation has been launched.
Preview it from the repository root (no models run and no outputs are created):

```bash
bash test_time/launch_eval.sh
```

Its proposed defaults are five batches of 200 trials: **1,000 selected designs per method**.
Each batch uses a different range of 200 design seeds, shared by all three methods.
Baseline, Best-of-2 and SVDD-PM (K=4, ten guided steps, temperature 0) run serially,
with method order rotated between batches. These are provisional settings; similar runtime
in one smoke test does not establish matched compute for a full evaluation.

The following is the explicit launch command for later, inside the activated Northflank environment:

```bash
bash test_time/launch_eval.sh --run
```

Override settings through `EVAL_SEEDS` (space-separated batch start seeds), `TRIALS_PER_RUN`,
`BEST_N`, `SVDD_K`, `GUIDED_STEPS`, `TEMPERATURE`, `EVAL_CONFIG`, `EVAL_OUTPUT`, or `GSDIFF_PYTHON`.
For example, preview one small calibration batch:

```bash
EVAL_SEEDS="1000" TRIALS_PER_RUN=20 bash test_time/launch_eval.sh --dry-run
```

Each batch gets method logs, saved candidates, reports and its own comparison table.
The script stops on failure; interrupted-run resume is not implemented.
The existing evaluator does **not** pool the five batches or calculate confidence intervals,
near-duplicate rates, or geometric/structural diversity distances. Those reporting additions,
the combined valid-and-all-checks-passed metric, and compute-budget calibration remain to be done
before treating this as the final evaluation. Aggregate by paired trial seed, not the Best-of-N winner's seed.

## Checkpoints and baseline audit

The existing Northflank setup already contains:

- `outputs/structure-1/model1000000.pt` — `gsdiff.house_nn1.HeterHouseModel`.
- `outputs/structure-2/model_stage2_best_061000.pt` — `gsdiff.house_nn2.EdgeModel`.

For another machine, download the official [unconstrained archive](https://drive.google.com/file/d/15gM0GtW2GwHmlpz0r-rpvo-k-BlNy_gu/view)
and extract its `outputs/` directory into the repository. Explicit checkpoint paths are also accepted by the CLI.
RPLAN is unnecessary for unconstrained generation and is not bundled here.
Checker/sampler unit tests work without checkpoints; missing weights produce setup instructions, never fake model output.

`scripts/test_main.py` has no importable sampling API and triggers dataset rendering/FID/KID at import.
The adapter therefore copies only its 1,000-step cosine DDPM formulas and calls the original models,
`inverse_normalize_and_remove_padding_100_4testing`, `edges_to_coordinates`, and
`get_cycle_basis_and_semantic_3_semansimplified`.
State shape is `(1, 53, 10)`: XY, seven semantic bits, and padding. Float64 state, original
thresholds (0.5 semantics, 0.75 padding), edge softmax/argmax, bridge removal and pixel decoding are retained.
No alignment or point merging is added. Raw predictions are retained before room extraction removes bridges.

The edge network draws random interpolation points even in evaluation mode. We clone the current
diffusion RNG state for its call: this reproduces the original final edge draw and gives common
edge randomness to the competing SVDD proposals without consuming diffusion noise.

To reproduce the initial seed-42 parity audit on the current service:

```bash
python -m test_time.audit \
  --reference /workspace/results/smoke-20261028T110433-573119Z/raw_prediction.npz
```

## Scope of the scores

`configs/constraints_v1.json` defines every enabled checker, threshold, weight and assumption.
C033/C035 share one bed placement when both pass. Finite placement-search misses are `unknown`;
they are not proofs of infeasibility. C034 supplies bed dimensions and an auxiliary footprint check,
so it does not inflate the six-constraint compliance denominator. C036 is empty-bathroom feasibility.
C076 uses `(area AND height) OR volume`; C077/C078 use the occupancy breakpoint after four people.
Main rooms are living/bedroom; estimated habitable area includes living, bedroom, kitchen and bathroom,
and excludes storage, balcony and outside. These are zero-thickness polygon estimates.

Coordinates are drawing pixels, not metres. The user-approved experimental scenario assumes
0.025 m per drawing pixel, two occupants and 2.5 m ceilings. Every resulting metric is assumption-based.
Set scale to `null` and provenance to `missing` to obtain `unknown` dimensionful checks.
Furniture, fixtures, doors, physical wall thickness and headroom are not generated; these scores
are conditional feasibility measurements, not verified building compliance.

Energy is the weighted sum of measurable normalized violations plus a separate validity penalty.
Unknowns have no invented severity. Selection orders valid graphs first, then fewer unknown checks,
then lower energy and seed. Within the best validity class, candidates may not introduce an unknown
ID that was measurable in the first candidate. This prevents swapping an expensive failure for a new unknown.
The guard is conservative and order-dependent; it can reject improvements with different unresolved checks.
SVDD applies the same coverage guard to surrogate candidates and falls back to an ordinary proposal
when none is scorable. It scores predicted clean nodes but continues from the selected **noisy** state.
Guided steps are `t=10,...,1` by default; the deterministic final step `t=0` is never duplicated.
This is approximate SVDD-PM, not an exact sampler from the reward-tilted target distribution.

Reports always pair measurable compliance with coverage, unknown fraction, assumptions and structural validity.
Measured compute includes sampling and checking; loading/rendering/export time is recorded separately.
Matched-compute labels require measured times within 20%. No RPLAN-relative FID/KID is claimed without reference data.

References: [GSDiff](https://arxiv.org/abs/2408.16258), [SVDD](https://arxiv.org/abs/2408.08252),
and the supplied `Technical_assessment_TTC.pdf`, Appendix A.
