# GSDiff test-time experiments

Edit on Kraken, push the code, pull on Northflank, and run on its H100.
The original GSDiff network architecture and pretrained weights are unchanged.

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
  gpu_graph.py               # tensor room extraction and graph checks
  gpu_constraints.py         # tensor constraint searches
  gpu_sampling.py            # GPU guidance and selection; final CPU audit
  audit_gpu.py               # compare tensor checks against saved results
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
python -m test_time.run --method svdd_pm --k 10 --guided_steps 999 --temperature 0
python -m test_time.evaluate --experiment_dir test_time/outputs/001_constraints
```

Defaults are one trial, seed 42, GPU rewards, and the same scenario/checkpoints for all methods.
SVDD-PM now defaults to K=10 and greedy selection at all 999 stochastic reverse steps.
This full-guidance configuration has not yet been evaluated; earlier results used shorter guidance windows.
Larger experiments and their budgets will be decided separately; `--trials` controls their size.
Use `--config path.json` to change the experiment, or `--output_dir path` for a new explicit run directory.
`--n`, `--k`, `--guided_steps`, `--temperature`, and `--seed` override JSON settings.
Positive temperature samples from stable softmax weights; zero selects greedily.

## Batched generation

All three methods support batching. `--batch_size` (default **4**) controls independent design
trials processed together. `--inference_batch_size` (default **10**) limits the number of states
in each neural-network call, including Best-of-N candidates and SVDD proposals. With one trial,
SVDD still evaluates its ten proposals in batches. Remaining trials form a smaller final batch.

For example, after pulling the code on Northflank:

```bash
python -m test_time.run --method svdd_pm --trials 8 --seed 1000 \
  --batch_size 4 --inference_batch_size 10
```

This uses the current K=10, 999-step, temperature-0 defaults. Lower `--inference_batch_size`
if the edge model needs too much GPU memory. Set both batch sizes to 1 for sequential inference.
By default, intermediate geometry extraction and constraint checking use the batched GPU backend.
Final saved designs still receive the CPU reference audit.

Each trajectory has its own seeded random stream. Competing SVDD proposals share the same edge
interpolation draw, as before. The edge network accepts optional precomputed interpolation draws
to preserve that behavior during batching; its default forward behavior and weights are unchanged.
GPU kernels can produce small floating-point differences at different batch sizes, so exact CUDA
output equality is not guaranteed. CPU tests check seed independence and selection consistency.

Timing is recorded once per generation batch. Per-trial seconds are the batch time divided by its
trial count (amortized throughput), not isolated latency. Reports record both actual forward calls
and `node_samples`/`edge_samples` (states processed). Compute comparisons require the same batching
settings and still need measured runtime calibration on the H100.

Each run records every complete candidate in `candidates.jsonl`, selected IDs,
PNG/SVG/JSON files, the config and source hashes, timings, model call counts, and a `report.md`.
With CPU rewards, SVDD additionally saves every intermediate surrogate graph/evaluation and selection
in `traces/`. With K=10 and 999 guided steps, that is 9,990 intermediate graphs per trial;
full-trace storage must be budgeted separately before a large evaluation.
GPU rewards use compact traces containing scores, check statuses and selection decisions.
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

## GPU room extraction and rewards

The default `--reward_backend gpu` runs intermediate graph extraction, constraint searches and
proposal selection with PyTorch tensors on the model's device. All six scored checks and auxiliary
C034 are retained, including the shared bed placement for C033/C035. Final saved candidates
are audited with the existing CPU checker; reports and overall validity always use that audit.

On Northflank, to generate **four final designs** with the current full-guidance defaults:

```bash
python -m test_time.run --method svdd_pm --trials 4 --seed 1000 \
  --batch_size 4 --inference_batch_size 10
```

`--inference_batch_size` also limits the number of graphs scored together. Progress is printed
at the first, every 50th and last guided step. Output locations are printed at startup;
the default is a new directory under `test_time/outputs/001_constraints/`.
Best-of-N uses the same GPU backend by default. Pass `--reward_backend cpu` to opt into the
reference CPU guidance implementation. Historical saved CPU runs retain their original backend.

The GPU backend uses planar faces instead of polygon unions: verified, non-overlapping room
areas can be summed directly. It uses the same configured .25/.05 m search grids and a maximum
of 20,000 bed positions per room, retaining up to 32 directional circle witnesses. These finite
searches can miss a valid placement; missed searches remain `unknown`. Bathroom failure requires
a proven bounding-box/area bound. Structurally invalid graphs skip dimensional checks and get the
structural penalty. This backend is therefore **not numerically identical** to the CPU checker.
Run configs and sample metadata identify the backend; final GPU scores are saved separately.
At positive temperature, GPU selection uses a separate PyTorch random stream and will not match
the CPU NumPy selection stream exactly. Greedy temperature 0 is unchanged.

Local synthetic tests and a saved-pilot comparison passed, but CUDA execution and H100 speedup
have not been measured. Python dispatch, dynamic tensor packing and final CPU auditing still
add overhead. This implementation does not guarantee full GPU utilization.
See the [implementation audit](experiments/002_gpu_backend/REPORT.md).

To check saved designs on the node without loading models or generating anything:

```bash
python -m test_time.audit_gpu --experiment_dir test_time/outputs/pilot_check \
  --device cuda:0 --output test_time/outputs/gpu_backend_check/audit_cuda.json
```

## Proposed larger evaluation: launcher

The launcher is prepared, but no larger evaluation has been launched.
Preview it from the repository root (no models run and no outputs are created):

```bash
bash test_time/launch_eval.sh
```

Its proposed defaults are five batches of 200 trials: **1,000 selected designs per method**.
Each batch uses a different range of 200 design seeds, shared by all three methods.
Baseline, Best-of-2 and SVDD-PM (K=10, 999 guided steps, temperature 0) run serially,
with method order rotated between batches. Best-of-2 remains a provisional launcher setting;
it has not been calibrated against full-guidance SVDD-PM. Recalibrate `BEST_N` using measured
runtime before a larger comparison.

The following is the explicit launch command for later, inside the activated Northflank environment:

```bash
bash test_time/launch_eval.sh --run
```

Override settings through `EVAL_SEEDS` (space-separated batch start seeds), `TRIALS_PER_RUN`,
`BEST_N`, `SVDD_K`, `GUIDED_STEPS`, `TEMPERATURE`, `BATCH_SIZE`, `INFERENCE_BATCH_SIZE`, `REWARD_BACKEND`,
`EVAL_CONFIG`, `EVAL_OUTPUT`, or `GSDIFF_PYTHON`.
For example, preview one small calibration batch:

```bash
EVAL_SEEDS="1000" TRIALS_PER_RUN=20 bash test_time/launch_eval.sh --dry-run
```

The launcher defaults to `REWARD_BACKEND=gpu`; set `REWARD_BACKEND=cpu` for reference CPU guidance.
Recalibrate matched compute after
changing this backend; CPU-guided timings do not establish a budget for GPU guidance.

Each batch gets method logs, saved candidates, reports and its own comparison table.
The script stops on failure; interrupted-run resume is not implemented.
The existing evaluator does **not** pool the five batches or calculate confidence intervals,
near-duplicate rates, or geometric/structural diversity distances. Those reporting additions
and compute-budget calibration remain to be done
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
Guided steps are `t=999,...,1` by default; the deterministic final step `t=0` is never duplicated.
This follows the [SVDD paper's §§4.2 and 6.1](https://arxiv.org/html/2408.08252v3):
guide throughout the reverse process, use greedy selection (alpha=0), and choose 5–20 proposals;
its experiments use 10 proposals outside image generation. Our graph-validity and coverage guards
remain GSDiff-specific adaptations.
This is approximate SVDD-PM, not an exact sampler from the reward-tilted target distribution.

The headline **overall validity** requires a structurally valid graph and every applicable scored
constraint to measurably pass. Any structural defect or failed scored check makes the sample invalid.
Without a confirmed failure, unresolved checks make it unknown; no applicable scored checks makes
it not applicable. Neither counts as valid. Auxiliary C034 and individual inapplicable checks are excluded.
The overall-valid rate divides confirmed valid samples by **all selected designs**.
This covers the six implemented scored checks, not every constraint in the assessment.

Reports retain structural validity, per-check compliance, coverage, unknown fraction and assumptions
as diagnostics. JSON uses `overall_valid_rate`, `overall_validity_counts` and `structural_valid_rate`;
legacy `valid_count` and `validity_rate` remain structural aliases. Figures show both overall and structural status.
Rebuild reports from saved evaluations without loading models or generating designs:

```bash
python -m test_time.evaluate --experiment_dir test_time/outputs/pilot_check/batch_1_seed_1000
```

Measured compute includes sampling and checking; loading/rendering/export time is recorded separately.
Matched-compute labels require measured times within 20%. No RPLAN-relative FID/KID is claimed without reference data.

References: [GSDiff](https://arxiv.org/abs/2408.16258), [SVDD](https://arxiv.org/abs/2408.08252),
and the supplied `Technical_assessment_TTC.pdf`, Appendix A.
