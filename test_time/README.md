# GSDiff test-time experiments

Edit on Kraken, push the code, pull on Northflank, and run on its H100.
The original GSDiff network architecture and pretrained weights are unchanged.

**Current profile:** eleven PDF-derived checks (C033/C035/C036/C037/C041/C076/C077/C078/C085/C086/C090),
three supporting rules, and **0.03515625 m/drawing pixel**. The declared ventilation system supplies
105 m³/h normal kitchen, 30 m³/h normal bathroom, 75 m³/h reduced total and 45 m³/h reduced kitchen flow.
These are fixed scenario inputs, not model outputs. Follow the [PDF-profile commands](../docs/bathroom_profile.md)
to rescore baseline, reselect Best-of-11, derive Best-of-3 and plot the eleven checks.
The [six-check rerun commands](../docs/rplan18_rerun.md) explicitly retain the previous profile.
The older `scenario_v2.json` and `experiment_v2.json` remain historical configurations
at the user-assumed 0.025 m/pixel; explicit commands using them retain that scale.

Read the [concise technical report](../docs/technical_report.md) or consult the
[technical notes](../docs/technical_notes.md) for full implementation and assessment details.

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
  reselect.py                # fixed-prefix Best-of-N from a scored candidate bank
  run.py                    # one experiment entry point
  launch_eval.sh            # preview/run paired batches; no automatic launch
  evaluate.py               # saved-result summaries and comparisons
  analyze.py                # offline figures, metrics, failures and complete image gallery
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
python -m test_time.evaluate --experiment_dir test_time/outputs/004_feasible_constraints
```

Defaults are one trial, seed 42, GPU rewards, and the same scenario/checkpoints for all methods.
The default configuration is `configs/experiment_bathroom_v1.json`, evaluating eleven PDF-derived constraints using
three supporting measurement rules. C085/C086/C090 use explicit ventilation-system assumptions. C036/C037/C041 use the same declared toilet, basin and inward-door completion. Use `--config test_time/configs/experiment_v1.json` to reproduce the original profile.
SVDD-PM now defaults to K=10 and greedy selection at all 999 stochastic reverse steps.
The eleven-check profile requires new guided runs; results from different constraint registries remain separate.
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
and `node_samples`/`edge_samples` (states processed). The selected compute budget is denoiser
states per selected output: the launcher matches Best-of-N to full SVDD using this count.
Keep the same batching settings for comparisons. This does not match elapsed time, edge-model
work or constraint-checking work; those costs are reported separately.

Each run records every complete candidate in `candidates.jsonl`, selected IDs,
PNG/SVG/JSON files, the config and source hashes, timings, model call counts, and a `report.md`.
Each trial log also lists failed and unresolved scored constraint IDs, for example
`failed=[C035], unknown_ids=[]`, using the selected design's final CPU reference evaluation.
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
proposal selection with PyTorch tensors on the model's device. All eleven scored checks run in this
backend. C034/C079/C080 define the bed/area measurements, including the shared placement for C033/C035. Final saved candidates
are audited with the existing CPU checker; reports and overall validity always use that audit.

On Northflank, to generate **four final designs** with the current full-guidance defaults:

```bash
python -m test_time.run --method svdd_pm --trials 4 --seed 1000 \
  --batch_size 4 --inference_batch_size 10
```

`--inference_batch_size` also limits the number of graphs scored together. All methods show a
`tqdm` bar per batch with diffusion steps completed, elapsed time, approximate steps/s (or s/step),
and estimated time remaining. It covers all 1,000 reverse steps; a step processes the whole batch,
including SVDD proposals when guided. Final validation and export follow the bar. Use `--no-progress`
to hide it. Output locations are printed at startup;
the default is a new directory under `test_time/outputs/004_feasible_constraints/`.
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

Preview the launcher from the repository root (no models run and no outputs are created):

```bash
bash test_time/launch_eval.sh
```

Its default is one seed batch starting at 10000, with 100 trials: **100 selected designs per variant**, or
**400 selected designs total**. The batch shares its 100 design seeds across four variants:

| Variant directory | Configuration |
|---|---|
| `baseline` | Unconstrained generation |
| `best_of_n` | Best-of-11 by default; N derived from full SVDD's denoiser budget |
| `svdd_pm_last20` | K=10, temperature 0, guidance at `t=200,...,1` |
| `svdd_pm_full` | K=10, temperature 0, guidance at `t=999,...,1` |

Every trajectory still denoises for all 1,000 steps. The late variant guides only the final
200 stochastic steps (approximately 20%); neither variant guides deterministic `t=0`.
Both SVDD variants use `--method svdd_pm` with different `--guided_steps` values.
Every seed batch runs serially in the order above, so late SVDD finishes before full SVDD.
After scoring the saved bank, the prefix-reselection command below adds **Best-of-3** as a
fifth analysis variant matching late SVDD, with no additional model generation.

For T denoising steps, K proposals and G guided steps, full SVDD evaluates T + K × G node states
per output; Best-of-N evaluates N × T. The launcher chooses the nearest integer N, rounding ties
upward. With T=1000, K=10 and G=999, the budgets are **10,990 for full SVDD** and **11,000 for
Best-of-11**, a **0.091% difference**. Late SVDD and **Best-of-3** each use a 3,000-state
budget per output. If K=1 or G=0,
the sampler takes its baseline shortcut, and the matched N is 1.

The root `compute_budget.json` records the planned match. Reports verify actual
`node_samples / selected_trials`, with differences within 1% labelled approximately matched.
Other K/G settings may have a larger rounding gap, which the report exposes. This is a denoiser
evaluation match, not a runtime or total-compute match.
See the [denoiser-budget report](experiments/005_denoiser_budget/REPORT.md), which supersedes
the provisional Best-of-2 plan in the [guidance-window report](experiments/003_guidance_window/REPORT.md).

The following is the explicit launch command for later, inside the activated Northflank environment:

```bash
bash test_time/launch_eval.sh --run
```

Override settings through `EVAL_SEEDS` (space-separated batch start seeds), `TRIALS_PER_RUN`,
`SVDD_K`, `GUIDED_STEPS` (full variant, default 999), `LATE_GUIDED_STEPS` (late variant, default 200),
`TEMPERATURE`, `BATCH_SIZE`, `INFERENCE_BATCH_SIZE`, `REWARD_BACKEND`,
`EVAL_CONFIG`, `EVAL_OUTPUT`, or `GSDIFF_PYTHON`.
`BEST_N` is rejected by the launcher because N is derived from the full SVDD settings; unset any
old override before launching. A standalone `python -m test_time.run --method best_of_n --n ...`
still accepts an explicit N, but does not automatically match another run.
For example, preview one small batch:

```bash
EVAL_SEEDS="1000" TRIALS_PER_RUN=20 bash test_time/launch_eval.sh --dry-run
```

The launcher defaults to `REWARD_BACKEND=gpu`; set `REWARD_BACKEND=cpu` for reference CPU guidance.
The denoiser-count formula holds for both backends. Their elapsed times differ, so CPU-guided
timings do not predict GPU-guided runtime.

Each batch gets separate variant directories, logs, configs, saved candidates and reports;
its comparison table includes both SVDD variants.
The script stops on failure; interrupted-run resume is not implemented.
The offline analysis command below adds overall-validity confidence intervals, duplicate and
semantic-layout diversity diagnostics, and paired trial outcomes. Seed batches remain separate;
cross-batch pooling is not implemented. Pair by trial seed, not the Best-of-N winner's seed.

### Add the Best-of-3 control from an existing bank

Retain candidate indices **0, 1 and 2** per trial before inspecting scores. Reapply the same
structural-validity, unknown-coverage, reward and seed selection rule to the saved GPU scores,
then CPU-audit the selected candidate. The existing Best-of-11 results and all 100 trials remain.

```bash
python -m test_time.reselect \
  --source_run test_time/outputs/bathroom_001/controls/batch_1_seed_10000/best_of_n \
  --output_dir test_time/outputs/bathroom_001/controls/batch_1_seed_10000/best_of_n_late \
  --n 3
python -m test_time.evaluate --experiment_dir test_time/outputs/bathroom_001/controls
python -m test_time.analyze --experiment_dir test_time/outputs/bathroom_001/controls
```

Use the complete scored bank, including rejected candidates, as the source. Rebuilding the
analysis includes both Best-of-3 and Best-of-11 with distinct labels. Their candidate banks are
nested, so outcomes are paired rather than independent. Reselection performs **zero new model
inference**: its 3,000 NFE/output is the equivalent cost of the retained prefix, and the original
full-bank cost stays recorded. Standalone runtime and batched forward-call counts are unknown.

## Analyse a completed sweep

After downloading results to Kraken, run from the repository root in the project environment:

```bash
python -m pip install -r test_time/requirements-analysis.txt
python -m test_time.analyze --experiment_dir test_time/outputs/sweep_001
```

This creates `sweep_001/analysis/REPORT.md`, a filterable `index.html` gallery, CSV/JSON metrics,
six PNG/PDF figures, and annotated PNG/SVG plus graph/evaluation JSON for **every selected design**.
Use `--output_dir PATH` to change the destination. The analysis needs no GPU or checkpoints and
does not rerun inference or constraint checks; original outputs remain unchanged.

Figures cover overall validity, per-constraint outcomes, the number of checks passed, measured
severity, layout distributions and compute. Failure tables include the recorded reasons and
the number of designs where a constraint is the sole failure. Method tables distinguish
passes/applicable checks from measurable-only compliance, so unknowns cannot inflate the
headline success rate. Best-of-N contributes only its selected design from each trial.

Diversity includes graph uniqueness, pairwise semantic-layout distance and near-duplicate
fraction, with eligible geometry counts. Distribution shifts compare against a compatible
generated baseline; these are not RPLAN fidelity scores. Completed runs are analysed separately;
running, failed and incomplete runs are excluded and listed. Full details and the assessment
requirements are in [the analysis guide](../docs/ANALYSIS.md).

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

`configs/constraints_bathroom_v1.json` contains fourteen implemented entries, with explicit groups, roles
and assumptions. The default profile is:

| Role | IDs | Meaning |
|---|---|---|
| Eight geometric checks | C033, C035, C036, C037, C041, C076, C077, C078 | Bedroom and shared furnished-bathroom feasibility; habitable area/volume |
| Three scored ventilation checks | C085, C086, C090 | Kitchen/bathroom extraction and occupant-adjustable reduced total/kitchen extraction |
| Three active supporting rules | C034, C079, C080 | Bed sizes; declared wall-area model; area exclusions |

The eleven scored checks determine overall validity and CPU audit energy; GPU guidance replaces the three coupled bathroom severity terms with one graded placement penalty. The ventilation group is also reported separately. The three supporting rules
define their measurements and add no separate reward or pass/fail result. A check that is
inconclusive for an individual design remains unknown. The selected set is fixed across all
four launcher variants. Reports include these IDs and per-group metrics.

List the configuration without model loading or generation:

```bash
python -m test_time.run --method baseline --list-constraints
```

C079 explicitly retains the existing **zero-thickness area proxy**; it does not infer actual wall
thickness. C080 centralizes room-class exclusions and excludes floor area below 1.80 m under the
declared uniform-height scenario (exactly 1.80 m is included). V2 area/volume checks become unknown
when height is missing or `habitable_area_mode` is `unknown`. Physical room areas used for bed/bath
placement remain unchanged. At the existing 2.5 m height, the new area rules retain the original
area estimates. C079/C080 add supporting definitions, not two new independent scored constraints.

The original `constraints_v1.json` and `scenario_v1.json` remain available through the v1 experiment
configuration. See the [constraint implementation report](experiments/004_feasible_constraints/REPORT.md).

C033/C035 share one bed placement when both pass. Finite placement-search misses are `unknown`;
they are not proofs of infeasibility. C034 supplies bed dimensions as a supporting rule;
its legacy v1 auxiliary footprint diagnostic is retained only in v1. C036/C037/C041 share one hypothetical bathroom/WC fixture and inward-door layout; see the [bathroom completion protocol](../docs/bathroom_profile.md).
C076 uses `(area AND height) OR volume`; C077/C078 use the occupancy breakpoint after four people.
Main rooms are living/bedroom; estimated habitable area includes living, bedroom, kitchen and bathroom,
and excludes storage, balcony and outside. These are zero-thickness polygon estimates.

Coordinates are drawing pixels, not metres. The current scenario uses the paper/code-derived
18/512 = 0.03515625 m per drawing pixel, two occupants and 2.5 m ceilings. Height, furnishings and
wall thickness remain scenario assumptions; the former 0.025 m/pixel profile remains historical.
Set scale to `null` and provenance to `missing` to obtain `unknown` dimensionful checks.
Furniture, fixtures, doors, physical wall thickness and headroom are not generated; these scores
are conditional feasibility measurements, not verified building compliance.

CPU audit energy is the weighted sum of measurable normalized violations plus a separate validity penalty.
GPU guidance uses that formula outside C036/C037/C041 and one joint bathroom placement penalty instead of their three severity terms.
The latter ranks near misses without turning an unresolved search into a proven failure.
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
it not applicable. Neither counts as valid. Supporting C034/C079/C080 and individual inapplicable checks are excluded.
The overall-valid rate divides confirmed valid samples by **all selected designs**.
This covers eleven assessment-derived checks under declared assumptions, not every constraint in the assessment.

Reports retain structural validity, per-check compliance, coverage, unknown fraction and assumptions
as diagnostics. JSON uses `overall_valid_rate`, `overall_validity_counts` and `structural_valid_rate`;
legacy `valid_count` and `validity_rate` remain structural aliases. Figures show both overall and structural status.
Rebuild reports from saved evaluations without loading models or generating designs:

```bash
python -m test_time.evaluate --experiment_dir test_time/outputs/pilot_check/batch_1_seed_1000
```

Measured runtime includes sampling and checking; loading/rendering/export time is recorded separately.
The chosen budget uses measured denoiser states per output, with a 1% tolerance for approximate
matches. Runtime similarity within 20% is reported separately and is not the budget criterion.
No RPLAN-relative FID/KID is claimed without reference data.

References: [GSDiff](https://arxiv.org/abs/2408.16258), [SVDD](https://arxiv.org/abs/2408.08252),
and the supplied `Technical_assessment_TTC.pdf`, Appendix A.
# Late extensions: hard feasibility and repair-aware SVDD

The separate launcher keeps the original experiments unchanged. Activate the existing
`gsdiff` environment on the node, pull the repository, then preview the commands:

```bash
bash test_time/launch_extensions.sh
```

Run two smoke-test attempts per method before a full experiment:

```bash
EVAL_OUTPUT=test_time/outputs/extensions_smoke_001 \
  bash test_time/launch_extensions.sh --smoke --run
```

The smoke run uses two proposals per decision and two rollouts/terminal proposals
for hard feasibility. It preserves the 1,000-step model schedule and the configured
guidance window (the final 10% in the default bathroom configuration);
it tests execution and outputs, not whether the method reliably finds successes.

Run the chosen configurations (100 attempts per method, seeds 10000–10099):

```bash
EVAL_OUTPUT=test_time/outputs/extensions_001 \
  bash test_time/launch_extensions.sh --run
```

Set `TRIALS_PER_RUN`, `EVAL_SEED`, `INFERENCE_BATCH_SIZE`, `WORKERS`, `DEVICE`, or `PYTHON`
to override launcher defaults. To launch just one method, set
`METHODS=hard_h_mc` or `METHODS=svdd_repair_aware`. Existing output directories are
never overwritten; choose a new `EVAL_OUTPUT` for another run. Methods execute
sequentially: **repair-aware SVDD first, then hard-h MC**. Both default to an
inference microbatch size of **32** for proposals/rollouts within each attempt.

To overlap CPU checking/export with GPU generation, use two independent workers
on the same GPU (the default remains one):

```bash
WORKERS=2 EVAL_OUTPUT=test_time/outputs/extensions_003 \
  bash test_time/launch_extensions.sh --run
```

This still runs **100 attempts per method**, split into seeds 10000–10049 and
10050–10099. Both repair-aware workers finish before either hard-h worker starts.
Each worker loads its own model and writes its own directory; the parent combines
their attempt ledgers and summaries, including failed attempts. Parallel workers
default to one OpenMP/MKL thread each unless those environment variables are set.
The benefit depends on available CPU cores and GPU contention; this does not
change the rollout or repair budgets. Stop the previous launcher before starting
a replacement, and use a new output root.

Settings are in [`configs/extensions_bathroom_v1.json`](configs/extensions_bathroom_v1.json):

- **`hard_h_mc`:** independent ordinary root prefixes, four proposals, four
  unguided completions per proposal, decisions at 100/50/15 transitions remaining,
  then eight terminal completions. Selection is proportional to verified joint
  success counts. All-zero estimates use a logged fresh ordinary continuation.
  Only a passing terminal graph is returned; failure and budget exhaustion remain
  recorded attempts. The NFE cap is 40,000 per attempt.
- **`svdd_repair_aware`:** four proposals at steps 100 through 1; bounded coordinate
  repair of predicted graphs, followed by selection of the original noisy proposal.
  Final repair is enabled. Default displacement is at most 0.20 m per junction,
  with at most two wall movements, twelve moved junctions, 2.4 m total displacement,
  sixteen inner candidates and 128 final candidates.
  Tensor repair scoring runs on the model device. The final repaired graph is
  checked with the CPU audit; a GPU score is not a compliance certificate.

The faster hard-h default reduces rollouts per proposal from 16 to 4, trading
estimate precision for speed and increasing the chance of an all-zero bank.
Saved per-run `config.json` files remain authoritative for previous and new runs.

Both methods use eleven scored checks and three supporting definitions by default, the same
unconditional checkpoint, and the declared metric scale/height/occupancy. Hard-h
uses the audited CPU checker for every rollout label. No new ablation sweep is
included; repair results measure guidance **plus** final repair together.

Outputs are grouped under the chosen root:

```text
extensions_001/
  comparison.md, comparison.csv
  hard_h_mc/                 # own config.json, run.json, summary.json, report.md
  svdd_repair_aware/          # same structure
    trials.jsonl             # every attempt, including failures without a graph
    candidates.jsonl         # returned graphs only
    events.jsonl             # completed hard-h lookahead/terminal outcomes
    traces/                  # decisions, costs and repair edits
    samples/                 # graphs, check evidence, PNG/SVG; pre-repair and edit overlays too
    lookaheads/              # verified successful hard-h lookahead graphs/evidence
```

With `WORKERS=2`, each method directory instead contains `worker_001/` and
`worker_002/` with those artifacts, plus `worker_001.log`, `worker_002.log`, a
`parallel.json` execution manifest and combined `trials.jsonl`, `summary.json`,
`run.json`, `config.json` and `report.md`. Follow individual progress with:

```bash
tail -f test_time/outputs/extensions_003/svdd_repair_aware/worker_*.log
```

Worker times overlap: summed episode seconds and mean seconds per attempt measure
work/latency, not the elapsed duration of the parallel run. The combined summary
also records elapsed time including setup/export and throughput separately.

Use `trials.jsonl` and the extension summaries for success rates. The old analysis
assumes one selected graph per trial and would omit hard-h failures with no returned
graph. Regenerate the extension comparison without loading models:

```bash
python -m test_time.run_extensions --summarize_root test_time/outputs/extensions_001
```

The configurations are not compute matched. Reports count actual denoiser states,
all sampling/checking/repair time, and unsuccessful attempts. Runtime excludes model
loading and artifact export. Generated output folders remain Git-ignored; download
the whole chosen output root from the node to preserve graphs and audit evidence.
