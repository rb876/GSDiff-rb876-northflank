# Analysing a saved GSDiff sweep

Run the analysis after copying the sweep to Kraken, or on the node. It uses saved graphs and
final CPU evaluations: no checkpoint, GPU inference or repeated constraint search is needed.
Use the existing project environment and install the additional plotting dependency:

```bash
python -m pip install -r test_time/requirements-analysis.txt
python -m test_time.analyze --experiment_dir test_time/outputs/sweep_002
```

The default destination is `test_time/outputs/sweep_002/analysis/`. Use `--output_dir PATH` to
choose another destination. Input files remain unchanged. Only completed runs with all planned
selected outputs are included; running, failed or incomplete runs are recorded as excluded.
Each variant and seed batch stays separate, with no pooling across batches.

## Re-audit with the improved CPU checker

To apply `cpu_bedroom_audit_v2` to the saved selected designs, then regenerate their analysis:

```bash
python -m test_time.reaudit \
  --experiment_dir test_time/outputs/sweep_002 \
  --output_dir test_time/outputs/sweep_002_cpu_audit_v2
python -m test_time.analyze \
  --experiment_dir test_time/outputs/sweep_002_cpu_audit_v2
```

The re-audit runs on CPU and writes a separate directory; original samples and evaluations
remain untouched. Without `--output_dir`, it uses the sibling directory `<sweep>_cpu_audit_v2`.
Only originally selected outputs are checked: generation and candidate selection are unchanged,
and rejected Best-of-N candidates are not copied or reconsidered. Saved-candidate statistics in
the audit therefore describe this selected population, not the original full candidate pool.

Open `AUDIT.md` for changes and `audit.json` for audit timings and provenance. Per-run
`evaluation_audit` metadata records the checker version, source hashes and unchanged-selection
policy; it is retained in the analysis. Generation times and compute counters are copied from
the original run and **exclude this new offline audit**. Reward/guidance settings remain the
original generation settings. Reports keep original and revised checker versions in separate
comparison groups, even when their generation settings match.

The completed 400-design re-audit is summarized in
[experiment 007](../test_time/experiments/007_cpu_checker_audit/REPORT.md).

## What to open

```text
analysis/
  REPORT.md                 # method comparison, figures, failures and limitations
  index.html                # filterable gallery of every selected design
  analysis.json             # full metrics, saved configuration and diagnostic details
  denoiser_comparison.json   # recorded Best-of-N / SVDD denoiser budgets
  figures/                  # six figures, each exported as PNG and PDF
  tables/
    methods.csv
    constraints.csv
    paired_validity.csv
    diversity.csv
    distribution_shift.csv  # when a compatible baseline is available
  samples/                  # annotated PNG/SVG, graph JSON and evaluation JSON
```

Open `REPORT.md` for the results and `index.html` in a browser to inspect samples. The gallery
filters by method, overall status, constraint and constraint status. Every selected design has
an image and links to the saved graph and exact check results. Constraint details show the full
name, requirement and plain-language reason; raw values sit under “Recorded measurements”.
Unresolved is the displayed name for saved unknown outcomes. The report includes the first valid, invalid and unknown trial
per method when present; these examples are chosen deterministically rather than by appearance.

Only one selected output per trial contributes to method comparisons. For Best-of-11, that
means 100 selected designs from a 100-trial run, not all 1,100 generated candidates.

## Figures

For a brief explanation of each plot, see the [HackMD-ready figure guide](figure_guide.md).

Figures use a compact paper style: white backgrounds, serif labels, thin axes and pale,
hatched bars. Method styling stays consistent: blue for baseline, orange for Best-of-N,
grey for late SVDD and green for full SVDD. Outcome plots separately identify pass, fail,
unknown and N/A in their status legends. Captions and denominator explanations appear
below the images in `REPORT.md`, keeping the exported figures uncluttered.

Rerun the analysis command after updating the plotting code to regenerate the six PNG/PDF
figures from saved results. This refreshes the analysis exports without running generation
or changing the saved samples. Use the vector PDFs when placing figures in the report.

| File stem | What it answers |
|---|---|
| `overall_validity` | What fraction of designs passes every applicable check and structural validity? Shows 95% intervals and the valid/invalid/unknown/N/A breakdown. |
| `constraint_outcomes` | Which checks are difficult? Shows pass/fail/unknown/N/A for each constraint and method; an asterisk marks the largest confirmed failure count. |
| `passed_constraints` | How many scored constraints does each design satisfy? Grouped, hatched bars at integer counts compare methods without overlapping distributions. Each method is normalized by its selected-output count. |
| `constraint_severity` | How severe are measured violations? Aligned method panels show normalized severities, including zero-severity passes and excluding unresolved checks. |
| `layout_distributions` | Does guidance change room counts or the sum of room areas relative to baseline? Aligned method panels use structurally valid layouts with usable room polygons. |
| `compute` | How many seconds and denoiser NFE (number of function evaluations) are needed per selected output? These are separate budgets. |

## Metrics and denominators

The current profile has **six scored constraints and three supporting measurement rules**.
Analysis reads the saved registry so older profiles retain their original definitions.

| Metric | Definition |
|---|---|
| Overall-valid rate | Structurally valid designs where every applicable scored check passes, divided by all selected designs. Unknown and N/A overall outcomes are not successes. |
| Structural-valid rate | Designs passing the saved graph checks, divided by all selected designs. |
| Constraint success | Passed scored checks divided by applicable scored checks. Unknown checks stay in this denominator. |
| Measurable-only compliance | Passed checks divided by measured pass/fail checks. Reported separately because it excludes unknowns. |
| Applicability coverage | Applicable scored checks divided by selected designs × configured scored constraints. |
| Measurement coverage | Measured pass/fail checks divided by applicable scored checks. |
| Unknown-check rate | Unresolved checks divided by applicable scored checks. |
| Mean measured severity | Mean normalized severity over measured scored checks, including passing checks at zero. Unknowns have no invented severity. |
| Sole-failure count | Designs with valid structure where this check fails and every other applicable scored check passes. |

For example, five passes and one unknown give **83.3% constraint success**, **100% measurable-only
compliance**, and an **unknown overall outcome**. Supporting rules do not add passes or failures.
Zero energy is not proof of validity because unknown checks add no numeric penalty.

Overall-validity intervals use the 95% Wilson method. They describe sampling uncertainty under
the saved scenario, not uncertainty about real-world compliance. The paired table records
both-valid, only-one-valid and neither-valid counts, plus the validity-rate difference on shared
trial seeds. It uses the trial seed rather than the winning Best-of-N candidate's seed; it does
not perform a significance test. Different settings or seed sets are not paired.

## Diversity and distribution preservation

These diagnostics use **structurally valid designs with usable geometry**. The tables show the
eligible and excluded counts; do not compare diversity without checking those populations.

- **Unique graph fraction:** distinct saved graphs divided by eligible graphs. Canonicalization
  ignores node order and edge direction and rounds node values to six decimals.
- **Mean semantic-layout distance:** average `1 − IoU` across all pairs of room-label masks on
  a 64×64 raster of the fixed 512-pixel drawing canvas. Background and outside label 6 are excluded.
- **Near-duplicate fraction:** fraction of raster-eligible designs with at least one other design
  at distance ≤0.05. This is a fraction of designs, not pairs; fewer than two eligible rasters
  leaves pairwise statistics unknown.
- **Room-type Jensen–Shannon divergence:** change in pooled room-type frequencies relative to
  a compatible generated baseline; zero means identical frequencies. This is in the report/JSON.
- **Wasserstein distances:** shifts in room, junction and edge counts and summed room-polygon
  area relative to baseline. Distances retain the original units; area uses drawing pixels² and
  excludes outside label 6. These rows are exported to `distribution_shift.csv`.

The raster metric has finite resolution, clips to the drawing canvas, does not separately
encode walls and does not align translations, rotations or scale. Large distance is not itself
evidence of good designs. Distribution diagnostics compare with generated baseline samples;
they do not measure fidelity to RPLAN. No FID/KID is claimed without reference data.

Runtime and NFE multipliers use a compatible baseline. Denoiser NFE counts each evaluated
sample within a batch, including candidate trajectories and SVDD lookahead, per selected output;
it excludes edge-model and reward evaluations. At default settings, Best-of-11
uses 11,000 NFE per output versus 10,990 for full SVDD; the recorded-counter
comparison marks differences within 1% as approximately matched. Late SVDD uses a smaller budget.
Equal denoiser counts do not equalize edge prediction, reward checking or elapsed time. Timing
is amortized across a generation batch and excludes model loading and rendering/export.

## Connection to the assessment

`Technical_assessment_TTC.pdf`, pages 2–3 (§§2.2 and 2.6–2.8), requires a shared checkpoint,
scenario distribution and output representation; a compute multiplier; and a comparison table
covering subset size, applicability, compliance, severity, structural validity and distribution
preservation. It also requests vector graphs with PNG/SVG views, annotated successful/failed
examples, an ablation, assumptions, failure cases and limitations. The full-versus-last-20%
comparison is the guidance-window ablation, with its different budget shown explicitly.

The assessment specifies no fixed sample count or mandatory FID/KID metric. Confidence intervals,
duplicate diagnostics and paired outcomes are additional descriptive evidence. The assumed
0.025 m/pixel scale, two occupants, 2.5 m ceiling and finite placement-search limitations remain
part of the experiment; these outputs do not establish legal compliance.
