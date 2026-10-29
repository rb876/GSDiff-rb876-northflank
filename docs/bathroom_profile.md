# Joint bathroom feasibility: C036, C037 and C041

The default profile checks **eleven PDF constraints**. C036/C037/C041 now share one hypothetical bathroom arrangement: a turning circle, toilet, basin, door swing and lateral transfer space. This tests coupled geometry that an empty-room circle check cannot establish. Whether it is harder for Best-of-11, or benefits SVDD, remains an empirical question.

## PDF requirements and declared objects

| Item | Requirement or assumption | Source |
|---|---|---|
| C036 | At least one bathroom contains a 1.50 m diameter turning circle | Assessment, Appendix A, p. 5 |
| C037 | At least one WC provides 0.80 × 1.30 m lateral transfer space outside door swing | Assessment, Appendix A, p. 5 |
| C041 | Circle overlap ≤0.25 m with door swing OR ≤0.15 m beneath a basin; only one overlap | Assessment, Appendix A, p. 6 |
| Combined bathroom/WC | One shared room and arrangement witnesses these checks | Declared scenario; the PDF does not require a combined room |
| Toilet | 0.40 m wide × 0.70 m deep | Declared fixture footprint |
| Basin | 0.60 m wide × 0.50 m deep; unobstructed space underneath | Declared fixture/mounting assumption |
| Door | 0.80 m wide; inward 90° swing | Declared door geometry |

**C041 is evaluated only for the selected bathroom turning circle.** The PDF wording concerns turning spaces generally; this implementation does not certify bedroom door/fixture overlaps.

GSDiff generates walls and room labels, **not these objects**. A pass means that a placement was found under the declared assumptions. It does not establish installed equipment, plumbing, a doorway to an adjacent room, or an accessible route through the dwelling. Shower, bathtub and door-approach requirements are outside this profile.

The scenario keeps the calibrated scale **18/512 m per drawing pixel**, two occupants, 2.5 m ceiling height and the previously declared ventilation capacities. The fixture assumptions are identical for baseline, Best-of-N, SVDD and the extensions.

## One bounded search, one shared arrangement

The tensor search considers **432 axis-aligned wall-attached templates**: four quarter-turn orientations × two reflections × three WC positions × three basin positions × three door positions × two hinge sides. Positions are the beginning, midpoint and end of the available wall interval. A WC and its lateral transfer rectangle lie at the back wall, a basin lies at a side wall, and an inward door lies at the opposite wall. Each template uses the same **49 circle centres** on a 7 × 7 grid over the room bounding box inset by the 0.75 m radius. This is a restricted furnishing family, not a search over every possible bathroom.

Every candidate is checked for complete room containment and geometric interference. The WC and basin are solid objects. The door sweep must avoid these objects and the transfer zone. The turning circle avoids the WC and uses only the basin/door overlap permitted by C041. **The transfer rectangle and turning circle may overlap each other** because they represent free space.

One candidate is selected jointly across the three rules. Independently finding a circle in one arrangement and transfer space in a different arrangement cannot establish joint success. The same saved placement budget is applied to every method, including rescoring the original Best-of-11 bank.

- **Pass:** a shared geometric witness satisfies the check; the CPU audit verifies positive witnesses independently.
- **Fail:** a necessary geometric bound establishes that the required placement is impossible, or the required bathroom is absent.
- **Unknown:** required scenario data are missing, or no verified shared arrangement was found in the finite bank. Other unsearched layouts may work.

A search miss must not be reported as a proven furnishing impossibility. Report unknown rates next to pass and failure rates.

## Overlap-depth convention

The PDF gives overlap limits without a coordinate-level algorithm. We explicitly formalize them as follows:

- **Door:** the swing is a quarter disk $D$. If $s_D(c)$ is the signed distance of circle centre $c$ to $D$ (negative inside), its penetration depth is $d_D=[r-s_D(c)]_+$, where $r=0.75$ m.
- **Basin:** $d_B$ is the largest depth by which the circle enters the basin footprint past its front edge, restricted to the basin's width and capped at its depth. This assumes the declared under-basin region is unobstructed.
- **C041 passes** only when $d_D\leq0.25$, $d_B\leq0.15$, and at most one of $d_D,d_B$ is positive, within numerical tolerance.

These lengths are neither intersection areas nor bounding-box overlaps. GPU checks use an analytic quarter-disk geometry. The independent CPU audit uses a conservative circumscribed sector for door containment/collision checks and independently recomputes overlap depths. The sector drawn in figures is a sampled outline for display. CPU/GPU agreement is tested, but a borderline positive tensor witness can be rejected conservatively by the audit and remain unknown.

## Guidance and final reporting

For a candidate completion $a$ in bathroom $b$, the search computes

$$
\ell(b,a)=\ell_{\mathrm{fixtures}}+\ell_{\mathrm{transfer}}+
\ell_{\mathrm{circle}}+\ell_{\mathrm{overlap}}.
$$

The fixture term sums containment and solid/door collision depths, normalized by the turning radius. The transfer term sums containment and basin/door collision depths, normalized by its 0.80 m width. The circle term adds room-boundary and WC-clearance deficits, normalized by its radius. A concave intrusion into a rectangle receives a small positive penalty even if all four corners lie inside; a corner-only fit cannot earn a pass. The overlap term is

$$
\ell_{\mathrm{overlap}}=
\max\left\{
\frac{[d_D-0.25]_+}{0.25},\,
\frac{[d_B-0.15]_+}{0.15},\,
\min\left(\frac{d_D}{0.25},\frac{d_B}{0.15}\right)
\right\}.
$$

The last term penalizes using both overlaps, even when each separately meets its depth limit. The search gives a fully feasible candidate priority, then minimizes the shared shortfall; remaining ties follow the fixed candidate order. With numerical tolerances omitted, the minimum is

$$
P_{\mathrm{bath}}(G)=\min_{b,a}\ell(b,a).
$$

GPU selection substitutes this **single joint penalty** for the three bathroom severities:

$$
E_{\mathrm{GPU}}(G)=100\mathbf1\{\mathrm{invalid}\}
+\sum_{j\in\mathcal M(G)\setminus\{\mathrm{C036,C037,C041}\}}v_j(G)
+P_{\mathrm{bath}}(G),\qquad r=-E_{\mathrm{GPU}}.
$$

The penalty provides graded feedback without turning an unresolved search into a failed regulation. Existing structural-validity and unknown-coverage filtering still applies. Best-of-N and SVDD use the same score and search budgets; the bathroom penalty is included once.

Final CPU metrics retain three separately named checks, a shared witness and the usual measured regulatory energy. The GPU search penalty is recorded separately. Final energy is therefore not interchangeable with the guidance objective. Hard-feasibility rollouts continue to use the explicit all-checks-pass predicate, not zero energy.

## Figures and inspection

Each saved sample keeps readable explanations for C036/C037/C041 and the shared placement coordinates. Sample SVG/PNG images show a bathroom inset with the WC, basin, transfer rectangle, door swing and turning circle. A proposal without verified joint success is labelled **unverified**. Fixture dimensions are explicitly described as assumptions.

Per-constraint plots include C037 and C041 in the bathroom group. Overall success still requires all eleven applicable scored checks to pass. Do not combine this profile's percentages with the earlier empty-bathroom/ventilation profile.

## Recheck baseline and reselect Best-of-11

On the node, use the original sweep containing **all 1,100 Best-of-11 candidates**, rather than an audit folder containing only the 100 previous winners:

```bash
cd /workspace/GSDiff
source /workspace/venvs/gsdiff/bin/activate
python -m pip install -r test_time/requirements-analysis.txt
python -m test_time.recalibrate \
  --experiment_dir test_time/outputs/sweep_002 \
  --output_dir test_time/outputs/bathroom_001/controls \
  --scenario test_time/configs/scenario_bathroom_v1.json \
  --constraints test_time/configs/constraints_bathroom_v1.json \
  --device cuda:0 --inference_batch_size 32
python -m test_time.evaluate --experiment_dir test_time/outputs/bathroom_001/controls
python -m test_time.analyze --experiment_dir test_time/outputs/bathroom_001/controls
```

This reuses unguided generated graphs, reselects Best-of-11 with the new score and CPU-checks selected outputs. It does not rerun diffusion or overwrite the original sweep. Rescoring old guided outputs cannot establish how guidance performs under the new objective: that comparison needs new guided runs.

## Add Best-of-3 for the late-SVDD comparison

Late SVDD with K=10 and 200 guided steps uses **3,000 denoiser NFE/output**. Its matched control is **Best-of-3**. Best-of-11 remains the control for full SVDD (11,000 versus 10,990 NFE/output).

Use the already rescored bank above. Retain original candidate indices **0, 1 and 2 in every trial**, apply the same selection rule to their saved GPU scores, and CPU-audit the selected design. This keeps all 100 outputs. The prefix is fixed independently of validity, reward or the previous winner.

```bash
python -m test_time.reselect \
  --source_run test_time/outputs/bathroom_001/controls/batch_1_seed_10000/best_of_n \
  --output_dir test_time/outputs/bathroom_001/controls/batch_1_seed_10000/best_of_n_late \
  --n 3
python -m test_time.evaluate --experiment_dir test_time/outputs/bathroom_001/controls
python -m test_time.analyze --experiment_dir test_time/outputs/bathroom_001/controls
```

The separate `best_of_n_late` directory adds Best-of-3 to tables, figures and the gallery while preserving Best-of-11. No new model inference is performed. The reported 3,000 NFE is the equivalent generation budget of the retained prefix; the original 11-candidate generation cost remains in provenance. Standalone Best-of-3 runtime and batched forward-call counts remain unknown. Because Best-of-3 is nested within Best-of-11, their trial outcomes are paired rather than independent.
