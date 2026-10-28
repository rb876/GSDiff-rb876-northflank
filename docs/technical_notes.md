# GSDiff: technical notes

[Concise report](technical_report.md)

Implementation details and assumptions supporting the presentation report. Evaluation tables await measured sweep results.

## 1. Model and experimental setting

[GSDiff: Synthesizing Vector Floorplans via Geometry-enhanced Structural Graph Generation](https://arxiv.org/abs/2408.16258) generates a structural graph rather than a raster image. It factorizes generation into node diffusion and conditional edge prediction:

$$
p_{\mathrm{pre}}(G)=p_\theta(V)\,p_\phi(E\mid V),\qquad G=(V,E).
$$

Nodes encode junction coordinates, surrounding room semantics and a background flag. Predicted edges form walls; polygon extraction produces labelled rooms. This provides geometry for room area and placement checks, but does not guarantee that every generated graph is valid.

We use the released **unconditional RPLAN** model. The experiment supplies a residential scenario and constraint objectives, without a prescribed boundary or topology. Unconditional generation lets us study how these objectives change the layout while keeping the same pretrained prior across methods.

The implementation uses **53 padded node slots**, not 53 actual junctions in every design. Each slot has 10 channels: two coordinates, seven surrounding-room semantic indicators and one background flag. The room categories are living room, bedroom, kitchen, bathroom, balcony, storage and outside; a junction can belong to several surrounding categories. Background slots are removed when decoding.

Sampling uses 1,000 cosine-schedule DDPM steps. The [main report's model introduction](technical_report.md#gsdiff-generating-the-geometry-of-a-floorplan) explains the paper's architecture and geometry-enhancement objectives. Both pretrained components remain frozen:

| Component | Checkpoint |
|---|---|
| Node denoiser | `outputs/structure-1/model1000000.pt` |
| Edge predictor | `outputs/structure-2/model_stage2_best_061000.pt` |

Each run saves checkpoint and source hashes. New sampling, checking and analysis code lives in [`test_time/`](../test_time/README.md). The upstream change in [`gsdiff/house_nn2.py`](../gsdiff/house_nn2.py) accepts supplied edge-interpolation randomness for reproducible batching; model parameters and checkpoint shapes are unchanged. There is no training or additional geometry-repair stage.

## 2. Constraint subset and assumptions

We evaluate **six constraints**, spanning accessibility and habitability. They use observable room polygons and labels, supplemented by explicit scenario assumptions. Source IDs refer to Appendix A of the [technical assessment](../Technical_assessment_TTC.pdf).

| ID | Implemented check | How it is checked |
|---|---|---|
| C033 | A bedroom fits a bed and a separate 1.50 m turning circle. | Search bed/circle placements inside the polygon, with no overlap. |
| C035 | Bed clearances: 0.90 m on both long sides and 1.20 m at one short side, or 1.20 m and 0.90 m respectively. | Fit the corresponding clearance envelope; joint success with C033 requires the same bed placement. |
| C036 | An **empty bathroom** fits a 1.50 m turning circle. | Search circle centres; geometric bounds can prove insufficient space. |
| C076 | One main room meets `(area ≥9 m² AND height ≥2.20 m) OR volume ≥20 m³`. | Evaluate both branches for each living room/bedroom. |
| C077 | Sufficient habitable area for the occupants. | Sum eligible areas: at least 14 m²/person for the first four, then 10 m²/person. |
| C078 | Sufficient habitable volume for the occupants. | Area × height: at least 33 m³/person for the first four, then 23 m³/person. |

Three supporting rules define measurements: **C034** supplies bed dimensions, **C079** declares the area estimator, and **C080** determines room-type/headroom exclusions. They contribute no independent reward or compliance result. The machine-readable specification is [`constraints_v2.json`](../test_time/configs/constraints_v2.json); full implementation details and formulas are in [`constraints.md`](constraints.md).

The fixed [scenario](../test_time/configs/scenario_v2.json) assumes:

- **Scale:** 0.025 m per drawing pixel, making the 512-pixel canvas 12.8 m wide. This is an experimental assumption, not recovered RPLAN scale.
- **Occupancy and height:** two people and uniform 2.5 m ceilings. Consequently, C077 requires 28 m² and C078 requires 66 m³.
- **Objects:** a hypothetical 1.40 × 1.90 m bed in either axis-aligned orientation; other furniture, fixtures and door swings are unobserved.
- **Area:** zero-thickness room polygons, including living rooms, bedrooms, kitchens and bathrooms; excluding storage, balconies and outside. Unrepresented wall footprints and other exclusions cannot be verified.

These checks provide both placement feasibility and dimensional deficits. Their scope is limited: GSDiff does not supply explicit door widths/swings, WC or shower positions, navigable routes, levels, or building-services metadata. Shared walls do not establish doorways. Requirements relying on these fields are excluded from the active subset. Reported success means **geometric feasibility under our assumptions**, not verified building compliance.

## 3. From constraints to a scalar reward

Each applicable check returns **pass**, **fail** or **unknown**, together with a severity when measurable. A failed finite placement search is not necessarily proof of impossibility: unresolved cases remain unknown.

Let $\mathcal M(G)$ contain the applicable, measurable scored checks. The implemented energy and reward are

$$
E(G)=100\,\mathbf 1\{G\text{ is structurally invalid}\}
     +\sum_{j\in\mathcal M(G)} w_j v_j(G),
\qquad r(G)=-E(G),\qquad w_j=1.
$$

Structural checks cover graph/geometry defects such as crossing walls, invalid room polygons and room overlaps. Their penalty is one binary term. Supporting rules carry no additional weight.

C033/C035 use binary severity: zero for a witnessed pass, one for an established failure. Dimensional checks use normalized shortfalls:

$$
d(q;b)=\max\!\left(0,\frac{b-q}{b}\right).
$$

For example, 21 m² against the 28 m² requirement gives severity 0.25. C077 uses $d(A;28)$ and C078 uses $d(2.5A;66)$ in this scenario. C076 combines area/height shortfalls with a maximum for its AND branch, then a minimum across the two alternative branches and eligible rooms. C036 uses a radius shortfall when failure is established; GPU guidance uses a geometric upper bound, while CPU reporting uses its reference radius estimate. Passing a threshold earns no extra reward for exceeding it.

**Unknown has no invented numeric severity and is omitted from the sum.** Therefore, zero energy alone does not establish validity. Overall validity requires valid structure and every applicable scored check to pass. Selection separately protects measurement coverage, as described below.

For positive temperature $\tau$, the ideal reward-tilted target is

$$
p_\tau^*(G)=\frac{p_{\mathrm{pre}}(G)\exp\{r(G)/\tau\}}{Z_\tau}.
$$

It favors higher reward while retaining the pretrained prior. Equivalently, it optimizes

$$
\max_q\left[\mathbb E_{G\sim q}[r(G)]
              -\tau\,\mathrm{KL}(q\Vert p_{\mathrm{pre}})\right].
$$

This is the motivation from [SVDD, §§3–4](https://arxiv.org/html/2408.08252v3), not a claim that our approximate sampler realizes this distribution exactly.

### Exact tilting of the two-stage generator

Write $\beta=1/\tau>0$, $w_\beta(G)=e^{-\beta E(G)}$ and

$$
H_t^\beta(z)=\mathbb E_{\mathrm{pre}}[w_\beta(G)\mid z_t=z],
\qquad v_t^\beta(z)=\beta^{-1}\log H_t^\beta(z).
$$

For this derivation, $z_T$ is the initial noise and $z_0=V$ denotes completed nodes. These are mathematical chain indices; the implementation separately executes code steps `999,…,0`, including its final deterministic update.

The expected terminal weight obeys

$$
H_t^\beta(z)=\int K_t(u\mid z)H_{t-1}^\beta(u)\,du,
\qquad
K_t^\beta(z'\mid z)=K_t(z'\mid z)
\frac{H_{t-1}^\beta(z')}{H_t^\beta(z)}.
$$

The second expression is a normalized transition wherever $H_t^\beta(z)>0$. It follows by reweighting the pretrained next-state distribution by the expected terminal weight of each next state.

For the complete path to induce $p_\beta(G)\propto p_{\mathrm{pre}}(G)e^{-\beta E(G)}$, its boundaries must also be weighted:

$$
p_T^\beta(z)=\frac{p_T(z)H_T^\beta(z)}{Z_\beta},
\qquad
p_\phi^\beta(e\mid V)
=\frac{p_\phi(e\mid V)e^{-\beta E(V,e)}}{H_0^\beta(V)}.
$$

Here $H_0^\beta(V)=\mathbb E_{e\sim p_\phi(\cdot\mid V)}[e^{-\beta E(V,e)}]$ includes edge-decoder randomness. Multiplying the initial density, guided transitions and final edge conditional cancels the intermediate $H$ factors, leaving the original path density times $e^{-\beta E(G)}/Z_\beta$.

Our sampler retains the original initial-noise and terminal edge procedures. Its interior weights use the approximation

$$
\widehat H_t^\beta(z)=e^{-\beta E_{\mathrm{GPU}}(\widehat G_t(z))},
\qquad
\widehat v_t(z)=-E_{\mathrm{GPU}}(\widehat G_t(z)),
$$

followed by finite-proposal selection and coverage guards. The current zero-temperature setting takes an argmax of candidate reward; it is not an exact realization of the finite-temperature construction above. GPU guidance and CPU reporting also use different geometric estimates in some checks.

This is energy-based guidance. A hard-feasibility interpretation would additionally require $E(G)=0$ exactly for fully valid graphs and positive pretrained probability of that event. Our energy omits unknown severities, so its zero set does not currently establish that equivalence.

## 4. Methods and SVDD adaptation

**Baseline** runs the original frozen generation pipeline. **Best-of-N** generates complete independent candidates, ranks them using structural validity, measurement coverage and reward, and retains one design. It is a simple inference-time search baseline.

We choose **SVDD-PM** because it uses reward evaluations without reward gradients, generator fine-tuning or a trained value network. This suits discrete room extraction and geometric feasibility tests. The source is [*Derivative-Free Guidance in Continuous and Discrete Diffusion Models with Soft Value-Based Decoding*](https://arxiv.org/abs/2408.08252).

The ideal soft value of a noisy state $z_t$ is

$$
v_t(z_t)=\tau\log\mathbb E_{\mathrm{pre}}
             [\exp\{r(G)/\tau\}\mid z_t].
$$

The PM approximation replaces this future-output calculation with a reward on the predicted clean output. Our graph-specific implementation uses

$$
\widehat v_t(z_t)=r_{\mathrm{GPU}}\!\left(
 D_\phi\left[\operatorname{Project}(\widehat z_0(z_t,t))\right]\right),
$$

where projection discretizes semantics/background and bounds coordinates; $D_\phi$ predicts edges and extracts rooms. Reward of this clean estimate is not generally expected terminal reward.

The implemented algorithm is:

1. Initialize an independently seeded Gaussian node state and traverse all 1,000 reverse steps, indexed `999,…,0`.
2. At an unguided step, take the ordinary DDPM transition. At a guided step, compute the parent clean prediction once and sample $K=10$ noisy next-state proposals.
3. Predict clean nodes for each proposal at the next timestep, decode complete graphs and compute GPU rewards. Sibling proposals share edge-decoder randomness to reduce comparison noise.
4. Keep structurally valid candidates with finite reward and at least one measurable check. Relative to the first such candidate, forbid new unknown constraint IDs; then prefer the smallest unknown count.
5. Among eligible proposals, select with probability proportional to $\exp(\widehat v/\tau)$ when $\tau>0$. The default **temperature zero** selects the highest reward. If none is scorable, retain proposal zero.
6. Continue from the selected **noisy proposal**, not its clean prediction. Decode the terminal graph and evaluate it with the CPU reference checker.

Full SVDD guides `t=999,…,1`; late SVDD guides `t=200,…,1`. Both perform the deterministic unguided `t=0` step. Finite proposals, the PM surrogate, coverage rules and greedy selection make this approximate inference-time optimization. It does not integrate over future edge predictions or explicitly tilt the edge conditional.

## 5. Engineering difficulties and limits

**Reward throughput.** Each guided step scores ten decoded graphs. CPU polygon operations and placement searches can stall the GPU between denoiser calls. The guidance path therefore batches edge inference, room extraction, geometry checks and rewards using PyTorch tensors on the GPU. Final saved-output metrics use the CPU reference checker. These evaluators have finite-search and severity differences; their numerical rewards should not be treated as identical.

**Geometry and uncertainty.** Bedroom searches use coarse/refined grids of 0.25/0.05 m and a bounded search budget. Successful placements are witnesses; inconclusive searches remain unknown. Early clean predictions can be structurally poor, limiting the usefulness of their rewards. Binary bedroom penalties also provide little signal for incremental improvement before feasibility is reached.

**Objective and diversity.** Equal weights are an explicit scalarization choice. At the assumed height, C077's 28 m² already implies C078's 66 m³, so these checks partly duplicate the same objective. Greedy selection may concentrate layouts; diversity and baseline-relative distribution diagnostics are therefore necessary alongside validity.

**Reproducibility and compute.** Per-design random streams and shared edge randomness support reproducible batching. Denoiser-state counts, model calls, edge evaluations and runtime are recorded separately. GPU memory use alone does not determine throughput; the measured runtime remains to be reported.

## 6. Evaluation protocol and results to populate

The planned sweep uses **100 selected designs per method**, with paired trial seeds `10000–10099`, one fixed scenario, batch size 4 and maximum inference batch size 10 on an H100 80 GB. Best-of-11 generates 1,100 candidates but contributes 100 selected outputs to the comparison.

For $T=1000$ reverse steps, $K=10$ proposals and $L$ guided steps, SVDD evaluates $T+KL$ denoiser states per output:

| Method | Guided steps | Denoiser states/output | Baseline multiplier |
|---|---:|---:|---:|
| Baseline | 0 | 1,000 | 1× |
| Best-of-11 | — | 11,000 | 11× |
| SVDD, last 20% | 200 | 3,000 | 3× |
| SVDD, full | 999 | 10,990 | 10.99× |

Best-of-11 approximately matches **full SVDD denoiser work**, within 0.091%. This does not match edge prediction, reward work or runtime. The late/full comparison is a guidance-window ablation with different budgets. Recorded counters must confirm the planned counts.

The primary metric is overall-valid rate over **all selected outputs**, with 95% Wilson intervals. Report per-constraint pass/fail/unknown/N/A counts, a histogram of checks passed per design, measurement coverage and mean measured severity. Constraint success uses all applicable checks in its denominator; measurable-only compliance is reported separately. Unknowns never count as passes.

| Method | Scored subset | Applicability coverage | Overall valid + 95% CI | Constraint success / measurable compliance | Mean measured severity | Structural valid |
|---|---:|---|---|---|---|---|
| Baseline | 6 | Pending | Pending | Pending | Pending | Pending |
| Best-of-11 | 6 | Pending | Pending | Pending | Pending | Pending |
| SVDD, last 20% | 6 | Pending | Pending | Pending | Pending | Pending |
| SVDD, full | 6 | Pending | Pending | Pending | Pending | Pending |

| Method | Unique / near-duplicate fraction | Mean pairwise layout distance | Distribution shift from baseline | Seconds/output |
|---|---|---|---|---|
| Baseline | Pending | Pending | Reference | Pending |
| Best-of-11 | Pending | Pending | Pending | Pending |
| SVDD, last 20% | Pending | Pending | Pending | Pending |
| SVDD, full | Pending | Pending | Pending | Pending |

Diversity uses canonical graph uniqueness and semantic-mask distances. Distribution diagnostics compare room-type frequencies and geometric/count distributions with baseline. They use structurally valid layouts with usable geometry and report eligible counts. Without RPLAN reference data, they do not establish dataset fidelity. Runtime is amortized generation time, excluding model loading and image export. Exact definitions are in [`analysis.md`](analysis.md).

**Results discussion — pending:** identify the hardest constraint and sole-failure counts; compare full SVDD with Best-of-11; assess late-guidance validity/runtime tradeoffs and any loss of diversity. Add representative success, failure and unknown examples with images, constraint IDs, measurements and assumptions. All selected designs remain available in the gallery. One hundred designs per method supports an initial comparison; intervals and additional seeds are needed before strong performance claims.

## 7. Assessment alignment and deliverables

The [assessment, pp. 2–3 (§§2.2, 2.6–2.8)](../Technical_assessment_TTC.pdf), asks for inference-time improvement of a released GSDiff model, a machine-readable constraint subset, comparable baseline/guided experiments, a compute multiplier, vector outputs with PNG/SVG views, and a report covering assumptions, metrics, an ablation, failure cases and limitations. This document follows that structure. Breadth is limited to six scored checks across two families; the three supporting rules are not counted as extra constraints.

The required comparison covers subset size, applicability, compliance, severity, structural validity and distribution preservation. Its grading emphasizes reasoning (50%), correctly implemented subset/breadth (25%) and constraint respect (25%). It specifies no fixed sample count or mandatory FID/KID metric; RPLAN reference data is optional.

After copying the completed sweep, populate the tables and figures from:

```bash
python -m test_time.analyze --experiment_dir test_time/outputs/sweep_001
```

The resulting `analysis/` contains the generated report, CSV tables, PNG/PDF figures and `index.html` gallery, with an annotated PNG/SVG and graph/evaluation JSON for every selected design. Preserve run configurations and hashes with these artifacts. **The remaining submission work is to insert the measured results, annotated examples and evidence-based conclusions.**
