# Constraint-guided floorplan generation with SVDD

We adapt **Soft Value-Based Decoding (SVDD)** to guide a frozen GSDiff model using geometric constraints. At each guided step, we compare possible denoising moves through their predicted floorplans and continue the most promising eligible branch. The experiment tests whether this improves feasibility over ordinary sampling and Best-of-N at comparable denoiser cost.

## GSDiff: generating the geometry of a floorplan

Hu et al.'s **GSDiff: Synthesizing Vector Floorplans via Geometry-enhanced Structural Graph Generation** represents a floorplan through shared wall junctions and wall segments. Its central contribution is to generate this vector structure directly, with room semantics attached to the graph. [Paper](https://arxiv.org/abs/2408.16258)

### Two pretrained components, one generator

For a graph $G=(V,E)$, generation factorizes as

$$
p_{\mathrm{pre}}(G)=p_\theta(V)\,p_\phi(E\mid V).
$$

1. **Node generation:** a Transformer diffusion model generates junction coordinates, surrounding-room semantic attributes and background flags.
2. **Edge prediction:** a second Transformer predicts wall connections between node pairs. Room extraction then identifies closed polygonal loops and assigns room labels.

```mermaid
flowchart LR
    A["Gaussian noise"] --> B["Node diffusion<br/>Junctions + semantics"]
    B --> C["Edge Transformer<br/>Wall connections"]
    C --> D["Room extraction<br/>Labelled polygons"]
```

### What makes it geometry-enhanced?

The paper introduces a **mixed-base alignment loss** to penalize junction misalignment, and **interpolation self-supervision** to improve edge prediction using random intermediate points. These training objectives encourage geometric consistency. The paper also supports topology conditioning through a room-adjacency graph and boundary conditioning through a prescribed exterior contour. [Method, Figures 1–3](https://arxiv.org/pdf/2408.16258)

### Our use of the model

We use the released **unconditional RPLAN** checkpoints because our scenario prescribes neither topology nor boundary. Every comparison shares the same frozen networks and output representation; the original training losses are not optimized during our experiments.

The resulting polygons provide areas and regions for placement checks. Our output schema has no explicit doors, fixtures or ceiling heights, and metric scale remains an assumption. Thus, we evaluate the observable geometric subset rather than infer missing building attributes.

This architecture determines where guidance must act: **we change node-sampling decisions, but score decoded graphs**. Edge prediction and room extraction introduce discrete decisions, making derivative-free SVDD suitable. Instability in early decoded predictions motivates comparing full-trajectory guidance with late guidance. The method must still establish structural validity and constraint satisfaction through final evaluation.

## SVDD-PM: search within the denoising trajectory

**Best-of-N** spends its budget on independent, complete designs and ranks them afterward. **SVDD** repeatedly branches from the current noisy state, using intermediate scores to decide where to continue. Each output retains one active trajectory; the other proposals are discarded at every guided step.

### The energy-tilted target

Our experiment uses **energy-based SVDD-PM**. Let $E(G)$ be the constraint energy defined below and $r(G)=-E(G)$. The [SVDD formulation](https://arxiv.org/html/2408.08252v3) motivates the target

$$
p_\beta(G)=\frac{p_{\mathrm{pre}}(G)e^{-\beta E(G)}}{Z_\beta},
\qquad \beta=1/\tau>0.
$$

This is the distribution minimizing expected violation while penalizing departure from the pretrained prior:

$$
p_\beta=\arg\min_q\left\{
\mathbb E_{G\sim q}[E(G)]
+\frac{1}{\beta}\mathrm{KL}(q\Vert p_{\mathrm{pre}})
\right\}.
$$

Larger $\beta$ strengthens the preference for low energy. Finite $\beta$ still permits violations.

### From final energy to a guided transition

Energy is defined on completed graphs, whereas diffusion operates on noisy nodes. The missing link is the expected terminal weight:

$$
H_t^\beta(z)=\mathbb E_{\mathrm{pre}}\!\left[e^{-\beta E(G)}\mid z_t=z\right],
\qquad
v_t^\beta(z)=\frac{1}{\beta}\log H_t^\beta(z).
$$

$H_t^\beta$ averages over the remaining denoising and edge-generation randomness; $v_t^\beta$ is the **soft value**. Given the pretrained transition $K_t(z'\mid z)$, the ideal reweighted transition is

$$
K_t^\beta(z'\mid z)
=K_t(z'\mid z)\frac{H_{t-1}^\beta(z')}{H_t^\beta(z)},
\qquad
H_t^\beta(z)=\int K_t(u\mid z)H_{t-1}^\beta(u)\,du.
$$

The ratio favors states with better expected terminal weights; the recursion makes it normalized wherever $H_t^\beta(z)>0$. These are ideal conditional transitions. Exact sampling of the full tilted process also requires consistent initial-noise and final-edge weighting, detailed in the [technical notes](technical_notes.md#exact-tilting-of-the-two-stage-generator).

### The posterior-mean approximation

A noisy node state cannot directly support room-area or clearance checks. For each proposal $z$, we predict clean nodes, discretize their semantics, predict edges and extract rooms:

$$
\widehat G_t(z)=D_\phi\!\left[\operatorname{Project}
    \bigl(\widehat z_{\mathrm{clean}}(z,t)\bigr)\right],
\qquad
\widehat H_t^\beta(z)=e^{-\beta E_{\mathrm{GPU}}(\widehat G_t(z))}.
$$

Thus $\widehat v_t(z)=\beta^{-1}\log\widehat H_t^\beta(z)=-E_{\mathrm{GPU}}(\widehat G_t(z))$. Here $D_\phi$ completes the graph. **One clean prediction and graph decode replace the expectation over future completions.** This needs neither rollouts, reward gradients nor a trained value network. The score is not expected terminal reward; early predictions can be unreliable, and GPU geometry is a proxy for the final CPU checker.

### Algorithm: one guided step

Given noisy nodes $z_t$ and $K=10$:

1. **Propose.** Compute the parent clean prediction once. Reuse it to draw $K$ noisy next states from the unchanged DDPM transition, each with independent Gaussian noise: $z_{t-1}^{(k)}\sim p_{\mathrm{pre}}(\cdot\mid z_t)$.
2. **Predict and score.** For each proposal, predict its clean nodes at timestep $t-1$, complete the graph and compute $\widehat v_{t-1}(z_{t-1}^{(k)})$. Sibling proposals share edge-decoder randomness to reduce comparison noise.
3. **Filter.** Require valid structure, finite reward and at least one measurable check. Relative to the first scorable proposal, forbid new unknown constraint IDs; then retain candidates with the fewest unknowns. Call this eligible set $\mathcal E_t$.
4. **Select.** For positive temperature, approximate the reweighted transition by sampling among eligible candidates:

$$
\Pr(k\mid\mathcal E_t)=
\frac{\widehat H_{t-1}^\beta(z_{t-1}^{(k)})}
{\sum_{\ell\in\mathcal E_t}\widehat H_{t-1}^\beta(z_{t-1}^{(\ell)})}
\propto e^{-\beta E_{\mathrm{GPU}}(\widehat G_{t-1}(z_{t-1}^{(k)}))}.
$$

5. **Continue.** Our default $\tau=0$ chooses the highest-scoring eligible proposal. Retain its **noisy state**, discard the other branches and resume denoising. If none is scorable, retain proposal zero.

The clean graph supplies the score; the noisy proposal supplies the next diffusion state. Replacing it with the clean estimate would skip the intended remaining denoising process.

**Guidance window.** Full SVDD applies this loop at code steps `999,…,1`; late SVDD applies it at `200,…,1`. Outside the window, take one ordinary DDPM transition. Both still execute all 1,000 steps, including deterministic, unguided step `0`, before final graph decoding and CPU evaluation.

Finite proposals, clean-estimate rewards, coverage filtering and the greedy default make our implementation **approximate inference-time optimization**. Temperature zero uses a deterministic argmax; it does not exactly sample the feasibility-conditioned model.

## Geometry defines the reward

We score six accessibility/habitability checks from Appendix A of the [assessment](../Technical_assessment_TTC.pdf):

| ID | Geometric requirement |
|---|---|
| **C033** | A bedroom fits a bed and a separate **1.50 m diameter** turning circle. |
| **C035** | Bed clearances: **0.90 m** on both long sides + **1.20 m** at one short side, or **1.20 m + 0.90 m**. Joint success with C033 requires the same bed placement. |
| **C036** | An **empty bathroom** fits a **1.50 m diameter** turning circle. |
| **C076** | One living room/bedroom satisfies **(area ≥9 m² AND height ≥2.20 m) OR volume ≥20 m³**. |
| **C077** | Habitable area ≥**28 m²** for two occupants. |
| **C078** | Habitable volume ≥**66 m³** for two occupants. |

**Assumptions:** 0.025 m/drawing pixel, two occupants, uniform 2.5 m ceilings, a hypothetical axis-aligned 1.40 × 1.90 m bed and zero-thickness walls. Scale and height are supplied. Doors, swings, fixtures and navigable routes are unobserved; requirements needing them are excluded. Supporting rules C034/C079/C080 define measurements without adding compliance points.

For applicable, measurable checks $\mathcal M(G)$, equal-weight scalarization gives

$$
E(G)=100\,\mathbf 1\{G\text{ structurally invalid}\}
      +\sum_{j\in\mathcal M(G)}v_j(G),
\qquad r(G)=-E(G).
$$

Bedroom feasibility uses binary penalties. Dimensional deficits use normalized shortfalls $\max(0,(b-q)/b)$, combined according to each rule. **Unknowns add no numeric severity and never count as passes.** Overall validity requires valid structure and every applicable scored check to pass; zero energy alone is insufficient. The [constraint specification](constraints.md) gives exact formulas and assumptions.

## Evaluation and compute

One parent prediction per reverse step plus $K$ extra predictions at each of $L$ guided steps gives **$T+KL$ denoiser states per output**. The selected state's prediction is recomputed on the next iteration. Best-of-N costs $NT$.

The planned sweep compares **100 selected designs per method**, paired seeds `10000–10099`, on an H100 80 GB:

| Method | Denoiser states/output | Overall valid | Seconds/output |
|---|---:|---:|---:|
| Baseline | 1,000 | — | — |
| Best-of-11 | 11,000 | — | — |
| SVDD, last 20% | 3,000 | — | — |
| SVDD, full | 10,990 | — | — |

*Measured results will populate the empty cells.* Best-of-11 matches full SVDD within **0.091%** in denoiser work; edge/reward costs and runtime remain unmatched. The late/full ablation changes both guidance timing and budget.

Report overall validity with 95% intervals, per-constraint outcomes, coverage, severity and structural validity. Measure diversity through graph uniqueness and semantic-layout distances, and distribution shifts relative to baseline. Annotated examples should explain successful, failed and unresolved designs. See [metric definitions](analysis.md).

## Practical limits

**GPU evaluation** batches room extraction and constraint searches, avoiding repeated CPU geometry work inside the guidance loop. Final metrics use the CPU reference checker; search outcomes and severities can differ between backends.

Finite searches may remain unknown, binary penalties provide weak incremental feedback, and greedy selection may reduce diversity. At the assumed height, C077 already implies C078, partly duplicating the objective. Results establish feasibility under the stated assumptions; baseline-relative diagnostics do not establish RPLAN fidelity.

[Technical notes](technical_notes.md) retain checkpoint details, exact process-weighting equations, assessment alignment and extended results tables.
