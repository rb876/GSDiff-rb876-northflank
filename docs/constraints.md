# GSDiff constraints: what we check and what we assume

The current experiment evaluates **six constraints**, using **three supporting measurement rules**.
The registry contains only these nine implemented entries. The six checks determine reward and validity; the three rules define their measurements and add no separate reward or pass/fail result.

GSDiff generates walls and room labels. It does not generate furniture, toilets, doors or real-world dimensions.
Our results therefore describe **geometric feasibility under the assumptions below**, rather than verified building compliance.

## Assumptions shared by the checks

| Input | Current assumption | What it means |
|---|---|---|
| Drawing scale | 0.025 metres per drawing pixel | A 512-pixel canvas represents 12.8 metres. This scale was chosen for the experiment, not recovered from the dataset. |
| Occupants | Two people | Used to choose bed dimensions and minimum dwelling area/volume. |
| Ceiling height | 2.5 metres everywhere | Height is supplied by the scenario; the model generates only a floorplan. |
| Wall thickness | Zero-thickness area estimate | We do not subtract real wall footprints because the model does not provide them. |
| Furniture and fixtures | Only the hypothetical bed used by the bedroom search | Other furniture, bathroom fittings and door swings are not obstacles in the current checks. |

Changing these assumptions can change whether a design passes. They are saved with each run.

## The six scored checks

### C033 — Space to turn a wheelchair in a bedroom

**Constraint:** At least one bedroom must fit the required bed and a turning circle **1.50 m in diameter**. The circle must not overlap the bed.

**Implementation:** Search positions for a rectangular bed and a circle inside the bedroom polygon. Check that both fit completely and remain separate. If C035 also passes, both checks must use the same bed arrangement.

**Assumptions:** The bed is hypothetical, not a generated object. It is aligned with the drawing axes, with either of two orientations. Other furniture and door swings are ignored. This establishes that a suitable arrangement was found under those assumptions.

### C035 — Circulation around the bed

**Constraint:** The bed must have either:

- **0.90 m** clearance along each long side and **1.20 m** at one free short side; or
- **1.20 m** along each long side and **0.90 m** at one free short side.

**Implementation:** For each proposed bed position, construct the rectangle enclosing the bed and the required clearances. Require the entire rectangle to fit inside the bedroom, including in concave rooms. Joint success with C033 requires one bed position satisfying both checks.

**Assumptions:** The clearance region is empty apart from the bed. This does not check access through a doorway or obstacles elsewhere in the dwelling.

### C036 — Turning space in an empty bathroom

**Constraint:** At least one bathroom must contain a **1.50 m diameter** turning circle.

**Implementation:** Search circle centres and check their distance from the room boundary. A fitting circle establishes a pass. A geometric bound can establish that the room is too small; an inconclusive search remains unknown.

**Assumptions:** The bathroom is treated as empty. A toilet, washbasin, shower or door swing could obstruct a circle that currently fits. A pass means **empty-room feasibility**, not that a furnished bathroom is accessible.

### C076 — Minimum size of a main room

**Constraint:** At least one main room must satisfy either:

- area **at least 9 m²** AND height **at least 2.20 m**; or
- volume **at least 20 m³**.

**Implementation:** Treat living rooms and bedrooms as main rooms. Calculate their habitable area using C079/C080, then calculate volume as area × scenario height. Evaluate the exact AND/OR rule separately for each room; one passing room is sufficient.

**Example:** At a height of 2.5 m, an 8 m² room reaches 20 m³ and passes through the volume branch.

**Assumptions:** Room labels are correct, scale is assumed and height is uniform. Area is the estimate described below.

### C077 — Minimum habitable area for the occupants

**Constraint:** Require **14 m² per person for the first four people**, plus **10 m² per additional person**.

**Implementation:** Add eligible room areas using C079/C080 and compare the total with:

`14 × min(occupants, 4) + 10 × max(occupants − 4, 0)`

For the current two-person scenario, the minimum is **28 m²**.

**Assumptions:** Occupancy is supplied, not inferred. Living rooms, bedrooms, kitchens and bathrooms contribute to our area estimate. Storage, balconies and outside regions do not.

### C078 — Minimum habitable volume for the occupants

**Constraint:** Require **33 m³ per person for the first four people**, plus **23 m³ per additional person**.

**Implementation:** Multiply the habitable area from C079/C080 by scenario height, then compare with:

`33 × min(occupants, 4) + 23 × max(occupants − 4, 0)`

For two people, the minimum is **66 m³**.

**Assumptions:** The same area and occupancy assumptions as C077 apply, together with a uniform ceiling height. At 2.5 m, 66 m³ corresponds to 26.4 m²; consequently, passing the current 28 m² area requirement also satisfies this volume requirement. These checks are not independent under that scenario.

## Three supporting rules — no extra compliance points

### C034 — Bed dimensions

**Rule:** Use a **0.90 × 1.90 m** bed for a one-person dwelling, or **1.40 × 1.90 m** otherwise.

**Implementation:** Supply those dimensions to the shared C033/C035 placement search. Our two-person scenario uses 1.40 × 1.90 m.

**Assumptions:** One standard rectangular bed is considered. Supplying its dimensions is not evidence that a bed exists, and is not an independent pass.

### C079 — What wall-related area should be excluded

**Rule:** Habitable area excludes walls, partitions, stairs, stairwells, shafts and door/window recesses.

**Implementation:** Explicitly declare the current `zero_thickness_proxy` area model. Room polygons supply the area estimate; no actual wall footprint is subtracted. CPU validation uses polygon unions; GPU scoring sums verified non-overlapping room areas.

**Assumptions:** Real wall thickness and the other unrepresented exclusions remain unverified. This rule records the limitation of the estimator; it does not certify that every required deduction was made. Setting the area model to `unknown` makes dependent area/volume checks unknown.

### C080 — Room-type and low-height exclusions

**Rule:** Habitable area excludes storage, balconies and other listed non-habitable spaces, plus area with height **below 1.80 m**. The complete list is recorded in the JSON registry.

**Implementation:** One shared classification table includes living rooms, bedrooms, kitchens and bathrooms, and excludes storage, balcony and outside labels. Under our uniform-height scenario, height below 1.80 m excludes all habitable area; exactly 1.80 m is included. This affects C076–C078, not physical bed/bathroom placement geometry.

**Assumptions:** The model does not separately identify every excluded category, such as stairwells, garages or unfinished attics. It also provides no map of varying headroom. Those exclusions cannot be individually verified. Missing height makes the v2 area/volume measurements unknown.

## How scoring and validity work

- **Pass:** The formalized check is satisfied under its stated assumptions.
- **Fail:** Available evidence establishes a violation of that check.
- **Unknown:** An active check lacks required information, or its bounded search is inconclusive. It does not count as passing.
- **Not applicable:** The check does not apply to the scenario. Supporting definitions also use this stored status because they are not independent checks.

Overall validity requires a structurally valid graph and **every applicable one of the six scored checks to pass**. Any failure makes the design invalid. With no failure but an unknown active check, overall validity is unknown. If no scored checks apply, the result is not applicable, not valid.

Energy adds the weighted measurable violations and a separate structural-invalidity penalty. Deficit-based checks give no additional reward once their threshold is reached. Unknowns receive no invented severity; candidate selection also protects measurement coverage so that hiding a known violation behind missing information is not treated as a straightforward improvement.

## Reward formulas — LaTeX for HackMD

These formulas describe the current code. They are written as LaTeX math blocks that can be copied into HackMD. Small numerical tolerances are omitted for readability.

### Total energy and reward

Each scored constraint produces a nonnegative **violation severity**. A satisfied check has severity zero. We minimize total energy, or equivalently maximize its negative:

$$
\mathcal{J}=\{\mathrm{C033},\mathrm{C035},\mathrm{C036},\mathrm{C076},\mathrm{C077},\mathrm{C078}\},
$$

$$
\mathcal{M}(x)=
\{j\in\mathcal{J}:j\text{ is enabled, applicable, and measurable for }x\},
$$

$$
E(x)=\lambda_{\mathrm{struct}}\,
\mathbf{1}\{x\text{ is structurally invalid}\}
+\sum_{j\in\mathcal{M}(x)}\lambda_j\,v_j(x),
\qquad
r(x)=-E(x).
$$

Here:

- $x$ is the candidate floorplan being evaluated.
- $v_j(x)$ is the violation severity for check $j$.
- Current scored-check weights are $\lambda_j=1$.
- The current structural-invalidity weight is $\lambda_{\mathrm{struct}}=100$.
- The structural term is one binary penalty, not 100 for every individual graph defect.
- Supporting rules C034/C079/C080 have zero independent weight.

**Unknown is not a zero violation.** Its severity is undefined and it is excluded from the sum; coverage and validity are handled separately during selection. Consequently, a numerical reward of zero alone does not establish overall validity.

### The normalized shortfall used by dimensional checks

For an observed quantity $q$ with required minimum $q_{\min}>0$, define:

$$
d(q;q_{\min})=
\max\left(0,\frac{q_{\min}-q}{q_{\min}}\right).
$$

For example, 21 m² against a 28 m² minimum gives $d=0.25$. Both 28 m² and 40 m² give zero: there is no extra reward for exceeding the requirement.

### C033 and C035 — binary placement penalties

These two checks currently use **binary feasibility penalties**, not a smooth clearance-distance reward:

$$
v_j(x)=
\begin{cases}
0, & \text{a qualifying placement is witnessed},\\
1, & \text{failure is established},\\
\text{undefined}, & \text{the result is unknown or not applicable},
\end{cases}
\qquad j\in\{\mathrm{C033},\mathrm{C035}\}.
$$

For C033, the witness contains the required bed and a separate 1.50 m turning circle. For C035, the clearance envelope must fit around the same bed used by the turning-circle witness. A missing required bedroom is a failure. A search that simply runs out of positions is unknown.

This means a proven failure that becomes “less bad” still contributes 1 until its status changes; these checks do not currently reward incremental clearance improvements.

### C036 — bathroom turning-circle shortfall

The required radius is:

$$
\rho_{\min}=\frac{1.50}{2}=0.75\ \mathrm{m}.
$$

For each bathroom $b$, GPU scoring computes an upper bound on the radius of any circle that could fit:

$$
U_b=
\min\left(
\frac{\min(w_b,h_b)}{2},
\sqrt{\frac{A_b}{\pi}}
\right).
$$

Here $w_b,h_b$ are the axis-aligned bounding-box dimensions, and $A_b$ is the physical bathroom polygon area. This is a bound, not a measured fitting-circle radius. If every bathroom has $U_b<0.75$, failure is established and the GPU severity is:

$$
v_{\mathrm{C036}}^{\mathrm{GPU}}(x)
=d\left(\max_b U_b;\,0.75\right).
$$

A witnessed fitting circle gives severity zero. If the bounds permit a fit but the finite search finds none, the result is unknown. With no bathroom, the severity is 1.

**CPU reporting uses a different failure estimate.** Its polygon search finds a radius $L_b$, stored as the radius lower bound. After its upper-bound test establishes failure in every bathroom, the reported severity is:

$$
v_{\mathrm{C036}}^{\mathrm{CPU}}(x)
=\min_b d(L_b;0.75)
=d\left(\max_b L_b;\,0.75\right).
$$

The CPU failure test uses $L_b+\varepsilon_b<0.75$, where $\varepsilon_b$ is the recorded search tolerance. CPU and GPU penalties can therefore differ even when both report failure.

### C076 — main-room area/height OR volume

Let $a_i$ be the habitable area of main room $i$, after C079/C080, and let $H$ be the scenario ceiling height. Its estimated volume is $a_iH$.

The code uses a maximum for the AND branch, and a minimum for the OR:

$$
v_{\mathrm{C076},i}(x)=
\min\left[
\max\left(d(a_i;9),\,d(H;2.20)\right),
d(a_iH;20)
\right].
$$

Only one main room needs to satisfy the requirement:

$$
v_{\mathrm{C076}}(x)=\min_{i\in\mathcal{R}_{\mathrm{main}}}
v_{\mathrm{C076},i}(x).
$$

Thus the penalty is zero exactly when at least one main room meets the area-and-height branch or the volume branch, within numerical tolerance. No main room gives severity 1; missing required measurements give unknown.

### C077 — total habitable-area shortfall

Let $n$ be the intended number of occupants and $A_{\mathrm{hab}}$ the total eligible area:

$$
A_{\min}(n)=14\min(n,4)+10\max(n-4,0),
$$

$$
v_{\mathrm{C077}}(x)=d\left(A_{\mathrm{hab}};\,A_{\min}(n)\right).
$$

For our two-person scenario, $A_{\min}=28\ \mathrm{m}^2$.

### C078 — total habitable-volume shortfall

Using the same area estimate and uniform height:

$$
V_{\mathrm{hab}}=A_{\mathrm{hab}}H,
\qquad
V_{\min}(n)=33\min(n,4)+23\max(n-4,0),
$$

$$
v_{\mathrm{C078}}(x)=d\left(V_{\mathrm{hab}};\,V_{\min}(n)\right).
$$

For two occupants, $V_{\min}=66\ \mathrm{m}^3$. Unknown scale, height or area-model inputs leave the corresponding active checks unknown.

### Worked example

Suppose a structurally valid design has measurable severities:

| Check | Severity |
|---|---:|
| C033 | 0 |
| C035 | 1 |
| C036 | 0.20 |
| C076 | 0 |
| C077 | 0.25 |
| C078 | 0.2045… |

For the current two occupants and 2.5 m height, 21 m² of habitable area gives 52.5 m³ of volume.
The area deficit is $7/28=0.25$; the volume deficit is $13.5/66\approx0.2045$.
This is an illustrative calculation, not an experimental result.

With unit weights:

$$
E(x)=0+1+0.20+0+0.25+\frac{13.5}{66}\approx1.6545,
\qquad r(x)\approx-1.6545.
$$

A comparable candidate with energy 1.20 has reward $-1.20$ and is preferred **if both survive the validity and coverage guards**.

The CPU formula adds 100 for structural invalidity when the other measurements remain available. The GPU backend skips constraint searches for invalid graphs, so their regulatory severities remain unknown and its energy is 100.

### How Best-of-N and SVDD use this reward

Baseline generation is not steered by the reward. Best-of-N evaluates completed candidates; SVDD evaluates graphs reconstructed from predicted clean states at guided steps.

Before comparing rewards, selection prioritizes structural validity, rejects newly introduced unknown IDs relative to its coverage anchor, and prefers fewer unknown checks. Let $\mathcal{A}$ denote the remaining eligible candidates. With the default temperature zero:

$$
k^*=\operatorname*{arg\,max}_{k\in\mathcal{A}}r(x_k)
=\operatorname*{arg\,min}_{k\in\mathcal{A}}E(x_k).
$$

For SVDD at positive temperature $\tau$, selection instead uses:

$$
\Pr(k\mid\mathcal{A})=
\frac{\exp(r(x_k)/\tau)}
{\sum_{\ell\in\mathcal{A}}\exp(r(x_\ell)/\tau)}.
$$

SVDD then continues the selected **noisy** diffusion state, not its clean prediction. If no valid measurable proposal is eligible, it continues the ordinary first proposal. The reward is a selection score, not a probability of compliance.

## GPU guidance and final CPU validation

During generation, room extraction, constraint geometry and proposal selection use tensors on the GPU. The placement search uses configured **0.25 m and 0.05 m grids**, at most **20,000 bed positions per room**, and up to **32 retained directional circle witnesses**. A missed placement is not proof of impossibility.

Every final design is checked with the CPU reference implementation. The CPU and GPU geometry searches can differ; **reported final validity uses the CPU result**. The new area definitions are shared between both implementations.

## Files and inspection

- [Constraint registry](../test_time/configs/constraints_v2.json): thresholds, groups, roles and assumptions for the six checks and three measurement rules.
- [Scenario](../test_time/configs/scenario_v2.json): scale, occupancy, height and area model.
- [Documentation snapshot](constraints.json): the same selection in a convenient JSON file.
- [CPU checks](../test_time/constraints.py), [GPU checks](../test_time/gpu_constraints.py), and [shared area rules](../test_time/area.py).

List the current selection without loading models or generating designs:

```bash
python -m test_time.run --method baseline --list-constraints
```

The original v1 configuration remains available for reproducing older results. This guide describes the current v2 default and the assessment's selected requirements, not a complete building-regulation assessment.
