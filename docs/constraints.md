# GSDiff constraints: what we check and what we assume

The default profile evaluates **nine constraints from Appendix A of the assessment**, using **three supporting measurement rules**.
All twelve registry entries are implemented. The nine checks determine reward and validity; supporting rules define measurements and add no separate reward or pass/fail result. The scored IDs are C033, C035, C036, C076, C077, C078, C085, C086 and C090.

GSDiff generates walls and room labels. It does not generate furniture, toilets, doors, ventilation equipment or airflow; metric dimensions are derived from the documented coordinate scale.
Our results therefore describe **geometric feasibility under the assumptions below**, rather than verified building compliance.

## Assumptions shared by the checks

| Input | Current assumption | What it means |
|---|---|---|
| Drawing scale | 18/512 = 0.03515625 metres per drawing pixel | Paper/code-derived calibration: RPLAN's 18 m canvas is normalized from 256 native pixels and decoded at 512 drawing pixels. This is not a field measurement. |
| Occupants | Two people | Used to choose bed dimensions and minimum dwelling area/volume. |
| Ceiling height | 2.5 metres everywhere | Height is supplied by the scenario; the model generates only a floorplan. |
| Wall thickness | Zero-thickness area estimate | We do not subtract real wall footprints because the model does not provide them. |
| Furniture and fixtures | Only the hypothetical bed used by the bedroom search | Other furniture, bathroom fittings and door swings are not obstacles in the current checks. |
| Ventilation system | Normal kitchen 105 and bathroom 30 m³/h per room; reduced total 75 and kitchen 45 m³/h per kitchen | Declared three-main-room system specification, fixed across candidates. These are scenario inputs, not GSDiff predictions. |
| Ventilation mapping/mode | All kitchen labels use the same declared equipment; all bathroom labels are full bathrooms; occupant-adjustable reduced mode | The graph does not identify equipment or distinguish an “other shower room” role. |

Ventilation capacities and the operating mode are declared in the [scenario](../test_time/configs/scenario_ventilation_v1.json), fixed before candidate generation and shared across methods. **The threshold tables are from the PDF; the available airflow values are experimental scenario assumptions, not observations or predictions of an installed system.** Changing these inputs can change whether a plan passes; every run saves them. Main rooms are distinct living-room and bedroom polygons (labels 0 and 1).

## Six assessment-derived scored checks

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

**Assumptions:** Room labels are correct, scale follows the paper/code calibration and height is uniform. Area is the estimate described below.

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

## Ventilation checks from Appendix A, page 7

C085/C086 evaluate normal extraction. C090 evaluates a separate occupant-adjustable **reduced operating mode**, so its lower flow values do not replace the normal-flow requirements. Room counts come from the graph; capacity comes from the declared ventilation scenario. The GPU performs room counting, table lookup and scalar deficits entirely with tensors.

### C085 — Kitchen extraction

**Constraint:** Each kitchen's normal extraction must reach the following flow for the dwelling's main-room count:

| Main rooms | 1 | 2 | 3 | 4 | 5 or more |
|---|---:|---:|---:|---:|---:|
| Minimum kitchen flow (m³/h) | 75 | 90 | 105 | 120 | 135 |

**Implementation:** Count living rooms and bedrooms, select the table entry and compare with the scenario's `kitchen_extraction_m3_h`. The same declared per-kitchen capability applies to every kitchen-labelled polygon.

**Assumptions:** Kitchens have the declared equipment and normal flow. No duct, inlet, outlet placement or physical airflow is inferred from wall geometry. An absent kitchen leaves the room-to-equipment mapping unknown; it does not create an automatic pass. A zero main-room count lies outside the table and remains unknown.

### C086 — Bathroom extraction

**Constraint:** Each bathroom's normal extraction must reach the following flow:

| Main rooms | 1 | 2 | 3 | 4 | 5 or more |
|---|---:|---:|---:|---:|---:|
| Minimum bathroom flow (m³/h) | 15 | 15 | 30 | 30 | 30 |

**Implementation:** Use the same main-room count and compare the threshold with `bathroom_extraction_m3_h`.

**Assumptions:** `bathroom_ventilation_role="bathroom"` explicitly maps every label-4 polygon to the bathroom category and the same declared per-room equipment. The model cannot distinguish the separately treated “other shower room” category in C087. Missing, mixed or unresolved role mappings remain unknown. No bathroom also leaves the mapping unknown.

### C090 — Occupant-adjustable reduced ventilation

**Constraint:** In occupant-adjustable reduced mode, both the total dwelling flow **and each kitchen's flow** must reach their respective minima:

| Main rooms | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Minimum total flow (m³/h) | 35 | 60 | 75 | 90 | 105 | 120 | 135 |
| Minimum kitchen flow (m³/h) | 20 | 30 | 45 | 45 | 45 | 45 | 45 |

**Implementation:** Compare `reduced_total_extraction_m3_h` and `reduced_kitchen_extraction_m3_h` with both table entries. The larger normalized deficit supplies the penalty, matching the AND condition. A declared total below the sum of per-kitchen reduced flows is inconsistent scenario metadata and remains unknown.

**Assumptions:** C090 applies when `ventilation_mode="occupant_adjustable"`. Another known mode is not applicable; an unspecified mode is unknown. The PDF supplies no values beyond seven main rooms, so counts outside 1–7 remain unknown rather than being extrapolated. Total flow is a dwelling-level input; it is not inferred by summing generated room equipment. C091 is not part of this profile.

Source for all three tables: [Technical assessment, Appendix A, page 7](../Technical_assessment_TTC.pdf). These checks are conditional capacity-versus-demand tests. Under the default 105/30/75/45 m³/h specification, C086 passes for every supported main-room count when its mapping is known; C085 and C090 both limit the count to at most three. They are correlated, not independent measures. Fixed capacities can favor fewer main rooms, so room-count distributions must be reported alongside success. The specification is not a guarantee that Best-of-N will become difficult.

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

Overall validity requires a structurally valid graph and **every applicable one of the nine scored checks to pass**. Any failure makes the design invalid. With no failure but an unknown active check, overall validity is unknown. If no scored checks apply, the result is not applicable, not valid. Plots and tables report the ventilation group separately from bedroom accessibility, bathroom turning and habitability.

Energy adds the weighted measurable violations and a separate structural-invalidity penalty. Deficit-based checks give no additional reward once their threshold is reached. Unknowns receive no invented severity; candidate selection also protects measurement coverage so that hiding a known violation behind missing information is not treated as a straightforward improvement.

## Reward formulas — LaTeX for HackMD

These formulas describe the current code. They are written as LaTeX math blocks that can be copied into HackMD. Small numerical tolerances are omitted for readability.

### Total energy and reward

Each scored constraint produces a nonnegative **violation severity**. A satisfied check has severity zero. We minimize total energy, or equivalently maximize its negative:

$$
\mathcal{J}=\{\mathrm{C033},\mathrm{C035},\mathrm{C036},\mathrm{C076},\mathrm{C077},\mathrm{C078},
\mathrm{C085},\mathrm{C086},\mathrm{C090}\},
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

### C085, C086 and C090 — ventilation penalties

Let $N$ be the number of living-room and bedroom polygons. Let $b_K(N)$ and $b_B(N)$ be the C085/C086 table thresholds; $b_T^r(N)$ and $b_K^r(N)$ are C090's reduced total/kitchen thresholds. The declared scenario flows are $q_K,q_B,q_T^r,q_K^r$, in m³/h. Using the normalized shortfall $d$ defined above:

$$
v_{\mathrm{C085}}=d(q_K;b_K(N)),\qquad
v_{\mathrm{C086}}=d(q_B;b_B(N)),
$$

$$
v_{\mathrm{C090}}=\max\left(d(q_T^r;b_T^r(N)),\ d(q_K^r;b_K^r(N))\right).
$$

The maximum in C090 enforces both requirements: sufficient total extraction cannot compensate for insufficient kitchen extraction. Every scored constraint has weight one. Missing inputs, unresolved mappings and unsupported table indices remain unknown and add no invented severity. A known inapplicable ventilation mode excludes C090 from the denominator.

**Illustrative calculation:** for four main rooms and declared flows $q_K=105$, $q_B=30$, $q_T^r=75$, $q_K^r=45$ m³/h:

| Check | Calculation | Severity |
|---|---|---:|
| C085 | $(120-105)/120$ | 0.125 |
| C086 | $\max(0,(30-30)/30)$ | 0 |
| C090 | $\max((90-75)/90,0)$ | 0.1667… |

These three checks contribute $7/24\approx0.2917$ to energy. This example illustrates the PDF tables and declared flows; it is not a generated experimental result. With equal validity and coverage, a smaller total penalty receives a higher reward.

The CPU formula adds 100 for structural invalidity when other measurements remain available. The GPU backend leaves scored severities unknown for invalid graphs and assigns energy 100.

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

Every final design is checked with the CPU reference implementation. The CPU and GPU placement searches can differ; **reported final validity uses the CPU result**. The three ventilation checks use the same room-count tables, scenario inputs and severity formulas on both backends. They add no CPU geometry calls inside GPU guidance.

## Files and inspection

- [Constraint registry](../test_time/configs/constraints_ventilation_v1.json): nine Appendix A checks and three measurement rules.
- [Scenario](../test_time/configs/scenario_ventilation_v1.json): declared ventilation mode/capacities, paper/code-derived scale, occupancy, height and area model.
- [Documentation snapshot](constraints.json): the same selection in a convenient JSON file.
- [CPU checks](../test_time/constraints.py), [GPU checks](../test_time/gpu_constraints.py), and [shared area rules](../test_time/area.py).

List the current selection without loading models or generating designs:

```bash
python -m test_time.run --method baseline --list-constraints
```

The default profile contains only the nine scored Appendix A checks listed above. Historical configurations remain available for reproducing their saved outputs, but their metrics must not be mixed with this profile. Baseline can be rechecked and the full Best-of-N bank reselected; guidance needs new runs under the same saved ventilation scenario. Freeze the scenario before the final comparison on fresh seeds.
