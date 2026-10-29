# GSDiff evaluation — figure guide

`sweep_002` compares **100 selected designs per method** against six scored constraints. Best-of-11 contributes one winner per trial. Blue denotes baseline, orange Best-of-11, grey late SVDD and green full SVDD; outcome plots use a separate status legend.

<!-- For HackMD, upload the six PNGs and replace the relative image links below with the uploaded image URLs. -->

## 1. Overall validity

![Overall validity](../test_time/outputs/sweep_002/analysis/figures/overall_validity.png)

The left panel shows the percentage of designs with **valid structure and all applicable constraints passing**, with 95% Wilson confidence intervals. The right panel separates valid, invalid, unknown and N/A outcomes; all selected designs remain in the denominator. Observed validity is 2%, 8%, 14% and 14%; the intervals alone do not test differences between methods.

## 2. Success by constraint

![Constraint outcomes](../test_time/outputs/sweep_002/analysis/figures/constraint_outcomes.png)

Each horizontal bar shows pass, fail, unknown and N/A percentages for one constraint, over all selected designs. The asterisk marks the largest confirmed failure count within each method. **C035 remains the main bottleneck**, while SVDD substantially improves C036 relative to baseline; supporting rules are excluded.

## 3. Number of constraints passed

![Number of constraints passed](../test_time/outputs/sweep_002/analysis/figures/passed_constraints.png)

The histogram shows the percentage of designs passing 0–6 checks. A shift to the right means more constraints are satisfied, even when overall validity remains low. Unknowns do not count as passes, and **six passes still require valid structure**: full SVDD has 15% with six passes but 14% overall validity.

## 4. Violation severity

![Violation severity](../test_time/outputs/sweep_002/analysis/figures/constraint_severity.png)

These histograms pool measured constraint severities using shared bins and axes; lower values indicate smaller violations. Percentages are normalized over measured checks, not designs: passing checks contribute zero, while unknowns and structural penalties are excluded. Most measured checks pass, but a concentration of large violations remains.

## 5. Layout distributions

![Layout distributions](../test_time/outputs/sweep_002/analysis/figures/layout_distributions.png)

The top row compares interior-room counts; the bottom compares summed room areas in drawing pixels squared. Percentages are normalized over structurally valid designs with usable geometry. Similar distributions suggest these features are preserved relative to baseline; they do not establish overall diversity or fidelity to real RPLAN designs.

## 6. Runtime and NFE

![Runtime and NFE](../test_time/outputs/sweep_002/analysis/figures/compute.png)

The left panel shows recorded generation/checking seconds per selected output, amortized across batches and excluding model loading and rendering/export. The right shows **denoiser NFE (number of function evaluations)** per output, counting each evaluated sample within a batch, including candidates and SVDD lookahead; edge-model and reward evaluations are excluded. Best-of-11 and full SVDD are approximately NFE-matched (11,000 versus 10,990), but their runtimes differ substantially; late SVDD uses 3,000 NFE.
