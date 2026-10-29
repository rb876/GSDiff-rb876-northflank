# 007 — Refine the final CPU checker

Re-audited all **400 originally selected designs** from `sweep_002` using the opt-in
`cpu_bedroom_audit_v2` checker. The frozen models, GPU rewards, generated graphs and original
candidate selections are unchanged. This measures checker sensitivity on existing outputs;
it is not a new generation experiment.

The CPU refinement adds bed positions aligned with clearance-envelope contacts, finer turning
circle witnesses and a circle search in the room's remaining free space after placing the bed.
Wall contact uses the existing **1e-8 m** geometry tolerance to avoid numerical rejection.
An axis-aligned bounding-box bound can certify that no permitted clearance envelope fits.
Search budgets remain finite: a search miss stays unknown unless a geometric bound proves failure.

Scale, bed dimensions, axis-aligned orientations, clearance requirements and the shared-bed
requirement for bedroom circulation and turning remain unchanged. Assumed geometry still does
not establish real-world accessibility compliance.

Each method has 100 selected designs; counts therefore also equal percentages.

| Method | Overall valid: original → revised | C035 unknown: original → revised |
|---|---:|---:|
| Baseline | 2 → 2 | 19 → 12 |
| Best-of-11 | 8 → 9 | 2 → 0 |
| SVDD, last 20% | 14 → 14 | 4 → 3 |
| SVDD, full trajectory | 14 → 15 | 6 → 4 |

All 12 changed check outcomes resolve unknowns: nine become confirmed failures and three become
passes. C035 remains a major bottleneck.

Original source files remain intact. Revised evaluations and provenance hashes are saved
separately; original generation timings exclude this offline CPU work. **130 focused tests passed.**

See the [detailed audit](../../outputs/sweep_002_cpu_audit_v2/AUDIT.md) and
[revised analysis](../../outputs/sweep_002_cpu_audit_v2/analysis/REPORT.md).
