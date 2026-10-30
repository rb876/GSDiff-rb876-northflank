Hi Davis,

I’ve implemented inference-time constraint guidance for GSDiff using **11 selected constraints** and explicit scenario assumptions, while keeping the pretrained model frozen.

The evaluation compares baseline generation, Best-of-N and SVDD, with matched denoiser-evaluation budgets for Best-of-N versus SVDD. I also implemented two proposed extensions: repair-aware SVDD and Monte Carlo feasibility guidance.

You already have access to the [repository](https://github.com/rb876/GSDiff-rb876-northflank). The main entry points are:

- **[docs/technical_report.md](docs/technical_report.md)** — model background, methods, assumptions, results and limitations.
- **[docs/constraints.md](docs/constraints.md)** and **[docs/constraints.json](docs/constraints.json)** — selected constraints and their specification.
- **[test_time/README.md](test_time/README.md)** — installation, tests, reproduction and analysis commands.

For a quick visual overview:

- **[Overall validity and ventilation failures](docs/assets/bathroom_001/ventilation_bottleneck.png)** — how often designs pass all eleven checks with valid structure, and what prevents success. [Per-constraint pass rates](docs/assets/bathroom_001/constraint_success.png) show each check separately; high individual rates do not imply that every design passes all checks.
- **[Six passing SVDD designs](docs/assets/bathroom_001/svdd_passing_examples.png)** — floorplans with labelled room types and areas.
- **[Annotated SVG example](docs/assets/bathroom_001/passing_svdd/svdd_last10_seed10000.svg)** — requirements and passing reasons for all eleven checks, plus the shared bathroom arrangement. PNGs, vector figures and source graph/evaluation JSONs are in **[docs/assets/bathroom_001](docs/assets/bathroom_001/)**.
- **[Hard-MC success and failure](docs/assets/hard_mc_overnight_001/hard_mc_examples.png)** — a valid returned floorplan and the check results explaining a failed terminal search.

The **late-guidance evaluation is complete**, with 100 outputs per method. SVDD guided over the last 10% of steps reaches **56% validity**, versus **47% for Best-of-2**, at the same 2,000 denoiser evaluations per output. The report includes uncertainty estimates, comparison tables, figures and annotated success/failure examples.

**Hard-MC is also complete:** 83 valid designs from 100 attempts, averaging 6,808 NFE per attempt, in about 86 minutes overall. The report includes its outcome and compute tables. Full evaluation of repair-aware SVDD was deferred due to **time constraints**. Full-trajectory SVDD remains **pending** and is estimated to take about **30 hours for 100 outputs** on the H100.

This work builds on [the original GSDiff implementation](https://github.com/SizheHu/GSDiff). Its dataset instructions and pretrained-weight links are preserved in [README.upstream.md](README.upstream.md).

Best,

Riccardo
