"""Create figures, tables and an image gallery from completed saved runs; no inference."""

import argparse
from collections import Counter
import csv
from itertools import combinations
import json
import math
from pathlib import Path

from .evaluate import (_configured_rows, check_summary, counted_results, denoiser_comparisons,
                       fraction, summarize_run)
from .types import overall_validity, read_json, write_json


def wilson_interval(successes, count):
    """95% Wilson score interval for a sample-level binomial proportion."""
    if not count:
        return None, None
    z = 1.959963984540054
    p = successes / count
    denominator = 1 + z * z / count
    centre = (p + z * z / (2 * count)) / denominator
    half = z * math.sqrt(p * (1 - p) / count + z * z / (4 * count * count)) / denominator
    return max(0.0, centre - half), min(1.0, centre + half)


def load_records(experiment_dir):
    """Keep each completed run separate; never pool different variants or seed ranges."""
    root = Path(experiment_dir).resolve()
    paths = [root] if (root / "candidates.jsonl").exists() else sorted(
        {path.parent for path in root.rglob("candidates.jsonl")})
    records, skipped = [], []
    order = {"baseline": 0, "best_of_n": 1, "svdd_pm_last20": 2, "svdd_pm_full": 3}
    for path in sorted(paths, key=lambda value: (str(value.parent), order.get(value.name, 4), value.name)):
        if not (path / "config.json").exists() or not (path / "run.json").exists():
            skipped.append({"run": str(path), "reason": "Missing config.json or run.json"})
            continue
        run = read_json(path / "run.json")
        if run.get("status", "complete") != "complete":
            skipped.append({"run": str(path), "reason": "Run status: " + run["status"]})
            continue
        if not (path / "candidates.jsonl").read_text().strip():
            skipped.append({"run": str(path), "reason": "No saved candidates"})
            continue
        summary = summarize_run(path)
        if summary["status"] != "complete":
            skipped.append({"run": str(path), "reason": "Saved trial count does not match the planned count"})
            continue
        config = read_json(path / "config.json")
        rows = [json.loads(line) for line in (path / "candidates.jsonl").read_text().splitlines() if line.strip()]
        rows = _configured_rows(rows, config.get("constraints", []))
        selected = sorted((row for row in rows if row.get("selected")), key=lambda row: row["trial"])
        records.append({"label": path.name, "run_dir": path, "config": config,
                        "summary": summary, "rows": selected})
    labels = Counter(record["label"] for record in records)
    for record in records:
        if labels[record["label"]] > 1:
            record["label"] = record["run_dir"].relative_to(root).as_posix()
    return records, skipped


def method_metrics(record):
    summary = record["summary"]
    n = summary["selected_trials"]
    ids = set(summary["active_scored_ids"]) | set(summary["per_constraint"])
    low, high = wilson_interval(summary["overall_valid_count"], n)
    return {
        "method": record["label"], "selected_outputs": n,
        **{key: summary["parameters"][key] for key in ("n", "k", "guided_steps", "temperature", "reward_backend")},
        "scored_constraints": len(ids), "supporting_rules": summary["constraint_counts"]["active_supporting"],
        "overall_valid_count": summary["overall_valid_count"], "overall_valid_rate": summary["overall_valid_rate"],
        "overall_ci_low": low, "overall_ci_high": high,
        "structural_valid_rate": summary["structural_valid_rate"],
        "overall_unknown_rate": summary["overall_validity_counts"]["unknown"] / n,
        "applicability_coverage": fraction(summary["applicable_count"], n * len(ids)),
        "applicable_pass_rate": fraction(summary["pass"], summary["applicable_count"]),
        "measurable_compliance": summary["compliance"], "measurable_coverage": summary["measurable_fraction"],
        "unknown_check_rate": summary["unknown_fraction"], "mean_measured_severity": summary["severity"]["mean"],
        "mean_energy": sum(row["evaluation"]["energy"] for row in record["rows"]) / n,
        "wall_seconds_per_trial": summary["compute"]["wall_seconds_per_trial"],
        "node_samples_per_trial": summary["compute"]["node_samples_per_trial"],
        "edge_samples_per_trial": fraction(summary["compute"]["edge_samples"], n)
            if summary["compute"]["edge_samples"] is not None else None,
        "comparison_group": summary["comparison_group"],
    }


def constraint_metrics(record):
    rows = record["rows"]
    specifications = {spec["id"]: spec for spec in record["config"].get("constraints", [])}
    ids = sorted(set(record["summary"]["active_scored_ids"]) | set(record["summary"]["per_constraint"]))
    output = []
    for identifier in ids:
        results = [check for row in rows for check in counted_results(row) if check["id"] == identifier]
        metrics = check_summary(results)
        sole_failure = 0
        for row in rows:
            applicable = [check for check in counted_results(row) if check["applicable"]]
            if (row["evaluation"]["structurally_valid"]
                    and any(check["id"] == identifier and check["status"] == "fail" for check in applicable)
                    and all(check["status"] == "pass" and check["measurable"]
                            for check in applicable if check["id"] != identifier)):
                sole_failure += 1
        failed_reasons = Counter(check.get("explanation") or "No explanation recorded"
                                 for check in results if check["status"] == "fail")
        unknown_reasons = Counter(check.get("explanation") or "No explanation recorded"
                                  for check in results if check["status"] == "unknown")
        spec = specifications.get(identifier, {})
        output.append({"method": record["label"], "id": identifier,
                       "definition": spec.get("definition", ""), "source_id": spec.get("source_id", ""),
                       **{key: metrics[key] for key in ("pass", "fail", "unknown", "not_applicable",
                                                      "applicable_count", "measurable_count")},
                       "pass_rate": fraction(metrics["pass"], metrics["applicable_count"]),
                       "measurable_compliance": metrics["compliance"],
                       "mean_measured_severity": metrics["severity"]["mean"],
                       "sole_failure_count": sole_failure,
                       "failure_reasons": dict(failed_reasons.most_common()),
                       "unknown_reasons": dict(unknown_reasons.most_common())})
    return output


def paired_validity(records):
    comparisons = []
    for first, second in combinations(records, 2):
        a, b = first["summary"], second["summary"]
        if (a["comparison_group"] != b["comparison_group"] or a["trial_seed_start"] is None
                or a["trial_seed_start"] != b["trial_seed_start"]
                or a["selected_trial_ids"] != b["selected_trial_ids"]):
            continue
        outcomes = Counter((overall_validity(x["evaluation"]) == "valid", overall_validity(y["evaluation"]) == "valid")
                           for x, y in zip(first["rows"], second["rows"]))
        comparisons.append({"method_a": first["label"], "method_b": second["label"],
                            "paired_outputs": len(first["rows"]), "both_valid": outcomes[True, True],
                            "only_a_valid": outcomes[True, False], "only_b_valid": outcomes[False, True],
                            "neither_valid": outcomes[False, False],
                            "b_minus_a_validity_points": 100 * (outcomes[False, True] - outcomes[True, False]) / len(first["rows"])})
    return comparisons


def baseline_for(record, records):
    matches = [candidate for candidate in records
               if candidate["summary"]["method"] == "baseline"
               and candidate["summary"]["comparison_group"] == record["summary"]["comparison_group"]]
    same_batch = [candidate for candidate in matches if candidate["run_dir"].parent == record["run_dir"].parent]
    return same_batch[0] if len(same_batch) == 1 else matches[0] if len(matches) == 1 else None


def write_csv(path, rows, fieldnames=None):
    if not rows and not fieldnames:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames or list(rows[0]))
        writer.writeheader()
        writer.writerows({key: json.dumps(value) if isinstance(value, (dict, list)) else value
                         for key, value in row.items()} for row in rows)


def _percent(value):
    return "n/a" if value is None else f"{100 * value:.1f}%"


def _number(value):
    return "n/a" if value is None else f"{value:.3g}"


def write_analysis_report(output, records, skipped, figures, gallery):
    from .analysis_plots import FIGURE_CAPTIONS

    lines = ["# Sweep analysis", "",
             "This report analyses completed saved runs without generating or rechecking designs. "
             "Only the selected output from each trial contributes to method comparisons. "
             "Best-of-N's rejected candidates are excluded. Final constraint outcomes come from the saved CPU reference audit.", ""]
    if any(record["summary"].get("scale_recalibration") for record in records):
        lines += ["## Scale recalibration", "",
                  "Baseline reuses saved unconditioned designs. Best-of-N reselects from its complete saved candidate bank "
                  "using tensor rewards at the new scale; selected designs receive the refined CPU checker. "
                  "Rejected candidates retain tensor scores and do not enter the selected-output comparisons. "
                  "Reused denoiser NFE is historical generation work, not additional inference. "
                  "Generation runtime is unavailable for recalibrated controls; their separate offline rescoring and audit "
                  "times are recorded in each run's `calibration.json`. No runtime multiplier is inferred from these controls.", ""]
    if any(record["summary"].get("evaluation_audit") for record in records):
        lines += ["## Evaluation audit", "",
                  "The re-audited runs reuse the **originally selected outputs**, with no reselection or new generation. "
                  "Their saved population contains only these outputs, not the original full candidate pool. "
                  "Generation timings and counters are preserved and **exclude the new offline CPU audit**; "
                  "audit time is recorded separately in [`audit.json`](../audit.json). "
                  "Guidance/reward metadata describes the original generation, not the revised final checker.", "",
                  "| Method | Final evaluation checker |", "|---|---|"]
        for record in records:
            audit = record["summary"].get("evaluation_audit") or {}
            lines.append(f"| {record['label']} | {audit.get('checker_version', 'original CPU reference')} |")
        lines += ["", "See the [audit report](../AUDIT.md) for checker changes, provenance and outcome changes. "
                  "Original and revised checker versions belong to separate comparison groups.", ""]
    lines += [
             "## Assessment requirements", "",
             "Technical_assessment_TTC.pdf, pages 2–3 (§§2.2, 2.6–2.8), asks for a shared checkpoint/scenario/representation, "
             "compute multiplier, subset size, applicability coverage, compliance, severity, structural validity, "
             "distribution-preservation diagnostics, vector outputs and annotated successful/failed examples. "
             "It also requests an ablation and honest assumptions/limitations. It specifies no sample count or mandatory FID/KID metric.", "",
             "The current profile has six evaluated constraints and three supporting measurement rules. "
             "The tables below use each run's saved registry, including historical profiles, rather than assuming today's configuration.", "",
             "## Saved run settings", "",
             "| Method | N | K | Guided steps | Temperature | Reward backend |",
             "|---|---:|---:|---:|---:|---|"]
    for record in records:
        parameters = record["summary"]["parameters"]
        settings = " | ".join(str(parameters[key]) if parameters[key] is not None else "—"
                              for key in ("n", "k", "guided_steps", "temperature", "reward_backend"))
        lines.append(f"| {record['label']} | {settings} |")
    lines += ["", "Full checkpoint paths/hashes, source hashes and scenario values are preserved per run in `analysis.json`.", "",
             "## Method comparison", "",
             "Overall validity requires valid structure and all applicable scored checks to pass; unknown is not a pass. "
             "Constraint success is passes / applicable checks, including unknowns in the denominator. "
             "Measurable-only compliance is shown separately in CSV. Applicability coverage is applicable checks / "
             "(selected outputs × configured scored constraints). Supporting rules contribute to neither denominator.", "",
             "| Method | Outputs | Checks | Overall valid (95% CI) | Structural valid | Applicability | Constraint success | Unknown checks | Mean measured severity | Seconds/output | Runtime × baseline | Denoiser NFE/output | NFE × baseline |",
             "|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for record in records:
        m = record["metrics"]
        lines.append(f"| {m['method']} | {m['selected_outputs']} | {m['scored_constraints']} | "
                     f"{_percent(m['overall_valid_rate'])} ({_percent(m['overall_ci_low'])}–{_percent(m['overall_ci_high'])}) | "
                     f"{_percent(m['structural_valid_rate'])} | {_percent(m['applicability_coverage'])} | "
                     f"{_percent(m['applicable_pass_rate'])} | {_percent(m['unknown_check_rate'])} | "
                     f"{_number(m['mean_measured_severity'])} | {_number(m['wall_seconds_per_trial'])} | "
                     f"{_number(m['runtime_multiplier_vs_baseline'])} | {_number(m['node_samples_per_trial'])} | "
                     f"{_number(m['denoiser_multiplier_vs_baseline'])} |")
    if len({record["summary"]["comparison_group"] for record in records}) > 1:
        lines += ["", "**Multiple configuration groups are present.** These figures describe each run, but differences across groups "
                  "are not controlled method comparisons. Baseline-relative metrics and paired comparisons are restricted to compatible groups; "
                  "`tables/methods.csv` records each group."]
    lines += ["", "Intervals are 95% Wilson intervals for each method's overall-valid indicator; "
              "they describe sampling uncertainty under the fixed scenario. They do not establish legal compliance or test a difference between methods.", "",
              "## Figures", ""]
    for name, paths in figures.items():
        png, pdf = (Path(paths[k]).relative_to(output).as_posix() for k in ("png", "pdf"))
        lines += [f"![{name.replace('_', ' ')}]({png})", ""]
        if caption := FIGURE_CAPTIONS.get(name):
            lines += [caption, ""]
        lines += [f"[Download PDF]({pdf})", ""]
    lines += ["## Constraint difficulty", "",
              "A sole failure is a structurally valid design where that constraint fails and all other applicable scored checks pass. "
              "This identifies a bottleneck without assuming that removing or weakening it is appropriate.", "",
              "| Method | Constraint | Pass | Fail | Unknown | N/A | Pass/applicable | Sole failures |",
              "|---|---|---:|---:|---:|---:|---:|---:|"]
    for record in records:
        for row in record["constraint_rows"]:
            lines.append(f"| {record['label']} | {row['id']} | {row['pass']} | {row['fail']} | {row['unknown']} | "
                         f"{row['not_applicable']} | {_percent(row['pass_rate'])} | {row['sole_failure_count']} |")
    lines += ["", "### Recorded failure explanations", ""]
    for record in records:
        hardest = sorted(record["constraint_rows"], key=lambda item: (-item["fail"], -item["unknown"], item["id"]))
        for row in hardest:
            reasons = list(row["failure_reasons"].items())[:2] + list(row["unknown_reasons"].items())[:1]
            if reasons:
                lines.append(f"- **{record['label']} / {row['id']}**: " + "; ".join(f"{reason} ({count})" for reason, count in reasons))
    lines += ["", "## Diversity and baseline-relative distribution", "",
              "These diagnostics use structurally valid samples with usable geometry; eligible counts are shown. "
              "Graph duplicates ignore node order and edge direction after rounding coordinates to six decimals. "
              "Semantic layout distance is one minus IoU across room-type masks on a fixed 64×64 version of the 512-pixel canvas. "
              "Near-duplicate fraction is the share with another layout at distance ≤0.05. "
              "The raster measure is resolution-dependent and has no translation/rotation alignment. "
              "Higher distance alone is not evidence of better designs.", "",
              "| Method | Geometry eligible / selected | Raster eligible | Unique graph fraction | Mean layout distance | Near-duplicate fraction | Room-type JS divergence vs baseline |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for record in records:
        d, shift = record["diversity"], record["distribution_shift"]
        lines.append(f"| {record['label']} | {d['eligible_count']}/{d['input_count']} | {d['raster_eligible_count']} | {_percent(d['graph_unique_fraction'])} | "
                     f"{_number(d['mean_pairwise_semantic_distance'])} | {_percent(d['near_duplicate_fraction'])} | "
                     f"{_number(shift.get('room_type_js_divergence_bits') if shift else None)} |")
    lines += ["", "Graph uniqueness uses the geometry-eligible population; pairwise distances and near-duplicate fractions use raster-eligible "
              "layouts with a nonempty mask inside the drawing canvas. CSV/JSON also record Wasserstein distances for room, junction and edge counts and total interior-room area "
              "against a compatible generated baseline. Distances retain each feature's units. The JSON records which baseline was used. "
              "These are descriptive distribution diagnostics, not fidelity to real floorplans. RPLAN reference data is unavailable; no FID/KID is claimed.", "",
              "## Paired outputs and compute", "",
              "`tables/paired_validity.csv` records changes on shared trial seeds, including outputs only one method validates. "
              "It does not pair by the winning Best-of-N candidate's seed. Runs with different saved comparison settings or seed sets are not paired. "
              "Different seed batches remain separate in this report; no cross-batch pooling is performed.", "",
              "Denoiser-count comparisons are in `denoiser_comparison.json`. Best-of-11 targets full SVDD: 11,000 versus "
              "10,990 evaluations per output at the default settings. Late SVDD is the guidance-window ablation and uses a smaller budget. "
              "Matching denoiser counts does not match edge-model work, constraint checking or elapsed time. "
              "Timing is amortized per selected output and excludes model loading/rendering/export.", "",
              "## Annotated examples", "",
              f"[Open the complete sample gallery](index.html): {gallery['sample_count']} selected samples, each with PNG, SVG, vector graph JSON and evaluation JSON. "
              "Filters show methods, validity status and failed/unknown constraints. The examples below are the first trial of each status, not hand-picked successes.", "",
              "| Method | Status | Trial seed | Failed / unknown constraints | Artifacts |",
              "|---|---|---:|---|---|"]
    for label, statuses in gallery["examples"].items():
        for status, example in statuses.items():
            if example is None:
                lines.append(f"| {label} | {status} | — | No selected sample with this status | — |")
                continue
            failures = ", ".join(example["failed_ids"]) or "none"
            unknown = ", ".join(example["unknown_ids"]) or "none"
            lines.append(f"| {label} | {status} | {example['seed']} | {failures} / {unknown} | "
                         f"[PNG]({example['png']}) · [SVG]({example['svg']}) · [graph]({example['graph']}) · [checks]({example['evaluation']}) |")
    lines += ["", "## Scope and reproducibility", "",
              "`analysis.json` preserves per-run summaries, parameters, scenario, checkpoint metadata, recorded source hashes, "
              "definitions and diagnostic details. Assumed scale and height remain assumptions; bedroom placement searches are finite "
              "and bathrooms exclude unobserved fixtures. Unknown checks add no numeric severity, so zero energy alone is not proof of validity.", "",
              "Saved input files are not changed. The analysis creates rendered views from the saved evaluations, without rerunning geometry checks or the pretrained networks."]
    if skipped:
        lines += ["", "### Excluded runs", ""]
        lines.extend(f"- `{entry['run']}`: {entry['reason']}" for entry in skipped)
    (output / "REPORT.md").write_text("\n".join(lines) + "\n")


def analyze(experiment_dir, output_dir=None):
    root = Path(experiment_dir).resolve()
    records, skipped = load_records(root)
    if not records:
        raise ValueError("No completed runs with selected outputs found; finish a method before analysing it.")
    output = Path(output_dir).resolve() if output_dir else root / "analysis"
    output.mkdir(parents=True, exist_ok=True)
    from .analysis_gallery import build_gallery
    from .analysis_plots import plot_analysis
    from .diversity import diversity_summary, distribution_shift

    for record in records:
        record["metrics"] = method_metrics(record)
        record["constraint_rows"] = constraint_metrics(record)
        record["diversity"] = diversity_summary(record["rows"])
        reference = baseline_for(record, records)
        record["distribution_shift"] = distribution_shift(record["rows"], reference["rows"]) if reference else None
        record["distribution_baseline"] = reference["label"] if reference else None
        record["metrics"]["comparison_baseline"] = record["distribution_baseline"]
        for key, name in (("wall_seconds_per_trial", "runtime_multiplier_vs_baseline"),
                          ("node_samples_per_trial", "denoiser_multiplier_vs_baseline")):
            value = record["metrics"][key]
            denominator = reference["summary"]["compute"].get(key) if reference else None
            record["metrics"][name] = value / denominator if value is not None and denominator else None
    paired = paired_validity(records)
    write_csv(output / "tables/methods.csv", [record["metrics"] for record in records])
    write_csv(output / "tables/constraints.csv", [row for record in records for row in record["constraint_rows"]])
    write_csv(output / "tables/paired_validity.csv", paired,
              ["method_a", "method_b", "paired_outputs", "both_valid", "only_a_valid", "only_b_valid", "neither_valid", "b_minus_a_validity_points"])
    write_csv(output / "tables/diversity.csv", [{"method": record["label"], **record["diversity"]} for record in records])
    shifts = [{"method": record["label"], "baseline": record["distribution_baseline"], "feature": feature, **values}
              for record in records if record["distribution_shift"]
              for feature, values in record["distribution_shift"]["features"].items()]
    write_csv(output / "tables/distribution_shift.csv", shifts)
    write_json(output / "denoiser_comparison.json", denoiser_comparisons(root, [record["summary"] for record in records]))
    figures = plot_analysis(records, output / "figures")
    gallery = build_gallery(records, output)
    write_analysis_report(output, records, skipped, figures, gallery)
    payload = {"experiment_dir": str(root), "selected_outputs_only": True, "skipped_runs": skipped,
               "methods": [{key: value for key, value in record.items() if key not in {"rows", "run_dir"}}
                           | {"run_dir": str(record["run_dir"])} for record in records],
               "paired_validity": paired, "gallery": gallery,
               "figures": {name: {kind: str(Path(path).relative_to(output)) for kind, path in paths.items()}
                           for name, paths in figures.items()}}
    write_json(output / "analysis.json", payload)
    return {"output_dir": str(output), "methods": len(records), "selected_outputs": gallery["sample_count"],
            "skipped_runs": skipped}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment_dir", "--experiment-dir", type=Path, required=True)
    parser.add_argument("--output_dir", "--output-dir", type=Path)
    args = parser.parse_args()
    try:
        result = analyze(args.experiment_dir, args.output_dir)
    except ImportError as error:
        parser.exit(2, f"Analysis dependency missing: {error}. Install test_time/requirements-analysis.txt.\n")
    except ValueError as error:
        parser.exit(2, str(error) + "\n")
    print(f"Analysed {result['methods']} completed runs and {result['selected_outputs']} selected outputs. "
          f"Open {result['output_dir']}/REPORT.md or index.html.")
    if result["skipped_runs"]:
        print(f"Skipped {len(result['skipped_runs'])} unfinished/unusable runs; see the report.")


if __name__ == "__main__":
    main()
