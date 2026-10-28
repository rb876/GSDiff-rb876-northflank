"""Summarize saved experiments without running either pretrained network."""

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, median

from .types import overall_validity, read_json, write_json


ROOM_NAMES = {0: "living", 1: "bedroom", 2: "storage", 3: "kitchen",
              4: "bathroom", 5: "balcony", 6: "outside"}


def fraction(numerator, denominator):
    return numerator / denominator if denominator else None


def describe(values):
    return {"count": len(values), "mean": mean(values) if values else None,
            "median": median(values) if values else None,
            "min": min(values) if values else None, "max": max(values) if values else None}


def counted_results(row):
    return [r for r in row["evaluation"]["results"] if r.get("counts_toward_compliance", True)]


def measurable(result):
    return (result["status"] in {"pass", "fail"}
            and result["applicable"] and result["measurable"])


def check_summary(results):
    counts = Counter(r["status"] for r in results)
    applicable = sum(bool(r["applicable"]) for r in results)
    measured = [r for r in results if measurable(r)]
    passes = sum(r["status"] == "pass" for r in measured)
    assumptions = sum(bool(r.get("assumption_flags")) for r in results if r["applicable"])
    severities = [r["severity"] for r in measured if r["severity"] is not None]
    return {"pass": passes, "fail": sum(r["status"] == "fail" for r in measured),
            "unknown": counts["unknown"], "not_applicable": counts["not_applicable"],
            "applicable_count": applicable, "measurable_count": len(measured),
            "assumption_based_count": assumptions,
            "compliance": fraction(passes, len(measured)),
            "measurable_fraction": fraction(len(measured), applicable),
            "unknown_fraction": fraction(counts["unknown"], applicable),
            "assumption_based_fraction": fraction(assumptions, applicable),
            "severity": describe(severities)}


def validity_summary(rows):
    counts = Counter(overall_validity(row["evaluation"]) for row in rows)
    structural_count = sum(bool(row["evaluation"]["structurally_valid"]) for row in rows)
    return {
        "overall_valid_count": counts["valid"],
        "overall_valid_rate": fraction(counts["valid"], len(rows)),
        "overall_validity_counts": {status: counts[status] for status in ("valid", "invalid", "unknown", "not_applicable")},
        "structural_valid_count": structural_count,
        "structural_valid_rate": fraction(structural_count, len(rows)),
        # Historical field names mean structural validity, never overall validity.
        "valid_count": structural_count,
        "validity_rate": fraction(structural_count, len(rows)),
    }


def polygon_area(points):
    if len(points) < 3:
        return 0.0
    return abs(sum(a[0] * b[1] - b[0] * a[1]
                   for a, b in zip(points, points[1:] + points[:1]))) / 2


def comparison_group(config):
    adapter = config.get("adapter", {})
    checkpoints = {name: data.get("sha256") for name, data in adapter.get("checkpoints", {}).items()}
    identity = {"scenario": config.get("scenario"), "constraints": config.get("constraints"),
                "variant": adapter.get("variant"), "steps": adapter.get("steps"),
                "checkpoints": checkpoints}
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:12]


def summarize_run(run_dir):
    run_dir = Path(run_dir)
    config = read_json(run_dir / "config.json")
    run = read_json(run_dir / "run.json")
    rows = [json.loads(line) for line in (run_dir / "candidates.jsonl").read_text().splitlines() if line.strip()]
    selected = [row for row in rows if row.get("selected", False)]
    if not selected:
        raise ValueError(f"{run_dir}: no selected candidates; complete the run before evaluating")
    trial_ids = [row["trial"] for row in selected]
    if len(set(trial_ids)) != len(trial_ids):
        raise ValueError(f"{run_dir}: expected exactly one selected candidate per trial")

    results = [result for row in selected for result in counted_results(row)]
    per_id = defaultdict(list)
    auxiliary = defaultdict(list)
    for row in selected:
        for result in row["evaluation"]["results"]:
            target = per_id if result.get("counts_toward_compliance", True) else auxiliary
            target[result["id"]].append(result)
    aggregate = check_summary(results)
    all_measurable_pass = 0
    all_applicable_pass = 0
    for row in selected:
        applicable = [r for r in counted_results(row) if r["applicable"]]
        known = [r for r in applicable if measurable(r)]
        all_measurable_pass += bool(known) and all(r["status"] == "pass" for r in known)
        all_applicable_pass += bool(applicable) and all(measurable(r) and r["status"] == "pass" for r in applicable)

    samples = [row["sample"] for row in selected]
    rooms = [room for sample in samples for room in sample["rooms"]]
    areas = [area for room in rooms if math.isfinite(area := polygon_area(room["polygon"]))]
    scale = config.get("scenario", {}).get("coordinate_scale_m_per_model_unit")
    categories = Counter(ROOM_NAMES.get(room["label"], str(room["label"])) for room in rooms)
    count = len(selected)
    expected_trials = config.get("trials", run.get("trials", count))
    status = run.get("status", "complete")
    if count != expected_trials:
        status = "incomplete"
    wall = run.get("total_wall_seconds")
    known_failures = [row for row in selected if any(r["status"] == "fail" for r in counted_results(row))
                      or not row["evaluation"]["structurally_valid"]]
    best = min(selected, key=lambda r: (not r["evaluation"]["structurally_valid"],
                                        r["evaluation"]["unknown_count"], r["evaluation"]["energy"], r["seed"]))
    worst = max(known_failures, key=lambda r: (not r["evaluation"]["structurally_valid"],
                                               sum(x["status"] == "fail" for x in counted_results(r)),
                                               r["evaluation"]["energy"])) if known_failures else None
    def example(row):
        return {"trial": row["trial"], "seed": row["seed"], "candidate_id": row.get("candidate_id", ""),
                "artifacts": row.get("artifacts", {}), "energy": row["evaluation"]["energy"],
                "failed_ids": [r["id"] for r in counted_results(row) if r["status"] == "fail"],
                "unknown_ids": [r["id"] for r in counted_results(row) if r["status"] == "unknown"]}
    return {
        "metric_schema_version": 2,
        "validity_definitions": {
            "overall": "Structurally valid and every applicable counted constraint measurably passes; at least one must apply.",
            "unknown": "No confirmed failure, but a required counted result is not measurable; not a confirmed valid sample.",
            "not_applicable": "Structurally valid, but no counted constraints apply; not a confirmed overall-valid sample.",
            "rate_denominator": "Every sample in the relevant selected/all-candidate population, including invalid, unknown and not_applicable.",
            "legacy_aliases": {"valid_count": "structural_valid_count", "validity_rate": "structural_valid_rate"},
        },
        "run_dir": str(run_dir), "method": config.get("method", run.get("method")),
        "parameters": {
            "n": 1 if config.get("method") == "baseline" else config.get("n") if config.get("method") == "best_of_n" else None,
            "k": config.get("k") if config.get("method") == "svdd_pm" else None,
            "guided_steps": config.get("guided_steps") if config.get("method") == "svdd_pm" else None,
            "temperature": config.get("temperature") if config.get("method") == "svdd_pm" else None,
        },
        "comparison_group": comparison_group(config), "scenario": config.get("scenario", {}),
        "status": status, "expected_trials": expected_trials,
        "selected_trials": count, "saved_final_candidates": len(rows),
        "selected_candidate_ids": [r.get("candidate_id", "") for r in selected],
        **validity_summary(selected),
        **aggregate,
        "all_measurable_pass_fraction": all_measurable_pass / count,
        "all_applicable_pass_fraction": all_applicable_pass / count,
        "per_constraint": {key: check_summary(value) for key, value in sorted(per_id.items())},
        "auxiliary_checks": {key: check_summary(value) for key, value in sorted(auxiliary.items())},
        "all_candidates": {
            "count": len(rows),
            **validity_summary(rows),
            **check_summary([result for row in rows for result in counted_results(row)]),
        },
        "graph_statistics": {
            "junctions": describe([len(sample["nodes"]) for sample in samples]),
            "edges": describe([len(sample["edges"]) for sample in samples]),
            "rooms": describe([len(sample["rooms"]) for sample in samples]),
            "room_categories": dict(sorted(categories.items())),
            "room_area_drawing_units2": describe(areas),
            "room_area_m2": describe([area * scale ** 2 for area in areas]) if scale else None,
            "reference_dataset": None,
            "note": "Generated-sample diagnostics only; no RPLAN reference, FID or KID was evaluated.",
        },
        "compute": {**{key: run.get(key) for key in (
            "total_wall_seconds", "node_calls", "edge_calls", "graph_completions", "candidate_evaluations",
            "peak_gpu_allocated_bytes", "peak_gpu_reserved_bytes", "fallback_count")},
            "wall_seconds_per_trial": wall / count if wall is not None else None},
        "examples": {"best": example(best), "failure": example(worst) if worst else None},
    }


def percent(value):
    return "unknown" if value is None else f"{100 * value:.1f}%"


def number(value):
    return "n/a" if value is None else f"{value:.3g}"


def write_report(run_dir):
    run_dir = Path(run_dir)
    result = summarize_run(run_dir)
    write_json(run_dir / "summary.json", result)
    scenario = result["scenario"]
    parameters = ", ".join(f"{key}={value}" for key, value in result["parameters"].items() if value is not None)
    lines = [f"# {result['method']} experiment", "",
             f"Status: **{result['status']}**. {result['selected_trials']}/{result['expected_trials']} selected trials; "
             f"{result['saved_final_candidates']} complete final candidates saved. {parameters}.", "",
             f"Scale: **{scenario.get('scale_provenance', 'missing')}**, {scenario.get('coordinate_scale_m_per_model_unit')} m/drawing pixel. "
             "Results under declared geometry/scenario assumptions are not verified real-world compliance.", "",
             "| Measure | Result |", "|---|---:|",
             f"| **Overall valid** | **{result['overall_valid_count']}/{result['selected_trials']} ({percent(result['overall_valid_rate'])})** |",
             f"| Structurally valid | {result['structural_valid_count']}/{result['selected_trials']} ({percent(result['structural_valid_rate'])}) |",
             f"| Overall invalid / unknown / not applicable | {result['overall_validity_counts']['invalid']} / "
             f"{result['overall_validity_counts']['unknown']} / {result['overall_validity_counts']['not_applicable']} |",
             f"| Measurable-constraint compliance | {percent(result['compliance'])} |",
             f"| Applicable checks | {result['applicable_count']} |",
             f"| Measurable / unknown coverage | {percent(result['measurable_fraction'])} / {percent(result['unknown_fraction'])} |",
             f"| Assumption-based applicable checks | {percent(result['assumption_based_fraction'])} |",
             f"| Mean / median normalized severity | {number(result['severity']['mean'])} / {number(result['severity']['median'])} |",
             f"| All measurable checks pass | {percent(result['all_measurable_pass_fraction'])} |",
             f"| All applicable checks measurably pass | {percent(result['all_applicable_pass_fraction'])} |",
             f"| Wall time per selected trial | {number(result['compute']['wall_seconds_per_trial'])} s |",
             f"| Node / edge forward calls | {result['compute']['node_calls']} / {result['compute']['edge_calls']} |",
             f"| Complete graph evaluations | {result['compute']['candidate_evaluations']} |", "",
             "| ID | Pass | Fail | Unknown | N/A | Compliance | Measurable coverage |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for cid, stats in result["per_constraint"].items():
        lines.append(f"| {cid} | {stats['pass']} | {stats['fail']} | {stats['unknown']} | {stats['not_applicable']} | "
                     f"{percent(stats['compliance'])} | {percent(stats['measurable_fraction'])} |")
    if result["auxiliary_checks"]:
        lines.extend(["", "Auxiliary checks excluded from compliance: " + ", ".join(result["auxiliary_checks"]) + "."])
    lines.extend(["", "**Overall valid** means structurally valid AND every applicable counted check measurably passes. "
                  "A structural defect or failed counted check makes it invalid; otherwise unresolved checks make it unknown. "
                  "Auxiliary checks and individual not-applicable checks are excluded. For a structurally valid graph with no applicable counted checks, "
                  "the overall status is not_applicable and does not count as success. All selected samples remain in the rate denominator.", "",
                  "Unknown and inapplicable checks contribute no numeric severity; "
                  "unknowns never count as passes. All-pass fractions use every selected trial, and require at least one evaluable/applicable check.", "",
                  "Legacy JSON fields `valid_count` and `validity_rate` remain aliases for **structural** validity; use `overall_valid_rate` for the headline.", "",
                  f"Mean junctions / edges / rooms: {number(result['graph_statistics']['junctions']['mean'])} / "
                  f"{number(result['graph_statistics']['edges']['mean'])} / {number(result['graph_statistics']['rooms']['mean'])}. "
                  "No RPLAN reference data: distribution statistics are descriptive, with no FID/KID claim.", ""])
    for kind, item in result["examples"].items():
        if item is None:
            lines.append("No observed failure example in this run; this does not establish general reliability.")
            continue
        image = item["artifacts"].get("png")
        label = f"[{kind} example]({image})" if image else f"{kind.capitalize()} example"
        lines.append(f"{label}: trial {item['trial']}, seed {item['seed']}; failed={item['failed_ids']}, unknown={item['unknown_ids']}.")
    lines.extend(["", "Full graph, per-check measurements, thresholds and candidate selections: `candidates.jsonl`. "
                  "Reproduction inputs: `config.json`; measured compute: `run.json`.", ""])
    (run_dir / "report.md").write_text("\n".join(lines))
    return result


def write_comparison(experiment_dir, summaries):
    experiment_dir = Path(experiment_dir)
    baselines = {s["comparison_group"]: s for s in summaries if s["method"] == "baseline" and s["status"] == "complete"}
    rows = []
    for s in summaries:
        baseline = baselines.get(s["comparison_group"]) if s["status"] == "complete" else None
        runtime = s["compute"]["wall_seconds_per_trial"]
        reference = baseline["compute"]["wall_seconds_per_trial"] if baseline else None
        rows.append({"run": str(Path(s["run_dir"]).resolve().relative_to(experiment_dir.resolve())), "method": s["method"], **s["parameters"],
                     "status": s["status"],
                     "comparison_group": s["comparison_group"], "trials": s["selected_trials"],
                     "final_candidates": s["saved_final_candidates"],
                     "overall_valid_count": s["overall_valid_count"], "overall_valid_rate": s["overall_valid_rate"],
                     "overall_invalid_count": s["overall_validity_counts"]["invalid"],
                     "overall_unknown_count": s["overall_validity_counts"]["unknown"],
                     "overall_not_applicable_count": s["overall_validity_counts"]["not_applicable"],
                     "structural_valid_count": s["structural_valid_count"], "structural_valid_rate": s["structural_valid_rate"],
                     "compliance": s["compliance"],
                     "measurable_fraction": s["measurable_fraction"], "unknown_fraction": s["unknown_fraction"],
                     "assumption_based_fraction": s["assumption_based_fraction"], "validity_rate": s["validity_rate"],
                     "validity_rate_alias_of": "structural_valid_rate",
                     "mean_severity": s["severity"]["mean"], "median_severity": s["severity"]["median"],
                     "wall_seconds_per_trial": runtime,
                     "compute_multiplier": runtime / reference if runtime is not None and reference else None,
                     "node_calls": s["compute"]["node_calls"], "edge_calls": s["compute"]["edge_calls"],
                     "candidate_evaluations": s["compute"]["candidate_evaluations"]})
    if not rows:
        raise ValueError("No completed experiment runs found")
    with (experiment_dir / "comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# Measured comparison", "", "Runtime multipliers use measured wall time per selected trial, relative to a baseline with the same scenario, "
             "constraint definitions and checkpoint hashes. Setup/inference timing boundaries are recorded in each run. Small runs are plumbing checks, not statistical evidence.", "",
             "| Run | Status | Settings | Trials | Overall valid | Structurally valid | Compliance | Measurable | Unknown checks | Severity | Seconds/trial | Compute |",
             "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        settings = ", ".join(f"{key}={row[key]}" for key in ("n", "k", "guided_steps", "temperature") if row[key] is not None)
        lines.append(f"| [{row['run']}]({row['run']}/report.md) | {row['status']} | {settings} | {row['trials']} | "
                     f"{percent(row['overall_valid_rate'])} | {percent(row['structural_valid_rate'])} | {percent(row['compliance'])} | "
                     f"{percent(row['measurable_fraction'])} | {percent(row['unknown_fraction'])} | "
                     f"{number(row['mean_severity'])} | {number(row['wall_seconds_per_trial'])} | {number(row['compute_multiplier'])}x |")
    matched = []
    for index, a in enumerate(rows):
        for b in rows[index + 1:]:
            ta, tb = a["wall_seconds_per_trial"], b["wall_seconds_per_trial"]
            if (a["status"] == b["status"] == "complete" and a["comparison_group"] == b["comparison_group"]
                    and ta and tb and max(ta, tb) / min(ta, tb) <= 1.2):
                matched.append(f"`{a['run']}` vs `{b['run']}` ({max(ta, tb) / min(ta, tb):.2f}x time ratio)")
    lines.extend(["", "Overall valid = structurally valid AND all applicable counted checks measurably pass. "
                  "A failure makes the result invalid; unresolved checks without a failure remain unknown. "
                  "A structurally valid graph with no applicable counted checks is not_applicable, not a success. All selected samples remain in the denominator. "
                  "CSV `validity_rate` is a legacy alias for `structural_valid_rate`, not overall validity.", "",
                  "Approximately matched compute (within 20% wall time per trial): " + ("; ".join(matched) if matched else "none measured yet") + ".", "",
                  "C034 is auxiliary where configured; see per-run reports for check counts and assumptions. "
                  "Unknown values are not zero violations or passes. No RPLAN-based FID/KID was measured.", ""])
    (experiment_dir / "comparison.md").write_text("\n".join(lines))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment_dir", "--experiment-dir", type=Path, required=True)
    args = parser.parse_args()
    directory = args.experiment_dir
    runs = [directory] if (directory / "candidates.jsonl").exists() else sorted(
        p.parent for p in directory.rglob("candidates.jsonl") if (p.parent / "run.json").exists())
    summaries = [write_report(run) for run in runs]
    write_comparison(directory, summaries)
    print(f"Wrote {len(summaries)} run reports and {directory / 'comparison.md'}")


if __name__ == "__main__":
    main()
