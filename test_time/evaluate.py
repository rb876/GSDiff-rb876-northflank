"""Summarize saved experiments without running either pretrained network."""

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict, is_dataclass
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, median

from .types import overall_validity, read_json, write_json


ROOM_NAMES = {0: "living", 1: "bedroom", 2: "storage", 3: "kitchen",
              4: "bathroom", 5: "balcony", 6: "outside"}


class _UnfinishedRunError(ValueError):
    """A readable run has not yet saved its complete selected population."""


def _supporting(record):
    # Legacy C034 may carry the new default role='check' while its historical
    # explicit denominator flag remains false. Never turn that into a check.
    return record.get("role") == "supporting_rule" or not record.get("counts_toward_compliance", True)


def _spec_dicts(specs):
    return [asdict(spec) if is_dataclass(spec) else dict(spec) for spec in (specs or [])]


def constraint_selection(specs):
    """Describe the saved registry without treating disabled IDs as results."""
    specs = _spec_dicts(specs)
    scored, supporting, disabled = [], [], []
    for spec in specs:
        role = "supporting_rule" if _supporting(spec) else "check"
        if not spec.get("enabled", True):
            disabled.append({"id": spec["id"], "group": spec.get("group") or "ungrouped", "role": role,
                             "reason": spec.get("disabled_reason") or "Disabled in saved configuration; reason not recorded"})
        else:
            (supporting if role == "supporting_rule" else scored).append(spec["id"])
    support_count = sum(_supporting(spec) for spec in specs)
    return {"active_scored_ids": scored, "active_supporting_ids": supporting, "disabled_constraints": disabled,
            "constraint_counts": {"registered": len(specs), "scored": len(specs) - support_count,
                                  "supporting": support_count, "active": len(scored) + len(supporting),
                                  "active_scored": len(scored), "active_supporting": len(supporting),
                                  "disabled": len(disabled)}}


def _configured_rows(rows, specs):
    """Apply saved enable/role metadata to reporting views, preserving files."""
    by_id = {spec["id"]: spec for spec in specs}
    complete_registry = bool(specs) and all("enabled" in spec for spec in specs)
    output = []
    for row in rows:
        results = []
        for result in row["evaluation"]["results"]:
            spec = by_id.get(result["id"], {})
            if not spec.get("enabled", True):
                continue
            role_recorded = "role" in spec or "counts_toward_compliance" in spec
            support = _supporting(spec) if role_recorded else _supporting(result)
            results.append({**result, "counts_toward_compliance": not support,
                            "role": "supporting_rule" if support else "check",
                            "group": spec.get("group") or result.get("group") or "ungrouped"})
        if complete_registry:
            observed = {result["id"] for result in results}
            for spec in specs:
                if spec["enabled"] and not _supporting(spec) and spec["id"] not in observed:
                    results.append({"id": spec["id"], "status": "unknown", "severity": None,
                                    "applicable": True, "measurable": False, "counts_toward_compliance": True,
                                    "role": "check", "group": spec.get("group") or "ungrouped",
                                    "claim_level": "unobservable", "reporting_missing_result": True,
                                    "source_id": spec.get("source_id"), "assumption_flags": [],
                                    "explanation": "Active check absent from saved evaluation; applicability and measurement unconfirmed."})
        output.append({**row, "evaluation": {**row["evaluation"], "results": results,
                                               "unknown_count": sum(result["status"] == "unknown" and not _supporting(result)
                                                                    for result in results)}})
    return output


def fraction(numerator, denominator):
    return numerator / denominator if denominator else None


def describe(values):
    return {"count": len(values), "mean": mean(values) if values else None,
            "median": median(values) if values else None,
            "min": min(values) if values else None, "max": max(values) if values else None}


def counted_results(row):
    return [r for r in row["evaluation"]["results"] if not _supporting(r)]


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


def group_summary(rows, specs):
    """Group only active checks; supporting/disabled-only groups are unassessed."""
    by_id = {spec["id"]: spec for spec in specs}
    # Older saved configs can be partial. Retain observed result IDs rather than
    # silently dropping their existing metrics or inventing registry entries.
    for row in rows:
        for result in row["evaluation"]["results"]:
            by_id.setdefault(result["id"], result)
    groups = defaultdict(list)
    for spec in by_id.values():
        groups[spec.get("group") or "ungrouped"].append(spec)
    output = {}
    for name, entries in sorted(groups.items()):
        selected = constraint_selection(entries)
        ids = set(selected["active_scored_ids"])
        results, outcomes, missing_count = [], Counter(), 0
        for row in rows:
            checks = [result for result in counted_results(row) if result["id"] in ids]
            results.extend(checks)
            missing = ids - {result["id"] for result in checks}
            missing_count += len(missing) + sum(bool(result.get("reporting_missing_result")) for result in checks)
            evaluation = {**row["evaluation"], "results": checks}
            status = overall_validity(evaluation)
            if not ids:
                status = "not_assessed"
            elif missing and status != "invalid":
                status = "unknown"
            outcomes[status] += 1
        metrics = check_summary(results)
        output[name] = {"status": "assessed" if ids else "not_assessed",
                        "active_scored_ids": selected["active_scored_ids"],
                        "active_supporting_ids": selected["active_supporting_ids"],
                        "disabled_ids": [entry["id"] for entry in selected["disabled_constraints"]],
                        "selected_trials": len(rows), "missing_result_count": missing_count,
                        "overall_valid_count": outcomes["valid"],
                        "overall_valid_rate": fraction(outcomes["valid"], len(rows)) if ids else None,
                        "overall_validity_counts": {status: outcomes[status] for status in
                                                    ("valid", "invalid", "unknown", "not_applicable", "not_assessed")},
                        **metrics,
                        "check_pass_rate": fraction(metrics["pass"], metrics["applicable_count"]),
                        "check_pass_rate_denominator": "observed applicable active scored checks, including unknowns",
                        "claim_levels": sorted({result.get("claim_level") or "unrecorded" for result in results})}
    return output


def polygon_area(points):
    if len(points) < 3:
        return 0.0
    return abs(sum(a[0] * b[1] - b[0] * a[1]
                   for a, b in zip(points, points[1:] + points[:1]))) / 2


def final_evaluation_backend(config):
    """Identify the checker of selected outputs, including offline re-evaluation."""
    return ((config.get("scale_recalibration") or {}).get("selected_evaluation_backend")
            or (config.get("evaluation_audit") or {}).get("checker_version")
            or config.get("final_checker")
            or config.get("guidance", {}).get("final_evaluator")
            or config.get("final_evaluator")
            or "cpu_reference")


def comparison_group(config):
    adapter = config.get("adapter", {})
    checkpoints = {name: data.get("sha256") for name, data in adapter.get("checkpoints", {}).items()}
    identity = {"scenario": config.get("scenario"), "constraints": config.get("constraints"),
                "variant": adapter.get("variant"), "steps": adapter.get("steps"),
                "checkpoints": checkpoints, "batch_size": config.get("batch_size", 1),
                "inference_batch_size": config.get("inference_batch_size", 1),
                "reward_backend": config.get("reward_backend", "cpu"),
                "tensor_settings": config.get("guidance", {}).get("tensor_settings")}
    checker = final_evaluation_backend(config)
    if checker != "cpu_reference":
        identity["evaluation_checker_version"] = checker
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:12]


def denoiser_comparison_group(config):
    """Match evaluated-state budgets across batching without relaxing runtime matches."""
    return comparison_group({**config, "batch_size": 1, "inference_batch_size": 1})


def summarize_run(run_dir):
    run_dir = Path(run_dir)
    config = read_json(run_dir / "config.json")
    run = read_json(run_dir / "run.json")
    rows = [json.loads(line) for line in (run_dir / "candidates.jsonl").read_text().splitlines() if line.strip()]
    specs = _spec_dicts(config.get("constraints", []))
    selection = constraint_selection(specs)
    rows = _configured_rows(rows, specs)
    selected = [row for row in rows if row.get("selected", False)]
    if not selected:
        raise _UnfinishedRunError(f"{run_dir}: no selected candidates; complete the run before evaluating")
    trial_ids = [row["trial"] for row in selected]
    if len(set(trial_ids)) != len(trial_ids):
        raise ValueError(f"{run_dir}: expected exactly one selected candidate per trial")

    results = [result for row in selected for result in counted_results(row)]
    per_id = defaultdict(list)
    auxiliary = defaultdict(list)
    for row in selected:
        for result in row["evaluation"]["results"]:
            target = auxiliary if _supporting(result) else per_id
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
    recalibration = config.get("scale_recalibration")
    prefix_reselection = config.get("prefix_reselection")
    mixed_population = any((metadata or {}).get("population_evaluation_backend") == "mixed_tensor_cpu"
                           for metadata in (recalibration, prefix_reselection))
    all_candidates = ({"count": len(rows), "status": "not_reported",
                       "reason": "Selected outputs have CPU audits; rejected candidates retain tensor scores. "
                                 "Mixed checker results are not pooled into compliance or validity rates."}
                      if mixed_population else {
                          "count": len(rows), **validity_summary(rows),
                          **check_summary([result for row in rows for result in counted_results(row)])})
    return {
        "metric_schema_version": 3,
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
            "batch_size": config.get("batch_size", 1),
            "inference_batch_size": config.get("inference_batch_size", 1),
            "reward_backend": config.get("reward_backend", "cpu"),
        },
        "guidance": config.get("guidance", {"backend": "cpu", "final_evaluator": "cpu_reference"}),
        "evaluation_audit": config.get("evaluation_audit"),
        "scale_recalibration": recalibration,
        "prefix_reselection": prefix_reselection,
        "final_evaluation_backend": final_evaluation_backend(config),
        "comparison_group": comparison_group(config), "scenario": config.get("scenario", {}),
        "denoiser_comparison_group": denoiser_comparison_group(config),
        "status": status, "expected_trials": expected_trials,
        "selected_trials": count, "saved_final_candidates": len(rows),
        "trial_seed_start": config.get("seed"), "selected_trial_ids": sorted(trial_ids),
        "selected_candidate_ids": [r.get("candidate_id", "") for r in selected],
        **selection,
        "missing_active_results": [{"trial": row["trial"], "candidate_id": row.get("candidate_id", ""),
                                    "ids": [result["id"] for result in row["evaluation"]["results"]
                                            if result.get("reporting_missing_result")]}
                                   for row in selected if any(result.get("reporting_missing_result")
                                                              for result in row["evaluation"]["results"])],
        **validity_summary(selected),
        **aggregate,
        "all_measurable_pass_fraction": all_measurable_pass / count,
        "all_applicable_pass_fraction": all_applicable_pass / count,
        "per_constraint": {key: check_summary(value) for key, value in sorted(per_id.items())},
        "auxiliary_checks": {key: check_summary(value) for key, value in sorted(auxiliary.items())},
        "per_group": group_summary(selected, specs),
        "all_candidates": all_candidates,
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
            "total_wall_seconds", "node_calls", "edge_calls", "node_samples", "edge_samples", "graph_completions", "candidate_evaluations",
            "gpu_candidate_evaluations", "cpu_candidate_evaluations",
            "peak_gpu_allocated_bytes", "peak_gpu_reserved_bytes", "fallback_count")},
            "nfe_provenance": "derived_prefix_budget" if prefix_reselection else "recorded_run_counter",
            "wall_seconds_per_trial": wall / count if wall is not None else None,
            "node_samples_per_trial": run["node_samples"] / count if run.get("node_samples") is not None else None,
            "time_per_trial_interpretation": "amortized throughput, not isolated latency" if config.get("batch_size", 1) > 1
                                             else "wall time per selected trial"},
        "examples": {"best": example(best), "failure": example(worst) if worst else None},
    }


def percent(value):
    return "unknown" if value is None else f"{100 * value:.1f}%"


def number(value):
    return "n/a" if value is None else f"{value:.3g}"


def recorded_count(value):
    return "unknown" if value is None else str(value)


def write_report(run_dir, *, require_complete=False):
    run_dir = Path(run_dir)
    result = summarize_run(run_dir)
    if require_complete and result["status"] != "complete":
        raise _UnfinishedRunError(f"{run_dir}: run status is {result['status']}; "
                                  f"saved {result['selected_trials']} selected trials; expected {result['expected_trials']}")
    write_json(run_dir / "summary.json", result)
    scenario = result["scenario"]
    counts = result["constraint_counts"]
    disabled_note = (f" {counts['disabled']} IDs disabled; these are excluded from compliance denominators "
                     "and are not counted as unknown or passed." if counts["disabled"] else "")
    parameters = ", ".join(f"{key}={value}" for key, value in result["parameters"].items() if value is not None)
    audit = result["evaluation_audit"]
    recalibration = result["scale_recalibration"]
    prefix = result["prefix_reselection"]
    saved_population = (f"{result['saved_final_candidates']} originally selected outputs saved for this audit; "
                        "the original full candidate pool is excluded" if audit and not recalibration else
                        f"{result['saved_final_candidates']} complete final candidates saved")
    audit_note = ([f"Offline CPU audit: **{audit['checker_version']}**. Original selections and generation are unchanged; "
                   "no candidates were reselected. All saved-candidate statistics refer only to the originally selected outputs. "
                   "Generation timings and counters are copied from the original run and exclude this offline audit; "
                   "the sweep's `audit.json` records audit time separately. Guidance metadata describes the original generation.", ""]
                  if audit and not recalibration else [])
    if recalibration:
        audit_note = ["Offline scenario/constraint re-evaluation: selected-output metrics use the recorded CPU checker. "
                      "No new diffusion was run; denoiser counters describe historical generation, "
                      "and generation runtime was not remeasured.", ""]
        if result["all_candidates"].get("status") == "not_reported":
            audit_note += [result["all_candidates"]["reason"], ""]
    if prefix:
        audit_note = [f"Fixed-prefix Best-of-{result['parameters']['n']}: retain the first "
                      f"{result['parameters']['n']} candidates in each original trial before inspecting their scores, "
                      "then reselect with the saved tensor rewards and audit the selected output on CPU. "
                      "The reported NFE is the equivalent budget of the retained complete trajectories, derived from "
                      "verified source counts. The larger source pool's generation cost remains recorded in "
                      "`reselection.json`; no new model inference was performed. Generation runtime and batched "
                      "forward-call counts are unavailable for this derived run.", "",
                      result["all_candidates"]["reason"], ""]
    lines = [f"# {result['method']} experiment", "",
             f"Status: **{result['status']}**. {result['selected_trials']}/{result['expected_trials']} selected trials; "
             f"{saved_population}. {parameters}.", "", *audit_note,
             f"Scale: **{scenario.get('scale_provenance', 'missing')}**, {scenario.get('coordinate_scale_m_per_model_unit')} m/drawing pixel. "
             "Results under declared geometry/scenario assumptions are not verified real-world compliance.", "",
             f"Guidance backend: **{result['parameters']['reward_backend']}**. Reported final validity and compliance use the **CPU reference checker** "
             f"(`{result['final_evaluation_backend']}`). "
             "GPU guidance uses conservative finite searches and can differ from the reference; its scores are saved separately in sample metadata.", "",
             f"Saved registry: **{counts['registered']} IDs** ({counts['scored']} checks, {counts['supporting']} supporting rules). "
             f"Active: **{counts['active_scored']} scored checks** and {counts['active_supporting']} supporting rules. "
             "Overall validity covers only the active scored checks." + disabled_note, "",
             "Active scored IDs: " + (", ".join(result["active_scored_ids"]) or "none") + ".",
             "Active supporting IDs: " + (", ".join(result["active_supporting_ids"]) or "none") +
             ". Supporting rules define measurements and assumptions; they add no separate reward or pass/fail result.", "",
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
             f"| Node / edge forward calls | {recorded_count(result['compute']['node_calls'])} / {recorded_count(result['compute']['edge_calls'])} |",
             f"| Node / edge evaluated states | {recorded_count(result['compute']['node_samples'])} / {recorded_count(result['compute']['edge_samples'])} |",
             f"| Denoiser evaluations per selected output | {recorded_count(result['compute']['node_samples_per_trial'])} |",
             f"| Tensor / CPU-reference constraint evaluations | {recorded_count(result['compute']['gpu_candidate_evaluations'])} / {recorded_count(result['compute']['cpu_candidate_evaluations'])} |",
             f"| Complete graph evaluations | {result['compute']['candidate_evaluations']} |", "",
             "| ID | Pass | Fail | Unknown | N/A | Compliance | Measurable coverage |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for cid, stats in result["per_constraint"].items():
        lines.append(f"| {cid} | {stats['pass']} | {stats['fail']} | {stats['unknown']} | {stats['not_applicable']} | "
                     f"{percent(stats['compliance'])} | {percent(stats['measurable_fraction'])} |")
    if result["auxiliary_checks"]:
        lines.extend(["", "Auxiliary checks excluded from compliance: " + ", ".join(result["auxiliary_checks"]) + "."])
    if result["per_group"]:
        lines.extend(["", "| Group | Assessment | Active checks | Overall valid | Check pass rate | Unknown checks |",
                      "|---|---|---|---:|---:|---:|"])
        for group, stats in result["per_group"].items():
            assessed = stats["status"] != "not_assessed"
            valid = f"{stats['overall_valid_count']}/{stats['selected_trials']}" if assessed else "not_assessed"
            rate = percent(stats["check_pass_rate"]) if assessed else "not_assessed"
            lines.append(f"| {group} | {stats['status']} | {', '.join(stats['active_scored_ids']) or 'none'} | "
                         f"{valid} | {rate} | {stats['unknown']} |")
        lines.extend(["", "Group overall validity also requires a valid graph and every applicable active group check to pass. "
                      "Group check pass rates divide passes by observed applicable checks, including unknowns; "
                      "measurable-only compliance remains separate in JSON. "
                      "Missing active results prevent a group from being reported valid and are recorded in JSON."])
        if any(stats["status"] == "not_assessed" for stats in result["per_group"].values()):
            lines.extend(["", "Groups without active scored checks are not_assessed."])
    if result["disabled_constraints"]:
        lines.extend(["", "| Disabled ID | Group | Reason |", "|---|---|---|"])
        for entry in result["disabled_constraints"]:
            reason = str(entry["reason"]).replace("|", "\\|").replace("\n", " ")
            lines.append(f"| {entry['id']} | {entry['group']} | {reason} |")
    if result["missing_active_results"]:
        lines.extend(["", "Some saved evaluations omit active scored IDs. These are reported as unknown and unmeasured, "
                      "with unconfirmed applicability; they cannot establish an overall pass. "
                      "See `missing_active_results` in summary.json. Original saved evaluations were not modified."])
    lines.extend(["", "**Overall valid** means structurally valid AND every applicable counted check measurably passes. "
                  "A structural defect or failed counted check makes it invalid; otherwise unresolved checks make it unknown. "
                  "Auxiliary checks and individual not-applicable checks are excluded. For a structurally valid graph with no applicable counted checks, "
                  "the overall status is not_applicable and does not count as success. All selected samples remain in the rate denominator.", "",
                  "Unknown and inapplicable checks contribute no numeric severity; "
                  "unknowns never count as passes. All-pass fractions use every selected trial, and require at least one evaluable/applicable check.", "",
                  "Legacy JSON fields `valid_count` and `validity_rate` remain aliases for **structural** validity; use `overall_valid_rate` for the headline.", "",
                  f"Timing: {result['compute']['time_per_trial_interpretation']}. Total measured batch time is counted once and divided by the number of selected trials. "
                  "Forward calls count network invocations; evaluated states count the inputs processed across those invocations. "
                  "Missing state counts in older runs remain unknown. `batch_size` groups independent trials; `inference_batch_size` caps neural inputs per forward call.", "",
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


def denoiser_comparisons(experiment_dir, summaries):
    """Compare recorded denoiser work only for compatible, paired complete runs."""
    root = Path(experiment_dir).resolve()
    comparisons = []
    for best in (summary for summary in summaries if summary["method"] == "best_of_n"):
        for svdd in (summary for summary in summaries if summary["method"] == "svdd_pm"):
            if (best["status"] != "complete" or svdd["status"] != "complete"
                    or best.get("denoiser_comparison_group", best["comparison_group"])
                    != svdd.get("denoiser_comparison_group", svdd["comparison_group"])
                    or best.get("trial_seed_start") is None
                    or best["trial_seed_start"] != svdd.get("trial_seed_start")
                    or best["selected_trials"] != svdd["selected_trials"]
                    or best.get("selected_trial_ids") != svdd.get("selected_trial_ids")):
                continue
            left = best["compute"].get("node_samples_per_trial")
            right = svdd["compute"].get("node_samples_per_trial")
            known = all(value is not None and math.isfinite(value) and value > 0 for value in (left, right))
            ratio = left / right if known else None
            difference = abs(ratio - 1) if known else None
            comparisons.append({
                "best_of_n": str(Path(best["run_dir"]).resolve().relative_to(root)),
                "svdd": str(Path(svdd["run_dir"]).resolve().relative_to(root)),
                "trial_seed_start": best["trial_seed_start"], "trials": best["selected_trials"],
                "best_of_n_evaluations_per_output": left, "svdd_evaluations_per_output": right,
                "best_of_n_nfe_provenance": best["compute"].get("nfe_provenance", "recorded_run_counter"),
                "ratio_best_of_n_to_svdd": ratio, "relative_difference": difference,
                "exact": left == right if known else None,
                "matched": abs(left - right) <= 0.01 * right if known else None,
            })
    return {"metric": "node_samples_per_selected_output", "relative_tolerance": 0.01,
            "comparisons": comparisons}


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
                     "trial_seed_start": s.get("trial_seed_start"),
                     "final_candidates": s["saved_final_candidates"],
                     "registered_constraint_ids": s.get("constraint_counts", {}).get("registered"),
                     "active_scored_count": s.get("constraint_counts", {}).get("active_scored"),
                     "active_supporting_count": s.get("constraint_counts", {}).get("active_supporting"),
                     "disabled_constraint_count": s.get("constraint_counts", {}).get("disabled"),
                     "active_scored_ids": ",".join(s.get("active_scored_ids", [])),
                     "active_supporting_ids": ",".join(s.get("active_supporting_ids", [])),
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
                     "node_samples": s["compute"]["node_samples"], "edge_samples": s["compute"]["edge_samples"],
                     "node_samples_per_trial": s["compute"].get("node_samples_per_trial"),
                     "gpu_candidate_evaluations": s["compute"]["gpu_candidate_evaluations"],
                     "cpu_candidate_evaluations": s["compute"]["cpu_candidate_evaluations"],
                     "time_per_trial_interpretation": s["compute"]["time_per_trial_interpretation"],
                     "candidate_evaluations": s["compute"]["candidate_evaluations"]})
    if not rows:
        raise ValueError("No completed experiment runs found")
    with (experiment_dir / "comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    denoiser = denoiser_comparisons(experiment_dir, summaries)
    write_json(experiment_dir / "denoiser_comparison.json", denoiser)
    lines = ["# Measured comparison", "", "Runtime multipliers use measured wall time per selected trial, relative to a baseline with the same scenario, "
             "constraint definitions, checkpoint hashes, guidance backend and batching controls. Older configs without batching fields mean batch_size=1 and inference_batch_size=1. "
             "For batch_size > 1, seconds/trial is amortized throughput, not isolated latency. "
             "Setup/inference timing boundaries are recorded in each run. Small runs are plumbing checks, not statistical evidence.", "",
             "| Run | Status | Settings | Trials | Overall valid | Structurally valid | Compliance | Measurable | Unknown checks | Severity | Seconds/trial | Node/edge calls | Node/edge states | Runtime vs baseline |",
             "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        settings = ", ".join(f"{key}={row[key]}" for key in ("n", "k", "guided_steps", "temperature", "batch_size", "inference_batch_size", "reward_backend") if row[key] is not None)
        lines.append(f"| [{row['run']}]({row['run']}/report.md) | {row['status']} | {settings} | {row['trials']} | "
                     f"{percent(row['overall_valid_rate'])} | {percent(row['structural_valid_rate'])} | {percent(row['compliance'])} | "
                     f"{percent(row['measurable_fraction'])} | {percent(row['unknown_fraction'])} | "
                     f"{number(row['mean_severity'])} | {number(row['wall_seconds_per_trial'])} | "
                     f"{recorded_count(row['node_calls'])}/{recorded_count(row['edge_calls'])} | "
                     f"{recorded_count(row['node_samples'])}/{recorded_count(row['edge_samples'])} | {number(row['compute_multiplier'])}x |")
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
                  "Similar measured runtime (within 20% wall time per trial): " + ("; ".join(matched) if matched else "none measured yet") + ". "
                  "CSV `compute_multiplier` is the legacy name for this runtime multiplier, not a denoiser-count match.", "",
                  "Supporting rules (including C034/C079/C080 where configured) are excluded from compliance denominators. "
                  "CSV records active scored/supporting IDs and registry counts; per-run reports list group metrics. "
                  "Do not compare percentages across different active pools without their denominators and assumptions. "
                  "Unknown values are not zero violations or passes. Missing evaluated-state counts are unknown (empty CSV cells), not inferred from forward calls. "
                  "All final validity/compliance metrics use the CPU reference checker, including runs guided by tensor scores. "
                  "No RPLAN-based FID/KID was measured.", ""])
    if any(s.get("disabled_constraints") for s in summaries):
        lines.extend(["Disabled IDs are excluded from compliance denominators; per-run reports list their reasons.", ""])
    if any(s.get("prefix_reselection") for s in summaries):
        lines.extend(["Fixed-prefix Best-of-N controls reuse the first N candidates from each saved trial. "
                      "Their denoiser counts describe the equivalent retained-trajectory budget; no new model "
                      "inference was run. The original larger pool's full cost is preserved in `reselection.json`. "
                      "Runtime and batched forward-call counts are unknown for these derived controls.", ""])
    lines.extend(["## Denoiser evaluation matching", "",
                  "The selected compute metric is evaluated denoiser states (`node_samples`) per selected output, "
                  "not batched forward calls (`node_calls`) or wall time. Counts include every Best-of-N candidate "
                  "and every SVDD proposal. Matching here means within 1%; exact matches are identified separately. "
                  "Edge prediction, constraint checks and elapsed time are not matched by this criterion.", "",
                  "Only complete runs with matching scenario, model, constraints, checker, reward backend, tensor settings, "
                  "trial seed start and trial IDs are paired. Batching may differ for denoiser matching; runtime matching "
                  "still requires identical batching controls. Missing recorded counts remain unknown. "
                  "Details are saved in `denoiser_comparison.json`.", ""])
    if denoiser["comparisons"]:
        lines.extend(["| Best-of-N run | SVDD run | BoN evaluations/output | SVDD evaluations/output | BoN / SVDD | Difference | Matched |",
                      "|---|---|---:|---:|---:|---:|---|"])
        for pair in denoiser["comparisons"]:
            difference = pair["relative_difference"]
            difference_text = "unknown" if difference is None else f"{100 * difference:.3f}%"
            match = "unknown" if pair["matched"] is None else "exact" if pair["exact"] else "within 1%" if pair["matched"] else "no"
            ratio = pair["ratio_best_of_n_to_svdd"]
            ratio_text = "unknown" if ratio is None else f"{ratio:.6f}"
            lines.append(f"| {pair['best_of_n']} | {pair['svdd']} | "
                         f"{recorded_count(pair['best_of_n_evaluations_per_output'])} | "
                         f"{recorded_count(pair['svdd_evaluations_per_output'])} | "
                         f"{ratio_text} | {difference_text} | {match} |")
    else:
        lines.append("No compatible complete Best-of-N/SVDD pair with recorded trial seed metadata is available yet.")
    (experiment_dir / "comparison.md").write_text("\n".join(lines))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment_dir", "--experiment-dir", type=Path, required=True)
    args = parser.parse_args()
    directory = args.experiment_dir
    runs = [directory] if (directory / "candidates.jsonl").exists() else sorted(
        p.parent for p in directory.rglob("candidates.jsonl") if (p.parent / "run.json").exists())
    summaries = []
    for run in runs:
        try:
            summaries.append(write_report(run, require_complete=True))
        except _UnfinishedRunError as error:
            print(f"Skipping unfinished run: {error}")
    write_comparison(directory, summaries)
    print(f"Wrote {len(summaries)} run reports and {directory / 'comparison.md'}")


if __name__ == "__main__":
    main()
