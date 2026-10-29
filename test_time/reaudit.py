"""Recheck originally selected designs with a versioned CPU audit; no inference or reselection."""

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import time

from tqdm.auto import tqdm

from .analyze import load_records, write_csv
from .policy import ConstraintEvaluator
from .types import ConstraintSpec, FloorplanSample, Scenario, overall_validity, read_json, write_json


CHECKER_VERSION = "cpu_bedroom_audit_v2"


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _evaluate_task(task):
    from .audit_geometry import refined_bedroom_search

    row, config = task
    evaluator = ConstraintEvaluator([ConstraintSpec(**spec) for spec in config["constraints"]],
                                    validity_weight=config.get("validity_weight", 100.0),
                                    bedroom_search_fn=refined_bedroom_search)
    started = time.perf_counter()
    evaluation = evaluator.evaluate(FloorplanSample.from_dict(row["sample"]), Scenario(**config["scenario"]))
    return asdict(evaluation), time.perf_counter() - started


def _counts(rows):
    return dict(Counter(overall_validity(row["evaluation"]) for row in rows))


def _constraint_counts(rows):
    counts = {}
    for row in rows:
        for result in row["evaluation"]["results"]:
            if result.get("counts_toward_compliance", True):
                counts.setdefault(result["id"], Counter())[result["status"]] += 1
    return {key: dict(value) for key, value in counts.items()}


def _write_audit_report(output, audit):
    lines = ["# CPU checker audit", "",
             f"Checker: **{audit['checker_version']}**. Source: `{audit['source_experiment']}`.", "",
             "The same originally selected designs were rechecked. No generation, reward-guided sampling "
             "or Best-of-N reselection was performed. Original evaluations are retained in each row as "
             "`original_evaluation`; the original sweep is unchanged.", "",
             "Only selected outputs are exported here; rejected candidates remain in the source sweep. "
             "The assumed scale, bed dimensions, clearance envelope and permitted orientations are unchanged. "
             "Search misses remain unknown, not inferred failures.", "",
             "The audit adds clearance-boundary placements, rechecks beds with refined circle positions, "
             "and searches the free room geometry for a turning circle when needed. It handles floating-point "
             "wall contacts with an explicit numerical tolerance and certifies failure when the room's bounding "
             "box cannot contain any permitted clearance envelope.", "",
             "| Method | Outputs | Overall valid: original → audited | Overall unknown: original → audited | C035 pass / fail / unknown: original → audited |",
             "|---|---:|---:|---:|---|", ]
    for method in audit["methods"]:
        old, new = method["original_counts"], method["audited_counts"]
        def c035(key):
            values = method[key].get("C035", {})
            return " / ".join(str(values.get(status, 0)) for status in ("pass", "fail", "unknown"))
        lines.append(f"| {method['method']} | {method['selected_outputs']} | "
                     f"{old.get('valid', 0)} → {new.get('valid', 0)} | "
                     f"{old.get('unknown', 0)} → {new.get('unknown', 0)} | "
                     f"{c035('original_constraint_counts')} → {c035('audited_constraint_counts')} |")
    lines += ["", "[Revised figures, metrics and gallery](analysis/REPORT.md)", "",
              "`changes.csv` lists changed scored-check outcomes; `audit.json` records all per-method "
              "counts, source/checker hashes and audit timings. Each run's `audit.json` records its structural "
              "and overall-status changes. Unresolved cases retain detailed search diagnostics.", "",
              f"Offline audit wall time: **{audit['audit_wall_seconds']:.1f} s** with {audit['workers']} worker(s). "
              "This excludes analysis rendering. Generation time and NFE in the revised plots are copied from "
              "the original runs and exclude this additional audit.", ""]
    (output / "AUDIT.md").write_text("\n".join(lines))


def reaudit(experiment_dir, output_dir=None, workers=4):
    root = Path(experiment_dir).resolve()
    output = Path(output_dir).resolve() if output_dir else root.with_name(root.name + "_cpu_audit_v2")
    if workers < 1:
        raise ValueError("workers must be positive")
    if output == root or root in output.parents or output in root.parents:
        raise ValueError("Use a separate sibling output directory, outside the original sweep")
    if output.exists():
        raise FileExistsError(f"Audit output already exists: {output}; choose a new directory")
    records, skipped = load_records(root)
    if not records:
        raise ValueError("No completed runs with selected outputs found")
    checker_files = ["audit_geometry.py", "geometry.py", "constraints.py", "policy.py", "area.py", "types.py", "reaudit.py"]
    checker_hashes = {name: file_hash(Path(__file__).parent / name) for name in checker_files}
    output.mkdir(parents=True)
    started = time.perf_counter()
    audit = {"checker_version": CHECKER_VERSION, "status": "running",
             "created_at": datetime.now(timezone.utc).isoformat(), "source_experiment": str(root),
             "scope": "originally_selected_outputs", "no_reselection": True, "generation_unchanged": True,
             "workers": workers, "checker_sha256": checker_hashes, "skipped_runs": skipped, "methods": []}
    write_json(output / "audit.json", audit)
    changes = []
    pool = ProcessPoolExecutor(max_workers=workers) if workers > 1 else None
    try:
        for record in records:
            source = record["run_dir"]
            relative = source.relative_to(root) if source != root else Path(source.name)
            destination = output / relative
            destination.mkdir(parents=True)
            source_hashes = {name: file_hash(source / name) for name in ("config.json", "run.json", "candidates.jsonl")}
            metadata = {"checker_version": CHECKER_VERSION, "scope": audit["scope"],
                        "source_run_dir": str(source), "source_hashes": source_hashes,
                        "checker_sha256": checker_hashes, "no_reselection": True, "generation_unchanged": True}
            config = {**record["config"], "evaluation_audit": metadata}
            write_json(destination / "config.json", config)
            run = read_json(source / "run.json")
            write_json(destination / "run.json", {**run, "status": "running"})
            rows = sorted((row for line in (source / "candidates.jsonl").read_text().splitlines()
                           if line.strip() and (row := json.loads(line)).get("selected")), key=lambda row: row["trial"])
            tasks = ((row, record["config"]) for row in rows)
            evaluations = pool.map(_evaluate_task, tasks) if pool else map(_evaluate_task, tasks)
            revised, task_seconds = [], []
            method_start = time.perf_counter()
            for row, (evaluation, seconds) in tqdm(zip(rows, evaluations), total=len(rows), desc=record["label"]):
                updated = {**row, "original_evaluation": row["evaluation"], "evaluation": evaluation,
                           "overall_validity": overall_validity(evaluation), "audit_seconds": seconds}
                if "artifacts" in updated:
                    updated["original_artifacts"] = updated.pop("artifacts")
                revised.append(updated)
                task_seconds.append(seconds)
                original_checks = {check["id"]: check for check in row["evaluation"]["results"]}
                for check in evaluation["results"]:
                    previous = original_checks.get(check["id"], {})
                    if check["counts_toward_compliance"] and previous.get("status") != check["status"]:
                        changes.append({"method": record["label"], "trial": row["trial"],
                                        "candidate_id": row.get("candidate_id", ""), "constraint": check["id"],
                                        "original_status": previous.get("status", "missing"), "audited_status": check["status"],
                                        "original_overall": overall_validity(row["evaluation"]),
                                        "audited_overall": overall_validity(evaluation),
                                        "original_reason": previous.get("explanation", ""), "audited_reason": check["explanation"]})
            method_audit = {**metadata, "method": record["label"], "selected_outputs": len(rows),
                            "source_saved_candidates": record["summary"]["saved_final_candidates"],
                            "original_counts": _counts(rows), "audited_counts": _counts(revised),
                            "original_constraint_counts": _constraint_counts(rows),
                            "audited_constraint_counts": _constraint_counts(revised),
                            "structural_status_changes": sum(a["evaluation"]["structurally_valid"] != b["evaluation"]["structurally_valid"]
                                                             for a, b in zip(rows, revised)),
                            "overall_status_changes": sum(overall_validity(a["evaluation"]) != overall_validity(b["evaluation"])
                                                          for a, b in zip(rows, revised)),
                            "audit_wall_seconds": time.perf_counter() - method_start,
                            "summed_sample_audit_seconds": sum(task_seconds)}
            if any(file_hash(source / name) != value for name, value in source_hashes.items()):
                raise RuntimeError(f"Source changed during audit: {source}")
            with (destination / "candidates.jsonl").open("w") as stream:
                for row in revised:
                    stream.write(json.dumps(row, allow_nan=False) + "\n")
            write_json(destination / "audit.json", method_audit)
            write_json(destination / "run.json", run)
            audit["methods"].append(method_audit)
            write_json(output / "audit.json", audit)
    finally:
        if pool:
            pool.shutdown()
    if (root / "compute_budget.json").is_file():
        shutil.copyfile(root / "compute_budget.json", output / "compute_budget.json")
    audit.update(status="complete", audit_wall_seconds=time.perf_counter() - started,
                 selected_outputs=sum(method["selected_outputs"] for method in audit["methods"]),
                 changed_check_outcomes=len(changes))
    write_json(output / "audit.json", audit)
    write_csv(output / "changes.csv", changes,
              ["method", "trial", "candidate_id", "constraint", "original_status", "audited_status",
               "original_overall", "audited_overall", "original_reason", "audited_reason"])
    _write_audit_report(output, audit)
    return {"output_dir": str(output), "selected_outputs": audit["selected_outputs"],
            "changed_check_outcomes": len(changes)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    result = reaudit(args.experiment_dir, args.output_dir, args.workers)
    print(f"Audited {result['selected_outputs']} fixed outputs; {result['changed_check_outcomes']} check outcomes changed.")
    print(f"Report: {result['output_dir']}/AUDIT.md")
    print(f"Refresh figures: python -m test_time.analyze --experiment_dir {result['output_dir']}")


if __name__ == "__main__":
    main()
