"""Run late hard-feasibility or repair-aware SVDD, counting unsuccessful attempts.

The authoritative episode ledger is trials.jsonl. Unlike legacy selected-output
analysis, it includes hard-h episodes that return no graph at all.
"""

import argparse
from collections import Counter
from dataclasses import asdict
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

from .types import Scenario, overall_validity, read_json, write_json


METHODS = ("hard_h_mc", "svdd_repair_aware")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def check_outcomes(evaluation):
    if evaluation is None:
        return {}
    data = asdict(evaluation) if not isinstance(evaluation, dict) else evaluation
    return {r["id"]: r["status"] for r in data["results"]
            if r.get("counts_toward_compliance", True) and r["applicable"]}


def summarize_trials(rows, method):
    """Failures remain in all per-attempt rates and compute denominators."""
    count = len(rows)
    returned = [r for r in rows if r["returned_graph"]]
    success = sum(r["overall_validity"] == "valid" for r in rows)
    total_seconds = sum(r["stats"].get("wall_seconds", 0) for r in rows)
    total_nfe = sum(r["stats"].get("node_samples", 0) for r in rows)
    per_constraint = {}
    for row in returned:
        for cid, status in row["checks"].items():
            per_constraint.setdefault(cid, Counter())[status] += 1
    return {
        "method": method, "attempts": count, "returned_graphs": len(returned),
        "successful_attempts": success,
        "success_rate": success / count if count else None,
        "any_feasible_found": sum(bool(r["stats"].get("any_feasible_found", r["overall_validity"] == "valid")) for r in rows),
        "no_graph_attempts": count - len(returned),
        "structurally_valid_attempts": sum(r.get("structurally_valid", False) for r in rows),
        "outcomes": dict(Counter(r["status"] for r in rows)),
        "total_wall_seconds": total_seconds, "total_nfe": total_nfe,
        "seconds_per_attempt": total_seconds / count if count else None,
        "nfe_per_attempt": total_nfe / count if count else None,
        "seconds_per_success": total_seconds / success if success else None,
        "nfe_per_success": total_nfe / success if success else None,
        "constraint_outcomes_on_returned_graphs": {cid: dict(v) for cid, v in per_constraint.items()},
        "constraint_pass_per_attempt": {cid: v.get("pass", 0) / count for cid, v in per_constraint.items()} if count else {},
        "first_success_nfe": [(r["stats"].get("first_success") or {}).get("node_samples", r["stats"].get("first_success_nfe")) for r in rows],
        "total_edge_samples": sum(r["stats"].get("edge_samples", 0) for r in rows),
        "total_node_calls": sum(r["stats"].get("node_calls", 0) for r in rows),
        "total_edge_calls": sum(r["stats"].get("edge_calls", 0) for r in rows),
        "fallback_count": sum(r["stats"].get("fallback_count", 0) for r in rows),
        "all_zero_count": sum(r["stats"].get("all_zero_count", 0) for r in rows),
        "checker_seconds": (sum(r["stats"].get("checker_seconds", r["stats"].get("cpu_checker_seconds", 0)) for r in rows)
                            if rows and all("checker_seconds" in r["stats"] or "cpu_checker_seconds" in r["stats"] for r in rows) else None),
        "denominator": "all completed attempts, including no-graph and budget-exhausted outcomes",
        "timing_scope": "sampling, lookaheads, repair and checking; excludes loading and artifact export",
    }


def number(value, digits=2):
    return "—" if value is None else f"{value:,.{digits}f}"


def write_summary(output, rows, method):
    summary = summarize_trials(rows, method)
    write_json(output / "summary.json", summary)
    lines = [f"# {method}", "",
             f"**{summary['successful_attempts']}/{summary['attempts']} successful attempts**; "
             f"{summary['returned_graphs']} returned graphs, {summary['no_graph_attempts']} attempts without a graph.", "",
             f"Attempts finding any feasible graph (including lookaheads): {summary['any_feasible_found']}.", "",
             "| NFE/attempt | Seconds/attempt | NFE/success | Seconds/success |",
             "|---:|---:|---:|---:|",
             f"| {number(summary['nfe_per_attempt'], 0)} | {number(summary['seconds_per_attempt'])} | "
             f"{number(summary['nfe_per_success'], 0)} | {number(summary['seconds_per_success'])} |", "",
             "The denominator includes unsuccessful and budget-exhausted episodes. Zero successes give undefined "
             "cost per success, not zero cost. `trials.jsonl` is the authoritative episode ledger; "
             "`candidates.jsonl` contains returned graphs only and must not be used alone to estimate success rates.", "",
             "| Constraint | Pass | Fail | Unknown | Pass / all attempts |",
             "|---|---:|---:|---:|---:|"]
    for cid, counts in sorted(summary["constraint_outcomes_on_returned_graphs"].items()):
        lines.append(f"| {cid} | {counts.get('pass', 0)} | {counts.get('fail', 0)} | {counts.get('unknown', 0)} | "
                     f"{summary['constraint_pass_per_attempt'][cid]:.1%} |")
    lines += ["", "No-graph attempts cannot be assigned individual constraint outcomes. "
              "Lookahead diagnostics are saved in `events.jsonl`, with successful lookahead graphs under `lookaheads/`. "
              "Returned graphs, audit evidence and PNG/SVG views are in `samples/`; repair runs also save the original graph.", "",
              "See `config.json` for the exact policy, assumptions, checkpoints and source hashes; "
              "`traces/` records decisions and edit provenance. GPU repair scores are proxies; "
              "reported final checks use the audited CPU evaluator. These configurations are not compute matched.", ""]
    (output / "report.md").write_text("\n".join(lines))
    return summary


def summarize_root(root):
    root = Path(root).resolve()
    records = []
    for method in METHODS:
        path = root / method / "summary.json"
        if path.exists():
            record = read_json(path)
            run_path = path.parent / "run.json"
            run = read_json(run_path) if run_path.exists() else {}
            records.append({**record, "run_status": run.get("status", "unknown"),
                            "planned_trials": run.get("planned_trials")})
    if not records:
        raise ValueError(f"No extension summaries found under {root}")
    fields = ["method", "run_status", "planned_trials", "attempts", "successful_attempts", "success_rate", "any_feasible_found",
              "no_graph_attempts", "nfe_per_attempt", "seconds_per_attempt", "nfe_per_success", "seconds_per_success"]
    with (root / "comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    lines = ["# Late extension experiments", "",
             "All rates include unsuccessful attempts. These methods have different compute budgets; "
             "NFE and measured runtime are reported separately. Results from repair-aware guidance include final repair.", "",
             "| Method | Run status | Successful / attempted | Success rate | Any feasible found | NFE/attempt | Seconds/attempt | Seconds/success |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    for r in records:
        rate = "—" if r["success_rate"] is None else f"{r['success_rate']:.1%}"
        lines.append(f"| [{r['method']}]({r['method']}/report.md) | {r['run_status']} | {r['successful_attempts']}/{r['attempts']} | {rate} | "
                     f"{r['any_feasible_found']} | {number(r['nfe_per_attempt'], 0)} | "
                     f"{number(r['seconds_per_attempt'])} | {number(r['seconds_per_success'])} |")
    (root / "comparison.md").write_text("\n".join(lines) + "\n")
    print(f"Saved {root / 'comparison.md'}")


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).parent / "configs/extensions_rplan18.json")
    parser.add_argument("--method", choices=METHODS)
    parser.add_argument("--workers", type=int, default=1,
                        help="Independent processes sharing the device; split the trial seeds (default: 1)")
    for name in ("trials", "seed", "inference_batch_size", "k", "rollouts", "terminal_k", "max_nfe"):
        parser.add_argument("--" + name, type=int)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--node-checkpoint")
    parser.add_argument("--edge-checkpoint")
    parser.add_argument("--output_dir", type=Path)
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--summarize_root", type=Path)
    return parser


def load_configuration(args):
    config_path = args.config.resolve()
    config = read_json(config_path)
    for name in ("trials", "seed", "inference_batch_size"):
        value = getattr(args, name)
        if value is not None:
            config[name] = value
    for name in ("trials", "inference_batch_size"):
        if type(config[name]) is not int or config[name] < 1:
            raise ValueError(f"{name} must be a positive integer")
    if type(config["seed"]) is not int or config["seed"] < 0:
        raise ValueError("seed must be a nonnegative integer")
    method = args.method
    parameters = dict(config[method])
    for name in ("k", "rollouts", "terminal_k", "max_nfe"):
        value = getattr(args, name)
        if value is not None:
            if method != "hard_h_mc" and name != "k":
                raise ValueError(f"--{name} applies only to hard_h_mc")
            parameters[name] = value
    if type(parameters["k"]) is not int or parameters["k"] < 1:
        raise ValueError("k must be a positive integer")
    config[method] = parameters
    scenario = Scenario(**read_json(config_path.parent / config["scenario"]))
    if scenario.coordinate_scale_m_per_model_unit is None or scenario.ceiling_height_m is None or scenario.intended_occupants is None:
        raise ValueError("The extension scenario requires explicit scale, ceiling height and occupants")
    from .constraints import load_specs
    specs = load_specs(config_path.parent / config["constraints"])
    return config, parameters, scenario, specs


def run(args):
    config, parameters, scenario, specs = load_configuration(args)
    from .adapter import GSDiffAdapter, file_hash
    from .audit_geometry import refined_bedroom_search
    from .evaluate import constraint_selection
    from .policy import ConstraintEvaluator
    from .visualize import save_sample
    import torch

    repo = Path(__file__).resolve().parents[1]
    output = (args.output_dir or repo / config["output_root"] / args.method).resolve()
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}; choose a new directory")
    evaluator = ConstraintEvaluator(specs, validity_weight=config["validity_weight"],
                                    bedroom_search_fn=refined_bedroom_search)
    policy_data = {"scenario": asdict(scenario), "constraints": [asdict(s) for s in specs],
                   "validity_weight": config["validity_weight"], "checker": "cpu_bedroom_audit_v2"}
    policy_hash = digest(policy_data)
    started_setup = time.perf_counter()
    adapter = GSDiffAdapter(repo=repo, node_checkpoint=args.node_checkpoint, edge_checkpoint=args.edge_checkpoint,
                           device=args.device, variant=config["variant"])
    if args.method == "hard_h_mc":
        if max(parameters["anchor_remaining"]) > adapter.steps * 0.2:
            raise ValueError("Hard-h anchors must remain within the last 20% of transitions")
        from .hard_feasibility import hard_h_mc as sampler
        tensor_evaluator = None
    else:
        if parameters["guided_steps"] > adapter.steps * 0.2:
            raise ValueError("Repair guidance must remain within the last 20%")
        from .repair_sampling import repair_aware_svdd as sampler
        from .gpu_constraints import TensorConstraintEvaluator
        tensor_evaluator = TensorConstraintEvaluator(specs, validity_weight=config["validity_weight"])
    before = adapter.parameter_hash()
    setup_seconds = time.perf_counter() - started_setup
    try:
        commit = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        commit = None
    output.mkdir(parents=True, exist_ok=False)
    snapshot = {**config, "method": args.method, "parameters": parameters, **policy_data,
                "policy_sha256": policy_hash, "adapter": adapter.metadata, "repo_commit": commit,
                "format": "extension_attempt_ledger_v1", "constraint_selection": constraint_selection(specs),
                "source_sha256": {str(p.relative_to(repo)): file_hash(p) for p in sorted((repo / "test_time").glob("*.py"))},
                "guidance_backend": "unguided_GPU_rollouts_with_CPU_audited_outcomes" if args.method == "hard_h_mc" else "GPU_repair_proxy",
                "final_evaluator": "cpu_bedroom_audit_v2", "exact_conditional": False,
                "timing_scope": "generation and policy/repair work, excluding setup and artifact export",
                "trial_batch_size": 1, "candidate_batching": "neural and repair candidates batched within each episode"}
    write_json(output / "config.json", snapshot)
    state = {"method": args.method, "status": "running", "planned_trials": config["trials"],
             "completed_trials": 0, "setup_seconds": setup_seconds, "parameters_unchanged": None}
    write_json(output / "run.json", state)
    print(f"Run: {output}\nMethod: {args.method}; final checker=cpu_bedroom_audit_v2; "
          f"attempts={config['trials']}; inference_batch_size={config['inference_batch_size']}", flush=True)
    rows = []
    wall_start = time.perf_counter()
    if adapter.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(adapter.device)

    def export(sample, evaluation, prefix):
        sample.scenario = asdict(scenario)
        write_json(prefix.with_suffix(".json"), sample.to_dict())
        evidence = prefix.with_name(prefix.name + "-evaluation.json")
        write_json(evidence, asdict(evaluation))
        paths = {**save_sample(sample, evaluation, prefix),
                 "graph": prefix.with_suffix(".json"), "evaluation": evidence}
        return {key: str(Path(path).relative_to(output)) for key, path in paths.items()}

    try:
        with (output / "trials.jsonl").open("w") as trials_file, \
             (output / "events.jsonl").open("w") as events_file, \
             (output / "candidates.jsonl").open("w") as candidates_file:
            for trial in range(config["trials"]):
                seed = config["seed"] + trial
                export_seconds = 0.0
                event_count = 0
                first_success_export_seconds = None

                def event_callback(sample, evaluation, event):
                    nonlocal export_seconds, event_count, first_success_export_seconds
                    start = time.perf_counter()
                    event_count += 1
                    event_id = f"trial-{trial:04d}-event-{event_count:05d}"
                    record = {**event, "id": event_id, "trial": trial, "trial_seed": seed,
                              "overall_validity": overall_validity(evaluation),
                              "structurally_valid": evaluation.structurally_valid,
                              "checks": check_outcomes(evaluation), "policy_sha256": policy_hash,
                              "graph_sha256": digest(sample.to_dict())}
                    if record["overall_validity"] == "valid":
                        if first_success_export_seconds is None:
                            first_success_export_seconds = export_seconds
                        prefix = output / "lookaheads" / event_id
                        write_json(prefix.with_suffix(".json"), sample.to_dict())
                        write_json(prefix.with_name(prefix.name + "-evaluation.json"), asdict(evaluation))
                        record["graph_path"] = str(prefix.with_suffix(".json").relative_to(output))
                    events_file.write(json.dumps(record, allow_nan=False) + "\n")
                    events_file.flush()
                    export_seconds += time.perf_counter() - start

                options = {"inference_batch_size": config["inference_batch_size"],
                           "progress": None if args.no_progress else f"{args.method} {trial + 1}/{config['trials']}"}
                if args.method == "hard_h_mc":
                    result = sampler(adapter, evaluator, scenario, seed, **parameters,
                                     event_callback=event_callback, **options)
                else:
                    repair_parameters = {key: value for key, value in parameters.items() if key != "repair"}
                    result = sampler(adapter, evaluator, scenario, seed, **repair_parameters,
                                     repair_config=parameters["repair"], tensor_evaluator=tensor_evaluator, **options)
                result.stats["wall_seconds_including_event_export"] = result.stats["wall_seconds"]
                result.stats["event_export_seconds"] = export_seconds
                result.stats["wall_seconds"] = max(0.0, result.stats["wall_seconds"] - export_seconds)
                result.stats["timing_scope"] = "sampling_and_checking_excluding_setup_and_artifact_export"
                if result.stats.get("first_success") is not None:
                    first = result.stats["first_success"]
                    first["wall_seconds_including_event_export"] = first["wall_seconds"]
                    first["wall_seconds"] = max(0.0, first["wall_seconds"] - (first_success_export_seconds or 0.0))
                # Failed searches have no selected graph, but still get a complete episode row.
                returned = result.sample is not None
                if returned != (result.evaluation is not None):
                    raise RuntimeError("Sampler returned an incomplete sample/evaluation pair")
                validity = overall_validity(result.evaluation) if returned else "no_graph"
                artifacts = {}
                if returned:
                    result.sample.metadata.update(method=args.method, trial=trial, trial_seed=seed,
                                                  policy_sha256=policy_hash, overall_validity=validity)
                    prefix = output / "samples" / f"trial-{trial:04d}-seed-{seed}"
                    if "pre_repair" in result.extra_samples:
                        result.sample.metadata["pre_repair_graph"] = str(
                            prefix.with_name(prefix.name + "-pre_repair.json").relative_to(output))
                    artifacts = export(result.sample, result.evaluation, prefix)
                    if "pre_repair" in result.extra_samples:
                        from .repair import save_repair_overlay
                        overlay = save_repair_overlay(result.extra_samples["pre_repair"][0], result.sample,
                                                      prefix.with_name(prefix.name + "-repair.svg"))
                        if overlay:
                            artifacts["repair_overlay"] = str(Path(overlay).relative_to(output))
                    candidate = {"trial": trial, "seed": seed, "method": args.method, "selected": True,
                                 "candidate_id": prefix.name, "sample": result.sample.to_dict(),
                                 "evaluation": asdict(result.evaluation), "overall_validity": validity,
                                 "stats": result.stats, "artifacts": artifacts}
                    candidates_file.write(json.dumps(candidate, allow_nan=False) + "\n")
                    candidates_file.flush()
                extra = {}
                for name, (sample, evaluation) in result.extra_samples.items():
                    extra[name] = export(sample, evaluation, output / "samples" / f"trial-{trial:04d}-seed-{seed}-{name}")
                write_json(output / "traces" / f"trial-{trial:04d}.json", result.trace)
                row = {"trial": trial, "seed": seed, "method": args.method, "status": result.status,
                       "returned_graph": returned, "overall_validity": validity,
                       "structurally_valid": bool(returned and result.evaluation.structurally_valid),
                       "checks": check_outcomes(result.evaluation), "stats": result.stats,
                       "artifacts": artifacts, "extra_artifacts": extra, "policy_sha256": policy_hash}
                rows.append(row)
                trials_file.write(json.dumps(row, allow_nan=False) + "\n")
                trials_file.flush()
                state["completed_trials"] = len(rows)
                write_json(output / "run.json", state)
                write_summary(output, rows, args.method)
                failed = [cid for cid, outcome in row["checks"].items() if outcome == "fail"]
                unknown = [cid for cid, outcome in row["checks"].items() if outcome == "unknown"]
                print(f"{args.method} trial {trial + 1}/{config['trials']}: seed={seed}, status={result.status}, "
                      f"overall={validity}, failed={failed}, unknown={unknown}, "
                      f"NFE={result.stats.get('node_samples', 0)}, seconds={result.stats['wall_seconds']:.2f}", flush=True)
                current_policy = {**policy_data, "scenario": asdict(scenario), "constraints": [asdict(s) for s in specs]}
                if digest(current_policy) != policy_hash:
                    raise RuntimeError("Policy or scenario changed during sampling")
        state["parameters_unchanged"] = before == adapter.parameter_hash()
        if not state["parameters_unchanged"]:
            raise RuntimeError("Frozen model parameters changed")
        state["status"] = "complete"
    except (Exception, KeyboardInterrupt) as error:
        state.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                     error=f"{type(error).__name__}: {error}")
        raise
    finally:
        state["end_to_end_seconds"] = time.perf_counter() - wall_start
        state["summary"] = write_summary(output, rows, args.method)
        if adapter.device.type == "cuda":
            state["peak_gpu_allocated_bytes"] = torch.cuda.max_memory_allocated(adapter.device)
        write_json(output / "run.json", state)
    print(f"Saved {output / 'report.md'}", flush=True)
    return output


def main(argv=None):
    parser = _parser()
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.summarize_root:
        summarize_root(args.summarize_root)
        return
    if args.method is None:
        parser.error("--method is required unless --summarize_root is used")
    if args.workers > 1:
        from .parallel_extensions import run_parallel
        from .extension_workers import summarize_workers
        try:
            output = run_parallel(args)
        except BaseException:
            if getattr(args, "parallel_output", None) is not None:
                try:
                    summarize_workers(args.parallel_output, args.method)
                except Exception as error:
                    print(f"Could not summarize partial workers: {error}", file=sys.stderr)
            raise
        summarize_workers(output, args.method)
        print(f"Saved {output / 'report.md'}", flush=True)
        return
    run(args)


if __name__ == "__main__":
    main()
