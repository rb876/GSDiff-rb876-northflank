"""Run a reproducible baseline, Best-of-N, or SVDD-PM experiment."""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import time

from .types import Scenario, overall_validity, read_json, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).parent / "configs/experiment_v2.json")
    parser.add_argument("--method", required=True, choices=["baseline", "best_of_n", "svdd_pm"])
    for name in ("trials", "seed", "n", "k", "guided_steps"):
        parser.add_argument("--" + name, type=int)
    parser.add_argument("--batch_size", type=int, help="Independent design trials generated together (default config: 4)")
    parser.add_argument("--inference_batch_size", type=int,
                        help="Maximum states per neural-network call, including proposals (default config: 10)")
    parser.add_argument("--reward_backend", choices=("cpu", "gpu"),
                        help="Reward backend (default: gpu); final validation uses the CPU checker")
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--output_dir", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--node-checkpoint")
    parser.add_argument("--edge-checkpoint")
    parser.add_argument("--no-progress", action="store_true", help="Hide the diffusion progress bar")
    parser.add_argument("--list-constraints", action="store_true",
                        help="List configured checks and supporting rules without loading models")
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = {"batch_size": 4, "inference_batch_size": 10, "reward_backend": "gpu", **read_json(config_path)}
    config.update({name: value for name, value in vars(args).items()
                   if name in {"method", "trials", "seed", "n", "k", "guided_steps", "temperature",
                               "batch_size", "inference_batch_size", "reward_backend"} and value is not None})
    if config["trials"] < 1 or config["seed"] < 0:
        parser.error("trials must be positive and seed nonnegative")
    if any(type(config[key]) is not int or config[key] < 1 for key in ("batch_size", "inference_batch_size")):
        parser.error("batch_size and inference_batch_size must be positive integers")
    if config["reward_backend"] not in {"cpu", "gpu"}:
        parser.error("reward_backend must be cpu or gpu")
    scenario = Scenario(**read_json(config_path.parent / config["scenario"]))
    from .constraints import load_specs
    from .policy import ConstraintEvaluator
    specs = load_specs(config_path.parent / config["constraints"])
    from .evaluate import constraint_selection
    selection = constraint_selection(specs)
    if args.list_constraints:
        print(json.dumps(selection, indent=2))
        return
    print("Scored constraints: " + ", ".join(selection["active_scored_ids"]), flush=True)
    print("Supporting rules: " + ", ".join(selection["active_supporting_ids"]), flush=True)
    if selection["disabled_constraints"]:
        print("Not assessed: " + ", ".join(item["id"] for item in selection["disabled_constraints"]), flush=True)
    evaluator = ConstraintEvaluator(specs, validity_weight=config["validity_weight"])
    from .adapter import GSDiffAdapter, file_hash
    from . import sampling
    from .evaluate import write_report
    from .visualize import save_sample
    import torch

    repo = Path(__file__).resolve().parents[1]
    setup_start = time.perf_counter()
    try:
        adapter = GSDiffAdapter(repo=repo, node_checkpoint=args.node_checkpoint,
                               edge_checkpoint=args.edge_checkpoint, device=args.device,
                               variant=config["variant"])
    except (FileNotFoundError, ImportError) as error:
        parser.exit(2, f"Setup needed: {error}\n")
    before = adapter.parameter_hash()
    tensor_evaluator = None
    if config["reward_backend"] == "gpu":
        from .gpu_constraints import TensorConstraintEvaluator
        tensor_evaluator = TensorConstraintEvaluator(specs, validity_weight=config["validity_weight"])
    setup_seconds = time.perf_counter() - setup_start
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S-%fZ")
    suffix = {"baseline": "", "best_of_n": f"-n{config['n']}",
              "svdd_pm": f"-k{config['k']}-s{config['guided_steps']}-t{config['temperature']:g}"}[args.method]
    output = args.output_dir or repo / config["output_root"] / f"{args.method}{suffix}-{stamp}"
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    try:
        commit = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        commit = None
    snapshot = {**config, "scenario": asdict(scenario), "constraints": [asdict(spec) for spec in specs],
                "constraint_selection": selection,
                "adapter": adapter.metadata, "repo_commit": commit,
                "guidance": {"backend": config["reward_backend"], "device": str(adapter.device),
                             "tensor_settings": tensor_evaluator.metadata if tensor_evaluator else None,
                             "final_evaluator": "cpu_reference",
                             "trace_format": "scores_and_selections" if tensor_evaluator else "full_candidate_graphs"},
                "source_sha256": {str(path.relative_to(repo)): file_hash(path)
                                  for path in sorted([*(repo / "test_time").glob("*.py"),
                                                      repo / "gsdiff/house_nn1.py", repo / "gsdiff/house_nn2.py"])},
                "metric_scope": "Assumption-based scenario and geometric feasibility checks; not verified building compliance.",
                "selection_order": "best structural validity class; no new unknown IDs relative to its first candidate; fewest unknown checks, energy, seed",
                "runtime_scope": "sum of generation-batch sampling plus policy evaluation, each batch counted once; excludes loading/rendering/serialization",
                "trial_timing_scope": "batch wall time divided by batch trial count; amortized throughput, not isolated trial latency"}
    snapshot.update(n=(1 if args.method == "baseline" else config["n"] if args.method == "best_of_n" else None),
                    k=config["k"] if args.method == "svdd_pm" else None,
                    guided_steps=config["guided_steps"] if args.method == "svdd_pm" else None,
                    temperature=config["temperature"] if args.method == "svdd_pm" else None)
    write_json(output / "config.json", snapshot)
    run = {"method": args.method, "status": "running", "planned_trials": config["trials"], "completed_trials": 0,
           "total_wall_seconds": 0.0, "setup_seconds": setup_seconds,
           "node_calls": 0, "edge_calls": 0, "graph_completions": 0,
           "node_samples": 0, "edge_samples": 0, "batches": [],
           "gpu_candidate_evaluations": 0, "cpu_candidate_evaluations": 0,
           "candidate_evaluations": 0, "fallback_count": 0, "peak_gpu_allocated_bytes": 0}
    write_json(output / "run.json", run)
    print(f"Run: {output}; batch_size={config['batch_size']}, inference_batch_size={config['inference_batch_size']}, "
          f"reward_backend={config['reward_backend']}, device={adapter.device}", flush=True)
    if adapter.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(adapter.device)
    started = time.perf_counter()
    try:
        with (output / "candidates.jsonl").open("w") as candidates_file:
            for first_trial in range(0, config["trials"], config["batch_size"]):
                trial_ids = list(range(first_trial, min(first_trial + config["batch_size"], config["trials"])))
                seeds = [config["seed"] + trial for trial in trial_ids]
                arguments = (adapter, evaluator, scenario, seeds)
                batch_options = {"inference_batch_size": config["inference_batch_size"],
                                 "progress": None if args.no_progress else
                                 f"{args.method} trials {first_trial + 1}-{trial_ids[-1] + 1}"}
                if args.method == "baseline":
                    results, batch_stats = sampling.baseline_batch(*arguments, **batch_options)
                elif tensor_evaluator is not None:
                    from . import gpu_sampling
                    if args.method == "best_of_n":
                        results, batch_stats = gpu_sampling.best_of_n_batch_gpu(
                            *arguments, n=config["n"], tensor_evaluator=tensor_evaluator, **batch_options)
                    else:
                        results, batch_stats = gpu_sampling.svdd_pm_batch_gpu(
                            *arguments, k=config["k"], guided_steps=config["guided_steps"],
                            temperature=config["temperature"], tensor_evaluator=tensor_evaluator, **batch_options)
                elif args.method == "best_of_n":
                    results, batch_stats = sampling.best_of_n_batch(*arguments, n=config["n"], **batch_options)
                else:
                    results, batch_stats = sampling.svdd_pm_batch(
                        *arguments, k=config["k"], guided_steps=config["guided_steps"],
                        temperature=config["temperature"], **batch_options)
                for key in ("node_calls", "edge_calls", "node_samples", "edge_samples", "graph_completions",
                            "candidate_evaluations", "fallback_count"):
                    run[key] += batch_stats.get(key, 0)
                run["gpu_candidate_evaluations"] += batch_stats.get("gpu_candidate_evaluations", 0)
                run["cpu_candidate_evaluations"] += batch_stats.get("cpu_candidate_evaluations", batch_stats["candidate_evaluations"])
                run["total_wall_seconds"] += batch_stats["wall_seconds"]
                run["batches"].append({"first_trial": first_trial, "trial_count": len(trial_ids), **batch_stats})
                if adapter.device.type == "cuda":
                    run["peak_gpu_allocated_bytes"] = torch.cuda.max_memory_allocated(adapter.device)
                    run["peak_gpu_reserved_bytes"] = torch.cuda.max_memory_reserved(adapter.device)
                for trial, seed, result in zip(trial_ids, seeds, results):
                    for index, candidate in enumerate(result.candidates):
                        candidate.candidate_id = f"trial-{trial:03d}-{candidate.candidate_id}"
                        candidate.sample.metadata.update({"config_file": "config.json", "trial": trial,
                                                          "trial_seed": seed, "candidate_id": candidate.candidate_id,
                                                          "active_scored_ids": selection["active_scored_ids"],
                                                          "not_assessed_ids": [item["id"] for item in selection["disabled_constraints"]],
                                                          "overall_validity": overall_validity(candidate.evaluation)})
                        prefix = output / "samples" / candidate.candidate_id
                        write_json(prefix.with_suffix(".json"), candidate.sample.to_dict())
                        artifacts = save_sample(candidate.sample, candidate.evaluation, prefix)
                        row = {**asdict(candidate), "trial": trial, "method": args.method,
                               "selected": index == result.selected_index,
                               "overall_validity": overall_validity(candidate.evaluation),
                               "artifacts": {key: str(Path(value).relative_to(output)) for key, value in artifacts.items()}}
                        candidates_file.write(json.dumps(row, allow_nan=False) + "\n")
                    candidates_file.flush()
                    if result.trace:
                        write_json(output / "traces" / f"trial-{trial:03d}.json", result.trace)
                    run["completed_trials"] += 1
                    write_json(output / "run.json", run)
                    chosen = result.candidates[result.selected_index]
                    checks = [check for check in chosen.evaluation.results
                              if check.counts_toward_compliance and check.applicable]
                    failed_ids = ",".join(check.id for check in checks if check.status == "fail")
                    unknown_ids = ",".join(check.id for check in checks if check.status == "unknown")
                    print(f"{args.method} trial {trial + 1}/{config['trials']}: seed={seed}, "
                          f"overall={overall_validity(chosen.evaluation)}, structural_valid={chosen.evaluation.structurally_valid}, "
                          f"energy={chosen.evaluation.energy:.4f}, unknown={chosen.evaluation.unknown_count}, "
                          f"failed=[{failed_ids}], unknown_ids=[{unknown_ids}], "
                          f"amortized_seconds={result.stats['wall_seconds']:.2f}", flush=True)
        run["parameters_unchanged"] = before == adapter.parameter_hash()
        if not run["parameters_unchanged"]:
            raise RuntimeError("Frozen model parameter check failed")
        run["status"] = "complete"
    except Exception as error:
        run.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        run["end_to_end_seconds"] = time.perf_counter() - started
        write_json(output / "run.json", run)
    summary = write_report(output)
    print(f"Saved {output / 'report.md'}; overall_validity={summary['overall_valid_rate']:.1%}, "
          f"structural_validity={summary['structural_valid_rate']:.1%}", flush=True)


if __name__ == "__main__":
    main()
