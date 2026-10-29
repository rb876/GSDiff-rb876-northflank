"""Select Best-of-N from a fixed prefix of an existing larger candidate bank.

Saved tensor scores determine selection. Only selected outputs receive a new
CPU audit; no tensor rescoring or model inference is performed.
"""

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time

import torch
from tqdm.auto import tqdm

from .audit_geometry import refined_bedroom_search
from .evaluate import final_evaluation_backend
from .gpu_sampling import STATUS_NAMES, tensor_selection
from .policy import ConstraintEvaluator
from .recalibrate import SOURCE_FILES, _checkpoint_identity, _validate_graph, file_hash
from .sampling import _candidate_seed
from .types import ConstraintSpec, FloorplanSample, Scenario, overall_validity, read_json, write_json


PROTOCOL = "best_of_n_prefix_v1"


def _load_pool(source, n):
    config, run = read_json(source / "config.json"), read_json(source / "run.json")
    if config.get("method") != "best_of_n" or config.get("reward_backend") != "gpu":
        raise ValueError("Source must be a GPU-scored Best-of-N run")
    if run.get("status") != "complete":
        raise ValueError("Source run must be complete")
    source_n, trials = config["n"], config["trials"]
    if type(n) is not int or not 1 <= n <= source_n:
        raise ValueError(f"n must be between 1 and the source budget {source_n}")
    checkpoints = _checkpoint_identity(config["adapter"])
    rows = [json.loads(line) for line in (source / "candidates.jsonl").read_text().splitlines() if line.strip()]
    groups = defaultdict(list)
    for row in rows:
        groups[row["trial"]].append(row)
        sample = row["sample"]
        if sample.get("scenario") != config["scenario"]:
            raise ValueError("Sample scenario differs from the saved scoring configuration")
        if _checkpoint_identity(sample.get("metadata", {})) != checkpoints:
            raise ValueError("Sample checkpoint identities differ from the source configuration")
        _validate_graph(sample)
    if set(groups) != set(range(trials)) or any(len(group) != source_n for group in groups.values()):
        raise ValueError("A complete source candidate bank is required, not only its previous winners")
    retained = []
    for trial in range(trials):
        group = sorted(groups[trial], key=lambda row: row["sample"]["metadata"].get("candidate_index", -1))
        if [row["sample"]["metadata"].get("candidate_index") for row in group] != list(range(source_n)):
            raise ValueError("Each source trial must contain every candidate_index exactly once")
        if sum(bool(row.get("selected")) for row in group) != 1:
            raise ValueError("Each source trial must have exactly one previous winner")
        if [row["seed"] for row in group] != [_candidate_seed(config["seed"] + trial, i) for i in range(source_n)]:
            raise ValueError("Candidate seeds differ from their saved trial/index order")
        retained.extend(group[:n])
    return config, run, retained


def _saved_tensor_scores(rows, specs):
    """Recover scores by ID; never mix selected CPU audits with tensor rewards."""
    ids = [spec.id for spec in specs if spec.enabled]
    records, statuses, applicable, rewards, valid, unknown = [], [], [], [], [], []
    for row in rows:
        record = row.get("selection_evaluation")
        if record is None:
            record = row["sample"].get("metadata", {}).get("guidance_evaluation")
        if record is None:
            raise ValueError("Every retained candidate needs a saved tensor selection_evaluation or guidance_evaluation")
        results = {item["id"]: item for item in record["results"]}
        if set(results) != set(ids) or len(record["results"]) != len(ids):
            raise ValueError("Saved tensor scores must cover exactly the enabled constraint IDs")
        if not isinstance(record.get("reward"), (int, float)) or not math.isfinite(record["reward"]):
            raise ValueError("Saved tensor reward must be finite")
        status = [STATUS_NAMES.index(results[key]["status"]) for key in ids]
        counted_unknowns = sum(results[spec.id]["status"] == "unknown" and results[spec.id]["applicable"]
                               for spec in specs if spec.enabled and spec.counts_toward_compliance)
        if record["unknown_count"] != counted_unknowns:
            raise ValueError("Saved tensor unknown_count disagrees with its constraint statuses")
        records.append(record)
        statuses.append(status)
        applicable.append([results[key]["applicable"] for key in ids])
        rewards.append(record["reward"])
        valid.append(record["structurally_valid"])
        unknown.append(record["unknown_count"])
    return records, {"status": torch.tensor(statuses, dtype=torch.long),
                     "applicable": torch.tensor(applicable, dtype=torch.bool),
                     "reward": torch.tensor(rewards, dtype=torch.float64),
                     "structurally_valid": torch.tensor(valid, dtype=torch.bool),
                     "unknown_count": torch.tensor(unknown, dtype=torch.long)}


def reselect(source_run, output_dir, n=3):
    source, output = Path(source_run).resolve(), Path(output_dir).resolve()
    if source == output or source in output.parents or output in source.parents:
        raise ValueError("Use a separate output run directory outside the source run")
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    source_hashes = {name: file_hash(source / name) for name in SOURCE_FILES}
    config, source_run_data, rows = _load_pool(source, n)
    specs = [ConstraintSpec(**spec) for spec in config["constraints"]]
    records, values = _saved_tensor_scores(rows, specs)
    trials, source_n = config["trials"], config["n"]
    steps, source_nfe = config["adapter"].get("steps"), source_run_data.get("node_samples")
    if type(steps) is not int or steps < 1 or source_nfe != trials * source_n * steps:
        raise ValueError("Recorded source NFE must verify one complete trajectory per candidate")
    checker = final_evaluation_backend(config)
    if checker not in {"cpu_reference", "cpu_bedroom_audit_v2", "cpu_bathroom_audit_v1"}:
        raise ValueError(f"Unsupported final CPU checker: {checker}")
    scenario = Scenario(**config["scenario"])
    evaluator = ConstraintEvaluator(specs, config.get("validity_weight", 100.0),
        bedroom_search_fn=None if checker == "cpu_reference" else refined_bedroom_search)
    started = time.perf_counter()
    selection = tensor_selection(values, trials, n,
        tie_seeds=torch.tensor([row["seed"] for row in rows], dtype=torch.long).reshape(trials, n))
    selected = {trial * n + int(selection["chosen"][trial]) for trial in range(trials)}
    metadata = {"protocol": PROTOCOL, "created_at": datetime.now(timezone.utc).isoformat(),
                "source_run_dir": str(source), "source_hashes": source_hashes,
                "source_n": source_n, "n": n, "retained_candidate_indices": list(range(n)),
                "saved_candidates": len(rows), "selected_outputs": trials,
                "selection_rule": "Original structural/unknown-ID/energy/seed rule applied only to the fixed candidate-index prefix",
                "selected_evaluation_backend": checker, "population_evaluation_backend": "mixed_tensor_cpu",
                "new_denoiser_evaluations": 0, "model_inference_performed": False,
                "effective_node_samples": trials * n * steps, "source_node_samples": source_nfe,
                "compute_provenance": "Effective NFE of retained complete trajectories, verified against the larger source bank; no new generation. Original total generation cost is preserved in source_run.",
                "source_provenance": {key: config[key] for key in ("scale_recalibration", "evaluation_audit", "prefix_reselection") if key in config},
                "code_sha256": {name: file_hash(Path(__file__).parent / name) for name in
                                ("reselect.py", "gpu_sampling.py", "constraints.py", "policy.py", "audit_geometry.py", "bathroom_geometry.py")}}
    revised_config = deepcopy(config)
    for key in ("scale_recalibration", "evaluation_audit"):
        revised_config.pop(key, None)
    revised_config.update(n=n, output_root=str(output), final_checker=checker, prefix_reselection=metadata,
        runtime_scope="Offline prefix reselection and CPU auditing; no standalone generation runtime is available")
    revised_config.setdefault("guidance", {})["final_evaluator"] = checker
    output.mkdir(parents=True)
    write_json(output / "config.json", revised_config)
    run = {"method": "best_of_n", "status": "running", "planned_trials": trials, "completed_trials": 0,
           "total_wall_seconds": None, "node_samples": trials * n * steps,
           "node_calls": None, "edge_calls": None, "edge_samples": None,
           "source_run": source_run_data, "compute_provenance": metadata["compute_provenance"],
           "prefix_reselection": {"new_denoiser_evaluations": 0, "cpu_candidates": 0}}
    write_json(output / "run.json", run)
    cpu_seconds, outcomes = 0.0, Counter()
    try:
        with (output / "candidates.jsonl").open("w") as stream:
            for index, original in enumerate(tqdm(rows, desc=f"Best-of-{n}: replay and selected CPU audits", unit="candidate")):
                row = deepcopy(original)
                row.update(prefix_source_selected=original["selected"], prefix_source_evaluation=original["evaluation"],
                           prefix_source_stats=original.get("stats", {}), selected=index in selected,
                           selection_evaluation=deepcopy(records[index]),
                           stats={"wall_seconds": None, "timing_scope": "Offline prefix replay; generation runtime unmeasured"})
                if "artifacts" in row:
                    row["prefix_source_artifacts"] = row.pop("artifacts")
                sample_metadata = row["sample"].setdefault("metadata", {})
                sample_metadata["prefix_source_method_parameters"] = sample_metadata.get("method_parameters")
                sample_metadata.update(method_parameters={"n": n}, guidance_evaluation=deepcopy(records[index]),
                    prefix_reselection=PROTOCOL, final_evaluation_backend=checker if row["selected"] else None)
                audit_seconds = 0.0
                if row["selected"]:
                    audit_started = time.perf_counter()
                    row["evaluation"] = asdict(evaluator.evaluate(FloorplanSample.from_dict(row["sample"]), scenario))
                    audit_seconds = time.perf_counter() - audit_started
                    cpu_seconds += audit_seconds
                    row["evaluation_backend"] = checker
                    run["completed_trials"] += 1
                else:
                    row["evaluation"] = deepcopy(records[index])
                    row["evaluation_backend"] = "tensor_gpu"
                row["overall_validity"] = overall_validity(row["evaluation"])
                sample_metadata["overall_validity"] = row["overall_validity"]
                row["prefix_reselection_seconds"] = {"cpu_audit": audit_seconds}
                if row["selected"]:
                    outcomes[row["overall_validity"]] += 1
                stream.write(json.dumps(row, allow_nan=False) + "\n")
        if any(file_hash(source / name) != value for name, value in source_hashes.items()):
            raise RuntimeError("Source candidate bank changed during prefix reselection")
        write_json(output / "selection.json", [
            {"trial": trial, "selected_index": int(selection["chosen"][trial]),
             "coverage_anchor_index": int(selection["anchor"][trial]),
             "eligible_indices": selection["eligible"][trial].nonzero().flatten().tolist(),
             "fallback": bool(selection["fallback"][trial])} for trial in range(trials)])
        run["status"] = "complete"
    except BaseException as error:
        run.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        run["prefix_reselection"].update(wall_seconds=time.perf_counter() - started,
            cpu_audit_seconds=cpu_seconds, cpu_candidates=run["completed_trials"])
        write_json(output / "run.json", run)
    result = {**metadata, "status": "complete", "output_dir": str(output),
              "overall_validity_counts": dict(outcomes), "timing": run["prefix_reselection"]}
    write_json(output / "reselection.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_run", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--n", type=int, default=3)
    args = parser.parse_args(argv)
    result = reselect(args.source_run, args.output_dir, args.n)
    print(f"Saved Best-of-{args.n} to {args.output_dir}: {result['selected_outputs']} outputs, "
          f"{result['effective_node_samples'] // result['selected_outputs']} effective NFE/output. No model inference performed.")


if __name__ == "__main__":
    main()
