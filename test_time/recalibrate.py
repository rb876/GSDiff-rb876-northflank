"""Reuse unguided trajectories under a new scale or explicit constraint registry.

No checkpoints are loaded. Selected outputs use the refined CPU checker; rejected
Best-of-N candidates retain tensor scores for reproducible selection, not CPU claims.
"""

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from tqdm.auto import tqdm

from .audit_geometry import refined_bedroom_search
from .constraints import load_specs
from .gpu_constraints import TensorConstraintEvaluator
from .gpu_graph import extract_rooms
from .gpu_sampling import SCORE_FIELDS, _evaluation_record, tensor_selection
from .policy import ConstraintEvaluator
from .sampling import _candidate_seed
from .types import ConstraintSpec, FloorplanSample, Scenario, overall_validity, read_json, write_json


CHECKER_VERSION = "cpu_bedroom_audit_v2"
PROTOCOL = "scale_recalibration_v1"
CONSTRAINT_PROTOCOL = "constraint_recalibration_v1"
SOURCE_FILES = ("config.json", "run.json", "candidates.jsonl")
SCALE_FIELDS = {"coordinate_scale_m_per_model_unit", "scale_provenance", "assumptions"}


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _checkpoint_identity(adapter):
    checkpoints = adapter.get("checkpoints", {})
    hashes = {name: checkpoints.get(name, {}).get("sha256") for name in ("node", "edge")}
    if not all(hashes.values()):
        raise ValueError("Source must record both node and edge checkpoint SHA256 identities")
    return hashes


def _compatible_scenario(old, new):
    # Scale documentation may change; operational scenario inputs may not.
    before, after = asdict(Scenario(**old)), asdict(Scenario(**new))
    changed = [key for key in before if key not in SCALE_FIELDS and before[key] != after[key]]
    if changed:
        raise ValueError(f"Only scale/provenance/documentation may change; scenario differs in {changed}")


def _validate_graph(sample):
    nodes = np.asarray(sample["nodes"], dtype=float).reshape(-1, 9)
    if len(nodes) > 53 or not np.isfinite(nodes).all():
        raise ValueError("Saved nodes must be finite XY-plus-seven-semantic rows (at most 53)")
    if "raw_edges" not in sample:
        raise ValueError("Original raw_edges are required; pruned edges cannot replay tensor selection")
    for edge in sample["raw_edges"]:
        if len(edge) != 2 or any(type(i) is not int or not 0 <= i < len(nodes) for i in edge):
            raise ValueError("raw_edges must reference saved compact node indices")
    raw = sample.get("raw_nodes", [])
    if raw:
        raw = np.asarray(raw, dtype=float)
        if raw.shape != (53, 10) or not np.isfinite(raw).all():
            raise ValueError("Saved raw_nodes are not a finite 53x10 decoded terminal state")
        decoded = raw[raw[:, 9] == 0, :9].copy()
        decoded[:, :2] = np.trunc(decoded[:, :2] * 256 + 256)
        decoded[:, 2:] = np.round(decoded[:, 2:])
        if decoded.shape != nodes.shape or not np.array_equal(decoded, nodes):
            raise ValueError("Saved nodes do not reproduce the original 512-pixel tensor decoder")


def graph_tensors(rows, device):
    """Replay the unpruned predicted graph, including bridges and self-loops.

    Never fall back from an explicitly empty raw_edges list to cleaned edges.
    The model already saved compact, quantized coordinates and rounded semantics.
    """
    count = len(rows)
    xy = torch.zeros((count, 53, 2), dtype=torch.float64, device=device)
    semantics = torch.zeros((count, 53, 7), dtype=torch.float64, device=device)
    mask = torch.zeros((count, 53), dtype=torch.bool, device=device)
    adjacency = torch.zeros((count, 53, 53), dtype=torch.bool, device=device)
    for index, row in enumerate(rows):
        sample = row["sample"]
        _validate_graph(sample)
        nodes = sample["nodes"]
        if nodes:
            values = torch.tensor(nodes, dtype=torch.float64, device=device)
            xy[index, :len(nodes)], semantics[index, :len(nodes)] = values[:, :2], values[:, 2:]
            mask[index, :len(nodes)] = True
        for first, second in sample["raw_edges"]:
            adjacency[index, min(first, second), max(first, second)] = True
    return dict(xy=xy, semantics=semantics, node_mask=mask, adjacency=adjacency)


def _prepare_sources(root, scenario):
    paths = [root] if (root / "candidates.jsonl").exists() else sorted(
        {path.parent for path in root.rglob("candidates.jsonl")})
    records, skipped, reference = [], [], None
    for path in paths:
        config, run = read_json(path / "config.json"), read_json(path / "run.json")
        method = config.get("method", run.get("method"))
        if method not in {"baseline", "best_of_n"}:
            skipped.append({"run": str(path), "method": method,
                            "reason": "Guided trajectories depend on the old reward and must be regenerated"})
            continue
        if run.get("status") != "complete":
            raise ValueError(f"Source run is not complete: {path}")
        if config.get("scale_recalibration"):
            raise ValueError("Use the original generation pool, not an already recalibrated output")
        _compatible_scenario(config["scenario"], scenario)
        adapter = config.get("adapter", {})
        checkpoints = _checkpoint_identity(adapter)
        if (adapter.get("variant") != "unconstrained_rplan" or adapter.get("drawing_resolution") != 512
                or adapter.get("merge_points") or adapter.get("align_points")):
            raise ValueError("Replay requires unmodified unconstrained_rplan graphs on the 512-pixel canvas")
        identity = {"checkpoints": checkpoints, "scenario": config["scenario"],
                    "constraints": config["constraints"], "steps": adapter.get("steps"),
                    "validity_weight": config.get("validity_weight", 100.0)}
        if reference is not None and reference != identity:
            raise ValueError("Source controls must use the same checkpoints, scenario, constraints and diffusion steps")
        reference = identity
        rows = [json.loads(line) for line in (path / "candidates.jsonl").read_text().splitlines() if line.strip()]
        groups = defaultdict(list)
        for row in rows:
            groups[row["trial"]].append(row)
            sample = row["sample"]
            if sample.get("scenario") != config["scenario"]:
                raise ValueError("Sample scenario does not match its generation config")
            if _checkpoint_identity(sample.get("metadata", {})) != checkpoints:
                raise ValueError("Sample checkpoint identities differ from the source config")
            _validate_graph(sample)
        trials, n = config["trials"], 1 if method == "baseline" else config["n"]
        if set(groups) != set(range(trials)) or any(len(group) != n for group in groups.values()):
            raise ValueError(f"{path}: complete original candidate pool required ({trials} trials x {n}); "
                             "selected-only CPU audit outputs cannot be reselected")
        ordered = []
        for trial in range(trials):
            group = groups[trial]
            if sum(bool(row.get("selected")) for row in group) != 1:
                raise ValueError("Each original trial must have exactly one selected candidate")
            if method == "best_of_n":
                group.sort(key=lambda row: row["sample"]["metadata"].get("candidate_index", -1))
                if [row["sample"]["metadata"].get("candidate_index") for row in group] != list(range(n)):
                    raise ValueError("Best-of-N pool must contain each original candidate_index exactly once")
            expected = [_candidate_seed(config["seed"] + trial, index) for index in range(n)]
            if [row["seed"] for row in group] != expected:
                raise ValueError("Candidate seeds do not match the original trial/index ordering")
            ordered.extend(group)
        if method == "best_of_n" and config.get("reward_backend") != "gpu":
            raise ValueError("This protocol replays the original GPU/tensor Best-of-N selection only")
        records.append(dict(path=path, config=config, run=run, rows=ordered, method=method, n=n,
                            hashes={name: file_hash(path / name) for name in SOURCE_FILES}))
    if not records:
        raise ValueError("No complete baseline or Best-of-N source runs found")
    return records, skipped


def _tensor_evaluator(config):
    settings = config.get("guidance", {}).get("tensor_settings", {})
    evaluator = TensorConstraintEvaluator([ConstraintSpec(**spec) for spec in config["constraints"]],
        config.get("validity_weight", 100.0),
        circle_directions=settings.get("circle_witness_directions", 32),
        max_positions=settings.get("max_bed_positions_per_room"))
    if settings and any(evaluator.metadata.get(key) != value for key, value in settings.items()):
        raise ValueError("Source tensor settings differ from the available scoring backend")
    return evaluator


@torch.no_grad()
def score_candidates(rows, evaluator, scenario, device, batch_size):
    parts = []
    for start in tqdm(range(0, len(rows), batch_size), desc="Tensor rescoring", unit="batch"):
        rooms = extract_rooms(**graph_tensors(rows[start:start + batch_size], device))
        values = evaluator.evaluate(rooms, scenario)
        values.setdefault("diagnostics", rooms["diagnostics"])
        parts.append({key: values[key].detach().cpu() for key in SCORE_FIELDS if key in values})
    return {key: torch.cat([part[key] for part in parts], dim=0) for key in parts[0]}


def select_candidates(values, rows, trials, n):
    seeds = torch.tensor([row["seed"] for row in rows], dtype=torch.long).reshape(trials, n)
    return tensor_selection(values, trials, n, tie_seeds=seeds)


def recalibrate(experiment_dir, output_dir, scenario_path, device="cpu", inference_batch_size=32,
                constraints_path=None):
    root, output = Path(experiment_dir).resolve(), Path(output_dir).resolve()
    if output == root or root in output.parents or output in root.parents:
        raise ValueError("Use a separate sibling output directory, outside the original sweep")
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}; choose a new directory")
    if inference_batch_size < 1:
        raise ValueError("inference_batch_size must be positive")
    scenario = Scenario(**read_json(scenario_path))
    if scenario.coordinate_scale_m_per_model_unit is None:
        raise ValueError("A supplied metric scale is required")
    torch.device(device)  # Fail invalid device syntax before creating output.
    records, skipped = _prepare_sources(root, asdict(scenario))
    constraints = ([asdict(spec) for spec in load_specs(constraints_path)]
                   if constraints_path is not None else None)
    override = ({"file": str(Path(constraints_path).resolve()), "sha256": file_hash(constraints_path),
                 "ids": [spec["id"] for spec in constraints]}
                if constraints is not None else None)
    protocol = CONSTRAINT_PROTOCOL if constraints is not None else PROTOCOL
    # Validate scorer compatibility for every run before writing any output.
    for record in records:
        record["effective_config"] = deepcopy(record["config"])
        if constraints is not None:
            record["effective_config"]["constraints"] = constraints
        if record["method"] == "best_of_n":
            _tensor_evaluator(record["config"])
            _tensor_evaluator(record["effective_config"])
    started = time.perf_counter()
    code_files = ("recalibrate.py", "gpu_constraints.py", "gpu_graph.py", "gpu_sampling.py",
                  "audit_geometry.py", "constraints.py", "policy.py", "geometry.py", "types.py", "area.py")
    manifest = {"protocol": protocol, "status": "running", "created_at": datetime.now(timezone.utc).isoformat(),
                "source_experiment": str(root), "scenario": asdict(scenario),
                "scenario_file": str(Path(scenario_path).resolve()), "scenario_sha256": file_hash(scenario_path),
                "constraints_override": override,
                "code_sha256": {name: file_hash(Path(__file__).parent / name) for name in code_files},
                "device": str(device), "inference_batch_size": inference_batch_size,
                "model_inference_performed": False, "skipped_runs": skipped, "methods": [],
                "steps": ["Validate complete original pools and source identities",
                          "Replay all Best-of-N raw graphs with tensor rewards under the requested scenario and constraints",
                          "Reselect with original structural/unknown-ID/energy/seed rule",
                          "Audit selected baseline and Best-of-N outputs with cpu_bedroom_audit_v2"],
                "runtime_scope": "Recalibration is separate from historical generation; no new generation runtime measured"}
    output.mkdir(parents=True)
    write_json(output / "calibration.json", manifest)
    try:
        for record in records:
            source, config, rows = record["path"], record["effective_config"], record["rows"]
            relative = source.relative_to(root) if source != root else Path(source.name)
            destination = output / relative
            destination.mkdir(parents=True)
            run_started = time.perf_counter()
            method, trials, n = record["method"], config["trials"], record["n"]
            selected_indices = set(range(len(rows)))
            tensor_records, selection, tensor_seconds = {}, None, 0.0
            if method == "best_of_n":
                evaluator = _tensor_evaluator(config)
                tensor_started = time.perf_counter()
                values = score_candidates(rows, evaluator, scenario, device, inference_batch_size)
                selection = select_candidates(values, rows, trials, n)
                selected_indices = {trial * n + int(selection["chosen"][trial]) for trial in range(trials)}
                tensor_records = {index: _evaluation_record(values, index, evaluator) for index in range(len(rows))}
                tensor_seconds = time.perf_counter() - tensor_started
            cpu = ConstraintEvaluator([ConstraintSpec(**spec) for spec in config["constraints"]],
                                      config.get("validity_weight", 100.0), bedroom_search_fn=refined_bedroom_search)
            revised, cpu_seconds = [], 0.0
            for index, original in enumerate(tqdm(rows, desc=f"{method}: final CPU audits", unit="candidate")):
                row = deepcopy(original)
                row.update(original_evaluation=original["evaluation"], original_selected=original["selected"],
                           original_overall_validity=original.get("overall_validity"),
                           original_stats=original.get("stats", {}), original_scenario=original["sample"]["scenario"],
                           original_sample_metadata=original["sample"].get("metadata", {}),
                           selected=index in selected_indices)
                row["stats"] = {"wall_seconds": None, "timing_scope": "No new generation; see original_stats and recalibration_seconds"}
                if "artifacts" in row:
                    row["original_artifacts"] = row.pop("artifacts")
                row["sample"]["scenario"] = asdict(scenario)
                metadata = row["sample"]["metadata"]
                metadata.update(scale_recalibration=protocol, final_evaluation_backend=CHECKER_VERSION if row["selected"] else None)
                if tensor_records:
                    row["selection_evaluation"] = tensor_records[index]
                    metadata.update(guidance_evaluation=tensor_records[index], guidance_device=str(device),
                                    guidance_backend_settings=dict(evaluator.metadata), selection_backend="tensor_gpu")
                audit_seconds = 0.0
                if row["selected"]:
                    audit_started = time.perf_counter()
                    row["evaluation"] = asdict(cpu.evaluate(FloorplanSample.from_dict(row["sample"]), scenario))
                    audit_seconds = time.perf_counter() - audit_started
                    cpu_seconds += audit_seconds
                    row["evaluation_backend"] = CHECKER_VERSION
                else:
                    row["evaluation"] = tensor_records[index]
                    row["evaluation_backend"] = "tensor_gpu"
                row["overall_validity"] = overall_validity(row["evaluation"])
                metadata["overall_validity"] = row["overall_validity"]
                row["recalibration_seconds"] = {"cpu_audit": audit_seconds}
                revised.append(row)
            changed = sum(not rows[index]["selected"] for index in selected_indices)
            metadata = {"protocol": protocol, "source_run_dir": str(source), "source_hashes": record["hashes"],
                        "checker_version": CHECKER_VERSION, "scenario": asdict(scenario),
                        "constraints_override": override,
                        "constraints_changed": config["constraints"] != record["config"]["constraints"],
                        "original_constraints": record["config"]["constraints"],
                        "reselection": method == "best_of_n", "generation_unchanged": True,
                        "saved_candidates": len(rows), "selected_outputs": trials, "changed_winners": changed,
                        "selected_evaluation_backend": CHECKER_VERSION,
                        "population_evaluation_backend": "mixed_tensor_cpu" if method == "best_of_n" else CHECKER_VERSION,
                        "selected_evaluation_scope": "Authoritative refined CPU audit of selected outputs",
                        "all_candidate_evaluation_scope": "Selected CPU audits plus rejected tensor proxy scores; do not report as homogeneous CPU validity"
                            if method == "best_of_n" else "Authoritative refined CPU audit of all baseline outputs",
                        "timing_scope": manifest["runtime_scope"]}
            if selection is not None:
                write_json(destination / "selection.json", [
                    {"trial": trial, "selected_index": int(selection["chosen"][trial]),
                     "coverage_anchor_index": int(selection["anchor"][trial]),
                     "eligible_indices": selection["eligible"][trial].nonzero().flatten().tolist(),
                     "fallback": bool(selection["fallback"][trial])} for trial in range(trials)])
            revised_config = deepcopy(config)
            revised_config.pop("evaluation_audit", None)
            revised_config.update(scenario=asdict(scenario), final_checker=CHECKER_VERSION,
                                  scale_recalibration=metadata, runtime_scope=manifest["runtime_scope"],
                                  output_root=str(output))
            revised_config.setdefault("guidance", {})["final_evaluator"] = CHECKER_VERSION
            write_json(destination / "config.json", revised_config)
            with (destination / "candidates.jsonl").open("w") as stream:
                for row in revised:
                    stream.write(json.dumps(row, allow_nan=False) + "\n")
            elapsed = time.perf_counter() - run_started
            run = {"method": method, "status": "complete", "planned_trials": trials, "completed_trials": trials,
                   "total_wall_seconds": None, "historical_generation": record["run"],
                   "compute_provenance": "node/edge counts below belong to reused historical generation, not recalibration",
                   "node_samples": record["run"].get("node_samples"), "edge_samples": record["run"].get("edge_samples"),
                   "node_calls": record["run"].get("node_calls"), "edge_calls": record["run"].get("edge_calls"),
                   "recalibration": {"wall_seconds": elapsed, "tensor_rescore_and_selection_seconds": tensor_seconds,
                                     "cpu_audit_seconds": cpu_seconds, "tensor_candidates": len(tensor_records),
                                     "cpu_candidates": trials, "new_denoiser_evaluations": 0}}
            write_json(destination / "run.json", run)
            summary = {**metadata, "method": method, "output_dir": str(destination),
                       "overall_validity_counts": dict(Counter(row["overall_validity"] for row in revised if row["selected"])),
                       "recalibration": run["recalibration"]}
            write_json(destination / "calibration.json", summary)
            if any(file_hash(source / name) != value for name, value in record["hashes"].items()):
                raise RuntimeError(f"Source changed during recalibration: {source}")
            manifest["methods"].append(summary)
            write_json(output / "calibration.json", manifest)
        manifest.update(status="complete", changed_winners=sum(item["changed_winners"] for item in manifest["methods"]))
    except BaseException as error:
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        manifest["wall_seconds"] = time.perf_counter() - started
        write_json(output / "calibration.json", manifest)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--constraints", help="Explicit replacement registry; omit to retain the source constraint definitions")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--inference_batch_size", type=int, default=32)
    args = parser.parse_args(argv)
    result = recalibrate(args.experiment_dir, args.output_dir, args.scenario, args.device,
                         args.inference_batch_size, constraints_path=args.constraints)
    print(f"Saved {args.output_dir}; {result['changed_winners']} Best-of-N winners changed. No model inference performed.")


if __name__ == "__main__":
    main()
