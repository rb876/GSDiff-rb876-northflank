"""Compare tensor geometry/rewards with saved CPU results; never generate samples."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path

import torch

from .gpu_constraints import TensorConstraintEvaluator
from .gpu_graph import extract_rooms
from .types import ConstraintSpec, Scenario


STATUS_NAMES = ("pass", "fail", "unknown", "not_applicable")


def _graph(rows, device):
    xy = torch.zeros((len(rows), 53, 2), dtype=torch.float64, device=device)
    semantics = torch.zeros((len(rows), 53, 7), dtype=torch.float64, device=device)
    mask = torch.zeros((len(rows), 53), dtype=torch.bool, device=device)
    adjacency = torch.zeros((len(rows), 53, 53), dtype=torch.bool, device=device)
    for index, row in enumerate(rows):
        sample = row["sample"]
        nodes = sample["nodes"]
        if len(nodes) > 53 or any(len(node) != 9 for node in nodes):
            raise ValueError("Saved nodes must contain at most 53 XY-plus-seven-semantic rows")
        if nodes:
            values = torch.tensor(nodes, dtype=xy.dtype, device=device)
            xy[index, :len(nodes)], semantics[index, :len(nodes)] = values[:, :2], values[:, 2:]
            mask[index, :len(nodes)] = True
        for first, second in sample.get("raw_edges") or sample.get("edges", []):
            if type(first) is not int or type(second) is not int or not (0 <= first < len(nodes) and 0 <= second < len(nodes)):
                raise ValueError("Saved edge has invalid compact-node references")
            adjacency[index, min(first, second), max(first, second)] = True
    return dict(xy=xy, semantics=semantics, node_mask=mask, adjacency=adjacency)


def _canonical(rooms):
    result = []
    for room in rooms:
        vertices = [tuple(point[:2]) for point in room["polygon"]]
        if vertices and vertices[0] == vertices[-1]:
            vertices.pop()
        cycles = [tuple(order[index:] + order[:index]) for order in (vertices, vertices[::-1])
                  for index in range(len(vertices))]
        result.append((room["label"], min(cycles) if cycles else ()))
    return sorted(result)


def _area(rooms):
    total = 0.0
    for room in rooms:
        points = room["polygon"]
        total += abs(sum(first[0] * second[1] - first[1] * second[0]
                         for first, second in zip(points, points[1:] + points[:1]))) / 2
    return total


def audit(experiment_dir, device="cpu", limit=None):
    directory = Path(experiment_dir).resolve()
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    files = sorted(directory.rglob("candidates.jsonl"))
    if not files:
        raise ValueError(f"No candidates.jsonl files found under {directory}")
    totals, checks, comparisons, runs = Counter(), {}, [], []
    available = sum(sum(bool(line.strip()) for line in path.open()) for path in files)
    for path in files:
        if limit is not None and len(comparisons) >= limit:
            break
        config_path = path.with_name("config.json")
        config = json.loads(config_path.read_text())
        scenario = Scenario(**config["scenario"])
        evaluator = TensorConstraintEvaluator([ConstraintSpec(**spec) for spec in config["constraints"]],
                                              validity_weight=config.get("validity_weight", 100.0))
        numbered = [(number, json.loads(line)) for number, line in enumerate(path.read_text().splitlines(), 1) if line.strip()]
        if limit is not None:
            numbered = numbered[:limit - len(comparisons)]
        for start in range(0, len(numbered), 4):
            chunk = numbered[start:start + 4]
            rooms = extract_rooms(**_graph([row for _, row in chunk], device))
            scored = evaluator.evaluate(rooms, scenario)
            # This is an offline reporting boundary; generation hot paths do not
            # perform these explicit tensor-to-host conversions.
            host = {key: value.cpu().tolist() for key, value in rooms.items()}
            statuses = scored["status"].cpu().tolist()
            valid = scored["structurally_valid"].cpu().tolist()
            for index, (line_number, row) in enumerate(chunk):
                cpu = row["evaluation"]
                old_rooms = row["sample"].get("rooms", [])
                new_rooms = [{"label": label, "polygon": [point for point, keep in zip(polygon, vertices) if keep]}
                             for label, polygon, vertices, keep in zip(host["labels"][index], host["polygons"][index],
                                                                       host["vertex_mask"][index], host["room_mask"][index]) if keep]
                old_status = {result["id"]: result["status"] for result in cpu["results"]}
                per_check = []
                for identifier, status in zip(evaluator.ids, statuses[index]):
                    new, old = STATUS_NAMES[status], old_status.get(identifier, "missing")
                    counts = checks.setdefault(identifier, Counter())
                    counts.update(compared=1, matches=int(new == old), gpu_unknown=int(new == "unknown"),
                                  gpu_pass_cpu_not_pass=int(new == "pass" and old != "pass"))
                    counts[f"cpu_{old}__gpu_{new}"] += 1
                    totals.update(gpu_unknown=int(new == "unknown"),
                                  gpu_pass_cpu_not_pass=int(new == "pass" and old != "pass"))
                    per_check.append({"id": identifier, "cpu": old, "gpu": new, "match": old == new})
                geometry_match = _canonical(old_rooms) == _canonical(new_rooms)
                validity_match = cpu["structurally_valid"] == valid[index]
                totals.update(compared=1, room_geometry_matches=int(geometry_match),
                              structural_validity_matches=int(validity_match),
                              all_statuses_match=int(all(check["match"] for check in per_check)))
                comparisons.append({"source": str(path), "line": line_number, "candidate_id": row.get("candidate_id"),
                                    "trial": row.get("trial"), "selected": row.get("selected"),
                                    "edge_source": "raw_edges" if row["sample"].get("raw_edges") else "edges",
                                    "room_counts": {"cpu": len(old_rooms), "gpu": len(new_rooms)},
                                    "room_geometry_match": geometry_match,
                                    "room_labels_match": sorted(r["label"] for r in old_rooms) == sorted(r["label"] for r in new_rooms),
                                    "summed_room_area_drawing_units2": {"cpu": _area(old_rooms), "gpu": _area(new_rooms)},
                                    "structurally_valid": {"cpu": cpu["structurally_valid"], "gpu": valid[index]},
                                    "gpu_diagnostics": host["diagnostics"][index], "checks": per_check})
        runs.append({"source": str(path), "config": str(config_path), "scenario": config["scenario"],
                     "constraint_ids": evaluator.ids, "backend_settings": evaluator.metadata,
                     "validity_weight": evaluator.validity_weight})
    return {"created_utc": datetime.now(timezone.utc).isoformat(), "experiment_dir": str(directory),
            "device": str(device), "torch_version": torch.__version__, "batch_size": 4, "limit": limit,
            "available_candidates": available, "candidate_files_found": len(files), "summary": dict(totals),
            "per_check": checks, "runs": runs, "comparisons": comparisons,
            "interpretation": "GPU fields identify the tensor backend, including when device=cpu. "
                              "Saved CPU evaluations are historical snapshots, not recomputed. Tensor finite-grid "
                              "misses may remain unknown. GPU-pass/CPU-not-pass is a disagreement requiring review, "
                              "not automatically a false pass. This comparison is not a compliance or speed proof. "
                              "Summed room areas are not polygon-union areas for invalid overlapping rooms."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment_dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    report = audit(args.experiment_dir, args.device, args.limit)
    output = args.output or args.experiment_dir / "gpu_audit.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(output), **report["summary"], "per_check": report["per_check"]}, indent=2))


if __name__ == "__main__":
    main()
