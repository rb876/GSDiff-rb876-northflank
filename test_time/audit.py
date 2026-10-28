"""Verify the adapter against the previously measured original one-sample run."""

import argparse
from pathlib import Path
import time

import numpy as np

from .adapter import GSDiffAdapter
from .types import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True, help="Original raw_prediction.npz")
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, default=Path("test_time/outputs/adapter_audit.json"))
    args = parser.parse_args()
    adapter = GSDiffAdapter(repo=args.repo)
    before = adapter.parameter_hash()
    start = time.perf_counter()
    sample = adapter.sample_baseline(42)
    reference = np.load(args.reference)
    checks = {
        "raw_nodes_identical": np.array_equal(np.array(sample.raw_nodes), reference["normalized_nodes"][0]),
        "coordinates_identical": np.array_equal(np.array(sample.nodes)[:, :2], reference["coordinates"]),
        "raw_edges_identical": np.array_equal(np.array(sample.raw_edges), reference["raw_edge_indices"]),
        "parameters_unchanged": before == adapter.parameter_hash(),
        "models_frozen": all(not p.requires_grad for model in (adapter.node, adapter.edge) for p in model.parameters()),
        "rooms_extracted": bool(sample.rooms),
    }
    report = {"checks": checks, "seconds": time.perf_counter() - start,
              "junctions": len(sample.nodes), "edges": len(sample.edges), "rooms": len(sample.rooms),
              "adapter": adapter.metadata, "counters": adapter.counters}
    write_json(args.output, report)
    write_json(args.output.with_name("baseline_sample.json"), sample.to_dict())
    print(report)
    if not all(checks.values()):
        raise SystemExit("Baseline parity failed; see audit JSON before running guided experiments.")


if __name__ == "__main__":
    main()
