"""Parallel reports preserve attempts, provenance and honest runtime accounting."""

from copy import deepcopy
import json

import pytest

from test_time.extension_workers import summarize_workers
from test_time.run_extensions import summarize_root, write_summary
from test_time.types import read_json, write_json


METHOD = "hard_h_mc"
CONFIG = {"method": METHOD, "parameters": {"k": 4, "rollouts": 4},
          "policy_sha256": "policy-a", "adapter": {"checkpoint": "frozen-a"},
          "source_sha256": {"sampler.py": "source-a"}, "inference_batch_size": 32}


def episode(seed, validity="valid"):
    returned = validity != "no_graph"
    return {"trial": 0, "seed": seed, "method": METHOD, "policy_sha256": "policy-a",
            "returned_graph": returned, "overall_validity": validity,
            "status": "success" if validity == "valid" else "infeasible" if returned else "no_feasible_terminal_sample",
            "structurally_valid": returned, "checks": {"C035": "pass" if validity == "valid" else "fail"} if returned else {},
            "stats": {"wall_seconds": 8, "node_samples": 10},
            "artifacts": {"graph": f"samples/{seed}.json"} if returned else {},
            "extra_artifacts": {"pre_repair": {"graph": f"samples/{seed}-before.json"}} if returned else {}}


def worker(parent, directory, seed, trials, rows, status="complete", config=None):
    folder = parent / directory
    folder.mkdir(parents=True)
    write_json(folder / "config.json", {**deepcopy(CONFIG if config is None else config), "seed": seed, "trials": trials})
    write_json(folder / "run.json", {"status": status, "planned_trials": trials,
                                    "completed_trials": len(rows), "parameters_unchanged": status == "complete"})
    for index, row in enumerate(rows):
        row["trial"] = index
        for paths in (row["artifacts"], *row["extra_artifacts"].values()):
            for path in paths.values():
                write_json(folder / path, {"seed": row["seed"]})
    (folder / "trials.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    write_summary(folder, rows, METHOD)
    return {"directory": directory, "seed": seed, "trials": trials, "returncode": 0 if status == "complete" else 1}


def manifest(output, workers, status="complete", elapsed=12):
    write_json(output / "parallel.json", {"method": METHOD, "seed": 100, "status": status,
               "planned_trials": sum(w["trials"] for w in workers), "workers": workers, "elapsed_seconds": elapsed})


def test_combines_all_attempts_and_preserves_artifact_paths_and_elapsed_semantics(tmp_path):
    output = tmp_path / METHOD
    # Intentionally list the higher seed first; attempt averaging must be weighted.
    second = worker(output, "worker_002", 101, 2, [episode(101), episode(102, "no_graph")])
    first = worker(output, "worker_001", 100, 1, [episode(100, "invalid")])
    original = (output / "worker_002/trials.jsonl").read_bytes()
    manifest(output, [second, first])
    assert summarize_workers(output, METHOD) == output
    summary = read_json(output / "summary.json")
    assert summary["attempts"] == 3 and summary["success_rate"] == pytest.approx(1 / 3)
    assert summary["no_graph_attempts"] == 1 and summary["total_nfe"] == 30
    assert summary["total_wall_seconds"] == 24 and summary["seconds_per_attempt"] == 8
    assert summary["parallel_elapsed_seconds"] == 12 and summary["attempts_per_elapsed_second"] == .25
    rows = [json.loads(line) for line in (output / "trials.jsonl").read_text().splitlines()]
    assert [row["seed"] for row in rows] == [100, 101, 102]
    assert [row["trial"] for row in rows] == [0, 1, 2]
    assert rows[1]["worker"] == "worker_002" and rows[1]["worker_trial"] == 0
    for row in rows:
        for paths in (row["artifacts"], *row["extra_artifacts"].values()):
            for path in paths.values():
                assert (output / path).exists()
    assert (output / "worker_002/trials.jsonl").read_bytes() == original
    assert read_json(output / "run.json")["status"] == "complete"
    report = (output / "report.md").read_text()
    assert "not elapsed parallel runtime" in report
    assert "worker_001/report.md" in report and "`worker_*/events.jsonl`" in report
    summarize_root(tmp_path)
    assert "1/3" in (tmp_path / "comparison.md").read_text()


@pytest.mark.parametrize("change", [
    {"policy_sha256": "different"}, {"parameters": {"rollouts": 16}},
    {"adapter": {"checkpoint": "different"}}, {"source_sha256": {"sampler.py": "different"}},
])
def test_refuses_mixed_policies_parameters_models_or_sources(tmp_path, change):
    first = worker(tmp_path, "worker_001", 100, 1, [episode(100)])
    second = worker(tmp_path, "worker_002", 101, 1, [episode(101)], config={**CONFIG, **change})
    manifest(tmp_path, [first, second])
    with pytest.raises(ValueError, match="Incompatible"):
        summarize_workers(tmp_path, METHOD)
    assert not (tmp_path / "trials.jsonl").exists()


def test_refuses_duplicate_seeds(tmp_path):
    first = worker(tmp_path, "worker_001", 100, 1, [episode(100)])
    second = worker(tmp_path, "worker_002", 100, 1, [episode(100)])
    manifest(tmp_path, [first, second])
    with pytest.raises(ValueError, match="Duplicate trial seed"):
        summarize_workers(tmp_path, METHOD)


def test_partial_failed_run_keeps_completed_rows_and_ignores_only_truncated_final_row(tmp_path):
    first = worker(tmp_path, "worker_001", 100, 2, [episode(100)], status="running")
    with (tmp_path / "worker_001/trials.jsonl").open("a") as stream:
        stream.write('{"trial": 1, "seed":')
    second = {"directory": "worker_002", "seed": 102, "trials": 2, "returncode": 1}
    manifest(tmp_path, [first, second], status="failed")
    summarize_workers(tmp_path, METHOD)
    state = read_json(tmp_path / "run.json")
    assert state["status"] == "failed" and state["completed_trials"] == 1 and state["planned_trials"] == 4
    assert len(state["issues"]) == 1
    assert state["summary"]["attempts"] == 1 and state["summary"]["success_rate"] == 1
    assert state["workers"][1]["status"] == "missing"


def test_complete_manifest_cannot_hide_missing_worker_attempts(tmp_path):
    first = worker(tmp_path, "worker_001", 100, 2, [episode(100)])
    manifest(tmp_path, [first])
    summarize_workers(tmp_path, METHOD)
    assert read_json(tmp_path / "run.json")["status"] == "incomplete"


def test_interrupted_before_first_sample_writes_empty_honest_summary(tmp_path):
    entry = {"directory": "worker_001", "seed": 100, "trials": 2, "returncode": -15}
    folder = tmp_path / entry["directory"]
    folder.mkdir()
    (folder / "config.json").write_text('{"parameters":')
    (folder / "run.json").write_text('{"status":')
    manifest(tmp_path, [entry], status="interrupted", elapsed=.5)
    summarize_workers(tmp_path, METHOD)
    state = read_json(tmp_path / "run.json")
    assert state["status"] == "interrupted" and state["completed_trials"] == 0
    assert state["summary"]["success_rate"] is None
    assert len(state["issues"]) == 2
