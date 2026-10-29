"""Exercise extension ledgers and artifacts with synthetic sampler outcomes."""

from dataclasses import asdict
import json
from pathlib import Path
import sys
from types import ModuleType

import pytest
import torch

from test_time import run_extensions
from test_time.extension_types import ExtensionResult
from test_time.types import ConstraintResult, FloorplanSample, PolicyEvaluation, Room


def sample(tag="returned", offset=0.0):
    nodes = [[offset, 0], [4 + offset, 0], [4 + offset, 3], [offset, 3]]
    return FloorplanSample(nodes=nodes, edges=[[0, 1], [1, 2], [2, 3], [3, 0]],
                           rooms=[Room(1, nodes + [nodes[0]])], metadata={"tag": tag})


def evaluation(failed=False):
    checks = [ConstraintResult(cid, "fail" if failed and cid == "C035" else "pass",
                               1.0 if failed and cid == "C035" else 0.0,
                               True, True, "synthetic", explanation="Synthetic integration outcome")
              for cid in ("C033", "C035", "C036", "C076", "C077", "C078")]
    energy = float(failed)
    return PolicyEvaluation(energy, -energy, checks, True, 0, 6, 0, 0.0, measurable_count=6)


def install_fakes(monkeypatch, sampler, method="hard_h_mc"):
    class FakeAdapter:
        steps = 1000
        device = torch.device("cpu")
        metadata = {"variant": "unconstrained_rplan", "checkpoints": {"node": {"sha256": "synthetic"}}}

        def __init__(self, **kwargs):
            self.arguments = kwargs

        def parameter_hash(self):
            return "unchanged-frozen-weights"

    def save_sample(graph, checks, prefix):
        prefix.parent.mkdir(parents=True, exist_ok=True)
        paths = {kind: prefix.with_suffix("." + kind) for kind in ("png", "svg")}
        for path in paths.values():
            path.write_text("synthetic rendered view")
        return {kind: str(path) for kind, path in paths.items()}

    monkeypatch.setattr("test_time.adapter.GSDiffAdapter", FakeAdapter)
    monkeypatch.setattr("test_time.visualize.save_sample", save_sample)
    name = "test_time.hard_feasibility" if method == "hard_h_mc" else "test_time.repair_sampling"
    module = ModuleType(name)
    setattr(module, "hard_h_mc" if method == "hard_h_mc" else "repair_aware_svdd", sampler)
    monkeypatch.setitem(sys.modules, name, module)
    if method == "svdd_repair_aware":
        monkeypatch.setattr("test_time.gpu_constraints.TensorConstraintEvaluator", lambda *args, **kwargs: "tensor-checker")


def run(output, method="hard_h_mc", trials=1, *extra):
    run_extensions.main(["--method", method, "--device", "cpu", "--trials", str(trials),
                         "--no-progress", "--output_dir", str(output), *extra])


def jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_failed_attempt_and_lookahead_success_remain_distinct_from_returned_success(tmp_path, monkeypatch):
    calls = []

    def sampler(adapter, evaluator, scenario, seed, **options):
        calls.append((seed, options))
        if seed == 10000:
            options["event_callback"](sample("failed-lookahead"), evaluation(True), {"kind": "rollout", "seed": 41})
            options["event_callback"](sample("passing-lookahead"), evaluation(), {"kind": "rollout", "seed": 42})
            return ExtensionResult(None, None, "no_terminal_success",
                                   stats={"wall_seconds": 2.0, "node_samples": 40, "any_feasible_found": True,
                                          "first_success_nfe": 30}, trace=[{"stage": "terminal", "success_count": 0}])
        return ExtensionResult(sample(), evaluation(), "success",
                               stats={"wall_seconds": 3.0, "node_samples": 60, "any_feasible_found": True})

    install_fakes(monkeypatch, sampler)
    output = tmp_path / "hard_h_mc"
    run(output, "hard_h_mc", 2, "--k", "2", "--rollouts", "2", "--terminal_k", "2")
    trials = jsonl(output / "trials.jsonl")
    candidates = jsonl(output / "candidates.jsonl")
    events = jsonl(output / "events.jsonl")
    summary = json.loads((output / "summary.json").read_text())
    state = json.loads((output / "run.json").read_text())
    assert [entry["returned_graph"] for entry in trials] == [False, True]
    assert trials[0]["overall_validity"] == "no_graph" and trials[0]["checks"] == {}
    assert len(candidates) == 1 and candidates[0]["trial"] == 1
    assert candidates[0]["sample"]["metadata"]["trial_seed"] == 10001
    assert summary["attempts"] == 2 and summary["success_rate"] == 0.5
    assert summary["any_feasible_found"] == 2 and summary["returned_graphs"] == 1
    assert summary["no_graph_attempts"] == 1
    assert summary["nfe_per_attempt"] == 50 and summary["nfe_per_success"] == 100
    assert summary["seconds_per_attempt"] == pytest.approx(sum(r["stats"]["wall_seconds"] for r in trials) / 2)
    assert all(0 <= r["stats"]["wall_seconds"] <= r["stats"]["wall_seconds_including_event_export"] for r in trials)
    assert summary["constraint_pass_per_attempt"]["C035"] == 0.5
    assert state["status"] == "complete" and state["parameters_unchanged"] is True
    assert "graph_path" not in events[0] and (output / events[1]["graph_path"]).exists()
    assert [e["overall_validity"] for e in events] == ["invalid", "valid"]
    assert [seed for seed, _ in calls] == [10000, 10001]
    for _, options in calls:
        assert (options["k"], options["rollouts"], options["terminal_k"]) == (2, 2, 2)
        assert options["anchor_remaining"] == [200, 100, 30]
        assert options["inference_batch_size"] == 10 and options["progress"] is None
    assert json.loads((output / "traces/trial-0000.json").read_text())[0]["success_count"] == 0
    run_extensions.main(["--summarize_root", str(tmp_path)])
    assert "1/2" in (tmp_path / "comparison.md").read_text()


def test_all_no_graph_attempts_have_zero_success_rate_and_undefined_cost_per_success(tmp_path, monkeypatch):
    def sampler(*args, **kwargs):
        return ExtensionResult(None, None, "budget_exhausted",
                               stats={"wall_seconds": 1.5, "node_samples": 10})

    install_fakes(monkeypatch, sampler)
    output = tmp_path / "hard_h_mc"
    run(output, "hard_h_mc", 2)
    summary = json.loads((output / "summary.json").read_text())
    assert summary["attempts"] == summary["no_graph_attempts"] == 2
    assert summary["success_rate"] == 0 and summary["nfe_per_attempt"] == 10
    assert summary["nfe_per_success"] is None and summary["seconds_per_success"] is None
    assert (output / "candidates.jsonl").read_text() == ""
    assert len(jsonl(output / "trials.jsonl")) == 2
    assert "undefined" in (output / "report.md").read_text()


def test_repair_preserves_before_after_graphs_and_cpu_evidence(tmp_path, monkeypatch):
    before, after = sample("before"), sample("after", offset=0.1)
    before_checks, after_checks = evaluation(True), evaluation()
    received = {}

    def sampler(adapter, evaluator, scenario, seed, **options):
        received.update(options)
        return ExtensionResult(after, after_checks, "success",
                               stats={"wall_seconds": 2.0, "node_samples": 1800},
                               trace=[{"move_m": 0.1}], extra_samples={"pre_repair": (before, before_checks)})

    install_fakes(monkeypatch, sampler, "svdd_repair_aware")
    output = tmp_path / "svdd_repair_aware"
    run(output, "svdd_repair_aware", 1, "--k", "2")
    row = jsonl(output / "trials.jsonl")[0]
    prefix = output / "samples/trial-0000-seed-10000"
    saved_after = json.loads(prefix.with_suffix(".json").read_text())
    saved_before = json.loads(prefix.with_name(prefix.name + "-pre_repair.json").read_text())
    assert saved_after["nodes"] == after.nodes and saved_before["nodes"] == before.nodes
    assert saved_after["nodes"] != saved_before["nodes"]
    assert json.loads(prefix.with_name(prefix.name + "-evaluation.json").read_text()) == asdict(after_checks)
    assert json.loads(prefix.with_name(prefix.name + "-pre_repair-evaluation.json").read_text()) == asdict(before_checks)
    assert set(row["extra_artifacts"]) == {"pre_repair"}
    for paths in (row["artifacts"], row["extra_artifacts"]["pre_repair"]):
        assert all(not Path(path).is_absolute() and (output / path).is_file() for path in paths.values())
    assert received["k"] == 2 and received["guided_steps"] == 200
    assert received["tensor_evaluator"] == "tensor-checker"
    assert received["repair_config"]["inner_evaluations"] == 16
    assert received["repair_config"]["final_evaluations"] == 128
    assert received["repair_config"]["final_enabled"] is True


def test_programming_error_propagates_and_marks_run_failed(tmp_path, monkeypatch):
    def sampler(*args, **kwargs):
        raise RuntimeError("synthetic programming bug")

    install_fakes(monkeypatch, sampler)
    output = tmp_path / "hard_h_mc"
    with pytest.raises(RuntimeError, match="synthetic programming bug"):
        run(output)
    state = json.loads((output / "run.json").read_text())
    assert state["status"] == "failed" and state["completed_trials"] == 0
    assert "synthetic programming bug" in state["error"]
    assert state["summary"]["attempts"] == 0
    assert (output / "trials.jsonl").read_text() == ""


def test_incomplete_sampler_pair_is_an_error_not_an_unsuccessful_attempt(tmp_path, monkeypatch):
    def sampler(*args, **kwargs):
        return ExtensionResult(sample(), None, "success", stats={"wall_seconds": 1.0, "node_samples": 1})

    install_fakes(monkeypatch, sampler)
    output = tmp_path / "hard_h_mc"
    with pytest.raises(RuntimeError, match="incomplete sample/evaluation pair"):
        run(output)
    assert json.loads((output / "run.json").read_text())["status"] == "failed"


def test_existing_output_is_refused_without_mutation(tmp_path, monkeypatch):
    def sampler(*args, **kwargs):
        pytest.fail("existing-output guard must run before sampling")

    install_fakes(monkeypatch, sampler)
    output = tmp_path / "hard_h_mc"
    output.mkdir()
    marker = output / "original.json"
    marker.write_text('{"preserve": true}')
    with pytest.raises(FileExistsError, match="Output already exists"):
        run(output)
    assert marker.read_text() == '{"preserve": true}'
    assert list(output.iterdir()) == [marker]
