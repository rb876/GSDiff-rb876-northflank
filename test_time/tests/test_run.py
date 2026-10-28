"""CLI batching/export integration without loading checkpoints or generating designs."""

import json
import sys
from types import SimpleNamespace

import pytest

from test_time import adapter, run, sampling, visualize
from test_time.types import Candidate, ConstraintResult, FloorplanSample, PolicyEvaluation, Room, SamplingResult


@pytest.mark.parametrize("method", ["baseline", "best_of_n", "svdd_pm"])
def test_runner_exports_tail_batch_and_counts_shared_compute_once(monkeypatch, tmp_path, method):
    class FakeAdapter:
        device = SimpleNamespace(type="cpu")
        metadata = {"variant": "test", "steps": 5}

        def __init__(self, **kwargs):
            pass

        def parameter_hash(self):
            return "unchanged"

    calls = []

    def sample_batch(model, evaluator, scenario, seeds, inference_batch_size, **settings):
        calls.append((seeds, inference_batch_size, settings))
        candidate_count = settings.get("n", 1)
        seconds = 2 * len(seeds) + 1
        results = []
        for seed in seeds:
            candidates = []
            for index in range(candidate_count):
                sample = FloorplanSample(
                    nodes=[[0, 0], [4, 0], [4, 3], [0, 3]], edges=[[0, 1], [1, 2], [2, 3], [3, 0]],
                    rooms=[Room(0, [[0, 0], [4, 0], [4, 3], [0, 3], [0, 0]])],
                )
                check = ConstraintResult("C076", "pass", 0.0, True, True, "test")
                evaluation = PolicyEvaluation(0.0, 0.0, [check], True, 0, 1, 0, 0.0, measurable_count=1)
                candidates.append(Candidate(sample, evaluation, seed + index, f"candidate-{index}-seed-{seed}"))
            results.append(SamplingResult(candidates, candidate_count - 1,
                                          trace=[{"trial_seed": seed}] if method == "svdd_pm" else [],
                                          stats={"wall_seconds": seconds / len(seeds)}))
        return results, {"wall_seconds": seconds, "node_calls": 4, "edge_calls": 1,
                         "node_samples": 4 * len(seeds), "edge_samples": len(seeds) * candidate_count,
                         "graph_completions": len(seeds) * candidate_count,
                         "candidate_evaluations": len(seeds) * candidate_count, "fallback_count": 0}

    output = tmp_path / method
    monkeypatch.setattr(adapter, "GSDiffAdapter", FakeAdapter)
    monkeypatch.setattr(sampling, method + "_batch", sample_batch)
    monkeypatch.setattr(visualize, "save_sample", lambda sample, evaluation, prefix:
                        {extension: str(prefix.with_suffix('.' + extension)) for extension in ("png", "svg")})
    monkeypatch.setattr(sys, "argv", ["run", "--method", method, "--trials", "3", "--seed", "1000",
                                    "--batch_size", "2", "--inference_batch_size", "3", "--n", "2",
                                    "--k", "2", "--guided_steps", "2", "--output_dir", str(output)])
    run.main()

    assert [(seeds, batch) for seeds, batch, _ in calls] == [([1000, 1001], 3), ([1002], 3)]
    metadata = json.loads((output / "run.json").read_text())
    assert metadata["status"] == "complete" and metadata["completed_trials"] == 3
    assert metadata["total_wall_seconds"] == 8
    assert metadata["node_calls"] == 8 and metadata["node_samples"] == 12
    assert [batch["trial_count"] for batch in metadata["batches"]] == [2, 1]
    rows = [json.loads(line) for line in (output / "candidates.jsonl").read_text().splitlines()]
    assert len(rows) == (6 if method == "best_of_n" else 3)
    selected = [row for row in rows if row["selected"]]
    assert [row["trial"] for row in selected] == [0, 1, 2]
    assert [row["sample"]["metadata"]["trial_seed"] for row in selected] == [1000, 1001, 1002]
    assert len({row["candidate_id"] for row in rows}) == len(rows)
    assert len(list((output / "samples").glob("*.json"))) == len(rows)
    if method == "svdd_pm":
        assert len(list((output / "traces").glob("*.json"))) == 3
    summary = json.loads((output / "summary.json").read_text())
    assert summary["compute"]["wall_seconds_per_trial"] == pytest.approx(8 / 3)
    assert summary["parameters"]["batch_size"] == 2


@pytest.mark.parametrize("flag", ["--batch_size", "--inference_batch_size"])
def test_nonpositive_batch_size_rejected_before_loading_models(monkeypatch, flag):
    monkeypatch.setattr(sys, "argv", ["run", "--method", "baseline", flag, "0"])
    with pytest.raises(SystemExit) as error:
        run.main()
    assert error.value.code == 2
