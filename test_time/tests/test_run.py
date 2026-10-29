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
    monkeypatch.setattr(sys, "argv", ["run", "--config", str(run.Path(run.__file__).parent / "configs/experiment_v2.json"),
                                    "--method", method, "--reward_backend", "cpu", "--trials", "3", "--seed", "1000",
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
    assert summary["parameters"]["reward_backend"] == "cpu"
    snapshot = json.loads((output / "config.json").read_text())
    assert snapshot["guidance"]["tensor_settings"] is None
    assert snapshot["guidance"]["final_evaluator"] == "cpu_reference"
    assert metadata["gpu_candidate_evaluations"] == 0
    assert metadata["cpu_candidate_evaluations"] == metadata["candidate_evaluations"]


@pytest.mark.parametrize("flag", ["--batch_size", "--inference_batch_size"])
def test_nonpositive_batch_size_rejected_before_loading_models(monkeypatch, flag):
    monkeypatch.setattr(sys, "argv", ["run", "--method", "baseline", flag, "0"])
    with pytest.raises(SystemExit) as error:
        run.main()
    assert error.value.code == 2


@pytest.mark.parametrize("legacy", [False, True])
def test_constraint_listing_needs_no_models_and_preserves_legacy_profile(monkeypatch, capsys, legacy):
    def no_models(**kwargs):
        pytest.fail("Constraint listing must not load pretrained models")

    monkeypatch.setattr(adapter, "GSDiffAdapter", no_models)
    arguments = ["run", "--method", "baseline", "--list-constraints"]
    if legacy:
        arguments += ["--config", str(run.Path(run.__file__).parent / "configs/experiment_v1.json")]
    monkeypatch.setattr(sys, "argv", arguments)
    run.main()
    selected = json.loads(capsys.readouterr().out)
    expected = {"C033", "C035", "C036", "C076", "C077", "C078"}
    if not legacy:
        expected |= {"D001", "D002", "D003", "D004"}
    assert set(selected["active_scored_ids"]) == expected
    assert selected["constraint_counts"]["registered"] == (7 if legacy else 13)
    assert selected["disabled_constraints"] == []


@pytest.mark.parametrize("method", ["baseline", "best_of_n", "svdd_pm"])
@pytest.mark.parametrize("final_checker", ["cpu_reference", "cpu_bedroom_audit_v2"])
def test_default_gpu_dispatch_metadata_and_final_cpu_results(monkeypatch, tmp_path, capsys, method, final_checker):
    from test_time import gpu_constraints, gpu_sampling, policy
    from test_time.audit_geometry import refined_bedroom_search
    import torch

    cpu_calls, tensor_instances, dispatched = [], [], []

    class FakeAdapter:
        device = torch.device("cpu")
        metadata = {"variant": "test", "steps": 5}

        def __init__(self, **kwargs):
            pass

        def parameter_hash(self):
            return "unchanged"

    class CpuReference:
        def __init__(self, specs, validity_weight, *, bedroom_search_fn=None):
            self.specs, self.validity_weight = specs, validity_weight
            self.bedroom_search_fn = bedroom_search_fn

        def evaluate(self, sample, scenario):
            cpu_calls.append(sample)
            check = ConstraintResult("C076", "fail", 0.5, True, True, "cpu_reference")
            return PolicyEvaluation(energy=0.5, reward=-0.5, results=[check], structurally_valid=True,
                                    unknown_count=0, applicable_count=1, assumption_based_count=0,
                                    validity_penalty=0.0, measurable_count=1)

    class TensorGuidance:
        metadata = {"backend": "test_tensor", "cpu_oracle_equivalence": False}

        def __init__(self, specs, validity_weight):
            self.specs, self.validity_weight = specs, validity_weight
            tensor_instances.append(self)

    def forbidden(*args, **kwargs):
        pytest.fail("Runner dispatched to the wrong reward backend")

    def sample_batch(model, evaluator, scenario, seeds, inference_batch_size, **settings):
        assert isinstance(evaluator, CpuReference)
        assert evaluator.bedroom_search_fn is (refined_bedroom_search if final_checker == "cpu_bedroom_audit_v2" else None)
        assert inference_batch_size == 3
        if method == "baseline":
            assert "tensor_evaluator" not in settings
        else:
            assert settings["tensor_evaluator"] is tensor_instances[0]
            assert settings["tensor_evaluator"].specs == evaluator.specs
            assert settings["tensor_evaluator"].validity_weight == evaluator.validity_weight
            assert settings["n" if method == "best_of_n" else "k"] == 2
        dispatched.append(seeds)
        results = []
        count = settings.get("n", 1)
        for seed in seeds:
            candidates = []
            for index in range(count):
                sample = FloorplanSample(nodes=[[0, 0], [4, 0], [4, 3], [0, 3]],
                                        edges=[[0, 1], [1, 2], [2, 3], [0, 3]],
                                        rooms=[Room(0, [[0, 0], [4, 0], [4, 3], [0, 3], [0, 0]])])
                sample.metadata["guidance_evaluation"] = {"results": [{"id": "C076", "status": "pass"}]}
                candidates.append(Candidate(sample, evaluator.evaluate(sample, scenario), seed,
                                            f"candidate-{index}-seed-{seed}"))
            results.append(SamplingResult(candidates, count - 1, stats={"wall_seconds": 1.0}))
        cpu_count = len(seeds) * count
        gpu_count = 0 if method == "baseline" else 7 * cpu_count
        return results, {"wall_seconds": float(len(seeds)), "candidate_evaluations": cpu_count + gpu_count,
                         "gpu_candidate_evaluations": gpu_count, "cpu_candidate_evaluations": cpu_count}

    monkeypatch.setattr(adapter, "GSDiffAdapter", FakeAdapter)
    monkeypatch.setattr(policy, "ConstraintEvaluator", CpuReference)
    monkeypatch.setattr(gpu_constraints, "TensorConstraintEvaluator", TensorGuidance)
    for name in ("baseline", "best_of_n", "svdd_pm"):
        monkeypatch.setattr(sampling, name + "_batch", sample_batch if name == method == "baseline" else forbidden)
    for name in ("best_of_n", "svdd_pm"):
        monkeypatch.setattr(gpu_sampling, name + "_batch_gpu", sample_batch if name == method else forbidden)
    monkeypatch.setattr(visualize, "save_sample", lambda *args: {})
    output = tmp_path / method
    arguments = ["run", "--method", method, "--device", "cpu",
                                    "--trials", "3", "--batch_size", "2", "--inference_batch_size", "3",
                                    "--n", "2", "--k", "2", "--guided_steps", "2", "--output_dir", str(output)]
    if final_checker == "cpu_reference":
        arguments += ["--config", str(run.Path(run.__file__).parent / "configs/experiment_v2.json")]
    monkeypatch.setattr(sys, "argv", arguments)
    run.main()

    trial_logs = [line for line in capsys.readouterr().out.splitlines() if "amortized_seconds=" in line]
    assert len(trial_logs) == 3
    assert all("failed=[C076], unknown_ids=[]" in line for line in trial_logs)
    assert [len(seeds) for seeds in dispatched] == [2, 1]
    snapshot = json.loads((output / "config.json").read_text())
    assert snapshot["reward_backend"] == snapshot["guidance"]["backend"] == "gpu"
    assert snapshot["guidance"]["device"] == "cpu"  # backend tests do not require CUDA
    assert snapshot["guidance"]["tensor_settings"] == TensorGuidance.metadata
    assert snapshot["final_checker"] == snapshot["guidance"]["final_evaluator"] == final_checker
    assert snapshot["guidance"]["trace_format"] == "scores_and_selections"
    saved = [json.loads(line) for line in (output / "candidates.jsonl").read_text().splitlines()]
    assert len(cpu_calls) == len(saved) == (6 if method == "best_of_n" else 3)
    assert all(row["evaluation"]["results"][0]["status"] == "fail" for row in saved)
    assert all(row["overall_validity"] == "invalid" for row in saved)
    assert all(row["sample"]["metadata"]["final_evaluation_backend"] == final_checker for row in saved)
    counters = json.loads((output / "run.json").read_text())
    assert counters["cpu_candidate_evaluations"] == len(cpu_calls)
    assert counters["gpu_candidate_evaluations"] == (0 if method == "baseline" else 7 * len(cpu_calls))
    assert counters["candidate_evaluations"] == counters["cpu_candidate_evaluations"] + counters["gpu_candidate_evaluations"]
    summary = json.loads((output / "summary.json").read_text())
    assert summary["parameters"]["reward_backend"] == "gpu"
    assert summary["overall_valid_rate"] == 0 and summary["structural_valid_rate"] == 1


def test_unknown_final_checker_rejected_before_loading_models(monkeypatch, tmp_path, capsys):
    config = json.loads((run.Path(run.__file__).parent / "configs/experiment_v2.json").read_text())
    config["final_checker"] = "unrecognised_checker"
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    monkeypatch.setattr(sys, "argv", ["run", "--config", str(path), "--method", "baseline"])
    monkeypatch.setattr(adapter, "GSDiffAdapter", lambda **kwargs: pytest.fail("Should reject before loading models"))
    with pytest.raises(SystemExit) as error:
        run.main()
    assert error.value.code == 2
    assert "final_checker must be" in capsys.readouterr().err
