"""Fixed-prefix controls reuse saved tensor scores without winner leakage."""

from copy import deepcopy
import pytest
import torch

from test_time.evaluate import comparison_group, summarize_run
from test_time.gpu_sampling import tensor_selection
from test_time.reselect import _saved_tensor_scores, reselect
from test_time.sampling import _candidate_seed
from test_time.tests.test_recalibrate import fixture_pool, read_rows, save_rows
from test_time.types import ConstraintSpec, read_json, write_json


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def bank(tmp_path, fresh=False):
    root, _, _ = fixture_pool(tmp_path)
    source = root / "best_of_n"
    config = read_json(source / "config.json")
    templates = read_rows(source / "candidates.jsonl")
    config.update(n=5, trials=2)
    if not fresh:
        config.update(final_checker="cpu_bedroom_audit_v2", scale_recalibration={
            "selected_evaluation_backend": "cpu_bedroom_audit_v2",
            "population_evaluation_backend": "mixed_tensor_cpu", "saved_candidates": 10})
    rows = []
    for trial in range(2):
        for index, energy in enumerate((.4, .8, .1, 0, .2)):
            row = deepcopy(templates[index % 2])
            row.update(trial=trial, seed=_candidate_seed(10000 + trial, index),
                       candidate_id=f"trial-{trial}-candidate-{index}", selected=index == 3)
            row["sample"]["metadata"].update(candidate_index=index, trial_seed=10000 + trial)
            score = dict(energy=energy, reward=-energy, structurally_valid=True, unknown_count=0,
                         applicable_count=1, measurable_count=1,
                         results=[dict(id="C036", status="pass" if energy == 0 else "fail", severity=energy,
                                       applicable=True, measurable=True, counts_toward_compliance=True)])
            if fresh:
                row["sample"]["metadata"]["guidance_evaluation"] = score
            else:
                row["selection_evaluation"] = score
            # Deliberately prefer a different winner if someone mistakenly uses CPU scores.
            row["evaluation"]["energy"] = 0 if index == 1 else 100
            row["evaluation"]["reward"] = -row["evaluation"]["energy"]
            rows.append(row)
    write_json(source / "config.json", config)
    write_json(source / "run.json", dict(method="best_of_n", status="complete", completed_trials=2,
        node_samples=10000, edge_samples=20, node_calls=4000, edge_calls=8, total_wall_seconds=123))
    save_rows(source / "candidates.jsonl", rows)
    return source


@pytest.mark.parametrize("fresh", [False, True])
def test_prefix_keeps_first_three_and_reaudits_their_winners_without_inference(tmp_path, fresh):
    source = bank(tmp_path, fresh=fresh)
    before = {path.name: path.read_bytes() for path in source.iterdir() if path.is_file()}
    output = tmp_path / "best_of_n_late"
    result = reselect(source, output, 3)
    rows = read_rows(output / "candidates.jsonl")
    assert len(rows) == 6 and sum(row["selected"] for row in rows) == 2
    assert [row["sample"]["metadata"]["candidate_index"] for row in rows] == [0, 1, 2, 0, 1, 2]
    assert [row["sample"]["metadata"]["candidate_index"] for row in rows if row["selected"]] == [2, 2]
    assert all(row["evaluation_backend"].startswith("cpu_") for row in rows if row["selected"])
    assert all(row["evaluation_backend"] == "tensor_gpu" for row in rows if not row["selected"])
    assert all("artifacts" not in row and row["stats"]["wall_seconds"] is None for row in rows)
    assert all(row["sample"]["metadata"]["method_parameters"] == {"n": 3} for row in rows)
    assert result["retained_candidate_indices"] == [0, 1, 2]
    assert result["model_inference_performed"] is False
    run = read_json(output / "run.json")
    assert run["node_samples"] == 6000 and run["source_run"]["node_samples"] == 10000
    assert all(run[key] is None for key in ("node_calls", "edge_calls", "edge_samples", "total_wall_seconds"))
    assert run["prefix_reselection"]["new_denoiser_evaluations"] == 0
    assert run["prefix_reselection"]["cpu_candidates"] == 2
    config = read_json(output / "config.json")
    assert config["n"] == 3 and config["prefix_reselection"]["saved_candidates"] == 6
    assert "scale_recalibration" not in config
    assert comparison_group(config) == comparison_group(read_json(source / "config.json"))
    summary = summarize_run(output)
    assert summary["compute"]["node_samples_per_trial"] == 3000
    assert summary["all_candidates"]["status"] == "not_reported"
    assert before == {path.name: path.read_bytes() for path in source.iterdir() if path.is_file()}
    with pytest.raises(FileExistsError):
        reselect(source, output, 3)


def test_prefix_selection_preserves_unknown_id_coverage_guard():
    specs = [ConstraintSpec(id=key, checker="bathroom_turning", definition="", applicability="residential",
                           thresholds={}, required_observations=[], weight=1, severity_type="normalized_deficit",
                           assumptions=[], source_id="test") for key in ("C036", "C076")]
    rows = []
    for statuses, reward in ((["fail", "unknown"], -1), (["unknown", "pass"], 10), (["pass", "unknown"], 0)):
        rows.append({"selection_evaluation": dict(reward=reward, structurally_valid=True, unknown_count=1,
            results=[dict(id=key, status=status, applicable=True) for key, status in zip(("C036", "C076"), statuses)])})
    _, values = _saved_tensor_scores(rows, specs)
    choice = tensor_selection(values, 1, 3, tie_seeds=torch.tensor([[100, 1, 2]]))
    assert choice["eligible"].tolist() == [[True, False, True]]
    assert choice["chosen"].tolist() == [2]


@pytest.mark.parametrize("problem", ["missing_scores", "selected_only", "unverified_nfe", "missing_index", "incomplete"])
def test_invalid_source_rejected_before_output_creation(tmp_path, problem):
    source = bank(tmp_path)
    if problem in {"missing_scores", "selected_only", "missing_index"}:
        rows = read_rows(source / "candidates.jsonl")
        if problem == "missing_scores":
            rows[0].pop("selection_evaluation")
        elif problem == "selected_only":
            rows = [row for row in rows if row["selected"]]
        else:
            rows[0]["sample"]["metadata"]["candidate_index"] = 1
        save_rows(source / "candidates.jsonl", rows)
    else:
        run = read_json(source / "run.json")
        run["node_samples" if problem == "unverified_nfe" else "status"] = None
        write_json(source / "run.json", run)
    output = tmp_path / "invalid"
    with pytest.raises(ValueError):
        reselect(source, output, 3)
    assert not output.exists()


@pytest.mark.parametrize("n", [0, 6])
def test_budget_must_fit_source_bank(tmp_path, n):
    source = bank(tmp_path)
    with pytest.raises(ValueError, match="source budget"):
        reselect(source, tmp_path / "bad", n)
