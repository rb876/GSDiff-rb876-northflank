"""Model-free replay tests using tiny synthetic floorplans and saved trajectories."""

from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path

import pytest
import torch

from test_time.constraints import load_specs
from test_time.gpu_constraints import TensorConstraintEvaluator
from test_time.gpu_graph import extract_rooms
from test_time.policy import ConstraintEvaluator
from test_time.recalibrate import (CHECKER_VERSION, CONSTRAINT_PROTOCOL, _prepare_sources, file_hash, graph_tensors,
                                  recalibrate, score_candidates, select_candidates)
from test_time.sampling import _candidate_seed
from test_time.types import FloorplanSample, Room, Scenario, overall_validity, read_json, write_json


@pytest.fixture(autouse=True)
def small_tensor_threads():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def save_rows(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def fixture_pool(tmp_path):
    source = tmp_path / "source"
    scenario = Scenario(coordinate_scale_m_per_model_unit=0.025, scale_provenance="assumed")
    corrected = Scenario(**{**asdict(scenario), "coordinate_scale_m_per_model_unit": 18 / 512,
                            "scale_provenance": "paper_derived", "assumptions": ["Paper-derived calibration"]})
    scenario_path = tmp_path / "scenario.json"
    write_json(scenario_path, asdict(corrected))
    specs = [spec for spec in load_specs(Path(__file__).resolve().parents[1] / "configs/constraints_v2.json")
             if spec.id == "C036"]
    tensor = TensorConstraintEvaluator(specs, max_positions=128)
    adapter = {"variant": "unconstrained_rplan", "steps": 1000, "drawing_resolution": 512,
               "merge_points": False, "align_points": False,
               "checkpoints": {"node": {"sha256": "a" * 64}, "edge": {"sha256": "b" * 64}}}
    for method in ("baseline", "best_of_n"):
        directory = source / method
        directory.mkdir(parents=True)
        n = 1 if method == "baseline" else 2
        config = {"method": method, "n": n, "trials": 1, "seed": 10000, "adapter": adapter,
                  "scenario": asdict(scenario), "constraints": [asdict(spec) for spec in specs],
                  "validity_weight": 100.0, "reward_backend": "gpu",
                  "guidance": {"tensor_settings": dict(tensor.metadata)}}
        rows = []
        for index, side in enumerate((50, 80)[:n]):
            points = [[0, 0], [side, 0], [side, side], [0, side]]
            semantics = [0, 0, 0, 0, 1, 0, 0]
            raw = [[0.0] * 9 + [1.0] for _ in range(53)]
            for vertex, (x, y) in enumerate(points):
                raw[vertex] = [(x - 256) / 256, (y - 256) / 256, *semantics, 0]
            edges = [[0, 1], [1, 2], [2, 3], [0, 3]]
            sample = FloorplanSample(nodes=[[*point, *semantics] for point in points], raw_nodes=raw,
                edges=edges, raw_edges=edges, rooms=[Room(4, points + [points[0]])], scenario=asdict(scenario),
                metadata={**adapter, "candidate_index": index, "trial_seed": 10000})
            evaluation = asdict(ConstraintEvaluator(specs).evaluate(sample, scenario))
            rows.append({"sample": asdict(sample), "evaluation": evaluation,
                         "overall_validity": overall_validity(evaluation), "trial": 0,
                         "candidate_id": f"candidate-{index}", "seed": _candidate_seed(10000, index),
                         "method": method, "selected": index == n - 1,
                         "stats": {"wall_seconds": 42.0}, "artifacts": {"svg": f"sample-{index}.svg"}})
        write_json(directory / "config.json", config)
        write_json(directory / "run.json", {"method": method, "status": "complete", "planned_trials": 1,
            "completed_trials": 1, "node_samples": 1000 * n, "total_wall_seconds": 42.0})
        save_rows(directory / "candidates.jsonl", rows)
    return source, scenario_path, corrected


def test_scale_reselects_full_pool_preserving_sources_and_historical_work(tmp_path):
    source, scenario_path, scenario = fixture_pool(tmp_path)
    before = {path.relative_to(source): path.read_bytes() for path in source.rglob("*") if path.is_file()}
    # Confirm this is an actual scale-dependent winner, not just altered flags.
    old_rows = read_rows(source / "best_of_n/candidates.jsonl")
    config = read_json(source / "best_of_n/config.json")
    tensor = TensorConstraintEvaluator([load_specs(Path(__file__).resolve().parents[1] / "configs/constraints_v2.json")[3]],
                                       max_positions=128)
    assert tensor.specs[0].id == "C036"
    old_values = score_candidates(old_rows, tensor, Scenario(**config["scenario"]), "cpu", 2)
    assert int(select_candidates(old_values, old_rows, 1, 2)["chosen"][0]) == 1
    output = tmp_path / "corrected"
    manifest = recalibrate(source, output, scenario_path, inference_batch_size=2)
    assert manifest["status"] == "complete" and manifest["changed_winners"] == 1
    assert before == {path.relative_to(source): path.read_bytes() for path in source.rglob("*") if path.is_file()}
    rows = read_rows(output / "best_of_n/candidates.jsonl")
    assert len(rows) == 2 and [row["selected"] for row in rows] == [True, False]
    for old, new in zip(old_rows, rows):
        assert new["original_evaluation"] == old["evaluation"]
        assert new["original_selected"] == old["selected"]
        assert new["original_stats"] == old["stats"] and new["stats"]["wall_seconds"] is None
        assert new["sample"]["scenario"] == asdict(scenario)
        assert new["sample"]["nodes"] == old["sample"]["nodes"]
        assert new["sample"]["raw_edges"] == old["sample"]["raw_edges"]
        assert new["original_artifacts"] == old["artifacts"] and "artifacts" not in new
        assert new["selection_evaluation"]["results"][0]["status"] == "pass"
    assert rows[0]["evaluation_backend"] == CHECKER_VERSION
    assert rows[1]["evaluation_backend"] == "tensor_gpu"
    assert rows[0]["overall_validity"] == "valid"
    run = read_json(output / "best_of_n/run.json")
    assert run["node_samples"] == 2000 and run["historical_generation"]["total_wall_seconds"] == 42
    assert run["total_wall_seconds"] is None and run["recalibration"]["new_denoiser_evaluations"] == 0
    assert run["recalibration"]["cpu_candidates"] == 1
    updated = read_json(output / "best_of_n/config.json")
    assert "evaluation_audit" not in updated
    assert updated["final_checker"] == CHECKER_VERSION
    assert updated["scale_recalibration"]["population_evaluation_backend"] == "mixed_tensor_cpu"
    assert read_rows(output / "baseline/candidates.jsonl")[0]["evaluation_backend"] == CHECKER_VERSION


def test_selected_only_audit_pool_is_rejected_before_output(tmp_path):
    source, scenario_path, _ = fixture_pool(tmp_path)
    path = source / "best_of_n/candidates.jsonl"
    save_rows(path, [row for row in read_rows(path) if row["selected"]])
    with pytest.raises(ValueError, match="complete original candidate pool"):
        recalibrate(source, tmp_path / "out", scenario_path)
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("change", ["scenario", "checkpoint", "candidate_index", "raw_edges"])
def test_incompatible_inputs_rejected_before_output(tmp_path, change):
    source, scenario_path, _ = fixture_pool(tmp_path)
    if change == "scenario":
        scenario = read_json(scenario_path)
        scenario["intended_occupants"] = 3
        write_json(scenario_path, scenario)
    else:
        path = source / "best_of_n/candidates.jsonl"
        rows = read_rows(path)
        if change == "checkpoint":
            rows[0]["sample"]["metadata"]["checkpoints"]["node"]["sha256"] = "other"
        elif change == "candidate_index":
            rows[1]["sample"]["metadata"]["candidate_index"] = 0
        else:
            rows[0]["sample"].pop("raw_edges")
        save_rows(path, rows)
    with pytest.raises(ValueError):
        recalibrate(source, tmp_path / "out", scenario_path)
    assert not (tmp_path / "out").exists()


def test_unknown_id_coverage_guard_is_identical_to_original_selection():
    # Candidate 1 has better energy but trades the original unknown for a new ID.
    status = torch.tensor([[1, 2], [2, 0], [0, 2]])
    values = {"status": status, "applicable": status != 3, "reward": torch.tensor([-1.0, 10.0, 0.0]),
              "structurally_valid": torch.ones(3, dtype=torch.bool), "unknown_count": (status == 2).sum(1)}
    result = select_candidates(values, [{"seed": 100}, {"seed": 1}, {"seed": 2}], 1, 3)
    assert result["eligible"].tolist() == [[True, False, True]]
    assert result["chosen"].tolist() == [2]


def test_raw_bridges_are_replayed_and_empty_raw_edges_never_fall_back(tmp_path):
    source, _, _ = fixture_pool(tmp_path)
    row = read_rows(source / "baseline/candidates.jsonl")[0]
    row["sample"]["raw_nodes"] = []
    row["sample"]["nodes"].append([80, 25, 0, 0, 0, 0, 1, 0, 0])
    row["sample"]["raw_edges"].append([0, 4])
    graph = graph_tensors([row], "cpu")
    assert graph["adjacency"][0, 0, 4]
    extracted = extract_rooms(**graph)
    assert extracted["removed_bridges"].all() and extracted["structurally_valid"].all()
    row["sample"]["raw_edges"] = []
    assert not graph_tensors([row], "cpu")["adjacency"].any()


def test_guided_sources_are_explicitly_skipped_and_decoder_checked(tmp_path):
    source, scenario_path, scenario = fixture_pool(tmp_path)
    guided = source / "svdd_pm_last20"
    guided.mkdir()
    write_json(guided / "config.json", {"method": "svdd_pm"})
    write_json(guided / "run.json", {"status": "complete"})
    (guided / "candidates.jsonl").write_text("{}\n")
    records, skipped = _prepare_sources(source, asdict(scenario))
    assert len(records) == 2 and skipped[0]["method"] == "svdd_pm"
    path = source / "baseline/candidates.jsonl"
    rows = read_rows(path)
    rows[0]["sample"]["nodes"][0][0] += 1
    save_rows(path, rows)
    with pytest.raises(ValueError, match="512-pixel tensor decoder"):
        recalibrate(source, tmp_path / "out", scenario_path)


def test_explicit_added_constraint_reselects_pool_and_audits_both_controls(tmp_path):
    source, scenario_path, _ = fixture_pool(tmp_path)
    config_dir = Path(__file__).resolve().parents[1] / "configs"
    # Both living rooms meet the original main-room requirement. Only the larger
    # one meets D003 at the corrected scale; changing the registry must reverse
    # the original tie-break winner without changing any generated graph.
    original_spec = next(spec for spec in load_specs(config_dir / "constraints_v1.json") if spec.id == "C076")
    added_spec = next(spec for spec in load_specs(config_dir / "constraints_design_v1.json") if spec.id == "D003")
    for directory in (source / "baseline", source / "best_of_n"):
        config = read_json(directory / "config.json")
        config["constraints"] = [asdict(original_spec)]
        config["guidance"]["tensor_settings"] = TensorConstraintEvaluator([original_spec], max_positions=128).metadata
        rows = read_rows(directory / "candidates.jsonl")
        for index, row in enumerate(rows):
            side = (120, 160)[index]
            points = [[0, 0], [side, 0], [side, side], [0, side]]
            semantics = [1, 0, 0, 0, 0, 0, 0]
            sample = row["sample"]
            sample["nodes"] = [[*point, *semantics] for point in points]
            sample["rooms"] = [{"label": 0, "polygon": points + [points[0]]}]
            for vertex, (x, y) in enumerate(points):
                sample["raw_nodes"][vertex] = [(x - 256) / 256, (y - 256) / 256, *semantics, 0]
            evaluation = ConstraintEvaluator([original_spec]).evaluate(
                FloorplanSample.from_dict(sample), Scenario(**config["scenario"]))
            row.update(evaluation=asdict(evaluation), overall_validity=overall_validity(asdict(evaluation)),
                       selected=index == 0)
        write_json(directory / "config.json", config)
        save_rows(directory / "candidates.jsonl", rows)
    registry = tmp_path / "expanded.json"
    write_json(registry, {"version": 1, "constraints": [asdict(original_spec), asdict(added_spec)]})
    before = {path.relative_to(source): path.read_bytes() for path in source.rglob("*") if path.is_file()}

    old_output, new_output = tmp_path / "same_checks", tmp_path / "expanded_checks"
    recalibrate(source, old_output, scenario_path, inference_batch_size=2)
    manifest = recalibrate(source, new_output, scenario_path, inference_batch_size=2, constraints_path=registry)
    old_rows = read_rows(old_output / "best_of_n/candidates.jsonl")
    new_rows = read_rows(new_output / "best_of_n/candidates.jsonl")
    assert [row["selected"] for row in old_rows] == [True, False]
    assert [row["selected"] for row in new_rows] == [False, True]
    assert manifest["protocol"] == CONSTRAINT_PROTOCOL and manifest["changed_winners"] == 1
    assert manifest["constraints_override"]["sha256"] == file_hash(registry)
    updated = read_json(new_output / "best_of_n/config.json")
    assert updated["constraints"] == [asdict(original_spec), asdict(added_spec)]
    assert updated["scale_recalibration"]["original_constraints"] == [asdict(original_spec)]
    assert updated["scale_recalibration"]["constraints_changed"]
    assert new_rows[1]["evaluation_backend"] == CHECKER_VERSION
    assert {check["id"]: check["status"] for check in new_rows[1]["evaluation"]["results"]} == {
        "C076": "pass", "D003": "pass"}
    baseline = read_rows(new_output / "baseline/candidates.jsonl")[0]
    assert baseline["overall_validity"] == "invalid"
    assert baseline["evaluation"]["results"][-1]["id"] == "D003"
    assert baseline["evaluation"]["results"][-1]["status"] == "fail"
    assert before == {path.relative_to(source): path.read_bytes() for path in source.rglob("*") if path.is_file()}


def test_constraints_cli_argument_is_explicit_and_optional(monkeypatch, capsys):
    import test_time.recalibrate as module

    calls = []
    def capture(*args, **kwargs):
        calls.append((args, kwargs))
        return {"changed_winners": 0}
    monkeypatch.setattr(module, "recalibrate", capture)
    arguments = ["--experiment_dir", "source", "--output_dir", "output", "--scenario", "scenario.json"]
    module.main(arguments)
    module.main(arguments + ["--constraints", "design.json", "--device", "cuda:0"])
    assert calls[0][1] == {"constraints_path": None}
    assert calls[1][1] == {"constraints_path": "design.json"}
    assert calls[1][0][3] == "cuda:0"
    assert "No model inference performed" in capsys.readouterr().out
