"""Model-free replay tests using tiny synthetic floorplans and saved trajectories."""

from dataclasses import asdict
import json
from pathlib import Path

import pytest
import torch

from test_time.constraints import load_specs
from test_time.evaluate import comparison_group
from test_time.gpu_constraints import TensorConstraintEvaluator
from test_time.gpu_graph import extract_rooms
from test_time.policy import ConstraintEvaluator
from test_time.recalibrate import (BATHROOM_CHECKER_VERSION, BATHROOM_FIELDS, CHECKER_VERSION,
                                  CONSTRAINT_PROTOCOL, _prepare_sources, _replacement_scenario_fields,
                                  file_hash, graph_tensors, recalibrate, score_candidates, select_candidates)
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


def test_constraints_cli_argument_is_explicit_and_optional(monkeypatch, capsys):
    import test_time.recalibrate as module

    calls = []
    def capture(*args, **kwargs):
        calls.append((args, kwargs))
        return {"changed_winners": 0}
    monkeypatch.setattr(module, "recalibrate", capture)
    arguments = ["--experiment_dir", "source", "--output_dir", "output", "--scenario", "scenario.json"]
    module.main(arguments)
    module.main(arguments + ["--constraints", "constraints.json", "--device", "cuda:0"])
    assert calls[0][1] == {"constraints_path": None}
    assert calls[1][1] == {"constraints_path": "constraints.json"}
    assert calls[1][0][3] == "cuda:0"
    assert "No model inference performed" in capsys.readouterr().out


def ventilation_pool(tmp_path):
    source, scenario_path, _ = fixture_pool(tmp_path)
    configs = Path(__file__).resolve().parents[1] / "configs"
    scenario = read_json(configs / "scenario_ventilation_v1.json")
    write_json(scenario_path, scenario)
    for directory in (source / "baseline", source / "best_of_n"):
        config = read_json(directory / "config.json")
        rows = read_rows(directory / "candidates.jsonl")
        original_specs = [load_specs(configs / "constraints_v2.json")[3]]
        for index, row in enumerate(rows):
            # Both candidates pass the original bathroom check. Candidate zero
            # has four main rooms; candidate one has three. The fixed system is
            # adequate only for candidate one under C085/C090.
            labels = [0] + [1] * (3 - index) + [3, 4]
            nodes, edges, rooms = [], [], []
            raw = [[0.0] * 9 + [1.0] for _ in range(53)]
            for room_index, label in enumerate(labels):
                x, y = (room_index % 3) * 160, (room_index // 3) * 160
                points = [[x, y], [x + 128, y], [x + 128, y + 128], [x, y + 128]]
                semantic = [int(value == label) for value in range(7)]
                first = len(nodes)
                for point in points:
                    raw[len(nodes)] = [(point[0] - 256) / 256, (point[1] - 256) / 256, *semantic, 0]
                    nodes.append([*point, *semantic])
                edges += [[first, first + 1], [first + 1, first + 2],
                          [first + 2, first + 3], [first, first + 3]]
                rooms.append({"label": label, "polygon": points + [points[0]]})
            sample = row["sample"]
            sample.update(nodes=nodes, raw_nodes=raw, edges=edges, raw_edges=edges, rooms=rooms)
            evaluation = ConstraintEvaluator(original_specs).evaluate(
                FloorplanSample.from_dict(sample), Scenario(**config["scenario"]))
            row.update(evaluation=asdict(evaluation), overall_validity=overall_validity(asdict(evaluation)),
                       selected=index == 0)
        save_rows(directory / "candidates.jsonl", rows)
    return source, scenario_path, configs / "constraints_ventilation_v1.json"


def test_explicit_ventilation_scenario_reselects_and_preserves_source_provenance(tmp_path):
    source, scenario_path, registry = ventilation_pool(tmp_path)
    before = {path.relative_to(source): path.read_bytes() for path in source.rglob("*") if path.is_file()}
    output = tmp_path / "ventilation"
    result = recalibrate(source, output, scenario_path, inference_batch_size=2, constraints_path=registry)
    assert result["protocol"] == CONSTRAINT_PROTOCOL and result["changed_winners"] == 1
    assert result["constraints_override"]["sha256"] == file_hash(registry)
    rows = read_rows(output / "best_of_n/candidates.jsonl")
    assert [row["selected"] for row in rows] == [False, True]
    checks = {item["id"]: item for item in rows[1]["evaluation"]["results"] if item["counts_toward_compliance"]}
    assert len(checks) == 9 and all(item["status"] == "pass" for item in checks.values())
    assert rows[1]["evaluation_backend"] == CHECKER_VERSION
    baseline = read_rows(output / "baseline/candidates.jsonl")[0]
    statuses = {item["id"]: item["status"] for item in baseline["evaluation"]["results"]}
    assert statuses["C085"] == statuses["C090"] == "fail"
    assert baseline["overall_validity"] == "invalid"
    assert baseline["evaluation_backend"] == CHECKER_VERSION
    updated = read_json(output / "best_of_n/config.json")
    provenance = updated["scale_recalibration"]
    assert updated["constraints"] == [asdict(spec) for spec in load_specs(registry)]
    assert provenance["original_constraints"] == read_json(source / "best_of_n/config.json")["constraints"]
    assert provenance["constraints_changed"]
    assert provenance["scenario_changes"]["kitchen_extraction_m3_h"] == {"from": None, "to": 105.0}
    assert provenance["original_scenario"] == read_json(source / "best_of_n/config.json")["scenario"]
    fresh = {key: value for key, value in updated.items() if key != "scale_recalibration"}
    assert comparison_group(updated) == comparison_group(fresh)
    assert before == {path.relative_to(source): path.read_bytes() for path in source.rglob("*") if path.is_file()}


@pytest.mark.parametrize("violation", ["no_registry", "unrelated_registry", "occupancy"])
def test_ventilation_change_requires_relevant_explicit_registry_and_keeps_other_guards(tmp_path, violation):
    source, scenario_path, registry = ventilation_pool(tmp_path)
    kwargs = {"constraints_path": registry}
    if violation == "no_registry":
        kwargs = {}
    elif violation == "unrelated_registry":
        kwargs["constraints_path"] = registry.parent / "constraints_v2.json"
    else:
        scenario = read_json(scenario_path)
        scenario["intended_occupants"] = 6
        write_json(scenario_path, scenario)
    with pytest.raises(ValueError, match="scenario differs"):
        recalibrate(source, tmp_path / "rejected", scenario_path, **kwargs)
    assert not (tmp_path / "rejected").exists()


def bathroom_pool(tmp_path):
    source, scenario_path, scenario = fixture_pool(tmp_path)
    configs = Path(__file__).resolve().parents[1] / "configs"
    target = read_json(configs / "scenario_bathroom_v1.json")
    write_json(scenario_path, {**asdict(scenario), **{name: target[name] for name in BATHROOM_FIELDS}})
    # Isolate the coupled bathroom checks so selection is driven by their
    # feasibility, not by the synthetic pool's missing bedrooms or kitchen.
    specs = [spec for spec in load_specs(configs / "constraints_bathroom_v1.json")
             if spec.id in {"C036", "C037", "C041"}]
    registry = tmp_path / "bathroom_constraints.json"
    write_json(registry, {"constraints": [asdict(spec) for spec in specs]})
    path = source / "best_of_n/candidates.jsonl"
    rows = read_rows(path)
    sample = rows[1]["sample"]
    sample["nodes"] = [[x * 1.6, y * 1.6, *rest] for x, y, *rest in sample["nodes"]]
    for index, node in enumerate(sample["nodes"]):
        sample["raw_nodes"][index][:2] = [(value - 256) / 256 for value in node[:2]]
    sample["rooms"][0]["polygon"] = [[x * 1.6, y * 1.6] for x, y in sample["rooms"][0]["polygon"]]
    old_specs = [spec for spec in load_specs(configs / "constraints_v2.json") if spec.id == "C036"]
    evaluation = ConstraintEvaluator(old_specs).evaluate(FloorplanSample.from_dict(sample), Scenario(**sample["scenario"]))
    rows[1].update(evaluation=asdict(evaluation), overall_validity=overall_validity(evaluation))
    save_rows(path, rows)
    return source, scenario_path, registry


def test_bathroom_replay_uses_full_pool_and_records_shared_witness_audit(tmp_path):
    source, scenario_path, registry = bathroom_pool(tmp_path)
    before = {path.relative_to(source): path.read_bytes() for path in source.rglob("*") if path.is_file()}
    output = tmp_path / "bathroom"
    manifest = recalibrate(source, output, scenario_path, inference_batch_size=2, constraints_path=registry)
    assert manifest["protocol"] == CONSTRAINT_PROTOCOL and manifest["model_inference_performed"] is False
    assert set(manifest["allowed_bathroom_scenario_changes"]) == BATHROOM_FIELDS
    assert "bathroom_geometry.py" in manifest["code_sha256"]
    rows = read_rows(output / "best_of_n/candidates.jsonl")
    assert len(rows) == 2 and [row["selected"] for row in rows] == [False, True]
    selected = rows[1]
    assert selected["evaluation_backend"] == BATHROOM_CHECKER_VERSION
    assert selected["overall_validity"] == "valid"
    assert {item["id"] for item in selected["evaluation"]["results"]} == {"C036", "C037", "C041"}
    witnesses = [item["observed_values"]["witness"] for item in selected["evaluation"]["results"]]
    assert witnesses[0] and all(witness == witnesses[0] for witness in witnesses)
    assert rows[0]["evaluation_backend"] == "tensor_gpu"
    config = read_json(output / "best_of_n/config.json")
    assert config["final_checker"] == config["guidance"]["final_evaluator"] == BATHROOM_CHECKER_VERSION
    assert config["scale_recalibration"]["scenario_changes"]["bathroom_layout"] == {
        "from": None, "to": "combined_wc_basin_v1"}
    original = read_json(source / "best_of_n/config.json")
    assert config["scale_recalibration"]["original_guidance"] == original["guidance"]
    fresh_settings = TensorConstraintEvaluator(load_specs(registry), max_positions=128).metadata
    assert config["guidance"]["tensor_settings"] == fresh_settings
    assert "bathroom_scenario" not in config["guidance"]["tensor_settings"]
    fresh = {**config, "method": "svdd_pm", "guidance": {**config["guidance"], "tensor_settings": fresh_settings}}
    fresh.pop("scale_recalibration")
    baseline = read_json(output / "baseline/config.json")
    assert comparison_group(config) == comparison_group(fresh) == comparison_group(baseline)
    assert read_json(output / "best_of_n/run.json")["recalibration"]["new_denoiser_evaluations"] == 0
    assert before == {path.relative_to(source): path.read_bytes() for path in source.rglob("*") if path.is_file()}


@pytest.mark.parametrize("violation", ["no_registry", "unrelated_registry", "occupancy"])
def test_bathroom_scenario_override_keeps_registry_and_operational_guards(tmp_path, violation):
    source, scenario_path, registry = bathroom_pool(tmp_path)
    kwargs = {"constraints_path": registry}
    if violation == "no_registry":
        kwargs = {}
    elif violation == "unrelated_registry":
        kwargs["constraints_path"] = Path(__file__).resolve().parents[1] / "configs/constraints_v2.json"
    else:
        scenario = read_json(scenario_path)
        scenario["intended_occupants"] = 3
        write_json(scenario_path, scenario)
    with pytest.raises(ValueError, match="scenario differs"):
        recalibrate(source, tmp_path / "rejected", scenario_path, **kwargs)
    assert not (tmp_path / "rejected").exists()


def test_bathroom_inputs_require_all_enabled_joint_checks_and_consumed_fields():
    specs = [{"id": identifier, "enabled": True, "checker": "joint_bathroom",
              "required_observations": ["bathroom_layout", "wc_width_m"]}
             for identifier in ("C036", "C037", "C041")]
    assert _replacement_scenario_fields(specs) == {"bathroom_layout", "wc_width_m"}
    specs[-1]["enabled"] = False
    assert _replacement_scenario_fields(specs) == set()
    specs[-1].update(enabled=True, checker="bathroom_turning")
    assert _replacement_scenario_fields(specs) == set()
