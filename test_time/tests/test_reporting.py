"""Reporting denominators, compute comparisons and annotated output artifacts."""

from dataclasses import asdict
import csv
import json
from pathlib import Path

from PIL import Image
import pytest

from test_time.evaluate import comparison_group, summarize_run, write_comparison, write_report
from test_time.types import ConstraintResult, FloorplanSample, PolicyEvaluation, Room, write_json
from test_time.visualize import save_sample


def result(cid, status, auxiliary=False, assumed=False):
    return ConstraintResult(cid, status, {"pass": 0.0, "fail": 0.5}.get(status),
                            status != "not_applicable", status in {"pass", "fail"}, "assessment",
                            {"area_m2": 8}, {"area_min_m2": 9},
                            ["assumed_scale"] if assumed else [], "Measured area versus required area.",
                            counts_toward_compliance=not auxiliary)


def candidate(trial, checks, valid=True, selected=True):
    sample = FloorplanSample(nodes=[[0, 0], [4, 0], [4, 3], [0, 3]],
                             edges=[[0, 1], [1, 2], [2, 3], [3, 0]],
                             rooms=[Room(0, [[0, 0], [4, 0], [4, 3], [0, 3], [0, 0]])],
                             scenario={"scale_provenance": "assumed", "coordinate_scale_m_per_model_unit": 0.025})
    evaluation = PolicyEvaluation(0.5, -0.5, checks, valid,
                                  sum(r.status == "unknown" for r in checks),
                                  sum(r.applicable for r in checks),
                                  sum(bool(r.assumption_flags) for r in checks), 0 if valid else 1)
    return {"trial": trial, "seed": trial, "selected": selected, "method": "baseline",
            "candidate_id": str(trial), "sample": asdict(sample), "evaluation": asdict(evaluation), "stats": {}}


def make_run(path, rows, method="baseline", seconds=20, scale=0.025):
    path.mkdir(parents=True)
    write_json(path / "config.json", {"method": method, "n": 1 if method == "baseline" else 4,
               "scenario": {"scale_provenance": "assumed", "coordinate_scale_m_per_model_unit": scale},
               "constraints": [{"id": "C076"}],
               "adapter": {"variant": "unconstrained_rplan", "steps": 1000,
                           "checkpoints": {"node": {"sha256": "node"}, "edge": {"sha256": "edge"}}}})
    write_json(path / "run.json", {"method": method, "trials": sum(row["selected"] for row in rows),
               "total_wall_seconds": seconds, "node_calls": 2000, "edge_calls": 2,
               "candidate_evaluations": 2})
    (path / "candidates.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    return path


def test_coverage_auxiliary_and_invalid_denominators(tmp_path):
    rows = [candidate(0, [result("C076", "pass", assumed=True), result("C033", "unknown"), result("C034", "pass", auxiliary=True)]),
            candidate(1, [result("C076", "fail", assumed=True), result("C033", "not_applicable"), result("C034", "pass", auxiliary=True)], valid=False),
            candidate(0, [result("C076", "pass")], selected=False)]
    summary = write_report(make_run(tmp_path / "baseline", rows))
    assert summary["selected_trials"] == 2 and summary["saved_final_candidates"] == 3
    assert summary["compliance"] == 0.5
    assert summary["validity_rate"] == 0.5
    assert summary["measurable_fraction"] == pytest.approx(2 / 3)
    assert summary["unknown_fraction"] == pytest.approx(1 / 3)
    assert summary["assumption_based_fraction"] == pytest.approx(2 / 3)
    assert summary["severity"]["mean"] == 0.25
    assert summary["all_measurable_pass_fraction"] == 0.5
    assert summary["all_applicable_pass_fraction"] == 0.0
    assert "C034" not in summary["per_constraint"] and summary["auxiliary_checks"]["C034"]["pass"] == 2
    assert summary["graph_statistics"]["room_area_drawing_units2"]["mean"] == 12
    assert (tmp_path / "baseline" / "report.md").exists()


def test_all_unknown_is_not_perfect_or_zero_severity(tmp_path):
    summary = summarize_run(make_run(tmp_path / "unknown", [candidate(0, [result("C076", "unknown")])]))
    assert summary["compliance"] is None
    assert summary["severity"]["mean"] is None
    assert summary["measurable_fraction"] == 0
    assert summary["all_measurable_pass_fraction"] == 0
    assert summary["overall_valid_rate"] == 0
    assert summary["overall_validity_counts"]["unknown"] == 1
    assert summary["overall_validity_counts"]["invalid"] == 0


def test_overall_validity_and_structural_validity_have_separate_denominators(tmp_path):
    rows = [
        candidate(0, [result("C076", "pass")]),
        candidate(1, [result("C076", "pass")], valid=False),
        candidate(2, [result("C076", "fail"), result("C033", "unknown")]),
        candidate(3, [result("C076", "unknown")]),
        candidate(4, [result("C076", "pass"), result("C034", "fail", auxiliary=True)]),
        candidate(5, [result("C076", "pass"), result("C033", "not_applicable")]),
        candidate(6, [result("C033", "not_applicable")]),
        candidate(7, []),
        candidate(0, [result("C076", "pass")], selected=False),
    ]
    summary = write_report(make_run(tmp_path / "overall", rows))
    assert summary["overall_validity_counts"] == {"valid": 3, "invalid": 2, "unknown": 1, "not_applicable": 2}
    assert summary["overall_valid_count"] == 3
    assert summary["overall_valid_rate"] == 3 / 8
    assert summary["structural_valid_count"] == summary["valid_count"] == 7
    assert summary["structural_valid_rate"] == summary["validity_rate"] == 7 / 8
    assert summary["validity_definitions"]["legacy_aliases"]["validity_rate"] == "structural_valid_rate"
    all_candidates = summary["all_candidates"]
    assert all_candidates["overall_validity_counts"] == {"valid": 4, "invalid": 2, "unknown": 1, "not_applicable": 2}
    assert all_candidates["overall_valid_rate"] == 4 / 9
    assert all_candidates["structural_valid_rate"] == 8 / 9
    report = (tmp_path / "overall" / "report.md").read_text()
    assert report.index("**Overall valid**") < report.index("Structurally valid")
    assert "not_applicable and does not count as success" in report


def test_comparison_labels_overall_and_structural_validity_explicitly(tmp_path):
    summary = write_report(make_run(tmp_path / "baseline", [candidate(0, [result("C076", "fail")])]))
    table = write_comparison(tmp_path, [summary])
    assert table[0]["overall_valid_rate"] == 0
    assert table[0]["structural_valid_rate"] == table[0]["validity_rate"] == 1
    assert table[0]["overall_invalid_count"] == 1
    with (tmp_path / "comparison.csv").open() as stream:
        row = next(csv.DictReader(stream))
    assert row["overall_valid_rate"] == "0.0"
    assert row["structural_valid_rate"] == "1.0"
    assert row["validity_rate_alias_of"] == "structural_valid_rate"
    markdown = (tmp_path / "comparison.md").read_text()
    assert "Overall valid | Structurally valid" in markdown
    assert "unresolved checks without a failure remain unknown" in markdown


def test_runtime_multiplier_is_per_trial_and_scenario_matched(tmp_path):
    baseline = write_report(make_run(tmp_path / "baseline", [candidate(0, [result("C076", "pass")])], seconds=10))
    best = write_report(make_run(tmp_path / "best", [candidate(i, [result("C076", "pass")]) for i in range(2)], method="best_of_n", seconds=40))
    other_scale = write_report(make_run(tmp_path / "other-scale", [candidate(0, [result("C076", "pass")])], method="svdd_pm", seconds=10, scale=0.05))
    table = write_comparison(tmp_path, [baseline, best, other_scale])
    assert table[0]["compute_multiplier"] == 1
    assert table[1]["compute_multiplier"] == 2
    assert table[2]["compute_multiplier"] is None
    assert "none measured yet" in (tmp_path / "comparison.md").read_text()


def test_batched_compute_preserves_calls_states_and_total_time(tmp_path):
    rows = [candidate(i, [result("C076", "pass")]) for i in range(4)]
    for row in rows:
        row["stats"] = {"wall_seconds": 3.0}
    directory = make_run(tmp_path / "batched", rows, seconds=12)
    config = json.loads((directory / "config.json").read_text())
    config.update(batch_size=4, inference_batch_size=10)
    write_json(directory / "config.json", config)
    run = json.loads((directory / "run.json").read_text())
    run.update(node_calls=1000, edge_calls=1, node_samples=4000, edge_samples=4)
    write_json(directory / "run.json", run)

    summary = write_report(directory)
    assert summary["parameters"]["batch_size"] == 4
    assert summary["parameters"]["inference_batch_size"] == 10
    assert summary["compute"]["total_wall_seconds"] == 12
    assert summary["compute"]["wall_seconds_per_trial"] == 3
    assert (summary["compute"]["node_calls"], summary["compute"]["node_samples"]) == (1000, 4000)
    assert (summary["compute"]["edge_calls"], summary["compute"]["edge_samples"]) == (1, 4)
    report = (directory / "report.md").read_text()
    assert "amortized throughput, not isolated latency" in report
    assert "| Node / edge evaluated states | 4000 / 4 |" in report
    table = write_comparison(tmp_path, [summary])
    assert table[0]["node_samples"] == 4000 and table[0]["edge_samples"] == 4
    assert table[0]["batch_size"] == 4 and table[0]["inference_batch_size"] == 10
    with (tmp_path / "comparison.csv").open() as stream:
        saved = next(csv.DictReader(stream))
    assert saved["node_calls"] == "1000" and saved["node_samples"] == "4000"


def test_legacy_state_counts_are_unknown_and_batching_defaults_to_one(tmp_path):
    directory = make_run(tmp_path / "legacy", [candidate(0, [result("C076", "pass")])])
    summary = write_report(directory)
    assert summary["parameters"]["batch_size"] == summary["parameters"]["inference_batch_size"] == 1
    assert summary["compute"]["node_samples"] is None and summary["compute"]["edge_samples"] is None
    assert summary["parameters"]["reward_backend"] == "cpu"
    assert summary["compute"]["gpu_candidate_evaluations"] is None
    assert summary["compute"]["cpu_candidate_evaluations"] is None
    assert "| Node / edge evaluated states | unknown / unknown |" in (directory / "report.md").read_text()
    assert "| Tensor / CPU-reference constraint evaluations | unknown / unknown |" in (directory / "report.md").read_text()
    config = json.loads((directory / "config.json").read_text())
    assert comparison_group(config) == comparison_group({**config, "batch_size": 1, "inference_batch_size": 1})
    assert comparison_group(config) == comparison_group({**config, "reward_backend": "cpu",
                                                         "guidance": {"tensor_settings": None}})
    write_comparison(tmp_path, [summary])
    with (tmp_path / "comparison.csv").open() as stream:
        row = next(csv.DictReader(stream))
    assert row["node_samples"] == row["edge_samples"] == ""


@pytest.mark.parametrize("batch_size,inference_batch_size", [(4, 1), (1, 10)])
def test_compute_matching_requires_identical_batching_controls(tmp_path, batch_size, inference_batch_size):
    baseline = write_report(make_run(tmp_path / "baseline", [candidate(0, [result("C076", "pass")])], seconds=10))
    directory = make_run(tmp_path / "best", [candidate(0, [result("C076", "pass")])], method="best_of_n", seconds=10)
    config = json.loads((directory / "config.json").read_text())
    config.update(batch_size=batch_size, inference_batch_size=inference_batch_size)
    write_json(directory / "config.json", config)
    best = write_report(directory)
    table = write_comparison(tmp_path, [baseline, best])
    assert best["comparison_group"] != baseline["comparison_group"]
    assert table[1]["compute_multiplier"] is None
    assert "none measured yet" in (tmp_path / "comparison.md").read_text()


def test_duplicate_selected_trial_is_not_silently_averaged(tmp_path):
    directory = make_run(tmp_path / "bad", [candidate(0, [result("C076", "pass")]) for _ in range(2)])
    with pytest.raises(ValueError, match="one selected candidate"):
        summarize_run(directory)


def test_partial_run_is_labelled_and_not_a_compute_reference(tmp_path):
    directory = make_run(tmp_path / "partial", [candidate(0, [result("C076", "pass")])])
    config = json.loads((directory / "config.json").read_text())
    config["trials"] = 8
    write_json(directory / "config.json", config)
    partial = write_report(directory)
    assert partial["status"] == "incomplete" and partial["expected_trials"] == 8
    rows = write_comparison(tmp_path, [partial])
    assert rows[0]["compute_multiplier"] is None
    assert "**incomplete**" in (directory / "report.md").read_text()


def test_png_and_svg_include_actual_checks_and_assumptions(tmp_path):
    row = candidate(0, [result("C076", "fail", assumed=True), result("C033", "unknown")])
    paths = save_sample(row["sample"], row["evaluation"], tmp_path / "sample")
    with Image.open(paths["png"]) as image:
        assert image.size[0] == 1120 and image.getextrema() != ((255, 255),) * 3
    svg = Path(paths["svg"]).read_text()
    assert "C076: fail" in svg and "C033: unknown" in svg
    assert "area_min_m2" in svg and "area_m2" in svg and "assumed" in svg


@pytest.mark.parametrize("check_status,overall_label", [("fail", "INVALID"), ("unknown", "UNKNOWN")])
def test_svg_distinguishes_overall_from_structural_validity(tmp_path, check_status, overall_label):
    row = candidate(0, [result("C076", check_status)], valid=True)
    paths = save_sample(row["sample"], row["evaluation"], tmp_path / check_status)
    svg = Path(paths["svg"]).read_text()
    assert f"overall: {overall_label} | structure: valid" in svg


def test_gpu_comparison_matches_backend_and_search_settings_before_runtime(tmp_path):
    def configured(name, method, seconds, backend="cpu", positions=20000):
        directory = make_run(tmp_path / name, [candidate(0, [result("C076", "pass")])],
                             method=method, seconds=seconds)
        config = json.loads((directory / "config.json").read_text())
        config["reward_backend"] = backend
        config["guidance"] = {"backend": backend, "final_evaluator": "cpu_reference",
                              "tensor_settings": {"backend": "torch_finite_grid_v1", "max_bed_positions_per_room": positions}
                              if backend == "gpu" else None}
        write_json(directory / "config.json", config)
        return write_report(directory)

    cpu_baseline = configured("cpu-baseline", "baseline", 10)
    gpu_best = configured("gpu-best", "best_of_n", 40, "gpu")
    first = write_comparison(tmp_path, [cpu_baseline, gpu_best])
    assert first[1]["compute_multiplier"] is None
    assert cpu_baseline["comparison_group"] != gpu_best["comparison_group"]

    gpu_baseline = configured("gpu-baseline", "baseline", 20, "gpu")
    changed_search = configured("gpu-different-search", "best_of_n", 40, "gpu", positions=1000)
    table = write_comparison(tmp_path, [cpu_baseline, gpu_baseline, gpu_best, changed_search])
    assert gpu_baseline["comparison_group"] == gpu_best["comparison_group"]
    assert changed_search["comparison_group"] != gpu_best["comparison_group"]
    assert [row["compute_multiplier"] for row in table] == [1, 1, 2, None]
    assert [row["reward_backend"] for row in table] == ["cpu", "gpu", "gpu", "gpu"]
    assert "guidance backend" in (tmp_path / "comparison.md").read_text()


def test_gpu_metadata_never_overrides_cpu_final_metrics_and_counters_export(tmp_path):
    row = candidate(0, [result("C076", "fail")])
    row["sample"]["metadata"] = {
        "guidance_backend": "tensor_gpu", "final_evaluation_backend": "cpu_reference",
        "guidance_evaluation": {"structurally_valid": True, "energy": 0.0,
                                "results": [{"id": "C076", "status": "pass"}]},
    }
    directory = make_run(tmp_path / "gpu", [row], method="svdd_pm")
    config = json.loads((directory / "config.json").read_text())
    config.update(reward_backend="gpu", guidance={"backend": "gpu", "final_evaluator": "cpu_reference",
                                                 "tensor_settings": {"cpu_oracle_equivalence": False}})
    write_json(directory / "config.json", config)
    counters = json.loads((directory / "run.json").read_text())
    counters.update(candidate_evaluations=101, gpu_candidate_evaluations=100, cpu_candidate_evaluations=1)
    write_json(directory / "run.json", counters)

    summary = write_report(directory)
    assert summary["overall_valid_rate"] == 0 and summary["structural_valid_rate"] == 1
    assert summary["compliance"] == 0 and summary["per_constraint"]["C076"]["fail"] == 1
    assert summary["guidance"]["final_evaluator"] == "cpu_reference"
    assert summary["compute"]["candidate_evaluations"] == 101
    assert summary["compute"]["gpu_candidate_evaluations"] == 100
    assert summary["compute"]["cpu_candidate_evaluations"] == 1
    markdown = (directory / "report.md").read_text()
    assert "Guidance backend: **gpu**" in markdown and "**CPU reference checker**" in markdown
    assert "| Tensor / CPU-reference constraint evaluations | 100 / 1 |" in markdown
    table = write_comparison(tmp_path, [summary])
    assert table[0]["gpu_candidate_evaluations"] == 100 and table[0]["cpu_candidate_evaluations"] == 1
    with (tmp_path / "comparison.csv").open() as stream:
        saved = next(csv.DictReader(stream))
    assert saved["reward_backend"] == "gpu"
    assert saved["gpu_candidate_evaluations"] == "100" and saved["cpu_candidate_evaluations"] == "1"
