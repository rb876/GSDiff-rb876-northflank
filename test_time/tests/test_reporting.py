"""Reporting denominators, compute comparisons and annotated output artifacts."""

from dataclasses import asdict
import csv
import json
from pathlib import Path

from PIL import Image
import pytest

from test_time.evaluate import constraint_selection, comparison_group, summarize_run, write_comparison, write_report
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
    assert summary["compute"]["node_samples_per_trial"] == 1000
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
    assert saved["node_samples_per_trial"] == "1000.0"


def test_legacy_state_counts_are_unknown_and_batching_defaults_to_one(tmp_path):
    directory = make_run(tmp_path / "legacy", [candidate(0, [result("C076", "pass")])])
    summary = write_report(directory)
    assert summary["parameters"]["batch_size"] == summary["parameters"]["inference_batch_size"] == 1
    assert summary["compute"]["node_samples"] is None and summary["compute"]["edge_samples"] is None
    assert summary["compute"]["node_samples_per_trial"] is None
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
    assert row["node_samples"] == row["edge_samples"] == row["node_samples_per_trial"] == ""


def test_comparison_groups_use_selected_checker_not_recalibration_provenance():
    legacy = {"scenario": {"coordinate_scale_m_per_model_unit": 18 / 512},
              "constraints": [{"id": "C035"}], "reward_backend": "gpu"}
    refined = {**legacy, "final_checker": "cpu_bedroom_audit_v2",
               "guidance": {"final_evaluator": "cpu_bedroom_audit_v2"}}
    audited = {**legacy, "evaluation_audit": {"checker_version": "cpu_bedroom_audit_v2"}}
    recalibrated = {**legacy, "scale_recalibration": {
        "selected_evaluation_backend": "cpu_bedroom_audit_v2",
        "population_evaluation_backend": "mixed_tensor_cpu", "source_root": "/historical/run"}}
    assert comparison_group(legacy) != comparison_group(refined)
    assert comparison_group(refined) == comparison_group(audited) == comparison_group(recalibrated)
    assert comparison_group(legacy) == comparison_group({**legacy, "final_checker": "cpu_reference"})
    # Latest selected-output evaluation takes precedence over historical settings.
    assert comparison_group({**refined, "scale_recalibration": {
        "selected_evaluation_backend": "cpu_reference"}}) == comparison_group(legacy)


def test_mixed_recalibration_reports_selected_cpu_outputs_without_pooling_tensor_scores(tmp_path):
    rows = [candidate(0, [result("C035", "fail")]),
            candidate(0, [result("C035", "pass")], selected=False)]
    directory = make_run(tmp_path / "recalibrated", rows, method="best_of_n", seconds=None)
    config = json.loads((directory / "config.json").read_text())
    config["scale_recalibration"] = {
        "selected_evaluation_backend": "cpu_bedroom_audit_v2",
        "population_evaluation_backend": "mixed_tensor_cpu"}
    write_json(directory / "config.json", config)
    summary = write_report(directory)
    assert summary["saved_final_candidates"] == 2 and summary["selected_trials"] == 1
    assert summary["overall_valid_rate"] == 0 and summary["per_constraint"]["C035"]["fail"] == 1
    assert summary["final_evaluation_backend"] == "cpu_bedroom_audit_v2"
    assert summary["all_candidates"]["count"] == 2
    assert summary["all_candidates"]["status"] == "not_reported"
    assert "overall_valid_rate" not in summary["all_candidates"]
    assert "compliance" not in summary["all_candidates"]
    assert summary["compute"]["wall_seconds_per_trial"] is None
    report = (directory / "report.md").read_text()
    assert "Mixed checker results are not pooled" in report
    assert "No new diffusion was run" in report
    assert "cpu_bedroom_audit_v2" in report


def test_prefix_control_reports_equivalent_nfe_without_pooling_mixed_checks(tmp_path):
    rows = [candidate(0, [result("C035", "fail")]),
            candidate(0, [result("C035", "pass")], selected=False),
            candidate(0, [result("C035", "unknown")], selected=False)]
    directory = make_run(tmp_path / "best_of_n_late", rows, method="best_of_n", seconds=None)
    config = json.loads((directory / "config.json").read_text())
    config.update(n=3, final_checker="cpu_bathroom_audit_v1", prefix_reselection={
        "source_n": 11, "n": 3, "population_evaluation_backend": "mixed_tensor_cpu"})
    write_json(directory / "config.json", config)
    run = json.loads((directory / "run.json").read_text())
    run.update(node_samples=3000, node_calls=None, edge_calls=None, edge_samples=None)
    write_json(directory / "run.json", run)
    summary = write_report(directory)
    assert summary["overall_valid_rate"] == 0
    assert summary["all_candidates"]["count"] == 3
    assert summary["all_candidates"]["status"] == "not_reported"
    assert summary["compute"]["node_samples_per_trial"] == 3000
    assert summary["compute"]["nfe_provenance"] == "derived_prefix_budget"
    assert summary["compute"]["node_calls"] is None
    assert summary["compute"]["wall_seconds_per_trial"] is None
    write_comparison(tmp_path, [summary])
    assert "Fixed-prefix Best-of-3" in (directory / "report.md").read_text()
    assert "equivalent retained-trajectory budget" in (tmp_path / "comparison.md").read_text()


def denoiser_run(path, method, evaluations, trial_ids=(0, 1), **settings):
    directory = make_run(path, [candidate(trial, [result("C076", "pass")]) for trial in trial_ids],
                         method=method)
    config = json.loads((directory / "config.json").read_text())
    config.update(seed=10000, batch_size=4, inference_batch_size=10, **settings)
    write_json(directory / "config.json", config)
    run = json.loads((directory / "run.json").read_text())
    run.update(status="complete", node_samples=evaluations * len(trial_ids) if evaluations is not None else None)
    write_json(directory / "run.json", run)
    return directory


def test_denoiser_matching_uses_recorded_states_for_both_svdd_budgets(tmp_path):
    directories = [
        denoiser_run(tmp_path / "best_late", "best_of_n", 3000, n=3),
        denoiser_run(tmp_path / "svdd_late", "svdd_pm", 3000, k=10, guided_steps=200),
        denoiser_run(tmp_path / "best_full", "best_of_n", 11000, n=11),
        denoiser_run(tmp_path / "svdd_full", "svdd_pm", 10990, k=10, guided_steps=999),
    ]
    summaries = [write_report(directory) for directory in directories]
    rows = write_comparison(tmp_path, summaries)
    assert [row["node_samples_per_trial"] for row in rows] == [3000, 3000, 11000, 10990]
    # Every fake run records identical calls and time; matches must use states instead.
    saved = json.loads((tmp_path / "denoiser_comparison.json").read_text())
    assert saved["metric"] == "node_samples_per_selected_output"
    pairs = {(pair["best_of_n"], pair["svdd"]): pair for pair in saved["comparisons"]}
    assert len(pairs) == 4
    late = pairs["best_late", "svdd_late"]
    assert late["matched"] and late["exact"] and late["ratio_best_of_n_to_svdd"] == 1
    full = pairs["best_full", "svdd_full"]
    assert full["matched"] and not full["exact"]
    assert full["relative_difference"] == pytest.approx(10 / 10990)
    assert not pairs["best_late", "svdd_full"]["matched"]
    assert not pairs["best_full", "svdd_late"]["matched"]
    markdown = (tmp_path / "comparison.md").read_text()
    assert "0.091%" in markdown and "within 1%" in markdown
    assert "Edge prediction, constraint checks and elapsed time are not matched" in markdown
    assert "Approximately matched compute" not in markdown


@pytest.mark.parametrize("difference", ["seed", "unknown_seed", "batch_size", "inference_batch_size", "partial", "trials", "trial_ids"])
def test_denoiser_matching_requires_complete_paired_compatible_runs(tmp_path, difference):
    best_dir = denoiser_run(tmp_path / "best", "best_of_n", 3000)
    trials = (0,) if difference == "trials" else (0, 2) if difference == "trial_ids" else (0, 1)
    svdd_dir = denoiser_run(tmp_path / "svdd", "svdd_pm", 3000, trial_ids=trials)
    config = json.loads((svdd_dir / "config.json").read_text())
    if difference in {"seed", "unknown_seed"}:
        config["seed"] = 20000 if difference == "seed" else None
    elif difference in {"batch_size", "inference_batch_size"}:
        config[difference] = 1
    elif difference == "partial":
        config["trials"] = 100
    write_json(svdd_dir / "config.json", config)
    write_comparison(tmp_path, [write_report(best_dir), write_report(svdd_dir)])
    saved = json.loads((tmp_path / "denoiser_comparison.json").read_text())
    assert saved["comparisons"] == []


def test_denoiser_matching_preserves_unknown_counts(tmp_path):
    best = write_report(denoiser_run(tmp_path / "best", "best_of_n", None))
    svdd = write_report(denoiser_run(tmp_path / "svdd", "svdd_pm", 3000))
    write_comparison(tmp_path, [best, svdd])
    pair = json.loads((tmp_path / "denoiser_comparison.json").read_text())["comparisons"][0]
    assert pair["best_of_n_evaluations_per_output"] is None
    assert pair["ratio_best_of_n_to_svdd"] is pair["relative_difference"] is pair["matched"] is pair["exact"] is None
    assert "| unknown |" in (tmp_path / "comparison.md").read_text()


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
        assert image.size[0] == 1440 and image.getextrema() != ((255, 255),) * 3
    svg = Path(paths["svg"]).read_text()
    assert "C076 Minimum main-room size" in svg and ">FAIL<" in svg
    assert "C033 Bedroom turning space" in svg and ">UNRESOLVED<" in svg
    assert "Requirement:" in svg and "Why:" in svg and "assumed" in svg
    assert "Observed:" not in svg and "area_min_m2" not in svg
    assert "Living" in svg


@pytest.mark.parametrize("check_status,overall_label", [("fail", "INVALID"), ("unknown", "UNRESOLVED")])
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


def registry_v2():
    groups = {"bedroom_accessibility": ["C033", "C034", "C035"],
              "bathroom_fixtures_turning": ["C036"],
              "habitability": ["C076", "C077", "C078", "C079", "C080"]}
    supporting = {"C034", "C079", "C080"}
    return [{"id": identifier, "group": group,
             "role": "supporting_rule" if identifier in supporting else "check",
             "counts_toward_compliance": identifier not in supporting,
             "enabled": True}
            for group, identifiers in groups.items() for identifier in identifiers]


def set_registry(directory, specs):
    config = json.loads((directory / "config.json").read_text())
    config["constraints"] = specs
    write_json(directory / "config.json", config)


def test_registry_selection_counts_and_legacy_auxiliary_role():
    selected = constraint_selection(registry_v2())
    assert selected["constraint_counts"] == {"registered": 9, "scored": 6, "supporting": 3,
                                             "active": 9, "active_scored": 6, "active_supporting": 3, "disabled": 0}
    assert selected["active_scored_ids"] == ["C033", "C035", "C036", "C076", "C077", "C078"]
    assert selected["active_supporting_ids"] == ["C034", "C079", "C080"]
    assert selected["disabled_constraints"] == []
    legacy = constraint_selection([{"id": "C034", "role": "check", "counts_toward_compliance": False},
                                   {"id": "C076"}, {"id": "C030", "enabled": False}])
    assert legacy["active_supporting_ids"] == ["C034"] and legacy["active_scored_ids"] == ["C076"]
    assert "reason not recorded" in legacy["disabled_constraints"][0]["reason"]


def test_v2_groups_and_supporting_rules_do_not_inflate_denominators(tmp_path):
    ids = ["C033", "C035", "C036", "C076", "C077", "C078"]
    first = [result(identifier, "pass") for identifier in ids]
    second = [result(identifier, "unknown" if identifier == "C033" else "fail" if identifier == "C036" else "pass")
              for identifier in ids]
    for checks in (first, second):
        # Registry roles must remain authoritative even if a stored result used
        # default counts=True.
        checks.extend([result("C034", "fail"), result("C079", "unknown"), result("C080", "pass")])
    directory = make_run(tmp_path / "v2", [candidate(0, first), candidate(1, second)])
    set_registry(directory, registry_v2())
    saved_rows = (directory / "candidates.jsonl").read_text()
    summary = write_report(directory)
    assert summary["constraint_counts"]["registered"] == 9
    assert summary["applicable_count"] == 12 and summary["measurable_count"] == 11
    assert summary["pass"] == 10 and summary["fail"] == summary["unknown"] == 1
    assert summary["compliance"] == pytest.approx(10 / 11)
    assert summary["overall_valid_rate"] == 0.5
    assert set(summary["per_constraint"]) == set(ids)
    assert set(summary["auxiliary_checks"]) == {"C034", "C079", "C080"}
    assert summary["all_candidates"]["applicable_count"] == 12
    bedroom = summary["per_group"]["bedroom_accessibility"]
    assert bedroom["overall_validity_counts"] == {"valid": 1, "invalid": 0, "unknown": 1,
                                                 "not_applicable": 0, "not_assessed": 0}
    assert bedroom["check_pass_rate"] == 0.75 and bedroom["compliance"] == 1.0
    assert summary["per_group"]["habitability"]["overall_valid_rate"] == 1.0
    assert set(summary["per_group"]) == {"bedroom_accessibility", "bathroom_fixtures_turning", "habitability"}
    assert all(group["status"] == "assessed" for group in summary["per_group"].values())
    assert (directory / "candidates.jsonl").read_text() == saved_rows
    markdown = (directory / "report.md").read_text()
    assert "9 IDs" in markdown and "6 scored checks" in markdown
    assert "disabled" not in markdown.lower() and "not_assessed" not in markdown
    comparison = write_comparison(tmp_path, [summary])[0]
    assert comparison["registered_constraint_ids"] == 9 and comparison["disabled_constraint_count"] == 0
    assert "disabled" not in (tmp_path / "comparison.md").read_text().lower()
    assert comparison["active_scored_count"] == 6 and comparison["active_supporting_count"] == 3


def test_missing_active_evaluation_cannot_become_headline_or_group_pass(tmp_path):
    checks = [result(identifier, "pass") for identifier in ["C033", "C035", "C036", "C076", "C077"]]
    directory = make_run(tmp_path / "missing", [candidate(0, checks)])
    set_registry(directory, registry_v2())
    saved = (directory / "candidates.jsonl").read_text()
    summary = write_report(directory)
    assert summary["overall_valid_rate"] == 0
    assert summary["overall_validity_counts"]["unknown"] == 1
    assert summary["per_constraint"]["C078"]["unknown"] == 1
    assert summary["per_constraint"]["C078"]["measurable_count"] == 0
    assert summary["missing_active_results"] == [{"trial": 0, "candidate_id": "0", "ids": ["C078"]}]
    group = summary["per_group"]["habitability"]
    assert group["overall_valid_rate"] == 0 and group["overall_validity_counts"]["unknown"] == 1
    assert group["missing_result_count"] == 1
    assert (directory / "candidates.jsonl").read_text() == saved


def test_group_with_only_supporting_rules_is_not_assessed(tmp_path):
    directory = make_run(tmp_path / "supporting", [candidate(0, [result("C034", "unknown"), result("C030", "fail")])])
    specs = [{"id": "C034", "group": "bedroom", "enabled": True, "role": "supporting_rule"},
             {"id": "C030", "group": "programme", "enabled": False, "disabled_reason": "No WC metadata"}]
    set_registry(directory, specs)
    summary = write_report(directory)
    assert summary["overall_valid_rate"] == 0 and summary["overall_validity_counts"]["not_applicable"] == 1
    assert summary["applicable_count"] == summary["pass"] == summary["unknown"] == 0
    assert summary["per_group"]["bedroom"]["status"] == "not_assessed"
    assert summary["per_group"]["programme"]["status"] == "not_assessed"
    assert summary["per_group"]["bedroom"]["active_supporting_ids"] == ["C034"]
