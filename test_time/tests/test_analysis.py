"""Offline analysis preserves populations, unresolved outcomes and paired trial identity."""

import json
from pathlib import Path

import pytest

from test_time.analyze import (analyze, baseline_for, constraint_metrics, load_records,
                               method_metrics, paired_validity, wilson_interval)
from test_time.tests.test_reporting import candidate, make_run, result
from test_time.types import read_json, write_json


def saved_run(path, rows, method="baseline", seed=10000, status="complete", scale=0.025):
    directory = make_run(path, rows, method=method, seconds=40, scale=scale)
    config = read_json(directory / "config.json")
    config.update(seed=seed, trials=sum(row["selected"] for row in rows), constraints=[
        {"id": "C035", "enabled": True, "counts_toward_compliance": True},
        {"id": "C077", "enabled": True, "counts_toward_compliance": True},
        {"id": "C034", "enabled": True, "counts_toward_compliance": False, "role": "supporting_rule"},
    ])
    write_json(directory / "config.json", config)
    run = read_json(directory / "run.json")
    run.update(status=status, node_samples=1000 * config["trials"])
    write_json(directory / "run.json", run)
    return directory


def mixed_rows():
    return [candidate(0, [result("C035", "pass"), result("C077", "pass"), result("C034", "fail", auxiliary=True)]),
            candidate(1, [result("C035", "fail"), result("C077", "pass")]),
            candidate(2, [result("C035", "unknown"), result("C077", "pass")]),
            candidate(3, [result("C035", "pass"), result("C077", "not_applicable")], valid=False),
            candidate(0, [result("C035", "fail"), result("C077", "fail")], selected=False)]


def test_selected_denominators_and_sole_failures(tmp_path):
    saved_run(tmp_path / "baseline", mixed_rows())
    records, skipped = load_records(tmp_path)
    metrics = method_metrics(records[0])
    assert skipped == [] and len(records[0]["rows"]) == 4
    assert metrics["scored_constraints"] == 2 and metrics["supporting_rules"] == 1
    assert metrics["overall_valid_rate"] == 0.25 and metrics["structural_valid_rate"] == 0.75
    assert metrics["applicability_coverage"] == 7 / 8
    assert metrics["applicable_pass_rate"] == 5 / 7 and metrics["measurable_compliance"] == 5 / 6
    constraints = {row["id"]: row for row in constraint_metrics(records[0])}
    assert set(constraints) == {"C035", "C077"}
    assert constraints["C035"]["pass"] == 2 and constraints["C035"]["unknown"] == 1
    assert constraints["C035"]["sole_failure_count"] == 1
    assert constraints["C077"]["not_applicable"] == 1


def test_completed_methods_are_usable_while_other_runs_are_unfinished(tmp_path):
    saved_run(tmp_path / "baseline", mixed_rows())
    saved_run(tmp_path / "svdd_pm_full", mixed_rows(), method="svdd_pm", status="running")
    records, skipped = load_records(tmp_path)
    assert [record["label"] for record in records] == ["baseline"]
    assert len(skipped) == 1 and skipped[0]["reason"] == "Run status: running"


def test_pairing_uses_trial_seeds_not_candidate_seeds_and_checks_compatibility(tmp_path):
    saved_run(tmp_path / "baseline", mixed_rows())
    selected = [candidate(index, [result("C035", "pass"), result("C077", "pass")]) for index in range(4)]
    for row in selected:
        row["seed"] += 999999  # Winners' seeds differ from the paired trial seeds.
    saved_run(tmp_path / "best_of_n", selected, method="best_of_n")
    saved_run(tmp_path / "different_seed", selected, method="svdd_pm", seed=20000)
    saved_run(tmp_path / "different_scenario", selected, method="svdd_pm", scale=0.05)
    records, _ = load_records(tmp_path)
    pairs = paired_validity(records)
    assert len(pairs) == 1
    assert pairs[0]["only_b_valid"] == 3 and pairs[0]["b_minus_a_validity_points"] == 75
    incompatible = next(record for record in records if record["label"] == "different_scenario")
    assert baseline_for(incompatible, records) is None


def test_wilson_intervals_include_uncertainty_at_zero_and_one():
    assert wilson_interval(0, 0) == (None, None)
    low, high = wilson_interval(50, 100)
    assert low == pytest.approx(0.40383, abs=1e-5) and high == pytest.approx(0.59617, abs=1e-5)
    assert wilson_interval(0, 100)[1] > 0.03
    assert wilson_interval(100, 100)[0] < 0.97


def test_analysis_exports_all_selected_samples_without_changing_inputs(tmp_path):
    pytest.importorskip("matplotlib")
    source = tmp_path / "sweep"
    directory = saved_run(source / "baseline", mixed_rows())
    original = {path.name: path.read_bytes() for path in directory.iterdir()}
    result_summary = analyze(source)
    output = Path(result_summary["output_dir"])
    assert result_summary["selected_outputs"] == 4
    assert len(list((output / "figures").glob("*.png"))) == 6
    assert len(list((output / "figures").glob("*.pdf"))) == 6
    assert len(list((output / "samples").rglob("*.png"))) == 4
    report = (output / "REPORT.md").read_text()
    assert "Technical_assessment_TTC.pdf" in report and "Constraint success" in report
    payload = json.loads((output / "analysis.json").read_text())
    metrics = payload["methods"][0]["metrics"]
    assert metrics["runtime_multiplier_vs_baseline"] == metrics["denoiser_multiplier_vs_baseline"] == 1
    assert payload["gallery"]["sample_count"] == 4
    assert (output / "tables/paired_validity.csv").read_text().startswith("method_a,method_b")
    assert {path.name: path.read_bytes() for path in directory.iterdir()} == original
