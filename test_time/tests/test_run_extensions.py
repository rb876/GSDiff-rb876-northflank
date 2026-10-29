"""Episode accounting must never hide unsuccessful hard-feasibility attempts."""

import json

import pytest

from test_time.run_extensions import _parser, load_configuration, summarize_root, summarize_trials, write_summary


def row(valid=False, returned=True, status=None):
    return {"returned_graph": returned, "overall_validity": "valid" if valid else "invalid" if returned else "no_graph",
            "status": status or ("success" if valid else "infeasible"),
            "structurally_valid": returned,
            "checks": {"C035": "pass" if valid else "fail"} if returned else {},
            "stats": {"wall_seconds": 3., "node_samples": 1000, "any_feasible_found": valid}}


def test_success_denominator_includes_no_graph_and_exhausted_attempts():
    rows = [row(True), row(False), row(returned=False), row(returned=False, status="budget_exhausted")]
    summary = summarize_trials(rows, "hard_h_mc")
    assert summary["success_rate"] == .25
    assert summary["returned_graphs"] == 2
    assert summary["no_graph_attempts"] == 2
    assert summary["nfe_per_success"] == 4000
    assert summary["seconds_per_success"] == 12
    assert summary["constraint_pass_per_attempt"]["C035"] == .25
    assert summary["constraint_outcomes_on_returned_graphs"]["C035"] == {"pass": 1, "fail": 1}


def test_successful_lookahead_does_not_turn_failed_return_into_success():
    failed = row(returned=False)
    failed["stats"]["any_feasible_found"] = True
    summary = summarize_trials([failed], "hard_h_mc")
    assert summary["success_rate"] == 0
    assert summary["any_feasible_found"] == 1
    assert summary["seconds_per_success"] is None
    assert summary["nfe_per_success"] is None


def test_empty_and_all_failure_reports_serialize_without_nan(tmp_path):
    for rows in ([], [row(returned=False)]):
        result = write_summary(tmp_path, rows, "hard_h_mc")
        assert json.loads((tmp_path / "summary.json").read_text()) == result
        assert "NaN" not in (tmp_path / "summary.json").read_text()
        assert "attempt" in (tmp_path / "report.md").read_text()


def test_combined_report_keeps_attempt_denominators(tmp_path):
    for method, rows in (("hard_h_mc", [row(True), row(returned=False)]),
                         ("svdd_repair_aware", [row(True), row(False)])):
        folder = tmp_path / method
        folder.mkdir()
        write_summary(folder, rows, method)
    summarize_root(tmp_path)
    report = (tmp_path / "comparison.md").read_text()
    assert report.count("50.0%") == 2
    assert report.count("1/2") == 2
    assert (tmp_path / "comparison.csv").exists()


def test_cli_overrides_and_existing_policy_are_reused():
    args = _parser().parse_args(["--method", "hard_h_mc", "--trials", "2", "--k", "2", "--rollouts", "2"])
    config, parameters, scenario, specs = load_configuration(args)
    assert config["trials"] == 2
    assert parameters["k"] == parameters["rollouts"] == 2
    assert parameters["anchor_remaining"] == [200, 100, 30]
    assert scenario.coordinate_scale_m_per_model_unit == .03515625
    assert {s.id for s in specs if s.enabled and s.counts_toward_compliance} == {
        "C033", "C035", "C036", "C076", "C077", "C078", "C085", "C086", "C090"}


def test_repair_cli_rejects_irrelevant_rollout_override():
    args = _parser().parse_args(["--method", "svdd_repair_aware", "--rollouts", "3"])
    with pytest.raises(ValueError, match="only to hard_h_mc"):
        load_configuration(args)
