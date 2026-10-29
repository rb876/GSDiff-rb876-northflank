"""Re-auditing changes evaluations, never selected designs or generation provenance."""

from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path

import pytest

from test_time.constraints import load_specs
from test_time.evaluate import comparison_group, summarize_run
from test_time.policy import ConstraintEvaluator
from test_time.reaudit import CHECKER_VERSION, reaudit
from test_time.types import FloorplanSample, Room, Scenario, overall_validity, read_json, write_json


def saved_candidates(path):
    path.mkdir(parents=True)
    specs = load_specs(Path(__file__).resolve().parents[1] / "configs/constraints_v2.json")
    scenario = Scenario(coordinate_scale_m_per_model_unit=1.0, scale_provenance="assumed")
    points = [[0, 0], [4, 0], [4, 4], [0, 4]]
    sample = FloorplanSample(nodes=points, edges=[[0, 1], [1, 2], [2, 3], [3, 0]],
                             rooms=[Room(1, points + [points[0]])])
    evaluation = asdict(ConstraintEvaluator(specs).evaluate(sample, scenario))
    # A historical unresolved result exercises the changed-outcome export.
    c035 = next(check for check in evaluation["results"] if check["id"] == "C035")
    c035.update(status="unknown", severity=None, measurable=False, explanation="Historical search was unresolved")
    c035.pop("group")  # Retain the literal source, not the reporting view's added defaults.
    evaluation["unknown_count"] += 1
    evaluation["measurable_count"] -= 1
    selected = {"trial": 0, "seed": 456, "candidate_id": "chosen", "selected": True, "method": "best_of_n",
                "sample": asdict(sample), "evaluation": evaluation, "overall_validity": overall_validity(evaluation),
                "stats": {"wall_seconds": 3.5}, "artifacts": {"png": "samples/chosen.png"}}
    rejected = {**deepcopy(selected), "candidate_id": "rejected", "seed": 123, "selected": False}
    config = {"method": "best_of_n", "n": 2, "seed": 10000, "trials": 1,
              "scenario": asdict(scenario), "constraints": [asdict(spec) for spec in specs],
              "validity_weight": 100.0, "reward_backend": "gpu"}
    run = {"status": "complete", "trials": 1, "node_samples": 2000, "total_wall_seconds": 3.5}
    write_json(path / "config.json", config)
    write_json(path / "run.json", run)
    (path / "candidates.jsonl").write_text("\n".join(json.dumps(row) for row in [rejected, selected]) + "\n")
    (path / "original-report.md").write_text("Keep the original report.\n")
    return selected, config, run


def test_reaudit_preserves_sources_selected_identity_and_compute(tmp_path):
    source = tmp_path / "sweep"
    run_dir = source / "batch_1" / "best_of_n"
    selected, config, run = saved_candidates(run_dir)
    before = {str(path.relative_to(source)): path.read_bytes() for path in source.rglob("*") if path.is_file()}
    result = reaudit(source, workers=1)
    output = Path(result["output_dir"])
    revised_dir = output / "batch_1" / "best_of_n"
    rows = [json.loads(line) for line in (revised_dir / "candidates.jsonl").read_text().splitlines()]
    assert len(rows) == result["selected_outputs"] == 1
    row = rows[0]
    for field in ("sample", "seed", "candidate_id", "trial", "selected", "stats"):
        assert row[field] == selected[field]
    assert row["original_evaluation"] == selected["evaluation"]
    assert row["original_artifacts"] == selected["artifacts"] and "artifacts" not in row
    assert next(check for check in row["evaluation"]["results"] if check["id"] == "C035")["status"] == "pass"
    assert before == {str(path.relative_to(source)): path.read_bytes() for path in source.rglob("*") if path.is_file()}
    assert read_json(revised_dir / "run.json") == run
    revised_config = read_json(revised_dir / "config.json")
    assert {key: value for key, value in revised_config.items() if key != "evaluation_audit"} == config
    assert comparison_group(config) != comparison_group(revised_config)
    summary = summarize_run(revised_dir)
    assert summary["compute"]["node_samples_per_trial"] == 2000
    assert summary["compute"]["wall_seconds_per_trial"] == 3.5
    assert summary["evaluation_audit"]["checker_version"] == CHECKER_VERSION
    audit = read_json(output / "audit.json")
    assert audit["status"] == "complete" and audit["methods"][0]["source_saved_candidates"] == 2
    assert audit["methods"][0]["structural_status_changes"] == 0
    assert "C035,unknown,pass" in (output / "changes.csv").read_text()
    assert (output / "AUDIT.md").is_file()


def test_reaudit_refuses_source_nested_and_existing_destinations(tmp_path):
    source = tmp_path / "sweep"
    saved_candidates(source / "baseline")
    for output in (source, source / "audit", tmp_path):
        with pytest.raises(ValueError, match="separate sibling"):
            reaudit(source, output, workers=1)
    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(FileExistsError, match="already exists"):
        reaudit(source, existing, workers=1)
