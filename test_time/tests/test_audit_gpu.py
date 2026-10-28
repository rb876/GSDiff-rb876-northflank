"""Offline audits exercise actual tensor scoring, without models or inference."""

from dataclasses import asdict
import json

import pytest
import torch

from test_time.audit_gpu import audit, main
from test_time.constraints import load_specs
from test_time.types import Scenario


@pytest.fixture
def experiment(tmp_path):
    specs = load_specs()
    for spec in specs:
        spec.enabled = spec.id in {"C076", "C077", "C078"}
    scenario = Scenario(intended_occupants=1, coordinate_scale_m_per_model_unit=1.0,
                        scale_provenance="assumed")
    (tmp_path / "config.json").write_text(json.dumps({
        "scenario": asdict(scenario), "constraints": [asdict(spec) for spec in specs], "validity_weight": 123.0}))
    points = [[0, 0], [4, 0], [4, 3], [0, 3]]
    edges = [[0, 1], [1, 2], [2, 3], [0, 3]]
    sample = {"nodes": [point + [1, 0, 0, 0, 0, 0, 0] for point in points],
              "raw_edges": edges, "edges": edges[:-1],
              "rooms": [{"label": 0, "polygon": points + [points[0]]}]}
    evaluation = {"structurally_valid": True, "results": [
        {"id": "C076", "status": "unknown"}, {"id": "C077", "status": "fail"},
        {"id": "C078", "status": "fail"}]}
    rows = [{"sample": sample, "evaluation": evaluation, "candidate_id": f"candidate-{index}",
             "trial": index, "selected": index == 0} for index in range(2)]
    (tmp_path / "candidates.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    return tmp_path


def test_actual_scoring_records_disagreement_and_historical_scope(experiment):
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        report = audit(experiment, limit=1)
    finally:
        torch.set_num_threads(previous)
    assert report["available_candidates"] == 2
    assert report["summary"]["compared"] == 1
    assert report["summary"]["room_geometry_matches"] == 1
    assert report["per_check"]["C076"]["gpu_pass_cpu_not_pass"] == 1
    assert report["per_check"]["C077"]["matches"] == 1
    assert report["runs"][0]["validity_weight"] == 123.0
    result = report["comparisons"][0]
    assert result["candidate_id"] == "candidate-0" and result["selected"]
    assert result["edge_source"] == "raw_edges"
    assert result["summed_room_area_drawing_units2"] == {"cpu": 12.0, "gpu": 12.0}
    assert "historical snapshots" in report["interpretation"]


def test_cli_writes_report_and_falls_back_to_saved_edges(experiment):
    path = experiment / "candidates.jsonl"
    row = json.loads(path.read_text().splitlines()[0])
    row["sample"]["edges"] = row["sample"].pop("raw_edges")
    path.write_text(json.dumps(row) + "\n")
    main(["--experiment_dir", str(experiment), "--limit", "1"])
    report = json.loads((experiment / "gpu_audit.json").read_text())
    assert report["comparisons"][0]["edge_source"] == "edges"
    assert report["summary"]["structural_validity_matches"] == 1


def test_invalid_input_is_not_silently_truncated_or_accepted(experiment):
    with pytest.raises(ValueError, match="positive"):
        audit(experiment, limit=0)
    path = experiment / "candidates.jsonl"
    row = json.loads(path.read_text().splitlines()[0])
    row["sample"]["raw_edges"].append([0, 53])
    path.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="invalid compact-node references"):
        audit(experiment)
