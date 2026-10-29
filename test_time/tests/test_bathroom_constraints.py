"""Coupled bathroom checks preserve one completion and honest unknown semantics."""

from dataclasses import replace
from pathlib import Path

import pytest
import torch

from test_time.constraints import load_specs
from test_time.gpu_constraints import TensorConstraintEvaluator
from test_time.gpu_sampling import SCORE_FIELDS, _evaluation_record
from test_time.policy import ConstraintEvaluator
from test_time.tests.test_gpu_constraints import packed, rectangle
from test_time.tests.test_gpu_constraints_v2 import sample_from_rooms
from test_time.types import Scenario, read_json

CONFIGS = Path(__file__).parents[1] / "configs"
IDS = ("C036", "C037", "C041")
STATUS = ("pass", "fail", "unknown", "not_applicable")


@pytest.fixture(autouse=True)
def cpu_threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


@pytest.fixture
def specs():
    return [spec for spec in load_specs(CONFIGS / "constraints_bathroom_v1.json") if spec.id in IDS]


@pytest.fixture
def scenario():
    return replace(Scenario(**read_json(CONFIGS / "scenario_bathroom_v1.json")),
                   coordinate_scale_m_per_model_unit=1.0, scale_provenance="assumed")


def compare(specs, scenario, layouts, device="cpu"):
    tensor = TensorConstraintEvaluator(specs).evaluate(packed([layouts], device=device), scenario)
    cpu = ConstraintEvaluator(specs).evaluate(sample_from_rooms(layouts), scenario)
    for column, result in enumerate(cpu.results):
        assert STATUS[int(tensor["status"][0, column])] == result.status
        if result.severity is None:
            assert torch.isnan(tensor["severity"][0, column])
        else:
            assert tensor["severity"][0, column].item() == pytest.approx(result.severity)
    return cpu, tensor


def test_registry_records_pdf_thresholds_separately_from_approved_fixtures():
    all_specs = load_specs(CONFIGS / "constraints_bathroom_v1.json")
    assert len(all_specs) == 14
    assert sum(spec.counts_toward_compliance for spec in all_specs) == 11
    assert all(spec.source_id.startswith("Technical_assessment_TTC.pdf:AppendixA:") for spec in all_specs)
    registry = {spec.id: spec for spec in all_specs}
    assert registry["C036"].thresholds["diameter_m"] == 1.5
    assert registry["C037"].thresholds == {"transfer_width_m": .8, "transfer_length_m": 1.3}
    assert registry["C041"].thresholds == {"door_overlap_m": .25, "basin_overlap_m": .15}
    assert "wc_width_m" not in registry["C037"].thresholds


def test_joint_registry_cannot_silently_omit_fixture_observations(tmp_path):
    import json
    registry = read_json(CONFIGS / "constraints_bathroom_v1.json")
    next(spec for spec in registry["constraints"] if spec["id"] == "C037")["required_observations"].remove("wc_width_m")
    path = tmp_path / "incomplete.json"
    path.write_text(json.dumps(registry))
    with pytest.raises(ValueError, match="all declared fixture/search inputs"):
        load_specs(path)


def test_all_three_checks_use_the_same_completed_room(specs, scenario):
    cpu, tensor = compare(specs, scenario, [(4, rectangle(1, 1)), (4, rectangle(4, 4, x=10))])
    assert all(result.status == "pass" for result in cpu.results)
    assert {result.observed_values["room_index"] for result in cpu.results} == {1}
    assert cpu.results[0].observed_values["witness"] == cpu.results[1].observed_values["witness"]
    assert cpu.results[1].observed_values["witness"] == cpu.results[2].observed_values["witness"]
    assert tensor["bathroom_guidance_penalty"].item() == pytest.approx(0, abs=1e-8)


def test_no_bathroom_fails_and_missing_observations_remain_unknown(specs, scenario):
    cpu, _ = compare(specs, scenario, [(0, rectangle(4, 4))])
    assert all(result.status == "fail" for result in cpu.results)
    for name in ("bathroom_layout", "wc_width_m", "basin_under_clearance", "bathroom_door_width_m"):
        cpu, tensor = compare(specs, replace(scenario, **{name: None}), [(4, rectangle(4, 4))])
        assert all(result.status == "unknown" and name in result.missing_fields for result in cpu.results)
        assert tensor["bathroom_guidance_penalty"].item() == 0


def test_small_bathroom_has_proven_failure_and_distinct_guidance_energy(specs, scenario):
    cpu, tensor = compare(specs, scenario, [(4, rectangle(1, 1))])
    assert cpu.results[0].status == "fail"
    assert tensor["energy"].item() == pytest.approx(tensor["bathroom_guidance_penalty"].item())
    assert tensor["energy"].item() > 0
    assert "bathroom_guidance_penalty" in SCORE_FIELDS
    record = _evaluation_record(tensor, 0, TensorConstraintEvaluator(specs))
    assert record["bathroom_guidance_penalty"] == pytest.approx(tensor["energy"].item())


def test_inapplicable_does_not_add_bathroom_guidance_penalty(specs, scenario):
    cpu, tensor = compare(specs, replace(scenario, accessible_dwelling=False), [(4, rectangle(1, 1))])
    assert all(result.status == "not_applicable" for result in cpu.results)
    assert tensor["energy"].item() == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_matches_tensor_cpu_and_final_audit(specs, scenario):
    layouts = [(4, rectangle(4, 4))]
    _, gpu = compare(specs, scenario, layouts, device="cuda")
    host = TensorConstraintEvaluator(specs).evaluate(packed([layouts]), scenario)
    assert gpu["status"].is_cuda
    assert torch.equal(gpu["status"].cpu(), host["status"])
    assert torch.allclose(gpu["energy"].cpu(), host["energy"])
