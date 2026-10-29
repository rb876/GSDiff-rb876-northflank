"""Four custom design requirements agree between tensor rewards and CPU checks."""

from dataclasses import replace
import json
from pathlib import Path

import pytest
import torch

from test_time.constraints import CheckContext, check, load_specs
from test_time.gpu_constraints import TensorConstraintEvaluator
from test_time.policy import ConstraintEvaluator
from test_time.tests.test_gpu_constraints import packed, rectangle
from test_time.tests.test_gpu_constraints_v2 import sample_from_rooms
from test_time.types import Scenario


CONFIG = Path(__file__).parents[1] / "configs" / "constraints_design_v1.json"
DESIGN_IDS = ("D001", "D002", "D003", "D004")
STATUS = ("pass", "fail", "unknown", "not_applicable")


@pytest.fixture(autouse=True)
def cpu_threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


@pytest.fixture
def specs():
    # Isolate the new rules so bedroom placement search cannot mask their result.
    return [spec for spec in load_specs(CONFIG) if spec.id in {*DESIGN_IDS, "C079", "C080"}]


@pytest.fixture
def scenario():
    return Scenario(coordinate_scale_m_per_model_unit=1, scale_provenance="verified")


def rooms_with_areas(entries):
    return [(label, rectangle(area, 1, x=index * 1000)) for index, (label, area) in enumerate(entries)]


def compare(specs, scenario, entries):
    rooms = rooms_with_areas(entries)
    tensor_evaluator = TensorConstraintEvaluator(specs)
    tensor = tensor_evaluator.evaluate(packed([rooms]), scenario)
    cpu = ConstraintEvaluator(specs).evaluate(sample_from_rooms(rooms), scenario)
    by_id = {item.id: item for item in cpu.results}
    for identifier in DESIGN_IDS:
        column = tensor_evaluator.ids.index(identifier)
        result = by_id[identifier]
        assert STATUS[int(tensor["status"][0, column])] == result.status
        if result.severity is None:
            assert torch.isnan(tensor["severity"][0, column])
        else:
            assert tensor["severity"][0, column].item() == pytest.approx(result.severity)
    assert tensor["energy"].item() == pytest.approx(cpu.energy)
    return by_id, tensor


def test_registry_separates_ten_scored_rules_from_three_supporting_definitions():
    specs = load_specs(CONFIG)
    assert len(specs) == 13
    assert sum(spec.counts_toward_compliance for spec in specs) == 10
    assert {spec.id for spec in specs if spec.role == "supporting_rule"} == {"C034", "C079", "C080"}
    custom = [spec for spec in specs if spec.id in DESIGN_IDS]
    assert all(spec.group == "design_brief" and spec.claim_level == "custom_design_requirement" for spec in custom)
    assert all("Appendix" not in spec.source_id for spec in custom)


def test_threshold_boundaries_pass_and_no_large_bedroom_compensates_for_small(specs, scenario):
    exact, _ = compare(specs, scenario, [(1, 10), (1, 10), (1, 10), (0, 18), (3, 52)])
    assert all(exact[id].status == "pass" for id in DESIGN_IDS)
    undersized, _ = compare(specs, scenario, [(1, 5), (1, 20), (1, 20), (0, 18)])
    assert undersized["D002"].severity == .5
    assert undersized["D002"].observed_values["worst_room_index"] == 0
    assert undersized["D002"].observed_values["minimum_bedroom_area_m2"] == 5


@pytest.mark.parametrize("count,penalty", [(0, 1), (1, 2 / 3), (2, 1 / 3), (3, 0), (4, 0)])
def test_bedroom_count_and_absence_have_explicit_penalties(specs, scenario, count, penalty):
    results, _ = compare(specs, scenario, [(1, 10)] * count + [(0, 18)])
    assert results["D001"].severity == pytest.approx(penalty)
    assert results["D001"].observed_values["bedroom_count"] == count
    assert results["D002"].severity == (0 if count else 1)


def test_living_rooms_are_not_summed_and_a_missing_living_room_fails(specs, scenario):
    separate, _ = compare(specs, scenario, [(1, 10), (0, 10), (0, 10)])
    assert separate["D003"].severity == pytest.approx(8 / 18)
    assert separate["D003"].observed_values["largest_living_room_area_m2"] == 10
    absent, _ = compare(specs, scenario, [(1, 10)])
    assert absent["D003"].status == "fail" and absent["D003"].severity == 1


@pytest.mark.parametrize("area,status,penalty", [(99, "pass", 0), (100, "pass", 0), (110, "fail", .1), (250, "fail", 1.5)])
def test_maximum_area_has_continuous_uncapped_excess(specs, scenario, area, status, penalty):
    results, _ = compare(specs, scenario, [(0, area), (2, 500), (5, 500), (6, 500)])
    assert results["D004"].status == status
    assert results["D004"].severity == pytest.approx(penalty)
    assert results["D004"].observed_values["estimated_habitable_area_m2"] == area


def test_count_survives_unknown_scale_while_dimensional_rules_remain_unknown(specs, scenario):
    scenario = replace(scenario, coordinate_scale_m_per_model_unit=None, scale_provenance="missing")
    results, _ = compare(specs, scenario, [(1, 10)] * 3 + [(0, 18)])
    assert results["D001"].status == "pass"
    assert all(results[id].status == "unknown" for id in DESIGN_IDS[1:])


@pytest.mark.parametrize("change", [{"ceiling_height_m": None}, {"habitable_area_mode": "unknown"}])
def test_habitable_area_inputs_only_gate_the_area_budget(specs, scenario, change):
    results, _ = compare(specs, replace(scenario, **change), [(1, 10)] * 3 + [(0, 18)])
    assert all(results[id].status == "pass" for id in DESIGN_IDS[:3])
    assert results["D004"].status == "unknown"


def test_low_headroom_exclusion_does_not_redefine_physical_room_areas(specs, scenario):
    results, tensor = compare(specs, replace(scenario, ceiling_height_m=1.7), [(1, 10)] * 3 + [(0, 18)])
    assert all(results[id].status == "pass" for id in DESIGN_IDS)
    assert results["D004"].observed_values["estimated_habitable_area_m2"] == 0
    assert tensor["habitable_area_m2"].item() == 0


def test_unreliable_extracted_geometry_is_never_a_pass(specs, scenario):
    rooms = rooms_with_areas([(1, 10)] * 3 + [(0, 18)])
    sample = sample_from_rooms(rooms)
    sample.diagnostics = ["Ambiguous room extraction"]
    cpu = ConstraintEvaluator(specs).evaluate(sample, scenario)
    tensor = TensorConstraintEvaluator(specs).evaluate(packed([rooms], valid=[False]), scenario)
    assert not cpu.structurally_valid and not tensor["structurally_valid"].item()
    assert all(item.status == "unknown" for item in cpu.results if item.id in DESIGN_IDS)
    assert cpu.energy == tensor["energy"].item() == 100


def test_unavailable_supporting_definitions_make_area_budget_unknown(specs, scenario):
    edited = [replace(spec, enabled=False) if spec.id == "C079" else spec for spec in specs]
    result, _ = compare(edited, scenario, [(0, 18)])
    assert result["D004"].status == "unknown"
    assert "supporting_area_definitions" in result["D004"].missing_fields


def test_area_budget_uses_polygon_union_on_cpu(specs, scenario):
    # Geometry checks reject overlaps globally, but area measurement must not double-count them.
    from test_time.types import FloorplanSample, Room
    polygons = [rectangle(8, 10), rectangle(8, 10, x=2)]
    sample = FloorplanSample(rooms=[Room(0, [*points, points[0]]) for points in polygons])
    context = CheckContext(sample, scenario, specs)
    result = check(sample, scenario, next(spec for spec in specs if spec.id == "D004"), context)
    assert result.observed_values["estimated_habitable_area_m2"] == 100


def test_count_threshold_is_integer_and_design_registry_is_complete(tmp_path):
    config = json.loads(CONFIG.read_text())
    next(spec for spec in config["constraints"] if spec["id"] == "D001")["thresholds"]["minimum_count"] = 2.5
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="minimum_count must be an integer"):
        load_specs(path)
    config = json.loads(CONFIG.read_text())
    config["constraints"] = [spec for spec in config["constraints"] if spec["id"] != "D001"]
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="thirteen implemented IDs"):
        load_specs(path)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_design_rules_cuda_match_tensor_cpu_and_stay_on_device(specs, scenario):
    batches = [rooms_with_areas([(1, 10)] * 3 + [(0, 18)]),
               rooms_with_areas([(1, 5), (0, 120)])]
    evaluator = TensorConstraintEvaluator(specs)
    cpu = evaluator.evaluate(packed(batches), scenario)
    cuda = evaluator.evaluate(packed(batches, device="cuda"), scenario)
    for key in ("status", "severity", "energy", "unknown_count", "applicable_count"):
        assert cuda[key].device.type == "cuda"
        torch.testing.assert_close(cuda[key].cpu(), cpu[key], equal_nan=True)
