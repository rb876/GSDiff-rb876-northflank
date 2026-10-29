"""PDF airflow tables use graph room counts and explicit system parameters."""

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
from test_time.types import Scenario, read_json


CONFIGS = Path(__file__).parents[1] / "configs"
IDS = ("C085", "C086", "C090")
STATUS = ("pass", "fail", "unknown", "not_applicable")
KITCHEN = [75, 90, 105, 120, 135]
BATHROOM = [15, 15, 30, 30, 30]
REDUCED_TOTAL = [35, 60, 75, 90, 105, 120, 135]
REDUCED_KITCHEN = [20, 30, 45, 45, 45, 45, 45]


@pytest.fixture(autouse=True)
def cpu_threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


@pytest.fixture
def specs():
    return [spec for spec in load_specs(CONFIGS / "constraints_ventilation_v1.json") if spec.id in IDS]


@pytest.fixture
def scenario():
    # None metric scale is intentional: room counts/declared flows are sufficient.
    return Scenario(ventilation_mode="occupant_adjustable", kitchen_extraction_m3_h=105,
                    bathroom_extraction_m3_h=30, reduced_total_extraction_m3_h=75,
                    reduced_kitchen_extraction_m3_h=45, bathroom_ventilation_role="bathroom")


def rooms(main_count, kitchens=1, bathrooms=1):
    labels = ([0] + [1] * (main_count - 1) if main_count else []) + [3] * kitchens + [4] * bathrooms
    return [(label, rectangle(4, 4, x=10 * index)) for index, label in enumerate(labels)]


def compare(specs, scenario, main_count, kitchens=1, bathrooms=1, device="cpu"):
    layouts = rooms(main_count, kitchens, bathrooms)
    evaluator = TensorConstraintEvaluator(specs)
    tensor = evaluator.evaluate(packed([layouts], device=device), scenario)
    cpu = ConstraintEvaluator(specs).evaluate(sample_from_rooms(layouts), scenario)
    by_id = {result.id: result for result in cpu.results}
    for index, spec in enumerate(specs):
        result = by_id[spec.id]
        assert STATUS[int(tensor["status"][0, index])] == result.status
        if result.severity is None:
            assert torch.isnan(tensor["severity"][0, index])
        else:
            assert tensor["severity"][0, index].item() == pytest.approx(result.severity)
    assert tensor["energy"].item() == pytest.approx(cpu.energy)
    return by_id, tensor


def test_default_profile_is_nine_pdf_checks_and_three_supporting_rules():
    specs = load_specs(CONFIGS / "constraints_ventilation_v1.json")
    scored = [spec for spec in specs if spec.counts_toward_compliance]
    assert len(specs) == 12 and len(scored) == 9
    assert all(spec.source_id.startswith("Technical_assessment_TTC.pdf:AppendixA:") for spec in specs)
    assert {spec.id for spec in specs if spec.group == "ventilation"} == set(IDS)
    assert not any(spec.id.startswith("D") for spec in specs)
    approved = Scenario(**read_json(CONFIGS / "scenario_ventilation_v1.json"))
    assert (approved.kitchen_extraction_m3_h, approved.bathroom_extraction_m3_h,
            approved.reduced_total_extraction_m3_h, approved.reduced_kitchen_extraction_m3_h) == (105, 30, 75, 45)


@pytest.mark.parametrize("count", range(1, 8))
def test_every_pdf_table_threshold_is_inclusive(specs, scenario, count):
    limits = replace(scenario, kitchen_extraction_m3_h=KITCHEN[min(count, 5)-1],
                     bathroom_extraction_m3_h=BATHROOM[min(count, 5)-1],
                     reduced_total_extraction_m3_h=REDUCED_TOTAL[count-1],
                     reduced_kitchen_extraction_m3_h=REDUCED_KITCHEN[count-1])
    outcomes, _ = compare(specs, limits, count)
    assert all(outcomes[id].status == "pass" and outcomes[id].severity == 0 for id in IDS)
    assert outcomes["C085"].observed_values["required_flow_m3_h"] == KITCHEN[min(count, 5)-1]
    assert outcomes["C086"].observed_values["required_flow_m3_h"] == BATHROOM[min(count, 5)-1]
    assert outcomes["C090"].observed_values["required_total_flow_m3_h"] == REDUCED_TOTAL[count-1]


@pytest.mark.parametrize("count", range(1, 8))
def test_normal_flow_deficits_follow_each_room_count(specs, scenario, count):
    kitchen, bath = KITCHEN[min(count, 5)-1], BATHROOM[min(count, 5)-1]
    below = replace(scenario, kitchen_extraction_m3_h=kitchen-1, bathroom_extraction_m3_h=bath-1)
    outcomes, _ = compare(specs, below, count)
    assert outcomes["C085"].status == outcomes["C086"].status == "fail"
    assert outcomes["C085"].severity == pytest.approx(1/kitchen)
    assert outcomes["C086"].severity == pytest.approx(1/bath)


@pytest.mark.parametrize("total,kitchen,penalty", [(75, 44, 1/45), (74, 45, 1/75), (60, 30, 1/3), (0, 0, 1)])
def test_reduced_check_requires_both_flows_and_uses_worst_deficit(specs, scenario, total, kitchen, penalty):
    outcomes, _ = compare(specs, replace(scenario, reduced_total_extraction_m3_h=total,
                                        reduced_kitchen_extraction_m3_h=kitchen), 3)
    assert outcomes["C090"].status == "fail"
    assert outcomes["C090"].severity == pytest.approx(penalty)


def test_zero_normal_flow_is_a_measured_failure(specs, scenario):
    outcomes, _ = compare(specs, replace(scenario, kitchen_extraction_m3_h=0, bathroom_extraction_m3_h=0), 3)
    assert outcomes["C085"].severity == outcomes["C086"].severity == 1


@pytest.mark.parametrize("field,affected", [
    ("kitchen_extraction_m3_h", {"C085"}), ("bathroom_extraction_m3_h", {"C086"}),
    ("bathroom_ventilation_role", {"C086"}), ("ventilation_mode", {"C090"}),
    ("reduced_total_extraction_m3_h", {"C090"}), ("reduced_kitchen_extraction_m3_h", {"C090"}),
])
def test_missing_declared_inputs_are_unknown_not_free_passes(specs, scenario, field, affected):
    outcomes, _ = compare(specs, replace(scenario, **{field: None}), 3)
    assert {id for id in IDS if outcomes[id].status == "unknown"} == affected
    assert all(outcomes[id].severity is None for id in affected)


@pytest.mark.parametrize("mode", ["automatic", "fixed"])
def test_reduced_rule_is_not_applicable_in_other_known_modes(specs, scenario, mode):
    outcomes, _ = compare(specs, replace(scenario, ventilation_mode=mode, accessible_dwelling=False), 3)
    assert outcomes["C090"].status == "not_applicable"
    assert outcomes["C085"].status == outcomes["C086"].status == "pass"


@pytest.mark.parametrize("role", ["other_shower", "mixed", "unknown"])
def test_bathroom_role_must_be_declared_not_inferred_from_label(specs, scenario, role):
    outcomes, _ = compare(specs, replace(scenario, bathroom_ventilation_role=role), 3)
    assert outcomes["C086"].status == "unknown"


def test_missing_service_rooms_and_inconsistent_flow_mapping_remain_unknown(specs, scenario):
    no_kitchen, _ = compare(specs, scenario, 3, kitchens=0)
    assert no_kitchen["C085"].status == no_kitchen["C090"].status == "unknown"
    no_bath, _ = compare(specs, scenario, 3, bathrooms=0)
    assert no_bath["C086"].status == "unknown"
    multiple, _ = compare(specs, scenario, 3, kitchens=2, bathrooms=2)
    assert multiple["C085"].status == multiple["C086"].status == "pass"
    assert multiple["C090"].status == "unknown"  # 75 total cannot deliver 2 x 45 kitchen flow.
    consistent, _ = compare(specs, replace(scenario, reduced_total_extraction_m3_h=90), 3, kitchens=2)
    assert consistent["C090"].status == "pass"


def test_pdf_table_ranges_are_not_extrapolated(specs, scenario):
    zero, _ = compare(specs, scenario, 0)
    assert all(zero[id].status == "unknown" for id in IDS)
    eight, _ = compare(specs, replace(scenario, kitchen_extraction_m3_h=135), 8)
    assert eight["C085"].status == eight["C086"].status == "pass"
    assert eight["C090"].status == "unknown"


def test_declared_system_is_fixed_and_changes_feasibility_with_layout(specs, scenario):
    good, _ = compare(specs, scenario, 3)
    crowded, _ = compare(specs, scenario, 4)
    assert all(good[id].status == "pass" for id in IDS)
    assert crowded["C085"].severity == pytest.approx(15/120)
    assert crowded["C090"].severity == pytest.approx(15/90)
    assert crowded["C086"].status == "pass"
    assert scenario.kitchen_extraction_m3_h == 105  # Never fitted to each layout.


def test_batched_counts_ignore_padding_and_keep_table_ranges_per_design(specs, scenario):
    evaluator = TensorConstraintEvaluator(specs)
    result = evaluator.evaluate(packed([rooms(count) for count in (1, 3, 4, 8)]), scenario)
    assert result["status"].tolist() == [[0, 0, 0], [0, 0, 0], [1, 0, 1], [1, 0, 2]]
    torch.testing.assert_close(result["severity"][:3],
                               torch.tensor([[0, 0, 0], [0, 0, 0], [15/120, 0, 15/90]], dtype=torch.float64))


def test_nonresidential_is_inapplicable_and_invalid_geometry_is_unresolved(specs, scenario):
    outcomes, _ = compare(specs, replace(scenario, building_use="office"), 3)
    assert all(outcomes[id].status == "not_applicable" for id in IDS)
    layouts = rooms(3)
    sample = sample_from_rooms(layouts)
    sample.diagnostics = ["invalid room topology"]
    context = CheckContext(sample, scenario, specs)
    assert all(check(sample, scenario, spec, context).status == "unknown" for spec in specs)
    tensor = TensorConstraintEvaluator(specs).evaluate(packed([layouts], valid=[False]), scenario)
    assert tensor["status"].tolist() == [[2, 2, 2]]


@pytest.mark.parametrize("value", [-1, float("nan"), float("inf"), True, "105"])
def test_scenario_rejects_invalid_flow_values(value):
    with pytest.raises(ValueError, match="nonnegative finite"):
        Scenario(kitchen_extraction_m3_h=value)


def test_registry_rejects_wrong_table_length(tmp_path):
    data = read_json(CONFIGS / "constraints_ventilation_v1.json")
    next(spec for spec in data["constraints"] if spec["id"] == "C090")["thresholds"]["total_flows_m3_h"] = [35]
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="7 positive finite"):
        load_specs(path)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_scoring_matches_cpu_reference(specs, scenario):
    compare(specs, scenario, 4, device="cuda")
