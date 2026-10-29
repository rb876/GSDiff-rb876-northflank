"""The native 256-pixel canvas decodes to 512 drawing pixels, not 256."""

from pathlib import Path

import pytest

from test_time.constraints import CheckContext, check, load_specs
from test_time.gpu_constraints import TensorConstraintEvaluator
from test_time.tests.test_gpu_constraints import packed, rectangle
from test_time.types import FloorplanSample, Room, Scenario, read_json


CONFIGS = Path(__file__).parents[1] / "configs"


def test_documented_canvas_has_324_square_metres_in_both_checkers():
    scenario = Scenario(**read_json(CONFIGS / "scenario_rplan18.json"))
    specs = load_specs(CONFIGS / "constraints_v2.json")
    # Half the drawing width represents 9 m, so a 256x256 room has 81 m².
    rooms = [(0, rectangle(256, 256))]
    sample = FloorplanSample(rooms=[Room(0, [*rooms[0][1], rooms[0][1][0]])])
    context = CheckContext(sample, scenario, specs)
    area_spec = next(spec for spec in specs if spec.id == "C077")
    cpu = check(sample, scenario, area_spec, context)
    tensor = TensorConstraintEvaluator(specs).evaluate(packed([rooms]), scenario)
    assert cpu.observed_values["estimated_habitable_area_m2"] == pytest.approx(81)
    assert tensor["habitable_area_m2"].item() == pytest.approx(81)
    assert "paper_derived_metric_scale" in cpu.assumption_flags
    assert (scenario.coordinate_scale_m_per_model_unit * 512) ** 2 == 324


def test_new_experiments_share_scale_and_old_configuration_is_preserved():
    normal = read_json(CONFIGS / "experiment_rplan18.json")
    extensions = read_json(CONFIGS / "extensions_rplan18.json")
    assert normal["scenario"] == extensions["scenario"] == "scenario_rplan18.json"
    assert normal["constraints"] == extensions["constraints"]
    assert normal["final_checker"] == "cpu_bedroom_audit_v2"
    assert read_json(CONFIGS / "scenario_v2.json")["coordinate_scale_m_per_model_unit"] == .025
