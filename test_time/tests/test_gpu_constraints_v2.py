"""V2 habitable-area semantics shared by tensor rewards and CPU final audits."""

from dataclasses import replace
from pathlib import Path

import pytest
import torch

from test_time.constraints import load_specs
from test_time.gpu_constraints import FAIL, NOT_APPLICABLE, PASS, UNKNOWN, TensorConstraintEvaluator
from test_time.policy import ConstraintEvaluator
from test_time.tests.test_gpu_constraints import packed, rectangle
from test_time.types import FloorplanSample, Room, Scenario


CONFIG = Path(__file__).parents[1] / "configs" / "constraints_v2.json"
STATUS_NAMES = ("pass", "fail", "unknown", "not_applicable")
AREA_IDS = ("C076", "C077", "C078")
SUPPORT_IDS = ("C034", "C079", "C080")
ROOMS = [(0, rectangle(4, 4)), (3, rectangle(2, 2, x=6)),
         (4, rectangle(2, 2, x=10)), (2, rectangle(10, 10, x=20)),
         (5, rectangle(10, 10, x=40)), (6, rectangle(10, 10, x=60))]


def sample_from_rooms(rooms):
    sample = FloorplanSample()
    for label, polygon in rooms:
        first = len(sample.nodes)
        sample.nodes.extend(polygon)
        sample.edges.extend([[first + index, first + (index + 1) % len(polygon)]
                             for index in range(len(polygon))])
        sample.rooms.append(Room(label, [*polygon, polygon[0]]))
    return sample


@pytest.fixture(autouse=True)
def cpu_threads():
    count = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(count)


@pytest.fixture
def specs():
    return load_specs(CONFIG)


@pytest.fixture
def scenario():
    return Scenario(coordinate_scale_m_per_model_unit=1, scale_provenance="assumed")


def assert_cpu_agreement(specs, scenario, rooms=ROOMS):
    evaluator = TensorConstraintEvaluator(specs, max_positions=128)
    tensor = evaluator.evaluate(packed([rooms]), scenario)
    reference = ConstraintEvaluator(specs).evaluate(sample_from_rooms(rooms), scenario)
    assert tensor["structurally_valid"].tolist() == [reference.structurally_valid] == [True]
    assert [STATUS_NAMES[int(status)] for status in tensor["status"][0]] == [item.status for item in reference.results]
    assert tensor["energy"].tolist() == pytest.approx([reference.energy])
    assert tensor["applicable_count"].tolist() == [reference.applicable_count] == [6]
    assert tensor["unknown_count"].tolist() == [reference.unknown_count]
    assert tensor["measurable_count"].tolist() == [reference.measurable_count]
    return evaluator, tensor, reference


@pytest.mark.parametrize("height,expected_area", [(1.79, 0), (1.8, 24), (2.2, 24), (2.5, 24)])
def test_height_boundary_and_central_classification_match_cpu(specs, scenario, height, expected_area):
    evaluator, result, reference = assert_cpu_agreement(specs, replace(scenario, ceiling_height_m=height))
    assert result["habitable_area_m2"].tolist() == [expected_area]
    assert result["room_areas_m2"].tolist() == [[16, 4, 4, 100, 100, 100]]
    assert result["status"][0, evaluator.ids.index("C076")] == (FAIL if height < 1.8 else PASS)
    for constraint_id in SUPPORT_IDS:
        column = evaluator.ids.index(constraint_id)
        assert result["status"][0, column] == NOT_APPLICABLE
        assert not result["applicable"][0, column]
        assert not result["measurable"][0, column]
        assert torch.isnan(result["severity"][0, column])
    assert len(specs) == 9 and all(spec.enabled for spec in specs)
    assert len(reference.results) == len(evaluator.ids) == 9


@pytest.mark.parametrize("missing", ["height", "area_mode", "scale"])
def test_missing_area_metadata_is_unknown_and_never_creates_support_passes(specs, scenario, missing):
    changes = {"height": {"ceiling_height_m": None},
               "area_mode": {"habitable_area_mode": "unknown"},
               "scale": {"coordinate_scale_m_per_model_unit": None, "scale_provenance": "missing"}}
    evaluator, result, _ = assert_cpu_agreement(specs, replace(scenario, **changes[missing]))
    assert all(result["status"][0, evaluator.ids.index(id)] == UNKNOWN for id in AREA_IDS)
    assert all(result["status"][0, evaluator.ids.index(id)] == NOT_APPLICABLE for id in SUPPORT_IDS)


def test_v2_label_table_is_used_once_for_all_area_checks(specs, scenario):
    # Override the supporting table alone: kitchen/bath area must no longer
    # count even though those labels were included by legacy C077/C078.
    edited = [replace(spec, thresholds={**spec.thresholds, "included_room_labels": [0],
                                       "excluded_room_labels": [1, 2, 3, 4, 5, 6]})
              if spec.id == "C080" else spec for spec in specs]
    _, result, _ = assert_cpu_agreement(edited, scenario)
    assert result["habitable_area_m2"].tolist() == [16]


@pytest.mark.parametrize("area,height,expected", [(9, 2.2, PASS), (8, 2.5, PASS),
                                                 (9, 2.0, FAIL), (10, 2.0, PASS)])
def test_v2_main_room_keeps_exact_area_and_height_or_volume_logic(specs, scenario, area, height, expected):
    evaluator, result, _ = assert_cpu_agreement(specs, replace(scenario, ceiling_height_m=height),
                                               [(0, rectangle(area, 1))])
    assert result["status"][0, evaluator.ids.index("C076")] == expected


def test_disabled_area_definition_keeps_dependent_scores_unknown(specs, scenario):
    edited = [replace(spec, enabled=False) if spec.id == "C079" else spec for spec in specs]
    evaluator, result, _ = assert_cpu_agreement(edited, scenario)
    assert "C079" not in evaluator.ids
    assert all(result["status"][0, evaluator.ids.index(id)] == UNKNOWN for id in AREA_IDS)
    assert torch.isnan(result["habitable_area_m2"]).all()
    assert evaluator.metadata["habitable_area_policy"]["missing_fields"] == ["supporting_area_definitions"]


def test_low_headroom_and_unknown_area_model_do_not_change_bed_or_bath_geometry(specs, scenario):
    evaluator = TensorConstraintEvaluator(specs, max_positions=256)
    rooms = packed([[(1, rectangle(4, 4)), (4, rectangle(2, 2, x=5))]])
    ordinary = evaluator.evaluate(rooms, scenario)
    low = evaluator.evaluate(rooms, replace(scenario, ceiling_height_m=1.79))
    unknown = evaluator.evaluate(rooms, replace(scenario, habitable_area_mode="unknown"))
    columns = [evaluator.ids.index(id) for id in ("C033", "C035", "C036")]
    assert ordinary["status"][:, columns].eq(PASS).all()
    assert torch.equal(ordinary["status"][:, columns], low["status"][:, columns])
    assert torch.equal(ordinary["status"][:, columns], unknown["status"][:, columns])
    assert torch.equal(ordinary["room_areas_m2"], low["room_areas_m2"])
    assert low["habitable_area_m2"].tolist() == [0]


def test_legacy_v1_surface_does_not_acquire_height_or_area_model_requirements(scenario):
    specs = load_specs(Path(__file__).parents[1] / "configs" / "constraints_v1.json")
    evaluator = TensorConstraintEvaluator(specs, max_positions=64)
    rooms = packed([[(0, rectangle(28, 1))]])
    for height in (1.0, None):
        result = evaluator.evaluate(rooms, replace(scenario, ceiling_height_m=height,
                                                  habitable_area_mode="unknown"))
        assert result["status"][0, evaluator.ids.index("C077")] == PASS
        assert "habitable_area_m2" not in result
        assert not evaluator.metadata["habitable_area_policy"]["enabled"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_v2_cuda_and_cpu_tensor_paths_agree(specs, scenario):
    evaluator = TensorConstraintEvaluator(specs, max_positions=128)
    for height in (1.79, 1.8):
        current = replace(scenario, ceiling_height_m=height)
        cpu = evaluator.evaluate(packed([ROOMS]), current)
        cuda = evaluator.evaluate(packed([ROOMS], device="cuda"), current)
        for key in ("status", "energy", "habitable_area_m2", "applicable_count"):
            torch.testing.assert_close(cuda[key].cpu(), cpu[key])
