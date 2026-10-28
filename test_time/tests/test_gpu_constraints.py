from dataclasses import replace
import inspect

import pytest
import torch

from test_time.constraints import load_specs
from test_time.gpu_constraints import (
    PASS, FAIL, UNKNOWN, NOT_APPLICABLE, TensorConstraintEvaluator,
    _edges, _rectangles_fit,
)
from test_time.types import Scenario


def rectangle(width, height, x=0, y=0):
    return [[x, y], [x + width, y], [x + width, y + height], [x, y + height]]


def packed(batches, valid=None, device="cpu"):
    room_slots = max(len(rooms) for rooms in batches)
    vertex_slots = max(len(points) for rooms in batches for _, points in rooms)
    polygons = torch.zeros((len(batches), room_slots, vertex_slots, 2), dtype=torch.float64, device=device)
    vertex_mask = torch.zeros(polygons.shape[:-1], dtype=torch.bool, device=device)
    room_mask = torch.zeros(polygons.shape[:2], dtype=torch.bool, device=device)
    labels = torch.full(polygons.shape[:2], -1, dtype=torch.long, device=device)
    for batch, rooms in enumerate(batches):
        for index, (label, points) in enumerate(rooms):
            polygons[batch, index, :len(points)] = torch.tensor(points, dtype=torch.float64, device=device)
            vertex_mask[batch, index, :len(points)] = True
            room_mask[batch, index] = True
            labels[batch, index] = label
    return dict(polygons=polygons, vertex_mask=vertex_mask, room_mask=room_mask, labels=labels,
                structurally_valid=torch.tensor(valid or [True] * len(batches), device=device),
                diagnostics=torch.zeros(len(batches), dtype=torch.long, device=device))


@pytest.fixture(autouse=True)
def cpu_threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


@pytest.fixture
def scenario():
    return Scenario(coordinate_scale_m_per_model_unit=1, scale_provenance="assumed")


@pytest.fixture
def evaluator():
    return TensorConstraintEvaluator(load_specs(), max_positions=256)


def states(evaluator, result, constraint):
    return result["status"][:, evaluator.ids.index(constraint)].tolist()


def test_batched_shared_bed_and_incompatible_independent_witnesses(evaluator, scenario):
    rooms = packed([[(1, rectangle(4, 4))], [(1, rectangle(3.2, 3.1))], [(1, rectangle(1, 1))]])
    result = evaluator.evaluate(rooms, scenario)
    assert states(evaluator, result, "C033") == [PASS, PASS, FAIL]
    assert states(evaluator, result, "C034") == [PASS, PASS, FAIL]
    assert states(evaluator, result, "C035") == [PASS, UNKNOWN, FAIL]
    assert result["geometry_joint"][0, 0]
    assert not result["geometry_joint"][1, 0]
    assert result["applicable_count"].tolist() == [6, 6, 6]


def test_bathroom_circle_witness_and_sound_bound(evaluator, scenario):
    rooms = packed([[(4, rectangle(2, 2))], [(4, rectangle(1.4, 2))]])
    result = evaluator.evaluate(rooms, scenario)
    assert states(evaluator, result, "C036") == [PASS, FAIL]
    column = evaluator.ids.index("C036")
    assert result["severity"][1, column] == pytest.approx((0.75 - 0.7) / 0.75)


@pytest.mark.parametrize("area,height,expected", [(9, 2.2, PASS), (8, 2.5, PASS), (9, 2.0, FAIL), (10, 2.0, PASS)])
def test_c076_or_branches(evaluator, scenario, area, height, expected):
    result = evaluator.evaluate(packed([[(0, rectangle(area, 1))]]), replace(scenario, ceiling_height_m=height))
    assert states(evaluator, result, "C076") == [expected]


@pytest.mark.parametrize("occupants,area,volume", [(1, 14, 33), (4, 56, 132), (5, 66, 155)])
def test_occupancy_breakpoints(evaluator, scenario, occupants, area, volume):
    scenario = replace(scenario, intended_occupants=occupants, ceiling_height_m=1)
    result = evaluator.evaluate(packed([[(0, rectangle(area, 1))], [(0, rectangle(area - 0.1, 1))]]), scenario)
    assert states(evaluator, result, "C077") == [PASS, FAIL]
    result = evaluator.evaluate(packed([[(0, rectangle(volume, 1))], [(0, rectangle(volume - 0.1, 1))]]), scenario)
    assert states(evaluator, result, "C078") == [PASS, FAIL]


def test_missing_metadata_invalid_geometry_and_not_applicable(evaluator, scenario):
    rooms = packed([[(0, rectangle(4, 4))]])
    result = evaluator.evaluate(rooms, Scenario())
    assert result["status"].eq(UNKNOWN).all()
    assert result["severity"].isnan().all()
    assert result["unknown_count"].tolist() == [6]
    no_height = evaluator.evaluate(rooms, replace(scenario, ceiling_height_m=None))
    assert states(evaluator, no_height, "C076") == states(evaluator, no_height, "C078") == [UNKNOWN]
    no_occupants = evaluator.evaluate(rooms, replace(scenario, intended_occupants=None))
    assert states(evaluator, no_occupants, "C077") == [UNKNOWN]
    invalid = evaluator.evaluate(packed([[(0, rectangle(4, 4))]], valid=[False]), scenario)
    assert invalid["status"].eq(UNKNOWN).all()
    assert invalid["energy"].tolist() == [100.0]
    nonapplicable = evaluator.evaluate(rooms, replace(scenario, accessible_dwelling=False))
    assert states(evaluator, nonapplicable, "C033") == [NOT_APPLICABLE]
    assert nonapplicable["applicable_count"].tolist() == [3]


def test_entire_rectangle_containment_rejects_concave_intrusion():
    # All four rectangle corners lie inside the U-shaped polygon; its notch
    # enters the rectangle interior and must still cause rejection.
    polygon = torch.tensor([[[0, 0], [4, 0], [4, 4], [3, 4], [3, 2], [1, 2], [1, 4], [0, 4]]], dtype=torch.float64)
    starts, ends, mask = _edges(polygon, torch.ones((1, 8), dtype=torch.bool))
    assert not _rectangles_fit(torch.tensor([[0.5, 0.5]]), torch.tensor([[3.5, 3.5]]), starts, ends, mask)[0]
    assert _rectangles_fit(torch.tensor([[0.1, 0.1]]), torch.tensor([[3.9, 1.9]]), starts, ends, mask)[0]


def test_holes_nonfinite_and_overlap_invalid_gate(evaluator, scenario):
    rooms = packed([[(0, rectangle(4, 4))]])
    rooms["has_holes"] = torch.tensor([[True]])
    assert evaluator.evaluate(rooms, scenario)["status"].eq(UNKNOWN).all()
    rooms = packed([[(0, rectangle(4, 4))]])
    rooms["polygons"][0, 0, 1, 0] = torch.nan
    assert evaluator.evaluate(rooms, scenario)["status"].eq(UNKNOWN).all()
    rooms = packed([[(0, rectangle(4, 4))]])
    rooms["polygons"] *= 1e200
    assert evaluator.evaluate(rooms, scenario)["status"].eq(UNKNOWN).all()


def test_capped_grid_miss_is_unknown_not_failure(scenario):
    # A 2 x 2 m lobe contains a turning circle around (1, 2.2), but the
    # deliberately tiny search budget visits only boundary/midpoint/grid-prefix
    # positions elsewhere. Its failure to find a witness is not a failed rule.
    lobe = [[0, 0], [6, 0], [6, 1], [1, 1], [1, 1.2], [2, 1.2],
            [2, 3.2], [1, 3.2], [1, 6], [0, 6]]
    evaluator = TensorConstraintEvaluator(load_specs(), max_positions=36)
    result = evaluator.evaluate(packed([[(4, lobe)]]), scenario)
    assert states(evaluator, result, "C036") == [UNKNOWN]
    assert torch.isnan(result["severity"][0, evaluator.ids.index("C036")])


def test_runtime_has_no_host_tensor_conversions():
    import test_time.gpu_constraints as module
    source = inspect.getsource(module)
    for forbidden in (".cpu(", ".numpy(", ".item(", ".tolist("):
        assert forbidden not in source


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_matches_same_tensor_backend_on_cpu(evaluator, scenario):
    rooms = [[(1, rectangle(4, 4)), (4, rectangle(1.4, 2, x=5))]]
    cpu = evaluator.evaluate(packed(rooms), scenario)
    gpu = evaluator.evaluate(packed(rooms, device="cuda"), scenario)
    assert torch.equal(cpu["status"], gpu["status"].cpu())
    assert torch.allclose(cpu["energy"], gpu["energy"].cpu())
