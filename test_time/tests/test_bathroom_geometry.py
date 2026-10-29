"""Geometric witnesses and conservative misses for the shared bathroom search."""

from copy import deepcopy
from dataclasses import replace
import math

import pytest
from shapely import affinity
from shapely.geometry import Polygon, box
import torch

from test_time.bathroom_geometry import (_audit_witness, _rectangle_containment,
    _sector_containment, _sector_distance, _sector_rectangle_depth,
    cpu_search_bathroom, search_bathrooms, serialized_witness)
from test_time.types import Scenario


LIMITS = dict(diameter_m=1.5, transfer_width_m=.8, transfer_length_m=1.3,
              door_overlap_m=.25, basin_overlap_m=.15)


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def scenario():
    return Scenario(bathroom_layout="combined_wc_basin_v1", wc_width_m=.4, wc_depth_m=.7,
                    basin_width_m=.6, basin_depth_m=.5, basin_under_clearance=True,
                    bathroom_door_width_m=.8, bathroom_search_budget=432, bathroom_circle_positions=49)


def packed_polygons(polygons, dtype=torch.float64, device="cpu"):
    points = [list(p.exterior.coords)[:-1] for p in polygons]
    vertices = max(map(len, points))
    starts = torch.zeros((len(points), vertices, 2), dtype=dtype, device=device)
    ends = starts.clone()
    masks = torch.zeros((len(points), vertices), dtype=torch.bool, device=device)
    for i, xy in enumerate(points):
        starts[i, :len(xy)] = torch.tensor(xy, dtype=dtype, device=device)
        ends[i, :len(xy)] = starts[i, :len(xy)].roll(-1, 0)
        masks[i, :len(xy)] = True
    lows = torch.tensor([p.bounds[:2] for p in polygons], dtype=dtype, device=device)
    highs = torch.tensor([p.bounds[2:] for p in polygons], dtype=dtype, device=device)
    areas = torch.tensor([p.area for p in polygons], dtype=dtype, device=device)
    return starts, ends, masks, lows, highs, areas


def test_shared_witness_is_contained_and_audited(scenario):
    room = box(0, 0, 3, 3)
    result = cpu_search_bathroom(room, scenario, LIMITS)
    assert set(result["status_by_id"].values()) == {"pass"}
    assert result["guidance_penalty"] == pytest.approx(0.)
    assert result["audit_rejected_ids"] == []
    witness = result["witness"]
    assert all(_audit_witness(room, witness, LIMITS).values())
    wc = Polygon(witness["wc_polygon_m"])
    transfer = Polygon(witness["transfer_polygon_m"])
    assert wc.distance(transfer) == pytest.approx(0.)
    assert wc.intersection(transfer).area == pytest.approx(0.)
    assert transfer.area == pytest.approx(.8 * 1.3)
    assert result["search_budget"] == 432
    assert result["circle_positions"] == 49


def test_necessary_bounds_fail_but_finite_misses_stay_unknown(scenario):
    small = cpu_search_bathroom(box(0, 0, 1, 1), scenario, LIMITS)
    unresolved = cpu_search_bathroom(box(0, 0, 1.758, 1.758), scenario, LIMITS)
    assert set(small["status_by_id"].values()) == {"fail"}
    assert all(value > 0 for value in small["severity_by_id"].values())
    assert set(unresolved["status_by_id"].values()) == {"unknown"}
    assert all(value is None for value in unresolved["severity_by_id"].values())
    assert unresolved["guidance_penalty"] > 0
    assert small["bounds"]["room_area_m2"] == 1


def test_batch_and_rotations_keep_shared_feasibility(scenario):
    shapes = [box(0, 0, 1.7, 2.6), affinity.rotate(box(0, 0, 1.7, 2.6), 90, origin=(0, 0)),
              box(4, 2, 7, 5), box(0, 0, 1, 1)]
    result = search_bathrooms(*packed_polygons(shapes), scenario, LIMITS)
    assert result["status"].tolist() == [[0, 0, 0], [0, 0, 0], [0, 0, 0], [1, 1, 1]]
    for index, shape in enumerate(shapes[:3]):
        assert all(_audit_witness(shape, serialized_witness(result, index), LIMITS).values())


def test_concave_intrusion_cannot_pass_on_corners_alone():
    room = Polygon([(0, 0), (4, 0), (4, 4), (2.1, 4), (2.1, 1.8), (1.9, 1.8), (1.9, 4), (0, 4)])
    starts, ends, masks, *_ = packed_polygons([room])
    rectangle = torch.tensor([[.5, .5, 3.5, 3.5]], dtype=torch.float64)
    fit, penalty = _rectangle_containment(rectangle, starts, ends, masks)
    assert not fit[0]
    assert penalty[0] > 0


def test_exact_door_sector_allows_empty_bbox_corner_and_rejects_true_intrusion():
    hinge = torch.zeros((1, 2), dtype=torch.float64)
    u = torch.tensor([[1., 0.]], dtype=torch.float64)
    v = torch.tensor([[0., 1.]], dtype=torch.float64)
    outside_arc = torch.tensor([[.7, .7, .8, .8]], dtype=torch.float64)
    inside_arc = torch.tensor([[.2, .2, .3, .3]], dtype=torch.float64)
    assert _sector_rectangle_depth(outside_arc, hinge, u, v, .8)[0] == 0
    assert _sector_rectangle_depth(inside_arc, hinge, u, v, .8)[0] > 0
    points = torch.tensor([[[1.3, 0.], [1.299, 0.]]], dtype=torch.float64)
    depths = .75 - _sector_distance(points, hinge, u, v, .8)
    assert depths[0, 0] == pytest.approx(.25)
    assert depths[0, 1] > .25


def test_sector_containment_detects_concave_room_boundary():
    # A notch enters the quarter disk although its hinge and arc endpoints fit.
    room = Polygon([(0, 0), (2, 0), (2, .2), (.4, .2), (.4, .3), (2, .3), (2, 2), (0, 2)])
    starts, ends, masks, *_ = packed_polygons([room])
    fit, cost = _sector_containment(torch.zeros((1, 2), dtype=torch.float64),
        torch.tensor([[1., 0.]], dtype=torch.float64), torch.tensor([[0., 1.]], dtype=torch.float64),
        .8, starts, ends, masks)
    assert not fit[0]
    assert cost[0] > 0


def test_c041_rejects_two_individually_small_overlaps(scenario):
    room = box(0, 0, 2, 2)
    witness = cpu_search_bathroom(room, scenario, LIMITS)["witness"]
    assert witness["template_index"] == 0
    witness["circle_center_m"] = [1.1, 1.15]
    witness["basin_overlap_m"] = 1.1 + math.sqrt(.75**2 - .55**2) - 1.5
    audited = _audit_witness(room, witness, LIMITS)
    assert audited["C036"] and audited["C037"]
    assert 0 < witness["basin_overlap_m"] < .15
    assert not audited["C041"]


def test_under_basin_overlap_requires_declared_clearance(scenario):
    room = box(0, 0, 1.7, 2.6)
    allowed = cpu_search_bathroom(room, scenario, LIMITS)
    assert allowed["status_by_id"]["C041"] == "pass"
    assert 0 < allowed["witness"]["basin_overlap_m"] < .15
    witness = deepcopy(allowed["witness"])
    witness["basin_under_clearance"] = False
    assert not _audit_witness(room, witness, LIMITS)["C041"]
    blocked = cpu_search_bathroom(room, replace(scenario, basin_under_clearance=False), LIMITS)
    if blocked["status_by_id"]["C041"] == "pass":
        assert blocked["witness"]["basin_overlap_m"] <= 1e-7
    assert blocked["guidance_penalty"] >= allowed["guidance_penalty"]


def test_empty_batch_and_single_center_budget(scenario):
    args = packed_polygons([box(0, 0, 3, 3)])
    empty = search_bathrooms(*(arg[:0] for arg in args), scenario, LIMITS)
    assert empty["status"].shape == (0, 3)
    limited = replace(scenario, bathroom_search_budget=1, bathroom_circle_positions=1)
    result = search_bathrooms(*args, limited, LIMITS)
    assert result["search_budget"] == result["circle_positions"] == 1
    assert result["witness"]["circle_center"].tolist() == [[1.5, 1.5]]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_matches_cpu_and_keeps_outputs_on_device(scenario):
    rooms = [box(0, 0, 3, 3), box(0, 0, 1.758, 1.758), box(0, 0, 1, 1)]
    cpu = search_bathrooms(*packed_polygons(rooms), scenario, LIMITS)
    gpu = search_bathrooms(*packed_polygons(rooms, device="cuda"), scenario, LIMITS)
    assert gpu["status"].is_cuda and gpu["guidance_penalty"].is_cuda
    assert gpu["status"].cpu().tolist() == cpu["status"].tolist()
    torch.testing.assert_close(gpu["guidance_penalty"].cpu(), cpu["guidance_penalty"])
