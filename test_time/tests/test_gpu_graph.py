"""Tensor room extraction against synthetic geometry and saved upstream output."""

import ast
import copy
import json
import math
from pathlib import Path

import networkx as nx
import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from test_time.gpu_graph import (DUPLICATE_XY, EDGE_CAPACITY, INVALID_SEMANTICS,
                                 NESTED_ROOMS, NONFINITE, NO_ROOMS, SELF_LOOP,
                                 WALL_INTERSECTION, extract_rooms)


ROOT = Path(__file__).resolve().parents[2]
SQUARE = [(0, 0), (4, 0), (4, 3), (0, 3)]
SQUARE_EDGES = [(0, 1), (1, 2), (2, 3), (0, 3)]


@pytest.fixture(autouse=True)
def small_tensor_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def graph(points, edges, slots=None, label=0):
    slots = len(points) if slots is None else slots
    xy = torch.zeros((1, slots, 2), dtype=torch.float64)
    xy[0, :len(points)] = torch.tensor(points, dtype=xy.dtype)
    semantics = torch.zeros((1, slots, 7), dtype=torch.long)
    semantics[0, :len(points), label] = 1
    mask = torch.arange(slots)[None] < len(points)
    adjacency = torch.zeros((1, slots, slots), dtype=torch.bool)
    for first, second in edges:
        adjacency[0, min(first, second), max(first, second)] = True
    return dict(xy=xy, semantics=semantics, node_mask=mask, adjacency=adjacency)


def canonical(polygons, labels):
    result = []
    for polygon, label in zip(polygons, labels):
        vertices = [tuple(point[:2]) for point in polygon]
        if vertices[-1] == vertices[0]:
            vertices.pop()
        rotations = [tuple(vertices[index:] + vertices[:index]) for index in range(len(vertices))]
        backward = list(reversed(vertices))
        rotations += [tuple(backward[index:] + backward[:index]) for index in range(len(backward))]
        result.append((int(label), min(rotations)))
    return sorted(result)


def tensor_rooms(result, index=0):
    polygons, labels = [], []
    for room in torch.where(result["room_mask"][index])[0]:
        polygons.append(result["polygons"][index, room][result["vertex_mask"][index, room]].tolist())
        labels.append(result["labels"][index, room].item())
    return canonical(polygons, labels)


@pytest.fixture(scope="module")
def upstream():
    # Execute the actual upstream functions without importing unrelated OpenCV
    # rendering utilities, which are not needed by this read-only CPU oracle.
    names = {"get_cycle_basis_and_semantic_3_semansimplified", "x_axis_angle",
             "poly_area", "get_quadrant", "rotate_degree_counterclockwise_from_counter_degree"}
    tree = ast.parse((ROOT / "gsdiff/utils.py").read_text())
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    namespace = {"nx": nx, "copy": copy, "math": math}
    exec(compile(ast.Module(body=functions, type_ignores=[]), "gsdiff/utils.py", "exec"), namespace)
    return namespace["get_cycle_basis_and_semantic_3_semansimplified"]


def oracle(data, upstream):
    xy, semantics, mask, adjacency = (data[key][0] for key in ("xy", "semantics", "node_mask", "adjacency"))
    points = [tuple(xy[i].tolist() + [label if flag == 1 else 99999
                                    for label, flag in enumerate(semantics[i].tolist())])
              for i in torch.where(mask)[0]]
    edges = [(points[i], points[j]) for i, j in torch.nonzero(torch.triu(adjacency), as_tuple=False).tolist()]
    _, polygons, labels = upstream(points, edges)
    return canonical(polygons, labels)


@pytest.mark.parametrize("points,edges,rooms", [
    (SQUARE, SQUARE_EDGES, 1),
    ([(0, 0), (2, 0), (4, 0), (4, 3), (2, 3), (0, 3)],
     [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (0, 5), (1, 4)], 2),
    ([(0, 0), (4, 0), (4, 1), (1, 1), (1, 4), (0, 4)],
     [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (0, 5)], 1),
    (SQUARE + [(8, 0), (12, 0), (12, 3), (8, 3)],
     SQUARE_EDGES + [(4, 5), (5, 6), (6, 7), (4, 7)], 2),
    (SQUARE + [(4, 6), (8, 6), (8, 3)],
     SQUARE_EDGES + [(2, 4), (4, 5), (5, 6), (2, 6)], 2),
])
def test_valid_faces_match_actual_upstream(points, edges, rooms, upstream):
    data = graph(points, edges, slots=53)
    result = extract_rooms(**data)
    assert result["structurally_valid"].all()
    assert result["room_mask"].sum() == rooms
    assert result["polygons"].shape == (1, 106, 53, 2)
    assert tensor_rooms(result) == oracle(data, upstream)
    assert (result["labels"][~result["room_mask"]] == -1).all()
    assert (result["polygons"][~result["vertex_mask"]] == 0).all()


def test_bridges_pruned_including_crossing_dangling_wall(upstream):
    data = graph(SQUARE + [(6, 2)], SQUARE_EDGES + [(0, 4)])
    result = extract_rooms(**data)
    assert result["removed_bridges"].all()
    assert result["raw_wall_intersection"].all()
    assert result["structurally_valid"].all()
    assert result["diagnostics"].item() == 0
    assert tensor_rooms(result) == oracle(data, upstream)


def test_crossings_overlap_and_nested_components_are_invalid():
    crossing = graph(SQUARE, SQUARE_EDGES + [(0, 2), (1, 3)])
    assert extract_rooms(**crossing)["diagnostics"].item() & WALL_INTERSECTION
    overlapping = graph([(0, 0), (4, 0), (2, 0), (2, 3)],
                        [(0, 1), (1, 3), (0, 3), (0, 2), (2, 3)])
    assert extract_rooms(**overlapping)["diagnostics"].item() & WALL_INTERSECTION
    nested = graph([(0, 0), (10, 0), (10, 10), (0, 10),
                    (2, 2), (4, 2), (4, 4), (2, 4)],
                   SQUARE_EDGES + [(4, 5), (5, 6), (6, 7), (4, 7)])
    result = extract_rooms(**nested)
    assert result["room_mask"].sum() == 2
    assert result["diagnostics"].item() & NESTED_ROOMS
    assert not result["structurally_valid"].any()


@pytest.mark.parametrize("failure", ["duplicate", "nonfinite", "selfloop", "semantics"])
def test_malformed_active_nodes_and_selfloops(failure):
    data = graph(SQUARE + [(8, 8)], SQUARE_EDGES, slots=7)
    if failure == "duplicate":
        data["xy"][0, 4] = data["xy"][0, 0]
        bit = DUPLICATE_XY
    elif failure == "nonfinite":
        data["xy"][0, 4, 0] = float("nan")
        bit = NONFINITE
    elif failure == "selfloop":
        data["adjacency"][0, 4, 4] = True
        bit = SELF_LOOP
    else:
        data["semantics"][0, 4, 0] = 2
        bit = INVALID_SEMANTICS
    result = extract_rooms(**data)
    assert not result["structurally_valid"].any()
    assert result["diagnostics"].item() & bit


def test_empty_dense_and_padding_bound_work_without_accepting_invalid_graphs():
    empty = graph(SQUARE, [])
    assert extract_rooms(**empty)["diagnostics"].item() & NO_ROOMS
    dense = graph([(i, i * i) for i in range(53)], [])
    dense["adjacency"][:] = True
    result = extract_rooms(**dense)
    assert result["diagnostics"].item() & EDGE_CAPACITY
    assert result["polygons"].shape == (1, 106, 53, 2)
    padded = graph(SQUARE, SQUARE_EDGES, slots=53)
    padded["xy"][0, 4:] = float("nan")
    padded["semantics"][0, 4:] = 8
    padded["adjacency"][0, 4:, 4:] = True
    assert extract_rooms(**padded)["structurally_valid"].all()


def test_semantic_presence_vote_priority_and_batch_isolation(upstream):
    first = graph(SQUARE, SQUARE_EDGES, slots=8)
    second = graph(SQUARE, SQUARE_EDGES, slots=8)
    first["semantics"][:, :, :] = 1  # all classes tie; storage wins, outside excluded
    second["semantics"][:, :, :] = 0
    second["semantics"][0, :4, 6] = 1
    second["semantics"][0, :3, 4] = 1
    second["semantics"][0, :3, 1] = 1  # bathroom wins tie over bedroom
    data = {key: torch.cat((first[key], second[key])) for key in first}
    result = extract_rooms(**data)
    assert result["labels"][:, 0].tolist() == [2, 4]
    assert tensor_rooms(result, 0) == oracle(first, upstream)
    assert tensor_rooms(result, 1) == oracle(second, upstream)
    for index, single in enumerate((first, second)):
        expected = extract_rooms(**single)
        for key in expected:
            torch.testing.assert_close(result[key][index:index + 1], expected[key])


@pytest.mark.parametrize("path", sorted((ROOT / "results").glob("**/graph.json")))
def test_saved_pretrained_samples_match_polygons_and_labels(path, upstream):
    saved = json.loads(path.read_text())
    points = saved["output_points_test"]
    data = graph([point[:2] for point in points], saved["raw_edge_indices"], slots=53)
    data["semantics"][0, :len(points)] = torch.tensor(
        [[int(value == label) for label, value in enumerate(point[2:])] for point in points])
    result = extract_rooms(**data)
    expected = canonical(saved["simple_cycles_test"], saved["simple_cycles_semantics_test"])
    assert tensor_rooms(result) == expected == oracle(data, upstream)


class NoScalarRead(TorchDispatchMode):
    def __torch_dispatch__(self, function, types, args=(), kwargs=None):
        assert function != torch.ops.aten._local_scalar_dense.default, "Host scalar read in graph hot path"
        return function(*args, **(kwargs or {}))


def test_hot_path_does_not_read_tensor_scalars_on_host():
    with NoScalarRead():
        result = extract_rooms(**graph(SQUARE, SQUARE_EDGES))
    assert result["structurally_valid"].all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_outputs_preserve_device_and_match_cpu():
    data = graph(SQUARE, SQUARE_EDGES, slots=53)
    expected = extract_rooms(**data)
    with NoScalarRead():
        actual = extract_rooms(**{key: value.cuda() for key, value in data.items()})
    for key in expected:
        assert actual[key].device.type == "cuda"
        torch.testing.assert_close(actual[key].cpu(), expected[key])
