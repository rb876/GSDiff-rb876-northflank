"""Bounded shared-wall edits checked against real vector geometry and policies."""

from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from shapely.geometry import Polygon
from shapely.ops import unary_union

from test_time.adapter import GSDiffAdapter
from test_time.audit_geometry import refined_bedroom_search
from test_time.constraints import load_specs
from test_time.gpu_constraints import TensorConstraintEvaluator
from test_time.gpu_graph import extract_rooms
from test_time.policy import ConstraintEvaluator
from test_time.repair import (candidate_bank, feasibility_mask, final_repair, graph_tensors,
                              repair_settings, score_repairs)
from test_time.tests.test_gpu_graph import graph, upstream
from test_time.types import FloorplanSample, Scenario, overall_validity


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def scenario():
    return Scenario(coordinate_scale_m_per_model_unit=1, scale_provenance="assumed")


@pytest.fixture
def specs():
    return [spec for spec in load_specs('test_time/configs/constraints_v2.json')
            if spec.id in {"C033", "C034", "C035"}]


def divided(width=3.0):
    result = graph([(0, 0), (width, 0), (8.4, 0), (8.4, 3.4), (width, 3.4), (0, 3.4)],
                   [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (0, 5), (1, 4)])
    result['semantics'].zero_()
    result['semantics'][0, [0, 1, 4, 5], 1] = 1
    result['semantics'][0, [1, 2, 3, 4], 0] = 1
    return result


@pytest.fixture
def fixed_adapter(upstream):
    adapter = GSDiffAdapter.__new__(GSDiffAdapter)
    adapter.device = torch.device('cpu')
    adapter.utils = SimpleNamespace(
        edges_to_coordinates=lambda adjacency, points:
            [(points[i // len(points)], points[i % len(points)]) for i, present in enumerate(adjacency) if present],
        get_cycle_basis_and_semantic_3_semansimplified=upstream)
    return adapter


def sample_from_graph(data, adapter):
    xy = data['xy'].numpy()
    semantics = data['semantics'].numpy()
    sample = FloorplanSample(nodes=np.concatenate((xy, semantics), -1)[0].tolist(),
                             metadata={'original_marker': True})
    return adapter._finish_graph(sample, xy, semantics, data['adjacency'][0].numpy())


def test_tensor_bank_is_bounded_and_moves_shared_wall_without_moving_corners(scenario):
    original = divided()
    saved = {key: value.clone() for key, value in original.items()}
    bank = candidate_bank(original, scenario, max_evaluations=16)
    assert len(bank['owner']) <= 16
    torch.testing.assert_close(bank['graph']['xy'][0], original['xy'][0])
    assert (bank['displacement_m'] <= .20 + 1e-8).all()
    assert (bank['displacement_m'][:, [0, 2, 3, 5]] == 0).all()
    assert torch.equal(bank['graph']['xy'][:, 1, 0], bank['graph']['xy'][:, 4, 0])
    for key, value in saved.items():
        torch.testing.assert_close(original[key], value, rtol=0, atol=0)
    rooms = extract_rooms(**bank['graph'])
    assert rooms['structurally_valid'].all()
    for row in range(len(bank['owner'])):
        shape = unary_union([Polygon(rooms['polygons'][row, room][rooms['vertex_mask'][row, room]].tolist())
                             for room in rooms['room_mask'][row].nonzero().flatten()])
        assert shape.equals(Polygon([(0, 0), (8.4, 0), (8.4, 3.4), (0, 3.4)]))


def test_repair_finds_joint_bed_circle_and_clearance_witness(scenario, specs):
    policy = TensorConstraintEvaluator(specs, max_positions=2048)
    bank = score_repairs(divided(), policy, scenario, max_evaluations=16, batch_size=2)
    selected = int(bank['selected'][0])
    assert feasibility_mask(bank['scores'], policy.specs)[selected]
    assert bank['graph']['xy'][selected, 1, 0] == pytest.approx(3.2)
    assert bank['scores']['status'][selected].tolist() == [0, 3, 0]


def test_already_feasible_identity_wins(scenario, specs):
    policy = TensorConstraintEvaluator(specs, max_positions=2048)
    bank = score_repairs(divided(3.2), policy, scenario, max_evaluations=16)
    assert bank['candidate'][bank['selected']].tolist() == [0]
    assert bank['distortion'][bank['selected']].tolist() == [0]


def test_too_small_budget_does_not_claim_feasibility(scenario, specs):
    policy = TensorConstraintEvaluator(specs, max_positions=2048)
    bank = score_repairs(divided(2.8), policy, scenario, {'max_displacement_m': .10})
    assert not feasibility_mask(bank['scores'], policy.specs).any()
    assert (bank['displacement_m'] <= .10 + 1e-8).all()


def test_new_unknown_id_cannot_hide_failure(scenario):
    class Policy:
        specs = [SimpleNamespace(counts_toward_compliance=True)] * 2
        def evaluate(self, rooms, scenario):
            count = len(rooms['labels'])
            width = rooms['polygons'][:, 0, :, 0].amax(-1)
            # Identity has known failure A and unknown B; positive edits swap them.
            status = torch.tensor([[1, 2]] * count)
            status[width > 3.01] = torch.tensor([2, 0])
            return dict(energy=(status == 1).sum(-1).double(), status=status,
                        applicable=torch.ones_like(status, dtype=torch.bool), structurally_valid=rooms['structurally_valid'],
                        unknown_count=(status == 2).sum(-1), measurable_count=(status < 2).sum(-1),
                        applicable_count=torch.full((count,), 2))
    bank = score_repairs(divided(), Policy(), scenario)
    changed_to_unknown = bank['scores']['status'][:, 0] == 2
    assert changed_to_unknown.any()
    assert not bank['eligible'][changed_to_unknown].any()


def test_crossing_and_single_room_graphs_receive_identity_only(scenario):
    crossed = graph([(0, 0), (4, 0), (4, 4), (0, 4)],
                    [(0, 1), (1, 2), (2, 3), (0, 3), (0, 2), (1, 3)])
    assert len(candidate_bank(crossed, scenario)['owner']) == 1
    square = graph([(0, 0), (4, 0), (4, 4), (0, 4)], [(0, 1), (1, 2), (2, 3), (0, 3)])
    assert len(candidate_bank(square, scenario)['owner']) == 1


def test_final_cpu_repair_preserves_input_and_decodes_fixed_edges(scenario, specs, fixed_adapter):
    sample = sample_from_graph(divided(), fixed_adapter)
    saved = deepcopy(sample.to_dict())
    cpu = ConstraintEvaluator(specs, bedroom_search_fn=refined_bedroom_search)
    before = cpu.evaluate(sample, scenario)
    assert overall_validity(before) == 'invalid'
    gpu = TensorConstraintEvaluator(specs, max_positions=2048)
    repaired, evaluation, record = final_repair(fixed_adapter, sample, cpu, scenario, gpu)
    assert overall_validity(evaluation) == 'valid'
    assert sample.to_dict() == saved
    assert repaired.edges == sample.edges
    assert len(record['edits']) == 2
    assert record['max_displacement_m'] <= .20 + 1e-8
    assert record['cpu_evaluations'] <= 128
    assert repaired.metadata['source_geometry'] == 'solver_modified_existing_junctions'


def test_cpu_rejects_a_gpu_success_if_joint_checker_disagrees(scenario, specs, fixed_adapter):
    sample = sample_from_graph(divided(), fixed_adapter)
    original_cpu = ConstraintEvaluator(specs, bedroom_search_fn=refined_bedroom_search)
    original = original_cpu.evaluate(sample, scenario)
    class StrictAudit:
        def evaluate(self, graph, scenario):
            return deepcopy(original)
    result, evaluation, record = final_repair(fixed_adapter, sample, StrictAudit(), scenario,
                                               TensorConstraintEvaluator(specs, max_positions=2048))
    assert result.nodes == sample.nodes
    assert overall_validity(evaluation) != 'valid'
    assert record['search_status'] == 'no_improvement'
    assert record['feasible'] is False
    assert record['energy'] == original.energy


def test_bank_batches_do_not_change_choices(scenario, specs):
    data = {key: value.expand(2, *value.shape[1:]).clone() for key, value in divided().items()}
    outputs = [score_repairs(data, TensorConstraintEvaluator(specs, max_positions=1024), scenario,
                            batch_size=batch) for batch in (1, 4)]
    assert outputs[0]['selected'].tolist() == outputs[1]['selected'].tolist()
    torch.testing.assert_close(outputs[0]['scores']['energy'], outputs[1]['scores']['energy'])


def test_preflight_rejects_missing_scale_and_unknown_settings():
    with pytest.raises(ValueError, match='scale'):
        candidate_bank(divided(), Scenario())
    with pytest.raises(ValueError, match='Unknown repair'):
        repair_settings({'max_edge_flips': 2})


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
def test_cuda_repair_preserves_tensor_device_and_agrees_with_cpu(scenario, specs):
    original = divided()
    cpu = score_repairs(original, TensorConstraintEvaluator(specs, max_positions=1024), scenario, batch_size=2)
    gpu = score_repairs({key: value.cuda() for key, value in original.items()},
                        TensorConstraintEvaluator(specs, max_positions=1024), scenario, batch_size=2)
    assert gpu['graph']['xy'].is_cuda
    assert gpu['selected'].is_cuda
    torch.testing.assert_close(cpu['scores']['status'], gpu['scores']['status'].cpu())
    torch.testing.assert_close(cpu['selected'], gpu['selected'].cpu())


def test_overlay_marks_only_actual_junction_moves(tmp_path, fixed_adapter):
    from test_time.repair import save_repair_overlay
    before = sample_from_graph(divided(), fixed_adapter)
    after = sample_from_graph(divided(3.2), fixed_adapter)
    path = tmp_path / 'overlay.svg'
    assert save_repair_overlay(before, after, path) == str(path)
    text = path.read_text()
    assert '2 junctions moved' in text
    assert text.count('marker-end="url(#move)"') == 2
