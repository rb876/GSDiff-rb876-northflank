"""Actual adapter math/RNG contracts with tiny networks, no checkpoints or GPU."""

from types import ModuleType, SimpleNamespace
import sys

import numpy as np
import pytest
import torch

from test_time.adapter import GSDiffAdapter


class TinyNode(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1))

    def forward(self, state, mask, timestep):
        return torch.zeros_like(state[:, :, :2]), torch.zeros_like(state[:, :, 2:])


class RandomEdge(TinyNode):
    def forward(self, corners, mask, padding, semantics):
        self.last_draw = torch.rand((1, 2809, 1), device=corners.device)
        return torch.cat((1 - self.last_draw, self.last_draw), dim=2), self.last_draw, None


@pytest.fixture
def adapter(monkeypatch, tmp_path):
    package = ModuleType("gsdiff")
    node_module, edge_module = ModuleType("gsdiff.house_nn1"), ModuleType("gsdiff.house_nn2")
    node_module.HeterHouseModel, edge_module.EdgeModel = TinyNode, RandomEdge
    coordinates = np.array([[[0, 0], [256, 0], [0, 256]]])
    package.utils = SimpleNamespace(
        inverse_normalize_and_remove_padding_100_4testing=lambda *args, **kwargs:
            ([coordinates], [np.zeros((1, 3, 7), dtype=np.int8)]),
        edges_to_coordinates=lambda adjacency, points:
            [(points[i // 3], points[i % 3]) for i, present in enumerate(adjacency) if present],
        get_cycle_basis_and_semantic_3_semansimplified=lambda points, edges:
            ({}, [[*points, points[0]]], [0]),
    )
    for name, module in (("gsdiff", package), ("gsdiff.house_nn1", node_module),
                         ("gsdiff.house_nn2", edge_module)):
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: {"weight": torch.ones(1)})
    dummy = tmp_path / "mock-weights.pt"
    dummy.touch()
    return GSDiffAdapter(node_checkpoint=dummy, edge_checkpoint=dummy, device="cpu")


def test_schedule_has_stochastic_reverse_steps_and_deterministic_final_step(adapter):
    assert adapter.variance[0].item() == 0
    assert torch.all(adapter.variance[1:] > 0)
    assert adapter.coefficient1[0].item() == pytest.approx(1)
    assert adapter.coefficient2[0].item() == pytest.approx(0)
    state = torch.full((1, 53, 10), 0.2, dtype=torch.float64)
    clean = torch.zeros_like(state)
    first_rng, second_rng = adapter.new_rng(1), adapter.new_rng(2)
    before = first_rng.get_state().clone()
    first = adapter.sample_transition(state, 0, first_rng, x0=clean)
    second = adapter.sample_transition(state, 0, second_rng, x0=clean)
    torch.testing.assert_close(first, clean)
    torch.testing.assert_close(second, clean)
    # Upstream still draws noise at t=0; retaining this matters for edge RNG.
    assert not torch.equal(before, first_rng.get_state())
    a = adapter.sample_transition(state, 1, first_rng, x0=clean)
    b = adapter.sample_transition(state, 1, second_rng, x0=clean)
    assert not torch.equal(a, b)


def test_clean_prediction_retains_original_coordinate_semantic_padding_thresholds(adapter):
    state = torch.zeros((1, 53, 10), dtype=torch.float64)
    state[:, :, :2] = torch.tensor([-2, 2])
    state[:, :, 2:4] = torch.tensor([0.4, 0.6])
    state[0, 0, 9], state[0, 1, 9] = 0.74, 0.76
    result = adapter.predict_x0(state.requires_grad_(), 0)
    assert result[0, 0, :2].tolist() == [-1, 1]
    assert result[0, 0, 2:4].tolist() == [0, 1]
    assert result[0, :2, 9].tolist() == [0, 1]
    assert not result.requires_grad
    assert all(not parameter.requires_grad for model in (adapter.node, adapter.edge)
               for parameter in model.parameters())


def test_edge_rng_matches_original_stream_and_surrogates_do_not_advance_it(adapter):
    seed = 42
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        # Original script: initial noise + one posterior noise at each timestep.
        for _ in range(adapter.steps + 1):
            torch.randn((1, 53, 10), dtype=torch.float64)
        expected_diffusion_state = torch.get_rng_state().clone()
        expected_edge_draw = torch.rand((1, 2809, 1))
    global_state = torch.get_rng_state().clone()
    sample = adapter.sample_baseline(seed)
    torch.testing.assert_close(adapter.edge.last_draw, expected_edge_draw, rtol=0, atol=0)
    assert torch.equal(adapter.active_rng.get_state(), expected_diffusion_state)
    assert torch.equal(torch.get_rng_state(), global_state)

    clean = torch.tensor([sample.raw_nodes], dtype=torch.float64)
    repeated = adapter.complete_graph(clean)
    assert repeated.nodes == sample.nodes
    assert repeated.raw_edges == sample.raw_edges
    assert repeated.edges == sample.edges
    assert repeated.rooms == sample.rooms
    assert repeated.metadata["edge_rng_state_sha256"] == sample.metadata["edge_rng_state_sha256"]
    assert torch.equal(adapter.active_rng.get_state(), expected_diffusion_state)
    assert torch.equal(torch.get_rng_state(), global_state)
    torch.testing.assert_close(adapter.edge.last_draw, expected_edge_draw, rtol=0, atol=0)
