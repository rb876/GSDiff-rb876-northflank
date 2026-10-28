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
        self.batch_sizes = []

    def forward(self, state, mask, timestep):
        assert mask.shape == (state.shape[0], 53, 53)
        assert timestep.shape == (state.shape[0],)
        assert torch.all(mask == 1)
        self.batch_sizes.append(state.shape[0])
        return torch.zeros_like(state[:, :, :2]), torch.zeros_like(state[:, :, 2:])


class RandomEdge(TinyNode):
    def forward(self, corners, mask, padding, semantics, random_lambda=None):
        count = corners.shape[0]
        assert corners.shape == (count, 53, 2)
        assert semantics.shape == (count, 53, 7)
        expected_mask = padding.squeeze(2).bool()
        assert torch.equal(mask, expected_mask[:, :, None] & expected_mask[:, None, :])
        self.batch_sizes.append(count)
        self.last_draw = (torch.rand((count, 2809, 1), device=corners.device)
                          if random_lambda is None else random_lambda)
        self.draws = getattr(self, "draws", []) + [self.last_draw.clone()]
        self.inputs = getattr(self, "inputs", []) + [(corners.clone(), padding.clone(), semantics.clone())]
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
            [(points[i // len(points)], points[i % len(points)])
             for i, present in enumerate(adjacency) if present],
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


def reset_counts(adapter):
    adapter.counters = dict.fromkeys(adapter.counters, 0)
    adapter.node.batch_sizes.clear()
    adapter.edge.batch_sizes.clear()
    adapter.edge.draws = []
    adapter.edge.inputs = []


def test_node_microbatches_match_single_predictions_and_limit_forward_size(adapter):
    states = [adapter.initial_state(adapter.new_rng(seed)) for seed in range(5)]
    expected = [adapter.predict_x0(state, 500) for state in states]
    reset_counts(adapter)
    actual = adapter.predict_x0_batch(states, 500, batch_size=2)
    for result, reference in zip(actual, expected):
        assert result.shape == (1, 53, 10)
        torch.testing.assert_close(result, reference, rtol=0, atol=0)
    assert adapter.node.batch_sizes == [2, 2, 1]
    assert adapter.counters["node_calls"] == 3
    assert adapter.counters["node_samples"] == 5


def decoded_state(count):
    state = torch.zeros((1, 53, 10), dtype=torch.float64)
    state[0, :, 0] = torch.arange(53, dtype=torch.float64) / 53 - 0.5
    state[0, :, 1] = 0.2
    state[0, :count, 2] = 1
    state[0, count:, 9] = 1
    return state


def variable_decode(corners, semantics, padding, resolution):
    assert resolution == 512
    valid = padding[0].cpu().numpy() == 0
    return ([(corners[0].cpu().numpy() * 256 + 256).astype(np.int32)[:, valid]],
            [np.round(semantics[0].cpu().numpy()).astype(np.int8)[:, valid]])


def test_edge_microbatches_preserve_padding_order_and_invalid_samples(adapter):
    adapter.utils.inverse_normalize_and_remove_padding_100_4testing = variable_decode
    states = [decoded_state(count) for count in (3, 2, 5, 4, 1, 3)]
    states[-1][0, 0, 0] = float("nan")
    rngs = [adapter.new_rng(seed) for seed in range(len(states))]
    expected = [adapter.complete_graph(state, rng=rng) for state, rng in zip(states, rngs)]
    reset_counts(adapter)
    actual = adapter.complete_graph_batch(states, rngs, batch_size=2)
    assert [sample.to_dict() for sample in actual] == [sample.to_dict() for sample in expected]
    assert [len(sample.nodes) for sample in actual] == [3, 2, 5, 4, 1, 0]
    assert actual[1].diagnostics == ["fewer_than_three_junctions"]
    assert actual[-1].diagnostics == ["nonfinite_node_state"]
    assert actual[-1].raw_nodes == []
    assert adapter.edge.batch_sizes == [2, 1]
    assert adapter.counters["edge_calls"] == 2
    assert adapter.counters["edge_samples"] == 3
    assert adapter.counters["graph_completions"] == 6
    valid_counts = [3, 5, 4]
    index = 0
    for corners, padding, semantics in adapter.edge.inputs:
        for item in range(corners.shape[0]):
            count = valid_counts[index]
            assert padding[item].sum().item() == count
            assert torch.count_nonzero(corners[item, count:]) == 0
            assert torch.count_nonzero(semantics[item, count:]) == 0
            index += 1


def test_batch_edge_rng_is_per_seed_common_for_proposals_and_unchanged(adapter):
    first, second = adapter.new_rng(11), adapter.new_rng(22)
    # Emulate different diffusion trajectories before graph completion.
    torch.randn((1, 53, 10), generator=first)
    torch.randn((3, 53, 10), generator=second)
    rngs = [first, second, first, second, first]
    states = [decoded_state(3) for _ in rngs]
    snapshots = [rng.get_state().clone() for rng in rngs]
    expected = [torch.rand((1, 2809, 1), generator=torch.Generator().set_state(state))
                for state in snapshots]
    global_state = torch.get_rng_state().clone()
    batched = adapter.complete_graph_batch(states, rngs, batch_size=2)
    draws = torch.cat(adapter.edge.draws)
    torch.testing.assert_close(draws, torch.cat(expected), rtol=0, atol=0)
    torch.testing.assert_close(draws[0], draws[2], rtol=0, atol=0)
    assert not torch.equal(draws[0], draws[1])
    assert torch.equal(torch.get_rng_state(), global_state)
    assert all(torch.equal(rng.get_state(), snapshot) for rng, snapshot in zip(rngs, snapshots))
    reset_counts(adapter)
    singles = adapter.complete_graph_batch(states, rngs, batch_size=1)
    assert [sample.to_dict() for sample in batched] == [sample.to_dict() for sample in singles]
    assert torch.equal(torch.get_rng_state(), global_state)
    assert all(torch.equal(rng.get_state(), snapshot) for rng, snapshot in zip(rngs, snapshots))


def test_empty_and_malformed_batches_do_not_forward(adapter):
    assert adapter.predict_x0_batch([], 0) == []
    assert adapter.complete_graph_batch([], []) == []
    with pytest.raises(ValueError, match="positive integer"):
        adapter.predict_x0_batch([], 0, batch_size=0)
    with pytest.raises(ValueError, match="one trajectory RNG"):
        adapter.complete_graph_batch([decoded_state(3)], [])
    with pytest.raises(ValueError, match="shape"):
        adapter.predict_x0_batch([torch.zeros((2, 53, 10))], 0)
    assert not any(adapter.counters.values())


@pytest.mark.parametrize("batch_size", [1, 2])
def test_original_edge_forward_accepts_explicit_draws_without_global_rng_changes(batch_size):
    # Exercise the actual forward implementation with tiny projections and no
    # transformer layers; no checkpoints, large networks, or GPU are needed.
    from gsdiff.house_nn2 import EdgeModel

    class SmallEdge(torch.nn.Module):
        forward = EdgeModel.forward

        def __init__(self):
            super().__init__()
            self.d_model = 8
            self.semantics_embedding = torch.nn.Linear(7, 4)
            self.transformer_layers = torch.nn.ModuleList()
            self.edges_MLP = torch.nn.Linear(8, 2)
            self.lambdas_MLP = torch.nn.Linear(8, 1)

    model = SmallEdge()
    corners = torch.stack((torch.linspace(-1, 1, 53), torch.linspace(0.5, -0.5, 53)), dim=1)
    corners = corners.unsqueeze(0).expand(batch_size, -1, -1)
    mask = torch.ones((batch_size, 53, 53), dtype=torch.bool)
    padding = torch.ones((batch_size, 53, 1), dtype=torch.uint8)
    semantics = torch.zeros((batch_size, 53, 7))
    before = torch.get_rng_state().clone()
    draws = torch.rand((batch_size, 2809, 1), generator=torch.Generator().set_state(before))
    output, returned_draws, _ = model(corners, mask, padding, semantics, random_lambda=draws)
    assert output.shape == (batch_size, 2809, 2)
    torch.testing.assert_close(returned_draws, draws, rtol=0, atol=0)
    assert torch.equal(torch.get_rng_state(), before)
    default_output, default_draws, _ = model(corners, mask, padding, semantics)
    torch.testing.assert_close(default_draws, draws, rtol=0, atol=0)
    torch.testing.assert_close(default_output, output, rtol=0, atol=0)
    assert not torch.equal(torch.get_rng_state(), before)
