"""Tensor graph decoding uses the same node quantization and edge noise."""

import numpy as np
import torch

from test_time.tests.test_adapter import adapter, decoded_state, reset_counts, variable_decode


def test_tensor_graph_decode_matches_cpu_reference_without_calling_cpu_geometry(adapter):
    adapter.utils.inverse_normalize_and_remove_padding_100_4testing = variable_decode
    states = [decoded_state(count) for count in (3, 5, 2, 3)]
    # Padding holes must be compacted without changing the surviving node order.
    states[0][0, 1, 9], states[0][0, 4, 9] = 1, 0
    states[-1][0, 0, 0] = float("nan")
    rngs = [adapter.new_rng(seed) for seed in (11, 22, 33, 44)]
    expected = [adapter.complete_graph(state, rng=rng) for state, rng in zip(states, rngs)]
    snapshots = [rng.get_state().clone() for rng in rngs]
    global_state = torch.get_rng_state().clone()

    def no_cpu_geometry(*args, **kwargs):
        raise AssertionError("Tensor decoding called the CPU geometry path")

    adapter.utils.inverse_normalize_and_remove_padding_100_4testing = no_cpu_geometry
    adapter.utils.get_cycle_basis_and_semantic_3_semansimplified = no_cpu_geometry
    reset_counts(adapter)
    tensors = adapter.predict_graph_tensors(states, rngs, batch_size=2)
    assert tensors["node_mask"].sum(dim=1).tolist() == [3, 5, 2, 0]
    for index, reference in enumerate(expected):
        mask = tensors["node_mask"][index]
        actual = torch.cat((tensors["xy"][index, mask], tensors["semantics"][index, mask]), dim=1)
        np.testing.assert_array_equal(actual.numpy(), np.asarray(reference.nodes).reshape(-1, 9))
        assert tensors["adjacency"][index].nonzero().tolist() == reference.raw_edges
    assert adapter.edge.batch_sizes == [2]
    assert adapter.counters["edge_samples"] == 2
    assert adapter.counters["graph_completions"] == 4
    assert torch.equal(torch.get_rng_state(), global_state)
    assert all(torch.equal(rng.get_state(), saved) for rng, saved in zip(rngs, snapshots))


def test_tensor_graph_proposals_share_edge_noise_across_microbatches(adapter):
    rng = adapter.new_rng(42)
    states = [decoded_state(3) for _ in range(3)]
    first = adapter.predict_graph_tensors(states, [rng] * 3, batch_size=1)
    second = adapter.predict_graph_tensors(states, [rng] * 3, batch_size=3)
    for key in first:
        torch.testing.assert_close(first[key], second[key], rtol=0, atol=0)
    assert torch.equal(first["adjacency"][0], first["adjacency"][2])


def test_empty_tensor_graph_batch(adapter):
    result = adapter.predict_graph_tensors([], [])
    assert result["xy"].shape == (0, 53, 2)
    assert result["adjacency"].shape == (0, 53, 53)
    assert not any(adapter.counters.values())
