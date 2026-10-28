"""Batch grouping changes model-call counts, never the stochastic toy result."""

from copy import deepcopy
import math

import numpy as np
import pytest

from test_time import sampling
from test_time.tests.test_adapter import adapter
from test_time.tests.test_sampling import ToyAdapter, ToyEvaluator, evaluated_candidate
from test_time.types import FloorplanSample, Scenario


class ToyBatchAdapter(ToyAdapter):
    """Tiny batched denoiser with an explicitly cloned, stochastic edge draw."""

    def __init__(self):
        super().__init__()
        self.counters.update(node_samples=0, edge_samples=0)
        self.node_batches, self.edge_batches = [], []

    def new_rng(self, seed):
        self.active_rng = super().new_rng(seed)
        return self.active_rng

    def predict_x0(self, state, step):
        self.counters["node_samples"] += 1
        return super().predict_x0(state, step)

    def predict_x0_batch(self, states, step, batch_size=10):
        predictions = []
        for start in range(0, len(states), batch_size):
            chunk = states[start:start + batch_size]
            self.node_batches.append(len(chunk))
            self.counters["node_calls"] += 1
            self.counters["node_samples"] += len(chunk)
            predictions.extend(state / 2 + 100 for state in chunk)
        return predictions

    def _graph(self, state, rng):
        clone = np.random.default_rng(0)
        clone.bit_generator.state = deepcopy(rng.bit_generator.state)
        edge_random = float(clone.normal())
        return FloorplanSample(nodes=[[float(state[0]) + 0.1 * edge_random, 0]],
                               raw_nodes=state.tolist(), metadata={"edge_random": edge_random})

    def complete_graph(self, state):
        self.counters["edge_calls"] += 1
        self.counters["edge_samples"] += 1
        self.counters["graph_completions"] += 1
        return self._graph(state, self.active_rng)

    def complete_graph_batch(self, states, rngs, batch_size=10):
        samples = []
        for start in range(0, len(states), batch_size):
            chunk = states[start:start + batch_size]
            self.edge_batches.append(len(chunk))
            self.counters["edge_calls"] += 1
            self.counters["edge_samples"] += len(chunk)
            self.counters["graph_completions"] += len(chunk)
            samples.extend(self._graph(state, rng) for state, rng in zip(chunk, rngs[start:start + batch_size]))
        return samples


def comparable(result):
    return {"selected_index": result.selected_index, "trace": result.trace,
            "candidates": [(item.seed, item.candidate_id, item.sample.nodes, item.sample.raw_nodes,
                            item.sample.metadata, item.evaluation) for item in result.candidates]}


@pytest.mark.parametrize("method,parameters", [
    ("baseline", {}), ("best_of_n", {"n": 3}),
    ("svdd_pm", {"k": 3, "guided_steps": 2, "temperature": 0}),
    ("svdd_pm", {"k": 3, "guided_steps": 2, "temperature": 0.1}),
])
@pytest.mark.parametrize("microbatch", [1, 2, 10])
def test_batched_methods_match_serial_graphs_seeds_choices_and_traces(method, parameters, microbatch):
    seeds, scenario = [42, 43, 44], Scenario()
    serial_adapter, batch_adapter = ToyBatchAdapter(), ToyBatchAdapter()
    reference = [getattr(sampling, method)(serial_adapter, ToyEvaluator(), scenario, seed, **parameters)
                 for seed in seeds]
    results, aggregate = getattr(sampling, method + "_batch")(
        batch_adapter, ToyEvaluator(), scenario, seeds, inference_batch_size=microbatch, **parameters)
    assert [comparable(result) for result in results] == [comparable(result) for result in reference]
    assert aggregate["node_samples"] == serial_adapter.counters["node_samples"]
    assert aggregate["edge_samples"] == serial_adapter.counters["edge_samples"]
    assert aggregate["candidate_evaluations"] == sum(result.stats["candidate_evaluations"] for result in results)
    assert aggregate["graph_completions"] == sum(result.stats["graph_completions"] for result in results)
    assert sum(result.stats["wall_seconds"] for result in results) == pytest.approx(aggregate["wall_seconds"])
    assert all(result.stats["timing_scope"] == "amortized_batch_time_not_isolated_latency" for result in results)
    assert all("node_calls" not in result.stats and "edge_calls" not in result.stats for result in results)
    if microbatch > 1:
        assert aggregate["node_calls"] < serial_adapter.counters["node_calls"]
        assert aggregate["edge_calls"] < serial_adapter.counters["edge_calls"]
    assert max(batch_adapter.node_batches + batch_adapter.edge_batches) <= microbatch


def test_microbatch_tail_and_counters_include_every_state_once():
    adapter = ToyBatchAdapter()
    results, aggregate = sampling.svdd_pm_batch(adapter, ToyEvaluator(), Scenario(), [10, 11, 12],
                                               k=3, guided_steps=2, inference_batch_size=2)
    assert aggregate["node_calls"] == adapter.steps * math.ceil(3 / 2) + 2 * math.ceil(9 / 2)
    assert aggregate["edge_calls"] == math.ceil(3 / 2) + 2 * math.ceil(9 / 2)
    assert aggregate["node_samples"] == 3 * (adapter.steps + 2 * 3)
    assert aggregate["edge_samples"] == aggregate["candidate_evaluations"] == 3 * (1 + 2 * 3)
    assert aggregate["guided_steps_applied"] == 3 * 2
    assert 1 in adapter.node_batches and 1 in adapter.edge_batches
    assert [result.candidates[0].seed for result in results] == [10, 11, 12]


@pytest.mark.parametrize("k,guided_steps", [(1, 3), (4, 0)])
def test_batch_k1_or_disabled_guidance_matches_baseline(k, guided_steps):
    seeds = [42, 43, 44]
    reference, baseline_stats = sampling.baseline_batch(ToyBatchAdapter(), ToyEvaluator(), Scenario(), seeds)
    results, stats = sampling.svdd_pm_batch(ToyBatchAdapter(), ToyEvaluator(), Scenario(), seeds,
                                           k=k, guided_steps=guided_steps)
    assert [result.candidates[0].sample.raw_nodes for result in results] == [result.candidates[0].sample.raw_nodes for result in reference]
    assert [result.candidates[0].sample.nodes for result in results] == [result.candidates[0].sample.nodes for result in reference]
    assert stats["node_calls"] == baseline_stats["node_calls"]
    assert stats["edge_calls"] == baseline_stats["edge_calls"]
    assert stats["guided_steps_applied"] == 0
    assert all(not result.trace for result in results)


def test_batch_size_one_matches_reference_and_trial_order_does_not_change_designs():
    args = (ToyEvaluator(), Scenario())
    parameters = {"k": 3, "guided_steps": 2, "temperature": 0.1}
    first, _ = sampling.svdd_pm_batch(ToyBatchAdapter(), *args, [42], **parameters)
    ordinary = sampling.svdd_pm(ToyBatchAdapter(), *args, 42, **parameters)
    assert comparable(first[0]) == comparable(ordinary)
    forward, _ = sampling.svdd_pm_batch(ToyBatchAdapter(), *args, [42, 43, 44], **parameters)
    backward, _ = sampling.svdd_pm_batch(ToyBatchAdapter(), *args, [44, 43, 42], **parameters)
    assert [comparable(result) for result in forward] == [comparable(result) for result in reversed(backward)]


@pytest.mark.parametrize("valid,measurable", [(False, True), (True, False)])
def test_batch_unscorable_fallback_keeps_first_noisy_proposal(valid, measurable):
    parameters = {"k": 3, "guided_steps": 2}
    results, stats = sampling.svdd_pm_batch(ToyBatchAdapter(), ToyEvaluator(valid, measurable),
                                           Scenario(), [42, 43], **parameters)
    expected = [sampling.svdd_pm(ToyBatchAdapter(), ToyEvaluator(valid, measurable), Scenario(), seed, **parameters)
                for seed in [42, 43]]
    assert [comparable(result) for result in results] == [comparable(result) for result in expected]
    assert stats["fallback_count"] == 4
    assert all(event["selected_index"] == 0 and event["fallback"] for result in results for event in result.trace)


@pytest.mark.parametrize("method", ["best_of_n", "svdd_pm"])
def test_unknown_identity_guard_applies_to_each_trial_in_a_batch(method):
    evaluations = [evaluated_candidate({"C033": "fail", "C036": "unknown"}, 1).evaluation,
                   evaluated_candidate({"C033": "unknown", "C036": "pass"}, 0).evaluation]

    class AlternatingEvaluator:
        calls = 0

        def evaluate(self, sample, scenario):
            evaluation = evaluations[self.calls % 2]
            self.calls += 1
            return evaluation

    args = (ToyBatchAdapter(), AlternatingEvaluator(), Scenario(), [42, 43])
    if method == "best_of_n":
        results, _ = sampling.best_of_n_batch(*args, n=2)
        assert all(result.selected_index == 0 and result.stats["eligible_candidate_count"] == 1 for result in results)
    else:
        results, _ = sampling.svdd_pm_batch(*args, k=2, guided_steps=1)
        assert all(result.trace[0]["selected_index"] == 0 and result.trace[0]["eligible_indices"] == [0] for result in results)


@pytest.mark.parametrize("seeds,batch_size", [([], 1), ([42], 0)])
def test_batch_configuration_errors_are_clear(seeds, batch_size):
    with pytest.raises(ValueError):
        sampling.baseline_batch(ToyBatchAdapter(), ToyEvaluator(), Scenario(), seeds, batch_size)


@pytest.mark.parametrize("method,parameters", [
    ("baseline", {}), ("best_of_n", {"n": 2}),
    ("svdd_pm", {"k": 3, "guided_steps": 2, "temperature": 0.1}),
])
def test_sampler_and_real_adapter_contract_with_tiny_cpu_networks(adapter, method, parameters):
    # Keep the real adapter/scheduler/RNG implementation, with tiny fixture
    # networks and a short reverse-loop prefix. This is not pretrained inference.
    adapter.steps = 5
    seeds = [41, 42, 43]
    reference = [getattr(sampling, method)(adapter, ToyEvaluator(), Scenario(), seed, **parameters)
                 for seed in seeds]
    expected = [comparable(result) for result in reference]
    for microbatch in [1, 2, 10]:
        actual, aggregate = getattr(sampling, method + "_batch")(
            adapter, ToyEvaluator(), Scenario(), seeds, inference_batch_size=microbatch, **parameters)
        assert [comparable(result) for result in actual] == expected
        assert aggregate["graph_completions"] == sum(result.stats["graph_completions"] for result in actual)
