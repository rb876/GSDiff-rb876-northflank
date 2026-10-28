"""Sampler contracts use a stochastic toy kernel, not fabricated model results."""

from dataclasses import replace

import numpy as np
import pytest

from test_time.sampling import baseline, best_of_n, choose_index, coverage_pool, selection_key, svdd_pm
from test_time.types import Candidate, ConstraintResult, FloorplanSample, PolicyEvaluation, Scenario


class ToyAdapter:
    steps = 5

    def __init__(self):
        self.counters = {"node_calls": 0, "edge_calls": 0, "graph_completions": 0}
        self.transitions = []
        self.clean_predictions = []

    def new_rng(self, seed):
        return np.random.default_rng(seed)

    def initial_state(self, rng):
        return np.array([rng.normal()])

    def predict_x0(self, state, step):
        self.counters["node_calls"] += 1
        clean = state / 2 + 100
        self.clean_predictions.append((step, state.copy(), clean.copy()))
        return clean

    def sample_transition(self, state, step, rng, x0=None):
        clean = self.predict_x0(state, step) if x0 is None else x0
        result = clean if step == 0 else state / 2 + rng.normal(size=state.shape)
        self.transitions.append((step, state.copy(), result.copy()))
        return result

    def complete_graph(self, state):
        self.counters["edge_calls"] += 1
        self.counters["graph_completions"] += 1
        return FloorplanSample(nodes=[[float(state[0]), 0]], raw_nodes=state.tolist())


class ToyEvaluator:
    def __init__(self, valid=True, measurable=True):
        self.valid = valid
        self.measurable = measurable

    def evaluate(self, sample, scenario):
        energy = abs(sample.nodes[0][0])
        return PolicyEvaluation(energy, -energy, [], self.valid,
                                int(not self.measurable), 1, 0, int(not self.valid),
                                measurable_count=int(self.measurable))


def test_n1_and_k1_reproduce_baseline_rng_and_graph():
    adapters = [ToyAdapter() for _ in range(3)]
    evaluator, scenario = ToyEvaluator(), Scenario()
    ordinary = baseline(adapters[0], evaluator, scenario, 42)
    reranked = best_of_n(adapters[1], evaluator, scenario, 42, n=1)
    guided = svdd_pm(adapters[2], evaluator, scenario, 42, k=1, guided_steps=3)
    expected = ordinary.candidates[0].sample.raw_nodes
    assert reranked.candidates[0].sample.raw_nodes == expected
    assert guided.candidates[0].sample.raw_nodes == expected
    assert adapters[0].counters == adapters[1].counters == adapters[2].counters


def test_best_of_n_saves_every_independent_candidate_and_selects_lowest_energy():
    result = best_of_n(ToyAdapter(), ToyEvaluator(), Scenario(), 42, n=4)
    assert len(result.candidates) == 4
    assert len({item.seed for item in result.candidates}) == 4
    assert result.candidates[0].seed == 42
    assert result.selected_index == min(range(4), key=lambda i: result.candidates[i].evaluation.energy)
    other_trial = best_of_n(ToyAdapter(), ToyEvaluator(), Scenario(), 43, n=4)
    assert not ({item.seed for item in result.candidates} & {item.seed for item in other_trial.candidates})
    assert result.stats["node_calls"] == 4 * ToyAdapter.steps
    assert result.stats["candidate_evaluations"] == 4


def test_selection_prefers_validity_then_coverage_then_energy():
    evaluation = ToyEvaluator().evaluate(FloorplanSample(nodes=[[5, 0]]), Scenario())
    valid = Candidate(FloorplanSample(), evaluation, 10)
    invalid = replace(valid, evaluation=replace(evaluation, structurally_valid=False, energy=0))
    unknown = replace(valid, evaluation=replace(evaluation, unknown_count=1, energy=0))
    lower = replace(valid, evaluation=replace(evaluation, energy=2))
    assert min([invalid, unknown, valid, lower], key=selection_key) is lower
    assert selection_key(valid) < selection_key(unknown) < selection_key(invalid)


def evaluated_candidate(statuses, energy, valid=True):
    results = [ConstraintResult(id, status, None if status == "unknown" else float(status == "fail"),
                                True, status != "unknown", id)
               for id, status in statuses.items()]
    unknown = sum(result.status == "unknown" for result in results)
    evaluation = PolicyEvaluation(energy, -energy, results, valid, unknown, len(results),
                                  0, int(not valid), measurable_count=len(results) - unknown)
    return Candidate(FloorplanSample(), evaluation, 42)


def test_unknown_identity_swap_cannot_hide_a_known_failure():
    anchor = evaluated_candidate({"C033": "fail", "C036": "unknown"}, 1)
    hidden_failure = evaluated_candidate({"C033": "unknown", "C036": "pass"}, 0)
    recovered = evaluated_candidate({"C033": "fail", "C036": "pass"}, 0.5)
    assert coverage_pool([anchor, hidden_failure]) == ([0], 0)
    assert coverage_pool([anchor, hidden_failure, recovered]) == ([2], 0)


def test_coverage_anchor_is_first_candidate_in_best_validity_class():
    invalid = evaluated_candidate({"C033": "unknown", "C036": "unknown"}, 0, valid=False)
    valid = evaluated_candidate({"C033": "fail", "C036": "unknown"}, 1)
    swapped = evaluated_candidate({"C033": "unknown", "C036": "pass"}, 0)
    assert coverage_pool([invalid, valid, swapped]) == ([1], 1)
    a = evaluated_candidate({"C033": "fail"}, 1, valid=False)
    b = evaluated_candidate({"C033": "fail"}, 0, valid=False)
    eligible, anchor = coverage_pool([a, b])
    assert anchor == 0
    assert min(eligible, key=lambda i: selection_key([a, b][i])) == 1


@pytest.mark.parametrize("method", ["best_of_n", "svdd_pm"])
def test_samplers_enforce_unknown_id_guard_end_to_end(method):
    evaluations = [evaluated_candidate({"C033": "fail", "C036": "unknown"}, 1).evaluation,
                   evaluated_candidate({"C033": "unknown", "C036": "pass"}, 0).evaluation]

    class ScriptedEvaluator:
        calls = 0

        def evaluate(self, sample, scenario):
            evaluation = evaluations[min(self.calls, 1)]
            self.calls += 1
            return evaluation

    args = (ToyAdapter(), ScriptedEvaluator(), Scenario(), 42)
    if method == "best_of_n":
        result = best_of_n(*args, n=2)
        assert result.selected_index == 0
        assert result.stats["eligible_candidate_count"] == 1
    else:
        result = svdd_pm(*args, k=2, guided_steps=1)
        assert result.trace[0]["selected_index"] == 0
        assert result.trace[0]["eligible_indices"] == [0]
        assert result.trace[0]["coverage_anchor_index"] == 0


def test_softmax_is_stable_and_greedy_uses_best_finite_reward():
    rng = np.random.default_rng(42)
    assert choose_index([-10, 4, 3, -np.inf], 0, rng) == 1
    assert choose_index([-1e300, -1e300 - 1e290], 1e-100, rng) == 0
    draws = [choose_index([0, 0], 0.1, rng) for _ in range(100)]
    assert set(draws) == {0, 1}
    with pytest.raises(ValueError, match="finite"):
        choose_index([-np.inf, np.nan], 1, rng)


def test_svdd_proposes_distinct_states_and_continues_selected_intermediate():
    adapter = ToyAdapter()
    result = svdd_pm(adapter, ToyEvaluator(), Scenario(), 42, k=4, guided_steps=1)
    proposal_calls = [call for call in adapter.transitions if call[0] == 1]
    assert len(proposal_calls) == 4
    assert len({float(call[2][0]) for call in proposal_calls}) == 4
    selected = result.trace[0]["selected_index"]
    final_input = [call for call in adapter.transitions if call[0] == 0][0][1]
    np.testing.assert_array_equal(final_input, proposal_calls[selected][2])
    clean = result.trace[0]["candidates"][selected]["sample"]["raw_nodes"]
    assert not np.array_equal(final_input, clean)
    assert result.trace[0]["timestep"] == 1
    assert result.trace[0]["candidate_state_timestep"] == 0
    assert result.stats["node_calls"] == adapter.steps + 4
    assert result.stats["edge_calls"] == 5
    assert result.stats["candidate_evaluations"] == 5


@pytest.mark.parametrize("valid,measurable", [(False, True), (True, False)])
def test_all_unscorable_proposals_fall_back_to_first_transition(valid, measurable):
    adapter = ToyAdapter()
    result = svdd_pm(adapter, ToyEvaluator(valid, measurable), Scenario(), 42, k=2, guided_steps=1)
    assert result.trace[0]["selected_index"] == 0
    assert result.trace[0]["fallback"]
    assert result.stats["fallback_count"] == 1
    proposal = [call[2] for call in adapter.transitions if call[0] == 1][0]
    resumed = [call[1] for call in adapter.transitions if call[0] == 0][0]
    np.testing.assert_array_equal(proposal, resumed)


def test_zero_guidance_has_no_surrogate_calls_and_soft_choice_is_reproducible():
    args = (ToyEvaluator(), Scenario(), 42)
    ordinary = baseline(ToyAdapter(), *args)
    disabled = svdd_pm(ToyAdapter(), *args, k=4, guided_steps=0)
    assert disabled.candidates[0].sample.raw_nodes == ordinary.candidates[0].sample.raw_nodes
    a = svdd_pm(ToyAdapter(), *args, k=3, guided_steps=2, temperature=0.1)
    b = svdd_pm(ToyAdapter(), *args, k=3, guided_steps=2, temperature=0.1)
    assert a.candidates[0].sample.raw_nodes == b.candidates[0].sample.raw_nodes
    assert [x["selected_index"] for x in a.trace] == [x["selected_index"] for x in b.trace]


@pytest.mark.parametrize("parameters", [{"k": 0}, {"guided_steps": 5}, {"temperature": -1}])
def test_invalid_sampler_parameters_have_actionable_errors(parameters):
    with pytest.raises(ValueError):
        svdd_pm(ToyAdapter(), ToyEvaluator(), Scenario(), 42, **parameters)
