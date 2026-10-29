"""Model-free contracts for binary continuations, budgets and returned graphs."""

from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest

from test_time.hard_feasibility import (
    advance_base, complete_base, hard_h_mc, select_h, stream_seed,
)
from test_time.tests.test_adapter import adapter
from test_time.tests.test_sampling_batch import ToyBatchAdapter
from test_time.types import ConstraintResult, PolicyEvaluation, Scenario, overall_validity


class SmallAdapter(ToyBatchAdapter):
    steps = 20

    def __init__(self):
        super().__init__()
        self.roots = []

    def initial_state(self, rng):
        state = super().initial_state(rng)
        self.roots.append(state.copy())
        return state


def evaluation(status="pass", *, structural=True):
    check = ConstraintResult("C035", status, None if status == "unknown" else float(status == "fail"),
                             True, status != "unknown", "test:C035")
    # Deliberately unusable energies: only binary outcomes may drive this method.
    return PolicyEvaluation(float("nan"), float("nan"), [check], structural,
                            int(status == "unknown"), 1, 0, float(not structural),
                            measurable_count=int(status != "unknown"))


class FixedEvaluator:
    def __init__(self, status="pass", structural=True):
        self.result = evaluation(status, structural=structural)

    def evaluate(self, sample, scenario):
        return self.result


PARAMETERS = {"k": 2, "rollouts": 3, "anchor_remaining": (4, 2, 1), "terminal_k": 3}


def run(adapter=None, evaluator=None, **kwargs):
    return hard_h_mc(adapter or SmallAdapter(), evaluator or FixedEvaluator(), Scenario(), 42,
                     **{**PARAMETERS, **kwargs})


def test_success_labels_every_completion_and_counts_all_network_work():
    adapter, events = SmallAdapter(), []
    result = run(adapter, event_callback=lambda sample, evaluation, event: events.append((sample, evaluation, event)))
    assert result.status == "success" and overall_validity(result.evaluation) == "valid"
    # 2*16 + 2*3*4 + 2*2 + 2*3*2 + 2*1 + 2*3*1 + 3*1
    assert result.stats["node_samples"] == 83
    assert result.stats["graph_completions"] == result.stats["candidate_evaluations"] == 21
    assert len(events) == 21
    assert len({event["event_id"] for _, _, event in events}) == 21
    assert result.stats["first_success"]["node_samples"] == 56
    assert result.stats["first_success"]["graph_completions"] == 6
    assert result.stats["first_success"]["batched_work_already_charged"]
    assert result.stats["any_feasible_found"] and result.stats["returned_success"]
    assert result.stats["exact_conditional"] is False
    returned_event = result.sample.metadata["event_id"]
    assert next(sample for sample, _, event in events if event["event_id"] == returned_event) is result.sample
    assert result.sample.metadata["origin"] == "terminal"
    assert adapter.counters["edge_samples"] == 21  # No new decoder draw after selection.


def test_root_proposals_have_independent_initial_noise_and_never_share_one_prefix():
    adapter = SmallAdapter()
    result = run(adapter)
    assert len(adapter.roots) == PARAMETERS["k"]
    assert not np.array_equal(adapter.roots[0], adapter.roots[1])
    expected = [np.random.default_rng(stream_seed(42, 0, proposal=i)).normal() for i in range(2)]
    assert [state.item() for state in adapter.roots] == expected
    assert [(stage["from_remaining"], stage["to_remaining"]) for stage in result.trace] == [
        (20, 4), (4, 2), (2, 1), (1, 0)]


def test_selection_is_proportional_not_greedy_or_softmax():
    rng = np.random.default_rng(51)
    draws = np.bincount([select_h([1, 3, 6], rng) for _ in range(8000)], minlength=3) / 8000
    np.testing.assert_allclose(draws, [0.1, 0.3, 0.6], atol=0.02)
    assert select_h([0, 0], rng) is None


def test_finite_bank_selection_is_not_exact_even_with_exact_h():
    rng = np.random.default_rng(52)
    selected_a = 0
    for _ in range(8000):
        bank = rng.integers(0, 2, size=2)
        chosen = select_h(np.where(bank == 0, 0.9, 0.1), rng)
        selected_a += bank[chosen] == 0
    # Exact h-conditioned P(A) is .9; a size-two proposal bank gives .7.
    assert selected_a / 8000 == pytest.approx(0.7, abs=0.025)


@pytest.mark.parametrize("status,structural", [("unknown", True), ("fail", True), ("pass", False)])
def test_unknown_violation_and_invalid_structure_never_count_as_success(status, structural):
    result = run(evaluator=FixedEvaluator(status, structural), zero_policy="abort")
    assert result.status == "no_positive_h_estimate"
    assert result.sample is result.evaluation is None
    assert not result.stats["any_feasible_found"]
    assert result.stats["all_zero_count"] == 1
    assert result.stats["node_samples"] == 56


def test_fresh_base_fallback_is_new_independent_prefix_and_fully_charged():
    adapter = SmallAdapter()
    result = run(adapter, FixedEvaluator("fail"))
    assert result.status == "no_feasible_terminal_sample"
    assert result.sample is result.evaluation is None
    assert result.stats["fallback_count"] == result.stats["all_zero_count"] == 3
    assert result.stats["node_samples"] == 83 + 16 + 2 + 1
    assert len(adapter.roots) == 3
    assert result.trace[0]["fallback_seed"] not in result.trace[0]["proposal_seeds"]
    assert adapter.roots[2].item() == np.random.default_rng(result.trace[0]["fallback_seed"]).normal()
    assert result.stats["outcome_counts"]["invalid"] == 21


def test_successful_lookahead_is_recorded_but_not_returned_after_terminal_failure():
    class LookaheadOnly:
        def evaluate(self, sample, scenario):
            return evaluation("pass" if sample.metadata["origin"] == "lookahead" else "fail")

    result = run(evaluator=LookaheadOnly())
    assert result.status == "no_feasible_terminal_sample"
    assert result.sample is None
    assert result.stats["any_feasible_found"] and not result.stats["returned_success"]
    assert result.stats["feasible_graphs_found"] == 18


def test_budget_reserves_balanced_bank_and_possible_fallback_before_starting():
    # Root needs 56 NFE plus a reserved 16-NFE fallback.
    result = run(max_nfe=71)
    assert result.status == "budget_exhausted"
    assert result.stats["node_samples"] == result.stats["completed_graphs"] == 0
    assert result.trace[0]["nfe_required"] == 72
    # Root succeeds, then second balanced stage+fallback no longer fits.
    result = run(max_nfe=73)
    assert result.status == "budget_exhausted"
    assert result.stats["node_samples"] == 56
    assert result.stats["completed_graphs"] == 6
    assert result.stats["any_feasible_found"] and result.sample is None
    assert run(max_nfe=83).status == "success"


def test_abort_policy_does_not_reserve_unused_fallback_work():
    result = run(evaluator=FixedEvaluator("fail"), zero_policy="abort", max_nfe=56)
    assert result.status == "no_positive_h_estimate"
    assert result.stats["node_samples"] == 56


def _without_timing(value):
    if isinstance(value, dict):
        return {key: _without_timing(item) for key, item in value.items()
                if "seconds" not in key and key not in {"node_calls", "edge_calls", "inference_batch_size"}}
    if isinstance(value, list):
        return [_without_timing(item) for item in value]
    return value


def test_microbatch_size_and_repeated_runs_preserve_random_streams_and_decisions():
    first = run(inference_batch_size=1)
    second = run(inference_batch_size=10)
    repeat = run(inference_batch_size=10)
    assert _without_timing(first.trace) == _without_timing(second.trace) == _without_timing(repeat.trace)
    assert first.sample.nodes == second.sample.nodes == repeat.sample.nodes
    assert first.sample.raw_nodes == second.sample.raw_nodes == repeat.sample.raw_nodes
    assert first.stats["node_samples"] == second.stats["node_samples"]
    assert first.stats["node_calls"] > second.stats["node_calls"]


def test_more_rollouts_do_not_change_existing_root_or_rollout_streams():
    first, second = run(rollouts=2), run(rollouts=5)
    assert first.trace[0]["proposal_seeds"] == second.trace[0]["proposal_seeds"]
    old = {event["event_id"]: event["seed"] for event in first.trace[0]["events"]}
    new = {event["event_id"]: event["seed"] for event in second.trace[0]["events"]}
    assert all(new[key] == value for key, value in old.items())


@pytest.mark.parametrize("model", ["toy", "real_adapter"])
def test_resuming_base_suffix_reproduces_uninterrupted_rng_and_graph(adapter, model):
    current = SmallAdapter() if model == "toy" else adapter
    current.steps = 20
    rng = current.new_rng(10)
    state = current.initial_state(rng)
    original = deepcopy(state)
    whole = complete_base(current, [state], [rng], 20)[0]
    rng = current.new_rng(10)
    state = current.initial_state(rng)
    prefix = advance_base(current, [state], [rng], 20, 4)[0]
    resumed = complete_base(current, [prefix], [rng], 4)[0]
    assert whole.raw_nodes == resumed.raw_nodes
    assert whole.nodes == resumed.nodes
    assert whole.edges == resumed.edges
    np.testing.assert_array_equal(np.asarray(state), np.asarray(original))


def test_lookahead_suffixes_do_not_mutate_candidate_states():
    adapter = SmallAdapter()
    state = np.array([2.0])
    complete_base(adapter, [state, state], [adapter.new_rng(1), adapter.new_rng(2)], 4)
    np.testing.assert_array_equal(state, [2.0])


def test_policy_and_unexpected_checker_errors_fail_loudly():
    class BrokenEvaluator:
        def evaluate(self, sample, scenario):
            raise RuntimeError("checker bug")

    with pytest.raises(RuntimeError, match="checker bug"):
        run(evaluator=BrokenEvaluator())
    scenario = Scenario()

    def change_policy(sample, evaluation, event):
        scenario.ceiling_height_m = 3.5

    with pytest.raises(RuntimeError, match="policy changed"):
        hard_h_mc(SmallAdapter(), FixedEvaluator(), scenario, 42,
                  **PARAMETERS, event_callback=change_policy)


@pytest.mark.parametrize("parameters", [
    {"k": 0}, {"rollouts": 0}, {"terminal_k": 0}, {"anchor_remaining": (5, 2)},
    {"anchor_remaining": (4, 4)}, {"anchor_remaining": ()},
    {"anchor_remaining": (4, 0)}, {"zero_policy": "softmax"}, {"max_nfe": 0},
])
def test_invalid_configuration_is_rejected_before_sampling(parameters):
    adapter = SmallAdapter()
    with pytest.raises(ValueError):
        run(adapter, **parameters)
    assert adapter.counters["node_samples"] == 0
