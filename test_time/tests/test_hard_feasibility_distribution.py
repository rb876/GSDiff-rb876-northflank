"""Compare a complete finite-bank episode with its enumerated probability law."""

from itertools import product
from math import comb

import numpy as np

from test_time.hard_feasibility import hard_h_mc
from test_time.types import ConstraintResult, FloorplanSample, PolicyEvaluation, Scenario


INITIAL = np.array([0.2, 0.8])
TRANSITION = np.array([[0.9, 0.1], [0.4, 0.6]])
# Two feasible graph outcomes and one infeasible outcome; decoding is stochastic.
DECODER = np.array([[0.85, 0.05, 0.10], [0.02, 0.08, 0.90]])
ROLLOUTS = 4
TERMINAL_K = 2


class FiniteChain:
    steps = 5

    def __init__(self):
        self.counters = {"node_samples": 0, "edge_samples": 0, "graph_completions": 0}
        self.anchor_states = []

    def new_rng(self, seed):
        return np.random.default_rng(seed)

    def initial_state(self, rng):
        return int(rng.choice(2, p=INITIAL))

    def predict_x0_batch(self, states, step, batch_size):
        self.counters["node_samples"] += len(states)
        return states

    def sample_transition(self, state, step, rng, x0):
        if step == 1:
            state = int(rng.choice(2, p=TRANSITION[state]))
            self.anchor_states.append(state)
        return state

    def complete_graph_batch(self, states, rngs, batch_size):
        self.counters["edge_samples"] += len(states)
        self.counters["graph_completions"] += len(states)
        return [FloorplanSample(metadata={"outcome": int(rng.choice(3, p=DECODER[state]))})
                for state, rng in zip(states, rngs)]


class BinaryOracle:
    def evaluate(self, sample, scenario):
        passed = sample.metadata["outcome"] < 2
        check = ConstraintResult("toy", "pass" if passed else "fail", float(not passed),
                                 True, True, "finite-chain")
        return PolicyEvaluation(float(not passed), -float(not passed), [check], True, 0, 1, 0, 0)


def enumerated_bank_law(marginal, h):
    """Unnormalized retained-state mass, summing both proposals and binomial counts.

    The omitted mass is the all-zero/abort event. This is finite-bank importance
    resampling, not the ideal marginal*h reweighting.
    """
    retained = np.zeros(2)
    for states in product(range(2), repeat=2):
        for counts in product(range(ROLLOUTS + 1), repeat=2):
            total = sum(counts)
            if not total:
                continue
            mass = np.prod([
                marginal[z] * comb(ROLLOUTS, s) * h[z] ** s * (1 - h[z]) ** (ROLLOUTS - s)
                for z, s in zip(states, counts)
            ])
            for z, s in zip(states, counts):
                retained[z] += mass * s / total
    return retained


def test_complete_finite_chain_matches_enumerated_bank_law():
    # Backward values and the first-anchor marginal are known exactly.
    h = DECODER[:, :2].sum(axis=1)
    marginal = INITIAL @ TRANSITION
    retained = enumerated_bank_law(marginal, h)
    terminal_success = 1 - (1 - h) ** TERMINAL_K
    outputs = (retained * terminal_success) @ (DECODER[:, :2] / h[:, None])
    expected = np.r_[outputs, retained @ (1 - terminal_success), 1 - retained.sum()]
    np.testing.assert_allclose(expected.sum(), 1)

    # The finite terminal cap and proposal bank do not preserve exact p(G|F).
    base_outputs = (marginal @ DECODER)[:2]
    exact_conditional = base_outputs / base_outputs.sum()
    assert abs(outputs[0] / outputs.sum() - exact_conditional[0]) > 0.02

    episodes = 1500
    observed = np.zeros(4)  # feasible A, feasible B, terminal failure, root abort
    selected = np.zeros(2)
    for seed in range(episodes):
        chain = FiniteChain()
        result = hard_h_mc(chain, BinaryOracle(), Scenario(), seed, k=2, rollouts=ROLLOUTS,
                           anchor_remaining=(1,), terminal_k=TERMINAL_K, zero_policy="abort")
        chosen = result.trace[0]["selected_index"]
        if chosen is None:
            assert result.status == "no_positive_h_estimate"
            observed[3] += 1
        else:
            selected[chain.anchor_states[chosen]] += 1
            if result.status == "success":
                outcome = result.sample.metadata["outcome"]
                assert outcome < 2
                observed[outcome] += 1
            else:
                assert result.status == "no_feasible_terminal_sample"
                observed[2] += 1
        # All stochastic graph draws are accounted for; no extra terminal decode.
        assert chain.counters["edge_samples"] == 2 * ROLLOUTS + (TERMINAL_K if chosen is not None else 0)

    # Union-bound Hoeffding tolerance: six empirical frequencies, <=0.5% bound.
    tolerance = np.sqrt(np.log(12 / 0.005) / (2 * episodes))
    np.testing.assert_allclose(selected / episodes, retained, atol=tolerance, rtol=0)
    np.testing.assert_allclose(observed / episodes, expected, atol=tolerance, rtol=0)
