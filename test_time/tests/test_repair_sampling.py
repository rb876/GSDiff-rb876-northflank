"""Repair guidance keeps the diffusion state and accounts for all neural work."""

from copy import deepcopy
from dataclasses import asdict
import math
from types import SimpleNamespace

import pytest
import torch

from test_time import repair_sampling
from test_time.types import ConstraintResult, FloorplanSample, PolicyEvaluation, Scenario


class Adapter:
    steps = 10
    device = torch.device('cpu')
    def __init__(self):
        self.counters = dict.fromkeys(('node_calls', 'node_samples', 'edge_calls', 'edge_samples', 'graph_completions'), 0)
        self.transitions = []
        self.final_export = False
    def new_rng(self, seed):
        return torch.Generator().manual_seed(seed)
    def initial_state(self, rng):
        return torch.randn((1, 3, 2), generator=rng, dtype=torch.float64)
    def predict_x0_batch(self, states, step, batch_size=10):
        self.counters['node_calls'] += math.ceil(len(states) / batch_size)
        self.counters['node_samples'] += len(states)
        return [state / 2 + 5 for state in states]
    def sample_transition(self, state, step, rng, x0=None):
        noise = torch.randn(state.shape, dtype=state.dtype, generator=rng)
        result = x0 if step == 0 else state / 2 + noise * .1
        self.transitions.append((step, state.clone(), result.clone()))
        return result
    def predict_graph_tensors(self, states, rngs, batch_size=10):
        self.counters['edge_calls'] += math.ceil(len(states) / batch_size)
        self.counters['edge_samples'] += len(states)
        self.counters['graph_completions'] += len(states)
        return dict(xy=torch.cat(states), semantics=torch.zeros((len(states), 3, 7)),
                    node_mask=torch.ones((len(states), 3), dtype=torch.bool),
                    adjacency=torch.zeros((len(states), 3, 3), dtype=torch.bool))
    def complete_graph_batch(self, states, rngs, batch_size=10):
        self.final_export = True
        graph = self.predict_graph_tensors(states, rngs, batch_size)
        return [FloorplanSample(nodes=xy.tolist(), raw_nodes=state[0].tolist())
                for xy, state in zip(graph['xy'], states)]


def evaluation(sample):
    check = ConstraintResult('C035', 'fail', 1., True, True, 'test', {}, {}, [], 'synthetic')
    return PolicyEvaluation(1., -1., [check], True, 0, 1, 0, 0., measurable_count=1)


class Audit:
    def __init__(self, adapter):
        self.adapter = adapter
        self.calls = 0
    def evaluate(self, sample, scenario):
        assert self.adapter.final_export, 'CPU geometry must not run inside guidance'
        self.calls += 1
        return evaluation(sample)


@pytest.fixture
def proxy(monkeypatch):
    """Strongly favour last proposal while giving it a wildly edited clean graph."""
    calls = []
    def score(graphs, policy, scenario, config, max_evaluations, batch_size):
        calls.append(graphs)
        count = len(graphs['xy'])
        values = dict(energy=torch.arange(count - 1, -1, -1).double(),
                      status=torch.ones((count, 1), dtype=torch.long),
                      applicable=torch.ones((count, 1), dtype=torch.bool),
                      structurally_valid=torch.ones(count, dtype=torch.bool),
                      unknown_count=torch.zeros(count, dtype=torch.long),
                      measurable_count=torch.ones(count, dtype=torch.long),
                      applicable_count=torch.ones(count, dtype=torch.long))
        changed = deepcopy(graphs)
        changed['xy'] = changed['xy'] + 1000
        return dict(scores=values, owner=torch.arange(count), candidate=torch.zeros(count, dtype=torch.long),
                    selected=torch.arange(count), distortion=torch.zeros(count), graph=changed,
                    evaluations=count, displacement_m=torch.zeros((count, 3)))
    monkeypatch.setattr(repair_sampling, 'score_repairs', score)
    return calls


@pytest.mark.parametrize('microbatch', [1, 4])
def test_last_steps_only_keep_original_noisy_proposal_and_cpu_audit_final(monkeypatch, proxy, microbatch):
    adapter = Adapter()
    audit = Audit(adapter)
    policy = SimpleNamespace(specs=[SimpleNamespace(id='C035', counts_toward_compliance=True)])
    scenario = Scenario(coordinate_scale_m_per_model_unit=.025, scale_provenance='assumed')
    result = repair_sampling.repair_aware_svdd(adapter, audit, scenario, 42, k=4, guided_steps=2,
                inference_batch_size=microbatch, repair_config={'final_enabled': False}, tensor_evaluator=policy)
    assert [row['timestep'] for row in result.trace] == [2, 1]
    assert [row['selected_index'] for row in result.trace] == [3, 3]
    proposals = [row[2] for row in adapter.transitions if row[0] == 1]
    final_input = next(row[1] for row in adapter.transitions if row[0] == 0)
    torch.testing.assert_close(final_input, proposals[3], rtol=0, atol=0)
    assert result.stats['node_samples'] == adapter.steps + 4 * 2
    assert result.stats['edge_samples'] == 4 * 2 + 1
    assert result.stats['gpu_candidate_evaluations'] == 8
    assert len(proxy) == 2
    assert audit.calls == 2
    assert 'pre_repair' in result.extra_samples
    assert result.status == 'infeasible'


@pytest.mark.parametrize('settings', [dict(k=1, guided_steps=2), dict(k=4, guided_steps=0)])
def test_disabled_guidance_matches_ordinary_rng_and_state(proxy, settings):
    seed = 91
    reference = Adapter()
    rng = reference.new_rng(seed)
    state = reference.initial_state(rng)
    for step in range(reference.steps - 1, -1, -1):
        clean = reference.predict_x0_batch([state], step)[0]
        state = reference.sample_transition(state, step, rng, x0=clean)
    expected = reference.complete_graph_batch([state], [rng])[0]
    actual = Adapter()
    result = repair_sampling.repair_aware_svdd(actual, Audit(actual),
        Scenario(coordinate_scale_m_per_model_unit=.025, scale_provenance='assumed'), seed,
        tensor_evaluator=SimpleNamespace(specs=[]), repair_config={'final_enabled': False}, **settings)
    assert result.sample.nodes == expected.nodes
    assert result.sample.raw_nodes == expected.raw_nodes
    assert result.stats['node_samples'] == reference.steps
    assert not result.trace
    assert not proxy


def test_final_repair_is_applied_without_injecting_it_into_diffusion(monkeypatch, proxy):
    adapter = Adapter()
    def repair(adapter, sample, evaluator, scenario, tensor_evaluator, config, batch_size):
        edited = deepcopy(sample)
        edited.nodes[0][0] += .1
        return edited, evaluation(edited), dict(gpu_evaluations=2, cpu_evaluations=2, edits=[{'node': 0}], search_status='improved')
    monkeypatch.setattr(repair_sampling, 'final_repair', repair)
    policy = SimpleNamespace(specs=[SimpleNamespace(id='C035', counts_toward_compliance=True)])
    result = repair_sampling.repair_aware_svdd(adapter, Audit(adapter),
        Scenario(coordinate_scale_m_per_model_unit=.025, scale_provenance='assumed'), 1,
        k=2, guided_steps=2, tensor_evaluator=policy)
    original = result.extra_samples['pre_repair'][0]
    assert result.sample.nodes[0][0] == pytest.approx(original.nodes[0][0] + .1)
    assert result.stats['cpu_candidate_evaluations'] == 3
    assert result.stats['final_repair']['edits'] == [{'node': 0}]


def test_invalid_repair_budget_fails_before_neural_inference():
    adapter = Adapter()
    with pytest.raises(ValueError, match='max_displacement'):
        repair_sampling.repair_aware_svdd(adapter, Audit(adapter),
            Scenario(coordinate_scale_m_per_model_unit=.025, scale_provenance='assumed'), 1,
            guided_steps=2, repair_config={'max_displacement_m': -1})
    assert not any(adapter.counters.values())


def test_no_scorable_repair_falls_back_to_first_noisy_proposal(monkeypatch, proxy):
    original = repair_sampling.score_repairs
    def invalid(*args, **kwargs):
        bank = original(*args, **kwargs)
        bank['scores']['structurally_valid'].fill_(False)
        return bank
    monkeypatch.setattr(repair_sampling, 'score_repairs', invalid)
    adapter = Adapter()
    result = repair_sampling.repair_aware_svdd(adapter, Audit(adapter),
        Scenario(coordinate_scale_m_per_model_unit=.025, scale_provenance='assumed'), 42,
        guided_steps=2, tensor_evaluator=SimpleNamespace(specs=[SimpleNamespace(id='C035', counts_toward_compliance=True)]),
        repair_config={'final_enabled': False})
    assert result.stats['fallback_count'] == 2
    assert all(row['selected_index'] == 0 for row in result.trace)
    assert all(row['fallback'] == 'ordinary_proposal_zero' for row in result.trace)


def test_guidance_cannot_escape_last_twenty_percent():
    adapter = Adapter()
    with pytest.raises(ValueError, match='final 20%'):
        repair_sampling.repair_aware_svdd(adapter, Audit(adapter),
            Scenario(coordinate_scale_m_per_model_unit=.025, scale_provenance='assumed'), 42, guided_steps=3)
    assert not any(adapter.counters.values())
