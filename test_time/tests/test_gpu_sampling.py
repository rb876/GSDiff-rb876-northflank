"""Synthetic tensor tests: no pretrained generation or regulatory claims."""

import math
from types import SimpleNamespace

import pytest
import torch

from test_time import gpu_sampling
from test_time.sampling import baseline_batch
from test_time.types import FloorplanSample, PolicyEvaluation, Scenario


class TensorAdapter:
    steps = 4
    device = torch.device("cpu")

    def __init__(self):
        self.counters = dict.fromkeys(("node_calls", "node_samples", "edge_calls", "edge_samples", "graph_completions"), 0)
        self.transitions = []
        self.final_export = False
        self.final_steps = 0

    def new_rng(self, seed):
        return torch.Generator().manual_seed(seed)

    def initial_state(self, rng):
        return torch.randn((1, 3, 2), generator=rng, dtype=torch.float64)

    def predict_x0_batch(self, states, t, batch_size=10):
        self.counters["node_calls"] += math.ceil(len(states) / batch_size)
        self.counters["node_samples"] += len(states)
        return [state / 2 + 5 for state in states]

    def sample_transition(self, state, t, rng, x0=None):
        noise = torch.randn(state.shape, dtype=state.dtype, generator=rng)
        result = x0 if t == 0 else state / 2 + 0.1 * noise
        self.transitions.append((t, state.clone(), result.clone()))
        self.final_steps += int(t == 0)
        return result

    def predict_graph_tensors(self, states, rngs, batch_size=10):
        self.counters["edge_calls"] += math.ceil(len(states) / batch_size)
        self.counters["edge_samples"] += len(states)
        self.counters["graph_completions"] += len(states)
        noise = torch.stack([torch.rand((), generator=torch.Generator().set_state(rng.get_state())) for rng in rngs])
        xy = torch.cat(states) + 0.01 * noise[:, None, None]
        return {"xy": xy, "semantics": torch.zeros((len(states), 3, 7)),
                "node_mask": torch.ones((len(states), 3), dtype=torch.bool),
                "adjacency": torch.zeros((len(states), 3, 3), dtype=torch.bool)}

    def complete_graph_batch(self, states, rngs, batch_size=10):
        self.final_export = True
        graph = self.predict_graph_tensors(states, rngs, batch_size)
        return [FloorplanSample(nodes=xy.tolist(), raw_nodes=state[0].tolist())
                for xy, state in zip(graph["xy"], states)]


class CpuAudit:
    def __init__(self, adapter, reverse_energy=False):
        self.adapter = adapter
        self.calls = 0
        self.reverse_energy = reverse_energy

    def evaluate(self, sample, scenario):
        assert self.adapter.final_export, "CPU checker called during guidance"
        self.calls += 1
        energy = abs(sample.nodes[0][0])
        if self.reverse_energy:
            energy = 100 - energy
        return PolicyEvaluation(energy, -energy, [], True, 0, 2, 0, 0, measurable_count=2)


class TensorPolicy:
    ids = ["C033", "C036", "C034"]
    specs = [SimpleNamespace(id=id, counts_toward_compliance=(id != "C034"), assumptions=[])
             for id in ids]
    metadata = {"backend": "synthetic_test", "cpu_oracle_equivalence": False}

    def __init__(self, mode="valid"):
        self.mode = mode
        self.sizes = []

    def evaluate(self, rooms, scenario):
        energy = rooms["value"].abs()
        self.sizes.append(len(energy))
        status = torch.zeros((len(energy), 3), dtype=torch.long)
        if self.mode == "unknown":
            status.fill_(2)
        measurable = status < 2
        return {"energy": energy, "reward": -energy, "status": status,
                "severity": torch.where(measurable, torch.zeros_like(status, dtype=torch.float64), torch.nan),
                "applicable": torch.ones_like(status, dtype=torch.bool), "measurable": measurable,
                "structurally_valid": torch.full_like(energy, self.mode != "invalid", dtype=torch.bool),
                "unknown_count": (status[:, :2] == 2).sum(dim=1),
                "measurable_count": measurable[:, :2].sum(dim=1), "diagnostics": rooms["diagnostics"],
                "geometry_unused_large_array": torch.zeros((len(energy), 20, 10))}


@pytest.fixture(autouse=True)
def fake_room_extraction(monkeypatch):
    monkeypatch.setattr(gpu_sampling, "_extract_rooms", lambda graph:
                        {"value": graph["xy"][:, 0, 0],
                         "diagnostics": torch.zeros(len(graph["xy"]), dtype=torch.long)})


def score_table(energies, statuses, valid=None):
    energy = torch.tensor(energies, dtype=torch.float64)
    status = torch.tensor(statuses, dtype=torch.long)
    measurable = status < 2
    return {"energy": energy, "reward": -energy, "status": status,
            "applicable": status != 3, "measurable": measurable,
            "structurally_valid": torch.tensor(valid if valid is not None else [True] * len(energy)),
            "unknown_count": (status[:, :2] == 2).sum(dim=1),
            "measurable_count": measurable[:, :2].sum(dim=1)}


def test_gpu_selection_preserves_unknown_ids_including_auxiliary_and_best_of_n_seed_ties():
    swapped = score_table([1, 0], [[1, 2, 0], [2, 0, 0]])
    selected = gpu_sampling.tensor_selection(swapped, 1, 2)
    assert selected["chosen"].tolist() == [0]
    assert selected["eligible"].tolist() == [[True, False]]
    auxiliary = score_table([1, 0], [[1, 0, 0], [0, 0, 2]])
    assert gpu_sampling.tensor_selection(auxiliary, 1, 2)["chosen"].tolist() == [0]
    ties = score_table([1, 1], [[1, 0, 0], [1, 0, 0]], valid=[False, False])
    chosen = gpu_sampling.tensor_selection(ties, 1, 2, tie_seeds=torch.tensor([[99, 2]]))
    assert chosen["chosen"].tolist() == [1]
    assert not chosen["fallback"].any()


def test_gpu_categorical_selection_is_stable_and_never_selects_excluded_proposals():
    values = score_table([1e300, 1e300, 0], [[1, 0, 0]] * 3, valid=[True, True, False])
    a = gpu_sampling.tensor_selection(values, 1, 3, temperature=1e-100, uniforms=torch.tensor([0.1]))
    b = gpu_sampling.tensor_selection(values, 1, 3, temperature=1e-100, uniforms=torch.tensor([0.9]))
    assert a["chosen"].tolist() == [0]
    assert b["chosen"].tolist() == [1]


@pytest.mark.parametrize("temperature", [0, 0.1])
def test_tensor_guidance_keeps_cpu_audit_final_only_and_preserves_microbatch_results(temperature):
    outputs = []
    for microbatch in [1, 2, 10]:
        adapter, policy = TensorAdapter(), TensorPolicy()
        audit = CpuAudit(adapter)
        progress = []
        results, stats = gpu_sampling.svdd_pm_batch_gpu(
            adapter, audit, Scenario(), [42, 43, 44], k=3, guided_steps=2, temperature=temperature,
            inference_batch_size=microbatch, tensor_evaluator=policy,
            progress_callback=lambda current, total: progress.append((current, total)))
        outputs.append([(result.candidates[0].sample.to_dict(), result.trace) for result in results])
        assert audit.calls == 3
        assert stats["gpu_candidate_evaluations"] == 3 * (3 * 2 + 1)
        assert stats["cpu_candidate_evaluations"] == 3
        assert stats["candidate_evaluations"] == stats["graph_completions"] == 24
        assert max(policy.sizes) <= microbatch
        assert progress == [(1, 2), (2, 2)]
        for result in results:
            assert result.trace[0]["format"] == gpu_sampling.TRACE_FORMAT
            assert "sample" not in result.trace[0]["candidates"][0]
            metadata = result.candidates[0].sample.metadata
            assert metadata["guidance_device"] == "cpu"
            assert metadata["guidance_evaluation"]["results"][2]["counts_toward_compliance"] is False
        # t=0 must consume the chosen noisy proposal, never its x0 prediction.
        last_proposals = [record[2] for record in adapter.transitions if record[0] == 1]
        final_inputs = [record[1] for record in adapter.transitions if record[0] == 0]
        for trial, result in enumerate(results):
            chosen = result.trace[-1]["selected_index"]
            torch.testing.assert_close(final_inputs[trial], last_proposals[trial * 3 + chosen], rtol=0, atol=0)
    assert outputs[0] == outputs[1] == outputs[2]


def test_guidance_scores_do_not_transfer_to_cpu_before_reverse_loop_finishes(monkeypatch):
    adapter = TensorAdapter()
    original_cpu, original_tolist = torch.Tensor.cpu, torch.Tensor.tolist

    def checked_cpu(tensor, *args, **kwargs):
        assert adapter.final_steps == 2
        return original_cpu(tensor, *args, **kwargs)

    def checked_tolist(tensor, *args, **kwargs):
        assert adapter.final_steps == 2
        return original_tolist(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "cpu", checked_cpu)
    monkeypatch.setattr(torch.Tensor, "tolist", checked_tolist)
    results, _ = gpu_sampling.svdd_pm_batch_gpu(adapter, CpuAudit(adapter), Scenario(), [42, 43],
                                                k=2, guided_steps=2, tensor_evaluator=TensorPolicy())
    assert len(results) == 2


@pytest.mark.parametrize("mode", ["invalid", "unknown"])
def test_all_unscorable_tensor_proposals_take_ordinary_first_proposal(mode):
    adapter = TensorAdapter()
    results, stats = gpu_sampling.svdd_pm_batch_gpu(adapter, CpuAudit(adapter), Scenario(), [42, 43],
                                                    k=3, guided_steps=2, tensor_evaluator=TensorPolicy(mode))
    assert stats["fallback_count"] == 4
    assert all(event["selected_index"] == 0 and event["fallback"] for result in results for event in result.trace)


@pytest.mark.parametrize("k,guided_steps", [(1, 2), (4, 0)])
def test_k1_and_disabled_guidance_match_ordinary_generation(k, guided_steps):
    adapter, reference_adapter = TensorAdapter(), TensorAdapter()
    seeds = [42, 43]
    reference, _ = baseline_batch(reference_adapter, CpuAudit(reference_adapter), Scenario(), seeds)
    policy = TensorPolicy()
    results, stats = gpu_sampling.svdd_pm_batch_gpu(adapter, CpuAudit(adapter), Scenario(), seeds,
                                                    k=k, guided_steps=guided_steps, tensor_evaluator=policy)
    assert [result.candidates[0].sample.raw_nodes for result in results] == [result.candidates[0].sample.raw_nodes for result in reference]
    assert [result.candidates[0].sample.nodes for result in results] == [result.candidates[0].sample.nodes for result in reference]
    assert not policy.sizes
    assert stats["gpu_candidate_evaluations"] == 0
    assert stats["cpu_candidate_evaluations"] == 2


def test_best_of_n_uses_gpu_selection_and_preserves_every_cpu_audit():
    adapter, policy = TensorAdapter(), TensorPolicy()
    audit = CpuAudit(adapter, reverse_energy=True)
    results, stats = gpu_sampling.best_of_n_batch_gpu(adapter, audit, Scenario(), [42, 43],
                                                      n=3, inference_batch_size=2, tensor_evaluator=policy)
    assert audit.calls == 6
    assert stats["gpu_candidate_evaluations"] == stats["cpu_candidate_evaluations"] == 6
    assert stats["candidate_evaluations"] == stats["graph_completions"] == 12
    for result in results:
        energies = [candidate.sample.metadata["guidance_evaluation"]["energy"] for candidate in result.candidates]
        assert result.selected_index == min(range(3), key=lambda index: energies[index])
        assert result.selected_index != min(range(3), key=lambda index: result.candidates[index].evaluation.energy)
        assert len(result.candidates) == 3
        assert all(candidate.sample.metadata["selection_backend"] == "tensor_gpu" for candidate in result.candidates)


@pytest.mark.parametrize("version", [1, 2])
def test_real_tensor_room_and_constraint_layers_work_through_guided_sampler(monkeypatch, version):
    from pathlib import Path
    from test_time.constraints import load_specs
    from test_time.gpu_constraints import TensorConstraintEvaluator
    from test_time.gpu_graph import extract_rooms

    class RectangleAdapter(TensorAdapter):
        def predict_graph_tensors(self, states, rngs, batch_size=10):
            super().predict_graph_tensors(states, rngs, batch_size)
            count = len(states)
            xy = torch.tensor([[0, 0], [4, 0], [4, 4], [0, 4]], dtype=torch.float64)
            semantics = torch.zeros((count, 4, 7))
            semantics[:, :, 1] = 1
            walls = torch.tensor([[0, 1, 0, 1], [1, 0, 1, 0],
                                  [0, 1, 0, 1], [1, 0, 1, 0]], dtype=torch.bool)
            return {"xy": xy[None].expand(count, -1, -1), "semantics": semantics,
                    "node_mask": torch.ones((count, 4), dtype=torch.bool),
                    "adjacency": walls[None].expand(count, -1, -1)}

    monkeypatch.setattr(gpu_sampling, "_extract_rooms", lambda graph: extract_rooms(**graph))
    adapter = RectangleAdapter()
    registry = Path(__file__).parents[1] / "configs" / f"constraints_v{version}.json"
    policy = TensorConstraintEvaluator(load_specs(registry), max_positions=64)
    audit = CpuAudit(adapter)
    scenario = Scenario(coordinate_scale_m_per_model_unit=1, scale_provenance="assumed")
    threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        results, stats = gpu_sampling.svdd_pm_batch_gpu(
            adapter, audit, scenario, [42, 43], k=2, guided_steps=1,
            inference_batch_size=3, tensor_evaluator=policy)
    finally:
        torch.set_num_threads(threads)
    assert audit.calls == 2
    assert stats["gpu_candidate_evaluations"] == 6
    assert stats["cpu_candidate_evaluations"] == 2
    assert stats["fallback_count"] == 0
    for result in results:
        metadata = result.candidates[0].sample.metadata
        score = metadata["guidance_evaluation"]
        assert score["structurally_valid"]
        assert score["applicable_count"] == 6
        assert score["diagnostic_bits"] == 0
        assert [item["id"] for item in score["results"]] == policy.ids
        assert metadata["guidance_backend_settings"]["backend"] == "torch_finite_grid_v1"
        assert result.trace[0]["selected_index"] == 0
        assert len(result.trace[0]["candidates"][0]["evaluation"]["results"]) == (7 if version == 1 else 9)
