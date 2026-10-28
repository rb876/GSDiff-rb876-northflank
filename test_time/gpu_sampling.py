"""Tensor rewards and selection, with the CPU checker used only for final audits.

GPU guidance is an explicit alternative objective backend. Compact traces retain
scores and decisions, not complete surrogate graphs. The reference sampler lives
in sampling.py and remains available for parity and audit work.
"""

import math

import torch

from .sampling import (_batch_inputs, _candidate_seed, _finish_batch, _scored_sample,
                       _start, _svdd_parameters, baseline_batch)
from .types import SamplingResult


STATUS_NAMES = ("pass", "fail", "unknown", "not_applicable")
TRACE_FORMAT = "tensor_gpu_compact_v1"
SCORE_FIELDS = ("energy", "reward", "status", "severity", "applicable", "measurable",
                "structurally_valid", "unknown_count", "measurable_count", "applicable_count",
                "assumption_based_count", "validity_penalty", "diagnostics")


def _evaluator(cpu_evaluator, tensor_evaluator):
    if tensor_evaluator is None:
        from .gpu_constraints import TensorConstraintEvaluator
        tensor_evaluator = TensorConstraintEvaluator(cpu_evaluator.specs, cpu_evaluator.validity_weight)
    return tensor_evaluator


def _extract_rooms(graph):
    from .gpu_graph import extract_rooms
    return extract_rooms(**graph)


def _score_states(adapter, evaluator, scenario, states, rngs, batch_size):
    """Bound graph extraction and geometry memory as well as neural forwards."""
    batches = []
    for start in range(0, len(states), batch_size):
        graph = adapter.predict_graph_tensors(states[start:start + batch_size],
                                              rngs[start:start + batch_size], batch_size=batch_size)
        rooms = _extract_rooms(graph)
        output = evaluator.evaluate(rooms, scenario)
        evaluation = {name: output[name] for name in SCORE_FIELDS if name in output}
        evaluation.setdefault("diagnostics", rooms["diagnostics"])
        batches.append(evaluation)
    return {name: torch.cat([batch[name] for batch in batches], dim=0) for name in batches[0]}


def tensor_selection(evaluation, batch_size, k, temperature=0.0, uniforms=None, tie_seeds=None):
    """GPU validity/unknown-ID guard and categorical or greedy selection.

    Supplying tie_seeds selects complete trajectories (Best-of-N), including an
    invalid fallback if necessary. Otherwise only valid, measurable PM proposals
    are scorable, with ordinary proposal zero as the all-unscorable fallback.
    """
    reward = evaluation["reward"].reshape(batch_size, k)
    valid = evaluation["structurally_valid"].reshape(batch_size, k)
    unknown_count = evaluation["unknown_count"].reshape(batch_size, k)
    statuses = evaluation["status"].reshape(batch_size, k, -1)
    unknown = (statuses == 2) & evaluation["applicable"].reshape_as(statuses)
    if tie_seeds is None:
        pool = valid & (evaluation["measurable_count"].reshape(batch_size, k) > 0) & torch.isfinite(reward)
        ordinary = torch.zeros(batch_size, dtype=torch.long, device=reward.device)
    else:
        validity_class = torch.where(valid.any(dim=1, keepdim=True), valid, torch.ones_like(valid))
        ordinary = validity_class.long().argmax(dim=1)
        pool = validity_class & torch.isfinite(reward)
    fallback = ~pool.any(dim=1)
    anchor = pool.long().argmax(dim=1)
    allowed = unknown.gather(1, anchor[:, None, None].expand(-1, 1, unknown.shape[-1]))
    covered = pool & ~(unknown & ~allowed).any(dim=2)
    minimum = unknown_count.masked_fill(~covered, statuses.shape[-1] + 1).amin(dim=1, keepdim=True)
    eligible = covered & (unknown_count == minimum)
    masked_reward = reward.masked_fill(~eligible, -torch.inf)
    if temperature == 0:
        chosen = masked_reward.argmax(dim=1)
        if tie_seeds is not None:
            ties = eligible & (masked_reward == masked_reward.amax(dim=1, keepdim=True))
            chosen = tie_seeds.masked_fill(~ties, torch.iinfo(torch.long).max).argmin(dim=1)
    else:
        maximum = masked_reward.amax(dim=1, keepdim=True)
        maximum = torch.where(fallback[:, None], torch.zeros_like(maximum), maximum)
        weights = torch.exp((masked_reward - maximum) / temperature)
        probabilities = weights / weights.sum(dim=1, keepdim=True).clamp_min(torch.finfo(weights.dtype).tiny)
        cumulative = probabilities.cumsum(dim=1)
        positions = torch.arange(k, device=reward.device)[None, :]
        last = positions.expand_as(eligible).masked_fill(~eligible, -1).amax(dim=1, keepdim=True)
        # Guarantee a terminal CDF of one even with floating-point accumulation.
        cumulative = torch.where(positions >= last, torch.ones_like(cumulative), cumulative)
        chosen = (uniforms[:, None] < cumulative).long().argmax(dim=1)
    return {"chosen": torch.where(fallback, ordinary, chosen), "eligible": eligible,
            "anchor": anchor.masked_fill(fallback, -1), "fallback": fallback}


def _export_records(records):
    """One score/trace transfer after generation; no hot-loop scalar reads."""
    layout = [[(name, value.shape, value.dtype, value.numel()) for name, value in record.items()]
              for record in records]
    packed = torch.cat([value.reshape(-1).to(torch.float64) for record in records for value in record.values()]).cpu()
    output, offset = [], 0
    for fields in layout:
        record = {}
        for name, shape, dtype, size in fields:
            record[name] = packed[offset:offset + size].reshape(shape).to(dtype)
            offset += size
        output.append(record)
    return output


def _finite(value):
    value = float(value)
    return value if math.isfinite(value) else None


def _evaluation_record(values, index, evaluator):
    results = []
    for column, spec in enumerate(evaluator.specs):
        status = int(values["status"][index, column])
        result = {"id": spec.id, "status": STATUS_NAMES[status],
                  "applicable": bool(values["applicable"][index, column]),
                  "measurable": bool(values["measurable"][index, column]),
                  "counts_toward_compliance": spec.counts_toward_compliance,
                  "assumption_flags": [*spec.assumptions, "tensor_gpu_finite_grid"]}
        if "severity" in values:
            result["severity"] = _finite(values["severity"][index, column]) if result["measurable"] else None
        results.append(result)
    evaluation = {"energy": _finite(values["energy"][index]), "reward": _finite(values["reward"][index]),
                  "results": results, "structurally_valid": bool(values["structurally_valid"][index]),
                  "unknown_count": int(values["unknown_count"][index]),
                  "measurable_count": int(values["measurable_count"][index]),
                  "diagnostic_bits": int(values["diagnostics"][index]),
                  "assumption_note": "Finite-grid tensor guidance under the supplied scenario; final reported checks use the CPU reference."}
    for name in ("applicable_count", "assumption_based_count"):
        if name in values:
            evaluation[name] = int(values[name][index])
    if "validity_penalty" in values:
        evaluation["validity_penalty"] = _finite(values["validity_penalty"][index])
    return evaluation


def _backend_metadata(candidate, evaluator, device, guidance_evaluation):
    candidate.sample.metadata.update({"guidance_backend": "tensor_gpu", "guidance_device": str(device),
                                      "guidance_backend_settings": dict(evaluator.metadata),
                                      "guidance_evaluation": guidance_evaluation,
                                      "final_evaluation_backend": "cpu_reference",
                                      "selection_rng_policy": "Independent per-seed PyTorch uniforms; differs from the CPU NumPy categorical stream.",
                                      "surrogate_trace_format": TRACE_FORMAT})


def _finish(adapter, started, counts, results, inference_batch_size):
    results, stats = _finish_batch(adapter, started, counts, results, inference_batch_size)
    for key in ("gpu_candidate_evaluations", "cpu_candidate_evaluations"):
        stats[key] = sum(result.stats[key] for result in results)
    stats["guidance_backend"] = "tensor_gpu"
    return results, stats


@torch.no_grad()
def svdd_pm_batch_gpu(adapter, cpu_evaluator, scenario, seeds, k=10, guided_steps=999,
                      temperature=0.0, inference_batch_size=10, tensor_evaluator=None,
                      progress_callback=None):
    seeds = _batch_inputs(seeds, inference_batch_size)
    parameters = _svdd_parameters(adapter, k, guided_steps, temperature)
    if k == 1 or guided_steps == 0:
        results, stats = baseline_batch(adapter, cpu_evaluator, scenario, seeds, inference_batch_size)
        for result in results:
            result.candidates[0].sample.metadata.update({"method": "svdd_pm", "method_parameters": parameters,
                                                        "guidance_backend": "tensor_gpu", "guidance_applied": False,
                                                        "guidance_device": str(adapter.device),
                                                        "final_evaluation_backend": "cpu_reference"})
            result.stats.update({"guided_steps_applied": 0, "fallback_count": 0,
                                 "gpu_candidate_evaluations": 0, "cpu_candidate_evaluations": 1})
            result.candidates[0].stats.update(result.stats)
        stats.update({"gpu_candidate_evaluations": 0, "cpu_candidate_evaluations": len(seeds),
                      "guidance_backend": "tensor_gpu"})
        return results, stats

    evaluator = _evaluator(cpu_evaluator, tensor_evaluator)
    started, counts = _start(adapter)
    rngs = [adapter.new_rng(seed) for seed in seeds]
    states = [adapter.initial_state(rng) for rng in rngs]
    device, batch_size = states[0].device, len(seeds)
    # Independent per-seed device-side selection randomness, separate from DDPM.
    uniforms = None
    if temperature > 0:
        uniforms = torch.stack([torch.rand(guided_steps, dtype=torch.float64, device=device,
                                generator=torch.Generator(device=device).manual_seed(_candidate_seed(seed, 0x53564444)))
                                for seed in seeds])
    records, timesteps = [], []
    for step in range(adapter.steps - 1, -1, -1):
        parent_clean = adapter.predict_x0_batch(states, step, batch_size=inference_batch_size)
        if not 1 <= step <= guided_steps:
            states = [adapter.sample_transition(state, step, rng, x0=clean)
                      for state, rng, clean in zip(states, rngs, parent_clean)]
            continue
        proposals = [[adapter.sample_transition(state, step, rng, x0=clean) for _ in range(k)]
                     for state, rng, clean in zip(states, rngs, parent_clean)]
        flat = [proposal for group in proposals for proposal in group]
        clean = adapter.predict_x0_batch(flat, step - 1, batch_size=inference_batch_size)
        values = _score_states(adapter, evaluator, scenario, clean, [rng for rng in rngs for _ in range(k)],
                               inference_batch_size)
        selection = tensor_selection(values, batch_size, k, temperature,
                                     uniforms[:, len(records)] if uniforms is not None else None)
        records.append({**values, **selection})
        timesteps.append(step)
        if progress_callback is not None and (len(records) == 1 or len(records) % 50 == 0 or step == 1):
            progress_callback(len(records), guided_steps)
        stacked = torch.cat(flat, dim=0).reshape(batch_size, k, *flat[0].shape[1:])
        indices = selection["chosen"].reshape(batch_size, 1, *([1] * (stacked.ndim - 2)))
        selected = stacked.gather(1, indices.expand(batch_size, 1, *stacked.shape[2:])).squeeze(1)
        states = list(selected.split(1, dim=0))

    final_values = _score_states(adapter, evaluator, scenario, states, rngs, inference_batch_size)
    exported = _export_records([*records, final_values])
    samples = adapter.complete_graph_batch(states, rngs, batch_size=inference_batch_size)
    results = []
    for trial, (sample, seed) in enumerate(zip(samples, seeds)):
        trace = []
        for step, values in zip(timesteps, exported[:-1]):
            fallback = bool(values["fallback"][trial])
            trace.append({"format": TRACE_FORMAT, "timestep": step, "candidate_state_timestep": step - 1,
                          "selected_index": int(values["chosen"][trial]),
                          "eligible_indices": values["eligible"][trial].nonzero().flatten().tolist(),
                          "coverage_anchor_index": None if fallback else int(values["anchor"][trial]),
                          "fallback": "No scorable GPU proposal; continue ordinary proposal 0." if fallback else None,
                          "candidates": [{"candidate_id": f"t-{step:04d}-proposal-{index:03d}",
                                          "evaluation": _evaluation_record(values, trial * k + index, evaluator)}
                                         for index in range(k)]})
        candidate = _scored_sample(sample, cpu_evaluator, scenario, seed, "svdd_pm", parameters)
        _backend_metadata(candidate, evaluator, device, _evaluation_record(exported[-1], trial, evaluator))
        gpu_evaluations = k * len(records) + 1
        results.append(SamplingResult([candidate], 0, trace=trace,
                                      stats={"candidate_evaluations": gpu_evaluations + 1,
                                             "gpu_candidate_evaluations": gpu_evaluations, "cpu_candidate_evaluations": 1,
                                             "graph_completions": gpu_evaluations + 1,
                                             "guided_steps_applied": len(records),
                                             "fallback_count": sum(bool(event["fallback"]) for event in trace)}))
    return _finish(adapter, started, counts, results, inference_batch_size)


@torch.no_grad()
def best_of_n_batch_gpu(adapter, cpu_evaluator, scenario, seeds, n=4,
                        inference_batch_size=10, tensor_evaluator=None):
    seeds = _batch_inputs(seeds, inference_batch_size)
    if not isinstance(n, int) or n < 1:
        raise ValueError("n must be a positive integer")
    evaluator = _evaluator(cpu_evaluator, tensor_evaluator)
    started, counts = _start(adapter)
    candidate_seeds = [_candidate_seed(seed, index) for seed in seeds for index in range(n)]
    rngs = [adapter.new_rng(seed) for seed in candidate_seeds]
    states = [adapter.initial_state(rng) for rng in rngs]
    for step in range(adapter.steps - 1, -1, -1):
        clean = adapter.predict_x0_batch(states, step, batch_size=inference_batch_size)
        states = [adapter.sample_transition(state, step, rng, x0=x0)
                  for state, rng, x0 in zip(states, rngs, clean)]
    values = _score_states(adapter, evaluator, scenario, states, rngs, inference_batch_size)
    seed_tensor = torch.tensor(candidate_seeds, dtype=torch.long, device=states[0].device).reshape(len(seeds), n)
    selection = tensor_selection(values, len(seeds), n, tie_seeds=seed_tensor)
    exported = _export_records([{**values, **selection}])[0]
    samples = adapter.complete_graph_batch(states, rngs, batch_size=inference_batch_size)
    results = []
    for trial, seed in enumerate(seeds):
        candidates = []
        for index in range(n):
            flat_index = trial * n + index
            candidate = _scored_sample(samples[flat_index], cpu_evaluator, scenario, candidate_seeds[flat_index],
                                       "best_of_n", {"n": n})
            candidate.candidate_id = f"candidate-{index:03d}-seed-{candidate.seed}"
            candidate.sample.metadata.update({"trial_seed": seed, "candidate_index": index,
                                              "selection_backend": "tensor_gpu"})
            _backend_metadata(candidate, evaluator, states[0].device,
                              _evaluation_record(exported, flat_index, evaluator))
            candidates.append(candidate)
        results.append(SamplingResult(candidates, int(exported["chosen"][trial]),
                                      stats={"candidate_evaluations": 2 * n, "gpu_candidate_evaluations": n,
                                             "cpu_candidate_evaluations": n, "graph_completions": 2 * n,
                                             "coverage_anchor_index": int(exported["anchor"][trial]),
                                             "eligible_candidate_count": int(exported["eligible"][trial].sum())}))
    return _finish(adapter, started, counts, results, inference_batch_size)
