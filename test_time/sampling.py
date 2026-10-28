"""Baseline, complete-trajectory reranking, and late-step SVDD-PM.

Selection is valid first, then coverage, then energy. Within a validity class,
the first candidate anchors which constraint IDs may remain unknown: another
candidate cannot hide a known failure by making that ID unknown. Prefer fewer
unknowns among the remaining candidates. This conservative, order-dependent
guard may reject useful candidates with different unresolved checks. SVDD-PM
uses the same guard; its clean-node reward is an approximation.
"""

from dataclasses import asdict
import math
import time

import numpy as np

from .types import Candidate, SamplingResult


def selection_key(candidate):
    evaluation = candidate.evaluation
    return (not evaluation.structurally_valid, evaluation.unknown_count,
            evaluation.energy, candidate.seed)


def coverage_pool(candidates, indices=None):
    """Keep validity first and prohibit new unknown IDs relative to an anchor."""
    indices = list(range(len(candidates))) if indices is None else list(indices)
    valid = [index for index in indices if candidates[index].evaluation.structurally_valid]
    pool = valid or indices
    anchor = pool[0]

    def unknown_ids(index):
        return {result.id for result in candidates[index].evaluation.results
                if result.applicable and result.status == "unknown"}

    allowed_unknown = unknown_ids(anchor)
    pool = [index for index in pool if unknown_ids(index) <= allowed_unknown]
    fewest_unknown = min(candidates[index].evaluation.unknown_count for index in pool)
    return [index for index in pool
            if candidates[index].evaluation.unknown_count == fewest_unknown], anchor


def choose_index(rewards, temperature, rng):
    """Choose among finite rewards; the caller handles unscorable proposals."""
    if not math.isfinite(temperature) or temperature < 0:
        raise ValueError("temperature must be a finite nonnegative number")
    rewards = np.asarray(rewards, dtype=float)
    eligible = np.flatnonzero(np.isfinite(rewards))
    if not len(eligible):
        raise ValueError("No finite candidate reward")
    if temperature == 0:
        return int(eligible[np.argmax(rewards[eligible])])
    # Subtract before division: large equal rewards must not become inf - inf.
    with np.errstate(over="ignore", under="ignore"):
        weights = np.exp((rewards[eligible] - rewards[eligible].max()) / temperature)
    return int(rng.choice(eligible, p=weights / weights.sum()))


def _sync(adapter):
    if hasattr(adapter, "synchronize"):
        adapter.synchronize()


def _start(adapter):
    _sync(adapter)
    return time.perf_counter(), dict(adapter.counters)


def _stats(adapter, started, counts, evaluated):
    _sync(adapter)
    return {**{key: value - counts.get(key, 0) for key, value in adapter.counters.items()},
            "candidate_evaluations": evaluated,
            "wall_seconds": time.perf_counter() - started}


def _candidate(adapter, evaluator, scenario, state, seed, method, parameters):
    sample = adapter.complete_graph(state)
    sample.scenario = asdict(scenario)
    sample.metadata.update({"seed": seed, "method": method,
                            "method_parameters": parameters})
    return Candidate(sample, evaluator.evaluate(sample, scenario), seed,
                     candidate_id=f"seed-{seed}")


def _ordinary_state(adapter, rng):
    state = adapter.initial_state(rng)
    for step in range(adapter.steps - 1, -1, -1):
        state = adapter.sample_transition(state, step, rng)
    return state


def baseline(adapter, evaluator, scenario, seed):
    started, counts = _start(adapter)
    state = _ordinary_state(adapter, adapter.new_rng(seed))
    candidate = _candidate(adapter, evaluator, scenario, state, seed, "baseline", {"n": 1})
    stats = _stats(adapter, started, counts, 1)
    candidate.stats = dict(stats)
    return SamplingResult([candidate], 0, stats=stats)


def _candidate_seed(seed, index):
    if index == 0:
        return seed
    # Avoid overlapping candidate seed sets for adjacent independent trial seeds.
    value = np.random.SeedSequence([seed, index]).generate_state(1, dtype=np.uint64)[0]
    return int(value % np.uint64(2**63))


def best_of_n(adapter, evaluator, scenario, seed, n=4):
    if not isinstance(n, int) or n < 1:
        raise ValueError("n must be a positive integer")
    started, counts = _start(adapter)
    candidates = []
    for index in range(n):
        candidate = baseline(adapter, evaluator, scenario, _candidate_seed(seed, index)).candidates[0]
        candidate.candidate_id = f"candidate-{index:03d}-seed-{candidate.seed}"
        candidate.sample.metadata.update({"method": "best_of_n",
                                          "method_parameters": {"n": n},
                                          "trial_seed": seed, "candidate_index": index})
        candidates.append(candidate)
    eligible, anchor = coverage_pool(candidates)
    selected = min(eligible, key=lambda index: selection_key(candidates[index]))
    stats = _stats(adapter, started, counts, n)
    stats.update({"coverage_anchor_index": anchor, "eligible_candidate_count": len(eligible)})
    return SamplingResult(candidates, selected, stats=stats)


def svdd_pm(adapter, evaluator, scenario, seed, k=4, guided_steps=10, temperature=0.0):
    if not isinstance(k, int) or k < 1:
        raise ValueError("k must be a positive integer")
    if not isinstance(guided_steps, int) or not 0 <= guided_steps < adapter.steps:
        raise ValueError(f"guided_steps must be between 0 and {adapter.steps - 1}")
    if not math.isfinite(temperature) or temperature < 0:
        raise ValueError("temperature must be a finite nonnegative number")
    parameters = {"k": k, "guided_steps": guided_steps, "temperature": temperature}
    if k == 1 or guided_steps == 0:
        # No redundant PM evaluations or categorical draws: identical RNG stream.
        result = baseline(adapter, evaluator, scenario, seed)
        result.candidates[0].sample.metadata.update({"method": "svdd_pm",
                                                    "method_parameters": parameters})
        result.stats.update({"guided_steps_applied": 0, "fallback_count": 0})
        return result

    started, counts = _start(adapter)
    rng = adapter.new_rng(seed)
    choice_rng = np.random.default_rng(np.random.SeedSequence([seed, 0x53564444]))
    state = adapter.initial_state(rng)
    trace = []
    evaluated = 0
    fallback_count = 0
    for step in range(adapter.steps - 1, -1, -1):
        if not 1 <= step <= guided_steps:
            state = adapter.sample_transition(state, step, rng)
            continue

        # All proposals share the same frozen denoiser prediction at the parent.
        parent_clean = adapter.predict_x0(state, step)
        proposals = [adapter.sample_transition(state, step, rng, x0=parent_clean)
                     for _ in range(k)]
        candidates = []
        for index, proposal in enumerate(proposals):
            clean = adapter.predict_x0(proposal, step - 1)
            candidate = _candidate(adapter, evaluator, scenario, clean, seed,
                                   "svdd_pm_surrogate", parameters)
            candidate.candidate_id = f"t-{step:04d}-proposal-{index:03d}"
            candidate.sample.metadata.update({"parent_timestep": step,
                                              "surrogate_timestep": step - 1,
                                              "proposal_index": index})
            candidates.append(candidate)
        evaluated += k
        scorable = [index for index, candidate in enumerate(candidates)
                    if candidate.evaluation.structurally_valid
                    and candidate.evaluation.measurable_count > 0
                    and math.isfinite(candidate.evaluation.reward)]
        if scorable:
            eligible, anchor = coverage_pool(candidates, scorable)
            chosen = eligible[choose_index([candidates[index].evaluation.reward for index in eligible],
                                           temperature, choice_rng)]
            fallback = None
        else:
            # Proposal zero is an ordinary transition, not a clean approximation.
            eligible = []
            anchor = None
            chosen = 0
            fallback = "No valid proposal with measurable constraints; continue ordinary proposal 0."
            fallback_count += 1
        trace.append({"timestep": step, "candidate_state_timestep": step - 1,
                      "selected_index": chosen, "eligible_indices": eligible,
                      "coverage_anchor_index": anchor,
                      "fallback": fallback, "candidates": [asdict(item) for item in candidates]})
        state = proposals[chosen]

    candidate = _candidate(adapter, evaluator, scenario, state, seed, "svdd_pm", parameters)
    stats = _stats(adapter, started, counts, evaluated + 1)
    stats.update({"guided_steps_applied": len(trace), "fallback_count": fallback_count})
    candidate.stats = dict(stats)
    return SamplingResult([candidate], 0, trace=trace, stats=stats)
