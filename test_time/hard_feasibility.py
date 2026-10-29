"""Online binary lookaheads for an approximate blockwise Doob h-transform.

All lookaheads use the ordinary DDPM and its actual edge decoder. Only the
verified pass predicate controls selection; neither energy nor severity does.
Finite proposal banks, Monte Carlo values and fallbacks are approximations.
"""

from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
import time

import numpy as np
from tqdm.auto import tqdm

from .extension_types import ExtensionResult
from .types import overall_validity


_PURPOSES = {"proposal": 1, "rollout": 2, "selection": 3, "fallback": 4}


def stream_seed(seed, stage, proposal=0, rollout=0, purpose="proposal"):
    """Independent streams whose keys do not depend on bank size or execution."""
    sequence = np.random.SeedSequence([seed, stage, proposal, rollout,
                                      _PURPOSES[purpose], 0x48415244])
    return int(sequence.generate_state(1, dtype=np.uint64)[0] % np.uint64(2**63))


def select_h(success_counts, rng):
    """Equal-size rollout banks: select proportionally to successes, or None."""
    counts = np.asarray(success_counts, dtype=float)
    if counts.ndim != 1 or not len(counts) or not np.isfinite(counts).all() or (counts < 0).any():
        raise ValueError("h weights must be a nonempty finite nonnegative vector")
    total = counts.sum()
    return None if total == 0 else int(rng.choice(len(counts), p=counts / total))


def clone_state(state):
    return state.clone() if hasattr(state, "clone") else deepcopy(state)


def advance_base(adapter, states, rngs, remaining, destination, *,
                 inference_batch_size=10, progress=None):
    """Resume ordinary transitions, without changing the schedule or inputs.

For this adapter DDPM is Markov in the node tensor; remaining transitions and
the independent RNG complete the resumable state. A state with n transitions
remaining next executes adapter timestep n-1, including timestep zero.
"""
    if not 0 <= destination <= remaining <= adapter.steps or len(states) != len(rngs):
        raise ValueError("Invalid remaining-transition interval or RNG bank")
    states = [clone_state(state) for state in states]
    for step in tqdm(range(remaining - 1, destination - 1, -1), desc=progress,
                     unit="step", disable=progress is None, dynamic_ncols=True,
                     mininterval=1, leave=False):
        clean = adapter.predict_x0_batch(states, step, batch_size=inference_batch_size)
        states = [adapter.sample_transition(state, step, rng, x0=x0)
                  for state, rng, x0 in zip(states, rngs, clean)]
    return states


def complete_base(adapter, states, rngs, remaining, *, inference_batch_size=10, progress=None):
    """Full unguided suffix, including the frozen stochastic edge machinery."""
    states = advance_base(adapter, states, rngs, remaining, 0,
                          inference_batch_size=inference_batch_size, progress=progress)
    return adapter.complete_graph_batch(states, rngs, batch_size=inference_batch_size)


def _policy_hash(evaluator, scenario):
    specs = [asdict(spec) for spec in getattr(evaluator, "specs", [])]
    search = getattr(evaluator, "bedroom_search_fn", None)
    payload = {"scenario": asdict(scenario), "specs": specs,
               "validity_weight": getattr(evaluator, "validity_weight", None),
               "placement_checker": None if search is None else
                   f"{search.__module__}.{search.__qualname__}"}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def hard_h_mc(adapter, evaluator, scenario, seed, *, k=4, rollouts=16,
              anchor_remaining=(200, 100, 30), terminal_k=8,
              inference_batch_size=10, zero_policy="fresh_base", progress=None,
              event_callback=None, max_nfe=None):
    """Return a verified terminal graph, or an explicit unsuccessful result.

Root proposals use independent initial noise and ordinary prefixes. Each later
bank branches from the retained noisy state. The fixed rollout count is shared
by every proposal. Every completed graph reaches ``event_callback`` in logical
proposal/rollout order, including successes not returned by the main sampler.

``max_nfe`` caps sample-wise denoiser work. Before starting a stage we reserve
its entire balanced bank and, when configured, one possible fresh-base fallback.
Already executed batched work is charged at the first observed success.
"""
    for name, value in (("k", k), ("rollouts", rollouts), ("terminal_k", terminal_k),
                        ("inference_batch_size", inference_batch_size)):
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    anchors = tuple(anchor_remaining)
    if (not anchors or any(type(value) is not int or value < 1 for value in anchors)
            or anchors[0] > adapter.steps // 5
            or any(first <= second for first, second in zip(anchors, anchors[1:]))):
        raise ValueError("Anchors must decrease strictly within the last 20% of transitions")
    if zero_policy not in {"fresh_base", "abort"}:
        raise ValueError("zero_policy must be fresh_base or abort")
    if max_nfe is not None and (type(max_nfe) is not int or max_nfe < 1):
        raise ValueError("max_nfe must be a positive integer or None")
    if "node_samples" not in adapter.counters:
        raise ValueError("The adapter must count sample-wise node_samples")

    if hasattr(adapter, "synchronize"):
        adapter.synchronize()
    started, initial_counts = time.perf_counter(), dict(adapter.counters)
    policy_hash = _policy_hash(evaluator, scenario)
    parameters = {"k": k, "rollouts": rollouts, "anchor_remaining": list(anchors),
                  "terminal_k": terminal_k, "zero_policy": zero_policy,
                  "max_nfe": max_nfe, "inference_batch_size": inference_batch_size}
    trace = []
    evaluated = fallback_count = all_zero_count = successes = 0
    checker_seconds = 0.0
    first_success = None
    outcomes = {"valid": 0, "invalid": 0, "unknown": 0, "not_applicable": 0}

    def counts_now():
        return {key: value - initial_counts.get(key, 0)
                for key, value in adapter.counters.items()}

    def finish(status, sample=None, evaluation=None):
        if _policy_hash(evaluator, scenario) != policy_hash:
            raise RuntimeError("Scenario or checker policy changed during hard_h_mc")
        if hasattr(adapter, "synchronize"):
            adapter.synchronize()
        stats = {**counts_now(), "candidate_evaluations": evaluated,
                 "completed_graphs": evaluated, "feasible_graphs_found": successes,
                 "returned_success": status == "success", "any_feasible_found": successes > 0,
                 "first_success": first_success, "outcome_counts": outcomes,
                 "fallback_count": fallback_count, "all_zero_count": all_zero_count,
                 "checker_seconds": checker_seconds, "wall_seconds": time.perf_counter() - started,
                 "policy_hash": policy_hash, "exact_conditional": False,
                 "method_parameters": parameters,
                 "timing_scope": "episode_including_checking_and_event_callbacks"}
        return ExtensionResult(sample, evaluation, status, stats=stats, trace=trace)

    def reserve(required, stage, kind):
        used = counts_now()["node_samples"]
        if max_nfe is not None and used + required > max_nfe:
            trace.append({"stage": stage, "kind": kind, "status": "budget_exhausted",
                          "nfe_used": used, "nfe_required": required, "max_nfe": max_nfe})
            return False
        return True

    def label(stage, proposal, rollout, origin, draw_seed, sample):
        nonlocal evaluated, checker_seconds, successes, first_success
        if _policy_hash(evaluator, scenario) != policy_hash:
            raise RuntimeError("Scenario or checker policy changed during hard_h_mc")
        event_id = f"stage-{stage:02d}-{origin}-p{proposal:03d}-r{rollout:04d}"
        sample.scenario = asdict(scenario)
        sample.metadata.update(method="hard_h_mc", seed=draw_seed, trial_seed=seed,
                               method_parameters=parameters, event_id=event_id,
                               origin=origin, policy_hash=policy_hash)
        before = time.perf_counter()
        evaluation = evaluator.evaluate(sample, scenario)
        checker_seconds += time.perf_counter() - before
        validity = overall_validity(evaluation)
        passed = validity == "valid"
        evaluated += 1
        successes += int(passed)
        outcomes[validity] += 1
        event = {"event_id": event_id, "stage": stage, "proposal": proposal,
                 "rollout": rollout, "origin": origin, "seed": draw_seed,
                 "passed": passed, "overall": validity,
                 "structurally_valid": evaluation.structurally_valid,
                 "unknown_ids": [r.id for r in evaluation.results
                                 if r.applicable and r.counts_toward_compliance and r.status == "unknown"],
                 "failing_ids": [r.id for r in evaluation.results
                                 if r.applicable and r.counts_toward_compliance and r.status == "fail"],
                 "compute_at_observation": counts_now(),
                 "wall_seconds_at_observation": time.perf_counter() - started}
        if passed and first_success is None:
            first_success = {"event_id": event_id, **counts_now(),
                             "candidate_evaluations": evaluated,
                             "wall_seconds": event["wall_seconds_at_observation"],
                             "batched_work_already_charged": True}
        if event_callback is not None:
            event_callback(sample, evaluation, event)
        return evaluation, event

    parent, remaining = None, adapter.steps
    for stage, destination in enumerate((*anchors, 0)):
        terminal = destination == 0
        width = terminal_k if terminal else k
        ordinary_cost = width * (remaining - destination)
        lookahead_cost = 0 if terminal else k * rollouts * destination
        fallback_reserve = remaining - destination if not terminal and zero_policy == "fresh_base" else 0
        if not reserve(ordinary_cost + lookahead_cost + fallback_reserve, stage,
                       "terminal" if terminal else "latent"):
            return finish("budget_exhausted")

        seeds = [stream_seed(seed, stage, proposal=index) for index in range(width)]
        rngs = [adapter.new_rng(draw_seed) for draw_seed in seeds]
        states = ([adapter.initial_state(rng) for rng in rngs] if parent is None
                  else [clone_state(parent) for _ in range(width)])
        prefix = None if progress is None else f"{progress} | stage {stage + 1}/{len(anchors) + 1}"
        candidates = advance_base(adapter, states, rngs, remaining, destination,
                                  inference_batch_size=inference_batch_size,
                                  progress=None if prefix is None else prefix + " proposals")
        selection_seed = stream_seed(seed, stage, purpose="selection")
        selection_rng = np.random.default_rng(selection_seed)
        stage_trace = {"stage": stage, "kind": "terminal" if terminal else "latent",
                       "from_remaining": remaining, "to_remaining": destination,
                       "proposal_seeds": seeds, "selection_seed": selection_seed,
                       "events": []}
        trace.append(stage_trace)

        if terminal:
            samples = adapter.complete_graph_batch(candidates, rngs, batch_size=inference_batch_size)
            evaluations, good = [], []
            for index, sample in tqdm(enumerate(samples), total=len(samples),
                                      desc=None if prefix is None else prefix + " CPU checks",
                                      disable=prefix is None, unit="graph", mininterval=1, leave=False):
                evaluation, event = label(stage, index, 0, "terminal", seeds[index], sample)
                evaluations.append(evaluation)
                stage_trace["events"].append(event)
                if event["passed"]:
                    good.append(index)
            stage_trace["passing_indices"] = good
            if not good:
                stage_trace["status"] = "no_feasible_terminal_sample"
                return finish("no_feasible_terminal_sample")
            selected = int(selection_rng.choice(good))
            stage_trace.update(selected_index=selected, status="success")
            # Return the already decoded graph; a fresh edge draw could invalidate it.
            return finish("success", samples[selected], evaluations[selected])

        rollout_keys = [(index, ell) for index in range(k) for ell in range(rollouts)]
        draw_seeds = [stream_seed(seed, stage, index, ell, "rollout") for index, ell in rollout_keys]
        rollout_rngs = [adapter.new_rng(draw_seed) for draw_seed in draw_seeds]
        suffixes = complete_base(adapter, [candidates[index] for index, _ in rollout_keys],
                                 rollout_rngs, destination, inference_batch_size=inference_batch_size,
                                 progress=None if prefix is None else prefix + " lookaheads")
        success_counts = [0] * k
        for (index, ell), draw_seed, sample in tqdm(zip(rollout_keys, draw_seeds, suffixes), total=len(suffixes),
                                                   desc=None if prefix is None else prefix + " CPU checks",
                                                   disable=prefix is None, unit="graph", mininterval=1, leave=False):
            _, event = label(stage, index, ell, "lookahead", draw_seed, sample)
            success_counts[index] += int(event["passed"])
            stage_trace["events"].append(event)
        total = sum(success_counts)
        stage_trace.update(success_counts=success_counts, rollout_count_per_proposal=rollouts,
                           h_hat=[count / rollouts for count in success_counts],
                           selection_probabilities=[count / total for count in success_counts] if total else None)
        selected = select_h(success_counts, selection_rng)
        if selected is not None:
            parent = candidates[selected]
            stage_trace.update(selected_index=selected, status="selected")
        else:
            all_zero_count += 1
            stage_trace.update(selected_index=None, status="all_zero", fallback=zero_policy)
            if zero_policy == "abort":
                return finish("no_positive_h_estimate")
            fallback_count += 1
            fallback_seed = stream_seed(seed, stage, purpose="fallback")
            fallback_rng = adapter.new_rng(fallback_seed)
            initial = adapter.initial_state(fallback_rng) if parent is None else clone_state(parent)
            parent = advance_base(adapter, [initial], [fallback_rng], remaining, destination,
                                  inference_batch_size=inference_batch_size,
                                  progress=None if prefix is None else prefix + " fallback")[0]
            stage_trace["fallback_seed"] = fallback_seed
        remaining = destination
