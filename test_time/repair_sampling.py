"""Late SVDD with GPU repair lookahead and separately audited final repair."""

from dataclasses import asdict
import time

import torch
from tqdm.auto import tqdm

from .extension_types import ExtensionResult
from .gpu_constraints import TensorConstraintEvaluator
from .repair import (SCORE_KEYS, final_repair, ordered_indices, repair_settings,
                     score_repairs, selection_record)
from .types import overall_validity


@torch.no_grad()
def repair_aware_svdd(adapter, evaluator, scenario, seed, *, k=4, guided_steps=200,
                      inference_batch_size=10, repair_config=None, tensor_evaluator=None,
                      progress=None):
    """Score repairs of clean predictions but retain unchanged noisy proposals.

    Trials run independently; all siblings and their repair banks use bounded
    GPU microbatches. Fixed evaluation counts, rather than wall-clock stopping,
    define reproducible local-search budgets.
    """
    if type(k) is not int or k < 1:
        raise ValueError("k must be a positive integer")
    last_window = min(adapter.steps - 1, max(1, adapter.steps // 5))
    if type(guided_steps) is not int or not 0 <= guided_steps <= last_window:
        raise ValueError(f"guided_steps must fit within the final 20% (0..{last_window})")
    if type(inference_batch_size) is not int or inference_batch_size < 1:
        raise ValueError("inference_batch_size must be a positive integer")
    config = repair_settings(repair_config)
    if scenario.coordinate_scale_m_per_model_unit is None:
        raise ValueError("Repair requires an explicit metric scale")
    if tensor_evaluator is None:
        tensor_evaluator = TensorConstraintEvaluator(evaluator.specs, evaluator.validity_weight)
    sync = getattr(adapter, "synchronize", lambda: None)
    sync()
    started = time.perf_counter()
    before_counts = dict(adapter.counters)
    rng = adapter.new_rng(seed)
    state = adapter.initial_state(rng)
    trace, fallback_count = [], 0
    repair_evaluations = 0
    repair_seconds = 0.0
    step_iter = tqdm(range(adapter.steps - 1, -1, -1), desc=progress or "repair-aware SVDD",
                     disable=not bool(progress), unit="step")
    for step in step_iter:
        clean = adapter.predict_x0_batch([state], step, batch_size=inference_batch_size)[0]
        if k == 1 or not 1 <= step <= guided_steps:
            state = adapter.sample_transition(state, step, rng, x0=clean)
            continue
        proposals = [adapter.sample_transition(state, step, rng, x0=clean) for _ in range(k)]
        predictions = adapter.predict_x0_batch(proposals, step - 1, batch_size=inference_batch_size)
        graphs = adapter.predict_graph_tensors(predictions, [rng] * k, batch_size=inference_batch_size)
        sync()
        repair_started = time.perf_counter()
        bank = score_repairs(graphs, tensor_evaluator, scenario, config,
                             max_evaluations=config["inner_evaluations"], batch_size=inference_batch_size)
        selected = bank["selected"]
        values = {key: bank["scores"][key][selected] for key in SCORE_KEYS}
        valid = (values["structurally_valid"] & torch.isfinite(values["energy"]) &
                 (values["measurable_count"] > 0))
        unknown = (values["status"] == 2) & values["applicable"]
        anchor = valid.long().argmax()
        eligible = valid & ~(unknown & ~unknown[anchor]).any(-1)
        ordering = ordered_indices(values, bank["distortion"][selected], eligible, tensor_evaluator.specs)
        fallback = not bool(eligible.any())
        chosen = 0 if fallback else int(ordering[0])
        fallback_count += int(fallback)
        repair_evaluations += bank["evaluations"]
        records = []
        for proposal, index in enumerate(selected.tolist()):
            record = selection_record(bank, index, tensor_evaluator.specs, graphs["xy"],
                                      scenario.coordinate_scale_m_per_model_unit)
            identity = int(((bank["owner"] == proposal) & (bank["candidate"] == 0)).nonzero()[0])
            record.update(proposal=proposal, before_energy=float(bank["scores"]["energy"][identity]),
                          repair_evaluations=int((bank["owner"] == proposal).sum()),
                          eligible=bool(eligible[proposal]), graph_id=f"seed-{seed}-t-{step}-proposal-{proposal}")
            records.append(record)
        sync()
        duration = time.perf_counter() - repair_started
        repair_seconds += duration
        trace.append(dict(timestep=step, selected_index=chosen, candidates=records,
                          fallback="ordinary_proposal_zero" if fallback else None,
                          repair_wall_seconds=duration, scoring_backend="tensor_proxy",
                          retained_state="original_noisy_proposal"))
        state = proposals[chosen]
        step_iter.set_postfix(repair_checks=repair_evaluations, refresh=False)
    sample = adapter.complete_graph_batch([state], [rng], batch_size=inference_batch_size)[0]
    sample.scenario = asdict(scenario)
    sample.metadata.update(method="svdd_repair_aware", seed=seed, guided_steps=guided_steps,
                           proposals=k, repair_settings=config,
                           guidance_backend="tensor_repair_proxy", guidance_device=str(adapter.device))
    cpu_started = time.perf_counter()
    before_evaluation = evaluator.evaluate(sample, scenario)
    original_audit_seconds = time.perf_counter() - cpu_started
    sync()
    final_started = time.perf_counter()
    repaired, evaluation, repair_record = final_repair(
        adapter, sample, evaluator, scenario, tensor_evaluator, config, batch_size=inference_batch_size)
    sync()
    final_seconds = time.perf_counter() - final_started
    status = overall_validity(evaluation)
    counters = {key: int(value - before_counts.get(key, 0)) for key, value in adapter.counters.items()}
    stats = dict(counters, wall_seconds=time.perf_counter() - started,
                 candidate_evaluations=repair_evaluations + repair_record["gpu_evaluations"] + repair_record["cpu_evaluations"] + 1,
                 gpu_candidate_evaluations=repair_evaluations + repair_record["gpu_evaluations"],
                 cpu_candidate_evaluations=repair_record["cpu_evaluations"] + 1,
                 fallback_count=fallback_count, guided_steps_applied=len(trace),
                 inner_repair_wall_seconds=repair_seconds, final_repair_wall_seconds=final_seconds,
                 final_repair=repair_record, returned_success=status == "valid",
                 cpu_checker_seconds=original_audit_seconds + repair_record.get("cpu_checker_seconds", 0.0),
                 any_feasible_found=status == "valid", final_evaluation_backend="cpu_placement_audit_v2",
                 exact_conditional=False, timing_scope="complete_episode_including_repair_and_cpu_audit")
    repaired.metadata.update(method="svdd_repair_aware", seed=seed, repair_settings=config,
                             overall_validity=status, final_repair=repair_record)
    return ExtensionResult(repaired, evaluation, "success" if status == "valid" else "unverified" if status == "unknown" else "infeasible",
                           stats=stats, trace=trace, extra_samples={"pre_repair": (sample, before_evaluation)})
