"""Bounded, tensor-scored wall repair; CPU auditing remains authoritative.

The fixed graph topology supplies shared junctions. Candidate edits translate
collinear internal-wall groups, never independently resize extracted polygons.
The inexpensive proposal ordering favours area in the largest bedroom; only the
unchanged constraint checker decides whether a repair is useful.
"""

from copy import deepcopy
from dataclasses import asdict
import math
import time

import numpy as np
import torch

from .gpu_graph import extract_rooms
from .types import FloorplanSample, overall_validity


DEFAULT_REPAIR = dict(max_displacement_m=0.20, inner_evaluations=16,
                      final_evaluations=128, max_wall_moves=2,
                      max_moved_junctions=12, max_total_displacement_m=2.4,
                      final_enabled=True)
SCORE_KEYS = ("energy", "status", "applicable", "structurally_valid",
              "unknown_count", "measurable_count", "applicable_count")


def repair_settings(config=None):
    config = config or {}
    unexpected = set(config) - DEFAULT_REPAIR.keys()
    if unexpected:
        raise ValueError("Unknown repair settings: " + ", ".join(sorted(unexpected)))
    values = {**DEFAULT_REPAIR, **config}
    for name in ("inner_evaluations", "final_evaluations", "max_moved_junctions"):
        if type(values[name]) is not int or values[name] < 1:
            raise ValueError(f"{name} must be a positive integer")
    if type(values["max_wall_moves"]) is not int or values["max_wall_moves"] not in (1, 2):
        raise ValueError("max_wall_moves must be 1 or 2")
    for name in ("max_displacement_m", "max_total_displacement_m"):
        if not math.isfinite(values[name]) or values[name] <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if type(values["final_enabled"]) is not bool:
        raise ValueError("final_enabled must be boolean")
    return values


def _parts(rooms, graph):
    """Recover face-node membership and boundary edges on the tensor device."""
    xy = graph["xy"]
    batch, nodes, _ = xy.shape
    differences = rooms["polygons"][..., None, :] - xy[:, None, None]
    indices = differences.square().sum(-1).argmin(-1)
    vertex = rooms["vertex_mask"]
    count = vertex.sum(-1)
    positions = torch.arange(indices.shape[-1], device=xy.device)
    following = torch.where(positions[None, None] + 1 < count[..., None], positions[None, None] + 1, 0)
    ends = indices.gather(2, following)
    pairs = torch.minimum(indices, ends) * nodes + torch.maximum(indices, ends)
    incidence = torch.zeros((batch, nodes * nodes), dtype=torch.long, device=xy.device)
    incidence.scatter_add_(1, pairs.flatten(1), vertex.long().flatten(1))
    incidence = incidence.reshape(batch, nodes, nodes)
    incidence = incidence + incidence.transpose(1, 2)
    membership = torch.zeros((batch, indices.shape[1], nodes), dtype=torch.long, device=xy.device)
    membership.scatter_add_(2, indices, vertex.long())
    # Up to 53 active node IDs: a face membership and label fit in int64.
    bits = 2 ** torch.arange(nodes, device=xy.device, dtype=torch.long)
    signature = ((membership > 0).long() * bits).sum(-1) * 8 + rooms["labels"] + 1
    signature = torch.where(rooms["room_mask"], signature, 0).sort(-1).values
    return indices, incidence, signature


def _boundary_limits(xy, boundary):
    delta = xy[:, None] - xy[:, :, None]
    degree = boundary.sum(-1)
    # Movement along x requires every boundary neighbour to share y (and vice versa).
    movable = []
    low, high = [], []
    for axis in range(2):
        parallel = (delta[..., 1 - axis].abs() <= 1e-8) | ~boundary
        movable.append((degree == 0) | ((degree == 2) & parallel.all(-1)))
        values = xy[:, None, :, axis].expand_as(boundary)
        low.append(values.masked_fill(~boundary, torch.inf).amin(-1))
        high.append(values.masked_fill(~boundary, -torch.inf).amax(-1))
    return torch.stack(movable, -1), torch.stack(low, -1), torch.stack(high, -1), degree > 0


def _target_area(coordinates, indices, mask):
    # coordinates [B,C,N,2], selected face indices/mask [B,N].
    selected = coordinates.gather(2, indices[:, None, :, None].expand(-1, coordinates.shape[1], -1, 2))
    positions = torch.arange(indices.shape[-1], device=coordinates.device)
    following = torch.where(positions[None] + 1 < mask.sum(-1)[:, None], positions[None] + 1, 0)
    end = selected.gather(2, following[:, None, :, None].expand_as(selected))
    cross = selected[..., 0] * end[..., 1] - selected[..., 1] * end[..., 0]
    return (cross * mask[:, None]).sum(-1).abs() / 2


@torch.no_grad()
def candidate_bank(graph, scenario, config=None, max_evaluations=None):
    """Return a bounded packed bank, always including each original graph.

    No Shapely/CPU geometry is used. Two-group proposals are composed before
    scoring: an intermediate move need not improve a binary penalty.
    """
    config = repair_settings(config)
    scale = scenario.coordinate_scale_m_per_model_unit
    if scale is None:
        raise ValueError("Repair requires the declared metric scale")
    limit = config["inner_evaluations"] if max_evaluations is None else max_evaluations
    if type(limit) is not int or limit < 1:
        raise ValueError("max_evaluations must be positive")
    xy, active = graph["xy"], graph["node_mask"].bool()
    batch, nodes, _ = xy.shape
    if nodes > 53:
        raise ValueError("Repair supports the existing maximum of 53 junction slots")
    rooms = extract_rooms(**graph)
    indices, incidence, signature = _parts(rooms, graph)
    internal, boundary = incidence == 2, incidence == 1
    movable, lower, upper, on_boundary = _boundary_limits(xy, boundary)
    groups, permitted = [], []
    for axis in range(2):
        same_line = (xy[:, :, None, axis] - xy[:, None, :, axis]).abs() <= 1e-8
        aligned = internal & same_line
        incident = aligned.any(-1)
        members = same_line & incident[:, None] & active[:, None]
        # Group every collinear internal partition with the same coordinate.
        group_ok = incident & ~((members & ~movable[:, None, :, axis]).any(-1))
        first = torch.where(members, torch.arange(nodes, device=xy.device), nodes).amin(-1)
        group_ok &= first == torch.arange(nodes, device=xy.device)[None]
        displacement = xy.new_zeros((batch, nodes, nodes, 2))
        displacement[..., axis] = members.to(xy.dtype)
        groups.append(displacement)
        permitted.append(group_ok)
    directions = torch.cat(groups, 1)
    available = torch.cat(permitted, 1)
    amounts = xy.new_tensor([1.0, -1.0, 0.5, -0.5]) * config["max_displacement_m"] / scale
    edits = (directions[:, :, None] * amounts[None, None, :, None, None]).flatten(1, 2)
    valid = available[:, :, None].expand(-1, -1, 4).flatten(1)
    valid &= rooms["structurally_valid"][:, None] & ~rooms["removed_bridges"][:, None]
    # A fixed target only orders local search; it is never a regulatory score.
    polygons = rooms["polygons"]
    widths = torch.where(rooms["vertex_mask"][..., None], polygons, torch.inf).amin(-2)
    heights = torch.where(rooms["vertex_mask"][..., None], polygons, -torch.inf).amax(-2)
    area_hint = (heights - widths).prod(-1)
    target_score = area_hint.masked_fill(~(rooms["room_mask"] & (rooms["labels"] == 1)), -torch.inf)
    target = target_score.argmax(-1)
    row = torch.arange(batch, device=xy.device)
    target_indices, target_mask = indices[row, target], rooms["vertex_mask"][row, target]

    def allowed(delta):
        distance = delta.square().sum(-1).sqrt() * scale
        inside = (distance.amax(-1) <= config["max_displacement_m"] + 1e-8)
        inside &= distance.sum(-1) <= config["max_total_displacement_m"] + 1e-8
        inside &= (distance > 1e-8).sum(-1) <= config["max_moved_junctions"]
        candidate = xy[:, None] + delta
        moved = delta.abs() > 1e-8
        boundary_ok = (~moved | ~on_boundary[:, None, :, None] |
                       (movable[:, None] & (candidate >= lower[:, None] - 1e-8) &
                        (candidate <= upper[:, None] + 1e-8))).flatten(-2).all(-1)
        return inside & boundary_ok

    valid &= allowed(edits)
    if config["max_wall_moves"] == 2:
        hint = _target_area(xy[:, None] + edits, target_indices, target_mask).masked_fill(~valid, -torch.inf)
        top = hint.argsort(dim=-1, descending=True, stable=True)[:, :min(8, edits.shape[1])]
        selected = edits.gather(1, top[:, :, None, None].expand(-1, -1, nodes, 2))
        selected_valid = valid.gather(1, top)
        pair_a, pair_b = torch.triu_indices(top.shape[1], top.shape[1], offset=1, device=xy.device)
        paired = selected[:, pair_a] + selected[:, pair_b]
        # Different groups only; don't turn two half moves into one hidden larger move.
        pair_valid = selected_valid[:, pair_a] & selected_valid[:, pair_b]
        pair_valid &= top[:, pair_a] // 4 != top[:, pair_b] // 4
        pair_valid &= allowed(paired)
        edits = torch.cat((edits, paired), 1)
        valid = torch.cat((valid, pair_valid), 1)
    edits = torch.cat((xy.new_zeros((batch, 1, nodes, 2)), edits), 1)
    valid = torch.cat((torch.ones((batch, 1), device=xy.device, dtype=torch.bool), valid), 1)
    coordinates = xy[:, None] + edits
    hint = _target_area(coordinates, target_indices, target_mask)
    hint = hint.masked_fill(~valid, -torch.inf)
    hint[:, 0] = torch.inf  # Identity always occupies the first evaluation.
    order = hint.argsort(dim=-1, descending=True, stable=True)[:, :min(limit, edits.shape[1])]
    chosen_valid = valid.gather(1, order)
    owner, slot = torch.nonzero(chosen_valid, as_tuple=True)
    candidate = order[owner, slot]
    selected_edits = edits[owner, candidate]
    packed = {name: value[owner].clone() for name, value in graph.items()}
    packed["xy"] += selected_edits
    distances = selected_edits.square().sum(-1).sqrt() * scale
    return dict(graph=packed, owner=owner, candidate=candidate,
                displacement_m=distances, distortion=distances.sum(-1) / config["max_total_displacement_m"],
                reference_signature=signature, config=config)


def feasibility_mask(values, specs):
    counted = torch.tensor([spec.counts_toward_compliance for spec in specs],
                           device=values["status"].device, dtype=torch.bool)
    applicable = values["applicable"] & counted[None]
    return (values["structurally_valid"] & applicable.any(-1) &
            (~applicable | (values["status"] == 0)).all(-1))


def ordered_indices(values, distortion, eligible, specs):
    feasible = feasibility_mask(values, specs)
    order = torch.arange(len(distortion), device=distortion.device)
    # Stable sorts from the least significant key implement lexicographic order.
    for key in (distortion, values["energy"], values["unknown_count"], (~feasible).long(), (~eligible).long()):
        order = order[key[order].argsort(stable=True)]
    return order


@torch.no_grad()
def score_repairs(graph, tensor_evaluator, scenario, config=None, *, max_evaluations=None,
                  batch_size=10):
    """GPU proxy search with full face, boundary and unknown-ID guards."""
    bank = candidate_bank(graph, scenario, config, max_evaluations)
    values, invariant = [], []
    total = len(bank["owner"])
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        part = {name: value[start:end] for name, value in bank["graph"].items()}
        rooms = extract_rooms(**part)
        signatures = _parts(rooms, part)[2]
        unchanged = (signatures == bank["reference_signature"][bank["owner"][start:end]]).all(-1)
        invariant.append(unchanged & ~rooms["removed_bridges"])
        output = tensor_evaluator.evaluate(rooms, scenario)
        values.append({key: output[key] for key in SCORE_KEYS})
    scores = {key: torch.cat([value[key] for value in values]) for key in SCORE_KEYS}
    invariant = torch.cat(invariant)
    selected, rankings = [], []
    eligible = torch.zeros(total, dtype=torch.bool, device=bank["owner"].device)
    for owner in range(graph["xy"].shape[0]):
        ids = (bank["owner"] == owner).nonzero().flatten()
        identity = ids[0]
        unknown = (scores["status"] == 2) & scores["applicable"]
        pool = invariant[ids] & scores["structurally_valid"][ids] & torch.isfinite(scores["energy"][ids])
        pool &= scores["measurable_count"][ids] > 0
        pool &= ~(unknown[ids] & ~unknown[identity]).any(-1)
        eligible[ids] = pool
        local = {key: value[ids] for key, value in scores.items()}
        order = ids[ordered_indices(local, bank["distortion"][ids], pool, tensor_evaluator.specs)]
        rankings.append(order)
        selected.append(torch.where(pool.any(), order[0], identity))
    bank.update(scores=scores, eligible=eligible, selected=torch.stack(selected), rankings=rankings,
                evaluations=total)
    return bank


def selection_record(bank, index, specs, original_xy, scale):
    scores = bank["scores"]
    owner = int(bank["owner"][index])
    changed = bank["displacement_m"][index] > 1e-8
    node_ids = changed.nonzero().flatten().tolist()
    xy = bank["graph"]["xy"][index]
    return dict(structurally_valid=bool(scores["structurally_valid"][index]),
                energy=float(scores["energy"][index]), unknown_count=int(scores["unknown_count"][index]),
                feasible=bool(feasibility_mask(scores, specs)[index]),
                status_by_id={spec.id: int(scores["status"][index, column]) for column, spec in enumerate(specs)},
                distortion=float(bank["distortion"][index]),
                total_displacement_m=float(bank["displacement_m"][index].sum()),
                max_displacement_m=float(bank["displacement_m"][index].max()),
                edits=[dict(node=node, before=original_xy[owner, node].tolist(), after=xy[node].tolist(),
                            displacement_m=float(bank["displacement_m"][index, node])) for node in node_ids])


def graph_tensors(sample, device):
    nodes = torch.tensor(sample.nodes, device=device, dtype=torch.float64)
    if nodes.ndim != 2 or nodes.shape[1] < 9 or not len(nodes):
        raise ValueError("Repair needs decoded XY and seven semantic fields")
    count = len(nodes)
    adjacency = torch.zeros((1, count, count), device=device, dtype=torch.bool)
    for first, second in sample.edges:
        adjacency[0, min(first, second), max(first, second)] = True
    return dict(xy=nodes[None, :, :2], semantics=nodes[None, :, 2:9],
                node_mask=torch.ones((1, count), device=device, dtype=torch.bool), adjacency=adjacency)


def fixed_graph_sample(adapter, original, graph, index):
    """Re-extract the edited fixed-edge graph; never predict its edges again."""
    mask = graph["node_mask"][index]
    xy = graph["xy"][index, mask].cpu().numpy()[None]
    semantics = graph["semantics"][index, mask].cpu().numpy()[None]
    adjacency = graph["adjacency"][index][mask][:, mask].cpu().numpy()
    sample = FloorplanSample(nodes=np.concatenate((xy, semantics), -1)[0].tolist(),
                             metadata=deepcopy(original.metadata), scenario=deepcopy(original.scenario))
    adapter._finish_graph(sample, xy, semantics, adjacency)
    sample.metadata["source_geometry"] = "solver_modified_existing_junctions"
    sample.metadata["original_geometry_artifact"] = "pre_repair"
    return sample


def _cpu_guard(original, candidate, before, after, scenario):
    from shapely.geometry import Polygon
    from shapely.ops import unary_union
    if not after.structurally_valid or not math.isfinite(after.energy):
        return False
    old = {result.id: result for result in before.results if result.counts_toward_compliance and result.applicable}
    new = {result.id: result for result in after.results if result.counts_toward_compliance and result.applicable}
    if old.keys() != new.keys():
        return False
    for name, value in old.items():
        other = new[name]
        if other.status == "unknown" and value.status != "unknown":
            return False
        if value.status == "pass" and other.status != "pass":
            return False
        if value.measurable and other.measurable and other.severity > value.severity + 1e-8:
            return False
    if [node[2:] for node in original.nodes] != [node[2:] for node in candidate.nodes]:
        return False
    if sorted(map(sorted, original.edges)) != sorted(map(sorted, candidate.edges)):
        return False
    # Compare each labelled face by junction membership, not its possibly reordered index.
    def signatures(sample):
        lookup = {tuple(node[:2]): index for index, node in enumerate(sample.nodes)}
        return sorted((room.label, tuple(sorted({lookup[tuple(point[:2])] for point in room.polygon})))
                      for room in sample.rooms)
    if signatures(original) != signatures(candidate):
        return False
    old_shape = unary_union([Polygon(room.polygon) for room in original.rooms])
    new_shape = unary_union([Polygon(room.polygon) for room in candidate.rooms])
    return old_shape.symmetric_difference(new_shape).area * scenario.coordinate_scale_m_per_model_unit**2 <= 1e-8


def final_repair(adapter, sample, evaluator, scenario, tensor_evaluator, config=None, batch_size=10):
    """Audit a bounded GPU-ranked bank; identity remains the safe fallback."""
    config = repair_settings(config)
    sample = deepcopy(sample)
    cpu_started = time.perf_counter()
    before = evaluator.evaluate(sample, scenario)
    cpu_checker_seconds = time.perf_counter() - cpu_started
    if not config["final_enabled"] or not before.structurally_valid or overall_validity(before) == "valid":
        return sample, before, dict(search_status="already_feasible" if overall_validity(before) == "valid" else "unchanged",
                                    gpu_evaluations=0, cpu_evaluations=1, cpu_checker_seconds=cpu_checker_seconds, edits=[], total_displacement_m=0.0,
                                    max_displacement_m=0.0)
    graph = graph_tensors(sample, adapter.device)
    bank = score_repairs(graph, tensor_evaluator, scenario, config,
                         max_evaluations=config["final_evaluations"], batch_size=batch_size)
    best, best_eval = sample, before
    best_key = (overall_validity(before) != "valid", before.unknown_count, before.energy, 0.0)
    winner = None
    audits = 1
    rejected = 0
    for index in bank["rankings"][0].tolist():
        if not bool(bank["eligible"][index]) or not bool((bank["displacement_m"][index] > 1e-8).any()):
            continue
        candidate = fixed_graph_sample(adapter, sample, bank["graph"], index)
        cpu_started = time.perf_counter()
        after = evaluator.evaluate(candidate, scenario)
        cpu_checker_seconds += time.perf_counter() - cpu_started
        audits += 1
        if not _cpu_guard(sample, candidate, before, after, scenario):
            rejected += 1
            continue
        key = (overall_validity(after) != "valid", after.unknown_count, after.energy, float(bank["distortion"][index]))
        if key < best_key:
            best, best_eval, best_key, winner = candidate, after, key, index
    record = (selection_record(bank, winner, tensor_evaluator.specs, graph["xy"], scenario.coordinate_scale_m_per_model_unit)
              if winner is not None else dict(edits=[], distortion=0.0, total_displacement_m=0.0, max_displacement_m=0.0))
    proxy = {name: record.pop(name) for name in ("structurally_valid", "energy", "unknown_count", "feasible", "status_by_id") if name in record}
    record.update(proxy_evaluation=proxy,
                  feasible=overall_validity(best_eval) == "valid", energy=best_eval.energy,
                  unknown_count=best_eval.unknown_count, structurally_valid=best_eval.structurally_valid,
                  status_by_id={value.id: value.status for value in best_eval.results},
                  search_status="feasible" if overall_validity(best_eval) == "valid" else "improved" if winner is not None else "no_improvement",
                  gpu_evaluations=bank["evaluations"], cpu_evaluations=audits, cpu_rejected=rejected,
                  cpu_checker_seconds=cpu_checker_seconds,
                  final_evaluation_backend="cpu_placement_audit_v2", global_minimum_proven=False)
    best.metadata["repair"] = record
    return best, best_eval, record


def save_repair_overlay(before, after, destination):
    """Export vector provenance: original walls, edited walls and junction arrows."""
    from html import escape
    from pathlib import Path
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    points = [node[:2] for sample in (before, after) for node in sample.nodes]
    if not points or not np.isfinite(points).all():
        return None
    coordinates = np.asarray(points)
    low, high = coordinates.min(0), coordinates.max(0)
    factor = min(900 / max(high[0] - low[0], 1), 620 / max(high[1] - low[1], 1))
    def point(node):
        value = (np.asarray(node[:2]) - low) * factor + [50, 100]
        return f'{value[0]:.3f}', f'{value[1]:.3f}'
    lines = ['<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="780" viewBox="0 0 1000 780">',
             '<rect width="100%" height="100%" fill="white"/>',
             '<defs><marker id="move" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0,0 L7,3.5 L0,7 Z" fill="#c43a31"/></marker></defs>',
             '<g font-family="Arial,sans-serif" fill="#222"><text x="30" y="32" font-size="22">Bounded final repair: actual geometric edits</text>',
             '<text x="30" y="61" font-size="15">Dashed grey: original walls · Teal: returned walls · Red arrows: moved junctions</text></g>']
    for sample, color, dash in ((before, '#9d9d9d', ' stroke-dasharray="7 5"'), (after, '#147d75', '')):
        for first, second in sample.edges:
            x1, y1 = point(sample.nodes[first]); x2, y2 = point(sample.nodes[second])
            lines.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="2.5"{dash}/>')
    moved = 0
    for index, (old, new) in enumerate(zip(before.nodes, after.nodes)):
        if np.linalg.norm(np.asarray(old[:2]) - np.asarray(new[:2])) <= 1e-8:
            continue
        moved += 1
        x1, y1 = point(old); x2, y2 = point(new)
        lines.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="#c43a31" stroke-width="2" marker-end="url(#move)"/>')
        lines.append(f'<text x="{float(x2)+6:.3f}" y="{float(y2)-6:.3f}" font-family="Arial,sans-serif" font-size="13" fill="#c43a31">J{index}</text>')
    caption = f'{moved} junctions moved; fixed node labels and wall connectivity. Metric distances use the declared scale.'
    lines.append(f'<text x="30" y="757" font-family="Arial,sans-serif" font-size="14">{escape(caption)}</text></svg>')
    destination.write_text('\n'.join(lines) + '\n')
    return str(destination)
