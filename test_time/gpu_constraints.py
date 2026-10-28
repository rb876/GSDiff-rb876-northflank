"""Tensor-only constraint scoring with conservative finite-grid witnesses.

The input polygons are simple, hole-free, non-overlapping faces verified by the
GPU graph extractor. Search misses remain unknown. This backend intentionally
does not claim numerical equivalence with the Shapely/polylabel evaluator.
"""

import math
import torch

from .area import area_policy

PASS, FAIL, UNKNOWN, NOT_APPLICABLE = 0, 1, 2, 3
EPS = 1e-8


def _labels_in(labels, allowed):
    result = torch.zeros_like(labels, dtype=torch.bool)
    for label in allowed:
        result |= labels == label
    return result


def _edges(polygons, vertex_mask):
    count = vertex_mask.sum(-1)
    next_index = (torch.arange(polygons.shape[-2], device=polygons.device) + 1) % count.clamp_min(1)[..., None]
    ends = polygons.gather(-2, next_index[..., None].expand_as(polygons))
    return polygons, ends, vertex_mask


def _point_metrics(points, starts, ends, edge_mask):
    """Covered-by and signed boundary distance for Q groups of P points."""
    p = points[..., None, :]
    a, b = starts[:, None], ends[:, None]
    delta = b - a
    length2 = (delta * delta).sum(-1)
    projection = ((p - a) * delta).sum(-1) / length2.clamp_min(1e-30)
    closest = a + projection.clamp(0, 1)[..., None] * delta
    distance2 = ((p - closest) ** 2).sum(-1).masked_fill(~edge_mask[:, None], torch.inf)
    distance = distance2.amin(-1).sqrt()
    py, px = p[..., 1], p[..., 0]
    ay, by = a[..., 1], b[..., 1]
    crossing_y = (ay > py) != (by > py)
    dy = by - ay
    safe_dy = torch.where(dy.abs() > 1e-30, dy, torch.ones_like(dy))
    intersect_x = a[..., 0] + (py - ay) * delta[..., 0] / safe_dy
    odd = ((crossing_y & (px < intersect_x) & edge_mask[:, None]).sum(-1) % 2) == 1
    inside = odd | (distance <= EPS)
    return inside, torch.where(inside, distance, -distance)


def _rectangles_fit(low, high, starts, ends, edge_mask):
    """Entire rectangle containment, including concave intrusions.

    Corner tests alone are insufficient. Reject any polygon edge entering the
    rectangle's open interior; boundary coincidence remains permitted.
    """
    corners = torch.stack((low, torch.stack((high[:, 0], low[:, 1]), -1), high,
                           torch.stack((low[:, 0], high[:, 1]), -1)), 1)
    covered, _ = _point_metrics(corners, starts, ends, edge_mask)
    direction = ends - starts
    parallel = direction.abs() <= 1e-30
    safe = torch.where(parallel, torch.ones_like(direction), direction)
    first = (low[:, None] + EPS - starts) / safe
    last = (high[:, None] - EPS - starts) / safe
    enter = torch.minimum(first, last)
    leave = torch.maximum(first, last)
    parallel_inside = (starts > low[:, None] + EPS) & (starts < high[:, None] - EPS)
    enter = torch.where(parallel, torch.where(parallel_inside, -torch.inf, torch.inf), enter)
    leave = torch.where(parallel, torch.where(parallel_inside, torch.inf, -torch.inf), leave)
    t0 = enter.amax(-1).clamp_min(0)
    t1 = leave.amin(-1).clamp_max(1)
    intrudes = ((t1 > t0 + 1e-12) & edge_mask).any(-1)
    return covered.all(-1) & ~intrudes


def _grid(low, high, step, budget):
    """Packed room/grid indices; boundaries and midpoint precede regular grid."""
    room_count = low.shape[0]
    ids = torch.arange(budget, device=low.device)
    special = torch.tensor([[0, 0], [1, 1], [.5, .5], [0, 1], [1, 0],
                            [.5, 0], [.5, 1], [0, .5], [1, .5]], device=low.device, dtype=low.dtype)
    span = high - low
    nx = (span[:, 0].clamp_min(0) / step).floor().to(torch.long) + 1
    ny = (span[:, 1].clamp_min(0) / step).floor().to(torch.long) + 1
    linear = (ids - 9).clamp_min(0)[None].expand(room_count, -1)
    ix, iy = linear % nx[:, None], linear // nx[:, None]
    regular = low[:, None] + torch.stack((ix, iy), -1) * step
    endpoints = low[:, None] + special[ids.clamp_max(8)][None] * span[:, None]
    points = torch.where((ids < 9)[None, :, None], endpoints, regular)
    valid = (span >= -EPS).all(-1)[:, None] & ((ids < 9)[None] | (linear < (nx * ny)[:, None]))
    room, position = torch.nonzero(valid, as_tuple=True)
    complete = nx * ny + 9 <= budget
    return room, position, points[room, position], complete, points


def _circle_search(starts, ends, masks, low, high, areas, radii, steps, budget, chunk, directions):
    rooms = starts.shape[0]
    device, dtype = starts.device, starts.dtype
    theta = torch.arange(directions, device=device, dtype=dtype) * (2 * math.pi / directions)
    axes = torch.stack((theta.cos(), theta.sin()), -1)
    best_scores = torch.full((rooms, directions), -torch.inf, device=device, dtype=dtype)
    witnesses = torch.zeros((rooms, directions, 2), device=device, dtype=dtype)
    lower = torch.zeros(rooms, device=device, dtype=dtype)
    upper = torch.minimum((high - low).amin(-1) / 2, (areas / math.pi).clamp_min(0).sqrt())
    positions_examined = torch.zeros(rooms, device=device, dtype=torch.long)
    for step in steps:
        room, position, points, _, dense_points = _grid(
            low + radii[:, None], high - radii[:, None], step, budget)
        signed = torch.full((rooms, budget), -torch.inf, device=device, dtype=dtype)
        for begin in range(0, points.shape[0], chunk):
            selected = room[begin:begin + chunk]
            _, distance = _point_metrics(points[begin:begin + chunk, None], starts[selected], ends[selected], masks[selected])
            signed[selected, position[begin:begin + chunk]] = distance[:, 0]
        phase_lower = signed.amax(-1)
        lower = torch.maximum(lower, phase_lower.clamp_min(0))
        fits = signed + EPS >= radii[:, None]
        scores = torch.einsum("rpd,kd->rpk", dense_points, axes).masked_fill(~fits[..., None], -torch.inf)
        phase_scores, choices = scores.max(1)
        coordinates = dense_points.gather(1, choices[..., None].expand(-1, -1, 2))
        improved = phase_scores > best_scores
        witnesses = torch.where(improved[..., None], coordinates, witnesses)
        best_scores = torch.maximum(best_scores, phase_scores)
        positions_examined.scatter_add_(0, room, torch.ones_like(room))
    witnessed = torch.isfinite(best_scores)
    return witnesses, witnessed, lower, upper.clamp_min(0), positions_examined


def _bed_search(starts, ends, masks, low, high, areas, circles, circle_mask,
                width, length, radius, clearances, steps, budget, chunk):
    rooms = starts.shape[0]
    device, dtype = starts.device, starts.dtype
    # PyTorch 2.1 CUDA scatter_reduce_ does not dispatch boolean reductions.
    bed_found = torch.zeros(rooms, device=device, dtype=torch.long)
    turning_found, circulation_found, joint_found = bed_found.clone(), bed_found.clone(), bed_found.clone()
    bed_witness = torch.zeros((rooms, 4), device=device, dtype=dtype)
    circle_witness = torch.zeros((rooms, 2), device=device, dtype=dtype)
    examined = torch.zeros(rooms, device=device, dtype=torch.long)
    for step in steps:
        for horizontal in (False, True):
            size = torch.tensor([length, width] if horizontal else [width, length], device=device, dtype=dtype)
            room, _, points, _, _ = _grid(low, high - size, step, budget)
            for begin in range(0, points.shape[0], chunk):
                selected = room[begin:begin + chunk]
                bottom = points[begin:begin + chunk]
                top = bottom + size
                fits = _rectangles_fit(bottom, top, starts[selected], ends[selected], masks[selected])
                examined.scatter_add_(0, selected, torch.ones_like(selected))
                # Dynamic packing avoids testing circulation for rejected beds.
                keep = torch.nonzero(fits, as_tuple=True)[0]
                selected, bottom, top = selected[keep], bottom[keep], top[keep]
                if selected.numel() == 0:
                    continue
                bed_found.scatter_reduce_(0, selected, torch.ones_like(selected), reduce="amax", include_self=True)
                centers = circles[selected]
                offset = torch.maximum(torch.maximum(bottom[:, None] - centers, centers - top[:, None]), torch.zeros_like(centers))
                free = ((offset * offset).sum(-1) + EPS >= radius**2) & circle_mask[selected]
                turning = free.any(-1)
                turning_found.scatter_reduce_(0, selected, turning.long(), reduce="amax", include_self=True)
                circulation = torch.zeros_like(turning)
                for side, foot in clearances:
                    for direction in (1, -1):
                        if horizontal:
                            left = [-foot if direction < 0 else 0, -side]
                            right = [foot if direction > 0 else 0, side]
                        else:
                            left = [-side, -foot if direction < 0 else 0]
                            right = [side, foot if direction > 0 else 0]
                        envelope_low = bottom + bottom.new_tensor(left)
                        envelope_high = top + top.new_tensor(right)
                        inside_bbox = ((envelope_low + EPS >= low[selected]) &
                                       (envelope_high <= high[selected] + EPS)).all(-1)
                        positions = torch.nonzero(inside_bbox, as_tuple=True)[0]
                        accepted = _rectangles_fit(envelope_low[positions], envelope_high[positions],
                                                   starts[selected[positions]], ends[selected[positions]], masks[selected[positions]])
                        circulation[positions] |= accepted
                circulation_found.scatter_reduce_(0, selected, circulation.long(), reduce="amax", include_self=True)
                joint = turning & circulation
                joint_found.scatter_reduce_(0, selected, joint.long(), reduce="amax", include_self=True)
                # Select one deterministic (lowest packed index) joint witness per room.
                index = torch.arange(selected.shape[0], device=device)
                chosen = torch.full((rooms,), selected.shape[0], device=device, dtype=torch.long)
                chosen.scatter_reduce_(0, selected, torch.where(joint, index, selected.shape[0]), reduce="amin", include_self=True)
                witness_rooms = torch.nonzero(chosen < selected.shape[0], as_tuple=True)[0]
                chosen_indices = chosen[witness_rooms]
                bed_witness[witness_rooms] = torch.cat((bottom[chosen_indices], top[chosen_indices]), -1)
                which_circle = free[chosen_indices].to(torch.long).argmax(-1)
                circle_witness[witness_rooms] = centers[chosen_indices, which_circle]
    return {"bed": bed_found.bool(), "turning": turning_found.bool(), "circulation": circulation_found.bool(),
            "joint": joint_found.bool(), "bed_witness": bed_witness,
            "circle_witness": circle_witness, "positions_examined": examined}


class TensorConstraintEvaluator:
    def __init__(self, specs, validity_weight=100.0, *, chunk_positions=512,
                 circle_directions=32, max_positions=None):
        self.all_specs = list(specs)
        self.specs = [spec for spec in self.all_specs if spec.enabled]
        self.ids = [spec.id for spec in self.specs]
        self.by_id = {spec.id: spec for spec in self.all_specs}
        self.validity_weight = float(validity_weight)
        self.chunk_positions = int(chunk_positions)
        self.circle_directions = int(circle_directions)
        configured = self.by_id["C033"].thresholds["max_positions"] if "C033" in self.by_id else 20000
        self.max_positions = configured if max_positions is None else int(max_positions)
        if self.chunk_positions < 1 or self.circle_directions < 1 or self.max_positions < 36:
            raise ValueError("GPU search needs positive chunks/directions and max_positions >= 36")
        self.metadata = {
            "backend": "torch_finite_grid_v1", "coordinate_unit": "drawing_pixel",
            "circle_witness_directions": self.circle_directions,
            "max_bed_positions_per_room": self.max_positions,
            "geometry": "Simple hole-free verified non-overlapping faces; full rectangle containment",
            "search_limitations": "Finite grids and retained directional circle witnesses can miss feasible placements; misses are unknown",
            "cpu_oracle_equivalence": False,
            "circle_failure_severity": "Normalized deficit of the proven bounding-box/area inradius upper bound",
        }

    @torch.no_grad()
    def evaluate(self, rooms, scenario):
        polygons = rooms["polygons"].to(torch.float64)
        vertex_mask, room_mask = rooms["vertex_mask"].bool(), rooms["room_mask"].bool()
        labels = rooms["labels"]
        batch, room_slots = room_mask.shape
        device, dtype = polygons.device, polygons.dtype
        finite = (torch.isfinite(polygons) | ~vertex_mask[..., None]).flatten(-2).all(-1)
        polygons = torch.nan_to_num(polygons)
        scale = scenario.coordinate_scale_m_per_model_unit
        metric = polygons * (scale if scale is not None else 1.0)
        starts, ends, edge_mask = _edges(metric, vertex_mask)
        cross = starts[..., 0] * ends[..., 1] - starts[..., 1] * ends[..., 0]
        areas = torch.where(vertex_mask, cross, 0).sum(-1).abs() / 2
        area_rules = area_policy(self.all_specs, scenario)
        self.metadata["habitable_area_policy"] = area_rules
        habitable_areas = areas
        if area_rules["enabled"]:
            included = room_mask & _labels_in(labels, area_rules["included_room_labels"])
            if not area_rules["available"]:
                habitable_areas = torch.full_like(areas, torch.nan)
            elif area_rules["excluded_by_height"]:
                habitable_areas = torch.zeros_like(areas)
            else:
                habitable_areas = torch.where(included, areas, 0)
        valid_room = finite & torch.isfinite(areas) & (vertex_mask.sum(-1) >= 3) & (areas > EPS) & (labels >= 0) & (labels <= 6)
        valid = rooms["structurally_valid"].bool() & room_mask.any(-1) & (~room_mask | valid_room).all(-1)
        if "has_holes" in rooms:
            valid &= ~rooms["has_holes"].bool().reshape(batch, -1).any(-1)
        status = torch.full((batch, len(self.specs)), UNKNOWN, device=device, dtype=torch.long)
        severity = torch.full((batch, len(self.specs)), torch.nan, device=device, dtype=dtype)
        count_flags = torch.tensor([spec.counts_toward_compliance for spec in self.specs], device=device, dtype=torch.bool)
        weights = torch.tensor([spec.weight for spec in self.specs], device=device, dtype=dtype)
        assume_flags = torch.tensor([bool(spec.assumptions) or scenario.scale_provenance == "assumed" or
                                     "ceiling_height" in spec.required_observations for spec in self.specs], device=device)
        low = starts.masked_fill(~vertex_mask[..., None], torch.inf).amin(-2)
        high = starts.masked_fill(~vertex_mask[..., None], -torch.inf).amax(-2)
        low, high = torch.nan_to_num(low), torch.nan_to_num(high)
        geometry = {}
        bed_ids = {"C033", "C034", "C035"}
        need_beds = bool(bed_ids & set(self.ids)) and bed_ids <= self.by_id.keys()
        need_bath = "C036" in self.ids
        bedroom_labels = self.by_id["C034"].thresholds["bedroom_labels"] if need_beds else []
        bathroom_labels = self.by_id["C036"].thresholds["bathroom_labels"] if need_bath else []
        bed_rooms = room_mask & _labels_in(labels, bedroom_labels)
        bath_rooms = room_mask & _labels_in(labels, bathroom_labels)
        self.metadata.update(coarse_grid_m=scenario.geometry_grid_m, refine_grid_m=scenario.geometry_refine_m)
        if scale is not None and scenario.building_use == "residential" and scenario.accessible_dwelling and (need_beds or need_bath):
            packed_batch, packed_room = torch.nonzero((bed_rooms | bath_rooms) & valid[:, None], as_tuple=True)
            pstarts, pends = starts[packed_batch, packed_room], ends[packed_batch, packed_room]
            pmasks = edge_mask[packed_batch, packed_room]
            plow, phigh, parea = low[packed_batch, packed_room], high[packed_batch, packed_room], areas[packed_batch, packed_room]
            is_bed = bed_rooms[packed_batch, packed_room]
            bedroom_radius = self.by_id["C033"].thresholds["diameter_m"] / 2 if need_beds else 0.75
            bathroom_radius = self.by_id["C036"].thresholds["diameter_m"] / 2 if need_bath else 0.75
            radii = torch.where(is_bed, bedroom_radius, bathroom_radius).to(dtype)
            steps = tuple(dict.fromkeys((scenario.geometry_grid_m, scenario.geometry_refine_m)))
            circles, circle_mask, lower, upper, circle_examined = _circle_search(
                pstarts, pends, pmasks, plow, phigh, parea, radii, steps,
                max(9, self.max_positions // len(steps)), self.chunk_positions, self.circle_directions)
            for name, values in (("circle_pass", circle_mask.any(-1)), ("circle_lower", lower), ("circle_upper", upper),
                                 ("circle_positions", circle_examined)):
                dense = torch.zeros((batch, room_slots), device=device, dtype=values.dtype)
                dense[packed_batch, packed_room] = values
                geometry[name] = dense
            if need_beds and scenario.intended_occupants is not None:
                positions = torch.nonzero(is_bed, as_tuple=True)[0]
                bed = self.by_id["C034"].thresholds
                width = bed["single_width_m"] if scenario.intended_occupants == 1 else bed["multiple_width_m"]
                clearance = self.by_id["C035"].thresholds["clearance_options_m"]
                found = _bed_search(pstarts[positions], pends[positions], pmasks[positions], plow[positions], phigh[positions],
                                    parea[positions], circles[positions], circle_mask[positions], width, bed["length_m"],
                                    bedroom_radius, clearance, steps, max(9, self.max_positions // (2 * len(steps))), self.chunk_positions)
                for name, values in found.items():
                    dense = torch.zeros((batch, room_slots, *values.shape[1:]), device=device, dtype=values.dtype)
                    dense[packed_batch[positions], packed_room[positions]] = values
                    geometry[name] = dense
                geometry["bed_width_m"] = width
                geometry["bed_length_m"] = bed["length_m"]

        def area_for(allowed):
            return torch.where(room_mask & _labels_in(labels, allowed), areas, 0).sum(-1)

        def existence(witness, impossible, target):
            passed = (witness & target).any(-1)
            failed = (~target | impossible).all(-1)
            return torch.where(passed, PASS, torch.where(failed, FAIL, UNKNOWN))

        for column, spec in enumerate(self.specs):
            if spec.checker in {"area_definition", "bed_definition"}:
                status[:, column] = NOT_APPLICABLE
                continue
            applicable = scenario.building_use == "residential" and (spec.applicability == "residential" or scenario.accessible_dwelling)
            if not applicable:
                status[:, column] = NOT_APPLICABLE
                continue
            available = {"scale": scale, "occupants": scenario.intended_occupants, "ceiling_height": scenario.ceiling_height_m}
            if any(name in available and available[name] is None for name in spec.required_observations):
                continue
            if (spec.checker in {"minimum_main_room", "minimum_surface", "minimum_volume"}
                    and area_rules["enabled"] and not area_rules["available"]):
                continue
            limits = spec.thresholds
            measured = torch.zeros(batch, device=device, dtype=dtype)
            if spec.checker in {"bedroom_turning", "bed_footprint", "bedroom_circulation"}:
                if "bed" not in geometry:
                    continue
                bed_area = geometry["bed_width_m"] * geometry["bed_length_m"]
                circle_radius = self.by_id["C033"].thresholds["diameter_m"] / 2
                if spec.checker == "bed_footprint":
                    witness, impossible = geometry["bed"], areas + EPS < bed_area
                elif spec.checker == "bedroom_turning":
                    witness = geometry["turning"]
                    impossible = (areas + EPS < bed_area + math.pi * circle_radius**2) | (geometry["circle_upper"] + EPS < circle_radius)
                else:
                    shared = self.by_id["C033"].enabled
                    witness = geometry["joint"] if shared else geometry["circulation"]
                    minimum = min((geometry["bed_width_m"] + 2 * side) * (geometry["bed_length_m"] + foot)
                                  for side, foot in limits["clearance_options_m"])
                    impossible = areas + EPS < minimum
                current = existence(witness, impossible, bed_rooms)
                measured = (current == FAIL).to(dtype)
            elif spec.checker == "bathroom_turning":
                if "circle_pass" not in geometry:
                    continue
                radius = limits["diameter_m"] / 2
                current = existence(geometry["circle_pass"], geometry["circle_upper"] + EPS < radius, bath_rooms)
                radius_bound = geometry["circle_upper"].masked_fill(~bath_rooms, 0).amax(-1)
                measured = ((radius - radius_bound) / radius).clamp_min(0)
                measured = torch.where(current == PASS, 0, measured)
            elif spec.checker == "minimum_main_room":
                target = room_mask & _labels_in(labels, limits["main_room_labels"])
                if area_rules["enabled"]:
                    target &= _labels_in(labels, area_rules["included_room_labels"])
                height = scenario.ceiling_height_m
                area_deficit = ((limits["area_m2"] - habitable_areas) / limits["area_m2"]).clamp_min(0)
                height_deficit = max(0.0, (limits["height_m"] - height) / limits["height_m"])
                volume_deficit = ((limits["volume_m3"] - habitable_areas * height) / limits["volume_m3"]).clamp_min(0)
                room_severity = torch.minimum(torch.maximum(area_deficit, area_deficit.new_tensor(height_deficit)), volume_deficit)
                measured = room_severity.masked_fill(~target, torch.inf).amin(-1)
                measured = torch.where(target.any(-1), measured, 1.0)
                current = torch.where(measured <= EPS, PASS, FAIL)
            elif spec.checker in {"minimum_surface", "minimum_volume"}:
                occupants = scenario.intended_occupants
                observed = (torch.where(room_mask, habitable_areas, 0).sum(-1) if area_rules["enabled"]
                            else area_for(limits["included_room_labels"]))
                if spec.checker == "minimum_surface":
                    minimum = limits["first_four_m2"] * min(occupants, 4) + limits["additional_m2"] * max(occupants - 4, 0)
                else:
                    minimum = limits["first_four_m3"] * min(occupants, 4) + limits["additional_m3"] * max(occupants - 4, 0)
                    observed = observed * scenario.ceiling_height_m
                measured = ((minimum - observed) / minimum).clamp_min(0)
                current = torch.where(measured <= EPS, PASS, FAIL)
            else:
                raise ValueError(f"Unsupported GPU checker: {spec.checker}")
            current = torch.where(valid, current, UNKNOWN)
            status[:, column] = current
            severity[:, column] = torch.where(current == PASS, 0, torch.where(current == FAIL, measured, torch.nan))

        applicable = status != NOT_APPLICABLE
        measurable = (status == PASS) | (status == FAIL)
        counted = applicable & count_flags[None]
        regulatory = (torch.where(measurable, severity, 0) * weights[None]).sum(-1)
        energy = regulatory + self.validity_weight * (~valid).to(dtype)
        result = {"energy": energy, "reward": -energy, "status": status, "severity": severity,
                "applicable": applicable, "measurable": measurable, "structurally_valid": valid,
                "unknown_count": ((status == UNKNOWN) & counted).sum(-1),
                "measurable_count": (measurable & counted).sum(-1), "applicable_count": counted.sum(-1),
                "assumption_based_count": (counted & assume_flags[None]).sum(-1),
                "validity_penalty": (~valid).to(dtype), "diagnostics": rooms["diagnostics"],
                "room_areas_m2": areas}
        if area_rules["enabled"]:
            result.update(habitable_room_areas_m2=habitable_areas,
                          habitable_area_m2=torch.where(room_mask, habitable_areas, 0).sum(-1))
        result.update({"geometry_" + name: value for name, value in geometry.items()
                       if isinstance(value, torch.Tensor)})
        return result

    __call__ = evaluate
