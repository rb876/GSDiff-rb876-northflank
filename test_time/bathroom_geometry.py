"""Bounded, shared bathroom completions; a missed witness is not a proof.

Fixtures use declared scenario dimensions. C036/C037/C041 use one candidate
layout and one turning circle. The completion bank is deliberately finite:
four rotations, reflection, three wall offsets per object, and two hinges.
All guidance calculations stay on the input tensor device.
"""

import itertools
import math

import torch


IDS = ("C036", "C037", "C041")
EPS = 1e-7


def _rect_corners(bounds):
    low, high = bounds[..., :2], bounds[..., 2:]
    return torch.stack((low, torch.stack((high[..., 0], low[..., 1]), -1), high,
                        torch.stack((low[..., 0], high[..., 1]), -1)), -2)


def _world_rectangle(bounds, origin, u, v):
    corners = _rect_corners(bounds)
    world = origin[..., None, :] + corners[..., :1] * u[..., None, :] + corners[..., 1:] * v[..., None, :]
    return torch.cat((world.amin(-2), world.amax(-2)), -1)


def _rectangle_overlap(first, second):
    widths = torch.minimum(first[..., 2:], second[..., 2:]) - torch.maximum(first[..., :2], second[..., :2])
    return widths.amin(-1).clamp_min(0)


def _circle_rectangle_distance(center, rectangle):
    delta = torch.maximum(torch.maximum(rectangle[..., :2] - center, center - rectangle[..., 2:]),
                          torch.zeros_like(center))
    return delta.square().sum(-1).sqrt()


def _sector_rectangle_depth(rectangle, hinge, u, v, radius):
    """Positive iff an axis-aligned rectangle overlaps a cardinal quarter disk."""
    corners = _rect_corners(rectangle) - hinge[..., None, :]
    local = torch.stack(((corners * u[..., None, :]).sum(-1),
                         (corners * v[..., None, :]).sum(-1)), -1)
    low, high = local.amin(-2), local.amax(-2)
    clipped = low.clamp_min(0)
    radial = (radius - clipped.square().sum(-1).sqrt()).clamp_min(0)
    # Positive width in both quadrant directions is required; boundary touch is OK.
    span = high - clipped
    return torch.minimum(radial, span.amin(-1).clamp_min(0))


def _sector_distance(points, hinge, u, v, radius):
    """Signed distance to a closed quarter disk, negative in its interior."""
    delta = points - hinge[..., None, :]
    x = (delta * u[..., None, :]).sum(-1)
    y = (delta * v[..., None, :]).sum(-1)
    norm = (x.square() + y.square()).sqrt()
    edge_x = ((x - x.clamp(0, radius)).square() + y.square()).sqrt()
    edge_y = (x.square() + (y - y.clamp(0, radius)).square()).sqrt()
    arc = torch.where((x >= 0) & (y >= 0), (norm - radius).abs(), torch.inf)
    outside = torch.minimum(torch.minimum(edge_x, edge_y), arc)
    inside = (x >= 0) & (y >= 0) & (norm <= radius)
    inner = torch.minimum(torch.minimum(x, y), radius - norm)
    return torch.where(inside, -inner, outside)


def _sector_containment(hinge, u, v, radius, starts, ends, masks):
    """Containment of an exact sector in a simple hole-free room polygon.

    The connected sector interior is contained when its interior reference point
    is inside and no room boundary segment enters its open interior.
    """
    from .gpu_constraints import _point_metrics

    midpoint = hinge + (u + v) * (radius / 4)
    covered, signed = _point_metrics(midpoint[:, None], starts, ends, masks)
    a = starts - hinge[:, None]
    b = ends - hinge[:, None]
    a = torch.stack(((a * u[:, None]).sum(-1), (a * v[:, None]).sum(-1)), -1)
    b = torch.stack(((b * u[:, None]).sum(-1), (b * v[:, None]).sum(-1)), -1)
    direction = b - a
    parallel = direction.abs() < 1e-20
    safe = torch.where(parallel, torch.ones_like(direction), direction)
    crossing = (EPS - a) / safe
    lower = torch.where(direction > 0, crossing, -torch.inf)
    upper = torch.where(direction < 0, crossing, torch.inf)
    lower = torch.where(parallel & (a <= EPS), torch.inf, lower)
    t0, t1 = lower.amax(-1).clamp_min(0), upper.amin(-1).clamp_max(1)
    nearest = (-(a * direction).sum(-1) / direction.square().sum(-1).clamp_min(1e-30))
    nearest = torch.maximum(t0, torch.minimum(t1, nearest))
    distance = (a + nearest[..., None] * direction).square().sum(-1).sqrt()
    penetration = (radius - distance).clamp_min(0)
    intrudes = (t1 > t0 + EPS) & (distance < radius - EPS) & masks
    violation = torch.where(intrudes, penetration, torch.zeros_like(penetration)).amax(-1)
    violation += (-signed[:, 0]).clamp_min(0)
    return covered[:, 0] & ~intrudes.any(-1), violation


def _rectangle_containment(rectangle, starts, ends, masks):
    from .gpu_constraints import _point_metrics, _rectangles_fit

    fit = _rectangles_fit(rectangle[:, :2], rectangle[:, 2:], starts, ends, masks)
    _, signed = _point_metrics(_rect_corners(rectangle), starts, ends, masks)
    violation = (-signed).clamp_min(0).amax(-1)
    # An edge can intrude into a concave room even when all corners are inside.
    violation = torch.maximum(violation, (~fit).to(rectangle.dtype) * .01)
    return fit, violation


def search_bathrooms(starts, ends, edge_mask, low, high, areas, scenario, limits):
    """Return one shared completion per room, with tensor-only geometry.

    ``status`` columns follow IDS. A pass has an explicit witness; failure needs
    a necessary size/area bound. Other misses are unknown. ``guidance_penalty``
    is a separate continuous search surrogate, not a certified violation count.
    """
    from .gpu_constraints import _point_metrics

    device, dtype = starts.device, starts.dtype
    rooms = starts.shape[0]
    radius = float(limits["diameter_m"]) / 2
    transfer_width = float(limits["transfer_width_m"])
    transfer_length = float(limits["transfer_length_m"])
    door_limit, basin_limit = float(limits["door_overlap_m"]), float(limits["basin_overlap_m"])
    wc_width, wc_depth = scenario.wc_width_m, scenario.wc_depth_m
    basin_width, basin_depth = scenario.basin_width_m, scenario.basin_depth_m
    door_width = scenario.bathroom_door_width_m
    templates = list(itertools.product(range(4), range(2), (0., .5, 1.), (0., .5, 1.),
                                       (0., .5, 1.), range(2)))[:scenario.bathroom_search_budget]
    bank = torch.tensor(templates, device=device, dtype=dtype)
    positions = scenario.bathroom_circle_positions
    if rooms == 0:
        return {"status": torch.empty((0, 3), device=device, dtype=torch.long),
                "severity": torch.empty((0, 3), device=device, dtype=dtype),
                "guidance_penalty": torch.empty(0, device=device, dtype=dtype), "witness": {},
                "search_budget": len(templates), "circle_positions": positions,
                "door_radius_m": door_width, "circle_radius_m": radius,
                "basin_under_clearance": bool(scenario.basin_under_clearance)}
    side = math.ceil(math.sqrt(positions))
    axis = (torch.tensor([.5], device=device, dtype=dtype) if positions == 1
            else torch.linspace(0, 1, side, device=device, dtype=dtype))
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    fractions = torch.stack((xx.flatten(), yy.flatten()), -1)[:positions]
    centers = low[:, None] + radius + fractions[None] * (high - low - 2 * radius)[:, None]
    _, center_distance = _point_metrics(centers, starts, ends, edge_mask)
    room_circle_fit = center_distance + EPS >= radius
    room_circle_cost = (radius - center_distance).clamp_min(0) / radius
    best_score = torch.full((rooms,), torch.inf, device=device, dtype=dtype)
    best_pass = torch.zeros((rooms, 3), device=device, dtype=torch.bool)
    best_complete = torch.zeros(rooms, device=device, dtype=torch.bool)
    witness = {
        "wc_bounds": torch.zeros((rooms, 4), device=device, dtype=dtype),
        "basin_bounds": torch.zeros((rooms, 4), device=device, dtype=dtype),
        "transfer_bounds": torch.zeros((rooms, 4), device=device, dtype=dtype),
        "door_hinge": torch.zeros((rooms, 2), device=device, dtype=dtype),
        "door_u": torch.zeros((rooms, 2), device=device, dtype=dtype),
        "door_v": torch.zeros((rooms, 2), device=device, dtype=dtype),
        "basin_front": torch.zeros((rooms, 2), device=device, dtype=dtype),
        "basin_inward": torch.zeros((rooms, 2), device=device, dtype=dtype),
        "basin_along": torch.zeros((rooms, 2), device=device, dtype=dtype),
        "circle_center": torch.zeros((rooms, 2), device=device, dtype=dtype),
        "door_overlap": torch.zeros(rooms, device=device, dtype=dtype),
        "basin_overlap": torch.zeros(rooms, device=device, dtype=dtype),
        "template_index": torch.zeros(rooms, device=device, dtype=torch.long),
    }
    axes_u = starts.new_tensor([[1, 0], [0, 1], [-1, 0], [0, -1]])
    axes_v = starts.new_tensor([[0, 1], [-1, 0], [0, -1], [1, 0]])
    origins = torch.stack((low, torch.stack((high[:, 0], low[:, 1]), -1), high,
                           torch.stack((low[:, 0], high[:, 1]), -1)), 1)
    for offset in range(0, bank.shape[0], 32):
        chunk = bank[offset:offset + 32]
        count = chunk.shape[0]
        rotation = chunk[:, 0].long()
        u = axes_u[rotation][None].expand(rooms, -1, -1)
        v = axes_v[rotation][None].expand(rooms, -1, -1)
        span = high - low
        width = torch.where((rotation % 2)[None] == 0, span[:, :1], span[:, 1:])
        height = torch.where((rotation % 2)[None] == 0, span[:, 1:], span[:, :1])
        origin = origins[:, rotation]
        reflected = chunk[None, :, 1]
        origin = origin + reflected[..., None] * width[..., None] * u
        u = u * (1 - 2 * reflected[..., None])
        wc_x = chunk[None, :, 2] * (width - wc_width - transfer_width).clamp_min(0)
        zero = torch.zeros_like(wc_x)
        wc_local = torch.stack((wc_x, zero, wc_x + wc_width, zero + wc_depth), -1)
        transfer_local = torch.stack((wc_x + wc_width, zero, wc_x + wc_width + transfer_width,
                                      zero + transfer_length), -1)
        basin_y = chunk[None, :, 3] * (height - basin_width).clamp_min(0)
        basin_local = torch.stack((width - basin_depth, basin_y, width, basin_y + basin_width), -1)
        wc = _world_rectangle(wc_local, origin, u, v).reshape(-1, 4)
        basin = _world_rectangle(basin_local, origin, u, v).reshape(-1, 4)
        transfer = _world_rectangle(transfer_local, origin, u, v).reshape(-1, 4)
        hinge_right = chunk[None, :, 5]
        door_x = chunk[None, :, 4] * (width - door_width).clamp_min(0) + hinge_right * door_width
        hinge = (origin + door_x[..., None] * u + height[..., None] * v).reshape(-1, 2)
        door_u = (u * (1 - 2 * hinge_right[..., None])).reshape(-1, 2)
        door_v = (-v).reshape(-1, 2)
        room_ids = torch.arange(rooms, device=device)[:, None].expand(-1, count).reshape(-1)
        a, b, mask = starts[room_ids], ends[room_ids], edge_mask[room_ids]
        wc_fit, wc_cost = _rectangle_containment(wc, a, b, mask)
        basin_fit, basin_cost = _rectangle_containment(basin, a, b, mask)
        transfer_fit, transfer_cost = _rectangle_containment(transfer, a, b, mask)
        door_fit, door_cost = _sector_containment(hinge, door_u, door_v, door_width, a, b, mask)
        solid_overlap = _rectangle_overlap(wc, basin)
        door_wc = _sector_rectangle_depth(wc, hinge, door_u, door_v, door_width)
        door_basin = _sector_rectangle_depth(basin, hinge, door_u, door_v, door_width)
        physical = wc_fit & basin_fit & door_fit & (solid_overlap <= EPS) & (door_wc <= EPS) & (door_basin <= EPS)
        base_cost = (wc_cost + basin_cost + door_cost + solid_overlap + door_wc + door_basin) / radius
        transfer_basin = _rectangle_overlap(transfer, basin)
        transfer_door = _sector_rectangle_depth(transfer, hinge, door_u, door_v, door_width)
        transfer_ok = transfer_fit & (transfer_basin <= EPS) & (transfer_door <= EPS)
        transfer_cost = (transfer_cost + transfer_basin + transfer_door) / transfer_width
        circle = centers[room_ids]
        wc_distance = _circle_rectangle_distance(circle, wc[:, None])
        circle_ok = room_circle_fit[room_ids] & (wc_distance + EPS >= radius)
        circle_cost = room_circle_cost[room_ids] + (radius - wc_distance).clamp_min(0) / radius
        door_overlap = (radius - _sector_distance(circle, hinge, door_u, door_v, door_width)).clamp_min(0)
        circle_local = circle - origin.reshape(-1, 2)[:, None]
        cx = (circle_local * u.reshape(-1, 2)[:, None]).sum(-1)
        cy = (circle_local * v.reshape(-1, 2)[:, None]).sum(-1)
        basin_lo = basin_y.reshape(-1, 1)
        distance_y = torch.maximum(torch.maximum(basin_lo - cy, cy - basin_lo - basin_width), torch.zeros_like(cy))
        reach = (radius**2 - distance_y.square()).clamp_min(0).sqrt()
        basin_overlap = (cx + reach - (width.reshape(-1, 1) - basin_depth)).clamp(0, basin_depth)
        basin_overlap = torch.where(distance_y < radius, basin_overlap, torch.zeros_like(basin_overlap))
        permitted_basin_overlap = basin_limit if scenario.basin_under_clearance else 0.
        overlap_ok = ((door_overlap <= door_limit + EPS) & (basin_overlap <= permitted_basin_overlap + EPS)
                      & ~((door_overlap > EPS) & (basin_overlap > EPS)))
        overlap_cost = torch.maximum(torch.maximum((door_overlap - door_limit).clamp_min(0) / door_limit,
                                                   (basin_overlap - permitted_basin_overlap).clamp_min(0) / basin_limit),
                                     torch.minimum(door_overlap / door_limit, basin_overlap / basin_limit))
        all_ok = physical[:, None] & transfer_ok[:, None] & circle_ok & overlap_ok
        score = base_cost[:, None] + transfer_cost[:, None] + circle_cost + overlap_cost
        # Feasible-first selection also handles harmless floating point residuals.
        rank = torch.where(all_ok, torch.full_like(score, -1), score)
        rank = rank.reshape(rooms, count * positions)
        value, choice = rank.min(-1)
        tchoice, cchoice = choice // positions, choice % positions
        flat = torch.arange(rooms, device=device) * count + tchoice
        complete = all_ok[flat, cchoice]
        selected_score = score[flat, cchoice]
        improve = ((complete & ~best_complete) | ((complete == best_complete) & (selected_score < best_score)))
        best_score = torch.where(improve, selected_score, best_score)
        best_complete = torch.where(improve, complete, best_complete)
        passes = torch.stack((physical[flat] & circle_ok[flat, cchoice], physical[flat] & transfer_ok[flat],
                              physical[flat] & circle_ok[flat, cchoice] & overlap_ok[flat, cchoice]), -1)
        best_pass = torch.where(improve[:, None], passes, best_pass)
        values = {"wc_bounds": wc[flat], "basin_bounds": basin[flat], "transfer_bounds": transfer[flat],
                  "door_hinge": hinge[flat], "door_u": door_u[flat], "door_v": door_v[flat],
                  "basin_front": (origin + (width - basin_depth)[..., None] * u + basin_y[..., None] * v).reshape(-1, 2)[flat],
                  "basin_inward": u.reshape(-1, 2)[flat], "basin_along": v.reshape(-1, 2)[flat],
                  "circle_center": circle[flat, cchoice], "door_overlap": door_overlap[flat, cchoice],
                  "basin_overlap": basin_overlap[flat, cchoice], "template_index": offset + tchoice}
        for key, incoming in values.items():
            choose = improve if incoming.ndim == 1 else improve[:, None]
            witness[key] = torch.where(choose, incoming, witness[key])
    span = high - low
    circle_bound = torch.minimum(span.amin(-1) / 2, (areas / math.pi).clamp_min(0).sqrt())
    circle_impossible = ((circle_bound + EPS < radius) | (areas + EPS < math.pi * radius**2 + wc_width * wc_depth))
    combined_width, combined_length = wc_width + transfer_width, max(wc_depth, transfer_length)
    envelope_fit = (((span[:, 0] + EPS >= combined_width) & (span[:, 1] + EPS >= combined_length)) |
                    ((span[:, 1] + EPS >= combined_width) & (span[:, 0] + EPS >= combined_length)))
    transfer_area = wc_width * wc_depth + transfer_width * transfer_length + basin_width * basin_depth + math.pi * door_width**2 / 4
    transfer_impossible = ~envelope_fit | (areas + EPS < transfer_area)
    impossible = torch.stack((circle_impossible, transfer_impossible, circle_impossible), -1)
    status = torch.where(best_pass, 0, torch.where(impossible, 1, 2))
    circle_severity = torch.maximum((radius - circle_bound).clamp_min(0) / radius,
                                   (math.pi * radius**2 + wc_width * wc_depth - areas).clamp_min(0) /
                                   (math.pi * radius**2 + wc_width * wc_depth))
    transfer_severity = torch.maximum((transfer_area - areas).clamp_min(0) / transfer_area,
                                     (~envelope_fit).to(dtype))
    proven = torch.stack((circle_severity, transfer_severity, circle_severity), -1)
    severity = torch.where(status == 0, torch.zeros_like(proven), torch.where(status == 1, proven, torch.nan))
    return {"status": status, "severity": severity, "guidance_penalty": best_score,
            "witness": witness, "search_budget": len(templates), "circle_positions": positions,
            "door_radius_m": door_width, "circle_radius_m": radius,
            "basin_under_clearance": bool(scenario.basin_under_clearance)}


def serialized_witness(search, index):
    """Serialize only at report/audit boundaries, never in GPU guidance."""
    raw = {key: value[index].detach().cpu().tolist() for key, value in search["witness"].items()}
    result = {}
    for name in ("wc", "basin", "transfer"):
        x0, y0, x1, y1 = raw[name + "_bounds"]
        result[name + "_polygon_m"] = [[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]
    hinge, u, v = raw["door_hinge"], raw["door_u"], raw["door_v"]
    radius = search["door_radius_m"]
    # Exact sector is evaluated analytically; this polygon is only its SVG outline.
    arc = [[hinge[j] + radius * (math.cos(k * math.pi / 64) * u[j] + math.sin(k * math.pi / 64) * v[j])
            for j in range(2)] for k in range(33)]
    result.update(door_hinge_m=hinge, door_axes=[u, v], door_radius_m=radius,
                  door_swing_polygon_m=[hinge, *arc, hinge], circle_center_m=raw["circle_center"],
                  circle_radius_m=search["circle_radius_m"], door_overlap_m=raw["door_overlap"],
                  basin_overlap_m=raw["basin_overlap"], template_index=raw["template_index"],
                  basin_front_m=raw["basin_front"], basin_inward=raw["basin_inward"], basin_along=raw["basin_along"],
                  basin_under_clearance=search["basin_under_clearance"])
    return result


def _audit_witness(polygon, witness, limits):
    """Independent Shapely containment/obstacle audit of positive witnesses.

    A circumscribed sector polygon makes the solid/transfer audit conservative;
    near-boundary rejection is unknown, never proof of infeasibility.
    """
    from shapely.geometry import Point, Polygon

    wc, basin, transfer = (Polygon(witness[name + "_polygon_m"]) for name in ("wc", "basin", "transfer"))
    hinge = witness["door_hinge_m"]
    u, v = witness["door_axes"]
    door_radius = witness["door_radius_m"]
    segment = math.pi / 64
    angles = [0., *[(k + .5) * segment for k in range(32)], math.pi / 2]
    radii = [door_radius, *[door_radius / math.cos(segment / 2)] * 32, door_radius]
    points = [[hinge[j] + r * (math.cos(theta) * u[j] + math.sin(theta) * v[j]) for j in range(2)]
              for theta, r in zip(angles, radii)]
    sector = Polygon([hinge, *points, hinge])
    covered = polygon.buffer(EPS)
    physical = (covered.covers(wc) and covered.covers(basin) and covered.covers(sector)
                and wc.intersection(basin).area <= EPS
                and sector.intersection(wc).area <= EPS and sector.intersection(basin).area <= EPS)
    center = Point(witness["circle_center_m"])
    radius = witness["circle_radius_m"]
    turning = (physical and polygon.covers(center) and center.distance(polygon.boundary) + EPS >= radius
               and center.distance(wc) + EPS >= radius)
    lateral = (physical and covered.covers(transfer) and transfer.intersection(basin).area <= EPS
               and transfer.intersection(sector).area <= EPS and transfer.intersection(wc).area <= EPS)
    # Door overlap uses the exact quarter-disk distance, independently in Python.
    dx = center.x - hinge[0]
    dy = center.y - hinge[1]
    x, y = dx * u[0] + dy * u[1], dx * v[0] + dy * v[1]
    norm = math.hypot(x, y)
    if x >= 0 and y >= 0 and norm <= door_radius:
        signed = -min(x, y, door_radius - norm)
    else:
        distances = [math.hypot(x - min(door_radius, max(0, x)), y),
                     math.hypot(x, y - min(door_radius, max(0, y)))]
        if x >= 0 and y >= 0:
            distances.append(abs(norm - door_radius))
        signed = min(distances)
    door_depth = max(0., radius - signed)
    # Reproduce directional basin intrusion independently from its explicit axes.
    front = witness["basin_front_m"]
    inward, along = witness["basin_inward"], witness["basin_along"]
    projected = [[sum((point[j] - front[j]) * axis[j] for j in range(2)) for axis in (inward, along)]
                 for point in witness["basin_polygon_m"]]
    depth, width = max(p[0] for p in projected), max(p[1] for p in projected)
    local_x = sum((witness["circle_center_m"][j] - front[j]) * inward[j] for j in range(2))
    local_y = sum((witness["circle_center_m"][j] - front[j]) * along[j] for j in range(2))
    distance_y = max(-local_y, local_y - width, 0.)
    basin_depth = (min(depth, max(0., local_x + math.sqrt(max(0., radius**2 - distance_y**2))))
                   if distance_y < radius else 0.)
    reproduced = abs(basin_depth - witness["basin_overlap_m"]) <= 1e-6
    permitted_basin_overlap = limits["basin_overlap_m"] if witness.get("basin_under_clearance") else 0.
    overlap = (turning and reproduced and door_depth <= limits["door_overlap_m"] + EPS
               and basin_depth <= permitted_basin_overlap + EPS
               and not (door_depth > EPS and basin_depth > EPS))
    return {"C036": bool(turning), "C037": bool(lateral), "C041": bool(overlap)}


def cpu_search_bathroom(polygon, scenario, limits):
    """Run the identical finite bank in float64 and audit all claimed passes."""
    vertices = torch.tensor(list(polygon.exterior.coords)[:-1], dtype=torch.float64)[None]
    search = search_bathrooms(vertices, vertices.roll(-1, 1), torch.ones(vertices.shape[:-1], dtype=torch.bool),
                             vertices.amin(1), vertices.amax(1), torch.tensor([polygon.area], dtype=torch.float64),
                             scenario, limits)
    witness = serialized_witness(search, 0)
    statuses = search["status"][0].tolist()
    severities = search["severity"][0].tolist()
    audit = _audit_witness(polygon, witness, limits)
    rejected = []
    for index, identifier in enumerate(IDS):
        if statuses[index] == 0 and not audit[identifier]:
            statuses[index] = 2
            severities[index] = None
            rejected.append(identifier)
    names = {0: "pass", 1: "fail", 2: "unknown"}
    x0, y0, x1, y1 = polygon.bounds
    width, height = x1 - x0, y1 - y0
    envelope_width = scenario.wc_width_m + limits["transfer_width_m"]
    envelope_length = max(scenario.wc_depth_m, limits["transfer_length_m"])
    radius = limits["diameter_m"] / 2
    bounds = {"room_area_m2": polygon.area, "bbox_dimensions_m": [width, height],
              "circle_radius_upper_bound_m": min(width / 2, height / 2, math.sqrt(polygon.area / math.pi)),
              "minimum_circle_wc_area_m2": math.pi * radius**2 + scenario.wc_width_m * scenario.wc_depth_m,
              "minimum_transfer_fixtures_door_area_m2": (scenario.wc_width_m * scenario.wc_depth_m
                  + limits["transfer_width_m"] * limits["transfer_length_m"]
                  + scenario.basin_width_m * scenario.basin_depth_m + math.pi * scenario.bathroom_door_width_m**2 / 4),
              "transfer_envelope_dimensions_m": [envelope_width, envelope_length],
              "transfer_envelope_fits": ((width + EPS >= envelope_width and height + EPS >= envelope_length)
                  or (height + EPS >= envelope_width and width + EPS >= envelope_length))}
    return {"status_by_id": dict(zip(IDS, (names[s] for s in statuses))),
            "severity_by_id": {identifier: value if status != 2 else None
                               for identifier, value, status in zip(IDS, severities, statuses)},
            "guidance_penalty": float(search["guidance_penalty"][0]), "witness": witness,
            "reason": ("One shared fixture, door, transfer and circle completion was witnessed" if all(s == 0 for s in statuses)
                       else "A necessary geometric bound fails" if any(s == 1 for s in statuses)
                       else "Finite shared-layout search has no complete audited witness; feasibility is unresolved"),
            "search_budget": search["search_budget"], "circle_positions": search["circle_positions"],
            "audit_rejected_ids": rejected, "bounds": bounds,
            "depth_convention": "door: signed-distance penetration into quarter disk; basin: intrusion inward from its front edge"}
