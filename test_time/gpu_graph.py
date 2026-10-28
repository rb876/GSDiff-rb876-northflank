"""Batched planar room extraction using only device-resident torch tensors.

Inputs use drawing coordinates and seven binary semantic-presence flags. As in
the original decoder, the adjacency's upper triangle defines undirected walls.
Bridges are removed before face extraction and structural crossing checks.
Faces have open, prefix-masked vertices; labels reproduce the original vote and
tie order. Nested components (which would need polygon holes) are rejected.

The edge capacity follows the planar graph bound; excessive/dense predictions
are invalid and truncated for bounded work, never silently accepted. All loops
depend on tensor shapes, not tensor values. There are no host transfers here.
"""

import math

import torch


NONFINITE = 1
DUPLICATE_XY = 2
SELF_LOOP = 4
EDGE_CAPACITY = 8
WALL_INTERSECTION = 16
NO_ROOMS = 32
INVALID_FACE = 64
NESTED_ROOMS = 128
INVALID_SEMANTICS = 256
FATAL_DIAGNOSTICS = 511

DIAGNOSTIC_NAMES = {
    NONFINITE: "nonfinite active node",
    DUPLICATE_XY: "duplicate active XY coordinates",
    SELF_LOOP: "predicted self-loop",
    EDGE_CAPACITY: "planar edge capacity exceeded",
    WALL_INTERSECTION: "retained walls cross or overlap without a shared junction",
    NO_ROOMS: "no bounded room faces",
    INVALID_FACE: "bounded face is not a simple polygon",
    NESTED_ROOMS: "nested room components require unsupported polygon holes",
    INVALID_SEMANTICS: "semantic presence flags are not binary",
}


def _cross(first, second):
    return first[..., 0] * second[..., 1] - first[..., 1] * second[..., 0]


def _gather_nodes(values, indices):
    batch = torch.arange(values.shape[0], device=values.device)
    return values[batch.view(-1, *([1] * (indices.ndim - 1))), indices]


def _intersections(xy, sources, targets, active, eps):
    """Find forbidden segment intersections, permitting a shared endpoint only."""
    first, second = _gather_nodes(xy, sources), _gather_nodes(xy, targets)
    a, b = first[:, :, None], second[:, :, None]
    c, d = first[:, None, :], second[:, None, :]
    ab, cd = b - a, d - c
    o1, o2 = _cross(ab, c - a), _cross(ab, d - a)
    o3, o4 = _cross(cd, a - c), _cross(cd, b - c)
    straddles = (((o1 <= eps) & (o2 >= -eps)) | ((o2 <= eps) & (o1 >= -eps)))
    straddles &= (((o3 <= eps) & (o4 >= -eps)) | ((o4 <= eps) & (o3 >= -eps)))
    low = torch.maximum(torch.minimum(a, b), torch.minimum(c, d))
    high = torch.minimum(torch.maximum(a, b), torch.maximum(c, d))
    intersects = straddles & (low <= high + eps).all(dim=-1)
    common = ((sources[:, :, None] == sources[:, None, :])
              | (sources[:, :, None] == targets[:, None, :])
              | (targets[:, :, None] == sources[:, None, :])
              | (targets[:, :, None] == targets[:, None, :]))
    collinear = (o1.abs() <= eps) & (o2.abs() <= eps)
    positive_overlap = (high - low).amax(dim=-1) > eps
    forbidden = intersects & (~common | (collinear & positive_overlap))
    pair_mask = active[:, :, None] & active[:, None, :]
    pair_mask &= torch.triu(torch.ones_like(pair_mask, dtype=torch.bool), diagonal=1)
    return (forbidden & pair_mask).flatten(1).any(dim=1)


def _bridges(adjacency, sources, targets, active, dtype):
    """Parallel reachability with each candidate edge excluded in its own row."""
    batch, nodes, _ = adjacency.shape
    edges = sources.shape[1]
    reach = torch.zeros((batch, edges, nodes), dtype=torch.bool, device=adjacency.device)
    reach.scatter_(2, sources[..., None], active[..., None])
    graph = adjacency.to(dtype)
    endpoints = torch.stack((sources, targets), dim=-1)
    for _ in range(max(nodes - 1, 0)):
        counts = torch.bmm(reach.to(dtype), graph)
        # Remove exactly the contributions travelling across this row's edge.
        contributions = torch.stack((reach.gather(2, targets[..., None]).squeeze(-1),
                                     reach.gather(2, sources[..., None]).squeeze(-1)), dim=-1)
        counts.scatter_add_(2, endpoints, -contributions.to(dtype))
        reach = reach | (counts > 0)
    return active & ~reach.gather(2, targets[..., None]).squeeze(-1)


def _nested(polygons, vertex_mask, room_mask, eps):
    """Strict first-vertex containment detects nesting of non-crossing faces."""
    vertices = polygons.shape[2]
    lengths = vertex_mask.sum(dim=-1)
    positions = torch.arange(vertices, device=polygons.device)
    next_index = torch.where(positions[None, None] + 1 < lengths[..., None],
                             positions[None, None] + 1, 0)
    following = polygons.gather(2, next_index[..., None].expand(-1, -1, -1, 2))
    point = polygons[:, :, 0, :][:, :, None, None]
    first, second = polygons[:, None], following[:, None]
    delta = second - first
    offset = point - first
    boundary = (_cross(delta, offset).abs() <= eps)
    boundary &= ((offset * delta).sum(-1) >= -eps)
    boundary &= ((offset * delta).sum(-1) <= (delta * delta).sum(-1) + eps)
    edge_mask = vertex_mask[:, None]
    boundary = (boundary & edge_mask).any(-1)
    dy = delta[..., 1]
    safe_dy = torch.where(dy == 0, torch.ones_like(dy), dy)
    x_intersect = first[..., 0] + offset[..., 1] * delta[..., 0] / safe_dy
    ray = ((first[..., 1] > point[..., 1]) != (second[..., 1] > point[..., 1]))
    ray &= point[..., 0] < x_intersect
    inside = ((ray & edge_mask).sum(-1) % 2 == 1) & ~boundary
    pair_mask = room_mask[:, :, None] & room_mask[:, None, :]
    return (inside & pair_mask).flatten(1).any(dim=1)


@torch.no_grad()
def extract_rooms(xy, semantics, node_mask, adjacency):
    """Return polygons[B,2N,N,2], masks, labels and graph validity on device.

    ``diagnostics`` is an integer bitmask of structural failures. Informational
    ``removed_bridges`` and ``raw_wall_intersection`` booleans describe raw walls
    that the original decoder may remove before structural evaluation.
    """
    if xy.ndim != 3 or xy.shape[-1] != 2 or xy.shape[1] < 1:
        raise ValueError("xy must have shape [B,N,2] with N >= 1")
    batch, nodes, _ = xy.shape
    if semantics.shape != (batch, nodes, 7) or node_mask.shape != (batch, nodes):
        raise ValueError("semantics/node_mask must have shapes [B,N,7]/[B,N]")
    if adjacency.shape != (batch, nodes, nodes):
        raise ValueError("adjacency must have shape [B,N,N]")
    if any(value.device != xy.device for value in (semantics, node_mask, adjacency)):
        raise ValueError("All graph tensors must be on the same device")
    dtype = torch.float64 if xy.dtype == torch.float64 else torch.float32
    xy = xy.to(dtype)
    node_mask = node_mask.bool()
    adjacency = adjacency.bool()
    eps = 1e-8
    diagnostics = torch.zeros(batch, dtype=torch.long, device=xy.device)
    finite = torch.isfinite(xy).all(-1) & torch.isfinite(semantics).all(-1)
    diagnostics |= ((~finite & node_mask).any(-1).long() * NONFINITE)
    bad_semantics = ~((semantics == 0) | (semantics == 1)).all(-1)
    diagnostics |= ((bad_semantics & node_mask).any(-1).long() * INVALID_SEMANTICS)
    safe_xy = torch.where(torch.isfinite(xy), xy, torch.zeros_like(xy))
    same = (safe_xy[:, :, None] == safe_xy[:, None, :]).all(-1)
    same &= node_mask[:, :, None] & node_mask[:, None, :]
    diagnostics |= torch.triu(same, diagonal=1).flatten(1).any(dim=1).long() * DUPLICATE_XY
    active_pair = node_mask[:, :, None] & node_mask[:, None, :]
    diagnostics |= (adjacency.diagonal(dim1=1, dim2=2) & node_mask).any(-1).long() * SELF_LOOP
    upper = torch.triu(adjacency & active_pair, diagonal=1)
    count = upper.sum(dim=(-1, -2))
    actual_nodes = node_mask.sum(-1)
    bound = torch.where(actual_nodes >= 3, 3 * actual_nodes - 6,
                        actual_nodes * (actual_nodes - 1) // 2)
    diagnostics |= (count > bound).long() * EDGE_CAPACITY
    capacity = max(1, min(nodes * (nodes - 1) // 2, max(3 * nodes - 6, 1)))
    flat_indices = torch.arange(nodes * nodes, device=xy.device).expand(batch, -1)
    ranked = torch.where(upper.reshape(batch, -1), flat_indices, nodes * nodes).sort(dim=1).values
    chosen = ranked[:, :capacity]
    edge_active = chosen < nodes * nodes
    chosen = chosen.clamp(max=nodes * nodes - 1)
    sources, targets = chosen // nodes, chosen % nodes
    # Build the bounded graph used for both reachability and the face walk.
    # Older CUDA PyTorch supports integer, but not bool, scatter reductions.
    bounded = torch.zeros_like(adjacency, dtype=torch.long).reshape(batch, -1)
    bounded.scatter_reduce_(1, chosen, edge_active.long(), reduce="amax", include_self=True)
    bounded = bounded.reshape(batch, nodes, nodes).bool()
    bounded = bounded | bounded.transpose(1, 2)
    bridges = _bridges(bounded, sources, targets, edge_active, dtype)
    raw_crossing = _intersections(safe_xy, sources, targets, edge_active, eps)
    retained = edge_active & ~bridges
    diagnostics |= _intersections(safe_xy, sources, targets, retained, eps).long() * WALL_INTERSECTION

    # Each directed edge belongs to one face. Choose the most clockwise exit
    # relative to its reversed incoming direction in Cartesian (x,-y) space.
    starts, ends = torch.cat((sources, targets), 1), torch.cat((targets, sources), 1)
    directed_active = torch.cat((retained, retained), 1)
    directed_count = 2 * capacity
    ids = torch.arange(directed_count, device=xy.device).expand(batch, -1)
    lookup = torch.zeros((batch, nodes * nodes), dtype=torch.long, device=xy.device)
    lookup.scatter_reduce_(1, starts * nodes + ends,
                           torch.where(directed_active, ids + 1, 0), reduce="amax", include_self=True)
    lookup = lookup.reshape(batch, nodes, nodes)
    vectors = safe_xy[:, None] - _gather_nodes(safe_xy, ends)[:, :, None]
    angles = torch.atan2(-vectors[..., 1], vectors[..., 0])
    incoming = _gather_nodes(safe_xy, starts) - _gather_nodes(safe_xy, ends)
    incoming_angle = torch.atan2(-incoming[..., 1], incoming[..., 0])
    delta = torch.remainder(angles - incoming_angle[..., None], 2 * math.pi)
    outgoing_ids = _gather_nodes(lookup, ends)
    delta = torch.where(outgoing_ids > 0, delta, -torch.ones_like(delta))
    next_node = delta.argmax(-1)
    successor = outgoing_ids.gather(2, next_node[..., None]).squeeze(-1) - 1
    successor = torch.where(directed_active, successor.clamp(min=0), ids)

    # Pointer doubling finds a canonical representative for each permutation
    # cycle without discovering a variable-length list on the host.
    representatives, pointer = ids.clone(), successor
    for _ in range(max(1, (directed_count - 1).bit_length())):
        representatives = torch.minimum(representatives, representatives.gather(1, pointer))
        pointer = pointer.gather(1, pointer)
    representative = (representatives == ids) & directed_active
    current = ids
    running = directed_active.clone()
    paths, masks = [], []
    for _ in range(nodes):
        paths.append(starts.gather(1, current))
        masks.append(running)
        following_edge = successor.gather(1, current)
        running = running & (following_edge != ids)
        current = following_edge
    path = torch.stack(paths, -1)
    mask = torch.stack(masks, -1)
    lengths = mask.sum(-1)
    closed = ~running
    polygons = _gather_nodes(safe_xy, path)
    positions = torch.arange(nodes, device=xy.device)
    next_position = torch.where(positions[None, None] + 1 < lengths[..., None],
                                positions[None, None] + 1, 0)
    following_xy = polygons.gather(2, next_position[..., None].expand(-1, -1, -1, 2))
    area2 = (_cross(polygons, following_xy) * mask).sum(-1)
    occurrences = torch.zeros((batch, directed_count, nodes), dtype=torch.long, device=xy.device)
    occurrences.scatter_add_(2, path, mask.long())
    simple = occurrences.amax(-1) <= 1
    # Bounded faces have negative drawing-coordinate signed area.
    bounded_face = representative & (area2 < -2 * eps)
    malformed = bounded_face & (~closed | ~simple | (lengths < 3))
    diagnostics |= malformed.any(-1).long() * INVALID_FACE
    keep = bounded_face & closed & simple & (lengths >= 3)
    votes = (_gather_nodes(semantics == 1, path) & mask[..., None]).sum(-2)
    priority = torch.tensor([2, 4, 3, 1, 5, 0], dtype=torch.long, device=xy.device)
    labels = priority[votes[..., priority].argmax(-1)]

    room_capacity = 2 * nodes
    selected = torch.where(keep, ids, directed_count).sort(dim=1).values[:, :room_capacity]
    room_mask = selected < directed_count
    selected = selected.clamp(max=directed_count - 1)
    polygons = _gather_nodes(polygons, selected)
    vertex_mask = _gather_nodes(mask, selected) & room_mask[..., None]
    labels = _gather_nodes(labels, selected)
    if directed_count < room_capacity:
        extra = room_capacity - directed_count
        polygons = torch.cat((polygons, polygons.new_zeros((batch, extra, nodes, 2))), 1)
        vertex_mask = torch.cat((vertex_mask, vertex_mask.new_zeros((batch, extra, nodes))), 1)
        labels = torch.cat((labels, labels.new_zeros((batch, extra))), 1)
        room_mask = torch.cat((room_mask, room_mask.new_zeros((batch, extra))), 1)
    polygons = torch.where(vertex_mask[..., None], polygons, torch.zeros_like(polygons))
    labels = torch.where(room_mask, labels, -torch.ones_like(labels))
    diagnostics |= (~room_mask.any(-1)).long() * NO_ROOMS
    diagnostics |= _nested(polygons, vertex_mask, room_mask, eps).long() * NESTED_ROOMS
    return {"polygons": polygons, "vertex_mask": vertex_mask, "room_mask": room_mask,
            "labels": labels, "structurally_valid": (diagnostics & FATAL_DIAGNOSTICS) == 0,
            "diagnostics": diagnostics, "removed_bridges": bridges.any(-1),
            "raw_wall_intersection": raw_crossing}
