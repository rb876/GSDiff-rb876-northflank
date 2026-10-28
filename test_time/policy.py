"""One scoring policy shared by sampling and offline evaluation."""

import math
import numpy as np
from shapely.geometry import LineString, Point
from shapely.ops import unary_union

from .constraints import CheckContext, check, load_specs
from .geometry import EPS, polygon_from_points
from .types import PolicyEvaluation


def graph_validity(sample):
    diagnostics = list(sample.diagnostics)
    if not sample.nodes or not sample.edges or not sample.rooms:
        diagnostics.append("Graph must contain nodes, wall segments and room polygons")
    try:
        coordinates = np.asarray([node[:2] for node in sample.nodes], dtype=float)
    except (TypeError, ValueError):
        coordinates = np.array([])
    coordinates_ok = coordinates.ndim == 2 and coordinates.shape[1:] == (2,) and np.isfinite(coordinates).all()
    if not coordinates_ok:
        diagnostics.append("Node coordinates are not finite XY pairs")
    elif len({tuple(xy) for xy in coordinates}) != len(coordinates):
        diagnostics.append("Multiple graph nodes occupy the same drawing coordinate")
    edges, seen = [], set()
    for index, edge in enumerate(sample.edges):
        if len(edge) != 2 or any(type(value) is not int or value < 0 or value >= len(sample.nodes) for value in edge):
            diagnostics.append(f"Edge {index} has inconsistent node references")
            continue
        first, second = edge
        key = tuple(sorted(edge))
        if first == second:
            diagnostics.append(f"Edge {index} is a self-loop")
            continue
        if key in seen:
            diagnostics.append(f"Edge {index} duplicates an undirected wall segment")
            continue
        seen.add(key)
        if coordinates_ok:
            segment = LineString(coordinates[[first, second]])
            if segment.length <= EPS:
                diagnostics.append(f"Edge {index} has zero length")
            else:
                edges.append((set(edge), segment))
    for index, (ids, segment) in enumerate(edges):
        for other_ids, other_segment in edges[index + 1:]:
            intersection = segment.intersection(other_segment)
            if intersection.is_empty:
                continue
            common = ids & other_ids
            if intersection.geom_type != "Point" or not any(intersection.distance(Point(coordinates[node])) <= EPS for node in common):
                diagnostics.append("Wall segments cross or overlap without a shared junction")
                break
    polygons = []
    for index, room in enumerate(sample.rooms):
        if room.label not in range(7):
            diagnostics.append(f"Room {index} has an unknown semantic label")
        if len(room.polygon) < 4 or not np.allclose(room.polygon[0][:2], room.polygon[-1][:2], atol=EPS, rtol=0):
            diagnostics.append(f"Room {index} polygon is not explicitly closed")
        try:
            polygons.append(polygon_from_points(room.polygon))
        except ValueError as error:
            diagnostics.append(f"Room {index}: {error}")
    for index, polygon in enumerate(polygons):
        if any(polygon.intersection(other).area > EPS for other in polygons[index + 1:]):
            diagnostics.append("Room interiors improperly overlap")
            break
    if edges and polygons:
        walls = unary_union([segment for _, segment in edges]).buffer(EPS)
        if any(polygon.boundary.difference(walls).length > EPS for polygon in polygons):
            diagnostics.append("A room boundary is not supported by predicted wall segments")
    diagnostics = list(dict.fromkeys(diagnostics))
    return not diagnostics, float(bool(diagnostics)), diagnostics


class ConstraintEvaluator:
    def __init__(self, specs=None, validity_weight=100.0):
        self.specs = load_specs() if specs is None else list(specs)
        if not math.isfinite(validity_weight) or validity_weight < 0:
            raise ValueError("validity_weight must be finite and nonnegative")
        self.validity_weight = validity_weight

    def evaluate(self, sample, scenario):
        valid, penalty, diagnostics = graph_validity(sample)
        context = CheckContext(sample, scenario, self.specs)
        enabled = [spec for spec in self.specs if spec.enabled]
        results = [check(sample, scenario, spec, context) for spec in enabled]
        regulatory_energy = sum(spec.weight * result.severity for spec, result in zip(enabled, results)
                                if result.applicable and result.measurable)
        energy = regulatory_energy + self.validity_weight * penalty
        counted = [result for result in results if result.counts_toward_compliance and result.applicable]
        return PolicyEvaluation(
            energy=float(energy), reward=-float(energy), results=results,
            structurally_valid=valid, validity_penalty=penalty, diagnostics=diagnostics,
            unknown_count=sum(result.status == "unknown" for result in counted),
            applicable_count=len(counted),
            assumption_based_count=sum(bool(result.assumption_flags) for result in counted),
            measurable_count=sum(result.measurable for result in counted),
        )

    __call__ = evaluate
