"""Descriptive diversity and baseline-relative geometry diagnostics; no model inference.

Callers provide selected candidate rows. Geometry metrics are conditional on structural
validity and usable saved geometry, so their eligible counts must accompany quality rates.
These diagnostics do not establish fidelity to RPLAN or real floorplans.
"""

from collections import Counter
import math

import numpy as np
from PIL import Image, ImageDraw


ROOM_LABELS = tuple(range(6))  # Outside (6) and raster background are excluded.
CANVAS_SIZE = 512
GRAPH_DECIMALS = 6
_BIT_COUNTS = np.array([number.bit_count() for number in range(256)], dtype=np.uint8)


def _area(points):
    return abs(sum(first[0] * second[1] - second[0] * first[1]
                   for first, second in zip(points, points[1:] + points[:1]))) / 2


def _eligible_samples(rows):
    samples, excluded = [], Counter()
    for row in rows:
        if not row.get("evaluation", {}).get("structurally_valid", False):
            excluded["structurally_invalid_or_unrecorded"] += 1
            continue
        sample = row.get("sample", {})
        try:
            nodes, edges, rooms = sample["nodes"], sample["edges"], sample["rooms"]
            usable = len(nodes) >= 3 and len(edges) >= 3 and bool(rooms)
            usable &= all(len(node) >= 2 and all(math.isfinite(value) for value in node) for node in nodes)
            usable &= all(len(edge) == 2 and all(type(index) is int and 0 <= index < len(nodes)
                                                for index in edge) for edge in edges)
            usable &= all(room["label"] in range(7) and len(room["polygon"]) >= 3
                          and all(len(point) >= 2 and all(math.isfinite(value) for value in point[:2])
                                  for point in room["polygon"])
                          and math.isfinite(area := _area(room["polygon"])) and area > 0 for room in rooms)
            usable &= any(room["label"] in ROOM_LABELS for room in rooms)
        except (KeyError, TypeError, ValueError, IndexError):
            usable = False
        if usable:
            samples.append(sample)
        else:
            excluded["missing_or_unusable_geometry"] += 1
    return samples, dict(excluded)


def _graph_key(sample):
    nodes = [tuple(round(float(value), GRAPH_DECIMALS) for value in node) for node in sample["nodes"]]
    edges = [tuple(sorted((nodes[first], nodes[second]))) for first, second in sample["edges"]]
    return tuple(sorted(nodes)), tuple(sorted(edges))


def _semantic_raster(sample, size):
    channels = []
    for label in ROOM_LABELS:
        image = Image.new("1", (size, size))
        draw = ImageDraw.Draw(image)
        for room in sample["rooms"]:
            if room["label"] == label:
                # Fixed drawing coordinates: no translation, rotation or scale alignment.
                points = [(x * size / CANVAS_SIZE, y * size / CANVAS_SIZE)
                          for x, y, *_ in room["polygon"]]
                draw.polygon(points, fill=1)
        channels.append(np.asarray(image, dtype=bool).reshape(-1))
    return np.packbits(np.concatenate(channels))


def diversity_summary(rows, raster_size=64, near_duplicate_threshold=0.05):
    """Graph duplicates and pairwise semantic IoU distance on a fixed drawing canvas.

    A near duplicate has another eligible layout with distance <= the threshold.
    The fraction is over raster-eligible layouts, not over pairs. Graph duplication
    is 1 - unique graphs / eligible graphs and is invariant to node/edge ordering.
    """
    if not isinstance(raster_size, int) or raster_size < 1:
        raise ValueError("raster_size must be a positive integer")
    if not 0 <= near_duplicate_threshold <= 1:
        raise ValueError("near_duplicate_threshold must be between 0 and 1")
    rows = list(rows)
    samples, exclusions = _eligible_samples(rows)
    count = len(samples)
    unique = len({_graph_key(sample) for sample in samples})
    rasters = [_semantic_raster(sample, raster_size) for sample in samples]
    rasters = [raster for raster in rasters if raster.any()]
    raster_count = len(rasters)
    pair_count = raster_count * (raster_count - 1) // 2
    mean_distance = near_fraction = None
    if pair_count:
        packed = np.stack(rasters)
        areas = _BIT_COUNTS[packed].sum(axis=1, dtype=np.int64)
        near = np.zeros(raster_count, dtype=bool)
        distance_sum = 0.0
        for index in range(raster_count - 1):
            intersection = _BIT_COUNTS[np.bitwise_and(packed[index], packed[index + 1:])].sum(axis=1, dtype=np.int64)
            union = areas[index] + areas[index + 1:] - intersection
            distance = 1 - intersection / union
            distance_sum += float(distance.sum())
            neighbors = distance <= near_duplicate_threshold + 1e-12
            near[index] |= bool(neighbors.any())
            near[index + 1:] |= neighbors
        mean_distance = distance_sum / pair_count
        near_fraction = float(near.mean())
    return {
        "population": "structurally_valid_with_usable_geometry",
        "input_count": len(rows), "eligible_count": count, "excluded_count": len(rows) - count,
        "exclusion_reasons": exclusions,
        "graph_unique_count": unique,
        "graph_unique_fraction": unique / count if count else None,
        "graph_duplicate_fraction": 1 - unique / count if count else None,
        "graph_coordinate_rounding_decimals": GRAPH_DECIMALS,
        "raster_eligible_count": raster_count, "raster_excluded_count": count - raster_count,
        "raster_size": raster_size, "canvas_size_drawing_pixels": CANVAS_SIZE,
        "room_labels": list(ROOM_LABELS), "pair_count": pair_count,
        "mean_pairwise_semantic_distance": mean_distance,
        "near_duplicate_threshold": near_duplicate_threshold,
        "near_duplicate_fraction": near_fraction,
        "interpretation": "Semantic distance is 1 minus intersection/union across room-label masks; background and outside are excluded. "
                          "Near-duplicate fraction counts layouts with at least one close neighbor. All pairs are evaluated.",
        "limitations": "Fixed-canvas low-resolution diagnostics; no translation/rotation/scale alignment. "
                       "Raster geometry is clipped to the 512-pixel canvas and does not separately encode walls. "
                       "Eligible counts differ if methods have different structural validity. This is not reference-data fidelity.",
    }


def _features(sample):
    rooms = [room for room in sample["rooms"] if room["label"] in ROOM_LABELS]
    return {"rooms": len(rooms), "junctions": len(sample["nodes"]), "edges": len(sample["edges"]),
            "total_room_area_drawing_units2": sum(_area(room["polygon"]) for room in rooms)}


def _wasserstein(first, second):
    """Exact 1D empirical Wasserstein distance, including unequal sample counts."""
    first, second = np.sort(first), np.sort(second)
    positions = np.sort(np.concatenate((first, second)))
    first_cdf = np.searchsorted(first, positions[:-1], side="right") / len(first)
    second_cdf = np.searchsorted(second, positions[:-1], side="right") / len(second)
    return float(np.sum(np.abs(first_cdf - second_cdf) * np.diff(positions)))


def _room_frequencies(samples):
    counts = Counter(room["label"] for sample in samples for room in sample["rooms"]
                     if room["label"] in ROOM_LABELS)
    total = sum(counts.values())
    return np.array([counts[label] / total for label in ROOM_LABELS]) if total else None


def distribution_shift(rows, baseline_rows):
    """Compare valid generated layouts with a compatible generated baseline.

    Distances are absolute, unstandardized Wasserstein distances; lower means
    similar marginal distributions, not higher quality. Semantic composition uses
    base-2 Jensen-Shannon divergence (0 identical, 1 disjoint room-label support).
    The caller must establish matching scenario, model and representation.
    """
    rows, baseline_rows = list(rows), list(baseline_rows)
    samples, exclusions = _eligible_samples(rows)
    baseline, baseline_exclusions = _eligible_samples(baseline_rows)
    observed, reference = [_features(sample) for sample in samples], [_features(sample) for sample in baseline]
    features = {}
    for name in ("rooms", "junctions", "edges", "total_room_area_drawing_units2"):
        left, right = [item[name] for item in observed], [item[name] for item in reference]
        features[name] = {"method_mean": float(np.mean(left)) if left else None,
                          "baseline_mean": float(np.mean(right)) if right else None,
                          "wasserstein_distance": _wasserstein(left, right) if left and right else None}
    left, right = _room_frequencies(samples), _room_frequencies(baseline)
    divergence = None
    if left is not None and right is not None:
        midpoint = (left + right) / 2
        terms = []
        for values in (left, right):
            positive = values > 0
            terms.append(float(np.sum(values[positive] * np.log2(values[positive] / midpoint[positive]))))
        divergence = min(1.0, max(0.0, sum(terms) / 2))
    return {
        "reference": "generated_baseline_not_RPLAN",
        "population": "structurally_valid_with_usable_geometry",
        "input_count": len(rows), "eligible_count": len(samples), "excluded_count": len(rows) - len(samples),
        "baseline_input_count": len(baseline_rows), "baseline_eligible_count": len(baseline),
        "baseline_excluded_count": len(baseline_rows) - len(baseline),
        "exclusion_reasons": exclusions, "baseline_exclusion_reasons": baseline_exclusions,
        "features": features, "room_type_js_divergence_bits": divergence,
        "room_type_frequencies": dict(zip(map(str, ROOM_LABELS), left.tolist())) if left is not None else None,
        "baseline_room_type_frequencies": dict(zip(map(str, ROOM_LABELS), right.tolist())) if right is not None else None,
        "interpretation": "Room counts/areas exclude outside label 6; area is a sum of polygons in drawing-pixel squared units. "
                          "Wasserstein distances have the original feature units. Composition is pooled over room instances. "
                          "These marginal diagnostics do not establish perceptual quality or real-data fidelity.",
    }
