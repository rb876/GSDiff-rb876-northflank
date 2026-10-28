"""Diversity diagnostics distinguish duplicates, missing evidence and distribution shift."""

from copy import deepcopy
import json

import pytest

from test_time.diversity import diversity_summary, distribution_shift


def rectangle(label=0, width=128, x=32, valid=True):
    points = [[x, 32], [x + width, 32], [x + width, 160], [x, 160]]
    return {"sample": {"nodes": [point + [label] for point in points],
                       "edges": [[0, 1], [1, 2], [2, 3], [3, 0]],
                       "rooms": [{"label": label, "polygon": points + [points[0]]}]},
            "evaluation": {"structurally_valid": valid}}


def test_duplicate_graphs_ignore_node_order_edge_direction_and_room_start():
    first = rectangle()
    second = deepcopy(first)
    order = [2, 0, 3, 1]
    inverse = {old: new for new, old in enumerate(order)}
    second["sample"]["nodes"] = [first["sample"]["nodes"][index] for index in order]
    second["sample"]["edges"] = [[inverse[b], inverse[a]] for a, b in reversed(first["sample"]["edges"])]
    polygon = first["sample"]["rooms"][0]["polygon"][:-1]
    polygon = polygon[2:] + polygon[:2]
    second["sample"]["rooms"][0]["polygon"] = polygon + [polygon[0]]
    summary = diversity_summary([first, second])
    assert summary["graph_unique_count"] == 1
    assert summary["graph_duplicate_fraction"] == 0.5
    assert summary["pair_count"] == 1
    assert summary["mean_pairwise_semantic_distance"] == 0
    assert summary["near_duplicate_fraction"] == 1
    json.dumps(summary, allow_nan=False)


def test_disjoint_geometry_and_room_semantics_are_distinct():
    for other in (rectangle(x=300), rectangle(label=1)):
        summary = diversity_summary([rectangle(), other])
        assert summary["graph_unique_count"] == 2
        assert summary["graph_duplicate_fraction"] == 0
        assert summary["mean_pairwise_semantic_distance"] == 1
        assert summary["near_duplicate_fraction"] == 0


def test_near_duplicate_fraction_counts_layouts_not_pairs():
    summary = diversity_summary([rectangle(), rectangle(), rectangle(x=300)])
    assert summary["pair_count"] == 3
    assert summary["near_duplicate_fraction"] == pytest.approx(2 / 3)
    assert summary["mean_pairwise_semantic_distance"] == pytest.approx(2 / 3)


def test_invalid_missing_and_nonfinite_geometry_are_excluded():
    missing = {"sample": {}, "evaluation": {"structurally_valid": True}}
    nonfinite = rectangle()
    nonfinite["sample"]["rooms"][0]["polygon"][0][0] = float("nan")
    summary = diversity_summary([rectangle(), rectangle(valid=False), missing, nonfinite])
    assert summary["input_count"] == 4 and summary["eligible_count"] == 1
    assert summary["excluded_count"] == 3
    assert summary["pair_count"] == 0
    assert summary["mean_pairwise_semantic_distance"] is None
    assert summary["near_duplicate_fraction"] is None
    assert summary["exclusion_reasons"] == {"structurally_invalid_or_unrecorded": 1,
                                             "missing_or_unusable_geometry": 2}


def test_empty_population_and_clipped_rasters_do_not_claim_diversity():
    empty = diversity_summary([])
    assert empty["graph_unique_fraction"] is None and empty["near_duplicate_fraction"] is None
    clipped = diversity_summary([rectangle(x=700), rectangle(x=900)])
    assert clipped["eligible_count"] == 2
    assert clipped["raster_eligible_count"] == 0 and clipped["raster_excluded_count"] == 2
    assert clipped["mean_pairwise_semantic_distance"] is None


def test_identical_distributions_with_different_counts_have_zero_shift():
    summary = distribution_shift([rectangle(), rectangle()], [rectangle()])
    assert all(item["wasserstein_distance"] == 0 for item in summary["features"].values())
    assert summary["room_type_js_divergence_bits"] == 0
    assert summary["room_type_frequencies"]["0"] == 1
    json.dumps(summary, allow_nan=False)


def test_distribution_shift_reports_original_units_and_semantic_difference():
    summary = distribution_shift([rectangle(label=1, width=256)], [rectangle(label=0)])
    assert summary["features"]["total_room_area_drawing_units2"]["wasserstein_distance"] == 128 * 128
    assert summary["room_type_js_divergence_bits"] == 1
    assert summary["features"]["rooms"]["wasserstein_distance"] == 0


def test_distribution_without_eligible_baseline_remains_unknown():
    summary = distribution_shift([rectangle()], [rectangle(valid=False)])
    assert summary["baseline_eligible_count"] == 0 and summary["baseline_excluded_count"] == 1
    assert all(item["wasserstein_distance"] is None for item in summary["features"].values())
    assert summary["room_type_js_divergence_bits"] is None
