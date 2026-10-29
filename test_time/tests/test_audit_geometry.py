from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
from shapely.geometry import Point, Polygon, box
from shapely.ops import unary_union

from test_time import audit_geometry
from test_time.audit_geometry import AUDIT_CONTAINMENT_TOLERANCE_M, refined_bedroom_search
from test_time.constraints import load_specs
from test_time.geometry import bedroom_search
from test_time.policy import ConstraintEvaluator
from test_time.types import FloorplanSample, Room, Scenario


BED = (1.4, 1.9, 0.75, [[0.9, 1.2], [1.2, 0.9]])


def contact_room():
    # The side protrusion shifts the drawing grid away from the only envelope
    # placement in the main rectangle: bed x=0.925, rather than a 5 cm grid point.
    return unary_union([box(0.025, 0, 3.225, 3.4), box(0, 1, 0.1, 1.2)])


def assert_joint_witness(room, result):
    witness = result["joint"]
    assert witness["status"] == "pass"
    bed = Polygon(witness["bed_polygon_m"])
    clearance = Polygon(witness["clearance_polygon_m"])
    center = Point(witness["circle_center_m"])
    tolerant_room = room.buffer(AUDIT_CONTAINMENT_TOLERANCE_M)
    assert tolerant_room.covers(bed) and tolerant_room.covers(clearance)
    assert center.distance(room.boundary) >= 0.75 - 1e-8
    assert room.covers(center) and center.distance(bed) >= 0.75 - 1e-8
    assert bed.area == pytest.approx(1.4 * 1.9)
    assert result["turning"]["bed_polygon_m"] == result["circulation"]["bed_polygon_m"]


def test_contact_placement_resolves_real_reference_search_miss():
    room = contact_room()
    original = bedroom_search(room, *BED)
    assert original["joint"]["status"] == "unknown"
    refined = refined_bedroom_search(room, *BED)
    assert_joint_witness(room, refined)
    assert refined["search"]["audit_positions_examined"] <= 20000
    assert refined["search"]["audit_circle_fit_calls"] <= 64


def test_free_space_circle_fallback_certifies_geometry(monkeypatch):
    # Model a finite circle-witness search that retains no useful center. The
    # fallback must find a real circle in room-minus-bed, not infer a pass.
    monkeypatch.setattr(audit_geometry, "_circle_centers", lambda *args: np.empty((0, 2)))
    room = contact_room()
    refined = refined_bedroom_search(room, *BED)
    assert_joint_witness(room, refined)
    assert refined["joint"]["circle_search_method"] == "free_space_polylabel"
    assert 0 < refined["search"]["audit_circle_fit_calls"] <= 64


@pytest.mark.parametrize("room", [box(0, 0, 4, 4), box(0, 0, 3, 3), box(0, 0, 1, 1)])
def test_existing_passes_and_geometric_failure_proofs_are_preserved(room):
    original = bedroom_search(room, *BED)
    refined = refined_bedroom_search(room, *BED)
    for key in ("bed", "turning", "circulation", "joint"):
        if original[key]["status"] in {"pass", "fail"}:
            assert refined[key] == original[key]


def test_separate_witnesses_and_exhausted_search_do_not_invent_success():
    # A 3.2 x 3.1 room supports separate placements but no shared arrangement
    # witnessed by either search. Sampling more beds is not a global fail proof.
    refined = refined_bedroom_search(box(0, 0, 3.2, 3.1), *BED, max_positions=500)
    assert refined["joint"]["status"] == "unknown"
    assert refined["circulation"]["status"] == "pass"
    assert refined["turning"]["status"] == "pass"
    limited = refined_bedroom_search(contact_room(), *BED, max_positions=1)
    assert limited["joint"]["status"] == "unknown"
    assert limited["search"]["audit_positions_examined"] <= 1
    assert limited["search"]["audit_budget_exhausted"]


def test_audit_is_opt_in_and_does_not_change_default_evaluation():
    polygon = [list(point) for point in contact_room().exterior.coords]
    nodes = polygon[:-1]
    sample = FloorplanSample(nodes=nodes, edges=[[i, (i + 1) % len(nodes)] for i in range(len(nodes))],
                            rooms=[Room(1, polygon)])
    scenario = Scenario(coordinate_scale_m_per_model_unit=1, scale_provenance="verified")
    specs = load_specs(Path(__file__).parents[1] / "configs" / "constraints_v2.json")
    original = ConstraintEvaluator(specs).evaluate(sample, scenario)
    refined = ConstraintEvaluator(specs, bedroom_search_fn=refined_bedroom_search).evaluate(sample, scenario)
    original_checks = {item.id: item for item in original.results}
    refined_checks = {item.id: item for item in refined.results}
    assert original_checks["C035"].status == "unknown"
    assert refined_checks["C035"].status == "pass"
    for constraint_id in original_checks.keys() - {"C033", "C035"}:
        assert original_checks[constraint_id] == refined_checks[constraint_id]
    assert asdict(ConstraintEvaluator(specs).evaluate(sample, scenario)) == asdict(original)


def test_saved_full_svdd_trial19_boundary_rounding_regression():
    # Actual room 4 from saved full-SVDD trial 19, converted using its original
    # 0.025 m/pixel scale. Grid rounding gives y=6.975 rather than
    # 6.9750000000000005 and strict containment previously rejected every witness.
    pixels = [[175, 421], [263, 421], [263, 279], [103, 279],
              [103, 391], [175, 391], [175, 421]]
    room = Polygon([(x * 0.025, y * 0.025) for x, y in pixels])
    assert bedroom_search(room, *BED)["joint"]["status"] == "unknown"
    refined = refined_bedroom_search(room, *BED)
    assert_joint_witness(room, refined)
    assert refined["joint"]["containment_tolerance_m"] == 1e-8


@pytest.mark.parametrize("pixels", [
    [[180, 300], [169, 287], [169, 121], [80, 121], [80, 300], [180, 300]],
    [[180, 300], [207, 299], [207, 178], [207, 121], [169, 121], [169, 287], [180, 300]],
    [[299, 121], [207, 121], [207, 178], [299, 178], [299, 121]],
    [[299, 121], [299, 178], [299, 276], [333, 276], [393, 276], [393, 121], [299, 121]],
])
def test_saved_trial50_bedrooms_cannot_fit_axis_aligned_clearance_envelope(pixels):
    room = Polygon([(x * 0.025, y * 0.025) for x, y in pixels])
    original = bedroom_search(room, *BED)
    refined = refined_bedroom_search(room, *BED)
    assert refined["circulation"]["status"] == refined["joint"]["status"] == "fail"
    if original["turning"]["status"] == "pass":
        assert refined["turning"]["status"] == "pass"
    if original["circulation"]["status"] == "unknown":
        assert "bounding box" in refined["circulation"]["reason"]


def test_bbox_proof_preserves_tolerance_and_does_not_fail_turning():
    within_tolerance = refined_bedroom_search(box(0, 0, 3.2 - 1e-9, 3.15), *BED)
    assert within_tolerance["circulation"]["status"] != "fail"
    too_narrow = refined_bedroom_search(box(0, 0, 3.2 - 1e-5, 3.15), *BED)
    assert too_narrow["circulation"]["status"] == "fail"
    assert "bounding box" in too_narrow["circulation"]["reason"]
    assert too_narrow["turning"]["status"] == "pass"
