from test_time.constraint_text import constraint_description, room_labels
from test_time.types import FloorplanSample, Room


SAMPLE = {"rooms": [{"label": kind} for kind in (0, 1, 3, 1, 4)]}


def saved(identifier, status, observed=None, thresholds=None, explanation=""):
    return {"id": identifier, "status": status, "observed_values": observed or {},
            "threshold_values": thresholds or {}, "explanation": explanation}


def test_room_labels_match_type_indices_and_dataclasses():
    assert room_labels(SAMPLE) == {0: "Living 1", 1: "Bedroom 1", 2: "Kitchen 1",
                                   3: "Bedroom 2", 4: "Bathroom 1"}
    assert room_labels(FloorplanSample(rooms=[Room(1, [])])) == {0: "Bedroom 1"}


def test_c035_actual_two_small_bedrooms_explains_necessary_bound():
    rooms = [{"room_index": index, "circulation": {"room_area_m2": area, "necessary_area_m2": 9.92}}
             for index, area in ((1, 4.6166), (3, 7.3716))]
    description = constraint_description(saved("C035", "fail", {"bedrooms": rooms},
                                             {"clearance_options_m": [[0.9, 1.2], [1.2, 0.9]]}), SAMPLE)
    assert "Bedroom 1: 4.62 m²" in description["reason"]
    assert "Bedroom 2: 7.37 m²" in description["reason"]
    assert "7.37 m² < 9.92 m² needed for bed + clearances" in description["reason"]
    assert "both long sides" in description["requirement"]
    assert "same bed" in description["requirement"]


def test_passed_placements_name_actual_room_and_shared_arrangement():
    observed = {"room_index": 3, "bed_polygon_m": [[0, 0]], "bed_dimensions_m": [1.4, 1.9],
                "circle_radius_m": 0.75, "joint_bed_placement": True}
    turning = constraint_description(saved("C033", "pass", observed, {"diameter_m": 1.5}), SAMPLE)
    clearance = constraint_description(saved("C035", "pass", observed), SAMPLE)
    assert "Bedroom 2 fits a 1.4 × 1.9 m bed" in turning["reason"]
    assert "separate 1.5 m diameter" in turning["reason"]
    assert "same arrangement" in clearance["reason"]


def test_c036_uses_witness_lower_bound_and_failure_upper_bound():
    passed = constraint_description(saved("C036", "pass", {"radius_lower_bound_m": 0.8,
                                          "radius_upper_bound_m": 0.805}, {"diameter_m": 1.5}), SAMPLE)
    assert "at least 1.6 m in diameter" in passed["reason"]
    assert "1.5 m diameter" in passed["requirement"]
    assert "1.61" not in passed["reason"]
    failed = constraint_description(saved("C036", "fail", {"bathrooms": [
        {"radius_lower_bound_m": 0.6, "radius_upper_bound_m": 0.605},
        {"radius_lower_bound_m": 0.7, "radius_upper_bound_m": 0.705}]}, {"diameter_m": 1.5}), SAMPLE)
    assert "upper bound of 1.41 m < 1.5 m" in failed["reason"]


def test_c076_area_and_height_are_alternative_to_volume():
    limits = {"area_m2": 9, "height_m": 2.2, "volume_m3": 20}
    volume_pass = constraint_description(saved("C076", "pass", {"area_m2": 8.5, "height_m": 2.5,
                                              "volume_m3": 21.25}, limits), SAMPLE)
    assert "AND" in volume_pass["requirement"] and "OR volume" in volume_pass["requirement"]
    assert "alternative volume route passes" in volume_pass["reason"]
    failed = constraint_description(saved("C076", "fail", {"area_m2": 9.5, "height_m": 2,
                                          "volume_m3": 19}, limits), SAMPLE)
    assert "Neither the area-and-height route nor the volume route" in failed["reason"]


def test_habitable_area_volume_use_saved_totals_and_occupants():
    area = constraint_description(saved("C077", "pass", {"estimated_habitable_area_m2": 31.123,
                                        "required_area_m2": 28, "occupants": 2}), SAMPLE)
    volume = constraint_description(saved("C078", "fail", {"estimated_habitable_volume_m3": 65.9999,
                                          "required_volume_m3": 66, "occupants": 2}), SAMPLE)
    assert "28 m² for 2 occupants" in area["requirement"]
    assert "31.12 m²" in area["reason"]
    assert "65.9999 m³; below the 66 m³" in volume["reason"]


def test_unknown_does_not_mean_infeasible_and_missing_data_is_clear():
    unknown = constraint_description(saved("C035", "unknown", explanation="Finite placement search found no witness; this is not proof of infeasibility"), SAMPLE)
    assert "did not prove" in unknown["reason"]
    shared = constraint_description(saved("C035", "unknown", explanation="Circulation fits, but no shared bed-and-turning-circle witness was found"), SAMPLE)
    assert "unresolved, not a proven failure" in shared["reason"]
    missing = saved("C077", "unknown")
    missing["missing_fields"] = ["scale"]
    assert "missing drawing-to-metre scale" in constraint_description(missing, SAMPLE)["reason"]
    geometry = saved("C033", "unknown", explanation="Unreliable extracted geometry: room polygon is invalid")
    assert "room geometry is unreliable" in constraint_description(geometry, SAMPLE)["reason"]


def test_bbox_fail_near_threshold_does_not_round_to_a_false_fit():
    check = saved("C035", "fail", {"bedrooms": [{"room_index": 1, "circulation": {
        "room_bbox_dimensions_m": [2.7999999, 4.2],
        "required_envelope_dimensions_m": [[3.2, 3.1], [3.1, 3.2], [3.8, 2.8], [2.8, 3.8]]}}]})
    text = constraint_description(check, SAMPLE)["reason"]
    assert "2.7999999 × 4.2 m" in text
    assert "3.2 × 3.1 m or 3.8 × 2.8 m" in text
    assert "either orientation" in text


def test_missing_legacy_measurements_use_text_without_inventing_values():
    partial = saved("C035", "fail", explanation="The required bedroom is absent")
    description = constraint_description(partial, {})
    assert description["reason"] == "The required bedroom is absent"
    assert "9.92" not in str(description)
    support = saved("C034", "not_applicable")
    support["counts_toward_compliance"] = False
    assert "not an independent pass/fail" in constraint_description(support, SAMPLE)["reason"]
