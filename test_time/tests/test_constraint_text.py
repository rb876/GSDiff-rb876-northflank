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


def test_ventilation_describes_saved_flow_and_main_room_count():
    kitchen = constraint_description(saved("C085", "fail", {
        "main_room_count": 4, "room_count": 1,
        "observed_flow_m3_h": 105, "required_flow_m3_h": 120}), SAMPLE)
    assert "normal extraction ≥ 120 m³/h for 4 main rooms" in kitchen["requirement"]
    assert "105 m³/h; below the 120 m³/h minimum" in kitchen["reason"]
    assert "scenario input, not a model prediction" in kitchen["reason"]
    bathroom = constraint_description(saved("C086", "pass", {
        "main_room_count": 3, "room_count": 2,
        "observed_flow_m3_h": 30, "required_flow_m3_h": 30}), SAMPLE)
    assert "2 bathroom(s)" in bathroom["reason"]
    assert "meets the 30 m³/h minimum" in bathroom["reason"]


def test_reduced_ventilation_shows_both_requirements_and_missing_inputs():
    reduced = constraint_description(saved("C090", "fail", {
        "main_room_count": 4, "observed_total_flow_m3_h": 75,
        "required_total_flow_m3_h": 90, "observed_kitchen_flow_m3_h": 45,
        "required_kitchen_flow_m3_h": 45}), SAMPLE)
    assert "total extraction ≥ 90 m³/h AND each kitchen ≥ 45 m³/h" in reduced["requirement"]
    assert "75 m³/h (below 90)" in reduced["reason"]
    assert "45 m³/h (meets 45)" in reduced["reason"]
    missing = saved("C090", "unknown")
    missing["missing_fields"] = ["ventilation_mode", "reduced_total_extraction_m3_h"]
    assert "missing ventilation operating mode, declared reduced total extraction" in constraint_description(missing, SAMPLE)["reason"]


def test_real_cpu_ventilation_measurements_flow_into_descriptions():
    from pathlib import Path
    from test_time.constraints import CheckContext, check, load_specs
    from test_time.types import Scenario, read_json

    configs = Path(__file__).parents[1] / "configs"
    scenario = Scenario(**read_json(configs / "scenario_ventilation_v1.json"))
    specs = load_specs(configs / "constraints_ventilation_v1.json")
    # One living room, three bedrooms, one kitchen and one bathroom: four main rooms.
    sample = FloorplanSample(rooms=[Room(label, [[x, 0], [x + 1, 0], [x + 1, 1], [x, 1]])
                                   for x, label in enumerate((0, 1, 1, 1, 3, 4))])
    context = CheckContext(sample, scenario, specs)
    checks = {spec.id: check(sample, scenario, spec, context) for spec in specs
              if spec.id in {"C085", "C086", "C090"}}
    descriptions = {identifier: constraint_description(value, sample) for identifier, value in checks.items()}
    assert {key: value.status for key, value in checks.items()} == {
        "C085": "fail", "C086": "pass", "C090": "fail"}
    assert "4 main rooms" in descriptions["C085"]["reason"]
    assert "105 m³/h; below the 120 m³/h minimum" in descriptions["C085"]["reason"]
    assert "30 m³/h; meets the 30 m³/h minimum" in descriptions["C086"]["reason"]
    assert "75 m³/h (below 90)" in descriptions["C090"]["reason"]
    assert "45 m³/h (meets 45)" in descriptions["C090"]["reason"]
    assert "not recorded" not in str(descriptions)
