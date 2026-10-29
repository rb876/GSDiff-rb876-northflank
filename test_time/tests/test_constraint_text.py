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


def test_design_count_and_worst_bedroom_have_readable_reasons():
    count = constraint_description(saved("D001", "fail", {"bedroom_count": 2, "required_count": 3},
                                       {"minimum_count": 3}), SAMPLE)
    assert count["name"] == "Bedroom count"
    assert count["requirement"] == "Design brief: at least 3 bedrooms."
    assert "Found 2 bedrooms; below the minimum of 3" in count["reason"]
    area = constraint_description(saved("D002", "fail", {"bedroom_count": 2,
        "minimum_bedroom_area_m2": 9.9999, "required_area_m2": 10, "worst_room_index": 3},
        {"area_m2": 10}), SAMPLE)
    assert "every bedroom" in area["requirement"]
    assert "Bedroom 2: 9.9999 m²; below 10 m²" in area["reason"]
    assert "cannot compensate" in area["reason"]
    passed = constraint_description(saved("D002", "pass", {"bedroom_count": 2,
        "minimum_bedroom_area_m2": 11, "required_area_m2": 10, "worst_room_index": 1}), SAMPLE)
    assert "Every bedroom meets the minimum" in passed["reason"]


def test_design_living_room_and_budget_respect_saved_thresholds():
    living = constraint_description(saved("D003", "pass", {"living_rooms": [{"room_index": 0, "area_m2": 20}],
        "largest_living_room_area_m2": 20, "required_area_m2": 19.5, "room_index": 0},
        {"area_m2": 19.5}), SAMPLE)
    assert "19.5 m²" in living["requirement"]
    assert "Living 1: 20 m²; meets the 19.5 m² minimum" in living["reason"]
    assert "not added together" in living["reason"]
    budget = constraint_description(saved("D004", "fail", {"estimated_habitable_area_m2": 90.0001,
        "maximum_area_m2": 90}, {"maximum_area_m2": 90}), SAMPLE)
    assert "≤ 90 m²" in budget["requirement"]
    assert "90.0001 m²; exceeds the 90 m² budget" in budget["reason"]
    passed = constraint_description(saved("D004", "pass", {"estimated_habitable_area_m2": 85,
        "maximum_area_m2": 90}), SAMPLE)
    assert "is within" in passed["reason"]


def test_design_missing_rooms_fail_and_missing_measurements_stay_unresolved():
    bed = constraint_description(saved("D002", "fail", {"bedroom_count": 0}), SAMPLE)
    living = constraint_description(saved("D003", "fail", {"living_rooms": []}), SAMPLE)
    assert "No bedroom is present" in bed["reason"]
    assert "No living room is present" in living["reason"]
    unknown = saved("D004", "unknown", thresholds={"maximum_area_m2": 100})
    unknown["missing_fields"] = ["scale"]
    assert "missing drawing-to-metre scale" in constraint_description(unknown, SAMPLE)["reason"]
    legacy = saved("D002", "fail", explanation="Saved design requirement failed")
    assert constraint_description(legacy, SAMPLE)["reason"] == legacy["explanation"]


def test_real_design_checks_flow_into_measured_gallery_reasons(tmp_path):
    from dataclasses import asdict
    from pathlib import Path

    from test_time.analysis_gallery import build_gallery
    from test_time.constraints import load_specs
    from test_time.policy import ConstraintEvaluator
    from test_time.types import Scenario, read_json

    configs = Path(__file__).parents[1] / "configs"
    scenario = Scenario(**read_json(configs / "scenario_rplan18.json"))
    specs = [spec for spec in load_specs(configs / "constraints_design_v1.json")
             if spec.id.startswith("D") or spec.id in {"C079", "C080"}]
    # A real tiled 10m x 6m synthetic plan: living=16, bedrooms=8/12,
    # bathroom=4 and kitchen=20 m². Shared wall junctions are explicit.
    metric_rooms = [
        (0, [(0, 0), (4, 0), (4, 4), (0, 4)]),
        (1, [(4, 0), (6, 0), (6, 4), (4, 4)]),
        (1, [(6, 0), (9, 0), (9, 4), (6, 4)]),
        (4, [(9, 0), (10, 0), (10, 4), (9, 4)]),
        (3, [(0, 4), (4, 4), (6, 4), (9, 4), (10, 4), (10, 6), (0, 6)]),
    ]
    sample = FloorplanSample(scenario=asdict(scenario), metadata={"trial_seed": 10000})
    vertex_ids, edges = {}, set()
    for label, polygon in metric_rooms:
        drawing = [[x / scenario.coordinate_scale_m_per_model_unit,
                    y / scenario.coordinate_scale_m_per_model_unit] for x, y in polygon]
        indices = []
        for point in drawing:
            key = tuple(point)
            if key not in vertex_ids:
                vertex_ids[key] = len(sample.nodes)
                sample.nodes.append(point)
            indices.append(vertex_ids[key])
        edges.update(tuple(sorted(edge)) for edge in zip(indices, indices[1:] + indices[:1]))
        sample.rooms.append(Room(label, drawing + [drawing[0]]))
    sample.edges = [list(edge) for edge in sorted(edges)]
    evaluation = ConstraintEvaluator(specs).evaluate(sample, scenario)
    assert evaluation.structurally_valid
    checks = {value.id: value for value in evaluation.results if value.id.startswith("D")}
    descriptions = {key: constraint_description(value, sample) for key, value in checks.items()}
    assert {key: value.status for key, value in checks.items()} == {
        "D001": "fail", "D002": "fail", "D003": "fail", "D004": "pass"}
    assert "Found 2 bedrooms; below the minimum of 3" in descriptions["D001"]["reason"]
    assert "Bedroom 1: 8 m²; below 10 m²" in descriptions["D002"]["reason"]
    assert "Living 1: 16 m²; below the 18 m² minimum" in descriptions["D003"]["reason"]
    assert "60 m²; is within the 100 m² budget" in descriptions["D004"]["reason"]
    assert "not recorded" not in str(descriptions)
    row = {"trial": 0, "seed": 10000, "selected": True, "candidate_id": "synthetic-measured",
           "sample": asdict(sample), "evaluation": asdict(evaluation)}
    output = tmp_path / "measured_gallery"
    gallery = build_gallery([{"label": "Synthetic measured design checks", "run_dir": tmp_path,
                              "rows": [row], "summary": {"trial_seed_start": 10000}}], output)
    html = Path(gallery["html"]).read_text()
    example = gallery["examples"]["Synthetic measured design checks"]["invalid"]
    graph = read_json(output / example["graph"])
    assert graph["scenario"]["coordinate_scale_m_per_model_unit"] == 0.03515625
    assert "paper_derived scale:" in (output / example["svg"]).read_text()
    assert "not recorded" not in html
    assert "Bedroom 1: 8 m²; below 10 m²" in html
