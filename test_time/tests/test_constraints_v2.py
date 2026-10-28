"""The feasible v2 pool adds supporting definitions, not invented observations."""

from collections import Counter
from dataclasses import asdict, replace
import json
from pathlib import Path

import pytest

from test_time.area import area_policy
from test_time.constraints import CheckContext, check, load_specs
from test_time.policy import ConstraintEvaluator
from test_time.types import FloorplanSample, Room, Scenario

CONFIGS = Path(__file__).parents[1] / "configs"


@pytest.fixture
def specs():
    return load_specs(CONFIGS / "constraints_v2.json")


@pytest.fixture
def scenario():
    return Scenario(coordinate_scale_m_per_model_unit=1, scale_provenance="assumed")


def rectangle(label, width=10, height=10, x=0):
    return Room(label, [[x, 0], [x + width, 0], [x + width, height], [x, height], [x, 0]])


def results(sample, scenario, specs):
    context = CheckContext(sample, scenario, specs)
    return {spec.id: check(sample, scenario, spec, context) for spec in specs if spec.enabled}


def test_registry_counts_selection_and_groups(specs):
    assert len(specs) == len({spec.id for spec in specs}) == 9
    assert Counter(spec.group for spec in specs) == {
        "bedroom_accessibility": 3, "bathroom_fixtures_turning": 1,
        "habitability": 5,
    }
    assert {spec.id for spec in specs if spec.enabled and spec.counts_toward_compliance} == {
        "C033", "C035", "C036", "C076", "C077", "C078"}
    assert {spec.id for spec in specs if spec.enabled and spec.role == "supporting_rule"} == {"C034", "C079", "C080"}
    assert all(spec.enabled for spec in specs)


def test_supporting_definitions_never_pass_or_enter_denominator(specs, scenario):
    sample = FloorplanSample(rooms=[rectangle(0)])
    checked = results(sample, scenario, specs)
    for id in ("C034", "C079", "C080"):
        item = checked[id]
        assert item.status == "not_applicable"
        assert item.severity is None and not item.measurable and not item.counts_toward_compliance
        assert item.observed_values["definition"]
    evaluation = ConstraintEvaluator(specs).evaluate(sample, scenario)
    assert evaluation.applicable_count == 6


@pytest.mark.parametrize("height,area,status", [(1.7999, 0, "fail"), (1.8, 100, "pass"), (2.5, 100, "pass")])
def test_headroom_boundary_affects_habitable_area_only(specs, scenario, height, area, status):
    scenario = replace(scenario, ceiling_height_m=height)
    sample = FloorplanSample(rooms=[rectangle(0)])
    checked = results(sample, scenario, specs)
    assert checked["C077"].observed_values["estimated_habitable_area_m2"] == area
    assert checked["C078"].observed_values["estimated_habitable_area_m2"] == area
    assert checked["C076"].observed_values["area_m2"] == area
    assert all(checked[id].status == status for id in ("C076", "C077", "C078"))
    physical_rooms = FloorplanSample(rooms=[rectangle(1, 4, 4), rectangle(4, 2, 2, x=5)])
    physical = results(physical_rooms, scenario, specs)
    assert all(physical[id].status == "pass" for id in ("C033", "C035", "C036"))


@pytest.mark.parametrize("changes,missing", [
    ({"ceiling_height_m": None}, "ceiling_height"),
    ({"habitable_area_mode": "unknown"}, "habitable_area_mode"),
    ({"coordinate_scale_m_per_model_unit": None, "scale_provenance": "missing"}, "scale"),
])
def test_unavailable_area_inputs_are_unknown(specs, scenario, changes, missing):
    scenario = replace(scenario, **changes)
    sample = FloorplanSample(rooms=[rectangle(0)])
    checked = results(sample, scenario, specs)
    assert not area_policy(specs, scenario)["available"]
    for id in ("C076", "C077", "C078"):
        assert checked[id].status == "unknown"
        assert checked[id].severity is None
        assert missing in checked[id].missing_fields
    assert checked["C079"].status == checked["C080"].status == "not_applicable"


def test_single_shared_label_table_and_polygon_union(specs, scenario):
    sample = FloorplanSample(rooms=[rectangle(0, 4, 4), rectangle(1, 4, 4, x=2),
                                   rectangle(2, 10, 10, x=20), rectangle(5, 10, 10, x=40),
                                   rectangle(6, 10, 10, x=60)])
    checked = results(sample, scenario, specs)
    assert checked["C077"].observed_values["estimated_habitable_area_m2"] == pytest.approx(24)
    assert checked["C078"].observed_values["estimated_habitable_volume_m3"] == pytest.approx(60)
    for spec in specs:
        if spec.id in {"C077", "C078"}:
            assert "included_room_labels" not in spec.thresholds
    serialized = asdict(checked["C077"])
    assert serialized["group"] == "habitability"
    assert serialized["claim_level"] == "assumption_based"
    assert "zero_thickness_habitable_area_proxy" in serialized["assumption_flags"]


def test_legacy_default_has_seven_and_surface_stays_height_independent(scenario):
    legacy = load_specs()
    assert len(legacy) == 7
    scenario = replace(scenario, ceiling_height_m=None, habitable_area_mode="unknown")
    checked = results(FloorplanSample(rooms=[rectangle(0)]), scenario, legacy)
    assert checked["C077"].status == "pass"
    assert checked["C076"].status == checked["C078"].status == "unknown"
    assert not area_policy(legacy, scenario)["enabled"]


@pytest.mark.parametrize("change,error", [
    ("unknown_checker", "unknown checker"), ("weight_support", "supporting rules"),
    ("drop_definition", "enabled dependencies"), ("bad_group", "incorrect v2 group"),
    ("extra_id", "nine implemented IDs"), ("overlapping_labels", "partition"),
])
def test_invalid_registry_changes_fail_actionably(tmp_path, change, error):
    data = json.loads((CONFIGS / "constraints_v2.json").read_text())
    by_id = {entry["id"]: entry for entry in data["constraints"]}
    if change == "unknown_checker":
        by_id["C033"]["checker"] = "unavailable"
        by_id["C033"]["enabled"] = False
    elif change == "weight_support":
        by_id["C079"]["weight"] = 1
    elif change == "drop_definition":
        by_id["C080"]["enabled"] = False
    elif change == "bad_group":
        by_id["C033"]["group"] = "habitability"
    elif change == "extra_id":
        data["constraints"].append({**by_id["C036"], "id": "C082"})
    else:
        by_id["C080"]["thresholds"]["included_room_labels"].append(2)
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match=error):
        load_specs(path)


def test_scenario_area_mode_is_explicit():
    with pytest.raises(ValueError, match="habitable_area_mode"):
        Scenario(habitable_area_mode="verified_wall_area")
