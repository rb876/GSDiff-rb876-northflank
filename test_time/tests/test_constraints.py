from dataclasses import replace
import json
from pathlib import Path

import pytest

from test_time.constraints import CheckContext, check, load_specs
from test_time.types import FloorplanSample, Room, Scenario


def rectangle(label, width, height, x=0, y=0):
    return Room(label, [[x, y], [x + width, y], [x + width, y + height], [x, y + height], [x, y]])


@pytest.fixture
def specs():
    return {spec.id: spec for spec in load_specs()}


@pytest.fixture
def scenario():
    return Scenario(coordinate_scale_m_per_model_unit=1.0, scale_provenance="verified")


@pytest.mark.parametrize("area,height,status", [(9, 2.2, "pass"), (8, 2.5, "pass"), (9, 2.0, "fail"), (10, 2.0, "pass")])
def test_c076_or_branches(area, height, status, specs, scenario):
    sample = FloorplanSample(rooms=[rectangle(0, area, 1)])
    result = check(sample, replace(scenario, ceiling_height_m=height), specs["C076"])
    assert result.status == status


@pytest.mark.parametrize("occupants,area,volume", [(1, 14, 33), (4, 56, 132), (5, 66, 155)])
def test_occupancy_threshold_changes_after_four(occupants, area, volume, specs, scenario):
    scenario = replace(scenario, intended_occupants=occupants, ceiling_height_m=1.0)
    sample = FloorplanSample(rooms=[rectangle(0, area, 1)])
    result = check(sample, scenario, specs["C077"])
    assert result.status == "pass"
    assert result.observed_values["required_area_m2"] == area
    sample.rooms = [rectangle(0, volume, 1)]
    result = check(sample, scenario, specs["C078"])
    assert result.status == "pass"
    assert result.observed_values["required_volume_m3"] == volume
    sample.rooms = [rectangle(0, volume - 1, 1)]
    assert check(sample, scenario, specs["C078"]).status == "fail"


def test_missing_scale_height_and_applicability_are_distinct(specs, scenario):
    sample = FloorplanSample(rooms=[rectangle(0, 5, 5)])
    for spec in specs.values():
        assert check(sample, Scenario(), spec).status == "unknown"
    no_height = replace(scenario, ceiling_height_m=None)
    assert check(sample, no_height, specs["C076"]).status == "unknown"
    assert check(sample, no_height, specs["C078"]).severity is None
    assert check(sample, replace(scenario, accessible_dwelling=False), specs["C033"]).status == "not_applicable"
    assert check(sample, scenario, specs["C033"]).status == "fail"
    assert check(sample, scenario, specs["C036"]).status == "fail"


def test_joint_bed_witness_and_auxiliary_exclusion(specs, scenario):
    sample = FloorplanSample(rooms=[rectangle(1, 4, 4)])
    context = CheckContext(sample, scenario, specs.values())
    turning = check(sample, scenario, specs["C033"], context)
    circulation = check(sample, scenario, specs["C035"], context)
    assert turning.status == circulation.status == "pass"
    assert turning.observed_values["bed_polygon_m"] == circulation.observed_values["bed_polygon_m"]
    assert not check(sample, scenario, specs["C034"], context).counts_toward_compliance


def test_separate_bed_witnesses_cannot_count_as_joint_pass(specs, scenario):
    # Exactly enough room for the first circulation envelope (3.2 x 3.1 m).
    # A corner bed leaves room for a circle, while the centred bed is needed for
    # side clearances; combining these two different placements would be wrong.
    sample = FloorplanSample(rooms=[rectangle(1, 3.2, 3.1)])
    context = CheckContext(sample, scenario, specs.values())
    placements = context.bedrooms()[0]
    assert placements["turning"]["status"] == placements["circulation"]["status"] == "pass"
    assert placements["turning"]["bed_polygon_m"] != placements["circulation"]["bed_polygon_m"]
    assert placements["joint"]["status"] == "unknown"
    assert check(sample, scenario, specs["C033"], context).status == "pass"
    result = check(sample, scenario, specs["C035"], context)
    assert result.status == "unknown"
    assert result.severity is None


def test_area_union_avoids_overlap_and_excludes_storage(specs, scenario):
    sample = FloorplanSample(rooms=[rectangle(0, 4, 4), rectangle(1, 4, 4, x=2), rectangle(2, 10, 10, x=20)])
    result = check(sample, scenario, specs["C077"])
    assert result.observed_values["estimated_habitable_area_m2"] == pytest.approx(24)


def test_invalid_geometry_becomes_unknown(specs, scenario):
    sample = FloorplanSample(rooms=[Room(0, [[0, 0], [2, 2], [0, 2], [2, 0], [0, 0]])])
    result = check(sample, scenario, specs["C076"])
    assert result.status == "unknown"
    assert "Unreliable" in result.explanation


@pytest.mark.parametrize("field,value,message", [("checker", "typo", "unknown checker"), ("weight", -1, "nonnegative"), ("thresholds", {}, "missing thresholds")])
def test_registry_rejects_bad_configuration(tmp_path, field, value, message):
    source = Path(__file__).parents[1] / "configs" / "constraints_v1.json"
    data = json.loads(source.read_text())
    data["constraints"][0][field] = value
    path = tmp_path / "constraints.json"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match=message):
        load_specs(path)
