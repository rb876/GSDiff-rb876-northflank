import pytest
from shapely.geometry import Point, box
from test_time.geometry import bedroom_search, circle_fit, polygon_from_points


def test_rectangle_area_points_and_circle_fit():
    room = polygon_from_points([[0, 0], [3, 0], [3, 2], [0, 2], [0, 0]])
    assert room.area == pytest.approx(6)
    assert room.contains(Point(1, 1))
    assert circle_fit(room, 0.75)["status"] == "pass"
    assert circle_fit(box(0, 0, 1.4, 2), 0.75)["status"] == "fail"


def test_invalid_polygon_is_not_repaired():
    with pytest.raises(ValueError, match="invalid"):
        polygon_from_points([[0, 0], [2, 2], [0, 2], [2, 0], [0, 0]])


def test_shared_bed_circle_and_circulation_witness():
    room = box(0, 0, 4, 4)
    found = bedroom_search(room, 1.4, 1.9, 0.75, [[0.9, 1.2], [1.2, 0.9]])
    assert found["joint"]["status"] == "pass"
    assert found["turning"]["bed_polygon_m"] == found["circulation"]["bed_polygon_m"]
    witness = found["joint"]
    bed = polygon_from_points(witness["bed_polygon_m"])
    center = Point(witness["circle_center_m"])
    assert room.covers(bed)
    assert center.distance(room.boundary) >= 0.75 - 1e-8
    assert center.distance(bed) >= 0.75 - 1e-8


def test_proven_small_area_failure_and_finite_search_unknown():
    small = bedroom_search(box(0, 0, 1, 1), 1.4, 1.9, 0.75, [[0.9, 1.2], [1.2, 0.9]])
    assert small["bed"]["status"] == "fail"
    limited = bedroom_search(box(0, 0, 4, 4), 1.4, 1.9, 0.75,
                             [[0.9, 1.2], [1.2, 0.9]], max_positions=1)
    assert limited["joint"]["status"] == "unknown"
    assert limited["search"]["budget_exhausted"]
