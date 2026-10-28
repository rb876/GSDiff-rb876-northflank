from test_time.policy import ConstraintEvaluator, graph_validity
from test_time.types import FloorplanSample, Room, Scenario


def sample():
    return FloorplanSample(nodes=[[0, 0], [4, 0], [4, 4], [0, 4]],
                           edges=[[0, 1], [1, 2], [2, 3], [3, 0]],
                           rooms=[Room(0, [[0, 0], [4, 0], [4, 4], [0, 4], [0, 0]])])


def test_validity_and_compliance_are_independent():
    scenario = Scenario(coordinate_scale_m_per_model_unit=1, scale_provenance="assumed")
    result = ConstraintEvaluator().evaluate(sample(), scenario)
    assert result.structurally_valid
    assert result.applicable_count == 6  # C034 is auxiliary.
    assert result.unknown_count == 0
    assert result.assumption_based_count == 6
    assert any(item.status == "fail" for item in result.results)
    assert result.energy > 0
    assert result.reward == -result.energy


def test_unknowns_have_no_numeric_violation_and_no_invented_pass():
    result = ConstraintEvaluator().evaluate(sample(), Scenario())
    assert result.unknown_count == result.applicable_count == 6
    assert result.measurable_count == 0
    assert result.energy == 0
    assert all(item.status == "unknown" and item.severity is None for item in result.results)


def test_invalid_graph_penalty_is_separate():
    invalid = sample()
    invalid.edges.append([0, 99])
    result = ConstraintEvaluator(validity_weight=123).evaluate(invalid, Scenario())
    assert not result.structurally_valid
    assert result.energy == 123
    assert any("references" in reason for reason in result.diagnostics)


def test_crossing_edges_unclosed_rooms_and_overlaps():
    invalid = sample()
    invalid.edges += [[0, 2], [1, 3]]
    assert not graph_validity(invalid)[0]
    invalid = sample()
    invalid.rooms[0].polygon.pop()
    assert not graph_validity(invalid)[0]
    invalid = sample()
    invalid.rooms.append(Room(1, [[1, 1], [3, 1], [3, 3], [1, 3], [1, 1]]))
    assert "overlap" in " ".join(graph_validity(invalid)[2])


def test_nonapplicable_accessibility_is_excluded():
    scenario = Scenario(accessible_dwelling=False, coordinate_scale_m_per_model_unit=1, scale_provenance="verified")
    result = ConstraintEvaluator().evaluate(sample(), scenario)
    assert result.applicable_count == 3
    assert sum(item.status == "not_applicable" for item in result.results) == 4
