"""Constraint registry and strict pass/fail/unknown semantics."""

import json
import math
from pathlib import Path

from shapely import affinity
from shapely.ops import unary_union

from .geometry import EPS, bedroom_search, circle_fit, polygon_from_points
from .types import ConstraintResult, ConstraintSpec


def deficit(observed, minimum):
    return max(0.0, (minimum - observed) / minimum)


APPLICABILITY = {
    "residential": lambda scenario: scenario.building_use == "residential",
    "accessible_residential": lambda scenario: scenario.building_use == "residential" and scenario.accessible_dwelling,
}


class CheckContext:
    """Shared decoded geometry and one bedroom search per complete graph."""

    def __init__(self, sample, scenario, specs):
        self.sample, self.scenario = sample, scenario
        self.specs = {spec.id: spec for spec in specs}
        self.rooms, self.geometry_error = [], None
        self._bedrooms = None
        scale = scenario.coordinate_scale_m_per_model_unit
        if scale is not None:
            try:
                self.rooms = [(room.label, affinity.scale(polygon_from_points(room.polygon),
                                                        xfact=scale, yfact=scale, origin=(0, 0)))
                              for room in sample.rooms]
            except ValueError as error:
                self.geometry_error = str(error)
        if sample.diagnostics:
            self.geometry_error = "; ".join(sample.diagnostics)

    def bedrooms(self):
        if self._bedrooms is None:
            bed = self.specs["C034"].thresholds
            turning = self.specs["C033"].thresholds
            circulation = self.specs["C035"].thresholds
            width = bed["single_width_m"] if self.scenario.intended_occupants == 1 else bed["multiple_width_m"]
            self._bedrooms = []
            for index, (label, polygon) in enumerate(self.rooms):
                if label in bed["bedroom_labels"]:
                    search = bedroom_search(polygon, width, bed["length_m"], turning["diameter_m"] / 2,
                                            circulation["clearance_options_m"], self.scenario.geometry_grid_m,
                                            self.scenario.geometry_refine_m, turning["max_positions"])
                    self._bedrooms.append({"room_index": index, **search})
        return self._bedrooms

    def area(self, labels):
        polygons = [polygon for label, polygon in self.rooms if label in labels]
        return unary_union(polygons).area if polygons else 0.0


def _result(spec, scenario, status, severity=None, observed=None, explanation=""):
    flags = list(spec.assumptions)
    if scenario.scale_provenance == "assumed":
        flags.append("assumed_metric_scale")
    if "ceiling_height" in spec.required_observations:
        flags.append("scenario_uniform_ceiling_height")
    return ConstraintResult(
        id=spec.id, status=status, severity=severity,
        applicable=status != "not_applicable", measurable=status in {"pass", "fail"},
        source_id=spec.source_id, observed_values=observed or {}, threshold_values=spec.thresholds,
        assumption_flags=flags, explanation=explanation,
        counts_toward_compliance=spec.counts_toward_compliance,
    )


def _bed_check(context, spec, part):
    bedrooms = context.bedrooms()
    if not bedrooms:
        return "fail", 1.0, {}, "The required bedroom is absent"
    joints = [room for room in bedrooms if room["joint"]["status"] == "pass"]
    if joints and part in {"turning", "circulation"}:
        room = joints[0]
        return "pass", 0.0, {"room_index": room["room_index"], **room["joint"], "search": room["search"]}, "One bed placement witnesses both bedroom checks"
    passes = [room for room in bedrooms if room[part]["status"] == "pass"]
    if passes:
        room = passes[0]
        observed = {"room_index": room["room_index"], **room[part], "search": room["search"]}
        # Avoid declaring two independently found, incompatible bed placements a joint success.
        if part == "circulation" and context.specs["C033"].enabled:
            return "unknown", None, observed, "Circulation fits, but no shared bed-and-turning-circle witness was found"
        return "pass", 0.0, observed, "A geometric placement witness was found; unobserved furnishings remain excluded"
    if all(room[part]["status"] == "fail" for room in bedrooms):
        return "fail", 1.0, {"bedrooms": bedrooms}, "Every bedroom violates a necessary geometric bound"
    return "unknown", None, {"bedrooms": bedrooms}, "Finite placement search found no witness; this is not proof of infeasibility"


def bedroom_turning(context, spec):
    return _bed_check(context, spec, "turning")


def bed_footprint(context, spec):
    return _bed_check(context, spec, "bed")


def bedroom_circulation(context, spec):
    return _bed_check(context, spec, "circulation")


def bathroom_turning(context, spec):
    polygons = [polygon for label, polygon in context.rooms if label in spec.thresholds["bathroom_labels"]]
    if not polygons:
        return "fail", 1.0, {}, "The required bathroom is absent"
    radius = spec.thresholds["diameter_m"] / 2
    checks = [circle_fit(polygon, radius, tolerance=min(context.scenario.geometry_refine_m / 4, 0.005))
              for polygon in polygons]
    passed = next((check for check in checks if check["status"] == "pass"), None)
    if passed:
        return "pass", 0.0, passed, "An empty bathroom can contain the circle; fixtures and door swings are unobserved"
    if all(check["status"] == "fail" for check in checks):
        severity = min(deficit(check["radius_lower_bound_m"], radius) for check in checks)
        return "fail", severity, {"bathrooms": checks}, "All bathroom inradius upper bounds are below the required radius"
    return "unknown", None, {"bathrooms": checks}, "Circle feasibility is unresolved at the search tolerance"


def minimum_main_room(context, spec):
    limits = spec.thresholds
    height = context.scenario.ceiling_height_m
    rooms = []
    for label, polygon in context.rooms:
        if label in limits["main_room_labels"]:
            area, volume = polygon.area, polygon.area * height
            branch_area_height = max(deficit(area, limits["area_m2"]), deficit(height, limits["height_m"]))
            branch_volume = deficit(volume, limits["volume_m3"])
            rooms.append({"area_m2": area, "height_m": height, "volume_m3": volume,
                          "severity": min(branch_area_height, branch_volume)})
    if not rooms:
        return "fail", 1.0, {}, "No main room (living room or bedroom) was extracted"
    best = min(rooms, key=lambda room: room["severity"])
    passed = best["severity"] <= EPS
    return ("pass" if passed else "fail"), (0.0 if passed else best["severity"]), best, "Evaluated (area AND height) OR volume over each main room"


def minimum_surface(context, spec):
    limits, occupants = spec.thresholds, context.scenario.intended_occupants
    threshold = limits["first_four_m2"] * min(occupants, 4) + limits["additional_m2"] * max(occupants - 4, 0)
    area = context.area(limits["included_room_labels"])
    severity = deficit(area, threshold)
    return ("pass" if severity <= EPS else "fail"), (0.0 if severity <= EPS else severity), {
        "estimated_habitable_area_m2": area, "required_area_m2": threshold, "occupants": occupants,
        "included_room_labels": limits["included_room_labels"], "excluded_room_labels": limits["excluded_room_labels"],
    }, "Estimated polygon-union area; wall thickness and headroom exclusions are unavailable"


def minimum_volume(context, spec):
    limits, occupants = spec.thresholds, context.scenario.intended_occupants
    threshold = limits["first_four_m3"] * min(occupants, 4) + limits["additional_m3"] * max(occupants - 4, 0)
    area = context.area(limits["included_room_labels"])
    volume = area * context.scenario.ceiling_height_m
    severity = deficit(volume, threshold)
    return ("pass" if severity <= EPS else "fail"), (0.0 if severity <= EPS else severity), {
        "estimated_habitable_area_m2": area, "height_m": context.scenario.ceiling_height_m,
        "estimated_habitable_volume_m3": volume, "required_volume_m3": threshold, "occupants": occupants,
    }, "Estimated polygon-union area times the scenario's uniform ceiling height"


CHECKERS = {
    "bedroom_turning": bedroom_turning, "bed_footprint": bed_footprint,
    "bedroom_circulation": bedroom_circulation, "bathroom_turning": bathroom_turning,
    "minimum_main_room": minimum_main_room, "minimum_surface": minimum_surface,
    "minimum_volume": minimum_volume,
}

REQUIRED_THRESHOLDS = {
    "bedroom_turning": {"diameter_m", "max_positions"},
    "bed_footprint": {"single_width_m", "multiple_width_m", "length_m", "bedroom_labels"},
    "bedroom_circulation": {"clearance_options_m"},
    "bathroom_turning": {"diameter_m", "bathroom_labels"},
    "minimum_main_room": {"area_m2", "height_m", "volume_m3", "main_room_labels"},
    "minimum_surface": {"first_four_m2", "additional_m2", "included_room_labels", "excluded_room_labels"},
    "minimum_volume": {"first_four_m3", "additional_m3", "included_room_labels", "excluded_room_labels"},
}


def load_specs(path=None):
    path = Path(path) if path else Path(__file__).parent / "configs" / "constraints_v1.json"
    data = json.loads(path.read_text())
    if not isinstance(data, dict) or not isinstance(data.get("constraints"), list):
        raise ValueError(f"{path}: expected an object with a constraints list")
    specs, seen = [], set()
    for entry in data["constraints"]:
        label = entry.get("id", "<missing id>") if isinstance(entry, dict) else "<invalid entry>"
        try:
            spec = ConstraintSpec(**entry)
        except TypeError as error:
            raise ValueError(f"{path}: invalid fields for {label}: {error}") from error
        if spec.id in seen:
            raise ValueError(f"Duplicate constraint id: {spec.id}")
        seen.add(spec.id)
        if spec.checker not in CHECKERS:
            raise ValueError(f"{spec.id}: unknown checker '{spec.checker}'")
        if spec.applicability not in APPLICABILITY:
            raise ValueError(f"{spec.id}: unknown applicability '{spec.applicability}'")
        if not isinstance(spec.weight, (int, float)) or isinstance(spec.weight, bool) or not math.isfinite(spec.weight) or spec.weight < 0:
            raise ValueError(f"{spec.id}: weight must be a finite nonnegative number")
        if not isinstance(spec.thresholds, dict):
            raise ValueError(f"{spec.id}: thresholds must be an object")
        missing = REQUIRED_THRESHOLDS[spec.checker] - spec.thresholds.keys()
        if missing:
            raise ValueError(f"{spec.id}: missing thresholds {sorted(missing)}")
        for name, value in spec.thresholds.items():
            if name.endswith("labels"):
                if not isinstance(value, list) or any(type(v) is not int or v not in range(7) for v in value):
                    raise ValueError(f"{spec.id}: {name} must list GSDiff labels 0..6")
            elif name == "clearance_options_m":
                if not isinstance(value, list) or not value or any(not isinstance(pair, list) or len(pair) != 2 or
                    any(not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0 for v in pair) for pair in value):
                    raise ValueError(f"{spec.id}: clearance_options_m must contain positive [side, foot] pairs")
            elif not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{spec.id}: threshold {name} must be a positive finite number")
        if "max_positions" in spec.thresholds and type(spec.thresholds["max_positions"]) is not int:
            raise ValueError(f"{spec.id}: max_positions must be an integer")
        if not isinstance(spec.required_observations, list) or not set(spec.required_observations) <= {"scale", "ceiling_height", "occupants", "room_polygons"}:
            raise ValueError(f"{spec.id}: unknown required observation")
        if not isinstance(spec.assumptions, list) or any(not isinstance(value, str) for value in spec.assumptions):
            raise ValueError(f"{spec.id}: assumptions must be a list of strings")
        if spec.severity_type not in {"normalized_deficit", "binary_feasibility"}:
            raise ValueError(f"{spec.id}: unknown severity_type '{spec.severity_type}'")
        if type(spec.enabled) is not bool or type(spec.counts_toward_compliance) is not bool:
            raise ValueError(f"{spec.id}: enabled/counts_toward_compliance must be boolean")
        specs.append(spec)
    if any(spec.checker in {"bedroom_turning", "bed_footprint", "bedroom_circulation"} for spec in specs):
        absent = {"C033", "C034", "C035"} - seen
        if absent:
            raise ValueError(f"Bedroom checks need joint parameter definitions: {sorted(absent)}")
    return specs


def check(sample, scenario, spec, context=None):
    if not APPLICABILITY[spec.applicability](scenario):
        return _result(spec, scenario, "not_applicable", explanation="The configured applicability predicate is false")
    observations = {"scale": scenario.coordinate_scale_m_per_model_unit,
                    "ceiling_height": scenario.ceiling_height_m, "occupants": scenario.intended_occupants}
    missing = [name for name in spec.required_observations if name in observations and observations[name] is None]
    if missing:
        return _result(spec, scenario, "unknown", explanation="Missing required observations: " + ", ".join(missing))
    if context is None:
        specs = {value.id: value for value in load_specs()}
        specs[spec.id] = spec
        context = CheckContext(sample, scenario, specs.values())
    if context.geometry_error:
        return _result(spec, scenario, "unknown", explanation="Unreliable extracted geometry: " + context.geometry_error)
    status, severity, observed, explanation = CHECKERS[spec.checker](context, spec)
    return _result(spec, scenario, status, severity, observed, explanation)
