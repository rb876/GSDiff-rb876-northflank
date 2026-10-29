"""Human-readable descriptions of saved checks; never rerun their geometry."""

from collections import Counter
from dataclasses import asdict, is_dataclass
import math

from .evaluate import ROOM_NAMES


NAMES = {"C033": "Bedroom turning space", "C035": "Clearance around the bed",
         "C036": "Bathroom turning space", "C076": "Minimum main-room size",
         "C077": "Habitable floor area", "C078": "Habitable volume"}


def _dict(value):
    return asdict(value) if is_dataclass(value) else value or {}


def room_labels(sample):
    """Map saved room indices to the same type-relative names used on images."""
    counts, labels = Counter(), {}
    for index, room in enumerate(_dict(sample).get("rooms", [])):
        kind = _dict(room).get("label")
        counts[kind] += 1
        name = ROOM_NAMES.get(kind, "room").capitalize()
        labels[index] = f"{name} {counts[kind]}"
    return labels


def _numeric(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _number(value, decimals=2):
    return f"{value:.{decimals}f}".rstrip("0").rstrip(".") if _numeric(value) else "not recorded"


def _pair(first, second):
    # Do not display a failed near-bound comparison as e.g. 9.92 < 9.92.
    for precision in (2, 4, 6, 8, 10, 14):
        left, right = _number(first, precision), _number(second, precision)
        if first == second or left != right:
            return left, right
    return repr(first), repr(second)


def _dimensions(values, decimals=2):
    return " × ".join(_number(value, decimals) for value in values) + " m"


def _room(observed, labels):
    index = observed.get("room_index")
    return labels.get(index, f"Room {index + 1}" if isinstance(index, int) else "A bedroom")


def _requirement(identifier, limits, observed):
    diameter = limits.get("diameter_m")
    if identifier == "C033":
        circle = f"a {_number(diameter)} m diameter turning circle" if _numeric(diameter) else "the required turning circle"
        return f"At least one bedroom must fit a bed and {circle}, without overlap."
    if identifier == "C035":
        options = limits.get("clearance_options_m", [])
        if options:
            choices = " OR ".join(f"{_number(side)} m on both long sides + {_number(foot)} m at one short side"
                                  for side, foot in options)
            return f"Bed clearances: {choices}. Use the same bed placement as the turning-space check."
        return "Fit the required clearances around the same bed used for bedroom turning space."
    if identifier == "C036":
        size = f"{_number(diameter)} m diameter" if _numeric(diameter) else "required"
        return f"At least one empty bathroom must fit the {size} turning circle; fixtures and door swings are excluded."
    if identifier == "C076":
        if all(_numeric(limits.get(key)) for key in ("area_m2", "height_m", "volume_m3")):
            return (f"One living room or bedroom: (area ≥ {_number(limits['area_m2'])} m² AND "
                    f"height ≥ {_number(limits['height_m'])} m) OR volume ≥ {_number(limits['volume_m3'])} m³.")
        return "One living room or bedroom must meet both area and height limits, or the alternative volume limit."
    if identifier in {"C077", "C078"}:
        is_area = identifier == "C077"
        quantity, unit = ("area", "m²") if is_area else ("volume", "m³")
        required = observed.get(f"required_{quantity}_{'m2' if is_area else 'm3'}")
        occupants = observed.get("occupants")
        suffix = f" for {_number(occupants)} occupant{'s' if occupants != 1 else ''}" if _numeric(occupants) else ""
        if _numeric(required):
            return f"Estimated habitable {quantity} ≥ {_number(required)} {unit}{suffix}."
        first = limits.get("first_four_m2" if is_area else "first_four_m3")
        additional = limits.get("additional_m2" if is_area else "additional_m3")
        if _numeric(first) and _numeric(additional):
            return f"Habitable {quantity}: {_number(first)} {unit} per first four occupants; {_number(additional)} {unit} per additional occupant."
        return f"Meet the habitable {quantity} required for the stated number of occupants."
    return "See the saved constraint definition."


def _unknown(check):
    explanation = check.get("explanation", "")
    if check.get("missing_fields"):
        fields = {"scale": "drawing-to-metre scale", "ceiling_height": "ceiling height",
                  "occupants": "number of occupants", "habitable_area_mode": "habitable-area assumptions"}
        missing = ", ".join(fields.get(name, name.replace("_", " ")) for name in check["missing_fields"])
        return f"Cannot decide: missing {missing}."
    if "Unreliable extracted geometry" in explanation:
        details = explanation.split(":", 1)[-1].strip()
        return f"Cannot decide because room geometry is unreliable: {details}."
    if "shared bed" in explanation or "shared bed-and-turning" in explanation:
        return "Bed clearances fit, but the search did not find one bed position that also leaves the required turning circle. This is unresolved, not a proven failure."
    if "tolerance" in explanation:
        return "The circle-size bounds straddle the required size at the saved search tolerance; the result is unresolved."
    if check.get("id") in {"C033", "C035"} and ("search" in explanation.lower() or "placement" in explanation.lower()):
        return "The finite placement search found no verified arrangement. It did not prove that a suitable arrangement is impossible."
    return explanation or "The saved check could establish neither a pass nor a failure."


def _bed_failure(observed, labels, part, limits):
    rooms = observed.get("bedrooms", [])
    if not rooms:
        return None
    entries = [(room, room.get(part, {})) for room in rooms]
    areas = [(room, value) for room, value in entries
             if _numeric(value.get("room_area_m2")) and _numeric(value.get("necessary_area_m2"))]
    if len(areas) == len(rooms) and len({value["necessary_area_m2"] for _, value in areas}) == 1:
        largest = max(value["room_area_m2"] for _, value in areas)
        minimum = areas[0][1]["necessary_area_m2"]
        left, right = _pair(largest, minimum)
        sizes = "; ".join(f"{_room(room, labels)}: {_number(value['room_area_m2'])} m²" for room, value in areas)
        object_name = "bed + clearances" if part == "circulation" else "bed + separate turning circle"
        return f"{sizes}. Even the largest is {left} m² < {right} m² needed for {object_name}."
    reasons, envelopes = [], []
    for room, value in entries:
        name = _room(room, labels)
        if _numeric(value.get("room_area_m2")) and _numeric(value.get("necessary_area_m2")):
            area, bound = _pair(value["room_area_m2"], value["necessary_area_m2"])
            reasons.append(f"{name}: {area} m² < {bound} m² required")
        elif value.get("room_bbox_dimensions_m"):
            reasons.append(f"{name}: bounding size {_dimensions(value['room_bbox_dimensions_m'], 8)}, too narrow for the clearance envelope")
            for dimensions in value.get("required_envelope_dimensions_m", []):
                if not any(sorted(dimensions) == sorted(previous) for previous in envelopes):
                    envelopes.append(dimensions)
        elif _numeric(value.get("radius_upper_bound_m")):
            diameter = 2 * value["radius_upper_bound_m"]
            upper = _pair(diameter, limits["diameter_m"])[0] if _numeric(limits.get("diameter_m")) else _number(diameter)
            reasons.append(f"{name}: even the empty room has a turning-circle diameter upper bound of {upper} m")
        else:
            reasons.append(f"{name}: {value.get('reason', 'the recorded geometric bound is not met')}")
    if envelopes:
        reasons.append("Required envelope: " + " or ".join(_dimensions(size) for size in envelopes) + ", allowing either orientation")
    return "; ".join(reasons) + "."


def constraint_description(check, sample):
    """Describe a saved result using its own thresholds and measurements only."""
    check, sample = _dict(check), _dict(sample)
    identifier, status = check.get("id", ""), check.get("status", "unknown")
    observed, limits = check.get("observed_values") or {}, check.get("threshold_values") or {}
    labels = room_labels(sample)
    description = {"name": NAMES.get(identifier, identifier or "Constraint"),
                   "requirement": _requirement(identifier, limits, observed), "reason": ""}
    fallback = check.get("explanation") or "No further measurement details were saved."
    if check.get("role") == "supporting_rule" or not check.get("counts_toward_compliance", True):
        description["reason"] = "Supporting measurement definition; not an independent pass/fail check."
    elif status == "not_applicable":
        description["reason"] = "This requirement does not apply to the saved scenario."
    elif status == "unknown":
        description["reason"] = _unknown(check)
    elif identifier in {"C033", "C035"}:
        if status == "fail":
            description["reason"] = _bed_failure(observed, labels, "turning" if identifier == "C033" else "circulation", limits) or fallback
        elif observed.get("bed_polygon_m"):
            name = _room(observed, labels)
            dimensions = observed.get("bed_dimensions_m")
            bed = f"a {_dimensions(dimensions)} bed" if dimensions else "the required bed"
            if identifier == "C033":
                radius = observed.get("circle_radius_m")
                circle = f"a separate {_number(2 * radius)} m diameter turning circle" if _numeric(radius) else "a separate turning circle"
                description["reason"] = f"{name} fits {bed} and {circle}."
            elif observed.get("joint_bed_placement"):
                description["reason"] = f"{name} fits {bed}, its clearances and the turning circle in the same arrangement."
            else:
                description["reason"] = f"{name} has a verified bed-and-clearance arrangement."
        else:
            description["reason"] = fallback
    elif identifier == "C036":
        if status == "pass" and _numeric(observed.get("radius_lower_bound_m")):
            description["reason"] = f"A bathroom contains a circle at least {_number(2 * observed['radius_lower_bound_m'])} m in diameter."
        elif status == "fail" and observed.get("bathrooms"):
            bounds = [room["radius_upper_bound_m"] for room in observed["bathrooms"] if _numeric(room.get("radius_upper_bound_m"))]
            diameter = limits.get("diameter_m")
            if len(bounds) == len(observed["bathrooms"]) and _numeric(diameter):
                found, required = _pair(2 * max(bounds), diameter)
                description["reason"] = f"Even the best bathroom has a circle-diameter upper bound of {found} m < {required} m required."
        description["reason"] = description["reason"] or fallback
    elif identifier == "C076" and all(_numeric(observed.get(key)) for key in ("area_m2", "height_m", "volume_m3")):
        values = {key: _pair(observed[key], limits[key])[0] if _numeric(limits.get(key)) else _number(observed[key])
                  for key in ("area_m2", "height_m", "volume_m3")}
        measurements = (f"Main-room candidate: {values['area_m2']} m², height {values['height_m']} m, "
                        f"volume {values['volume_m3']} m³.")
        if status == "fail":
            conclusion = "Neither the area-and-height route nor the volume route meets the requirement."
        elif all(_numeric(limits.get(key)) for key in ("area_m2", "height_m")) and observed["area_m2"] >= limits["area_m2"] and observed["height_m"] >= limits["height_m"]:
            conclusion = "The area-and-height route passes."
        elif _numeric(limits.get("volume_m3")) and observed["volume_m3"] >= limits["volume_m3"]:
            conclusion = "The alternative volume route passes."
        else:
            conclusion = "The saved checker records a pass within its numerical tolerance."
        description["reason"] = measurements + " " + conclusion
    elif identifier in {"C077", "C078"}:
        quantity, unit, key = ("area", "m²", "m2") if identifier == "C077" else ("volume", "m³", "m3")
        value, required = observed.get(f"estimated_habitable_{quantity}_{key}"), observed.get(f"required_{quantity}_{key}")
        if _numeric(value) and _numeric(required):
            left, right = _pair(value, required)
            comparison = "below" if status == "fail" else "meets"
            description["reason"] = f"Estimated habitable {quantity}: {left} {unit}; {comparison} the {right} {unit} requirement."
        else:
            description["reason"] = fallback
    else:
        description["reason"] = fallback
    return description
