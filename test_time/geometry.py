"""Deterministic geometric witnesses; a missed placement is not an infeasibility proof."""

import math
import numpy as np
import shapely
from shapely.geometry import Point, Polygon, box
from shapely.ops import polylabel

EPS = 1e-8


def polygon_from_points(points):
    xy = np.asarray(points, dtype=float)
    if xy.ndim != 2 or xy.shape[0] < 3 or xy.shape[1] < 2 or not np.isfinite(xy).all():
        raise ValueError("A room needs at least three finite XY points")
    polygon = Polygon(xy[:, :2])
    if not polygon.is_valid or polygon.area <= EPS:
        raise ValueError("Room polygon is invalid or has zero area")
    return polygon


def circle_fit(polygon, radius, tolerance=0.005):
    """Find a certified circle witness, or an inradius bound from polylabel."""
    if not polygon.is_valid:
        return {"status": "unknown", "reason": "invalid polygon"}
    pieces = list(polygon.geoms) if polygon.geom_type == "MultiPolygon" else [polygon]
    best_radius, center = 0.0, None
    for piece in pieces:
        if piece.is_empty or piece.area <= EPS:
            continue
        point = polylabel(piece, tolerance=tolerance)
        distance = point.distance(piece.boundary) if piece.covers(point) else 0.0
        if distance > best_radius:
            best_radius, center = distance, [point.x, point.y]
    if best_radius + EPS >= radius:
        status, reason = "pass", "A contained circle centre is witnessed"
    elif best_radius + tolerance + EPS < radius:
        status, reason = "fail", "The inradius upper bound is below the required radius"
    else:
        status, reason = "unknown", "Circle fit lies within numerical search tolerance"
    return {"status": status, "reason": reason, "center_m": center,
            "radius_lower_bound_m": best_radius, "radius_upper_bound_m": best_radius + tolerance,
            "tolerance_m": tolerance}


def _axis(low, high, step, extra=()):
    if high + EPS < low:
        return np.array([])
    # Boundary and central placements are tried before the regular grid.
    values = [low, high, (low + high) / 2, *extra, *np.arange(low, high + EPS, step)]
    return np.array(list(dict.fromkeys(round(float(v), 9) for v in values
                                    if low - EPS <= v <= high + EPS)))


def _circle_centers(polygon, radius, step, best):
    x0, y0, x1, y1 = polygon.bounds
    xs = _axis(x0 + radius, x1 - radius, step)
    ys = _axis(y0 + radius, y1 - radius, step)
    xx, yy = np.meshgrid(xs, ys)
    xy = np.column_stack((xx.ravel(), yy.ravel()))
    if best.get("center_m"):
        xy = np.vstack((best["center_m"], xy))
    if len(xy) == 0:
        return xy
    points = shapely.points(xy)
    fits = shapely.covers(polygon, points) & (shapely.distance(points, polygon.boundary) + EPS >= radius)
    return xy[fits]


def _free_circle(centers, bounds, radius):
    if len(centers) == 0:
        return None
    x0, y0, x1, y1 = bounds
    dx = np.maximum(np.maximum(x0 - centers[:, 0], centers[:, 0] - x1), 0)
    dy = np.maximum(np.maximum(y0 - centers[:, 1], centers[:, 1] - y1), 0)
    indices = np.flatnonzero(dx * dx + dy * dy + EPS >= radius * radius)
    return centers[indices[0]].tolist() if len(indices) else None


def bedroom_search(polygon, bed_width, bed_length, radius, clearances,
                   grid=0.25, refine=0.05, max_positions=20000):
    """Search axis-aligned bed translations and both head directions.

    Circle and circulation witnesses share an identical bed when ``joint`` passes.
    Area bounds can prove failure; exhausting the finite search only means unknown.
    """
    statuses = {name: {"status": "unknown", "reason": "No placement witnessed in finite search"}
                for name in ("bed", "turning", "circulation", "joint")}
    statuses["search"] = {"orientations_deg": [0, 90, 180, 270],
                          "grid_m": grid, "refine_m": refine, "max_positions": max_positions,
                          "positions_examined": 0}
    area = polygon.area
    bed_area = bed_width * bed_length
    clearance_area = min((bed_width + 2 * side) * (bed_length + foot) for side, foot in clearances)
    bounds = {"bed": bed_area, "turning": bed_area + math.pi * radius**2,
              "circulation": clearance_area, "joint": max(clearance_area, bed_area + math.pi * radius**2)}
    for name, minimum in bounds.items():
        if area + EPS < minimum:
            statuses[name] = {"status": "fail", "reason": "Room area is below a necessary placement bound",
                              "room_area_m2": area, "necessary_area_m2": minimum}
    room_circle = circle_fit(polygon, radius, tolerance=min(refine / 4, 0.005))
    if room_circle["status"] == "fail":
        for name in ("turning", "joint"):
            statuses[name] = {**room_circle, "reason": "Even the empty room cannot contain the turning circle"}
    if statuses["bed"]["status"] == "fail":
        return statuses
    x0, y0, x1, y1 = polygon.bounds
    vertices = np.asarray(polygon.exterior.coords)
    seen = set()
    for step in dict.fromkeys((grid, refine)):
        centers = _circle_centers(polygon, radius, step, room_circle)
        for horizontal in (False, True):
            width, length = (bed_length, bed_width) if horizontal else (bed_width, bed_length)
            xs = _axis(x0, x1 - width, step, (*vertices[:, 0], *(vertices[:, 0] - width)))
            ys = _axis(y0, y1 - length, step, (*vertices[:, 1], *(vertices[:, 1] - length)))
            for x in xs:
                for y in ys:
                    key = (horizontal, round(x, 8), round(y, 8))
                    if key in seen:
                        continue
                    seen.add(key)
                    statuses["search"]["positions_examined"] += 1
                    if statuses["search"]["positions_examined"] > max_positions:
                        statuses["search"]["budget_exhausted"] = True
                        return statuses
                    bed = box(x, y, x + width, y + length)
                    if not polygon.covers(bed):
                        continue
                    witness = {"status": "pass", "reason": "An explicit contained placement is witnessed",
                               "bed_polygon_m": [list(p) for p in bed.exterior.coords],
                               "bed_dimensions_m": [bed_width, bed_length], "search_resolution_m": step}
                    if statuses["bed"]["status"] != "pass":
                        statuses["bed"] = witness
                    circle = _free_circle(centers, bed.bounds, radius)
                    if circle is not None and statuses["turning"]["status"] != "pass":
                        statuses["turning"] = {**witness, "circle_center_m": circle, "circle_radius_m": radius}
                    if statuses["circulation"]["status"] == "fail":
                        if statuses["turning"]["status"] in {"pass", "fail"}:
                            return statuses
                        continue
                    for side, foot in clearances:
                        for direction in (1, -1):
                            if horizontal:
                                envelope = box(x - (foot if direction < 0 else 0), y - side,
                                               x + width + (foot if direction > 0 else 0), y + length + side)
                            else:
                                envelope = box(x - side, y - (foot if direction < 0 else 0),
                                               x + width + side, y + length + (foot if direction > 0 else 0))
                            if not polygon.covers(envelope):
                                continue
                            placement = {**witness, "clearance_polygon_m": [list(p) for p in envelope.exterior.coords],
                                         "long_side_clearance_m": side, "free_short_side_clearance_m": foot,
                                         "orientation_deg": (90 if horizontal else 0) + (180 if direction < 0 else 0)}
                            if statuses["circulation"]["status"] != "pass":
                                statuses["circulation"] = placement
                            if circle is not None:
                                joint = {**placement, "circle_center_m": circle, "circle_radius_m": radius,
                                         "joint_bed_placement": True}
                                statuses.update(turning=joint, circulation=joint, joint=joint)
                                return statuses
            if all(statuses[name]["status"] != "unknown" for name in ("bed", "turning", "circulation", "joint")):
                return statuses
    return statuses
