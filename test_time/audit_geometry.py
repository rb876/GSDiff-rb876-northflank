"""Opt-in CPU placement refinement for saved outputs, never used by sampling.

Keep the original metric scale, axis-aligned bed, rectangular clearance envelope
and shared-bed requirement. Extra search can establish witnesses; a search miss
is still unknown. Original geometric failure bounds remain unchanged; an
axis-aligned bounding-box bound can additionally certify an unresolved failure.
"""

import numpy as np
from shapely.geometry import box

from .geometry import EPS, _axis, _circle_centers, _free_circle, bedroom_search, circle_fit


AUDIT_GEOMETRY_VERSION = "cpu_placement_audit_v2"
AUDIT_CONTAINMENT_TOLERANCE_M = EPS


def refined_bedroom_search(polygon, bed_width, bed_length, radius, clearances,
                          grid=0.25, refine=0.05, max_positions=20000):
    """Extend the reference search with bounded contact and free-space checks.

    Try at most ``max_positions`` extra bed positions, split between the two
    axis-aligned orientations, and 64 free-space circle fits per bedroom. Retain
    at most 4096 fine-grid circle witnesses. None of these limits certify failure.
    Containment allows only the existing geometry epsilon (1e-8 metres), avoiding
    false rejection of boundary contacts rounded by the legacy grid helper.
    """
    result = bedroom_search(polygon, bed_width, bed_length, radius, clearances,
                            grid, refine, max_positions)
    search = result["search"]
    search.update(audit_version=AUDIT_GEOMETRY_VERSION, audit_positions_examined=0,
                  audit_max_positions=max_positions, audit_circle_fit_calls=0,
                  audit_max_circle_fit_calls=64, audit_budget_exhausted=False,
                  audit_circle_fit_budget_exhausted=False,
                  audit_containment_tolerance_m=AUDIT_CONTAINMENT_TOLERANCE_M)
    x0, y0, x1, y1 = polygon.bounds
    envelope_sizes = [(bed_width + 2 * side, bed_length + foot) for side, foot in clearances]
    orientations = [size for width, height in envelope_sizes for size in ((width, height), (height, width))]
    bbox_fits = any(x1 - x0 + 2 * AUDIT_CONTAINMENT_TOLERANCE_M >= width
                   and y1 - y0 + 2 * AUDIT_CONTAINMENT_TOLERANCE_M >= height
                   for width, height in orientations)
    if not bbox_fits:
        proof = {"status": "fail", "reason": "Room bounding box cannot contain any axis-aligned clearance envelope",
                 "room_bbox_dimensions_m": [x1 - x0, y1 - y0],
                 "required_envelope_dimensions_m": [list(size) for size in orientations],
                 "containment_tolerance_m": AUDIT_CONTAINMENT_TOLERANCE_M}
        for part in ("circulation", "joint"):
            if result[part]["status"] == "unknown":
                result[part] = dict(proof)
    if (result["joint"]["status"] == "pass" or result["bed"]["status"] == "fail"
            or (result["joint"]["status"] == "fail" and result["turning"]["status"] != "unknown")
            or (result["circulation"]["status"] == "fail"
                and result["turning"]["status"] != "unknown")):
        return result

    tolerance = min(refine / 4, 0.005)
    room_circle = circle_fit(polygon, radius, tolerance=tolerance)
    # The reference search skips previously seen beds at its refinement phase.
    # Here every bed is tested against the fine circle witnesses from the outset.
    centers = _circle_centers(polygon, radius, refine, room_circle)
    if len(centers) > 4096:
        centers = centers[np.linspace(0, len(centers) - 1, 4096, dtype=int)]
    search["audit_circle_witnesses"] = len(centers)
    # The rounded grid can move a contact by <1 nm. Round-join buffering is
    # contained in the EPS neighbourhood and does not change meaningful widths.
    containment_polygon = polygon.buffer(AUDIT_CONTAINMENT_TOLERANCE_M)
    vertices = np.asarray(polygon.exterior.coords)
    circle_cache = {}
    for horizontal in (False, True):
        width, length = (bed_length, bed_width) if horizontal else (bed_width, bed_length)
        # Align both bed and clearance-envelope edges with polygon vertices.
        x_extra = [*vertices[:, 0], *(vertices[:, 0] - width)]
        y_extra = [*vertices[:, 1], *(vertices[:, 1] - length)]
        for side, foot in clearances:
            x_margins, y_margins = ((0, foot), (side,)) if horizontal else ((side,), (0, foot))
            for margin in x_margins:
                x_extra.extend(vertices[:, 0] + margin)
                x_extra.extend(vertices[:, 0] - width - margin)
            for margin in y_margins:
                y_extra.extend(vertices[:, 1] + margin)
                y_extra.extend(vertices[:, 1] - length - margin)
        xs = _axis(x0, x1 - width, refine, x_extra)
        ys = _axis(y0, y1 - length, refine, y_extra)
        budget = max_positions // 2 + (max_positions % 2 if horizontal else 0)
        examined = 0
        for x in xs:
            for y in ys:
                if examined >= budget:
                    search["audit_budget_exhausted"] = True
                    break
                examined += 1
                search["audit_positions_examined"] += 1
                bed = box(x, y, x + width, y + length)
                if not containment_polygon.covers(bed):
                    continue
                witness = {"status": "pass", "reason": "An explicit contained placement is witnessed",
                           "bed_polygon_m": [list(point) for point in bed.exterior.coords],
                           "bed_dimensions_m": [bed_width, bed_length], "search_resolution_m": refine,
                           "audit_version": AUDIT_GEOMETRY_VERSION,
                           "containment_tolerance_m": AUDIT_CONTAINMENT_TOLERANCE_M}
                if result["bed"]["status"] == "unknown":
                    result["bed"] = witness
                placement = None
                if result["circulation"]["status"] != "fail":
                    for side, foot in clearances:
                        for direction in (1, -1):
                            if horizontal:
                                envelope = box(x - (foot if direction < 0 else 0), y - side,
                                               x + width + (foot if direction > 0 else 0), y + length + side)
                            else:
                                envelope = box(x - side, y - (foot if direction < 0 else 0),
                                               x + width + side, y + length + (foot if direction > 0 else 0))
                            if containment_polygon.covers(envelope):
                                placement = {**witness,
                                             "clearance_polygon_m": [list(point) for point in envelope.exterior.coords],
                                             "long_side_clearance_m": side, "free_short_side_clearance_m": foot,
                                             "orientation_deg": (90 if horizontal else 0) + (180 if direction < 0 else 0)}
                                break
                        if placement is not None:
                            break
                if placement is not None and result["circulation"]["status"] == "unknown":
                    result["circulation"] = placement
                circle = _free_circle(centers, bed.bounds, radius)
                circle_method = "fine_grid"
                needs_circle = (placement is not None or result["circulation"]["status"] == "fail")
                if circle is None and needs_circle and result["turning"]["status"] != "fail":
                    key = (horizontal, float(x), float(y))
                    if key not in circle_cache and search["audit_circle_fit_calls"] < 64:
                        # Search the actual remaining room, including any hole left
                        # by the bed. A local fail says nothing about other beds.
                        circle_cache[key] = circle_fit(polygon.difference(bed), radius, tolerance=tolerance)
                        search["audit_circle_fit_calls"] += 1
                    if key in circle_cache:
                        fitted = circle_cache[key]
                        if fitted["status"] == "pass":
                            circle = fitted["center_m"]
                            circle_method = "free_space_polylabel"
                    else:
                        search["audit_circle_fit_budget_exhausted"] = True
                if circle is not None:
                    turning = {**witness, "circle_center_m": circle, "circle_radius_m": radius,
                               "circle_search_method": circle_method}
                    if result["turning"]["status"] == "unknown":
                        result["turning"] = turning
                    if placement is not None and result["joint"]["status"] != "fail":
                        joint = {**placement, "circle_center_m": circle, "circle_radius_m": radius,
                                 "circle_search_method": circle_method, "joint_bed_placement": True}
                        result.update(turning=joint, circulation=joint, joint=joint)
                        return result
                if result["circulation"]["status"] == "fail" and result["turning"]["status"] != "unknown":
                    return result
            if examined >= budget:
                if examined < len(xs) * len(ys):
                    search["audit_budget_exhausted"] = True
                break
    return result
