"""Shared CPU/GPU interpretation of the supporting habitable-area definitions."""


def area_policy(specs, scenario):
    specs = {spec.id: spec for spec in specs}
    definitions = {key: specs[key] for key in ("C079", "C080") if key in specs}
    enabled = bool(definitions) or any(
        spec.enabled and {"C079", "C080"}.intersection(spec.depends_on) for spec in specs.values())
    result = {
        "enabled": enabled, "available": True, "reason": None, "missing_fields": [],
        "mode": scenario.habitable_area_mode,
        "included_room_labels": [0, 1, 3, 4], "excluded_room_labels": [2, 5, 6],
        "height_exclusion_m": 1.8, "excluded_by_height": False,
        "assumption_flags": [], "definitions": {},
    }
    if not enabled:
        return result  # v1 keeps its original per-check labels and height handling.
    result["definitions"] = {key: {"definition": spec.definition, "thresholds": spec.thresholds,
                                   "source_id": spec.source_id}
                             for key, spec in definitions.items()}
    if "C080" in definitions:
        thresholds = definitions["C080"].thresholds
        result.update(included_room_labels=thresholds["included_room_labels"],
                      excluded_room_labels=thresholds["excluded_room_labels"],
                      height_exclusion_m=thresholds["min_headroom_m"])
    missing = result["missing_fields"]
    if any(key not in definitions or not definitions[key].enabled for key in ("C079", "C080")):
        missing.append("supporting_area_definitions")
    if scenario.coordinate_scale_m_per_model_unit is None:
        missing.append("scale")
    if scenario.ceiling_height_m is None:
        missing.append("ceiling_height")
    if scenario.habitable_area_mode == "unknown":
        missing.append("habitable_area_mode")
    result["available"] = not missing
    result["reason"] = "Missing habitable-area inputs: " + ", ".join(missing) if missing else None
    result["excluded_by_height"] = (scenario.ceiling_height_m is not None and
                                     scenario.ceiling_height_m < result["height_exclusion_m"])
    mode_flag = "zero_thickness_habitable_area_proxy" if scenario.habitable_area_mode == "zero_thickness_proxy" else "unresolved_habitable_area_mode"
    result["assumption_flags"] = [mode_flag, "uniform_scenario_headroom",
                                  "unrepresented_area_exclusions_unverified"]
    if scenario.scale_provenance == "assumed":
        result["assumption_flags"].append("assumed_metric_scale")
    return result
