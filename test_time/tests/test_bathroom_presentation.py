"""Saved joint bathroom results remain readable and visible without rescoring."""

from copy import deepcopy
from html import unescape
from pathlib import Path
import xml.etree.ElementTree as ET

from PIL import Image, ImageFont
import pytest

from test_time.constraint_text import constraint_description
from test_time.visualize import save_sample


def saved_bathroom(status="pass"):
    witness = {
        "wc_polygon_m": [[0, 0], [.4, 0], [.4, .7], [0, .7]],
        "basin_polygon_m": [[0, 2], [.5, 2], [.5, 2.6], [0, 2.6]],
        "transfer_polygon_m": [[.4, 0], [1.2, 0], [1.2, 1.3], [.4, 1.3]],
        "door_swing_polygon_m": [[2.2, 3], [3, 3], [2.2, 2.2]],
        "circle_center_m": [1.5, 1.5], "circle_radius_m": .75,
        "door_overlap_m": 0, "basin_overlap_m": 0,
    }
    observed = {"completion_model": "combined_wc_basin_v1", "room_index": 0,
                "witness": witness, "guidance_penalty": 0 if status == "pass" else .15}
    sample = {"nodes": [[0, 0], [300, 0], [300, 300], [0, 300]],
              "edges": [[0, 1], [1, 2], [2, 3], [3, 0]],
              "rooms": [{"label": 4, "polygon": [[0, 0], [300, 0], [300, 300], [0, 300]]}],
              "scenario": {"coordinate_scale_m_per_model_unit": .01, "scale_provenance": "assumed"}}
    limits = {"diameter_m": 1.5, "transfer_width_m": .8, "transfer_length_m": 1.3,
              "door_overlap_m": .25, "basin_overlap_m": .15}
    checks = [{"id": identifier, "status": status, "severity": 0 if status == "pass" else None,
               "applicable": True, "measurable": status in {"pass", "fail"},
               "observed_values": deepcopy(observed), "threshold_values": limits,
               "explanation": "A shared bathroom arrangement was found" if status == "pass" else "Finite search exhausted"}
              for identifier in ("C036", "C037", "C041")]
    return sample, {"structurally_valid": True, "energy": 0, "results": checks}


def test_new_check_descriptions_share_room_and_explain_dimensions_and_overlaps():
    sample, evaluation = saved_bathroom()
    descriptions = {check["id"]: constraint_description(check, sample) for check in evaluation["results"]}
    assert "same fixture and door layout" in descriptions["C036"]["requirement"]
    assert "empty bathroom" not in descriptions["C036"]["requirement"]
    assert "0.8 × 1.3 m" in descriptions["C037"]["requirement"]
    assert "outside the door swing" in descriptions["C037"]["reason"]
    assert "≤ 0.25 m OR" in descriptions["C041"]["requirement"]
    assert "≤ 0.15 m" in descriptions["C041"]["requirement"]
    assert "door overlap 0 m; under-basin overlap 0 m" in descriptions["C041"]["reason"]
    assert all("Bathroom 1" in value["reason"] for value in descriptions.values())


def test_search_miss_is_unresolved_and_has_no_invented_failure():
    sample, evaluation = saved_bathroom("unknown")
    for check in evaluation["results"]:
        reason = constraint_description(check, sample)["reason"]
        assert "Other fixture or door placements may work" in reason
        assert "not a proven failure" in reason


def test_rejected_cpu_witness_is_explained_as_unresolved():
    sample, evaluation = saved_bathroom("unknown")
    check = evaluation["results"][2]
    check["observed_values"]["audit_rejected_ids"] = ["C041"]
    reason = constraint_description(check, sample)["reason"]
    assert "independent CPU geometry audit" in reason
    assert "unresolved" in reason


def test_bathroom_failure_explains_the_saved_bound_in_words():
    sample, evaluation = saved_bathroom("fail")
    check = evaluation["results"][1]
    check["observed_values"]["bounds"] = {"room_area_m2": 1.8,
        "minimum_transfer_fixtures_door_area_m2": 2.12265, "transfer_envelope_fits": True}
    reason = constraint_description(check, sample)["reason"]
    assert "Bathroom 1: 1.8 m² < 2.12 m²" in reason
    assert "WC, basin, transfer zone and door sweep" in reason
    check = evaluation["results"][0]
    check["observed_values"]["bathrooms"] = [{"room_index": 0,
        "necessary_bounds": {"circle_radius_upper_bound_m": .7}}]
    assert "cannot exceed 1.4 m, below 1.5 m required" in constraint_description(check, sample)["reason"]


@pytest.mark.parametrize("fallback_font", [False, True])
def test_bathroom_svg_and_png_display_all_objects_and_keep_saved_results_unchanged(tmp_path, monkeypatch, fallback_font):
    if fallback_font:
        bitmap_font = getattr(ImageFont, "load_default_imagefont", ImageFont.load_default)()

        def unavailable_system_font(*args, **kwargs):
            raise OSError("No system fonts installed")

        monkeypatch.setattr(ImageFont, "truetype", unavailable_system_font)
        monkeypatch.setattr(ImageFont, "load_default", lambda: bitmap_font)
    sample, evaluation = saved_bathroom()
    original = deepcopy((sample, evaluation))
    files = save_sample(sample, evaluation, tmp_path / "shared")
    svg = Path(files["svg"]).read_text()
    ET.fromstring(svg)
    assert "Verified bathroom arrangement" in svg
    assert "assumed fixtures" in svg
    assert "not GSDiff predictions" in svg
    for name in ("wc_polygon_m", "basin_polygon_m", "transfer_polygon_m", "door_swing_polygon_m", "turning_circle"):
        assert f'data-bathroom-object="{name}"' in svg
    assert "WC lateral transfer space" in svg and "Bathroom turning-space overlaps" in svg
    assert "circle_center_m" not in unescape(svg)  # Explain geometry rather than dumping JSON.
    with Image.open(files["png"]) as image:
        assert image.width == 1440 and image.height > 1100
        colors = {color for _, color in image.getcolors(image.width * image.height)}
        # Check the five bathroom drawing colours, independent of font antialiasing.
        assert {(55, 65, 81), (139, 90, 166), (22, 131, 77),
                (230, 126, 34), (39, 101, 176)} <= colors
    assert (sample, evaluation) == original


def test_unverified_candidate_is_never_labelled_a_verified_arrangement(tmp_path):
    sample, evaluation = saved_bathroom("unknown")
    files = save_sample(sample, evaluation, tmp_path / "unresolved")
    svg = Path(files["svg"]).read_text()
    assert "Unverified bathroom proposal" in svg
    assert "Verified bathroom arrangement" not in svg
    assert "overall: UNRESOLVED" in svg


def test_missing_metric_scale_does_not_draw_metre_objects_as_pixels(tmp_path):
    sample, evaluation = saved_bathroom()
    sample["scenario"] = {}
    files = save_sample(sample, evaluation, tmp_path / "missing_scale")
    svg = Path(files["svg"]).read_text()
    assert 'data-bathroom-object=' not in svg
    assert 'Metric scale unavailable' in svg


def test_bathroom_profile_includes_new_ids_in_plots_gallery_and_group_metrics(tmp_path, monkeypatch):
    import csv
    import json
    pytest.importorskip("matplotlib")
    from test_time import analysis_plots
    from test_time.analyze import analyze
    from test_time.tests.test_analysis import saved_run
    from test_time.tests.test_reporting import candidate, result
    from test_time.types import read_json, write_json

    configs = Path(__file__).parents[1] / "configs"
    registry = read_json(configs / "constraints_bathroom_v1.json")
    checks = [result(spec["id"], "pass" if spec["counts_toward_compliance"] else "not_applicable",
                     auxiliary=not spec["counts_toward_compliance"]) for spec in registry["constraints"]]
    directory = saved_run(tmp_path / "baseline", [candidate(0, checks)], scale=18 / 512)
    config = read_json(directory / "config.json")
    config.update(constraints=registry["constraints"], scenario=read_json(configs / "scenario_bathroom_v1.json"))
    write_json(directory / "config.json", config)
    captured = {}
    finish = analysis_plots._finish

    def capture(fig, output, name):
        if name == "constraint_outcomes":
            captured["ids"] = [tick.get_text() for tick in fig.axes[0].get_yticklabels()]
        elif name == "passed_constraints":
            captured["counts"] = fig.axes[0].get_xticks().tolist()
        return finish(fig, output, name)

    monkeypatch.setattr(analysis_plots, "_finish", capture)
    output = Path(analyze(tmp_path)["output_dir"])
    assert len(captured["ids"]) == 11 and captured["counts"][-1] == 11
    assert all(identifier in captured["ids"] for identifier in ("C036", "C037", "C041"))
    html = (output / "index.html").read_text()
    assert 'value="C037"' in html and 'value="C041"' in html
    with (output / "tables/groups.csv").open() as stream:
        groups = list(csv.DictReader(stream))
    bath = next(group for group in groups if group["group"] == "bathroom_fixtures_turning")
    assert json.loads(bath["active_scored_ids"]) == ["C036", "C037", "C041"]
