"""Gallery exports saved selected outputs without inference or source mutations."""

from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path

from PIL import Image

from test_time.analysis_gallery import build_gallery
from test_time.types import ConstraintResult, FloorplanSample, PolicyEvaluation, Room


def row(trial, status="pass", *, selected=True, structural=True):
    sample = FloorplanSample(nodes=[[0, 0], [4, 0], [4, 3], [0, 3]],
                             edges=[[0, 1], [1, 2], [2, 3], [3, 0]],
                             rooms=[Room(0, [[0, 0], [4, 0], [4, 3], [0, 3], [0, 0]])],
                             metadata={"trial_seed": 10000 + trial})
    check = ConstraintResult("C035", status, 1.0 if status == "fail" else 0.0 if status == "pass" else None,
                             True, status in {"pass", "fail"}, "assessment",
                             observed_values={"clearance_m": 0.7}, threshold_values={"clearance_m": 0.9},
                             explanation='Recorded witness <not HTML> & "quoted".')
    support = ConstraintResult("C034", "unknown", None, True, False, "assessment",
                               counts_toward_compliance=False, role="supporting_rule")
    evaluation = PolicyEvaluation(1.0, -1.0, [check, support], structural,
                                  int(status == "unknown"), 1, 0, float(not structural),
                                  diagnostics=[] if structural else ["No closed room polygons"],
                                  measurable_count=int(status in {"pass", "fail"}))
    return {"trial": trial, "seed": 987654 + trial, "selected": selected,
            "candidate_id": f"winner-{trial}", "sample": asdict(sample), "evaluation": asdict(evaluation)}


def record(label, rows, run_dir):
    return {"label": label, "rows": rows, "run_dir": run_dir, "summary": {"trial_seed_start": 10000}}


def test_selected_images_graphs_and_evaluations_are_portable_and_sources_unchanged(tmp_path):
    rows = [row(3, "unknown"), row(2, "fail"), row(1), row(0, selected=False)]
    original = deepcopy(rows)
    source = tmp_path / "original"
    source.mkdir()
    (source / "sentinel.txt").write_text("unchanged")
    output = tmp_path / "analysis"
    result = build_gallery([record("Best-of-11", rows, source)], output)
    assert result["sample_count"] == 3
    examples = result["examples"]["Best-of-11"]
    assert {status: item["trial"] for status, item in examples.items()} == {"valid": 1, "invalid": 2, "unknown": 3}
    for status, item in examples.items():
        for kind in ("png", "svg", "graph", "evaluation"):
            assert not Path(item[kind]).is_absolute()
            assert (output / item[kind]).is_file()
        with Image.open(output / item["png"]) as image:
            assert image.size[0] == 1120
        original_row = next(candidate for candidate in rows if candidate["trial"] == item["trial"])
        assert json.loads((output / item["graph"]).read_text()) == original_row["sample"]
        assert json.loads((output / item["evaluation"]).read_text()) == original_row["evaluation"]
        assert item["seed"] == 10000 + item["trial"]
        assert item["candidate_seed"] == 987654 + item["trial"]
    assert examples["invalid"]["failed_ids"] == ["C035"]
    assert examples["unknown"]["unknown_ids"] == ["C035"]
    assert examples["valid"]["unknown_ids"] == []
    assert rows == original
    assert [item.name for item in source.iterdir()] == ["sentinel.txt"]
    html = Path(result["html"]).read_text()
    assert 'href="samples/' in html and 'src="samples/' in html
    assert 'value="C035"' in html and 'value="C034"' not in html
    assert 'Recorded witness &lt;not HTML&gt; &amp; &quot;quoted&quot;.' in html


def test_invalid_graph_and_unicode_labels_render_with_safe_unique_paths(tmp_path):
    invalid = row(0, structural=False)
    invalid["sample"].update(nodes=[], edges=[], rooms=[])
    invalid["sample"]["metadata"] = {}
    labels = ['SVDD été </script><script>alert("bad")</script>', 'SVDD été /..']
    records = [record(label, [deepcopy(invalid)], tmp_path / str(index)) for index, label in enumerate(labels)]
    output = tmp_path / "analysis"
    result = build_gallery(records, output)
    assert result["sample_count"] == 2
    paths = [result["examples"][label]["invalid"]["png"] for label in labels]
    assert len(set(paths)) == 2
    assert all((output / path).is_relative_to(output) for path in paths)
    assert all(result["examples"][label]["invalid"]["seed"] == 10000 for label in labels)
    assert all(result["examples"][label]["valid"] is None for label in labels)
    html = Path(result["html"]).read_text()
    assert '</script><script>alert("bad")</script>' not in html
    assert 'été &lt;/script&gt;' in html
    assert 'No closed room polygons' in html
    assert html.count('<script>') == 1


def test_missing_trial_seed_is_not_replaced_by_best_of_n_winner_seed(tmp_path):
    candidate = row(0)
    candidate["sample"]["metadata"] = {}
    current = record("Best-of-N", [candidate], tmp_path)
    current["summary"] = {}
    result = build_gallery([current], tmp_path / "analysis")
    example = result["examples"]["Best-of-N"]["valid"]
    assert example["seed"] is None and example["candidate_seed"] == candidate["seed"]
    assert 'Trial seed: <strong>unrecorded</strong>' in Path(result["html"]).read_text()
