"""Two Best-of-N budgets remain separate in figures, galleries and compute pairs."""

from copy import deepcopy
import json
from pathlib import Path

import pytest

from test_time.analyze import analyze, load_records
from test_time.tests.test_analysis import saved_run
from test_time.tests.test_reporting import candidate, result
from test_time.types import read_json, write_json


def budget_runs(root, late_n=3, guided_steps=200):
    rows = [candidate(0, [result("C035", "pass"), result("C077", "pass")])]
    configurations = [
        ("svdd_pm_full", "svdd_pm", None, 999, 10990),
        ("best_of_n", "best_of_n", 11, None, 11000),
        ("baseline", "baseline", 1, None, 1000),
        ("best_of_n_late", "best_of_n", late_n, None, late_n * 1000),
        (f"svdd_pm_last{guided_steps // 10}", "svdd_pm", None, guided_steps, 1000 + 10 * guided_steps),
    ]
    for name, method, n, guided, nfe in configurations:
        directory = saved_run(root / name, deepcopy(rows), method=method)
        config = read_json(directory / "config.json")
        config.update(n=n, guided_steps=guided, k=10, reward_backend="gpu")
        if n == late_n:
            config["prefix_reselection"] = {"protocol": "best_of_n_prefix_v1", "source_n": 11,
                                             "n": late_n, "retained_candidate_indices": list(range(late_n)),
                                             "new_denoiser_evaluations": 0}
        write_json(directory / "config.json", config)
        run = read_json(directory / "run.json")
        run["node_samples"] = nfe
        if n == late_n:
            run.update(total_wall_seconds=None, node_calls=None, edge_calls=None, edge_samples=None)
        write_json(directory / "run.json", run)


@pytest.mark.parametrize("late_n,guided_steps", [(2, 100), (3, 200)])
def test_record_order_places_each_best_of_budget_by_its_svdd_comparison(tmp_path, late_n, guided_steps):
    budget_runs(tmp_path, late_n, guided_steps)
    records, skipped = load_records(tmp_path)
    assert not skipped
    assert [record["label"] for record in records] == [
        "baseline", "best_of_n_late", f"svdd_pm_last{guided_steps // 10}", "best_of_n", "svdd_pm_full"]
    assert [record["display_order"] for record in records] == list(range(5))


@pytest.mark.parametrize("n", [2, 3])
def test_best_of_plot_identity_comes_from_saved_n_not_folder_name(n):
    pytest.importorskip("matplotlib")
    from test_time.analysis_plots import METHOD_STYLES, _label, _method, _style

    small = {"label": "arbitrary_run", "config": {"method": "best_of_n", "n": n}}
    large = {"label": "best_of_n", "config": {"method": "best_of_n", "n": 11}}
    assert _method(small) == (f"best_of_n_{n}", f"Best-of-{n}")
    assert _method(large) == ("best_of_n_11", "Best-of-11")
    assert _style(small) != _style(large)
    assert _style(large) == METHOD_STYLES["best_of_n"]
    renamed = {**small, "label": "batch_2/renamed_control"}
    assert _style(renamed) == _style(small)
    assert _label(renamed) == f"Best-of-{n} · batch_2"


@pytest.mark.parametrize("late_n,guided_steps", [(2, 100), (3, 200)])
def test_analysis_keeps_both_budgets_and_exports_both_compute_matches(tmp_path, monkeypatch, late_n, guided_steps):
    pytest.importorskip("matplotlib")
    from test_time import analysis_plots

    source = tmp_path / "controls"
    budget_runs(source, late_n, guided_steps)
    late_method = f"svdd_pm_last{guided_steps // 10}"
    late_label = f"SVDD (last {guided_steps // 10}%)"
    original = {path: path.read_bytes() for path in source.rglob("*") if path.is_file()}
    captured = {}
    finish = analysis_plots._finish

    def capture(fig, output, name):
        if name == "compute":
            captured["legend"] = [text.get_text() for legend in fig.legends for text in legend.texts]
        return finish(fig, output, name)

    monkeypatch.setattr(analysis_plots, "_finish", capture)
    summary = analyze(source)
    output = Path(summary["output_dir"])
    assert summary["methods"] == summary["selected_outputs"] == 5
    assert captured["legend"] == ["Baseline", f"Best-of-{late_n}", late_label, "Best-of-11", "SVDD (full)"]
    comparisons = read_json(output / "denoiser_comparison.json")["comparisons"]
    matched = {(item["best_of_n"], item["svdd"]): item for item in comparisons if item["matched"]}
    assert set(matched) == {("best_of_n_late", late_method), ("best_of_n", "svdd_pm_full")}
    assert matched[("best_of_n_late", late_method)]["exact"]
    assert not matched[("best_of_n", "svdd_pm_full")]["exact"]
    payload = read_json(output / "analysis.json")
    small = next(record for record in payload["methods"] if record["config"].get("n") == late_n)
    assert small["metrics"]["node_samples_per_trial"] == late_n * 1000
    assert small["metrics"]["wall_seconds_per_trial"] is None
    paths = [item["valid"]["png"] for item in payload["gallery"]["examples"].values()]
    assert len(set(paths)) == 5 and all((output / path).is_file() for path in paths)
    gallery = (output / "index.html").read_text()
    assert f'<option value="best_of_n_late">Best-of-{late_n}</option>' in gallery
    assert '<option value="best_of_n">Best-of-11</option>' in gallery
    assert gallery.index(f'<h2>Best-of-{late_n}') < gallery.index('<h2>Best-of-11')
    report = (output / "REPORT.md").read_text()
    assert "first N candidates" in report and "equivalent generation budget" in report
    assert "evaluated denoiser sample states per selected output" in report
    assert "Best-of-3 targets late" not in report
    assert {path: path.read_bytes() for path in original} == original


def test_late_100_step_svdd_keeps_late_style_when_renamed():
    pytest.importorskip("matplotlib")
    from test_time.analysis_plots import METHOD_STYLES, _method, _style

    record = {"label": "arbitrary_run", "config": {"method": "svdd_pm", "guided_steps": 100}}
    assert _method(record) == ("svdd_pm_last10", "SVDD (last 10%)")
    assert _style(record) == METHOD_STYLES["svdd_pm_last20"]
    assert _style(record) != METHOD_STYLES["svdd_pm_full"]
