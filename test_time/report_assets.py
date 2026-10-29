"""Regenerate the compact bathroom report figures from saved CPU evaluations.

Run ``python3 -m test_time.report_assets --help``. No inference, geometry checks,
reselections, or edits to the experiment are performed.
"""

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Patch, Polygon as PolygonPatch
import matplotlib.patheffects as pe
from matplotlib.ticker import PercentFormatter
import numpy as np
from shapely.geometry import Polygon
from shapely.ops import polylabel

from .analysis_plots import PAPER_STYLE, _bar_style, _style
from .analyze import load_records
from .constraint_text import room_labels
from .evaluate import counted_results
from .types import overall_validity, write_json
from .visualize import COLORS, save_sample


METHODS = ["baseline", "best_of_n_2", "svdd_pm_last10", "best_of_n_11"]
LABELS = {"baseline": "Baseline", "best_of_n_2": "Best-of-2",
          "svdd_pm_last10": "SVDD (last 10%)", "best_of_n_11": "Best-of-11"}
CONSTRAINTS = {
    "C033": "Bedroom turning space", "C035": "Bed circulation",
    "C036": "Bathroom turning space", "C037": "WC lateral transfer",
    "C041": "Bathroom overlap limits", "C076": "Main-room area / volume",
    "C077": "Habitable area", "C078": "Habitable volume",
    "C085": "Kitchen extraction", "C086": "Bathroom extraction",
    "C090": "Reduced-mode extraction",
}
VENTILATION = {"C085", "C086", "C090"}
BATHROOM = {"C036", "C037", "C041"}
PARTITIONS = {
    "valid": ("Overall valid", ("#579b50", "")),
    "ventilation_failure_only": ("Ventilation failure only", ("#c27543", "////")),
    "ventilation_unknown_only": ("Ventilation unresolved only", ("#b89131", "....")),
    "other": ("Other outcomes", ("#8b8683", "xxxx")),
}
SELECTION = ("First trial, in ascending trial order, where the selected last-10% SVDD "
             "output is overall valid and its same-trial, same-seed baseline is overall invalid; "
             "show that baseline and SVDD pair, then the first overall-invalid SVDD output. "
             "This outcome-conditioned illustration is not a representative random sample or a repair trace.")
DEFINITIONS = {
    "denominator": "All 100 selected outputs per method, including invalid and unresolved outputs.",
    "constraint_success": "Applicable, measurable saved passes / all selected outputs; supporting rules excluded.",
    "valid": "Valid structure and every applicable scored check measurably passes, with at least one applicable check.",
    "ventilation_failure_only": "Valid structure, all eight non-ventilation checks measurably pass, and at least one ventilation check fails. This bin takes precedence over ventilation unknowns.",
    "ventilation_unknown_only": "Valid structure, all eight non-ventilation checks measurably pass, no ventilation check fails, and at least one applicable ventilation check is unknown or unmeasurable.",
    "other": "All remaining outputs, including structural failures and failed or unresolved non-ventilation checks.",
    "ventilation_checks": sorted(VENTILATION),
    "partition": "Mutually exclusive and exhaustive; every selected output belongs to exactly one bin.",
}


def _path(path):
    path = Path(path).resolve()
    try:
        return path.relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return str(path)


def _fingerprint(path):
    return {"path": _path(path), "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest()}


def _method(record):
    config = record["config"]
    if config["method"] == "baseline":
        return "baseline"
    if config["method"] == "best_of_n":
        return f"best_of_n_{config['n']}"
    if config["method"] == "svdd_pm" and config.get("guided_steps") == 100:
        return "svdd_pm_last10"
    return None


def _passed(check):
    return check["status"] == "pass" and check["applicable"] and check["measurable"]


def _partition(row):
    if overall_validity(row["evaluation"]) == "valid":
        return "valid"
    checks = counted_results(row)
    nonvent = [check for check in checks if check["id"] not in VENTILATION]
    vent = [check for check in checks if check["id"] in VENTILATION]
    if row["evaluation"]["structurally_valid"] and all(map(_passed, nonvent)):
        if any(check["applicable"] and check["status"] == "fail" for check in vent):
            return "ventilation_failure_only"
        if any(check["applicable"] and (check["status"] == "unknown" or not check["measurable"])
               for check in vent):
            return "ventilation_unknown_only"
    return "other"


def _csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _save(fig, output, name):
    files = []
    for extension in ("png", "pdf", "svg"):
        path = output / f"{name}.{extension}"
        fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.07, facecolor="white")
        if extension == "svg":
            path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
        files.append(path.name)
    plt.close(fig)
    return files


def _constraint_figure(records, counts, output):
    fig, axes = plt.subplots(3, 4, figsize=(11.5, 7.2), sharey=True)
    fig.subplots_adjust(left=0.06, right=0.99, top=0.94, bottom=0.11, hspace=0.52, wspace=0.22)
    positions = np.arange(len(records))
    for axis, identifier in zip(axes.flat, CONSTRAINTS):
        values = [next(row["pass_percent"] for row in counts
                       if row["method"] == _method(record) and row["constraint"] == identifier)
                  for record in records]
        for x, value, record in zip(positions, values, records):
            axis.bar(x, value, width=0.63, **_bar_style(_style(record)))
            axis.text(x, value + 2.0, f"{value:g}", ha="center", fontsize=9)
        axis.set_title(f"{identifier} · {CONSTRAINTS[identifier]}", fontsize=10, pad=9)
        axis.set(ylim=(0, 115), xticks=[], yticks=[0, 25, 50, 75, 100])
        axis.yaxis.set_major_formatter(PercentFormatter(100, decimals=0))
        axis.grid(axis="y")
    for axis in axes[:, 0]:
        axis.set_ylabel("Pass rate")
    notes = axes.flat[-1]
    notes.axis("off")
    notes.text(0.02, 0.80, "100 selected outputs per method\n11 scored checks\n\nAll outputs remain in each denominator.\nUnresolved checks do not count as passes.",
               transform=notes.transAxes, va="top", fontsize=10, linespacing=1.5)
    fig.legend(handles=[Patch(**_bar_style(_style(record)), label=LABELS[_method(record)])
                        for record in records], loc="lower center", bbox_to_anchor=(0.5, 0.015), ncol=4)
    return _save(fig, output, "constraint_success")


def _bottleneck_figure(partition, output):
    fig, ax = plt.subplots(figsize=(10.5, 3.45))
    fig.subplots_adjust(left=0.15, right=0.98, top=0.87, bottom=0.28)
    left = np.zeros(len(partition))
    for category, (label, style) in PARTITIONS.items():
        values = np.array([100 * row[category] / row["selected_outputs"] for row in partition])
        ax.barh(np.arange(len(partition)), values, left=left, height=0.56, **_bar_style(style), label=label)
        for i, value in enumerate(values):
            if value:
                ax.text(left[i] + value / 2, i, f"{value:g}", ha="center", va="center", fontsize=10)
        left += values
    ax.set(yticks=np.arange(len(partition)), yticklabels=[LABELS[row["method"]] for row in partition],
           xlim=(0, 100), xticks=[0, 20, 40, 60, 80, 100], xlabel="Selected outputs (%)")
    ax.invert_yaxis()
    ax.xaxis.set_major_formatter(PercentFormatter(100, decimals=0))
    ax.grid(axis="x")
    ax.set_title("Ventilation-only outcomes require valid structure and all eight non-ventilation checks to pass", fontsize=10)
    fig.legend(handles=[Patch(**_bar_style(style), label=label) for label, style in PARTITIONS.values()],
               loc="lower center", bbox_to_anchor=(0.5, 0.005), ncol=2)
    return _save(fig, output, "ventilation_bottleneck")


def _select_examples(records):
    baseline, svdd = records[0], records[2]
    paired = {row["trial"]: row for row in baseline["rows"]}
    good = next((row for row in svdd["rows"] if overall_validity(row["evaluation"]) == "valid"
                 and row["trial"] in paired and paired[row["trial"]]["seed"] == row["seed"]
                 and overall_validity(paired[row["trial"]]["evaluation"]) == "invalid"), None)
    bad = next((row for row in svdd["rows"] if overall_validity(row["evaluation"]) == "invalid"), None)
    if good is None or bad is None:
        raise ValueError("The deterministic selection requires a valid SVDD / invalid baseline pair and an invalid SVDD output")
    return [(baseline, paired[good["trial"]]), (svdd, good), (svdd, bad)]


def _bathroom(row):
    check = next(check for check in counted_results(row) if check["id"] == "C036")
    observed = check["observed_values"]
    if not observed.get("witness") or any(check["status"] != "pass" for check in counted_results(row)
                                           if check["id"] in BATHROOM):
        raise ValueError("The selected illustration requires a saved, passing shared bathroom witness")
    return observed


def _polygon(ax, coordinates, color, fill="none", linestyle="-", linewidth=1.0):
    ax.add_patch(PolygonPatch(coordinates, closed=True, facecolor=fill, edgecolor=color,
                             linestyle=linestyle, linewidth=linewidth))


def _floorplan(ax, row, limits):
    sample = row["sample"]
    scale = sample["scenario"]["coordinate_scale_m_per_model_unit"]
    labels = room_labels(sample)
    for i, room in enumerate(sample["rooms"]):
        points = np.array(room["polygon"])[:, :2] * scale
        _polygon(ax, points, "#666666", to_rgba(COLORS.get(room["label"], "#dddddd"), 0.55), linewidth=0.7)
        shape = Polygon(points)
        if not shape.is_valid or shape.is_empty:
            continue
        anchor = polylabel(shape, tolerance=0.03)
        name = labels[i].replace("Bedroom", "Bed").replace("Bathroom", "Bath").replace("Balcony", "Balc.").replace("Storage", "Store")
        if sum(other["label"] == room["label"] for other in sample["rooms"]) == 1:
            name = name.rsplit(" ", 1)[0]
        # Tiny storage closets receive an external callout; other saved labels
        # stay inside their room without displaying graph internals or wall IDs.
        if shape.area < 2.0:
            ax.annotate(f"{name}\n{shape.area:.1f} m²", (anchor.x, anchor.y),
                        xytext=(limits[1] - 0.1, anchor.y - 0.8), ha="right", va="center", fontsize=8,
                        arrowprops={"arrowstyle": "-", "color": "#777777", "lw": 0.65})
        else:
            ax.text(anchor.x, anchor.y, f"{name}\n{shape.area:.1f} m²", ha="center", va="center", fontsize=8.5,
                    path_effects=[pe.withStroke(linewidth=2.3, foreground="white", alpha=0.9)])
    ax.set(xlim=limits[:2], ylim=limits[2:][::-1], aspect="equal")
    ax.axis("off")
    x, y = limits[0] + 0.6, limits[3] - 0.4
    ax.plot([x, x + 2], [y, y], color="#444444", lw=1.3)
    ax.text(x + 1, y - 0.18, "2 m", ha="center", va="bottom", fontsize=8)


def _diagnostic(ax, row):
    checks = {check["id"]: check for check in counted_results(row)}
    kitchen, reduced = (checks[identifier]["observed_values"] for identifier in ("C085", "C090"))
    count = sum(map(_passed, checks.values()))
    ax.axis("off")
    ax.text(0.5, 0.98, f"{kitchen['main_room_count']} main rooms · {count}/11 checks pass", transform=ax.transAxes,
            ha="center", va="top", fontsize=10)
    for y, identifier, observed, required, label in (
            (0.66, "C085", kitchen["observed_flow_m3_h"], kitchen["required_flow_m3_h"], "kitchen"),
            (0.36, "C090", reduced["observed_total_flow_m3_h"], reduced["required_total_flow_m3_h"], "total reduced")):
        passed = _passed(checks[identifier])
        relation = "≥" if passed else "<"
        ax.text(0.5, y, f"{identifier}: {observed:g} {relation} {required:g} m³/h ({label})", transform=ax.transAxes,
                ha="center", va="top", fontsize=9, color="#386943" if passed else "#a54e3c")


def _bathroom_inset(ax, row, span):
    sample, observed = row["sample"], _bathroom(row)
    scale = sample["scenario"]["coordinate_scale_m_per_model_unit"]
    points = np.array(sample["rooms"][observed["room_index"]]["polygon"])[:, :2] * scale
    centre = (points.min(axis=0) + points.max(axis=0)) / 2
    _polygon(ax, points, "#667b7b", "#f2f6f5", linewidth=1.1)
    witness = observed["witness"]
    for key, color, fill, dash in (
            ("door_swing_polygon_m", "#d08335", "none", "--"),
            ("transfer_polygon_m", "#318255", "none", "--"),
            ("wc_polygon_m", "#4c586a", "#d9dee5", "-"),
            ("basin_polygon_m", "#8b5aa6", "#ece4f2", "-")):
        _polygon(ax, witness[key], color, fill, dash, linewidth=1.1)
    ax.add_patch(Circle(witness["circle_center_m"], witness["circle_radius_m"],
                        fill=False, color="#376d9f", linewidth=1.3, linestyle="--"))
    ax.set(xlim=(centre[0] - span / 2, centre[0] + span / 2),
           ylim=(centre[1] + span / 2, centre[1] - span / 2), aspect="equal")
    ax.axis("off")
    ax.set_title("Shared bathroom witness · 3/3 pass", fontsize=9.5, pad=1, color="#386943")


def _example_figure(examples, output):
    points = np.concatenate([np.array(room["polygon"])[:, :2]
                             * row["sample"]["scenario"]["coordinate_scale_m_per_model_unit"]
                             for _, row in examples for room in row["sample"]["rooms"]])
    midpoint = (points.max(axis=0) + points.min(axis=0)) / 2
    span = max(points.max(axis=0) - points.min(axis=0)) + 1.5
    limits = [midpoint[0] - span / 2, midpoint[0] + span / 2,
              midpoint[1] - span / 2, midpoint[1] + span / 2]
    bathroom_span = max(max(np.ptp(np.array(row["sample"]["rooms"][_bathroom(row)["room_index"]]["polygon"])
                                   * row["sample"]["scenario"]["coordinate_scale_m_per_model_unit"], axis=0))
                        for _, row in examples) + 0.55
    fig = plt.figure(figsize=(11.3, 7.6))
    grid = fig.add_gridspec(3, 3, height_ratios=[3.8, 1.0, 2.5], hspace=0.10, wspace=0.13,
                           left=0.02, right=0.98, top=0.91, bottom=0.13)
    for index, (record, row) in enumerate(examples):
        ax = fig.add_subplot(grid[0, index])
        _floorplan(ax, row, limits)
        status = overall_validity(row["evaluation"])
        ax.set_title(f"({chr(97 + index)}) {LABELS[_method(record)]} · seed {row['seed']}\n"
                     f"Overall {status.upper()} · structure valid", fontsize=10.5, pad=8,
                     color="#386943" if status == "valid" else "#a54e3c")
        _diagnostic(fig.add_subplot(grid[1, index]), row)
        _bathroom_inset(fig.add_subplot(grid[2, index]), row, bathroom_span)
    handles = [Patch(facecolor="#d9dee5", edgecolor="#4c586a", label="WC"),
               Patch(facecolor="#ece4f2", edgecolor="#8b5aa6", label="Basin"),
               Line2D([], [], color="#318255", linestyle="--", label="WC transfer"),
               Line2D([], [], color="#d08335", linestyle="--", label="Door swing"),
               Line2D([], [], color="#376d9f", linestyle="--", label="Turning circle")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.065), ncol=5, fontsize=9)
    fig.text(0.5, 0.035, "Saved hypothetical bathroom placements; shared C036/C037/C041 witness. Fixtures and doors are scenario assumptions.",
             ha="center", fontsize=9)
    fig.text(0.5, 0.01, "All plans use one common metric scale; bathroom insets use a second common scale. Room labels are saved model outputs.",
             ha="center", fontsize=9)
    return _save(fig, output, "qualitative_examples"), {"plan_limits_m": limits, "bathroom_inset_span_m": bathroom_span}


def _passing_examples(record, excluded_seed, output):
    """Six further successes, with the existing detailed SVG/PNG checker layout."""
    passing = [row for row in sorted(record["rows"], key=lambda row: row["trial"])
               if overall_validity(row["evaluation"]) == "valid"]
    selected = [row for row in passing if row["seed"] != excluded_seed][:6]
    points = np.concatenate([np.array(room["polygon"])[:, :2]
                             * row["sample"]["scenario"]["coordinate_scale_m_per_model_unit"]
                             for row in selected for room in row["sample"]["rooms"]])
    midpoint = (points.max(axis=0) + points.min(axis=0)) / 2
    span = max(points.max(axis=0) - points.min(axis=0)) + 1.5
    limits = [midpoint[0] - span / 2, midpoint[0] + span / 2,
              midpoint[1] - span / 2, midpoint[1] + span / 2]
    fig, axes = plt.subplots(2, 3, figsize=(11.8, 9.1))
    fig.subplots_adjust(left=0.015, right=0.985, top=0.91, bottom=0.07, hspace=0.20, wspace=0.08)
    fig.suptitle("SVDD (last 10%) · additional passing designs", fontsize=15, y=0.98)
    manifest, artifacts = [], []
    for ax, row in zip(axes.flat, selected):
        _floorplan(ax, row, limits)
        ax.set_title(f"Seed {row['seed']} · VALID\n11/11 checks pass · no unresolved checks",
                     fontsize=10, pad=7, color="#386943")
        prefix = output / "passing_svdd" / f"svdd_last10_seed{row['seed']}"
        rendered = save_sample(row["sample"], row["evaluation"], prefix)
        files = {key: Path(path).relative_to(output).as_posix() for key, path in rendered.items()}
        files.update(graph=f"passing_svdd/{prefix.name}_graph.json",
                     evaluation=f"passing_svdd/{prefix.name}_evaluation.json")
        write_json(output / files["graph"], row["sample"])
        write_json(output / files["evaluation"], row["evaluation"])
        manifest.append({"seed": row["seed"], "trial": row["trial"], "method": _method(record),
                         "overall_validity": "valid", "passed_checks": sum(map(_passed, counted_results(row))),
                         "artifacts": files})
        artifacts.extend(files[key] for key in ("png", "svg"))
    for ax in axes.flat[len(selected):]:
        ax.axis("off")
    fig.text(0.5, 0.025, "First six passing trials excluding seed " + str(excluded_seed)
             + "; common metric scale. These examples are selected from the passing subset.",
             ha="center", fontsize=9)
    artifacts += _save(fig, output, "svdd_passing_examples")
    write_json(output / "svdd_passing_examples_sources.json", {
        "method": "SVDD (last 10%)", "passing_outputs": len(passing),
        "selected_examples": len(selected), "excluded_seed_already_featured": excluded_seed,
        "selection": "First six overall-valid trials in ascending trial order, excluding the SVDD seed already shown in Figure 3.",
        "source": _fingerprint(record["run_dir"] / "candidates.jsonl"),
        "renderer": _fingerprint(Path(__file__).with_name("visualize.py")),
        "coordinate_scale_m_per_drawing_pixel": record["config"]["scenario"]["coordinate_scale_m_per_model_unit"],
        "plan_limits_m": limits, "examples": manifest,
    })
    return artifacts


def build_assets(experiment_dir, output_dir):
    """Export the four-method, 100-output bathroom comparison and provenance."""
    records, skipped = load_records(experiment_dir)
    by_method = {}
    for record in records:
        key = _method(record)
        if key in METHODS:
            if key in by_method:
                raise ValueError(f"Multiple {key} runs found; use a single matched experiment")
            by_method[key] = record
    if set(by_method) != set(METHODS):
        raise ValueError(f"Expected completed runs for {METHODS}; found {list(by_method)}; skipped: {skipped}")
    records = [by_method[key] for key in METHODS]
    for record in records:
        if len(record["rows"]) != 100:
            raise ValueError("The bathroom report requires exactly 100 selected outputs per method")
        if set(record["summary"]["active_scored_ids"]) != set(CONSTRAINTS):
            raise ValueError("The bathroom report requires the eleven configured scored checks")
        for row in record["rows"]:
            if Counter(check["id"] for check in counted_results(row)) != Counter(CONSTRAINTS.keys()):
                raise ValueError("Each selected output must have exactly one saved result for every scored check")
    examples = _select_examples(records)
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    constraints, partitions, assignments, sources = [], [], [], []
    for record in records:
        method, rows = _method(record), record["rows"]
        sources += [_fingerprint(record["run_dir"] / name) for name in ("candidates.jsonl", "config.json", "run.json")]
        for identifier in CONSTRAINTS:
            selected = [check for row in rows for check in counted_results(row) if check["id"] == identifier]
            counts = Counter(check["status"] for check in selected)
            passes = sum(map(_passed, selected))
            constraints.append({"method": method, "constraint": identifier, "description": CONSTRAINTS[identifier],
                                "selected_outputs": len(rows), "pass": passes, "fail": counts["fail"],
                                "unknown": counts["unknown"], "not_applicable": counts["not_applicable"],
                                "pass_percent": 100 * passes / len(rows)})
        counts = Counter(_partition(row) for row in rows)
        partitions.append({"method": method, "selected_outputs": len(rows),
                           **{key: counts[key] for key in PARTITIONS}})
        assignments.extend({"method": method, "trial": row["trial"], "seed": row["seed"],
                            "candidate_id": row["candidate_id"], "overall_validity": overall_validity(row["evaluation"]),
                            "partition": _partition(row)} for row in rows)
        if counts["valid"] != record["summary"]["overall_valid_count"] or sum(counts.values()) != len(rows):
            raise ValueError("Partition disagrees with the saved-output summary")
    _csv(output / "constraint_success.csv", constraints)
    _csv(output / "ventilation_bottleneck.csv", partitions)
    _csv(output / "selected_outcomes.csv", assignments)
    with plt.rc_context(PAPER_STYLE):
        artifacts = _constraint_figure(records, constraints, output)
        artifacts += _bottleneck_figure(partitions, output)
        files, geometry = _example_figure(examples, output)
        artifacts += files
        artifacts += _passing_examples(records[2], examples[1][1]["seed"], output)
    copied = []
    for record, row in examples:
        method = _method(record)
        prefix = f"examples/{method}_seed{row['seed']}"
        graph_path, evaluation_path = prefix + "_graph.json", prefix + "_evaluation.json"
        write_json(output / graph_path, row["sample"])
        write_json(output / evaluation_path, row["evaluation"])
        copied.append({"method": method, "seed": row["seed"], "trial": row["trial"], "candidate_id": row["candidate_id"],
                       "overall_validity": overall_validity(row["evaluation"]), "graph": graph_path,
                       "evaluation": evaluation_path, "source": _fingerprint(record["run_dir"] / "candidates.jsonl")})
    write_json(output / "qualitative_examples_sources.json", {
        "source": _path(experiment_dir), "selection": SELECTION, "examples": copied,
        "semantics": "Unedited saved room polygons and model labels. Bathroom overlays are saved, audited hypothetical witnesses; no new placements are searched.",
        **geometry,
    })
    data = {"source_experiment": _path(experiment_dir), "source_files": sources,
            "method_order": METHODS, "method_labels": LABELS, "constraint_labels": CONSTRAINTS,
            "definitions": DEFINITIONS, "constraint_success": constraints, "ventilation_bottleneck": partitions,
            "example_selection": SELECTION, "figures": artifacts,
            "generator": _fingerprint(Path(__file__)),
            "regenerate": f"python3 -m test_time.report_assets --experiment-dir {_path(experiment_dir)} --output-dir {_path(output)}"}
    write_json(output / "figure_data.json", data)
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", default="test_time/outputs/bathroom_001")
    parser.add_argument("--output-dir", default="docs/assets/bathroom_001")
    args = parser.parse_args()
    data = build_assets(args.experiment_dir, args.output_dir)
    print(json.dumps({"output_dir": args.output_dir, "partition": data["ventilation_bottleneck"]}, indent=2))


if __name__ == "__main__":
    main()
