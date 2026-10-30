"""Render Hard-MC success and failure evidence from saved outputs only."""

import argparse
import json
from pathlib import Path
import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

from .analysis_plots import PAPER_STYLE
from .report_assets import _fingerprint, _floorplan, _save
from .types import write_json
from .visualize import save_sample


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def select_examples(source):
    trials = sorted(read_rows(source / "trials.jsonl"), key=lambda row: row["trial"])
    success = next(row for row in trials if row["returned_graph"] and row["overall_validity"] == "valid")
    terminals = {}
    for event in read_rows(source / "events.jsonl"):
        if event["origin"] == "terminal":
            terminals.setdefault(event["trial"], []).append(event)
    terminal_k = json.loads((source / "config.json").read_text())["parameters"]["terminal_k"]
    for row in trials:
        events = sorted(terminals.get(row["trial"], []), key=lambda event: event["proposal"])
        if (not row["returned_graph"] and len(events) == terminal_k
                and all(event["structurally_valid"] and not event["unknown_ids"]
                        and set(event["failing_ids"]) == {"C085", "C090"}
                        and all(status == ("fail" if cid in {"C085", "C090"} else "pass")
                                for cid, status in event["checks"].items()) for event in events)):
            return success, row, events
    raise ValueError("No failed attempt with eight ventilation-only rejected terminal candidates was found")


def plan_limits(sample):
    scale = sample["scenario"]["coordinate_scale_m_per_model_unit"]
    points = np.concatenate([np.asarray(room["polygon"])[:, :2] * scale for room in sample["rooms"]])
    midpoint = (points.max(axis=0) + points.min(axis=0)) / 2
    span = max(points.max(axis=0) - points.min(axis=0)) + 1.8
    return [midpoint[0] - span / 2, midpoint[0] + span / 2,
            midpoint[1] - span / 2, midpoint[1] + span / 2]


def success_panel(ax, row, label=""):
    _floorplan(ax, row, plan_limits(row["sample"]))
    ax.set_title(f"{label}Returned success · seed {row['seed']}\n"
                 "Valid structure · all 11 checks pass", color="#386943", pad=13)


def failure_panel(ax, failed, events, label=""):
    identifiers = sorted(events[0]["checks"])
    values = np.array([[event["checks"][cid] == "fail" for cid in identifiers] for event in events])
    ax.imshow(values, cmap=ListedColormap(["#e2efdf", "#f3d4cb"]), vmin=0, vmax=1, aspect="auto")
    for i, event in enumerate(events):
        for j, identifier in enumerate(identifiers):
            fail = event["checks"][identifier] == "fail"
            ax.text(j, i, "F" if fail else "P", ha="center", va="center", fontsize=9,
                    color="#a1422e" if fail else "#386943", fontweight="bold" if fail else "normal")
    ax.set_xticks(range(len(identifiers)), identifiers, rotation=50, ha="right")
    ax.set_yticks(range(len(events)), [f"Candidate {event['proposal'] + 1}" for event in events])
    ax.set_xticks(np.arange(-0.5, len(identifiers)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(events)), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.tick_params(which="both", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(f"{label}Failed attempt · seed {failed['seed']}\n"
                 "0 / 8 terminal candidates accepted", color="#a1422e", pad=13)
    ax.text(0.5, -0.27, "P = pass     F = fail · all eight have valid structure\n"
            "C085: kitchen extraction · C090: reduced-mode extraction",
            transform=ax.transAxes, ha="center", va="top", fontsize=9, linespacing=1.5)


def build_assets(source, output):
    source, output = Path(source), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    success, failed, events = select_examples(source)
    graph_path = source / success["artifacts"]["graph"]
    evaluation_path = source / success["artifacts"]["evaluation"]
    row = {"seed": success["seed"], "sample": json.loads(graph_path.read_text()),
           "evaluation": json.loads(evaluation_path.read_text())}
    success_name = f"hard_mc_success_seed{success['seed']}"
    failure_name = f"hard_mc_failure_seed{failed['seed']}"
    shutil.copyfile(graph_path, output / f"{success_name}_graph.json")
    shutil.copyfile(evaluation_path, output / f"{success_name}_evaluation.json")
    write_json(output / f"{failure_name}_events.json", events)
    detailed = save_sample(row["sample"], row["evaluation"], output / f"{success_name}_annotated")
    with plt.rc_context(PAPER_STYLE):
        fig, ax = plt.subplots(figsize=(5.5, 6.0))
        fig.subplots_adjust(top=0.85, bottom=0.10)
        success_panel(ax, row)
        fig.text(0.5, 0.025, "Saved room labels and areas · scale: 18 m / 512 drawing pixels",
                 ha="center", fontsize=9)
        assets = _save(fig, output, success_name)

        fig, ax = plt.subplots(figsize=(7.4, 5.2))
        fig.subplots_adjust(left=0.16, right=0.98, top=0.83, bottom=0.29)
        failure_panel(ax, failed, events)
        fig.text(0.56, 0.015, "No graph returned; rejected floorplans were not retained.",
                 ha="center", fontsize=9, color="#555555")
        assets += _save(fig, output, failure_name)

        fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.5), gridspec_kw={"width_ratios": [1, 1.28]})
        fig.subplots_adjust(left=0.02, right=0.99, top=0.82, bottom=0.27, wspace=0.25)
        fig.suptitle("Hard-MC · successful output and failed terminal search", fontsize=14, y=0.98)
        success_panel(axes[0], row, "(a) ")
        failure_panel(axes[1], failed, events, "(b) ")
        fig.text(0.5, 0.025, "Selected examples from different attempts. Failed terminal floorplans were not retained; panel (b) shows saved check results.",
                 ha="center", fontsize=9)
        assets += _save(fig, output, "hard_mc_examples")

    write_json(output / "examples_sources.json", {
        "method": "hard_h_mc", "selection": {
            "success": "First returned valid design in ascending trial order.",
            "failure": "First failed attempt in trial order whose eight terminal candidates all have valid structure and fail only C085/C090, with no unknown checks.",
        },
        "success": {"trial": success["trial"], "seed": success["seed"],
                    "graph": f"{success_name}_graph.json", "evaluation": f"{success_name}_evaluation.json",
                    "detailed_figures": {key: Path(value).name for key, value in detailed.items()}},
        "failure": {"trial": failed["trial"], "seed": failed["seed"],
                    "trial_status": failed["status"], "returned_graph": False,
                    "terminal_events": f"{failure_name}_events.json", "candidate_count": len(events),
                    "failing_ids": ["C085", "C090"], "rejected_geometry_available": False,
                    "limitations": "Saved checks identify failures but do not retain the rejected layouts, room counts, or numerical airflow deficits."},
        "coordinate_scale_m_per_drawing_pixel": row["sample"]["scenario"]["coordinate_scale_m_per_model_unit"],
        "sources": [_fingerprint(source / name) for name in ("config.json", "trials.jsonl", "events.jsonl")]
                   + [_fingerprint(graph_path), _fingerprint(evaluation_path)],
        "renderers": [_fingerprint(Path(__file__)), _fingerprint(Path(__file__).with_name("report_assets.py")),
                      _fingerprint(Path(__file__).with_name("visualize.py"))],
        "figures": assets, "regenerate": "python3 -m test_time.hard_mc_assets",
    })
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("test_time/outputs/hard_mc_overnight_001"))
    parser.add_argument("--output", type=Path, default=Path("docs/assets/hard_mc_overnight_001"))
    args = parser.parse_args()
    print(f"Saved {build_assets(args.source, args.output)}")


if __name__ == "__main__":
    main()
