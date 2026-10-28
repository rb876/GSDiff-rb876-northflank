"""Standalone analysis figures from saved, selected designs; no model inference."""

import math
from pathlib import Path
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import PercentFormatter
import numpy as np

from .evaluate import counted_results, measurable, polygon_area
from .types import overall_validity


METHOD_COLORS = ("#4477aa", "#ee6677", "#228833", "#aa3377", "#ccbb44", "#66ccee")
STATUS_COLORS = {"pass": "#228833", "fail": "#cc4444", "unknown": "#ddaa33", "not_applicable": "#bbbbbb"}


def _label(record, width=24):
    return textwrap.fill(record["label"], width=width, break_long_words=False)


def _finish(fig, output, name, caption, title):
    fig.suptitle(title, fontsize=14)
    fig.text(0.02, 0.015, textwrap.fill(caption, 145), fontsize=8, va="bottom")
    fig.tight_layout(rect=(0, 0.095, 1, 0.94))
    paths = {}
    for extension in ("png", "pdf"):
        path = output / f"{name}.{extension}"
        fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
        paths[extension] = str(path)
    plt.close(fig)
    return paths


def _empty(ax, message="No eligible observations"):
    ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes)


def _fraction_axis(ax, label):
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.set_ylabel(label)
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)


def plot_analysis(records, output_dir):
    """Save PNG/PDF figures and return their absolute paths by figure name.

    Each record contains its label, selected candidate rows, per-constraint rows,
    and scalar metrics calculated by the analysis command. Unknown values stay
    unknown; only measured pass/fail results enter the severity histogram.
    """
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    artifacts = {}
    if not records:
        fig, ax = plt.subplots(figsize=(8, 4))
        _empty(ax, "No selected designs to analyse")
        artifacts["overall_validity"] = _finish(fig, output, "overall_validity", "No observations were supplied.", "Overall validity")
        return artifacts

    colors = [METHOD_COLORS[index % len(METHOD_COLORS)] for index in range(len(records))]
    labels = [_label(record) for record in records]
    positions = np.arange(len(records))

    fig, axes = plt.subplots(1, 2, figsize=(max(11, len(records) * 2.4), 5.8))
    for index, (record, color) in enumerate(zip(records, colors)):
        metrics, count = record["metrics"], len(record["rows"])
        value = metrics.get("overall_valid_rate")
        if value is None:
            axes[0].text(index, 0.03, "unknown", ha="center", rotation=90)
            continue
        axes[0].bar(index, value, color=color)
        low, high = metrics.get("overall_ci_low"), metrics.get("overall_ci_high")
        if low is not None and high is not None:
            axes[0].errorbar(index, value, yerr=[[max(0, value - low)], [max(0, high - value)]],
                             color="#222222", capsize=4, linewidth=1.2)
        axes[0].text(index, min(1.06, (high if high is not None else value) + 0.035),
                     f"{value:.1%}\nn={count}", ha="center", fontsize=9)
    axes[0].set(title="All applicable constraints pass + valid graph", ylim=(0, 1.15), xticks=positions, xticklabels=labels)
    _fraction_axis(axes[0], "Fraction of selected designs")
    bottom = np.zeros(len(records))
    outcomes = ["valid", "invalid", "unknown", "not_applicable"]
    for status, status_color in zip(outcomes, STATUS_COLORS.values()):
        values = [sum(overall_validity(row["evaluation"]) == status for row in record["rows"]) / len(record["rows"])
                  if record["rows"] else 0 for record in records]
        axes[1].bar(positions, values, bottom=bottom, color=status_color, label=status.replace("_", " "))
        bottom += values
    axes[1].set(title="Overall outcomes", ylim=(0, 1.05), xticks=positions, xticklabels=labels)
    _fraction_axis(axes[1], "Fraction of selected designs")
    axes[1].legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, 1.2), ncol=2)
    artifacts["overall_validity"] = _finish(fig, output, "overall_validity",
        "Denominator: every selected output, including invalid, unknown and not-applicable designs. Error bars are 95% Wilson intervals for overall validity.",
        "Overall validity by method")

    columns = min(2, len(records))
    rows = math.ceil(len(records) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(7.4 * columns, 3.8 * rows), squeeze=False)
    for ax, record in zip(axes.flat, records):
        constraints, count = record["constraint_rows"], len(record["rows"])
        if not constraints or not count:
            _empty(ax)
            ax.set_title(_label(record))
            continue
        y = np.arange(len(constraints))
        highest = max(item.get("fail", 0) for item in constraints)
        ids = [item["id"] + (" *" if highest and item.get("fail", 0) == highest else "") for item in constraints]
        left = np.zeros(len(constraints))
        for status, color in STATUS_COLORS.items():
            values = np.array([item.get(status, 0) / count for item in constraints])
            ax.barh(y, values, left=left, color=color, label=status.replace("_", " "))
            for index, value in enumerate(values):
                if value >= 0.09:
                    ax.text(left[index] + value / 2, index, f"{value:.0%}", ha="center", va="center", fontsize=8)
            left += values
        ax.set(yticks=y, yticklabels=ids, xlim=(0, 1), title=f"{_label(record, 45)} — n={count}", xlabel="Fraction of selected designs")
        ax.invert_yaxis()
        ax.xaxis.set_major_formatter(PercentFormatter(1))
    for ax in list(axes.flat)[len(records):]:
        ax.set_visible(False)
    fig.legend(handles=[Patch(facecolor=color, label=status.replace("_", " ")) for status, color in STATUS_COLORS.items()],
               loc="lower center", bbox_to_anchor=(0.5, 0.062), ncol=4, fontsize=9)
    artifacts["constraint_outcomes"] = _finish(fig, output, "constraint_outcomes",
        "Each bar uses all selected outputs. Unknown and N/A are separate from passes. Supporting definitions are excluded. * marks the highest confirmed failure count within each method.",
        "Which constraints are hardest?")

    successes = [[sum(measurable(check) and check["status"] == "pass" for check in counted_results(row))
                  for row in record["rows"]] for record in records]
    max_checks = max((len(counted_results(row)) for record in records for row in record["rows"]), default=0)
    fig, ax = plt.subplots(figsize=(9, 5))
    for record, values, color in zip(records, successes, colors):
        if values:
            ax.hist(values, bins=np.arange(max_checks + 2) - 0.5, weights=np.ones(len(values)) / len(values),
                    histtype="step", linewidth=2, color=color, label=f"{record['label']} (n={len(values)})")
    if any(successes):
        ax.legend(fontsize=8)
    else:
        _empty(ax)
    ax.set(xlabel="Number of scored constraints passed", xticks=np.arange(max_checks + 1))
    _fraction_axis(ax, "Fraction of selected designs")
    artifacts["passed_constraints"] = _finish(fig, output, "passed_constraints",
        "Each method's histogram is normalized by its selected-output count. Unknown, failed and inapplicable checks do not add a pass; supporting definitions are excluded. This does not itself establish structural validity.",
        "Constraints passed per design")

    fig, ax = plt.subplots(figsize=(9, 5))
    observed = []
    for record in records:
        checks = [check for row in record["rows"] for check in counted_results(row)]
        values = [check["severity"] for check in checks if measurable(check) and check.get("severity") is not None
                  and math.isfinite(check["severity"])]
        observed.append((values, sum(check["status"] == "unknown" for check in checks)))
    maximum = max([1.0, *(value for values, _ in observed for value in values)])
    for record, (values, unknown), color in zip(records, observed, colors):
        if values:
            ax.hist(values, bins=np.linspace(0, maximum, 21), weights=np.ones(len(values)) / len(values),
                    histtype="step", linewidth=2, color=color,
                    label=f"{record['label']} (measured={len(values)}, unknown={unknown})")
    if any(values for values, _ in observed):
        ax.legend(fontsize=8)
    else:
        _empty(ax, "No measurable constraint severities")
    ax.set_xlabel("Normalized constraint severity (0 = satisfied)")
    _fraction_axis(ax, "Fraction of measurable scored checks")
    artifacts["constraint_severity"] = _finish(fig, output, "constraint_severity",
        "Only finite severities of applicable, measured scored checks are included, including zero-severity passes. Unknown values are omitted, never converted to zero. Structural penalties are excluded.",
        "Measured violation severity")

    features = []
    for record in records:
        counts, areas = [], []
        for row in record["rows"]:
            if not row["evaluation"]["structurally_valid"]:
                continue
            rooms = [room for room in row["sample"].get("rooms", []) if room["label"] != 6]
            room_areas = [polygon_area(room["polygon"]) for room in rooms]
            if not rooms or not all(math.isfinite(value) and value > 0 for value in room_areas):
                continue
            counts.append(len(rooms))
            areas.append(sum(room_areas))
        features.append((counts, areas))
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    max_rooms = max((count for counts, _ in features for count in counts), default=0)
    all_areas = [area for _, areas in features for area in areas]
    area_bins = np.histogram_bin_edges(all_areas, bins=15) if all_areas else np.linspace(0, 1, 16)
    for record, (counts, areas), color in zip(records, features, colors):
        if counts:
            for ax, values, bins in ((axes[0], counts, np.arange(max_rooms + 2) - 0.5), (axes[1], areas, area_bins)):
                ax.hist(values, bins=bins, weights=np.ones(len(values)) / len(values), histtype="step",
                        linewidth=2, color=color, label=f"{record['label']} (n={len(values)})")
    for ax in axes:
        _fraction_axis(ax, "Fraction of eligible designs")
        if all_areas:
            ax.legend(fontsize=8)
        else:
            _empty(ax, "No structurally valid designs with usable rooms")
    axes[0].set_xlabel("Number of interior rooms")
    axes[1].set_xlabel("Sum of interior room areas (drawing pixels²)")
    artifacts["layout_distributions"] = _finish(fig, output, "layout_distributions",
        "Descriptive distributions of structurally valid selected designs with finite, positive room areas. Outside label 6 is excluded. Compare with the generated baseline; these are not RPLAN fidelity metrics.",
        "Layout distribution diagnostics")

    fig, axes = plt.subplots(1, 2, figsize=(max(11, len(records) * 2.4), 5.8))
    for ax, key, title, ylabel in ((axes[0], "wall_seconds_per_trial", "Measured runtime", "Seconds per selected output"),
                                   (axes[1], "node_samples_per_trial", "Denoiser work", "Evaluated states per selected output")):
        known = [record["metrics"].get(key) for record in records
                 if record["metrics"].get(key) is not None and math.isfinite(record["metrics"][key])]
        for index, (record, color) in enumerate(zip(records, colors)):
            value = record["metrics"].get(key)
            if value is None or not math.isfinite(value):
                if known:
                    ax.text(index, 0.03, "unknown", transform=ax.get_xaxis_transform(), ha="center", fontsize=8)
                continue
            ax.bar(index, value, color=color)
            ax.annotate(f"{value:,.1f}", (index, value), xytext=(0, 4), textcoords="offset points", ha="center", fontsize=8)
        ax.set(title=title, ylabel=ylabel, xticks=positions, xticklabels=labels)
        ax.set(xlim=(-0.6, len(records) - 0.4), ylim=(0, max(known, default=0) * 1.2 or 1))
        if not known:
            ax.set_yticks([])
            _empty(ax, "No recorded values\nMissing counters remain unknown")
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
    artifacts["compute"] = _finish(fig, output, "compute",
        "Recorded batch sampling/checking time is divided by selected outputs; model loading and rendering/export are excluded. Denoiser evaluations count states, not batched calls. Equal denoiser counts do not imply equal runtime.",
        "Compute and runtime per selected output")
    return artifacts
