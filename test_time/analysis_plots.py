"""Paper-style figures from saved, selected designs; no model inference."""

import math
from pathlib import Path
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator, PercentFormatter
import numpy as np

from .evaluate import counted_results, measurable, polygon_area
from .types import overall_validity


# Method identity determines style, so omitted/reordered runs keep their colours.
METHOD_STYLES = {
    "baseline": ("#4c80b5", "////"),
    "best_of_n": ("#df8134", "\\\\\\\\"),
    "best_of_n_3": ("#9e6aa8", "...."),
    "best_of_n_11": ("#df8134", "\\\\\\\\"),
    "svdd_pm_last10": ("#8b8683", ""),
    "svdd_pm_last20": ("#8b8683", ""),
    "svdd_pm_full": ("#579b50", "xxxx"),
}
STATUS_STYLES = {
    "pass": ("#579b50", ""),
    "fail": ("#b86555", "////"),
    "unknown": ("#b89131", "...."),
    "not_applicable": ("#939393", "xx"),
}
STATUS_LABELS = {"pass": "Pass", "fail": "Fail", "unknown": "Unknown", "not_applicable": "N/A"}
PAPER_STYLE = {
    "font.family": "serif",
    "font.serif": ["STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "axes.linewidth": 0.65,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.axisbelow": True,
    "axes.facecolor": "white",
    "figure.facecolor": "white",
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "grid.color": "#e7e7e7",
    "grid.linewidth": 0.45,
    "hatch.linewidth": 0.55,
    "legend.frameon": False,
    "legend.fontsize": 9,
    "legend.handlelength": 1.7,
    "legend.handleheight": 0.85,
    "legend.columnspacing": 1.4,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.dpi": 300,
}
FIGURE_CAPTIONS = {
    "overall_validity": "Every selected output remains in the denominator. Overall validity requires valid structure and all applicable scored constraints to pass; error bars are 95% Wilson intervals. The second panel retains invalid, unknown and N/A outcomes.",
    "constraint_outcomes": "Each bar uses all selected outputs. Unknown and N/A are separate from passes; supporting definitions are excluded. Ventilation checks are labelled and separated from geometric checks by a dotted line when both are present. An asterisk marks the highest confirmed failure count within each method. Labels show segments of at least 8%.",
    "passed_constraints": "Grouped bars are integer-bin histograms, normalized separately by each method's selected-output count. Unknown, failed and inapplicable checks do not add a pass; supporting definitions are excluded. Passing checks alone does not establish structural validity. Nonzero bars are labelled; zero bins remain empty.",
    "constraint_severity": "All panels share bins and axis limits. Only finite severities of applicable, measured scored checks are included, including zero-severity passes. Panel annotations give unknown check counts. Unknown values are omitted, never converted to zero; structural penalties are excluded.",
    "layout_distributions": "Each method is normalized by its structurally valid selected designs with finite, positive room areas; eligible counts are recorded in the report tables. Bins and axis limits are shared within each row. Outside label 6 is excluded. Area is in drawing pixels squared. These baseline-relative diagnostics are not RPLAN fidelity metrics.",
    "compute": "Recorded batch sampling/checking time is divided by selected outputs; model loading and rendering/export are excluded. Denoiser NFE (number of function evaluations) counts each evaluated sample within a batch, including candidate trajectories and SVDD lookahead, per selected output. For prefix-reselected controls, NFE is the equivalent cost of retained trajectories; reselection adds no model inference and the full source bank's cost remains in provenance. Edge-model and reward evaluations are excluded from NFE. Equal NFE does not imply equal runtime. No uncertainty intervals are inferred from these aggregate counters.",
}


def _method(record):
    name = record["label"].split("/")[-1]
    summary, config = record.get("summary", {}), record.get("config", {})
    method = config.get("method", summary.get("method", name))
    parameters = summary.get("parameters", {})
    parameters = {**parameters, **record.get("metrics", {})}
    if method == "baseline":
        return "baseline", "Baseline"
    if method == "best_of_n" or name.startswith("best_of_n"):
        n = config.get("n", parameters.get("n"))
        return (f"best_of_n_{n}", f"Best-of-{n}") if n is not None else ("best_of_n", "Best-of-N")
    if method in {"svdd_pm", "svdd_pm_full", "svdd_pm_last10", "svdd_pm_last20"}:
        steps = config.get("guided_steps", parameters.get("guided_steps"))
        if name == "svdd_pm_last10" or method == "svdd_pm_last10" or (method == "svdd_pm" and steps == 100):
            return "svdd_pm_last10", "SVDD (last 10%)"
        if name == "svdd_pm_last20" or (method == "svdd_pm" and steps == 200):
            return "svdd_pm_last20", "SVDD (last 20%)"
        if name == "svdd_pm_full" or (method == "svdd_pm" and steps == 999):
            return "svdd_pm_full", "SVDD (full)"
        return "svdd_pm_full", f"SVDD ({steps} steps)" if steps is not None else "SVDD"
    return "baseline", name.replace("_", " ")


def _label(record, width=None):
    _, name = _method(record)
    # Distinguish multiple seed batches without assigning different method colours.
    if "/" in record["label"]:
        name += " · " + record["label"].rsplit("/", 1)[0]
    return textwrap.fill(name, width, break_long_words=False) if width else name


def _style(record):
    key = _method(record)[0]
    if key not in METHOD_STYLES and key.startswith("best_of_n_"):
        # Keep additional N settings deterministic across subset/reordered plots.
        n = int(key.rsplit("_", 1)[1])
        palette = ("#c56078", "#8b744e", "#287c8e", "#b89b32")
        hatches = ("++", "oo", "--", "**")
        return palette[n % len(palette)], hatches[(n // len(palette)) % len(hatches)]
    return METHOD_STYLES[key]


def _bar_style(style):
    color, hatch = style
    return {"facecolor": to_rgba(color, 0.14), "edgecolor": color, "hatch": hatch, "linewidth": 0.8}


def _method_handles(records):
    return [Patch(**_bar_style(_style(record)), label=_label(record)) for record in records]


def _legend(fig, handles, columns=4):
    fig.legend(handles=handles, loc="outside lower center", ncol=min(columns, len(handles)), borderaxespad=0.3)


def _finish(fig, output, name):
    paths = {}
    for extension in ("png", "pdf"):
        path = output / f"{name}.{extension}"
        fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.05, facecolor="white")
        paths[extension] = str(path)
    plt.close(fig)
    return paths


def _empty(ax, message="No eligible observations"):
    ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes, fontsize=9)


def _fraction_axis(ax, label="Selected designs (%)"):
    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5, steps=[1, 2, 2.5, 5, 10]))
    ax.set_ylabel(label)
    ax.grid(axis="y")


def _frequencies(values, bins):
    counts = np.histogram(values, bins=bins)[0]
    return counts / len(values) if len(values) else counts.astype(float)


def _percentage(value):
    return f"{value:.0%}" if abs(value * 100 - round(value * 100)) < 1e-7 else f"{value:.1%}"


def _annotate_bars(ax, bars, values, percentage=False, fontsize=8):
    for bar, value in zip(bars, values):
        if percentage and value == 0:
            continue
        label = _percentage(value) if percentage else f"{value:,.0f}" if value >= 1000 else f"{value:.1f}"
        ax.annotate(label, (bar.get_x() + bar.get_width() / 2, value), xytext=(0, 3),
                    textcoords="offset points", ha="center", va="bottom", fontsize=fontsize)


def plot_analysis(records, output_dir):
    """Export compact PNG/PDF figures without changing metrics or global styles."""
    with plt.rc_context(PAPER_STYLE):
        return _plot_analysis(records, output_dir)


def _plot_analysis(records, output_dir):
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    artifacts = {}
    if not records:
        fig, ax = plt.subplots(figsize=(6, 3), layout="constrained")
        _empty(ax, "No selected designs to analyse")
        artifacts["overall_validity"] = _finish(fig, output, "overall_validity")
        return artifacts

    labels = [_label(record, 16).replace("SVDD (", "SVDD\n(") for record in records]
    positions = np.arange(len(records))
    styles = [_style(record) for record in records]

    fig, axes = plt.subplots(1, 2, figsize=(max(8.2, len(records) * 1.7), 3.5), layout="constrained")
    extent = 0.0
    for index, (record, style) in enumerate(zip(records, styles)):
        metrics = record["metrics"]
        value = metrics.get("overall_valid_rate")
        if value is None:
            axes[0].text(index, 0.02, "N/A", ha="center", fontsize=8)
            continue
        axes[0].bar(index, value, width=0.62, **_bar_style(style))
        low, high = metrics.get("overall_ci_low"), metrics.get("overall_ci_high")
        if low is not None and high is not None:
            axes[0].errorbar(index, value, yerr=[[max(0, value - low)], [max(0, high - value)]],
                             fmt="none", ecolor="#444444", capsize=2.5, elinewidth=0.8, capthick=0.8)
        top = high if high is not None else value
        extent = max(extent, top)
        axes[0].annotate(_percentage(value), (index, top), xytext=(0, 4), textcoords="offset points",
                         ha="center", va="bottom", fontsize=9)
    axes[0].set(title="(a) Overall validity · 95% CI", ylim=(0, max(0.1, extent * 1.25)))
    _fraction_axis(axes[0])
    bottom = np.zeros(len(records))
    outcomes = [("valid", "pass"), ("invalid", "fail"), ("unknown", "unknown"), ("not_applicable", "not_applicable")]
    for status, key in outcomes:
        values = np.array([sum(overall_validity(row["evaluation"]) == status for row in record["rows"]) / len(record["rows"])
                           if record["rows"] else 0 for record in records])
        axes[1].bar(positions, values, bottom=bottom, width=0.62, **_bar_style(STATUS_STYLES[key]))
        for index, value in enumerate(values):
            if value >= 0.08:
                axes[1].text(index, bottom[index] + value / 2, _percentage(value), ha="center", va="center", fontsize=8)
        bottom += values
    axes[1].set(title="(b) Overall outcomes", ylim=(0, 1.04))
    _fraction_axis(axes[1])
    for ax in axes:
        ax.set(xticks=positions, xticklabels=labels, xlim=(-0.6, len(records) - 0.4))
    _legend(fig, [Patch(**_bar_style(STATUS_STYLES[key]), label={"valid": "Valid", "invalid": "Invalid"}.get(status, STATUS_LABELS[key]))
                  for status, key in outcomes])
    artifacts["overall_validity"] = _finish(fig, output, "overall_validity")

    columns = min(2, len(records))
    rows = math.ceil(len(records) / columns)
    max_constraints = max((len(record["constraint_rows"]) for record in records), default=0)
    panel_height = max(2.7, 0.32 * max_constraints + 0.8)
    fig, axes = plt.subplots(rows, columns, figsize=(4.6 * columns, panel_height * rows + 0.35), squeeze=False, layout="constrained")
    for panel, (ax, record) in enumerate(zip(axes.flat, records)):
        constraints, count = record["constraint_rows"], len(record["rows"])
        ax.set_title(f"({chr(97 + panel)}) {_label(record)}")
        if not constraints or not count:
            _empty(ax)
            continue
        y = np.arange(len(constraints))
        highest = max(item.get("fail", 0) for item in constraints)
        family = lambda item: "ventilation" if item["id"] in {"C085", "C086", "C090"} else "geometry"
        ids = [item["id"] + (" · vent." if family(item) == "ventilation" else "")
               + (" *" if highest and item.get("fail", 0) == highest else "") for item in constraints]
        left = np.zeros(len(constraints))
        for status, style in STATUS_STYLES.items():
            values = np.array([item.get(status, 0) / count for item in constraints])
            ax.barh(y, values, left=left, height=0.64, **_bar_style(style))
            for index, value in enumerate(values):
                if value >= 0.08:
                    ax.text(left[index] + value / 2, index, _percentage(value), ha="center", va="center", fontsize=8)
            left += values
        ax.set(yticks=y, yticklabels=ids, xlim=(0, 1), xlabel="Selected designs (%)")
        for index in range(1, len(constraints)):
            if family(constraints[index]) != family(constraints[index - 1]):
                ax.axhline(index - 0.5, color="#999999", linewidth=0.65, linestyle=":")
        ax.invert_yaxis()
        ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        ax.grid(axis="x")
    for ax in list(axes.flat)[len(records):]:
        ax.set_visible(False)
    _legend(fig, [Patch(**_bar_style(style), label=STATUS_LABELS[status]) for status, style in STATUS_STYLES.items()])
    artifacts["constraint_outcomes"] = _finish(fig, output, "constraint_outcomes")

    successes = [[sum(measurable(check) and check["status"] == "pass" for check in counted_results(row))
                  for row in record["rows"]] for record in records]
    max_checks = max((len(counted_results(row)) for record in records for row in record["rows"]), default=0)
    bins = np.arange(max_checks + 2) - 0.5
    centres = np.arange(max_checks + 1)
    width = 0.82 / len(records)
    fig, ax = plt.subplots(figsize=(max(8.0, 0.9 * (max_checks + 1)), 3.5), layout="constrained")
    maximum = 0.0
    for index, (values, style) in enumerate(zip(successes, styles)):
        frequencies = _frequencies(values, bins)
        maximum = max(maximum, float(frequencies.max(initial=0)))
        bars = ax.bar(centres + (index - (len(records) - 1) / 2) * width, frequencies,
                      width=width * 0.94, **_bar_style(style))
        _annotate_bars(ax, bars, frequencies, percentage=True, fontsize=7)
    if not any(successes):
        _empty(ax)
    ax.set(xlabel="Number of constraints passed", xticks=centres, xlim=(-0.6, max_checks + 0.6),
           ylim=(0, maximum * 1.18 if maximum else 1))
    _fraction_axis(ax)
    _legend(fig, _method_handles(records))
    artifacts["passed_constraints"] = _finish(fig, output, "passed_constraints")

    observed = []
    for record in records:
        checks = [check for row in record["rows"] for check in counted_results(row)]
        values = [check["severity"] for check in checks if measurable(check) and check.get("severity") is not None
                  and math.isfinite(check["severity"])]
        observed.append((values, sum(check["status"] == "unknown" for check in checks)))
    maximum = max([1.0, *(value for values, _ in observed for value in values)])
    bins = np.linspace(0, maximum, 21)
    frequencies = [_frequencies(values, bins) for values, _ in observed]
    ymax = max((float(values.max(initial=0)) for values in frequencies), default=0)
    fig, axes = plt.subplots(1, len(records), figsize=(max(4, 2.5 * len(records)), 3.0),
                             squeeze=False, sharex=True, sharey=True, layout="constrained")
    for index, (ax, record, style, (values, unknown), frequency) in enumerate(zip(axes.flat, records, styles, observed, frequencies)):
        ax.bar(bins[:-1], frequency, width=np.diff(bins), align="edge", **_bar_style(style))
        ax.set(title=_label(record), xlabel="Violation severity", xlim=(-0.02 * maximum, maximum * 1.02), ylim=(0, ymax * 1.12 if ymax else 1))
        ax.text(0.96, 0.95, f"unknown={unknown}", transform=ax.transAxes, ha="right", va="top", fontsize=8)
        _fraction_axis(ax, "Measured checks (%)" if index == 0 else "")
        if not values:
            _empty(ax, "No measured severities")
    artifacts["constraint_severity"] = _finish(fig, output, "constraint_severity")

    features = []
    for record in records:
        room_counts, areas = [], []
        for row in record["rows"]:
            if not row["evaluation"]["structurally_valid"]:
                continue
            rooms = [room for room in row["sample"].get("rooms", []) if room["label"] != 6]
            room_areas = [polygon_area(room["polygon"]) for room in rooms]
            if not rooms or not all(math.isfinite(value) and value > 0 for value in room_areas):
                continue
            room_counts.append(len(rooms))
            areas.append(sum(room_areas))
        features.append((room_counts, areas))
    max_rooms = max((count for room_counts, _ in features for count in room_counts), default=0)
    count_bins = np.arange(max_rooms + 2) - 0.5
    all_areas = [area for _, areas in features for area in areas]
    area_bins = np.histogram_bin_edges(all_areas, bins=15) if all_areas else np.linspace(0, 1, 16)
    fig, axes = plt.subplots(2, len(records), figsize=(max(4, 2.5 * len(records)), 5.0),
                             squeeze=False, sharex="row", sharey="row", layout="constrained")
    for row, bins in enumerate((count_bins, area_bins)):
        frequencies = [_frequencies(feature[row], bins) for feature in features]
        ymax = max((float(value.max(initial=0)) for value in frequencies), default=0)
        for index, (record, style, feature, frequency) in enumerate(zip(records, styles, features, frequencies)):
            ax = axes[row, index]
            ax.bar((bins[:-1] + bins[1:]) / 2, frequency, width=np.diff(bins) * 0.9, **_bar_style(style))
            ax.set_ylim(0, ymax * 1.12 if ymax else 1)
            _fraction_axis(ax, "Eligible designs (%)" if index == 0 else "")
            if row == 0:
                ax.set(title=_label(record), xlabel="Interior rooms")
                ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))
            else:
                ax.set_xlabel("Area (drawing pixels²)")
                if all_areas:
                    ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0), useMathText=True)
                    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
            if not feature[row]:
                _empty(ax, "No eligible designs")
    artifacts["layout_distributions"] = _finish(fig, output, "layout_distributions")

    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.4), layout="constrained")
    for ax, key, title, ylabel in ((axes[0], "wall_seconds_per_trial", "(a) Generation time", "Seconds per output"),
                                   (axes[1], "node_samples_per_trial", "(b) Denoiser NFE", "NFE per output")):
        known = [record["metrics"].get(key) for record in records
                 if record["metrics"].get(key) is not None and math.isfinite(record["metrics"][key])]
        for index, (record, style) in enumerate(zip(records, styles)):
            value = record["metrics"].get(key)
            if value is None or not math.isfinite(value):
                ax.text(index, 0.03, "N/A", transform=ax.get_xaxis_transform(), ha="center", fontsize=8)
                continue
            bars = ax.bar(index, value, width=0.62, **_bar_style(style))
            _annotate_bars(ax, bars, [value])
        ax.set(title=title, ylabel=ylabel, xticks=[], xlim=(-0.6, len(records) - 0.4),
               ylim=(0, max(known, default=0) * 1.18 or 1))
        if not known:
            ax.set_yticks([])
            _empty(ax, "No recorded values")
        ax.grid(axis="y")
    _legend(fig, _method_handles(records))
    artifacts["compute"] = _finish(fig, output, "compute")
    return artifacts
