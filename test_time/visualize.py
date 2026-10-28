"""PNG/SVG floorplans annotated with the shared evaluator's actual results."""

from dataclasses import asdict, is_dataclass
from html import escape
import json
import math
from pathlib import Path
import textwrap

from PIL import Image, ImageDraw, ImageFont

from .types import overall_validity


COLORS = {0: "#f4f1de", 1: "#eab69f", 2: "#6b705c", 3: "#e07a5f",
          4: "#5f797b", 5: "#f2cc8f", 6: "#222222"}
STATUS_COLORS = {"pass": "#167343", "fail": "#b42318", "unknown": "#946200", "not_applicable": "#666666"}


def compact(value, limit=210):
    text = json.dumps(value, ensure_ascii=True, separators=(", ", ": "))
    return text if len(text) <= limit else text[:limit - 3] + "..."


def save_sample(sample, evaluation, output_prefix):
    sample = asdict(sample) if is_dataclass(sample) else sample
    evaluation = asdict(evaluation) if is_dataclass(evaluation) else evaluation
    prefix = Path(output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    text_lines = []
    for result in evaluation["results"]:
        auxiliary = " [auxiliary]" if not result.get("counts_toward_compliance", True) else ""
        text_lines.append((f"{result['id']}: {result['status']}{auxiliary}", STATUS_COLORS[result["status"]]))
        for label, values in (("Observed", result.get("observed_values", {})),
                              ("Thresholds", result.get("threshold_values", {}))):
            text_lines.extend((line, "#333333") for line in textwrap.wrap(f"{label}: {compact(values)}", 76))
        text_lines.extend((line, "#555555") for line in textwrap.wrap(result.get("explanation", "")[:220], 76))
        text_lines.append(("", "#333333"))
    width, height = 1120, max(640, 100 + 16 * len(text_lines))
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 12)
        title_font = ImageFont.truetype("DejaVuSans.ttf", 16)
        bitmap_font = False
    except OSError:
        font = title_font = ImageFont.load_default()
        bitmap_font = True
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
           '<rect width="100%" height="100%" fill="white"/>']

    def text(x, y, value, color="#222222", title=False):
        # Minimal CUDA images may only have Pillow's Latin-1 bitmap font.
        raster_text = value.encode("latin-1", "replace").decode("latin-1") if bitmap_font else value
        draw.text((x, y), raster_text, fill=color, font=title_font if title else font)
        svg.append(f'<text x="{x}" y="{y + (16 if title else 12)}" fill="{color}" font-family="sans-serif" '
                   f'font-size="{16 if title else 12}">{escape(value)}</text>')

    status = overall_validity(evaluation)
    structural = "valid" if evaluation["structurally_valid"] else "INVALID"
    color = {"valid": "#167343", "invalid": "#b42318", "unknown": "#946200", "not_applicable": "#666666"}[status]
    text(20, 16, f"GSDiff | overall: {status.upper()} | structure: {structural} | energy={evaluation['energy']:.4g}",
         color, title=True)
    scenario = sample.get("scenario", {})
    provenance = scenario.get("scale_provenance", "missing")
    scale = scenario.get("coordinate_scale_m_per_model_unit")
    text(20, 44, f"Scale: {provenance}, {scale} m/drawing pixel. Geometry/scenario checks; not verified building compliance.")
    finite_points = [p[:2] for p in sample["nodes"] if len(p) >= 2 and all(math.isfinite(x) for x in p[:2])]
    if finite_points:
        xmin, xmax = min(p[0] for p in finite_points), max(p[0] for p in finite_points)
        ymin, ymax = min(p[1] for p in finite_points), max(p[1] for p in finite_points)
        factor = 460 / max(xmax - xmin, ymax - ymin, 1)
    else:
        xmin, ymin, factor = 0, 0, 1

    def xy(point):
        return (40 + (point[0] - xmin) * factor, 95 + (point[1] - ymin) * factor)

    for room in sample["rooms"]:
        points = room["polygon"]
        if len(points) < 3 or not all(len(p) >= 2 and all(math.isfinite(x) for x in p[:2]) for p in points):
            continue
        transformed = [xy(p) for p in points]
        color = COLORS.get(room["label"], "#dddddd")
        draw.polygon(transformed, fill=color, outline="#888888")
        vertices = " ".join(f"{x:.2f},{y:.2f}" for x, y in transformed)
        svg.append(f'<polygon points="{vertices}" fill="{color}" stroke="#888888" stroke-width="2"/>')
    nodes = sample["nodes"]
    for edge in sample["edges"]:
        if len(edge) != 2 or any(not isinstance(i, int) or not 0 <= i < len(nodes) for i in edge):
            continue
        a, b = (nodes[i] for i in edge)
        if any(len(p) < 2 or not all(math.isfinite(x) for x in p[:2]) for p in (a, b)):
            continue
        p, q = xy(a), xy(b)
        draw.line([p, q], fill="#777777", width=3)
        svg.append(f'<line x1="{p[0]:.2f}" y1="{p[1]:.2f}" x2="{q[0]:.2f}" y2="{q[1]:.2f}" stroke="#777777" stroke-width="3"/>')
    if not sample["rooms"]:
        text(40, 110, "No extracted rooms", "#b42318")
    text(20, 578, f"{len(nodes)} junctions | {len(sample['edges'])} edges | {len(sample['rooms'])} extracted rooms")
    diagnostics = evaluation.get("diagnostics", []) + sample.get("diagnostics", [])
    for index, line in enumerate(textwrap.wrap("; ".join(diagnostics)[:230], 78)[:3]):
        text(20, 598 + 14 * index, line, "#b42318")
    for index, (line, color) in enumerate(text_lines):
        text(540, 84 + 16 * index, line, color)
    svg.append("</svg>")
    png_path, svg_path = prefix.with_suffix(".png"), prefix.with_suffix(".svg")
    image.save(png_path)
    svg_path.write_text("\n".join(svg) + "\n")
    return {"png": str(png_path), "svg": str(svg_path)}
