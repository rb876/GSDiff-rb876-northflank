"""Readable PNG/SVG floorplans from saved room labels and constraint outcomes."""

from dataclasses import asdict, is_dataclass
from html import escape
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from shapely.geometry import Polygon
from shapely.ops import polylabel

from .constraint_text import constraint_description, room_labels
from .types import overall_validity


COLORS = {0: "#f4f1de", 1: "#eab69f", 2: "#6b705c", 3: "#e07a5f",
          4: "#5f797b", 5: "#f2cc8f", 6: "#222222"}
STATUS_COLORS = {"pass": "#167343", "fail": "#b42318", "unknown": "#946200", "not_applicable": "#666666"}
STATUS_NAMES = {"pass": "PASS", "fail": "FAIL", "unknown": "UNRESOLVED", "not_applicable": "N/A"}
STATUS_FILLS = {"pass": "#eef7f1", "fail": "#fff0ed", "unknown": "#fff8e6", "not_applicable": "#f2f3f5"}


def save_sample(sample, evaluation, output_prefix):
    sample = asdict(sample) if is_dataclass(sample) else sample
    evaluation = asdict(evaluation) if is_dataclass(evaluation) else evaluation
    prefix = Path(output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    fonts, bitmap = {}, set()
    sizes = {"title": 23, "heading": 17, "body": 15, "small": 13, "badge": 15}
    for key, size in sizes.items():
        bold = key in {"title", "heading", "badge"}
        try:
            fonts[key] = ImageFont.truetype("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf", size)
        except OSError:
            fonts[key] = ImageFont.load_default()
            bitmap.add(key)
    measure = ImageDraw.Draw(Image.new("RGB", (1, 1)))

    def safe(value, font):
        return value.encode("latin-1", "replace").decode("latin-1") if font in bitmap else value

    def wrap(value, width, font="body"):
        lines, line = [], ""
        for word in str(value).split():
            candidate = (line + " " + word).strip()
            if line and measure.textlength(safe(candidate, font), font=fonts[font]) > width:
                lines.append(line)
                line = word
            else:
                line = candidate
        return lines + ([line] if line else [])

    scored = [check for check in evaluation["results"]
              if check.get("role") != "supporting_rule" and check.get("counts_toward_compliance", True)]
    supporting = [check["id"] for check in evaluation["results"] if check not in scored]
    order = {"fail": 0, "unknown": 1, "pass": 2, "not_applicable": 3}
    cards = []
    for check in sorted(scored, key=lambda value: (order[value["status"]], value["id"])):
        description = constraint_description(check, sample)
        title = wrap(f"{check['id']}  {description['name']}", 570, "heading")
        requirement = wrap("Requirement: " + description["requirement"], 730, "small")
        reason = wrap("Why: " + description["reason"], 730)
        height = 34 + 23 * len(title) + 18 * len(requirement) + 21 * len(reason)
        cards.append((check, title, requirement, reason, height))

    labels = room_labels(sample)
    scenario = sample.get("scenario", {})
    scale = scenario.get("coordinate_scale_m_per_model_unit")
    scale_ok = isinstance(scale, (int, float)) and math.isfinite(scale) and scale > 0
    provenance = scenario.get("scale_provenance", "missing")
    supporting_text = ("Supporting definitions: " + ", ".join(supporting) + ". These supply parameters; they are not scored checks."
                       if supporting else "")
    footer = ["Room types are the model's saved semantic labels. Metric dimensions use the declared scale; other furniture and door swings are unobserved."]
    if supporting_text:
        footer.append(supporting_text)
    footer.append("These geometry checks assess the stated assumptions, not verified building compliance. Full measurements remain in the evaluation JSON.")
    footer_lines = [line for item in footer for line in wrap(item, 1376, "small")]
    diagnostics = list(dict.fromkeys(evaluation.get("diagnostics", []) + sample.get("diagnostics", [])))
    diagnostic_lines = wrap("Structure: " + "; ".join(diagnostics), 540, "small") if diagnostics else []
    left_bottom = 740 + 29 * math.ceil(len(labels) / 2) + 18 * len(diagnostic_lines)
    right_bottom = 140 + sum(card[-1] + 10 for card in cards)
    content_bottom = max(left_bottom, right_bottom, 830)
    width, height = 1440, content_bottom + 32 + 19 * len(footer_lines) + 24
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
           '<rect width="100%" height="100%" fill="white"/>']

    def rect(x, y, w, h, fill, outline=None, radius=0):
        draw.rounded_rectangle((x, y, x + w, y + h), radius=radius, fill=fill, outline=outline)
        svg.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" '
                   f'fill="{fill}" stroke="{outline or "none"}"/>')

    def text(x, y, value, color="#263238", font="body"):
        draw.text((x, y), safe(value, font), fill=color, font=fonts[font])
        weight = "bold" if font in {"title", "heading", "badge"} else "normal"
        svg.append(f'<text x="{x}" y="{y + sizes[font]}" fill="{color}" font-family="DejaVu Sans, sans-serif" '
                   f'font-size="{sizes[font]}" font-weight="{weight}">{escape(value)}</text>')

    status = overall_validity({**evaluation, "results": scored})
    status_text = "UNRESOLVED" if status == "unknown" else status.upper()
    structural = "valid" if evaluation["structurally_valid"] else "INVALID"
    color = {"valid": "#167343", "invalid": "#b42318", "unknown": "#946200", "not_applicable": "#666666"}[status]
    text(32, 24, f"GSDiff | overall: {status_text} | structure: {structural}", color, "title")
    summary = "  |  ".join(f"{sum(check['status'] == key for check in scored)} {name}"
                            for key, name in (("pass", "passed"), ("fail", "failed"), ("unknown", "unresolved")))
    text(32, 63, summary + f"  |  Energy: {evaluation['energy']:.3g}")
    scale_text = f"{provenance} scale: {scale:g} m/drawing pixel" if scale_ok else "Metric scale unavailable"
    text(32, 91, f"{scale_text}. Room areas below use this scale.", "#56616b", "small")
    text(32, 122, "Room types and areas", font="heading")
    text(632, 112, "Constraint results", font="heading")

    # Use a fixed-size, centred viewport; render saved geometry without repairing it.
    points = [point[:2] for point in sample.get("nodes", [])]
    points += [point[:2] for room in sample.get("rooms", []) for point in room["polygon"]]
    points = [point for point in points if len(point) == 2 and all(math.isfinite(v) for v in point)]
    if points:
        xmin, xmax = min(p[0] for p in points), max(p[0] for p in points)
        ymin, ymax = min(p[1] for p in points), max(p[1] for p in points)
        factor = min(520 / max(xmax - xmin, 1), 500 / max(ymax - ymin, 1))
        ox, oy = 52 + (520 - (xmax - xmin) * factor) / 2, 164 + (500 - (ymax - ymin) * factor) / 2
    else:
        xmin, ymin, factor, ox, oy = 0, 0, 1, 52, 164

    def xy(point):
        return ox + (point[0] - xmin) * factor, oy + (point[1] - ymin) * factor

    rect(32, 150, 560, 530, "#ffffff", "#e1e5e9", 6)
    room_shapes, room_areas = {}, {}
    for index, room in enumerate(sample.get("rooms", [])):
        points = room["polygon"]
        if len(points) < 3 or not all(len(p) >= 2 and all(math.isfinite(v) for v in p[:2]) for p in points):
            continue
        transformed = [xy(point) for point in points]
        fill = COLORS.get(room["label"], "#dddddd")
        draw.polygon(transformed, fill=fill, outline="#777777")
        vertices = " ".join(f"{x:.3f},{y:.3f}" for x, y in transformed)
        svg.append(f'<polygon points="{vertices}" fill="{fill}" stroke="#777777" stroke-width="2"/>')
        polygon = Polygon([point[:2] for point in points])
        if polygon.is_valid and not polygon.is_empty and polygon.area > 0:
            room_shapes[index] = polygon
            if scale_ok:
                room_areas[index] = polygon.area * scale * scale
    nodes = sample.get("nodes", [])
    for edge in sample.get("edges", []):
        if len(edge) != 2 or any(not isinstance(i, int) or not 0 <= i < len(nodes) for i in edge):
            continue
        a, b = (nodes[index] for index in edge)
        if any(len(p) < 2 or not all(math.isfinite(v) for v in p[:2]) for p in (a, b)):
            continue
        p, q = xy(a), xy(b)
        draw.line((p, q), fill="#70777c", width=2)
        svg.append(f'<line x1="{p[0]:.3f}" y1="{p[1]:.3f}" x2="{q[0]:.3f}" y2="{q[1]:.3f}" stroke="#70777c" stroke-width="2"/>')
    for index, polygon in room_shapes.items():
        anchor = polylabel(polygon, tolerance=max(0.25, 1 / factor))
        x, y = xy((anchor.x, anchor.y))
        space = max(25, 1.8 * anchor.distance(polygon.boundary) * factor)
        lines = wrap(labels[index], space, "small")
        if index in room_areas:
            lines.append(f"{room_areas[index]:.2f} m²")
        if max((measure.textlength(safe(line, "small"), font=fonts["small"]) for line in lines), default=0) > space or len(lines) * 17 > space:
            lines = [f"R{index + 1}"]
        for line_index, line in enumerate(lines):
            raster_text = safe(line, "small")
            length = measure.textlength(raster_text, font=fonts["small"])
            tx, ty = x - length / 2, y - len(lines) * 17 / 2 + line_index * 17
            draw.text((tx, ty), raster_text, font=fonts["small"], fill="#20262c", stroke_width=2, stroke_fill="white")
            svg.append(f'<text x="{x:.3f}" y="{ty + 13:.3f}" text-anchor="middle" fill="#20262c" '
                       f'font-family="DejaVu Sans, sans-serif" font-size="13" stroke="white" stroke-width="3" '
                       f'paint-order="stroke fill" stroke-linejoin="round">{escape(line)}</text>')
    if not room_shapes:
        text(52, 180, "No usable room polygons to label", "#b42318")
    text(32, 696, "Room key (R numbers identify small rooms)", font="heading")
    for index, label in labels.items():
        x, y = 32 + (index % 2) * 286, 731 + (index // 2) * 29
        room = sample["rooms"][index]
        rect(x, y + 3, 13, 13, COLORS.get(room["label"], "#dddddd"), "#999999")
        area = f" · {room_areas[index]:.2f} m²" if index in room_areas else ""
        text(x + 21, y, f"R{index + 1} {label}{area}", font="small")
    for index, line in enumerate(diagnostic_lines):
        text(32, 740 + 29 * math.ceil(len(labels) / 2) + 18 * index, line, "#b42318", "small")

    y = 140
    for check, title, requirement, reason, card_height in cards:
        current = check["status"]
        rect(632, y, 776, card_height, "#ffffff", "#dce2e7", 7)
        rect(632, y + 7, 4, card_height - 14, STATUS_COLORS[current])
        rect(1256, y + 13, 134, 28, STATUS_FILLS[current], radius=5)
        text(1268, y + 17, STATUS_NAMES[current], STATUS_COLORS[current], "badge")
        row_y = y + 14
        for line in title:
            text(650, row_y, line, font="heading")
            row_y += 23
        row_y += 5
        for line in requirement:
            text(650, row_y, line, "#56616b", "small")
            row_y += 18
        row_y += 4
        for line in reason:
            text(650, row_y, line)
            row_y += 21
        y += card_height + 10
    if not cards:
        text(650, 155, "No scored checks recorded", "#666666")
    draw.line((32, content_bottom + 12, 1408, content_bottom + 12), fill="#dce2e7", width=1)
    svg.append(f'<line x1="32" y1="{content_bottom + 12}" x2="1408" y2="{content_bottom + 12}" stroke="#dce2e7"/>')
    for index, line in enumerate(footer_lines):
        text(32, content_bottom + 26 + index * 19, line, "#56616b", "small")
    svg.append("</svg>")
    png_path, svg_path = prefix.with_suffix(".png"), prefix.with_suffix(".svg")
    image.save(png_path)
    svg_path.write_text("\n".join(svg) + "\n")
    return {"png": str(png_path), "svg": str(svg_path)}
