"""Portable galleries rendered from saved selected designs and CPU evaluations."""

from html import escape
import json
from pathlib import Path
import re

from .constraint_text import constraint_description
from .types import overall_validity, write_json
from .visualize import save_sample


def _scored(check):
    return check.get("role") != "supporting_rule" and check.get("counts_toward_compliance", True)


def _trial_seed(row, summary):
    seed = row["sample"].get("metadata", {}).get("trial_seed", row.get("trial_seed"))
    if seed is None and summary.get("trial_seed_start") is not None:
        seed = summary["trial_seed_start"] + row["trial"]
    return seed


def _options(values, labels=None):
    labels = labels or {}
    return "".join(f'<option value="{escape(str(value), quote=True)}">{escape(str(labels.get(value, value)))}</option>'
                   for value in values)


def _details(checks, diagnostics, sample):
    parts = ['<details><summary>Constraint results and reasons</summary>']
    if diagnostics:
        parts.append('<p><strong>Structural diagnostics:</strong> ' +
                     escape("; ".join(dict.fromkeys(diagnostics))) + '</p>')
    for check in checks:
        description = constraint_description(check, sample)
        status = {"unknown": "UNRESOLVED", "not_applicable": "N/A"}.get(check["status"], check["status"].upper())
        parts.append('<section class="constraint-result">'
                     f'<h3>{escape(check["id"])} · {escape(description["name"])} '
                     f'<span class="status {escape(check["status"], quote=True)}">{escape(status)}</span></h3>'
                     f'<p><strong>Requirement:</strong> {escape(description["requirement"])}</p>'
                     f'<p><strong>Reason:</strong> {escape(description["reason"])}</p>'
                     '<details class="measurements"><summary>Recorded measurements</summary><pre>' +
                     escape(json.dumps(check, indent=2, ensure_ascii=False)) + '</pre></details></section>')
    parts.append('</details>')
    return "\n".join(parts)


def build_gallery(records, output_dir):
    """Write an image and saved-data bundle for every selected row, without inference.

    ``records`` contain ``label``, ``run_dir``, selected candidate ``rows`` and
    an existing run ``summary``. Explicitly unselected rows are ignored.
    The returned ``examples[label][status]`` contains the first example in
    stable run/trial order (or None) for valid, invalid and unknown outcomes.
    Artifact paths in examples are relative to ``output_dir`` for Markdown use;
    ``seed`` is the paired trial seed, while ``candidate_seed`` identifies a
    Best-of-N winner. Source files and saved evaluation values are not changed.
    """
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    cards, examples, constraint_ids, labels = [], {}, set(), set()
    records = sorted(records, key=lambda item: (item["label"], str(item["run_dir"])))
    for record_index, record in enumerate(records, 1):
        label, summary = str(record["label"]), record["summary"]
        labels.add(label)
        examples.setdefault(label, {status: None for status in ("valid", "invalid", "unknown")})
        slug = re.sub(r"[^A-Za-z0-9_-]+", "_", label).strip("_-")[:64] or "method"
        directory = output / "samples" / f"{record_index:02d}_{slug}"
        rows = sorted((row for row in record["rows"] if row.get("selected", True)),
                      key=lambda row: (row["trial"], str(row.get("candidate_id", ""))))
        for row_index, row in enumerate(rows):
            trial, evaluation = row["trial"], row["evaluation"]
            checks = [check for check in evaluation["results"] if _scored(check)]
            # Supporting definitions never make a sample fail, including legacy
            # results whose role is explicit but whose denominator flag is absent.
            status = overall_validity({**evaluation, "results": checks})
            failed = [check["id"] for check in checks if check["status"] == "fail" and check["applicable"]]
            unknown = [check["id"] for check in checks if check["status"] == "unknown" and check["applicable"]]
            constraint_ids.update(check["id"] for check in checks)
            seed = _trial_seed(row, summary)
            prefix = directory / f"trial-{trial:04d}-sample-{row_index:04d}"
            rendered_evaluation = {**evaluation, "results": [
                {**check, "counts_toward_compliance": _scored(check)} for check in evaluation["results"]]}
            paths = save_sample(row["sample"], rendered_evaluation, prefix)
            graph_path = prefix.with_suffix(".json")
            evaluation_path = prefix.with_name(prefix.name + "-evaluation.json")
            write_json(graph_path, row["sample"])
            write_json(evaluation_path, evaluation)
            paths.update(graph=str(graph_path), evaluation=str(evaluation_path))
            relative = {kind: Path(path).relative_to(output).as_posix() for kind, path in paths.items()}
            item = {**relative, "trial": trial, "seed": seed, "candidate_seed": row.get("seed"),
                    "failed_ids": failed, "unknown_ids": unknown}
            if status in examples[label] and examples[label][status] is None:
                examples[label][status] = item
            check_states = json.dumps({check["id"]: check["status"] for check in checks})
            seed_text = "unrecorded" if seed is None else str(seed)
            links = " · ".join(f'<a href="{escape(relative[kind], quote=True)}">{name}</a>'
                               for kind, name in (("png", "Full PNG"), ("svg", "SVG"),
                                                  ("graph", "Graph JSON"), ("evaluation", "Evaluation JSON")))
            diagnostics = list(evaluation.get("diagnostics", [])) + list(row["sample"].get("diagnostics", []))
            status_text = {"unknown": "UNRESOLVED", "not_applicable": "N/A"}.get(status, status.upper())
            cards.append(
                f'<article class="sample" data-method="{escape(label, quote=True)}" '
                f'data-status="{status}" data-checks="{escape(check_states, quote=True)}">\n'
                f'<h2>{escape(label)} · trial {trial + 1}</h2>\n'
                f'<p>Trial seed: <strong>{escape(seed_text)}</strong> · '
                f'Candidate seed: {escape(str(row.get("seed", "unrecorded")))}</p>\n'
                f'<p class="status {status}">{status_text} · '
                f'Structure: {"valid" if evaluation["structurally_valid"] else "invalid"}</p>\n'
                f'<p>Failed: {escape(", ".join(failed) or "none")} · '
                f'Unresolved: {escape(", ".join(unknown) or "none")}</p>\n'
                f'<a href="{escape(relative["png"], quote=True)}"><img loading="lazy" '
                f'src="{escape(relative["png"], quote=True)}" '
                f'alt="{escape(label, quote=True)} trial {trial + 1} annotated floorplan"></a>\n'
                f'<p class="links">{links}</p>\n{_details(checks, diagnostics, row["sample"])}\n</article>')

    page = ['<!doctype html><html lang="en"><head><meta charset="utf-8">',
            '<meta name="viewport" content="width=device-width, initial-scale=1">',
            '<title>GSDiff selected-design gallery</title>', _STYLE, '</head><body><main>',
            '<h1>Selected-design gallery</h1>',
            '<p>Every selected output is included. Labels and annotations use the saved final CPU reference '
            'evaluation; no generation or constraint checking is repeated. Unresolved checks are not passes. '
            'Supporting measurement rules are excluded from the constraint filters.</p>',
            '<p>Designs are shown in stable method, run and trial order. Click an image for its full-size annotation; '
            'open “Constraint results and reasons” for plain-language requirements and explanations. '
            'Raw check data is available under “Recorded measurements”.</p>',
            '<div class="filters">',
            '<label>Method<select id="method"><option value="">All methods</option>' + _options(sorted(labels)) + '</select></label>',
            '<label>Overall status<select id="status"><option value="">All statuses</option>' +
            _options(("valid", "invalid", "unknown", "not_applicable"),
                     {"valid": "Valid", "invalid": "Invalid", "unknown": "Unresolved", "not_applicable": "N/A"}) + '</select></label>',
            '<label>Constraint<select id="constraint"><option value="">Any constraint</option>' +
            _options(sorted(constraint_ids)) + '</select></label>',
            '<label>Constraint status<select id="check-status"><option value="">Any result</option>' +
            _options(("pass", "fail", "unknown", "not_applicable"),
                     {"pass": "Pass", "fail": "Fail", "unknown": "Unresolved", "not_applicable": "N/A"}) + '</select></label>',
            '</div><p id="count" aria-live="polite">' + str(len(cards)) + ' selected designs</p>',
            '<section class="gallery">', *cards, '</section></main>', _SCRIPT, '</body></html>']
    index = output / "index.html"
    index.write_text("\n".join(page), encoding="utf-8")
    return {"html": str(index), "sample_count": len(cards), "examples": examples}


_STYLE = """<style>
body{margin:0;background:#f4f5f7;color:#202832;font:15px/1.5 system-ui,sans-serif}
main{max-width:1450px;margin:auto;padding:24px}h1{margin-top:0}h2{font-size:18px;margin:0}
.filters{display:flex;gap:16px;flex-wrap:wrap;padding:16px;background:#fff;border-radius:8px}
label{display:flex;flex-direction:column;gap:5px}select{font:inherit;padding:7px;max-width:350px}
.gallery{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,380px),1fr));gap:20px}
.sample{background:#fff;border:1px solid #d6dce2;border-radius:8px;padding:16px;min-width:0}
.sample[hidden]{display:none}.sample p{margin:8px 0}.sample img{width:100%;height:290px;object-fit:contain}
a{color:#175b9a}.status{font-weight:bold}.valid,.pass{color:#167343}.invalid,.fail{color:#b42318}.unknown{color:#946200}
.constraint-result{margin-top:16px;border-top:1px solid #e1e5e9;padding-top:10px}
.constraint-result h3{font-size:15px;margin:0}.constraint-result h3 .status{display:block;font-size:13px}
.measurements{font-size:13px}
details{margin-top:14px}summary{cursor:pointer}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px;
background:#f4f5f7;padding:10px;max-height:350px;overflow:auto}.links{font-size:14px}
</style>"""


_SCRIPT = """<script>
const controls = ['method', 'status', 'constraint', 'check-status'].map(id => document.getElementById(id));
const cards = [...document.querySelectorAll('.sample')].map(element => ({element, checks: JSON.parse(element.dataset.checks)}));
function filterSamples() {
  const [method, status, constraint, checkStatus] = controls.map(control => control.value);
  let visible = 0;
  for (const {element, checks} of cards) {
    const outcomes = constraint ? (Object.hasOwn(checks, constraint) ? [checks[constraint]] : []) : Object.values(checks);
    const show = (!method || element.dataset.method === method) && (!status || element.dataset.status === status)
      && (!constraint || outcomes.length > 0) && (!checkStatus || outcomes.includes(checkStatus));
    element.hidden = !show;
    visible += Number(show);
  }
  document.getElementById('count').textContent = `${visible} of ${cards.length} selected designs`;
}
controls.forEach(control => control.addEventListener('change', filterSamples));
filterSamples();
</script>"""
