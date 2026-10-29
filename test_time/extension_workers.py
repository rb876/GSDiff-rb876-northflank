"""Combine independent extension workers without hiding unsuccessful attempts."""

from copy import deepcopy
import json
from pathlib import Path

from .types import read_json, write_json


def _relative_artifacts(paths, directory):
    result = {}
    for name, value in paths.items():
        if isinstance(value, dict):
            result[name] = _relative_artifacts(value, directory)
        else:
            path = Path(value)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError(f"Worker artifact must be relative: {value}")
            result[name] = str(Path(directory) / path)
    return result


def summarize_workers(output, method):
    """Write a combined ledger and report; original worker artifacts stay in place.

    Episode durations are summed for accounting. Elapsed wall time is supplied by
    the process supervisor and includes overlapping workers, loading and exports.
    """
    from .run_extensions import number, write_summary

    output = Path(output)
    manifest = read_json(output / "parallel.json")
    if manifest["method"] != method:
        raise ValueError("Parallel manifest method does not match requested method")
    rows, workers, configs, issues = [], [], [], []
    seen = set()
    reference = None
    for worker in manifest["workers"]:
        directory = worker["directory"]
        if Path(directory).name != directory or not directory.startswith("worker_"):
            raise ValueError(f"Invalid worker directory: {directory}")
        folder = output / directory
        config = None
        if (folder / "config.json").exists():
            try:
                config = read_json(folder / "config.json")
            except json.JSONDecodeError:
                issues.append(f"{directory}/config.json is incomplete")
        if config is not None:
            # Everything except the assigned seed range/output location must agree.
            signature = {key: value for key, value in config.items()
                         if key not in {"seed", "trials", "output_root", "workers"}}
            if config.get("method") != method or (reference is not None and signature != reference):
                raise ValueError(f"Incompatible policy, model or settings in {directory}")
            reference = signature
            configs.append(config)
            if config["seed"] != worker["seed"] or config["trials"] != worker["trials"]:
                raise ValueError(f"Worker seed range does not match manifest: {directory}")
        state = {}
        if (folder / "run.json").exists():
            try:
                state = read_json(folder / "run.json")
            except json.JSONDecodeError:
                issues.append(f"{directory}/run.json is incomplete")
        path = folder / "trials.jsonl"
        lines = path.read_text().splitlines() if path.exists() else []
        count = 0
        for index, line in enumerate(lines):
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                if index != len(lines) - 1 or manifest["status"] == "complete":
                    raise ValueError(f"Invalid completed-attempt ledger: {path}") from None
                issues.append(f"Ignored incomplete final row in {directory}/trials.jsonl")
                break
            if config is None:
                raise ValueError(f"Missing worker configuration for recorded attempts: {directory}")
            seed = row["seed"]
            if seed in seen:
                raise ValueError(f"Duplicate trial seed across workers: {seed}")
            if not worker["seed"] <= seed < worker["seed"] + worker["trials"]:
                raise ValueError(f"Trial seed outside worker assignment: {seed}")
            if row["method"] != method or row.get("policy_sha256") != config.get("policy_sha256"):
                raise ValueError(f"Trial policy or method differs from worker configuration: {directory}")
            seen.add(seed)
            row = deepcopy(row)
            row.update(worker=directory, worker_trial=row["trial"], trial=seed - manifest["seed"])
            for key in ("artifacts", "extra_artifacts"):
                row[key] = _relative_artifacts(row.get(key, {}), directory)
            rows.append(row)
            count += 1
        complete = (state.get("status") == "complete" and count == worker["trials"]
                    and state.get("completed_trials") == count)
        workers.append({**worker, "status": state.get("status", "missing"),
                        "completed_trials": count, "complete": complete})

    rows.sort(key=lambda row: row["seed"])
    planned = manifest["planned_trials"]
    if sum(worker["trials"] for worker in workers) != planned:
        raise ValueError("Worker assignments do not match planned trial count")
    status = manifest["status"]
    if status == "complete" and (len(rows) != planned or issues or not all(w["complete"] for w in workers)):
        status = "incomplete"
    elapsed = manifest.get("elapsed_seconds")
    summary = write_summary(output, rows, method)
    summary.update(worker_count=len(workers), parallel_elapsed_seconds=elapsed,
                   attempts_per_elapsed_second=(len(rows) / elapsed if elapsed and elapsed > 0 else None),
                   elapsed_timing_scope="supervisor wall time including loading, exports and overlapping workers",
                   episode_timing_scope="sum of worker episode durations; not elapsed parallel runtime")
    write_json(output / "summary.json", summary)
    ledger = output / "trials.jsonl"
    temporary = ledger.with_suffix(".jsonl.tmp")
    temporary.write_text("".join(json.dumps(row, allow_nan=False) + "\n" for row in rows))
    temporary.replace(ledger)
    config = deepcopy(configs[0]) if configs else {"method": method}
    config.update(format="extension_parallel_attempt_ledger_v1", seed=manifest["seed"], trials=planned,
                  parallel_workers=[{"directory": w["directory"], "seed": w["seed"], "trials": w["trials"]}
                                    for w in workers])
    write_json(output / "config.json", config)
    state = {"method": method, "status": status, "planned_trials": planned,
             "completed_trials": len(rows), "end_to_end_seconds": elapsed,
             "workers": workers, "issues": issues, "summary": summary}
    write_json(output / "run.json", state)
    report_path = output / "report.md"
    report = report_path.read_text()
    for name in ("candidates.jsonl", "events.jsonl", "lookaheads/", "samples/", "traces/"):
        report = report.replace(f"`{name}`", f"`worker_*/{name}`")
    report = report.replace("`worker_*/candidates.jsonl` contains", "`worker_*/candidates.jsonl` contain")
    report += ("\n## Parallel execution\n\n"
               f"Status: **{status}**; completed **{len(rows)}/{planned}** planned attempts. "
               f"Supervisor elapsed time: **{number(elapsed)} seconds**, including loading and artifact export. "
               "The seconds/attempt table sums worker episode durations; it is not elapsed parallel runtime "
               "and should not be interpreted as throughput. Concurrent execution can change individual timings.\n\n"
               "The combined `trials.jsonl` includes failures and points to the original worker artifacts. "
               "Raw candidate/event ledgers, traces and images remain in the worker directories.\n\n"
               "| Worker | Assigned seeds | Status | Completed |\n|---|---|---|---:|\n")
    for worker in workers:
        name = worker["directory"]
        link = f"[{name}]({name}/report.md)" if (output / name / "report.md").exists() else name
        report += (f"| {link} | {worker['seed']}–{worker['seed'] + worker['trials'] - 1} | "
                   f"{worker['status']} | {worker['completed_trials']}/{worker['trials']} |\n")
    if issues:
        report += "\n" + "\n".join(f"- {issue}" for issue in issues) + "\n"
    report_path.write_text(report)
    return output
