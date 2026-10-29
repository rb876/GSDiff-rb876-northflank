"""Run disjoint extension trial ranges in independent Python/CUDA processes."""

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from .types import write_json


def _worker_command(args, config, worker, output):
    command = [sys.executable, "-u", "-m", "test_time.run_extensions", "--workers", "1",
               "--method", args.method, "--config", str(args.config.resolve()),
               "--device", args.device, "--trials", str(worker["trials"]),
               "--seed", str(worker["seed"]),
               "--inference_batch_size", str(config["inference_batch_size"]),
               "--output_dir", str(output / worker["directory"])]
    for name in ("k", "rollouts", "terminal_k", "max_nfe", "node_checkpoint", "edge_checkpoint"):
        value = getattr(args, name, None)
        if value is not None:
            flag = "--" + (name.replace("_", "-") if name.endswith("checkpoint") else name)
            command.extend((flag, str(value)))
    if args.no_progress:
        command.append("--no-progress")
    return command


def _stop_workers(processes):
    # Workers own their process groups: stop any child processes as well.
    for process in processes:
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    for process in processes:
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()


def run_parallel(args):
    """Preserve per-seed algorithms while overlapping separate trial processes.

    Artifacts remain inside worker directories. ``parallel.json`` records the
    launch; the caller aggregates worker ledgers after completion or failure.
    """
    from .run_extensions import load_configuration

    if type(args.workers) is not int or args.workers < 1:
        raise ValueError("workers must be a positive integer")
    config, _, _, _ = load_configuration(args)
    repo = Path(__file__).resolve().parents[1]
    output = (args.output_dir or repo / config["output_root"] / args.method).resolve()
    output.mkdir(parents=True, exist_ok=False)
    args.parallel_output = output
    count = min(args.workers, config["trials"])
    base, extra = divmod(config["trials"], count)
    workers, seed = [], config["seed"]
    for index in range(count):
        trials = base + (index < extra)
        workers.append({"directory": f"worker_{index + 1:03d}", "seed": seed,
                        "trials": trials, "completed_trials": 0})
        seed += trials
    state = {"format": "parallel_extension_run_v1", "method": args.method,
             "status": "running", "planned_trials": config["trials"],
             "seed": config["seed"], "completed_trials": 0,
             "requested_workers": args.workers, "worker_count": count,
             "workers": workers, "config_path": str(args.config.resolve()),
             "device": args.device,
             "timing_scope": "elapsed launch time including worker loading and artifact export"}
    environment = os.environ.copy()
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        environment.setdefault(name, "1")
    state["worker_cpu_threads"] = {name: environment[name] for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS")}
    started = time.perf_counter()
    processes, logs = [], []

    def save_progress():
        for worker, process in zip(workers, processes):
            worker["returncode"] = process.poll()
            try:
                child = json.loads((output / worker["directory"] / "run.json").read_text())
                worker["completed_trials"] = child.get("completed_trials", 0)
            except (OSError, json.JSONDecodeError):
                pass  # The child's initial/updated manifest may not be written yet.
        state["completed_trials"] = sum(worker["completed_trials"] for worker in workers)
        state["elapsed_seconds"] = time.perf_counter() - started
        temporary = output / "parallel.json.tmp"
        write_json(temporary, state)
        temporary.replace(output / "parallel.json")

    def terminate(signum, frame):
        raise SystemExit(128 + signum)

    handlers = {signum: signal.getsignal(signum) for signum in (signal.SIGINT, signal.SIGTERM)}
    for signum in handlers:
        signal.signal(signum, terminate)
    try:
        save_progress()
        for worker in workers:
            log_path = output / f"{worker['directory']}.log"
            log = log_path.open("w")
            logs.append(log)
            process = subprocess.Popen(_worker_command(args, config, worker, output), cwd=repo,
                                       env=environment, stdin=subprocess.DEVNULL,
                                       stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            processes.append(process)
            worker["pid"] = process.pid
            print(f"{args.method} {worker['directory']}: seeds {worker['seed']}.."
                  f"{worker['seed'] + worker['trials'] - 1}; log: {log_path}", flush=True)
        next_report = 0.0
        while True:
            codes = [process.poll() for process in processes]
            failures = [(worker, code) for worker, code in zip(workers, codes) if code not in (None, 0)]
            if failures:
                worker, code = failures[0]
                raise RuntimeError(f"{worker['directory']} exited with status {code}; "
                                   f"see {output / (worker['directory'] + '.log')}")
            if all(code == 0 for code in codes):
                break
            if time.perf_counter() >= next_report:
                save_progress()
                print(f"{args.method}: {state['completed_trials']}/{state['planned_trials']} trials complete; "
                      f"{sum(code is None for code in codes)} workers running", flush=True)
                next_report = time.perf_counter() + 15
            time.sleep(0.2)
        state["status"] = "complete"
    except BaseException as error:
        state.update(status="interrupted" if isinstance(error, (KeyboardInterrupt, SystemExit)) else "failed",
                     error=f"{type(error).__name__}: {error}")
        _stop_workers(processes)
        raise
    finally:
        for signum, handler in handlers.items():
            signal.signal(signum, handler)
        for log in logs:
            log.close()
        save_progress()
    print(f"Completed {args.method}: {state['completed_trials']}/{state['planned_trials']} trials; "
          f"elapsed {state['elapsed_seconds']:.1f}s", flush=True)
    return output
