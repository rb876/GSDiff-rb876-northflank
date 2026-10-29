"""Real lightweight subprocesses exercise seed partitioning and worker cleanup."""

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

import pytest

from test_time import parallel_extensions
from test_time.run_extensions import _parser


def settings(tmp_path, *, trials=5, workers=2):
    args = _parser().parse_args(["--method", "hard_h_mc", "--trials", str(trials),
                                "--seed", "10000", "--device", "cpu", "--rollouts", "3",
                                "--no-progress", "--output_dir", str(tmp_path / "method with spaces")])
    args.workers = workers
    return args


@pytest.fixture
def fake_workers(tmp_path, monkeypatch):
    """Children rendezvous before finishing, so sequential launching cannot pass."""
    child = tmp_path / "fake_child.py"
    child.write_text('''
import argparse, json, os, pathlib, time
parser = argparse.ArgumentParser()
for name in ('seed', 'trials', 'workers'):
    parser.add_argument('--' + name, type=int)
parser.add_argument('--output_dir', type=pathlib.Path)
args, _ = parser.parse_known_args()
args.output_dir.mkdir()
start = time.monotonic()
barrier = pathlib.Path(os.environ['FAKE_BARRIER'])
(barrier / (args.output_dir.name + '.started')).write_text(str(os.getpid()))
while len(list(barrier.glob('*.started'))) < int(os.environ['FAKE_WORKERS']):
    if time.monotonic() - start > 5:
        raise RuntimeError('workers were not launched concurrently')
    time.sleep(.01)
if os.environ.get('FAKE_FAIL') == args.output_dir.name:
    raise SystemExit(7)
if os.environ.get('FAKE_HANG') == '1':
    while True:
        time.sleep(.1)
state = dict(status='complete', completed_trials=args.trials,
             seed=args.seed, started=start, finished=time.monotonic())
(args.output_dir / 'run.json').write_text(json.dumps(state))
print('synthetic worker complete', flush=True)
''')
    barrier = tmp_path / "barrier"
    barrier.mkdir()
    monkeypatch.setenv("FAKE_BARRIER", str(barrier))
    monkeypatch.setenv("FAKE_WORKERS", "2")
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    monkeypatch.setenv("MKL_NUM_THREADS", "3")
    actual_popen = subprocess.Popen
    calls, processes = [], []

    def launch(command, **kwargs):
        calls.append((command, kwargs))
        process = actual_popen([sys.executable, str(child), *command[4:]], **kwargs)
        processes.append(process)
        return process

    monkeypatch.setattr(parallel_extensions.subprocess, "Popen", launch)
    yield calls, processes, barrier
    for process in processes:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def test_workers_overlap_and_cover_disjoint_seed_ranges(tmp_path, fake_workers):
    args = settings(tmp_path)
    output = parallel_extensions.run_parallel(args)
    calls, processes, _ = fake_workers
    state = json.loads((output / "parallel.json").read_text())
    assert state["status"] == "complete"
    assert state["completed_trials"] == state["planned_trials"] == 5
    assert [(worker["seed"], worker["trials"]) for worker in state["workers"]] == [(10000, 3), (10003, 2)]
    children = [json.loads((output / worker["directory"] / "run.json").read_text()) for worker in state["workers"]]
    assert max(child["started"] for child in children) <= min(child["finished"] for child in children)
    assert state["elapsed_seconds"] > 0
    assert args.parallel_output == output
    assert all(process.poll() == 0 for process in processes)
    for command, kwargs in calls:
        assert command[command.index("--workers") + 1] == "1"
        assert command[command.index("--rollouts") + 1] == "3"
        assert command[command.index("--inference_batch_size") + 1] == "32"
        assert "--no-progress" in command
        assert kwargs["start_new_session"] is True
        assert kwargs["env"]["OMP_NUM_THREADS"] == "1"
        assert kwargs["env"]["MKL_NUM_THREADS"] == "3"
    assert state["worker_cpu_threads"] == {"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "3"}
    assert "synthetic worker complete" in (output / "worker_001.log").read_text()
    with pytest.raises(FileExistsError):
        parallel_extensions.run_parallel(settings(tmp_path))
    assert len(calls) == 2


def test_more_workers_than_trials_never_creates_empty_ranges(tmp_path, fake_workers, monkeypatch):
    monkeypatch.setenv("FAKE_WORKERS", "1")
    output = parallel_extensions.run_parallel(settings(tmp_path, trials=1, workers=4))
    state = json.loads((output / "parallel.json").read_text())
    assert state["worker_count"] == 1 and state["requested_workers"] == 4
    assert state["workers"][0]["trials"] == state["completed_trials"] == 1


def test_failed_worker_stops_and_reaps_running_sibling(tmp_path, fake_workers, monkeypatch):
    monkeypatch.setenv("FAKE_FAIL", "worker_001")
    monkeypatch.setenv("FAKE_HANG", "1")
    args = settings(tmp_path)
    with pytest.raises(RuntimeError, match="worker_001 exited with status 7"):
        parallel_extensions.run_parallel(args)
    state = json.loads((args.output_dir / "parallel.json").read_text())
    assert state["status"] == "failed"
    assert [worker["returncode"] for worker in state["workers"]] == [7, -signal.SIGTERM]
    assert all(process.poll() is not None for process in fake_workers[1])


def test_sigterm_to_parent_stops_workers_and_restores_handler(tmp_path, fake_workers, monkeypatch):
    monkeypatch.setenv("FAKE_HANG", "1")
    before = signal.getsignal(signal.SIGTERM)
    barrier = fake_workers[2]
    args = settings(tmp_path)

    def request_stop():
        deadline = time.monotonic() + 5
        while len(list(barrier.glob("*.started"))) < 2 and time.monotonic() < deadline:
            time.sleep(.01)
        if len(list(barrier.glob("*.started"))) == 2:
            os.kill(os.getpid(), signal.SIGTERM)

    thread = threading.Thread(target=request_stop, daemon=True)
    thread.start()
    with pytest.raises(SystemExit) as stopped:
        parallel_extensions.run_parallel(args)
    thread.join(timeout=5)
    assert stopped.value.code == 143
    assert signal.getsignal(signal.SIGTERM) == before
    state = json.loads((args.output_dir / "parallel.json").read_text())
    assert state["status"] == "interrupted"
    assert all(process.poll() == -signal.SIGTERM for process in fake_workers[1])
