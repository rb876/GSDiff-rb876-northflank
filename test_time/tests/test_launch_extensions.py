"""Preview and failure-safe extension orchestration without loading a model."""

import json
import os
from pathlib import Path
import shlex
import subprocess

import pytest


REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "test_time" / "launch_extensions.sh"


def launch(tmp_path, *arguments, **settings):
    env = os.environ.copy()
    for name in ("INFERENCE_BATCH_SIZE", "METHODS", "WORKERS", "EXTENSION_CONFIG"):
        env.pop(name, None)
    env.update(PYTHON="must-not-run-in-preview", TRIALS_PER_RUN="100", EVAL_SEED="10000",
               DEVICE="cuda:0", EVAL_OUTPUT=str(tmp_path / "output with spaces"))
    env.update(settings)
    return subprocess.run(["bash", str(SCRIPT), *arguments], cwd=tmp_path, env=env,
                          text=True, capture_output=True)


def commands(result):
    return [shlex.split(line) for line in result.stdout.splitlines()
            if "-m test_time.run_extensions " in line]


def fake_python(tmp_path):
    interpreter = tmp_path / "fake python"
    interpreter.write_text(
        "#!/usr/bin/env python3\n"
        "import json, os, sys\n"
        "with open(os.environ['LAUNCH_CALLS'], 'a') as stream:\n"
        "    stream.write(json.dumps(sys.argv[1:]) + '\\n')\n"
        "print('fake runner output', flush=True)\n"
        "sys.exit(int(os.environ.get('LAUNCH_EXIT', '0')))\n"
    )
    interpreter.chmod(0o755)
    return interpreter


def test_default_preview_has_both_methods_and_summary_without_creating_output(tmp_path):
    result = launch(tmp_path)
    assert result.returncode == 0, result.stderr
    planned = commands(result)
    assert len(planned) == 3
    for command, method in zip(planned[:2], ("svdd_repair_aware", "hard_h_mc")):
        assert command[command.index("--method") + 1] == method
        assert command[command.index("--trials") + 1] == "100"
        assert command[command.index("--seed") + 1] == "10000"
        assert command[command.index("--inference_batch_size") + 1] == "32"
        assert command[command.index("--workers") + 1] == "1"
        assert command[command.index("--device") + 1] == "cuda:0"
        assert command[command.index("--config") + 1] == "test_time/configs/extensions_ventilation_v1.json"
        assert command[command.index("--output_dir") + 1] == str(tmp_path / "output with spaces" / method)
        assert "--k" not in command
    assert planned[-1][-2:] == ["--summarize_root", str(tmp_path / "output with spaces")]
    assert not (tmp_path / "output with spaces").exists()


def test_smoke_changes_trial_and_proposal_counts_without_shortening_guidance(tmp_path):
    result = launch(tmp_path, "--smoke", "--dry-run", TRIALS_PER_RUN="500")
    assert result.returncode == 0, result.stderr
    repair, hard, _ = commands(result)
    for command in (hard, repair):
        assert command[command.index("--trials") + 1] == "2"
        assert command[command.index("--k") + 1] == "2"
        assert command[command.index("--config") + 1] == "test_time/configs/extensions_ventilation_v1.json"
        assert "--guided_steps" not in command
    assert hard[hard.index("--rollouts") + 1] == "2"
    assert hard[hard.index("--terminal_k") + 1] == "2"
    assert "--rollouts" not in repair and "--terminal_k" not in repair
    assert not (tmp_path / "output with spaces").exists()


def test_parallel_workers_keep_total_trials_and_method_order(tmp_path):
    result = launch(tmp_path, WORKERS="2", TRIALS_PER_RUN="5")
    assert result.returncode == 0, result.stderr
    repair, hard, summary = commands(result)
    assert repair[repair.index("--method") + 1] == "svdd_repair_aware"
    assert hard[hard.index("--method") + 1] == "hard_h_mc"
    for command in (repair, hard):
        assert command[command.index("--workers") + 1] == "2"
        assert command[command.index("--trials") + 1] == "5"
    assert "--summarize_root" in summary
    assert not (tmp_path / "output with spaces").exists()


def test_ventilation_output_default_and_custom_config_override(tmp_path):
    result = launch(tmp_path, EVAL_OUTPUT="", EXTENSION_CONFIG="custom extension config.json")
    assert result.returncode == 0, result.stderr
    for command in commands(result)[:2]:
        assert command[command.index("--config") + 1] == "custom extension config.json"
        assert command[command.index("--output_dir") + 1].startswith("test_time/outputs/ventilation_extensions_001/")
    assert "Config: custom extension config.json" in result.stdout


@pytest.mark.parametrize("settings", [{"METHODS": "baseline"}, {"METHODS": "hard_h_mc hard_h_mc"},
                                     {"TRIALS_PER_RUN": "0"}, {"INFERENCE_BATCH_SIZE": "-1"},
                                     {"WORKERS": "0"}, {"WORKERS": "two"},
                                     {"EVAL_SEED": "wrong"}])
def test_invalid_settings_are_rejected_before_execution(tmp_path, settings):
    result = launch(tmp_path, "--run", **settings)
    assert result.returncode == 2
    assert not (tmp_path / "output with spaces").exists()


def test_existing_second_method_prevents_any_run_and_preserves_data(tmp_path):
    output = tmp_path / "output with spaces"
    existing = output / "hard_h_mc"
    existing.mkdir(parents=True)
    marker = existing / "saved.txt"
    marker.write_text("original")
    result = launch(tmp_path, "--run")
    assert result.returncode == 2 and "Output already exists" in result.stderr
    assert marker.read_text() == "original"
    assert not (output / "svdd_repair_aware.log").exists()


def test_run_supports_paths_with_spaces_and_unselected_existing_method(tmp_path):
    interpreter = fake_python(tmp_path)
    calls = tmp_path / "calls.jsonl"
    output = tmp_path / "output with spaces"
    (output / "hard_h_mc").mkdir(parents=True)
    result = launch(tmp_path, "--run", "--smoke", PYTHON=str(interpreter),
                    METHODS="svdd_repair_aware", LAUNCH_CALLS=str(calls))
    assert result.returncode == 0, result.stderr
    called = [json.loads(line) for line in calls.read_text().splitlines()]
    assert len(called) == 2
    assert called[0][called[0].index("--output_dir") + 1] == str(output / "svdd_repair_aware")
    assert called[1] == ["-m", "test_time.run_extensions", "--summarize_root", str(output)]
    assert "fake runner output" in (output / "svdd_repair_aware.log").read_text()
    assert "fake runner output" in (output / "comparison.log").read_text()


def test_runner_failure_survives_tee_and_stops_later_methods(tmp_path):
    calls = tmp_path / "calls.jsonl"
    result = launch(tmp_path, "--smoke", "--run", PYTHON=str(fake_python(tmp_path)),
                    LAUNCH_CALLS=str(calls), LAUNCH_EXIT="7")
    assert result.returncode == 7
    assert len(calls.read_text().splitlines()) == 1
    assert not (tmp_path / "output with spaces" / "hard_h_mc.log").exists()
    assert not (tmp_path / "output with spaces" / "comparison.log").exists()
