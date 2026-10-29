"""Check the sweep's compute settings without loading models or generating designs."""

import os
from pathlib import Path
import shlex
import subprocess

import pytest


REPO = Path(__file__).resolve().parents[2]


def preview(tmp_path, **settings):
    env = os.environ.copy()
    for name in ("BEST_N", "VARIANTS", "EVAL_CONFIG"):
        env.pop(name, None)
    env.update(EVAL_OUTPUT=str(tmp_path / "preview"), EVAL_SEEDS="10000", TRIALS_PER_RUN="100",
               GSDIFF_PYTHON="must-not-run-during-preview", SVDD_K="10", GUIDED_STEPS="999",
               LATE_GUIDED_STEPS="200")
    env.update(settings)
    return subprocess.run(["bash", "test_time/launch_eval.sh", "--dry-run"], cwd=REPO,
                          env=env, text=True, capture_output=True)


def generation_commands(output):
    return [shlex.split(line) for line in output.splitlines() if "-m test_time.run " in line]


@pytest.mark.parametrize("k,steps,n", [(10, 999, 11), (10, 200, 3), (4, 10, 1),
                                      (1, 999, 1), (10, 0, 1), (10, 150, 3)])
def test_sweep_derives_best_n_from_full_svdd_and_keeps_late_before_full(tmp_path, k, steps, n):
    result = preview(tmp_path, SVDD_K=str(k), GUIDED_STEPS=str(steps))
    assert result.returncode == 0, result.stderr
    commands = generation_commands(result.stdout)
    assert [Path(command[command.index("--output_dir") + 1]).name for command in commands] == [
        "baseline", "best_of_n", "svdd_pm_last20", "svdd_pm_full"]
    best = commands[1]
    assert best[best.index("--method") + 1] == "best_of_n"
    assert int(best[best.index("--n") + 1]) == n
    assert not (tmp_path / "preview").exists()


def test_late_window_does_not_change_full_svdd_matching(tmp_path):
    result = preview(tmp_path, LATE_GUIDED_STEPS="10")
    assert result.returncode == 0, result.stderr
    best = generation_commands(result.stdout)[1]
    assert best[best.index("--n") + 1] == "11"


def test_old_fixed_n_is_rejected_instead_of_silently_breaking_matching(tmp_path):
    result = preview(tmp_path, BEST_N="2")
    assert result.returncode == 2
    assert "Unset BEST_N" in result.stderr


def test_rplan18_defaults_and_svdd_subset_preserve_seed_budget(tmp_path):
    result = preview(tmp_path, VARIANTS="svdd_pm_last20 svdd_pm_full", EVAL_OUTPUT="")
    assert result.returncode == 0, result.stderr
    commands = generation_commands(result.stdout)
    assert [Path(command[command.index("--output_dir") + 1]).name for command in commands] == [
        "svdd_pm_last20", "svdd_pm_full"]
    for command in commands:
        assert command[command.index("--config") + 1] == "test_time/configs/experiment_rplan18.json"
        assert command[command.index("--trials") + 1] == "100"
        assert command[command.index("--seed") + 1] == "10000"
        assert command[command.index("--output_dir") + 1].startswith("test_time/outputs/rplan18_sweep_001/")


def test_explicit_config_and_variant_order_are_preserved(tmp_path):
    result = preview(tmp_path, EVAL_CONFIG="custom config.json", VARIANTS="svdd_pm_full baseline")
    assert result.returncode == 0, result.stderr
    commands = generation_commands(result.stdout)
    assert [Path(command[command.index("--output_dir") + 1]).name for command in commands] == ["svdd_pm_full", "baseline"]
    assert all(command[command.index("--config") + 1] == "custom config.json" for command in commands)


@pytest.mark.parametrize("variants", ["", "wrong", "svdd_pm_last20 svdd_pm_last20"])
def test_empty_unknown_or_duplicate_variants_fail_before_execution(tmp_path, variants):
    result = preview(tmp_path, VARIANTS=variants)
    assert result.returncode == 2
    assert not (tmp_path / "preview").exists()
