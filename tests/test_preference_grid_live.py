"""Terminal display stays outside the experiment and its frozen conditions."""

import json
from pathlib import Path
import threading
import time

import pytest

from experiments.preference_grid.design import build_plan, write_plan


def prepare(root):
    write_plan(root, build_plan(treatments=["baseline"], repetitions=1,
                               run_settings={"backend": "scripted", "full_session": False,
                                             "completed_period_limit": 2}))


def test_watch_only_reads_files_without_loading_runtime_or_starting_batch(tmp_path, monkeypatch, capsys):
    from custom import config
    from experiments.preference_grid import live, run_batch

    root = tmp_path / "batch"
    prepare(root)
    before = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}

    def forbidden(*args, **kwargs):
        pytest.fail("Watching must not load credentials or launch experiments")

    monkeypatch.setattr(config, "load_runtime_config", forbidden)
    monkeypatch.setattr(run_batch, "run_batch", forbidden)
    assert live.main(["watch", "--batch-dir", str(root), "--once"]) == 0
    output = capsys.readouterr().out
    assert "只读查看" in output and "北京时间" in output and "81" in output
    assert before == {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}


def test_live_run_prints_before_work_and_while_waiting_and_preserves_arguments(tmp_path, monkeypatch, capsys):
    from experiments.preference_grid import live, run_batch

    root = tmp_path / "batch"
    prepare(root)
    observed = {}

    def run(path, **kwargs):
        observed.update(kwargs)
        assert path == root
        initial = capsys.readouterr().out
        assert "准备运行批次" in initial and "并发上限：2场 / 4个Agent" in initial
        # The watcher continues to report even before there are any worker logs.
        time.sleep(0.06)
        assert "北京时间" in capsys.readouterr().out
        return {"scheduled": 2, "completed_this_invocation": 2, "failed_this_invocation": 0}

    monkeypatch.setattr(run_batch, "run_batch", run)
    assert live.main(["run", "--batch-dir", str(root), "--interval", "0.01",
                      "--concurrency", "2", "--limit-jobs", "2", "--retry-failed",
                      "--treatments", "baseline", "--combinations", "P2__P8"]) == 0
    assert observed == {"concurrency": 2, "limit_jobs": 2, "retry_failed": True,
                        "treatments": ["baseline"], "combinations": ["P2__P8"]}


def test_display_error_does_not_cancel_or_retry_batch(tmp_path, monkeypatch, capsys):
    from experiments.preference_grid import live, run_batch

    root = tmp_path / "batch"
    prepare(root)
    original = live.print_progress
    failed = threading.Event()
    calls = []

    def display(reader):
        if threading.current_thread() is not threading.main_thread():
            failed.set()
            raise OSError("api_key=fake-display-secret")
        return original(reader)

    def run(*args, **kwargs):
        calls.append(True)
        assert failed.wait(2)
        return {"failed_this_invocation": 0}

    monkeypatch.setattr(live, "print_progress", display)
    monkeypatch.setattr(run_batch, "run_batch", run)
    assert live.main(["run", "--batch-dir", str(root), "--interval", "0.01"]) == 0
    assert len(calls) == 1
    error = capsys.readouterr().err
    assert "进度显示暂不可用" in error and "fake-display-secret" not in error


@pytest.mark.parametrize("command", ["watch", "run"])
def test_ctrl_c_message_distinguishes_watching_from_running(tmp_path, monkeypatch, capsys, command):
    from experiments.preference_grid import live, run_batch

    root = tmp_path / "batch"
    prepare(root)

    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt

    if command == "watch":
        monkeypatch.setattr(live, "print_progress", interrupted)
    else:
        monkeypatch.setattr(run_batch, "run_batch", interrupted)
    assert live.main([command, "--batch-dir", str(root)]) == 130
    output = capsys.readouterr().out
    assert ("原实验继续运行" if command == "watch" else "批次已中断") in output


def test_display_modules_are_outside_frozen_execution_identity():
    from experiments.preference_grid.run_batch import execution_identity, validate_settings

    identity = execution_identity(validate_settings({"backend": "scripted"}))
    sources = identity["source_sha256"]
    for name in ("run_batch.py", "worker.py", "design.py"):
        assert "experiments/preference_grid/" + name in sources
    for name in ("live.py", "progress.py"):
        assert "experiments/preference_grid/" + name not in sources


@pytest.mark.parametrize("interval", ["0", "-1", "nan", "inf"])
def test_invalid_refresh_interval_rejected_before_running(tmp_path, interval):
    from experiments.preference_grid.live import main

    with pytest.raises(SystemExit) as exc:
        main(["watch", "--batch-dir", str(tmp_path), "--interval", interval])
    assert exc.value.code == 2
