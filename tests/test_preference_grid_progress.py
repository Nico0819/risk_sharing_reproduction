"""Read-only live progress, including partial logs and interrupted attempts."""

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pytest

from experiments.preference_grid import progress


NOW = datetime(2026, 11, 10, 13, 30, tzinfo=timezone.utc)
FIRST = "baseline__P2__P8__r01"
SECOND = "baseline__P2__P8__r02"
OTHER = "25v__P1__P9__r01"


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def batch(tmp_path, jobs=(FIRST, SECOND, OTHER)):
    write_json(tmp_path / "plan.json", {"jobs": [{"job_id": name} for name in jobs]})
    return tmp_path


def attempt(root, job=FIRST, *, number=1, state="running", pid=12345, exit_code=None):
    path = root / "cases" / job / "attempts" / f"{number:04d}"
    write_json(path / "status.json", {"status": state, "worker_pid": pid, "exit_code": exit_code,
                                      "started_at": "2026-11-10T13:29:00+00:00"})
    (path / "run").mkdir(exist_ok=True)
    return path


def append(log, *rows):
    with log.open("ab") as handle:
        for row in rows:
            handle.write(json.dumps(row).encode() + b"\n")


@pytest.fixture(autouse=True)
def live_pid(monkeypatch):
    monkeypatch.setattr(progress, "_pid_alive", lambda pid: True if pid else None)


def test_snapshot_filters_and_incremental_events_are_read_only(tmp_path):
    root = batch(tmp_path)
    path = attempt(root)
    log = path / "run/run.log"
    append(log, {"event": "scheduler_step_started", "current_period": 3, "completed_periods": [1, 2]},
           {"event": "model_call_finished", "request_index": 12, "phase": "memory_update"})
    os.utime(log, (NOW.timestamp() - 4, NOW.timestamp() - 4))
    before = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
    reader = progress.ProgressReader(root, treatments=["baseline"], combinations=["P2__P8"])
    snap = reader.snapshot(now=NOW)
    assert snap["total_planned"] == 3 and snap["selected_planned"] == 2
    assert snap["counts"]["running"] == snap["counts"]["pending"] == 1
    current = snap["active"][0]
    assert (current["current_period"], current["settled_period"], current["model_requests"]) == (3, 2, 12)
    assert current["phase"] == "memory_update" and current["elapsed_seconds"] == 60
    assert current["last_activity_age"] == 4
    cursor = reader.cursors[log]
    first_offset = cursor.offset
    assert reader.snapshot(now=NOW) == snap
    assert cursor.offset == first_offset
    assert {p: p.read_bytes() for p in root.rglob("*") if p.is_file()} == before
    append(log, {"event": "model_call_finished", "request_index": 11, "phase": "routing"},
           {"event": "scheduler_step_finished", "period_after": 4, "settled_period": 3, "model_requests": 13})
    current = reader.snapshot(now=NOW)["active"][0]
    assert current["current_period"] == 3 and current["settled_period"] == 3
    assert current["model_requests"] == 13 and current["phase"] == "routing"
    assert cursor.offset > first_offset
    rendered = progress.render_snapshot(reader.snapshot(now=NOW))
    assert "最近调用：工具路由" in rendered and "进程完成" in rendered
    assert "最终以批次报告为准" in rendered


def test_closing_memory_does_not_look_like_a_new_economic_round(tmp_path):
    root = batch(tmp_path, [FIRST])
    log = attempt(root) / "run/run.log"
    append(log, {"event": "scheduler_step_started", "current_period": 3},
           {"event": "scheduler_step_finished", "period_before": 3, "period_after": 4,
            "settled_period": 3, "model_requests": 13},
           {"event": "model_call_finished", "phase": "memory_update", "memory_update_kind": "closing",
            "request_index": 14})
    rendered = progress.render_snapshot(progress.ProgressReader(root).snapshot())
    assert "当前第 3 轮" in rendered and "已结算 3 轮" in rendered
    assert "终局记忆更新" in rendered and "当前第 4 轮" not in rendered


def test_half_lines_malformed_records_and_long_sensitive_actions(tmp_path):
    root = batch(tmp_path, [FIRST])
    log = attempt(root) / "run/run.log"
    reader = progress.ProgressReader(root)
    log.write_bytes(b'{"event":"model_call_finished","request_index":9')
    assert reader.snapshot()["active"][0]["model_requests"] == 0
    with log.open("ab") as handle:
        handle.write(b',"phase":"decision"}\nNOT_JSON api_key=DO_NOT_PRINT\n')
        handle.write(b'{"event":"tool_actions","secret":"' + b"x" * progress.MAX_LINE_BYTES)
    snap = reader.snapshot()
    assert snap["active"][0]["model_requests"] == 9
    cursor = reader.cursors[log]
    assert cursor.discard and len(cursor.pending) <= progress.MAX_LINE_BYTES
    with log.open("ab") as handle:
        handle.write(b'api_key=DO_NOT_PRINT"}\n')
    append(log, {"event": "model_call_finished", "request_index": 10,
                 "phase": "Authorization: Bearer DO_NOT_PRINT", "error": "DO_NOT_PRINT"})
    snap = reader.snapshot()
    assert snap["active"][0]["model_requests"] == 10
    assert snap["active"][0]["phase"] is None
    assert "DO_NOT_PRINT" not in json.dumps(snap) + progress.render_snapshot(snap)
    assert cursor.malformed_lines == 1 and not cursor.discard


def test_log_rotation_and_truncation_reset_progress(tmp_path):
    root = batch(tmp_path, [FIRST])
    log = attempt(root) / "run/run.log"
    append(log, {"event": "scheduler_step_started", "current_period": 25, "completed_periods": list(range(1, 25))})
    reader = progress.ProgressReader(root)
    assert reader.snapshot()["active"][0]["settled_period"] == 24
    log.write_bytes(b'{"event":"model_call_finished","request_index":2}\n')
    snap = reader.snapshot()["active"][0]
    assert snap["settled_period"] == 0 and snap["model_requests"] == 2
    new = log.with_suffix(".new")
    new.write_bytes(b'{"event":"scheduler_step_started","current_period":1}\n')
    new.replace(log)
    snap = reader.snapshot()["active"][0]
    assert snap["current_period"] == 1 and snap["model_requests"] == 0


def test_latest_attempt_wins_and_no_logs_are_reported_as_waiting(tmp_path):
    root = batch(tmp_path, [FIRST, SECOND])
    attempt(root, state="completed", exit_code=0)
    attempt(root, number=2)
    attempt(root, SECOND, state="failed")
    snap = progress.ProgressReader(root).snapshot()
    assert snap["counts"]["running"] == snap["counts"]["failed"] == 1
    assert snap["counts"]["completed"] == 0
    assert snap["active"][0]["attempt"] == "0002"
    assert "等待启动/日志" in progress.render_snapshot(snap)


@pytest.mark.parametrize("state,pid,exit_code,alive,expected", [
    ("running", 5, None, False, "interrupted_or_stale"),
    ("running", None, None, None, "unknown"),
    ("interrupted", 5, None, True, "interrupted_or_stale"),
    ("completed", 5, 0, False, "completed"),
    ("completed", 5, True, False, "unknown"),
    ("completed", 5, None, False, "unknown"),
    ("something api_key=DO_NOT_PRINT", 5, 0, True, "unknown"),
])
def test_process_states_never_invent_success(tmp_path, monkeypatch, state, pid, exit_code, alive, expected):
    root = batch(tmp_path, [FIRST])
    attempt(root, state=state, pid=pid, exit_code=exit_code)
    monkeypatch.setattr(progress, "_pid_alive", lambda pid: alive)
    snap = progress.ProgressReader(root).snapshot()
    assert snap["counts"][expected] == 1
    assert "DO_NOT_PRINT" not in json.dumps(snap) + progress.render_snapshot(snap)


def test_missing_and_corrupt_status_are_unknown(tmp_path):
    root = batch(tmp_path, [FIRST, SECOND])
    first = attempt(root)
    second = attempt(root, SECOND)
    (first / "status.json").unlink()
    (second / "status.json").write_text('{"status":')
    snap = progress.ProgressReader(root).snapshot()
    assert snap["counts"]["unknown"] == 2


def test_malformed_field_types_do_not_break_progress_or_leak_content(tmp_path):
    root = batch(tmp_path, [FIRST])
    current = attempt(root, state={"api_key": "DO_NOT_PRINT"})
    log = current / "run/run.log"
    append(log, {"event": ["DO_NOT_PRINT"]}, {"event": {"api_key": "DO_NOT_PRINT"}},
           {"event": "model_call_finished", "request_index": 5, "phase": ["DO_NOT_PRINT"]},
           {"event": "scheduler_step_started", "current_period": {"bad": 3}, "completed_periods": [1, {}, False]})
    snap = progress.ProgressReader(root).snapshot()
    assert snap["counts"]["unknown"] == 1
    assert snap["active"][0]["model_requests"] == 5
    assert snap["active"][0]["settled_period"] == 1
    assert "DO_NOT_PRINT" not in json.dumps(snap) + progress.render_snapshot(snap)


def test_reader_never_opens_model_traces_profiles_or_config(tmp_path, monkeypatch):
    root = batch(tmp_path, [FIRST])
    current = attempt(root)
    append(current / "run/run.log", {"event": "model_call_finished", "request_index": 1, "phase": "decision"})
    original = Path.open

    def allowed_only(path, *args, **kwargs):
        assert path.name in {"plan.json", "status.json", "run.log"}
        mode = args[0] if args else kwargs.get("mode", "r")
        assert not any(marker in mode for marker in ("w", "a", "+"))
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", allowed_only)
    assert progress.ProgressReader(root).snapshot()["counts"]["running"] == 1


def test_unsafe_job_path_and_unknown_filters_rejected(tmp_path):
    root = batch(tmp_path, [FIRST])
    with pytest.raises(ValueError, match="Unknown treatment"):
        progress.ProgressReader(root, treatments=["25v"])
    with pytest.raises(ValueError, match="Unknown combination"):
        progress.ProgressReader(root, combinations=["P9__P9"])
    write_json(root / "plan.json", {"jobs": [{"job_id": "../outside"}]})
    with pytest.raises(ValueError, match="Invalid"):
        progress.ProgressReader(root)


def test_per_poll_read_is_bounded(tmp_path, monkeypatch):
    root = batch(tmp_path, [FIRST])
    log = attempt(root) / "run/run.log"
    monkeypatch.setattr(progress, "MAX_READ_BYTES", 80)
    append(log, *[{"event": "model_call_finished", "request_index": i} for i in range(1, 6)])
    reader = progress.ProgressReader(root)
    reader.snapshot()
    cursor = reader.cursors[log]
    assert cursor.offset == 80
    for _ in range(10):
        reader.snapshot()
    assert cursor.offset == log.stat().st_size and cursor.model_requests == 5
