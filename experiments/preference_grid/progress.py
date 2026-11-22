"""Read-only, bounded progress snapshots for existing batch runs.

This module never loads runtime configuration or imports the model runner. It
reads only the plan, attempt status and compact run events, not model traces.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
from typing import Any

MAX_LINE_BYTES = 64 * 1024
MAX_READ_BYTES = 2 * 1024 * 1024
_JOB_ID = re.compile(r"(baseline|25v|25f)__(P[1-9]__P[1-9])__r[0-9]{2,}\Z")
PHASES = {"decision": "经济决策", "memory_update": "记忆更新", "routing": "工具路由",
          "deliberation": "额外评估", "closing_memory": "终局记忆更新"}
STATUSES = ("pending", "running", "completed", "failed", "interrupted_or_stale", "unknown")


def _integer(value: Any, minimum: int = 0) -> int | None:
    return value if type(value) is int and value >= minimum else None


def _time(value: Any) -> datetime | None:
    try:
        result = datetime.fromisoformat(value) if isinstance(value, str) else None
        return result.astimezone(timezone.utc) if result and result.tzinfo else None
    except (ValueError, OverflowError):
        return None


def _pid_alive(pid: Any) -> bool | None:
    if _integer(pid, 1) is None:
        return None
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except (PermissionError, OSError):
        return None
    return True


def _read_object(path: Path) -> dict:
    try:
        if path.is_symlink():
            return {}
        with path.open("rb") as handle:
            data = handle.read(MAX_LINE_BYTES + 1)
        if len(data) > MAX_LINE_BYTES:
            return {}
        value = json.loads(data)
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


@dataclass
class _LogCursor:
    identity: tuple[int, int] | None = None
    offset: int = 0
    pending: bytes = b""
    discard: bool = False
    current_period: int | None = None
    settled_period: int = 0
    model_requests: int = 0
    phase: str | None = None
    modified: float | None = None
    malformed_lines: int = 0

    def _event(self, line: bytes) -> None:
        try:
            row = json.loads(line)
        except (ValueError, UnicodeDecodeError):
            self.malformed_lines += 1
            return
        if not isinstance(row, dict):
            return
        event = row.get("event")
        if not isinstance(event, str):
            return
        if event in {"scheduler_step_started", "scheduler_step_finished"}:
            # A settled turn advances the environment before closing notes are
            # written. Do not present period_after as a newly started decision.
            period = _integer(row.get("current_period") if event == "scheduler_step_started"
                              else row.get("period_before", row.get("settled_period")), 1)
            if period is not None:
                self.current_period = period
            settled = _integer(row.get("settled_period"))
            completed = row.get("completed_periods")
            if isinstance(completed, list):
                valid = [p for p in completed if _integer(p, 1) is not None]
                settled = max([settled or 0, *valid])
            if settled is not None:
                self.settled_period = max(self.settled_period, settled)
            requests = _integer(row.get("model_requests"))
            if requests is not None:
                self.model_requests = max(self.model_requests, requests)
        elif event == "model_call_finished":
            index = _integer(row.get("request_index"))
            if index is not None:
                self.model_requests = max(self.model_requests, index)
            phase = row.get("phase")
            self.phase = phase if isinstance(phase, str) and phase in PHASES else None
            if phase == "memory_update" and row.get("memory_update_kind") == "closing":
                self.phase = "closing_memory"

    def read(self, path: Path) -> None:
        try:
            with path.open("rb") as handle:
                info = os.fstat(handle.fileno())
                identity = (info.st_dev, info.st_ino)
                if identity != self.identity or info.st_size < self.offset:
                    self.__dict__.update(_LogCursor(identity=identity).__dict__)
                self.modified = info.st_mtime
                handle.seek(self.offset)
                data = handle.read(MAX_READ_BYTES)
                self.offset = handle.tell()
        except OSError:
            return
        # Never retain or parse an unbounded tool_actions line. A partial JSON
        # line is held until its newline arrives, rather than counted as broken.
        for part in data.splitlines(keepends=True):
            ended = part.endswith(b"\n")
            if not self.discard:
                if len(self.pending) + len(part) > MAX_LINE_BYTES:
                    self.pending, self.discard = b"", True
                else:
                    self.pending += part
            if ended:
                if not self.discard and self.pending.strip():
                    self._event(self.pending)
                self.pending, self.discard = b"", False


class ProgressReader:
    def __init__(self, batch_dir: Path, *, treatments=None, combinations=None):
        self.root = Path(batch_dir).expanduser().resolve()
        try:
            plan = json.loads((self.root / "plan.json").read_text(encoding="utf-8"))
            jobs = plan["jobs"]
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise ValueError("Cannot read a valid batch plan.json") from exc
        if not isinstance(jobs, list):
            raise ValueError("Batch jobs must be a list")
        self.jobs = []
        seen = set()
        for job in jobs:
            match = _JOB_ID.fullmatch(job.get("job_id", "")) if isinstance(job, dict) and isinstance(job.get("job_id"), str) else None
            if not match or job["job_id"] in seen:
                raise ValueError("Invalid or duplicate batch job ID")
            seen.add(job["job_id"])
            if treatments and match[1] not in treatments:
                continue
            if combinations and match[2] not in combinations:
                continue
            self.jobs.append(job["job_id"])
        if treatments and set(treatments) - {j.split("__")[0] for j in seen}:
            raise ValueError("Unknown treatment filter")
        if combinations and set(combinations) - {"__".join(j.split("__")[1:3]) for j in seen}:
            raise ValueError("Unknown combination filter")
        self.total_planned = len(jobs)
        self.cursors: dict[Path, _LogCursor] = {}

    def snapshot(self, *, now: datetime | None = None) -> dict:
        now = now or datetime.now(timezone.utc)
        counts, active = Counter({key: 0 for key in STATUSES}), []
        for job_id in self.jobs:
            attempts_root = self.root / "cases" / job_id / "attempts"
            try:
                attempts = [p for p in attempts_root.iterdir() if p.is_dir() and p.name.isdigit()]
            except OSError:
                attempts = []
            if not attempts:
                counts["pending"] += 1
                continue
            attempt = max(attempts, key=lambda p: int(p.name))
            if not attempt.resolve().is_relative_to(self.root):
                counts["unknown"] += 1
                continue
            status = _read_object(attempt / "status.json")
            state, pid = status.get("status"), _integer(status.get("worker_pid"), 1)
            if not isinstance(state, str):
                state = "unknown"
            if state == "running":
                alive = _pid_alive(pid)
                state = "running" if alive is True else "interrupted_or_stale" if alive is False else "unknown"
            elif state == "completed":
                state = "completed" if type(status.get("exit_code")) is int and status["exit_code"] == 0 else "unknown"
            elif state == "interrupted":
                state = "interrupted_or_stale"
            elif state != "failed":
                state = "unknown"
            counts[state] += 1
            if state in {"completed", "failed"}:
                continue
            log = attempt / "run/run.log"
            cursor = self.cursors.setdefault(log, _LogCursor())
            if log.resolve().is_relative_to(self.root) and not log.is_symlink():
                cursor.read(log)
            started = _time(status.get("started_at"))
            active.append({"job_id": job_id, "attempt": attempt.name, "pid": pid, "status": state,
                           "current_period": cursor.current_period, "settled_period": cursor.settled_period,
                           "model_requests": cursor.model_requests, "phase": cursor.phase,
                           "elapsed_seconds": max(0, (now - started).total_seconds()) if started else None,
                           "last_activity_age": max(0, now.timestamp() - cursor.modified) if cursor.modified else None,
                           "log_path": str(log.relative_to(self.root))})
        return {"total_planned": self.total_planned, "selected_planned": len(self.jobs),
                "counts": dict(counts), "active": active, "captured_at": now.isoformat()}


def render_snapshot(snapshot: dict) -> str:
    counts = snapshot["counts"]
    lines = [f"批次进度：查看 {snapshot['selected_planned']} / 全计划 {snapshot['total_planned']} 场",
             f"运行 {counts['running']} | 进程完成 {counts['completed']} | 失败 {counts['failed']} | "
             f"未运行 {counts['pending']} | 中断/待确认 {counts['interrupted_or_stale'] + counts['unknown']}"]
    for job in snapshot["active"]:
        period = f"当前第 {job['current_period']} 轮" if job["current_period"] is not None else "等待启动/日志"
        elapsed = f"{int(job['elapsed_seconds'])} 秒" if job["elapsed_seconds"] is not None else "未知"
        age = f"{int(job['last_activity_age'])} 秒前" if job["last_activity_age"] is not None else "尚无日志"
        phase = PHASES.get(job["phase"], "尚无记录")
        state = "运行中" if job["status"] == "running" else "状态待确认"
        lines.append(f"  {job['job_id']}（尝试 {job['attempt']}，{state}）：{period}，已结算 {job['settled_period']} 轮，"
                     f"已记录请求序号 {job['model_requests']}，最近调用：{phase}，耗时 {elapsed}，日志更新 {age}")
    lines.append("查看范围包含尚未启动的任务；进程完成不等于结果审核通过，最终以批次报告为准。")
    return "\n".join(lines)
