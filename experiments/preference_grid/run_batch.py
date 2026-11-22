"""Plan and run independent, ordered preference-pair repetitions."""

from __future__ import annotations

import argparse
import asyncio
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any

from custom.participant_trace import redact_credentials
from experiments.preference_grid.design import build_plan, load_plan, write_plan

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SETTINGS = {
    "backend": "llm", "full_session": True, "completed_period_limit": 3,
    "request_limit": 1000, "max_output_tokens": 8192, "router_steps": 6,
    "agent_turns": 4, "timeout_seconds": 86400.0, "request_timeout": 180.0,
    "memory_condition": "persistent",
}


def validate_settings(settings: dict) -> dict:
    if not isinstance(settings, dict) or set(settings) - set(DEFAULT_SETTINGS):
        raise ValueError("Unknown batch run settings")
    settings = {**DEFAULT_SETTINGS, **settings}
    if settings["backend"] not in {"llm", "scripted"}:
        raise ValueError("backend must be llm or scripted")
    if settings["memory_condition"] != "persistent":
        raise ValueError("This design uses persistent private memory")
    if type(settings["full_session"]) is not bool:
        raise ValueError("full_session must be boolean")
    for name in ("completed_period_limit", "request_limit", "max_output_tokens", "router_steps", "agent_turns"):
        if type(settings[name]) is not int or settings[name] < 1:
            raise ValueError(f"{name} must be a positive integer")
    for name in ("timeout_seconds", "request_timeout"):
        if type(settings[name]) not in (int, float) or not math.isfinite(settings[name]) or settings[name] <= 0:
            raise ValueError(f"{name} must be finite and positive")
    return settings


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def batch_lock(root: Path):
    with (root / ".execution.lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("This batch is already running in another process") from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def attempts_for(root: Path, job: dict) -> list[Path]:
    folder = root / "cases" / job["job_id"] / "attempts"
    return sorted((p for p in folder.glob("*") if p.is_dir() and p.name.isdigit()), key=lambda p: int(p.name))


def _previous_worker_alive(attempt: Path) -> bool:
    try:
        status = json.loads((attempt / "status.json").read_text())
    except (OSError, ValueError):
        return False
    if status.get("status") != "running" or type(status.get("worker_pid")) is not int:
        return False
    try:
        os.kill(status["worker_pid"], 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def execution_identity(settings: dict) -> dict:
    # Resolve configuration locally; this makes no API request. Keys and headers
    # never enter the batch files. A different key may be used for resumption.
    from custom.config import load_runtime_config
    from experiments.fixed_preferences_memory_v1.run_smoke import _version_source_hashes

    cfg = load_runtime_config(settings["backend"],
                              env_path=Path(os.devnull) if settings["backend"] == "scripted" else None,
                              request_timeout=settings["request_timeout"])
    sources = _version_source_hashes()
    for name in ("design.py", "run_batch.py", "worker.py"):
        path = Path(__file__).with_name(name)
        sources[str(path.relative_to(PROJECT_ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    endpoint = hashlib.sha256(cfg.base_url.encode()).hexdigest()
    return {"settings": settings, "runtime": cfg.safe_status(), "endpoint_sha256": endpoint,
            "source_sha256": sources}


def freeze_execution(root: Path, settings: dict) -> None:
    identity = execution_identity(settings)
    path = root / "execution.json"
    if path.exists():
        if json.loads(path.read_text()) != identity:
            raise ValueError("Model configuration or source code changed; create a new batch instead of mixing conditions")
    else:
        write_json(path, identity)


async def _execute(root: Path, jobs: list[dict], settings: dict, concurrency: int) -> list[dict]:
    queue = asyncio.Queue()
    for job in jobs:
        queue.put_nowait(job)
    outcomes = []

    async def consume():
        while not queue.empty():
            job = queue.get_nowait()
            attempts = attempts_for(root, job)
            number = int(attempts[-1].name) + 1 if attempts else 1
            attempt = root / "cases" / job["job_id"] / "attempts" / f"{number:04d}"
            attempt.mkdir(parents=True, exist_ok=False)
            status = {"job_id": job["job_id"], "status": "running", "attempt": number,
                      "started_at": timestamp(), "run_dir": "run"}
            write_json(attempt / "status.json", status)
            process = None
            try:
                with (attempt / "worker.log").open("wb") as output:
                    child_env = dict(os.environ)
                    child_env["PYTHON_DOTENV_DISABLED"] = "1"
                    child_env["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
                    process = await asyncio.create_subprocess_exec(
                        sys.executable, "-m", "experiments.preference_grid.worker",
                        "--batch-dir", str(root), "--job-id", job["job_id"], "--attempt", str(number),
                        cwd=PROJECT_ROOT, env=child_env, stdout=output, stderr=asyncio.subprocess.STDOUT,
                    )
                    status["worker_pid"] = process.pid
                    write_json(attempt / "status.json", status)
                    status["exit_code"] = await process.wait()
                status["status"] = "completed" if status["exit_code"] == 0 else "failed"
            except asyncio.CancelledError:
                status["status"] = "interrupted"
                raise
            except Exception as exc:
                status.update(status="failed", error=redact_credentials(str(exc)))
            finally:
                if process is not None and process.returncode is None:
                    process.terminate()
                    try:
                        await asyncio.wait_for(process.wait(), timeout=10)
                    except asyncio.TimeoutError:
                        process.kill()
                        await process.wait()
                status["finished_at"] = timestamp()
                write_json(attempt / "status.json", status)
                outcomes.append(status)
                print(json.dumps({"job_id": job["job_id"], "status": status["status"],
                                  "finished": len(outcomes), "scheduled": len(jobs)}, ensure_ascii=False), flush=True)
                queue.task_done()

    await asyncio.gather(*(consume() for _ in range(min(concurrency, len(jobs)))))
    return outcomes


def run_batch(batch_dir: Path, *, concurrency: int = 10, limit_jobs: int | None = None,
              retry_failed: bool = False, treatments: list[str] | None = None,
              combinations: list[str] | None = None) -> dict:
    root = Path(batch_dir).resolve()
    plan = load_plan(root)
    settings = validate_settings(plan["run_settings"])
    if type(concurrency) is not int or concurrency < 1:
        raise ValueError("concurrency must be a positive integer")
    if limit_jobs is not None and (type(limit_jobs) is not int or limit_jobs < 1):
        raise ValueError("limit_jobs must be a positive integer")
    if treatments and set(treatments) - set(plan["treatments"]):
        raise ValueError("Unknown treatment filter")
    if combinations and set(combinations) - {j["combination_id"] for j in plan["jobs"]}:
        raise ValueError("Unknown combination filter")
    from experiments.preference_grid.report import inspect_job, write_batch_report

    with batch_lock(root):
        jobs = []
        for job in plan["jobs"]:
            if treatments and job["treatment"] not in treatments:
                continue
            if combinations and job["combination_id"] not in combinations:
                continue
            attempts = attempts_for(root, job)
            if attempts:
                session, _, _ = inspect_job(root, job, settings["full_session"], settings["completed_period_limit"])
                if session["status"] == "complete" or not retry_failed:
                    continue
                if _previous_worker_alive(attempts[-1]):
                    raise ValueError(f"Previous worker still exists for {job['job_id']}; do not duplicate this run")
            jobs.append(job)
        if limit_jobs is not None:
            jobs = jobs[:limit_jobs]
        if jobs:
            freeze_execution(root, settings)
            try:
                outcomes = asyncio.run(_execute(root, jobs, settings, concurrency))
            except BaseException:
                try:
                    write_batch_report(root)
                except Exception as exc:
                    print(f"中断后的报告生成失败：{redact_credentials(str(exc))}", file=sys.stderr)
                raise
        else:
            outcomes = []
        report = write_batch_report(root)
    return {"scheduled": len(jobs), "completed_this_invocation": sum(o["status"] == "completed" for o in outcomes),
            "failed_this_invocation": sum(o["status"] != "completed" for o in outcomes), "report": report}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    plan_parser = sub.add_parser("plan", help="Write the design and profiles without calling models")
    plan_parser.add_argument("--batch-dir", type=Path, required=True)
    plan_parser.add_argument("--repetitions", type=int, default=10)
    plan_parser.add_argument("--treatments", nargs="+", choices=("baseline", "25v", "25f"), default=["baseline", "25v", "25f"])
    plan_parser.add_argument("--seed", type=int, default=2026)
    plan_parser.add_argument("--forced-assignment-probabilities", nargs=3, type=float)
    plan_parser.add_argument("--backend", choices=("llm", "scripted"), default="llm")
    mode = plan_parser.add_mutually_exclusive_group()
    mode.add_argument("--full-session", action="store_true", help="Default: run to natural termination")
    mode.add_argument("--completed-period-limit", type=int, help="Short validation only, explicitly marked smoke")
    for flag, value in DEFAULT_SETTINGS.items():
        if flag not in {"backend", "full_session", "completed_period_limit", "memory_condition"}:
            option = {"router_steps": "router-max-steps", "agent_turns": "agent-max-turns"}.get(flag, flag.replace("_", "-"))
            plan_parser.add_argument("--" + option, dest=flag, type=type(value), default=value)
    run_parser = sub.add_parser("run", help="Execute pending jobs; each child process runs exactly one pair")
    run_parser.add_argument("--batch-dir", type=Path, required=True)
    run_parser.add_argument("--concurrency", type=int, default=10, help="Concurrent independent sessions; up to twice this many participant calls")
    run_parser.add_argument("--limit-jobs", type=int)
    run_parser.add_argument("--retry-failed", action="store_true", help="New attempt for failed/interrupted jobs; preserve prior directories and seeds")
    run_parser.add_argument("--treatments", nargs="+", choices=("baseline", "25v", "25f"))
    run_parser.add_argument("--combinations", nargs="+")
    report_parser = sub.add_parser("report", help="Rebuild CSV and HTML without model calls")
    report_parser.add_argument("--batch-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "plan":
            settings = {k: getattr(args, k, v) for k, v in DEFAULT_SETTINGS.items()}
            settings["full_session"] = args.completed_period_limit is None
            settings["completed_period_limit"] = args.completed_period_limit if args.completed_period_limit is not None else 3
            plan = build_plan(repetitions=args.repetitions, treatments=args.treatments, master_seed=args.seed,
                              forced_assignment_probabilities=args.forced_assignment_probabilities,
                              run_settings=validate_settings(settings))
            write_plan(args.batch_dir, plan)
            result = {"batch_dir": str(args.batch_dir.resolve()), "jobs": len(plan["jobs"]), "model_calls": 0}
        elif args.command == "run":
            result = run_batch(args.batch_dir, concurrency=args.concurrency, limit_jobs=args.limit_jobs,
                               retry_failed=args.retry_failed, treatments=args.treatments, combinations=args.combinations)
        else:
            from experiments.preference_grid.report import write_batch_report
            result = write_batch_report(args.batch_dir)
        print(json.dumps(result, ensure_ascii=False))
        return 1 if result.get("failed_this_invocation", 0) else 0
    except KeyboardInterrupt:
        print("批次已中断；已有实验文件保留，未运行的任务可以稍后继续。", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"批次失败：{type(exc).__name__}: {redact_credentials(str(exc))}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
