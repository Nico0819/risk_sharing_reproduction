"""Run or observe a preference batch with periodic terminal progress.

This display wrapper does not modify the frozen runner or execution identity.
The watch command only reads local files and never imports a model client.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import math
from pathlib import Path
import sys
import threading
from zoneinfo import ZoneInfo

from custom.participant_trace import redact_credentials
from experiments.preference_grid.progress import ProgressReader, render_snapshot


def _interval(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("刷新间隔必须为正数")
    return number


def print_progress(reader: ProgressReader) -> None:
    stamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n[{stamp} 北京时间]\n{render_snapshot(reader.snapshot())}", flush=True)


def _monitor(reader: ProgressReader, stop: threading.Event, interval: float) -> None:
    while not stop.wait(interval):
        try:
            print_progress(reader)
        except Exception as exc:
            # A display problem must not cancel or relaunch an experiment.
            print(f"[进度显示暂不可用] {type(exc).__name__}: {redact_credentials(str(exc))}",
                  file=sys.stderr, flush=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="实时显示配对实验的启动、轮次、模型请求数和任务完成数量。")
    sub = parser.add_subparsers(dest="command", required=True)
    watch = sub.add_parser("watch", help="只读查看已经启动的实验；Ctrl+C仅退出查看")
    run = sub.add_parser("run", help="启动批量实验，同时定期显示进度")
    for command in (watch, run):
        command.add_argument("--batch-dir", type=Path, required=True)
        command.add_argument("--interval", type=_interval, default=10.0, help="进度刷新间隔，单位秒（默认10）")
        command.add_argument("--treatments", nargs="+", choices=("baseline", "25v", "25f"))
        command.add_argument("--combinations", nargs="+")
    watch.add_argument("--once", action="store_true", help="打印一次当前状态后退出")
    run.add_argument("--concurrency", type=int, default=10)
    run.add_argument("--limit-jobs", type=int)
    run.add_argument("--retry-failed", action="store_true")
    args = parser.parse_args(argv)
    stop = threading.Event()
    monitor = None
    reader = None
    try:
        reader = ProgressReader(args.batch_dir.expanduser().resolve(),
                                treatments=args.treatments, combinations=args.combinations)
        if args.command == "watch":
            print("只读查看已有实验；每次刷新显示最新进度。Ctrl+C退出查看，实验继续运行。", flush=True)
            while True:
                print_progress(reader)
                if args.once:
                    return 0
                stop.wait(args.interval)

        if args.concurrency < 1 or (args.limit_jobs is not None and args.limit_jobs < 1):
            raise ValueError("并发数和任务上限必须为正整数")
        print(f"准备运行批次：{args.batch_dir}\n并发上限：{args.concurrency}场 / {args.concurrency * 2}个Agent；"
              f"每{args.interval:g}秒显示一次。正在检查计划、已完成任务和运行条件……", flush=True)
        if args.limit_jobs is not None:
            print(f"本次最多启动 {args.limit_jobs} 场；进度表同时保留筛选范围内未启动任务的数量。", flush=True)
        print_progress(reader)
        monitor = threading.Thread(target=_monitor, args=(reader, stop, args.interval), daemon=True)
        monitor.start()
        # Import only for execution: watch never initializes runtime/provider settings.
        from experiments.preference_grid.run_batch import run_batch

        result = run_batch(args.batch_dir, concurrency=args.concurrency, limit_jobs=args.limit_jobs,
                           retry_failed=args.retry_failed, treatments=args.treatments,
                           combinations=args.combinations)
        print(json.dumps(result, ensure_ascii=False), flush=True)
        return 1 if result.get("failed_this_invocation", 0) else 0
    except KeyboardInterrupt:
        if args.command == "watch":
            print("已退出进度查看，原实验继续运行。", flush=True)
        else:
            print("批次已中断；已有实验文件保留，未运行的任务可以稍后继续。", flush=True)
        return 130
    except Exception as exc:
        print(f"{type(exc).__name__}: {redact_credentials(str(exc))}", file=sys.stderr, flush=True)
        return 2
    finally:
        stop.set()
        if monitor is not None:
            monitor.join()
            if reader is not None:
                try:
                    print_progress(reader)
                except Exception as exc:
                    print(f"[最终进度暂不可用] {type(exc).__name__}: {redact_credentials(str(exc))}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
