"""Read saved participant reasoning by Agent/period without calling a model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_rows(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="Directory of an experiment run")
    parser.add_argument("--agent-id", type=int)
    parser.add_argument("--period", type=int)
    parser.add_argument("--phase", choices=("decision", "memory_update", "deliberation"))
    parser.add_argument("--limit", type=int, default=10, help="Maximum records; 0 means all (default: 10)")
    args = parser.parse_args(argv)
    if args.limit < 0:
        parser.error("--limit must be non-negative")
    path = args.run_dir / "participant_reasoning.jsonl"
    if not path.is_file():
        parser.error("This run has no participant_reasoning.jsonl; old runs cannot recover missing traces.")
    receipts = {}
    action_file = args.run_dir / "agent_actions.jsonl"
    if action_file.is_file():
        for row in read_rows(action_file):
            key = (row.get("agent_id"), row.get("participant_request_index"))
            if key[1] is not None and row.get("environment_receipt") is not None:
                receipts.setdefault(key, []).append(row["environment_receipt"])
    shown = 0
    for row in read_rows(path):
        if any(value is not None and row.get(key) != value for key, value in (
            ("agent_id", args.agent_id), ("period", args.period), ("phase", args.phase),
        )):
            continue
        print(f"\nAgent {row.get('agent_id')} | period {row.get('period')} | "
              f"{row.get('phase')} | request {row.get('request_index')}")
        print(f"source={row.get('response_source')} | reasoning={row.get('reasoning_status')} | "
              f"finish={row.get('finish_reason')} | truncated={row.get('response_truncated')} | "
              f"redacted={row.get('redaction_applied')}")
        fields = row.get("reasoning_fields_returned") or []
        if not fields:
            print("[本次响应没有可供显示的推理字段；程序未补写。]")
        for field in fields:
            print(f"\n{field}:\n")
            value = row.get(field)
            print(value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, indent=2))
        content = row.get("content")
        rendered = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False, indent=2)
        print("\n最终正文：\n" + (rendered if content else "[无正文]"))
        print("工具调用：" + json.dumps(row.get("tool_calls") or [], ensure_ascii=False, indent=2))
        accepted = receipts.get((row.get("agent_id"), row.get("request_index")))
        if accepted:
            print("环境回执：" + json.dumps(accepted, ensure_ascii=False, indent=2))
        shown += 1
        if args.limit and shown >= args.limit:
            break
    print(f"\n显示 {shown} 条记录；内容是接口公开的文本，不代表完整内部计算。")
    return 0 if shown else 1


if __name__ == "__main__":
    raise SystemExit(main())
