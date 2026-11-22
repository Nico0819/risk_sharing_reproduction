"""Export saved reasoning into readable UTF-8 text, without model calls."""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from custom.agents.reasoning_trace import PROVIDER_REASONING_FIELDS
from custom.participant_trace import TRACE_FILE, redact_credentials


PHASES = {"decision": "经济决策", "memory_update": "更新私人记忆", "deliberation": "额外评估"}
TASKS = {"transfer": "转账", "waiting": "等待", "wait": "等待", "insurance": "投保"}
STATUSES = {"available": "已返回", "empty": "推理字段为空", "missing": "未返回推理字段",
            "request_failed": "请求失败"}


def _structured_text(value: Any) -> list[str]:
    """Read text only inside a reasoning field; omit signatures/ciphertext."""
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, list):
        return [text for item in value for text in _structured_text(item)]
    if not isinstance(value, dict):
        return []
    kind = str(value.get("type", "")).lower()
    if any(marker in kind for marker in ("encrypted", "redacted", "signature")):
        return []
    if kind and kind not in {"text", "thinking", "reasoning", "summary", "summary_text", "reasoning_text"} and not kind.startswith("reasoning."):
        return []
    return [text for key in ("text", "thinking", "reasoning", "summary", "content")
            if key in value for text in _structured_text(value[key])]


def reasoning_texts(row: dict[str, Any]) -> list[tuple[str, str]]:
    """Retain original text; deduplicate complete copies across provider fields."""
    result = []
    seen = set()
    for field in PROVIDER_REASONING_FIELDS:
        value = row.get(field)
        if field == "reasoning_content":
            text = value if isinstance(value, str) else ""
        else:
            text = "\n\n".join(_structured_text(value))
        normalized = text.replace("\r\n", "\n").strip()
        if normalized and normalized not in seen:
            seen.add(normalized)
            result.append((field, text))
    return result


def _order(row: dict[str, Any]) -> tuple[int, int, int]:
    def number(key: str) -> int:
        value = row.get(key)
        return value if type(value) is int else -1
    return number("agent_id"), number("period"), number("request_index")


def render_text(rows: list[dict[str, Any]], source_name: str) -> str:
    parts = ["参与者推理记录", f"来源：{source_name}",
             f"共 {len(rows)} 条调用记录，其中 {sum(bool(reasoning_texts(row)) for row in rows)} 条包含可读推理。",
             "以下文字直接摘自接口返回的推理字段，未翻译、概括或补写。",
             "记录中的缺失、截断和脱敏状态保留在各条目下。"]
    previous_agent = None
    for row in sorted(rows, key=_order):
        agent = row["agent_id"]
        if agent != previous_agent:
            parts.extend(["", "=" * 64, f"Agent {agent}", "=" * 64])
            previous_agent = agent
        phase = PHASES.get(row.get("phase"), row.get("phase") or "未知阶段")
        task = TASKS.get(row.get("task"), row.get("task") or "未记录任务")
        parts.extend(["", f"第 {row.get('period', '?')} 轮 · {phase} · {task} · 请求 {row.get('request_index', '?')}",
                      "-" * 64])
        origin = {"provider": "模型接口", "scripted": "离线脚本"}.get(row.get("response_source"), "未记录")
        status = STATUSES.get(row.get("reasoning_status"), row.get("reasoning_status") or "未记录")
        parts.append(f"来源：{origin}；推理状态：{status}")
        if row.get("response_truncated"):
            parts.append("[响应被截断；下面仅为已保存的内容。]")
        if row.get("redaction_applied"):
            parts.append("[原始日志经过凭据脱敏。]")
        if row.get("response_complete") is False and not row.get("response_truncated"):
            parts.append("[响应未完整结束。]")
        if row.get("reasoning_status") == "request_failed":
            parts.append(f"[请求异常：{row.get('error_type') or '未记录类型'}]")
        texts = reasoning_texts(row)
        if not texts:
            parts.extend(["", "[没有可读推理正文；不使用最终回答或私人记忆内容补充。]"])
        else:
            for field, text in texts:
                if len(texts) > 1:
                    parts.append(f"\n推理字段：{field}")
                parts.extend(["", text])
    return "\n".join(parts) + "\n"


def export_reasoning(source: Path, output_dir: Path | None = None) -> list[Path]:
    source = Path(source)
    if source.is_dir():
        source = source / "participant_reasoning.jsonl"
    rows = []
    with source.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"第 {line_number} 行不是完整 JSON：{exc.msg}") from exc
            if not isinstance(row, dict) or type(row.get("agent_id")) is not int or row["agent_id"] < 1:
                raise ValueError(f"第 {line_number} 行缺少合法的整数 agent_id")
            rows.append(row)
    if not rows:
        raise ValueError("源文件没有推理调用记录")

    grouped = defaultdict(list)
    for row in rows:
        grouped[row["agent_id"]].append(row)
    output_dir = Path(output_dir) if output_dir is not None else source.parent / "reasoning_text"
    outputs = {output_dir / "all_agents.txt": rows}
    outputs.update({output_dir / f"agent_{agent:04d}.txt": grouped[agent] for agent in sorted(grouped)})
    # Refuse symlinks so regenerating text cannot overwrite a raw input file.
    for path in outputs:
        if path.is_symlink() or path.resolve() == source.resolve():
            raise ValueError(f"导出文件不能指向原始日志或符号链接：{path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    source_name = f"{source.parent.name}/{source.name}"
    for path, selected in outputs.items():
        with path.open("w", encoding="utf-8", newline="") as handle:
            handle.write(render_text(selected, source_name))
    return list(outputs)


def finalize_reasoning_export(run_dir: Path) -> None:
    """Export saved calls after shutdown, without changing the run's outcome.

    Called from each runner's finally block after asyncio has joined/cancelled
    participant tasks, including closing memory updates. Diagnostics are also
    best-effort so a full disk cannot hide the original experiment exception.
    """
    run_dir = Path(run_dir)
    try:
        source = run_dir / TRACE_FILE
        if source.exists():
            with source.open(encoding="utf-8") as handle:
                has_records = any(line.strip() for line in handle)
        else:
            has_records = False
        if has_records:
            paths = export_reasoning(source)
            event = {"event": "reasoning_export_finished",
                     "output_files": [str(path.relative_to(run_dir)) for path in paths]}
            summary_note = "推理正文已自动导出：[全部 Agent](reasoning_text/all_agents.txt)；个人文件位于 `reasoning_text/agent_XXXX.txt`。"
        else:
            event = {"event": "reasoning_export_skipped", "reason": "no_saved_calls"}
            summary_note = "尚无已保存的参与者调用记录，未生成推理文本。"
    except Exception as exc:
        event = {"event": "reasoning_export_failed", "error_type": type(exc).__name__,
                 "error": redact_credentials(str(exc))}
        summary_note = "推理文本自动导出失败；原始日志未修改，原因见 `run.log`。可用 `experiments.export_reasoning` 重新导出。"
        logging.getLogger(__name__).warning("Reasoning export failed: %s", event["error"])

    # Write diagnostics independently; a missing/unwritable summary must not
    # prevent the run log from explaining why the text export failed.
    try:
        with (run_dir / "run.log").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"timestamp": datetime.now(timezone.utc).isoformat(),
                                     **event}, ensure_ascii=False) + "\n")
    except Exception as exc:
        logging.getLogger(__name__).warning("Cannot log reasoning export: %s", redact_credentials(str(exc)))
    try:
        summary_path = run_dir / "summary.md"
        if summary_path.exists():
            with summary_path.open("a", encoding="utf-8") as handle:
                handle.write("\n" + summary_note + "\n")
    except Exception as exc:
        logging.getLogger(__name__).warning("Cannot link reasoning export: %s", redact_credentials(str(exc)))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="将推理日志导出为按 Agent、轮次排列的纯文本；不调用模型。")
    parser.add_argument("source", type=Path, help="运行目录或 participant_reasoning.jsonl 文件路径")
    parser.add_argument("--output-dir", type=Path, help="导出目录；默认是运行目录下的 reasoning_text")
    args = parser.parse_args(argv)
    try:
        paths = export_reasoning(args.source, args.output_dir)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print("已导出推理正文（原始日志未修改）：")
    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
