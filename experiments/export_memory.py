"""Render saved private notes and their decision links without model calls."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path.name} 第 {number} 行不是完整 JSON：{exc.msg}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"{path.name} 第 {number} 行必须是 JSON 对象")
        rows.append(row)
    return rows


def _agent(value: Any) -> int:
    if type(value) is not int or value < 1:
        raise ValueError("记忆或决策记录缺少合法的整数 agent_id")
    return value


def _validate_state(state: Any, agent: int) -> None:
    if not isinstance(state, dict) or not isinstance(state.get("content"), dict):
        raise ValueError(f"Agent {agent} 的记忆状态缺少 content 对象")
    if state.get("owner_agent_id", agent) != agent:
        raise ValueError(f"Agent {agent} 的记忆记录混入了其他参与者的状态")
    revision = state.get("revision")
    if type(revision) is not int or revision < 0:
        raise ValueError(f"Agent {agent} 的记忆 revision 不合法")
    content = state["content"]
    if "note" in content and not isinstance(content["note"], str):
        raise ValueError(f"Agent {agent} 的私人笔记必须是文本")
    periods = content.get("evidence_periods", [])
    if not isinstance(periods, list) or any(type(n) is not int or n < 1 for n in periods):
        raise ValueError(f"Agent {agent} 的依据轮次不合法")


def _kind(row: dict[str, Any]) -> str:
    if row.get("memory_update_kind") == "closing" or row.get("task") == "closing":
        return "closing"
    return row.get("memory_update_kind") or "legacy"


def _request_order(row: dict[str, Any], key: str = "request_index") -> tuple[int, int]:
    value = row.get(key)
    period = row.get("period")
    return (value if type(value) is int else -1, period if type(period) is int else -1)


def _state_lines(state: dict[str, Any]) -> list[str]:
    content = state["content"]
    lines = [f"版本：{state['revision']}"]
    if "note" in content:
        lines.append(content["note"] if content["note"].strip() else "[空笔记]")
    else:
        lines.append("[旧版结构化记忆，以下保留原有字段含义]")
        for key, label in (
            ("partner_assessment", "对搭档的判断"),
            ("expected_partner_transfer", "预期搭档转账"),
            ("perceived_partner_expectation", "认为搭档对自己的预期"),
            ("uncertainties", "尚不确定的事情"),
        ):
            value = content.get(key)
            rendered = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
            lines.append(f"{label}：{rendered if rendered else '[空]'}")
    periods = content.get("evidence_periods", [])
    lines.append("依据轮次：" + (", ".join(map(str, periods)) if periods else "无"))
    return lines


def _decision_line(row: dict[str, Any]) -> str:
    receipt = row.get("environment_receipt")
    receipt = receipt if isinstance(receipt, dict) else {}
    period = receipt.get("period", row.get("period", "?"))
    value = "金额或投保选择未记录"
    if row.get("submitted_amount") is not None:
        value = f"转账金额 {row['submitted_amount']}"
    elif row.get("submitted_purchase") is not None:
        value = f"投保选择 {json.dumps(row['submitted_purchase'], ensure_ascii=False)}"
    return (f"第 {period} 轮 · 决策请求 {row.get('participant_request_index', '?')} · "
            f"记忆版本 {row.get('private_memory_revision', '未记录')} · {value}")


def _render(agents: list[int], updates: dict[int, list], decisions: dict[int, list],
            snapshots: dict[int, dict], source: str) -> str:
    count = sum(len(updates[agent]) for agent in agents)
    parts = ["参与者私人记忆记录", f"来源：{source}",
             f"共 {len(agents)} 个 Agent，{count} 次已保存的记忆更新。",
             "笔记正文直接来自已保存状态，未翻译、概括或补写。",
             "决策关联依据日志中的 Agent 和记忆版本；关联不证明记忆导致了该选择。",
             "终局补记发生在决策之后，仅用于收尾记录，不能解释此前的决定。"]
    if not agents:
        parts.append("\n[未发现参与者记忆状态、更新或已接受决策记录；这不表示实验失败。]")
    for agent in agents:
        parts.extend(["", "=" * 64, f"Agent {agent}", "=" * 64])
        rows = sorted(updates[agent], key=_request_order)
        actions = sorted(decisions[agent], key=lambda row: _request_order(row, "participant_request_index"))
        linked = set()
        initial = [row for row in actions if row.get("private_memory_revision") == 0]
        if initial:
            parts.extend(["", "使用初始记忆版本 0 的已接受决策："])
            for row in initial:
                parts.append(_decision_line(row))
                linked.add(id(row))
        if not rows:
            parts.append("\n[没有已保存的记忆更新；可能尚无新的合法反馈，不能据此判断失败。]")
        for row in rows:
            before, after = row["previous_state"], row["new_state"]
            kind = _kind(row)
            label = {"feedback": "新反馈更新", "closing": "终局补记"}.get(kind, "更新（旧记录未标注触发类型）")
            parts.extend(["", f"第 {row.get('period', '?')} 轮 · {label} · 请求 {row.get('request_index', '?')}",
                          "-" * 64])
            condition = row.get("memory_condition")
            if condition == "fresh":
                parts.extend([
                    "实验条件：fresh（当轮反思）",
                    "上一版仍保存在日志和文件中供审阅；本次更新模型没有读取它。",
                    "本次更新实际输入：空笔记、空依据。",
                    "更新前保存的状态（仅供审阅，不是本次更新的模型输入）：",
                ])
            else:
                if condition == "persistent":
                    parts.append("实验条件：persistent（持久私人记忆）")
                parts.append("更新前：")
            parts.extend([*_state_lines(before), "", "更新后：", *_state_lines(after)])
            if before["content"] == after["content"]:
                parts.append("[本次笔记内容保持不变。]")
            if kind == "closing":
                parts.append("[终局版本不用于此前的经济决策。]")
                continue
            matching = []
            for action in actions:
                if id(action) in linked or action.get("private_memory_revision") != after["revision"]:
                    continue
                request = action.get("participant_request_index")
                update_request = row.get("request_index")
                if type(request) is int and type(update_request) is int and request <= update_request:
                    continue
                matching.append(action)
                linked.add(id(action))
            parts.append("\n使用该版本的已接受决策：")
            parts.extend(_decision_line(action) for action in matching)
            if not matching:
                parts.append("[未找到关联的已接受决策；等待任务或尚未提交的决定可能没有此类记录。]")
        unlinked = [row for row in actions if id(row) not in linked]
        if unlinked:
            parts.append("\n未能关联到此前记忆更新的已接受决策（不推测所用笔记）：")
            parts.extend(_decision_line(row) for row in unlinked)
        if agent in snapshots:
            parts.extend(["", "导出时保存的最终记忆快照（不反推过去的决策输入）：",
                          *_state_lines(snapshots[agent])])
    return "\n".join(parts) + "\n"


def export_memory(run_dir: Path, output_dir: Path | None = None) -> dict[str, Any]:
    """Export all revisions, including legacy notes; never feed reports to agents."""
    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        raise ValueError(f"运行目录不存在：{run_dir}")
    source_paths = [run_dir / "private_memory_updates.jsonl", run_dir / "agent_actions.jsonl"]
    updates: dict[int, list] = defaultdict(list)
    decisions: dict[int, list] = defaultdict(list)
    snapshots = {}
    for row in _read_jsonl(source_paths[0]):
        agent = _agent(row.get("agent_id"))
        for field in ("previous_state", "new_state"):
            _validate_state(row.get(field), agent)
        updates[agent].append(row)
    for row in _read_jsonl(source_paths[1]):
        if row.get("action") == "submit_decision" and row.get("ok") is True:
            decisions[_agent(row.get("agent_id"))].append(row)
    for path in sorted((run_dir / "agents").glob("agent_*/private_memory.json")):
        if path.is_symlink():
            raise ValueError(f"记忆快照不能是符号链接：{path}")
        try:
            state = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path} 不是完整 JSON：{exc.msg}") from exc
        agent = _agent(state.get("owner_agent_id") if isinstance(state, dict) else None)
        if path.parent.name != f"agent_{agent:04d}":
            raise ValueError(f"记忆快照目录与参与者身份不一致：{path}")
        _validate_state(state, agent)
        snapshots[agent] = state
        source_paths.append(path)
    agents = sorted(set(updates) | set(decisions) | set(snapshots))
    output_dir = Path(output_dir) if output_dir is not None else run_dir / "memory_text"
    outputs = {output_dir / "all_agents.txt": agents}
    outputs.update({output_dir / f"agent_{agent:04d}.txt": [agent] for agent in agents})
    if output_dir.is_symlink():
        raise ValueError("导出目录不能是符号链接")
    for path in outputs:
        if path.is_symlink() or any(
            path.resolve() == source.resolve()
            or (path.exists() and source.exists() and path.samefile(source)) for source in source_paths
        ):
            raise ValueError(f"导出文件不能指向原始输入或符号链接：{path}")
    # Parse and validate all input before writing any report.
    output_dir.mkdir(parents=True, exist_ok=True)
    for path, selected in outputs.items():
        with path.open("w", encoding="utf-8", newline="") as handle:
            handle.write(_render(selected, updates, decisions, snapshots, run_dir.name))
    return {
        "agent_count": len(agents), "update_count": sum(map(len, updates.values())),
        "closing_update_count": sum(_kind(row) == "closing" for rows in updates.values() for row in rows),
        "decision_count": sum(map(len, decisions.values())),
        "output_files": [str(path) for path in outputs],
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="导出按 Agent 排列的私人记忆更新和关联决策；不调用模型。")
    parser.add_argument("run_dir", type=Path, help="实验运行目录")
    parser.add_argument("--output-dir", type=Path, help="导出目录；默认是运行目录下的 memory_text")
    args = parser.parse_args(argv)
    try:
        result = export_memory(args.run_dir, args.output_dir)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(f"已导出 {result['agent_count']} 个 Agent 的 {result['update_count']} 次私人记忆更新：")
    for path in result["output_files"]:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
