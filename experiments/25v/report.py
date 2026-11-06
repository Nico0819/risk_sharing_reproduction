"""Render researcher-facing insurance evidence, without running any models.

This module never changes decisions, settlements, manifests or raw logs.
Only derived reports are updated; the previous summary contents are backed up.
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from protocol import paper_spec as spec  # noqa: E402
from experiments.report import add_report_link, write_visual_report  # noqa: E402


START = "<!-- insurance-report:start -->"
END = "<!-- insurance-report:end -->"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _cell(value: Any) -> str:
    return html.escape(str(value)).replace("|", "&#124;").replace("\n", " ")


def _render_pair_purchases(decisions, records, actions) -> list[str]:
    """Group accepted choices by recorded pair, retaining missing submissions."""

    members: dict[Any, set[Any]] = {}

    def remember_members(row):
        pair_id = row.get("pair_id")
        if pair_id is not None:
            members.setdefault(pair_id, set()).update(
                row[key] for key in ("agent_id", "partner_id")
                if row.get(key) is not None
            )

    for row in [*records, *decisions]:
        remember_members(row)
    observations = []
    for action in actions:
        observation = action.get("observation")
        if (
            action.get("action") == "get_observation"
            and action.get("ok") is True
            and isinstance(observation, dict)
        ):
            observations.append(observation)
            remember_members({
                "pair_id": action.get("pair_id"),
                "agent_id": observation.get("agent_id"),
                "partner_id": observation.get("partner_id"),
            })

    choices = {(row["pair_id"], row["agent_id"]): row["purchase"] for row in decisions}
    reached_offer = bool(decisions) or any(
        row["period"] >= spec.INSURANCE_DECISION_PERIOD for row in records
    ) or any(
        row.get("current_period", 0) >= spec.INSURANCE_DECISION_PERIOD for row in observations
    )
    lines = [
        "### 各组购买情况", "",
        f"购买情况按第 {spec.INSURANCE_DECISION_PERIOD} 轮环境接受的选择记录，保险撤销后仍保留原选择。"
        "保费和赔付只累计已结算保险期的实际金额。", "",
    ]
    if not members:
        return lines + ["暂无可用的分组记录。", ""]
    lines.extend([
        "| pair | 组内成员及选择 | 购买人数 | 本组情况 | 保险期已结算轮数 | 保费合计 | 赔付合计 |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ])
    for pair_id in sorted(members, key=str):
        pair_members = sorted(members[pair_id], key=str)
        descriptions = []
        bought = declined = 0
        invalid = False
        for agent_id in pair_members:
            key = (pair_id, agent_id)
            if key not in choices:
                choice = "未提交"
            elif choices[key] is True:
                choice = "购买"
                bought += 1
            elif choices[key] is False:
                choice = "不购买"
                declined += 1
            else:
                choice = "非法值，需核对"
                invalid = True
            descriptions.append(f"Agent {agent_id}：{choice}")
        if invalid or len(pair_members) != spec.PAIR_SIZE:
            status = "分组或选择记录需核对"
        elif bought + declined < spec.PAIR_SIZE:
            status = "决定未收齐" if reached_offer else "尚未到投保阶段"
        else:
            status = "双方购买" if bought == spec.PAIR_SIZE else "单方购买" if bought else "双方不购买"
        settled = [row for row in records if (
            row["pair_id"] == pair_id
            and spec.INSURANCE_ACTIVE_START_PERIOD <= row["period"] <= spec.INSURANCE_ACTIVE_END_PERIOD
        )]
        values = (
            pair_id, "；".join(descriptions), f"{bought}/{spec.PAIR_SIZE}", status,
            len({row["period"] for row in settled}),
            sum(row["insurance_cost"] for row in settled),
            sum(row["insurance_payout"] for row in settled),
        )
        lines.append("| " + " | ".join(_cell(value) for value in values) + " |")
    return lines + [""]


def render_insurance_section(manifest, decisions, records, actions) -> str:
    """Report accepted choices and observed effects, not assumed participation."""

    bought = sum(row["purchase"] is True for row in decisions)
    declined = sum(row["purchase"] is False for row in decisions)
    expected = manifest["agents"]
    source = "真实模型（llm）" if manifest["backend"] == "llm" else "离线测试策略（scripted），不是外部 API 决策"
    lines = [
        START, "## 保险机制与 Agent 决策", "",
        f"决策来源：{source}。下表只统计环境实际接受的提交，缺失不视为拒保。", "",
        f"已接受投保决定：{len(decisions)} / {expected}；购买：{bought}；明确拒保：{declined}。", "",
        *_render_pair_purchases(decisions, records, actions),
        "### 各 Agent 的提交与回执", "",
        "| Agent | pair | 决策轮次 | 选择 | 原始提交与回执核对 |",
        "| --- | --- | --- | --- | --- |",
    ]
    matched = []
    for decision in sorted(decisions, key=lambda row: (str(row["pair_id"]), str(row["agent_id"]))):
        action = next((row for row in actions if (
            row.get("action") == "submit_insurance_decision"
            and row.get("ok") is True
            and row.get("agent_id") == decision["agent_id"]
            and row.get("decision_id") == decision["decision_id"]
            and row.get("decision_source") == "participant_model"
            and row.get("participant_model_output", {}).get("decision_type") == "insurance"
            and row.get("participant_model_output", {}).get("decision_id") == decision["decision_id"]
            and row.get("participant_model_output", {}).get("purchase") is decision["purchase"]
            and row.get("submitted_purchase") is decision["purchase"]
            and (row.get("environment_receipt") or {}).get("decision_id") == decision["decision_id"]
            and (row.get("environment_receipt") or {}).get("period") == decision["period"]
            and (row.get("environment_receipt") or {}).get("purchase") is decision["purchase"]
            and (row.get("environment_receipt") or {}).get("status") in {"submitted", "already_recorded"}
        )), None)
        choice = "购买（true）" if decision["purchase"] is True else "拒保（false）" if decision["purchase"] is False else "非法值，需核对"
        check = "一致，已接受" if action is not None else "未找到匹配证据，需核对"
        lines.append("| " + " | ".join(_cell(v) for v in (
            decision["agent_id"], decision["pair_id"], decision["period"], choice, check,
        )) + " |")
        matched.append((decision, action))

    insurance_rows = [r for r in records if spec.INSURANCE_ACTIVE_START_PERIOD <= r["period"] <= spec.INSURANCE_ACTIVE_END_PERIOD]
    if len(decisions) == expected and declined == expected:
        lines.extend(["", "所有 Agent 都明确选择了拒保：保险选项已提供，但本次没有人购买。"])
        if insurance_rows and all(r["insurance_cost"] == r["insurance_payout"] == 0 for r in insurance_rows):
            lines.append("因此已结算保险期的保费与赔付均为 0；本次轨迹没有触发投保后的扣费/赔付分支。")
    elif len(decisions) < expected:
        lines.extend(["", "投保决定尚未收齐或运行尚未到达投保阶段，不能把未提交解释为拒保。"])

    observations = [a["observation"] for a in actions if a.get("action") == "get_observation" and isinstance(a.get("observation"), dict)]
    offered = {o["agent_id"] for o in observations if o.get("current_period") == spec.INSURANCE_DECISION_PERIOD and o.get("task") == "insurance" and "insurance_offer" in o}
    removed = {o["agent_id"] for o in observations if o.get("current_period") == spec.INSURANCE_REMOVAL_REVEALED_PERIOD and "announcement" in o}
    lines.extend([
        "", f"第 {spec.INSURANCE_DECISION_PERIOD} 轮投保任务及条款的观察记录：{len(offered)} 个 Agent；"
        f"第 {spec.INSURANCE_REMOVAL_REVEALED_PERIOD} 轮撤销通知的观察记录：{len(removed)} 个 Agent。", "",
        "### 各阶段实际扣费和赔付", "",
        "这里只汇总已结算记录，金额为实验点数；投保人轮是有效投保的参与者—轮次数。", "",
        "| 阶段 | 已结算轮数 | 参与者记录数 | 投保人轮 | 保费合计 | 赔付合计 |",
        "| --- | --- | --- | --- | --- | --- |",
    ])
    last = max((row["period"] for row in records), default=0)
    windows = (
        (f"1—{spec.INSURANCE_ACTIVE_START_PERIOD - 1}：引入前", 1, spec.INSURANCE_ACTIVE_START_PERIOD - 1),
        (f"{spec.INSURANCE_ACTIVE_START_PERIOD}—{spec.INSURANCE_ACTIVE_END_PERIOD}：保险期", spec.INSURANCE_ACTIVE_START_PERIOD, spec.INSURANCE_ACTIVE_END_PERIOD),
        (f"{spec.INSURANCE_REMOVAL_REVEALED_PERIOD} 起：撤销后", spec.INSURANCE_REMOVAL_REVEALED_PERIOD, last),
    )
    for label, start, end in windows:
        rows = [r for r in records if start <= r["period"] <= end]
        values = (label, len({r["period"] for r in rows}), len(rows),
                  sum(r["insured"] is True for r in rows), sum(r["insurance_cost"] for r in rows),
                  sum(r["insurance_payout"] for r in rows))
        lines.append("| " + " | ".join(_cell(v) for v in values) + " |")

    first_rows = [r for r in records if r["period"] == spec.INSURANCE_ACTIVE_START_PERIOD]
    if first_rows:
        lines.extend([
            "", f"### 第 {spec.INSURANCE_ACTIVE_START_PERIOD} 轮收入明细", "",
            "转账前收入 = 固定收入 + 随机收入 − 保费 + 赔付；最终收入再减转出、加转入。", "",
            "| Agent | pair | 已投保 | 固定收入 | 随机收入 | 保费 | 赔付 | 转账前收入 | 转出 | 转入 | 最终收入 |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ])
        for row in first_rows:
            values = (row["agent_id"], row["pair_id"], "是" if row["insured"] else "否",
                      row["fixed_income"], row["random_income"], row["insurance_cost"], row["insurance_payout"],
                      row["income_before_transfer"], row["transfer_sent"], row["transfer_received"], row["final_income"])
            lines.append("| " + " | ".join(_cell(v) for v in values) + " |")

    lines.extend(["", "### Agent 投保决策原文", "",
                  "以下是已接受提交的原文（含理由与明确选择），不是事后补写的理由，也不保证模型的经济解释正确。"])
    for decision, action in matched:
        lines.extend(["", f"#### Agent {_cell(decision['agent_id'])}（pair {_cell(decision['pair_id'])}）", ""])
        raw = action.get("participant_raw_request") if action is not None else None
        if raw:
            lines.extend("> " + html.escape(line, quote=False) for line in str(raw).splitlines())
        else:
            lines.append("未找到匹配的原始请求；不推测决策理由。")
    lines.extend(["", "证据文件：[保险选择](insurance_decisions.jsonl)、[Agent 原文及回执](agent_actions.jsonl)、"
                  "[逐轮收入](period_records.jsonl)。", END])
    return "\n".join(lines)


def refresh_summary(run_dir: Path) -> Path:
    """Refresh only this derived section, preserving a first-update backup."""

    run_dir = run_dir.expanduser().resolve()
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("treatment") != spec.Treatment.VOLUNTARY_INSURANCE_25.value:
        raise ValueError("This reporter only accepts a recorded 25V run.")
    decisions = read_jsonl(run_dir / "insurance_decisions.jsonl")
    records = read_jsonl(run_dir / "period_records.jsonl")
    section = render_insurance_section(manifest, decisions, records, read_jsonl(run_dir / "agent_actions.jsonl"))
    path = run_dir / "summary.md"
    original = path.read_text(encoding="utf-8")
    if START in original or END in original:
        if original.count(START) != 1 or original.count(END) != 1 or original.index(START) >= original.index(END):
            raise ValueError("Invalid insurance summary markers; leaving the summary unchanged.")
        start, end = original.index(START), original.index(END) + len(END)
        updated = original[:start] + section + original[end:]
    else:
        anchor = "## Pair-period settlements"
        if anchor not in original:
            raise ValueError("Expected summary section is missing; leaving the summary unchanged.")
        updated = original.replace(anchor, section + "\n\n" + anchor, 1)
    write_visual_report(run_dir, manifest, records, decisions)
    updated = add_report_link(updated)
    if updated != original:
        backup = run_dir / "summary.before-insurance-report.md"
        try:
            with backup.open("x", encoding="utf-8") as handle:
                handle.write(original)
        except FileExistsError:
            pass  # preserve the first backup, including a previous user edit
        path.write_text(updated, encoding="utf-8")
    return path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    path = refresh_summary(args.run_dir)
    print(f"Updated insurance evidence in {path}; raw records unchanged; no model calls.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
