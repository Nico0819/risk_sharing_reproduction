"""Render recorded compulsory insurance assignments without inventing choices.

The runner supplies accepted environment assignments and settled period records.
This module performs no model calls and never changes those records.
"""

from __future__ import annotations

import html
from typing import Any

from protocol import paper_spec as spec


START = "<!-- forced-insurance-report:start -->"
END = "<!-- forced-insurance-report:end -->"


def _cell(value: Any) -> str:
    return html.escape(str(value)).replace("|", "&#124;").replace("\n", " ")


def _sort_id(value: Any) -> tuple[int, Any]:
    return (0, value) if type(value) is int else (1, str(value))


def _row(values) -> str:
    return "| " + " | ".join(_cell(value) for value in values) + " |"


def render_insurance_section(manifest, assignments, records) -> str:
    """Summarize original assignments separately from currently active coverage.

    Missing assignments remain missing, even after withdrawal. No participant
    decision, reason or receipt is inferred from a computer-generated assignment.
    """

    assignments, records = list(assignments), list(records)
    first = spec.INSURANCE_ACTIVE_START_PERIOD
    last = spec.INSURANCE_ACTIVE_END_PERIOD
    removal = spec.INSURANCE_REMOVAL_REVEALED_PERIOD
    reached_assignment = bool(assignments) or any(row["period"] >= first for row in records)
    configuration = manifest.get("insurance_assignment_configuration") or {}
    probabilities = configuration.get("probabilities") or {}
    lines = [
        START, "## 强制保险分配与收入", "",
        f"第 {first} 轮的保险状态由环境随机分配（environment_random），不是参与者的购买选择。"
        "参与者模型仍自行决定转账；保险分配没有参与者提交、投保理由或投保回执。", "",
        "分配概率来自本次实验配置；冻结协议尚未确定原论文的精确分配概率，"
        "因此本次配置不能直接视为论文参数。", "",
        "| 配置项 | 本次记录 |",
        "| --- | --- |",
        _row(("双方有保险的概率", probabilities.get("both_insured", "未记录"))),
        _row(("仅第一位成员有保险的概率", probabilities.get("first_only", "未记录"))),
        _row(("仅第二位成员有保险的概率", probabilities.get("second_only", "未记录"))),
        _row(("概率来源", configuration.get("probability_source", "未记录"))),
        _row(("分配随机数来源", configuration.get("rng_source", "未记录"))),
        _row(("分配随机种子", configuration.get("seed", "未记录"))),
        "", "第一、第二位成员的顺序沿用环境中固定 pair 的成员顺序；"
        "分配随机数流与收入、终止抽样分开。", "",
    ]
    if not reached_assignment:
        lines.extend([f"尚未到第 {first} 轮，尚未分配保险。未分配不能解释为未获保或拒保。", ""])
    elif not assignments:
        lines.extend(["运行记录已经到达保险期，但缺少分配记录，需要核对；不能按未获保处理。", ""])

    members: dict[Any, set[Any]] = {}
    for row in [*records, *assignments]:
        pair_id = row.get("pair_id")
        if pair_id is not None:
            members.setdefault(pair_id, set()).update(
                row[key] for key in ("agent_id", "partner_id") if row.get(key) is not None
            )
    indexed: dict[tuple[Any, Any], dict[str, Any]] = {}
    invalid: set[tuple[Any, Any]] = set()
    for row in assignments:
        key = (row.get("pair_id"), row.get("agent_id"))
        if (
            key in indexed
            or type(row.get("insured")) is not bool
            or row.get("assignment_source") != "environment_random"
            or row.get("period") != first
            or row.get("treatment") != spec.Treatment.FORCED_INSURANCE_25.value
        ):
            invalid.add(key)
        indexed[key] = row

    lines.extend([
        f"分配记录：{len(assignments)} / {manifest.get('agents', '未知')} 个参与者。", "",
        "### 各组原始分配", "",
        f"以下状态保留第 {first} 轮的原始分配。第 {removal} 轮撤销后，"
        "原分配仍保留，当轮有效保险为 false，不会改写成拒保。", "",
        "| pair | 成员及原始分配 | 获保人数 | 记录来源 | 保险期已结算轮数 | 保费合计 | 赔付合计 |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ])
    for pair_id in sorted(members, key=_sort_id):
        pair_members = sorted(members[pair_id], key=_sort_id)
        descriptions = []
        assigned_count = covered_count = 0
        problematic = len(pair_members) != spec.PAIR_SIZE
        for agent_id in pair_members:
            key = (pair_id, agent_id)
            assignment = indexed.get(key)
            if key in invalid:
                description = "分配记录需核对"
                problematic = True
            elif assignment is None:
                description = "缺少分配记录" if reached_assignment else "尚未分配"
            else:
                description = "有保险" if assignment["insured"] else "未获分配保险"
                assigned_count += 1
                covered_count += assignment["insured"] is True
            descriptions.append(f"Agent {agent_id}：{description}")
        if assigned_count == spec.PAIR_SIZE and covered_count == 0:
            problematic = True  # 25F excludes the no-insurance pair outcome.
        if problematic:
            count, source = "需核对", "分配记录需核对"
        elif assigned_count != spec.PAIR_SIZE:
            count = "未齐" if reached_assignment else "尚未分配"
            source = "记录未齐" if reached_assignment else "尚未分配"
        else:
            count, source = f"{covered_count}/{spec.PAIR_SIZE}", "environment_random"
        settled = [row for row in records if row["pair_id"] == pair_id and first <= row["period"] <= last]
        lines.append(_row((
            pair_id, "；".join(descriptions), count, source,
            len({row["period"] for row in settled}),
            sum(row["insurance_cost"] for row in settled),
            sum(row["insurance_payout"] for row in settled),
        )))
    if not members:
        lines.extend(["", "暂无可用的分组或分配记录。"])

    lines.extend([
        "", "### 已结算的保费和赔付", "",
        "金额为实验点数；有效保险人轮只统计当轮 insured=true 的参与者记录。", "",
        "| 阶段 | 已结算轮数 | 参与者记录数 | 有效保险人轮 | 保费合计 | 赔付合计 |",
        "| --- | --- | --- | --- | --- | --- |",
    ])
    final_period = max((row["period"] for row in records), default=0)
    for label, start, end in (
        (f"1—{first - 1}：引入前", 1, first - 1),
        (f"{first}—{last}：保险期", first, last),
        (f"{removal} 起：撤销后", removal, final_period),
    ):
        rows = [row for row in records if start <= row["period"] <= end]
        lines.append(_row((
            label, len({row["period"] for row in rows}), len(rows),
            sum(row["insured"] is True for row in rows),
            sum(row["insurance_cost"] for row in rows),
            sum(row["insurance_payout"] for row in rows),
        )))

    income_rows = [row for row in records if row["period"] == first]
    if income_rows:
        lines.extend([
            "", f"### 第 {first} 轮收入明细", "",
            "保险前收入 = 固定收入 + 随机收入；转账前收入 = 保险前收入 − 保费 + 赔付；"
            "最终收入再减转出、加转入。", "",
            "| Agent | pair | 当轮有保险 | 固定收入 | 随机收入 | 保险前收入 | 保费 | 赔付 | 转账前收入 | 转出 | 转入 | 最终收入 |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ])
        for row in sorted(income_rows, key=lambda row: (_sort_id(row["pair_id"]), _sort_id(row["agent_id"]))):
            lines.append(_row((
                row["agent_id"], row["pair_id"], "是" if row["insured"] is True else "否",
                row["fixed_income"], row["random_income"], row["fixed_income"] + row["random_income"],
                row["insurance_cost"], row["insurance_payout"], row["income_before_transfer"],
                row["transfer_sent"], row["transfer_received"], row["final_income"],
            )))
    lines.extend([
        "", "证据文件：[环境分配记录](insurance_assignments.jsonl)、"
        "[逐轮收入](period_records.jsonl)、[参与者转账及回执](agent_actions.jsonl)、"
        "[本次实验配置](manifest.json)。", END,
    ])
    return "\n".join(lines)
