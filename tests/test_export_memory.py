"""Readable notes must retain provenance and avoid inventing decision links."""

import json
import os

import pytest

from experiments.export_memory import export_memory


def state(agent, revision, note, periods=()):
    return {"owner_agent_id": agent, "schema_version": 2, "revision": revision,
            "content": {"note": note, "evidence_periods": list(periods)}}


def update(agent, period, request, before, after, **extra):
    return {"agent_id": agent, "period": period, "request_index": request,
            "memory_update_kind": "feedback", "previous_state": before, "new_state": after,
            "model_messages": [{"content": "DO_NOT_EXPORT_CONTEXT"}],
            "raw_output": "DO_NOT_EXPORT_RAW_OUTPUT", **extra}


def decision(agent, period, request, revision, **extra):
    return {"agent_id": agent, "action": "submit_decision", "ok": True,
            "participant_request_index": request, "private_memory_revision": revision,
            "submitted_amount": 40, "environment_receipt": {"period": period}, **extra}


def write_rows(path, rows):
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")


def test_export_preserves_notes_orders_requests_and_isolates_agents(tmp_path):
    raw_note = "只保存我自己的判断。\r\n保留原文尾部空格  "
    first = state(1, 1, raw_note, [1])
    second = state(1, 2, "证据不足，保留不确定性。", [1, 2])
    write_rows(tmp_path / "private_memory_updates.jsonl", [
        update(2, 2, 5, state(2, 0, ""), state(2, 1, "仅属于 Agent 2", [1])),
        update(1, 3, 9, first, second),
        update(1, 2, 4, state(1, 0, ""), first),
    ])
    write_rows(tmp_path / "agent_actions.jsonl", [
        decision(1, 3, 12, 2, submitted_amount=None, submitted_purchase=False),
        decision(2, 2, 7, 1, submitted_amount=76),
        decision(1, 1, 1, 0), decision(1, 2, 6, 1),
        decision(1, 2, 8, 1, ok=False, submitted_amount=999),
    ])
    sources = {p: p.read_bytes() for p in tmp_path.glob("*.jsonl")}
    result = export_memory(tmp_path)
    assert result["agent_count"] == 2 and result["update_count"] == 3
    assert result["decision_count"] == 4 and result["closing_update_count"] == 0
    agent1 = (tmp_path / "memory_text/agent_0001.txt").read_bytes().decode("utf-8")
    assert raw_note in agent1
    assert agent1.index("请求 4") < agent1.index("请求 9")
    assert "仅属于 Agent 2" not in agent1 and "转账金额 76" not in agent1
    assert "使用初始记忆版本 0" in agent1 and "决策请求 1" in agent1
    assert "依据轮次：1, 2" in agent1 and "投保选择 false" in agent1
    assert "转账金额 999" not in agent1 and "DO_NOT_EXPORT" not in agent1
    combined = (tmp_path / "memory_text/all_agents.txt").read_text(encoding="utf-8")
    assert combined.index("Agent 1\n") < combined.index("Agent 2\n")
    assert all(path.read_bytes() == data for path, data in sources.items())


def test_closing_never_explains_past_decisions_and_unchanged_notes_are_visible(tmp_path):
    before, after = state(1, 1, "原判断", [1]), state(1, 2, "原判断", [1])
    closing = state(1, 3, "最后一轮后修正了判断", [1, 2])
    write_rows(tmp_path / "private_memory_updates.jsonl", [
        update(1, 2, 4, before, after),
        update(1, 2, 8, after, closing, memory_update_kind="closing", task="closing"),
    ])
    write_rows(tmp_path / "agent_actions.jsonl", [
        decision(1, 2, 6, 2),
        # An inconsistent input must not attach a past choice to a closing note.
        decision(1, 1, 1, 3),
    ])
    result = export_memory(tmp_path)
    report = (tmp_path / "memory_text/agent_0001.txt").read_text(encoding="utf-8")
    assert result["closing_update_count"] == 1
    assert "本次笔记内容保持不变" in report and "终局版本不用于此前的经济决策" in report
    assert report.index("决策请求 6") < report.index("终局补记 · 请求 8")
    assert "未能关联到此前记忆更新的已接受决策" in report
    assert report.index("未能关联到此前") < report.index("决策请求 1")


def test_legacy_structured_memory_is_readable_without_schema_conversion(tmp_path):
    old = {"owner_agent_id": 1, "schema_version": 1, "revision": 1,
           "content": {"partner_assessment": "尚不了解搭档", "uncertainties": "需要继续观察",
                       "expected_partner_transfer": {"amount": None, "confidence": "unknown"},
                       "perceived_partner_expectation": {"amount": 100, "confidence": "low"},
                       "evidence_periods": [1]}}
    row = update(1, 2, 4, old, {**old, "revision": 2})
    row.pop("memory_update_kind")
    write_rows(tmp_path / "private_memory_updates.jsonl", [row])
    source = (tmp_path / "private_memory_updates.jsonl").read_bytes()
    export_memory(tmp_path)
    report = (tmp_path / "memory_text/agent_0001.txt").read_text(encoding="utf-8")
    assert "旧版结构化记忆" in report and "尚不了解搭档" in report
    assert '"amount": 100' in report and "需要继续观察" in report
    assert (tmp_path / "private_memory_updates.jsonl").read_bytes() == source


def test_empty_updates_and_initial_snapshot_are_not_reported_as_failure(tmp_path):
    folder = tmp_path / "agents/agent_0001"
    folder.mkdir(parents=True)
    (folder / "private_memory.json").write_text(json.dumps(state(1, 0, "")), encoding="utf-8")
    result = export_memory(tmp_path)
    report = (tmp_path / "memory_text/agent_0001.txt").read_text(encoding="utf-8")
    assert result["agent_count"] == 1 and result["update_count"] == 0
    assert "没有已保存的记忆更新" in report and "不能据此判断失败" in report
    assert "[空笔记]" in report and "最终记忆快照" in report
    empty = tmp_path / "empty"
    empty.mkdir()
    assert export_memory(empty)["agent_count"] == 0
    assert "不表示实验失败" in (empty / "memory_text/all_agents.txt").read_text(encoding="utf-8")


def test_bad_input_does_not_create_partial_output(tmp_path):
    source = tmp_path / "private_memory_updates.jsonl"
    write_rows(source, [update(1, 2, 4, state(1, 0, ""), state(1, 1, "判断"))])
    with source.open("a", encoding="utf-8") as handle:
        handle.write('{"broken":\n')
    original = source.read_bytes()
    with pytest.raises(ValueError, match="第 2 行"):
        export_memory(tmp_path)
    assert source.read_bytes() == original and not (tmp_path / "memory_text").exists()
    write_rows(source, [update(1, 2, 4, state(1, 0, ""), state(2, 1, "别人的状态"))])
    with pytest.raises(ValueError, match="其他参与者"):
        export_memory(tmp_path)
    assert not (tmp_path / "memory_text").exists()


@pytest.mark.parametrize("link_kind", ["symlink", "hardlink"])
def test_existing_output_cannot_overwrite_source_through_links(tmp_path, link_kind):
    source = tmp_path / "private_memory_updates.jsonl"
    write_rows(source, [])
    output = tmp_path / "memory_text"
    output.mkdir()
    target = output / "all_agents.txt"
    if link_kind == "symlink":
        target.symlink_to(source)
    else:
        os.link(source, target)
    original = source.read_bytes()
    with pytest.raises(ValueError, match="原始输入或符号链接"):
        export_memory(tmp_path)
    assert source.read_bytes() == original
