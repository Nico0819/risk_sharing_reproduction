"""Research-text export must preserve source text and distinguish missing data."""

import json

import pytest

from experiments.export_reasoning import export_reasoning, reasoning_texts


def write_trace(tmp_path, rows):
    path = tmp_path / "participant_reasoning.jsonl"
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")
    return path


def entry(agent, period, request, text, **extra):
    return {"agent_id": agent, "period": period, "request_index": request,
            "phase": "decision", "task": "transfer", "response_source": "provider",
            "reasoning_status": "available", "reasoning_content": text,
            "content": "DO_NOT_EXPORT_FINAL_ANSWER", "tool_calls": [{"secret": "DO_NOT_EXPORT_TOOL"}],
            **extra}


def test_export_preserves_long_text_sorts_calls_and_leaves_raw_log_unchanged(tmp_path):
    long_text = "原始开头\r\n" + "这是完整的一段推理。\n" * 1500 + "原始结尾  "
    source = write_trace(tmp_path, [
        entry(2, 2, 8, "Agent2的推理"),
        entry(1, 2, 7, "第二轮的推理"),
        entry(1, 1, 3, long_text),
        entry(1, 1, 1, "先更新记忆", phase="memory_update"),
    ])
    before = source.read_bytes()
    paths = export_reasoning(tmp_path)
    assert {path.name for path in paths} == {"all_agents.txt", "agent_0001.txt", "agent_0002.txt"}
    assert source.read_bytes() == before
    # Bytes preserve the returned CRLF and trailing whitespace as well as long text.
    agent1 = (tmp_path / "reasoning_text/agent_0001.txt").read_bytes().decode("utf-8")
    assert long_text in agent1 and len(long_text) > 10000
    assert agent1.index("先更新记忆") < agent1.index("原始开头") < agent1.index("第二轮的推理")
    assert "Agent2的推理" not in agent1
    combined = paths[0].read_text(encoding="utf-8")
    assert combined.index("Agent 1") < combined.index("Agent 2")
    assert "DO_NOT_EXPORT" not in combined


def test_structured_reasoning_omits_signatures_and_ciphertext_without_final_answer_fallback():
    row = {
        "reasoning_content": "同一段正文",
        "reasoning_details": [{"type": "reasoning.text", "text": "同一段正文"},
                              {"type": "reasoning.encrypted", "data": "SECRET_CIPHERTEXT"}],
        "thinking_blocks": [{"type": "thinking", "thinking": "另一段正文", "signature": "SECRET_SIGNATURE"}],
        "reasoning_items": [{"type": "reasoning", "encrypted_content": "SECRET_ENCRYPTED",
                             "summary": [{"type": "summary_text", "text": "提供的摘要"}]}],
        "content": "DO_NOT_EXPORT_FINAL",
    }
    texts = reasoning_texts(row)
    assert [text for _, text in texts] == ["同一段正文", "另一段正文", "提供的摘要"]
    assert "SECRET" not in str(texts) and "DO_NOT_EXPORT" not in str(texts)
    assert reasoning_texts({"content": "Only a final answer"}) == []
    assert reasoning_texts({"thinking_blocks": [{"type": "redacted_thinking", "text": "SECRET"}]}) == []
    assert reasoning_texts({"thinking_blocks": [{"type": "signature", "text": "SECRET"}]}) == []
    assert reasoning_texts({"reasoning_items": [{"type": "message", "content": [
        {"type": "output_text", "text": "DO_NOT_EXPORT_FINAL"},
    ]}]}) == []
    # Repeated passages within a single returned text remain untouched.
    assert reasoning_texts({"reasoning_content": "重复\n重复"}) == [("reasoning_content", "重复\n重复")]


def test_missing_empty_failed_and_truncated_records_remain_visible(tmp_path):
    source = write_trace(tmp_path, [
        entry(1, 1, 1, "", reasoning_status="empty"),
        entry(1, 1, 2, None, reasoning_status="missing"),
        entry(1, 1, 3, None, reasoning_status="request_failed", error_type="TimeoutError", response_complete=False),
        entry(1, 1, 4, "只返回了这部分", response_truncated=True, redaction_applied=True),
    ])
    exported = export_reasoning(source)[0].read_text(encoding="utf-8")
    assert "共 4 条调用记录，其中 1 条包含可读推理" in exported
    for label in ("推理字段为空", "未返回推理字段", "请求失败", "TimeoutError", "响应被截断", "凭据脱敏"):
        assert label in exported
    assert exported.count("[没有可读推理正文") == 3
    assert "DO_NOT_EXPORT" not in exported


def test_malformed_source_does_not_produce_a_partial_export(tmp_path):
    source = write_trace(tmp_path, [entry(1, 1, 1, "有效记录")])
    with source.open("a", encoding="utf-8") as handle:
        handle.write('{"unfinished":\n')
    before = source.read_bytes()
    with pytest.raises(ValueError, match="第 2 行"):
        export_reasoning(source)
    assert not (tmp_path / "reasoning_text").exists()
    assert source.read_bytes() == before
