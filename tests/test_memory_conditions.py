"""Fresh reflection and persistent notes differ only in old-note availability."""

from __future__ import annotations

import copy
import json

from experiments.export_memory import export_memory
from experiments.fixed_preferences_memory_v1 import run_smoke as version
from test_fixed_preferences_integration import (
    isolated_runtime,  # noqa: F401 -- real framework, isolated provider settings
    read_json,
    rows,
)


def _memory(messages):
    blocks = [message["content"].split("<private_memory>", 1)[1].split("</private_memory>", 1)[0]
              for message in messages if isinstance(message.get("content"), str)
              and "<private_memory>" in message["content"]]
    assert len(blocks) == 1
    return json.loads(blocks[0])


def _canary(agent, period, task):
    return f"PRIVATE_CANARY_AGENT_{agent}_PERIOD_{period}_TASK_{task}_END"


def test_real_framework_fresh_and_persistent_conditions_have_matched_calls_and_information(tmp_path, monkeypatch):
    original = version.ScriptedLLMClient.call
    observed = {"persistent": [], "fresh": []}
    condition = "persistent"

    async def spy(self, model=None, messages=None, **kwargs):
        supplied = list(messages or [])
        update = bool(supplied) and "<private_memory_update>" in str(supplied[-1].get("content", ""))
        turn = self._participant_turn(supplied) if self.kind == "participant" else None
        record = {"kind": self.kind, "update": update, "turn": copy.deepcopy(turn),
                  "messages": copy.deepcopy(supplied), "kwargs": copy.deepcopy(kwargs)}
        observed[condition].append(record)
        response = await original(self, model=model, messages=messages, **kwargs)
        if self.kind == "participant" and update:
            agent, observation = turn
            reply = response.choices[0].message
            content = json.loads(reply.content)
            if agent in {1, 2}:
                content["note"] = _canary(agent, observation["current_period"], observation["task"])
            elif agent == 3:
                # Empty is a valid participant choice, even after new feedback.
                content = {"note": "", "evidence_periods": []}
            else:
                # Retaining identical text is also valid after later observations.
                content = {"note": "I have no additional judgment to record.", "evidence_periods": []}
            reply.content = json.dumps(content)
            assert not reply.tool_calls
            record["output_content"] = content
        return response

    monkeypatch.setattr(version.ScriptedLLMClient, "call", spy)
    manifests = {}
    for condition in ("persistent", "fresh"):
        root = tmp_path / condition
        manifests[condition] = version.run_experiment(
            backend="scripted", memory_condition=condition,
            participants=4, agent_concurrency=4, completed_period_limit=3,
            seed=2026, preference_seed=2026, request_limit=100, run_dir=root,
        )
        manifest = manifests[condition]
        assert manifest["success"] is True
        assert manifest["memory_condition"] == condition
        assert manifest["agent_class"] == "FixedPreferenceMemoryParticipant"
        assert manifest["router"] == "ReActRouter"
        assert manifest["closing_memory_complete"] is True, manifest["closing_memory_failures"]
        assert manifest["memory_experiment_complete"] is True
        assert manifest["private_memory_update_requests"] == 12
        assert manifest["participant_model_requests"] == 24
        calls = [row for row in observed[condition] if row["kind"] == "participant"]
        assert len(calls) == 24
        assert len([row for row in calls if row["update"]]) == 12
        assert not any(row["update"] and row["turn"][1]["current_period"] == 1 for row in calls)
        closing = [row for row in calls if row["turn"][1]["task"] == "closing"]
        assert len(closing) == 4 and all(row["update"] for row in closing)
        contexts = rows(root / "participant_contexts.jsonl")
        assert all(row["phase"] == "memory_update" for row in contexts if row["task"] == "closing")
        for call in calls:
            agent, observation = call["turn"]
            memory = _memory(call["messages"])
            text = json.dumps(call["messages"])
            members = {agent, observation["partner_id"]}
            assert all({r["agent_id"], r["partner_id"]} == members for r in observation["history"])
            assert all(r["period"] <= 3 for r in observation["history"])
            for other in set(range(1, 5)) - {agent}:
                assert f"PRIVATE_CANARY_AGENT_{other}_" not in text
            if call["update"] and condition == "fresh":
                assert memory["content"] == {"note": "", "evidence_periods": []}
                assert memory["evidence"] == []
                assert "PRIVATE_CANARY_AGENT_" not in text
                assert "I have no additional judgment to record." not in text
            if agent in {1, 2} and observation["current_period"] == 3 and observation["task"] != "closing":
                if call["update"] and condition == "persistent":
                    assert memory["content"]["note"].startswith(f"PRIVATE_CANARY_AGENT_{agent}_PERIOD_2_")
                elif not call["update"]:
                    # Both arms make the current decision using the newly written note.
                    assert memory["content"]["note"] == _canary(agent, 3, observation["task"])
        updates = rows(root / "private_memory_updates.jsonl")
        assert all(row["memory_condition"] == condition for row in updates)
        assert sum(row["memory_update_kind"] == "closing" for row in updates) == 4
        empty_updates = [row for row in updates if row["agent_id"] == 3]
        assert len(empty_updates) == 3
        assert all(row["new_state"]["content"] == {"note": "", "evidence_periods": []} for row in empty_updates)
        unchanged = [row for row in updates if row["agent_id"] == 4 and row["new_state"]["revision"] > 1]
        assert len(unchanged) == 2
        assert all(row["previous_state"]["content"] == row["new_state"]["content"] for row in unchanged)
        assert all(read_json(root / "agents" / f"agent_{agent:04d}" / "private_memory.json")["revision"] == 3
                   for agent in range(1, 5))

    for key in ("actual_model_requests", "participant_model_requests", "router_model_requests",
                "private_memory_update_requests", "completed_periods"):
        assert manifests["persistent"][key] == manifests["fresh"][key]
    # The scripted policy is deliberately independent of notes: this checks the
    # architecture, not whether private memory causes human-like behavior.
    economic_results = {
        name: [{key: value for key, value in row.items() if key != "session_id"}
               for row in rows(tmp_path / name / "period_records.jsonl")]
        for name in ("persistent", "fresh")
    }
    assert economic_results["persistent"] == economic_results["fresh"]
    history_views = {}
    for condition, calls in observed.items():
        history_views[condition] = sorted([
            (call["turn"][0], call["turn"][1]["current_period"], call["turn"][1]["task"],
             call["update"], json.dumps(call["turn"][1]["history"], sort_keys=True),
             call["kwargs"].get("max_output_tokens"))
            for call in calls if call["kind"] == "participant"
        ])
    assert history_views["persistent"] == history_views["fresh"]


def test_fresh_report_distinguishes_saved_previous_state_from_empty_model_input(tmp_path):
    before = {"owner_agent_id": 1, "revision": 1,
              "content": {"note": "上一轮保存过的笔记", "evidence_periods": [1]}}
    after = {"owner_agent_id": 1, "revision": 2,
             "content": {"note": "只根据当前合法历史生成的新笔记", "evidence_periods": [1, 2]}}
    row = {"agent_id": 1, "period": 3, "request_index": 9,
           "memory_condition": "fresh", "memory_update_kind": "feedback",
           "previous_state": before, "new_state": after}
    (tmp_path / "private_memory_updates.jsonl").write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
    export_memory(tmp_path)
    report = (tmp_path / "memory_text/agent_0001.txt").read_text(encoding="utf-8")
    assert "实验条件：fresh" in report
    assert "上一版仍保存在日志和文件中供审阅；本次更新模型没有读取它" in report
    assert "本次更新实际输入：空笔记、空依据" in report
    assert "上一轮保存过的笔记" in report and "只根据当前合法历史生成的新笔记" in report


def test_fresh_repair_keeps_previous_note_private_and_uses_only_successful_update(tmp_path, monkeypatch):
    original = version.ScriptedLLMClient.call
    root = tmp_path / "fresh_repair"
    old_note = "OLD_PRIVATE_NOTE_AGENT_1_PERIOD_2_CANARY"
    rejected_draft = "REJECTED_MEMORY_DRAFT_CANARY"
    recovered_note = "RECOVERED_CURRENT_NOTE_AGENT_1_PERIOD_3"
    calls = []
    target_attempts = []

    async def spy(self, model=None, messages=None, **kwargs):
        supplied = list(messages or [])
        update = bool(supplied) and "<private_memory_update>" in str(supplied[-1].get("content", ""))
        turn = self._participant_turn(supplied) if self.kind == "participant" else None
        record = {"kind": self.kind, "update": update, "turn": copy.deepcopy(turn),
                  "messages": copy.deepcopy(supplied)}
        calls.append(record)
        response = await original(self, model=model, messages=messages, **kwargs)
        if self.kind == "participant" and update:
            agent, observation = turn
            reply = response.choices[0].message
            if agent == 1 and observation["current_period"] == 2:
                content = json.loads(reply.content)
                content["note"] = old_note
                reply.content = json.dumps(content)
            if agent == 1 and observation["current_period"] == 3 and observation["task"] != "closing":
                saved = read_json(root / "agents/agent_0001/private_memory.json")
                assert saved["revision"] == 1 and saved["content"]["note"] == old_note
                target_attempts.append(record)
                if len(target_attempts) == 1:
                    # Deliberately incomplete JSON, requiring a genuine repair call.
                    reply.content = '{"note":"' + rejected_draft
                else:
                    content = json.loads(reply.content)
                    content["note"] = recovered_note
                    reply.content = json.dumps(content)
        return response

    monkeypatch.setattr(version.ScriptedLLMClient, "call", spy)
    manifest = version.run_experiment(
        backend="scripted", memory_condition="fresh", participants=4, agent_concurrency=4,
        completed_period_limit=3, seed=2026, preference_seed=2026, request_limit=100, run_dir=root,
    )
    assert manifest["success"] is True and manifest["memory_experiment_complete"] is True
    assert manifest["private_memory_update_rejections"] == 1
    assert manifest["private_memory_update_requests"] == 13
    assert manifest["private_memory_update_count"] == 12
    assert len(target_attempts) == 2
    for attempt in target_attempts:
        assert _memory(attempt["messages"])["content"] == {"note": "", "evidence_periods": []}
        assert _memory(attempt["messages"])["evidence"] == []
        assert old_note not in json.dumps(attempt["messages"])
    assert rejected_draft not in json.dumps(target_attempts[0]["messages"])
    assert rejected_draft in json.dumps(target_attempts[1]["messages"])
    for call in calls:
        serialized = json.dumps(call["messages"])
        if call is not target_attempts[1]:
            assert rejected_draft not in serialized
        if call["kind"] == "participant" and call["update"]:
            assert old_note not in serialized
    target_decisions = [call for call in calls if call["kind"] == "participant" and not call["update"]
                        and call["turn"][0] == 1 and call["turn"][1]["current_period"] == 3]
    assert len(target_decisions) == 1
    assert _memory(target_decisions[0]["messages"])["content"]["note"] == recovered_note
    assert old_note not in json.dumps(target_decisions[0]["messages"])
    updates = rows(root / "private_memory_updates.jsonl")
    committed = [row for row in updates if row["agent_id"] == 1 and row["period"] == 3
                 and row["memory_update_kind"] == "feedback"]
    assert len(committed) == 1 and committed[0]["attempt"] == 2
    assert committed[0]["previous_state"]["content"]["note"] == old_note
    assert committed[0]["new_state"]["content"]["note"] == recovered_note
    assert committed[0]["new_state"]["revision"] == 2
    assert all(rejected_draft not in json.dumps(row["new_state"]) for row in updates)
    contexts = rows(root / "participant_contexts.jsonl")
    decision_context = next(row for row in contexts if row["agent_id"] == 1 and row["period"] == 3
                            and row["phase"] == "decision")
    assert decision_context["private_memory_snapshot"]["content"]["note"] == recovered_note
    assert any(row.get("action") == "submit_decision" and row.get("ok") is True
               and row.get("participant_request_index") == decision_context["request_index"]
               and row.get("private_memory_revision") == 2
               for row in rows(root / "agent_actions.jsonl"))
