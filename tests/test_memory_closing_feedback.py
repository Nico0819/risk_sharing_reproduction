"""Closing feedback uses real PersonAgent identity and only settled facts."""

from __future__ import annotations

import copy
import json

import pytest

from custom.envs.risk_sharing_env import RiskSharingRouterAdapter
from experiments.fixed_preferences_memory_v1 import run_smoke as version
from test_fixed_preferences_integration import isolated_runtime, read_json, rows  # noqa: F401
from test_fixed_preferences_memory_integration import install_spy, is_memory_update, memory_payload


@pytest.mark.parametrize("treatment,periods", [
    ("baseline", 1), ("25V", 15), ("25F", 15), ("25V", 30), ("25F", 30),
])
def test_closing_uses_real_identity_without_next_period_disclosures(tmp_path, monkeypatch, caplog, treatment, periods):
    spy = install_spy(monkeypatch, inject_canary=False)
    original = RiskSharingRouterAdapter.ask
    closing_identities = []

    async def inspect_real_context(self, ctx, instruction, **kwargs):
        if not self._closing_feedback:
            return await original(self, ctx, instruction, **kwargs)
        # These are delivered by PersonAgent.ask_env through its ServiceProxy;
        # the test never constructs a context or invokes a fake framework.
        assert ctx["id"] == ctx["agent_id"] == ctx["person_id"]
        assert instruction == "<observe>" and kwargs["readonly"] is True
        closing_identities.append(ctx["id"])
        env = self._environment
        before = copy.deepcopy((env.current_period, env.stage, env.history, env._decision_receipts))
        with pytest.raises(PermissionError, match="Closing feedback"):
            await original(
                self, ctx, "decision_type: transfer\ndecision_id: injected\namount: 1",
                readonly=False,
            )
        response = await original(self, ctx, instruction, **kwargs)
        feedback = response[0]["get_observation"]
        assert set(feedback) == {
            "agent_id", "partner_id", "current_period", "last_settled_period",
            "task", "stage", "history",
        }
        assert feedback["agent_id"] == ctx["id"]
        assert feedback["current_period"] == feedback["last_settled_period"] == periods
        assert feedback["task"] == feedback["stage"] == "closing"
        assert len(feedback["history"]) == periods * 2
        assert all({row["agent_id"], row["partner_id"]} == {
            feedback["agent_id"], feedback["partner_id"],
        } for row in feedback["history"])
        if periods < 16:
            assert "insurance" not in json.dumps(feedback).lower()
        assert copy.deepcopy((env.current_period, env.stage, env.history, env._decision_receipts)) == before
        return response

    monkeypatch.setattr(RiskSharingRouterAdapter, "ask", inspect_real_context)
    root = tmp_path / f"{treatment}-{periods}"
    extra = {"forced_assignment_probabilities": (0.5, 0.25, 0.25)} if treatment == "25F" else {}
    manifest = version.run_experiment(
        backend="scripted", treatment=treatment, completed_period_limit=periods,
        participants=4, agent_concurrency=4, request_limit=1000, run_dir=root, **extra,
    )
    assert manifest["success"] is True and manifest["closing_memory_complete"] is True
    assert "invalid ReAct decision" not in caplog.text
    assert sorted(closing_identities) == [1, 2, 3, 4]
    updates = rows(root / "private_memory_updates.jsonl")
    closing = [row for row in updates if row["memory_update_kind"] == "closing"]
    assert len(closing) == 4
    assert all(row["period"] == periods and row["task"] == "closing" for row in closing)
    closing_requests = {row["request_index"] for row in closing}
    actions = rows(root / "agent_actions.jsonl")
    assert not any(row.get("participant_request_index") in closing_requests for row in actions)
    for call in spy["calls"]:
        if call["kind"] == "participant" and call["turn"][1]["task"] == "closing":
            assert call["update"] and not call["tool_calls"]
    assert len(rows(root / "period_records.jsonl")) == periods * 4
    for path in root.glob("agents/*/private_memory.json"):
        state = read_json(path)
        assert state["last_reviewed_task"] == "closing"
        assert state["last_reviewed_period"] == periods
        assert periods in state["content"]["evidence_periods"]


def test_natural_end_includes_last_settlement_in_each_private_note(tmp_path):
    root = tmp_path / "full"
    manifest = version.run_experiment(
        backend="scripted", full_session=True, participants=4,
        agent_concurrency=4, request_limit=1200, run_dir=root,
    )
    assert manifest["success"] is True and manifest["stop_reason"] == "natural_termination"
    assert manifest["closing_memory_complete"] is True
    last = manifest["completed_periods"][-1]
    assert last in {48, 49, 50}
    closing = [row for row in rows(root / "private_memory_updates.jsonl") if row["task"] == "closing"]
    assert len(closing) == 4
    for update in closing:
        assert update["period"] == last
        assert last in update["new_state"]["content"]["evidence_periods"]
    assert not any(
        row["phase"] == "decision" and row["task"] == "closing"
        for row in rows(root / "participant_contexts.jsonl")
    )


def test_empty_notes_can_remain_unchanged_without_forced_predictions(tmp_path, monkeypatch):
    original = version.ScriptedLLMClient.call

    async def keep_note(self, model=None, messages=None, **kwargs):
        response = await original(self, model=model, messages=messages, **kwargs)
        if self.kind == "participant" and is_memory_update(messages):
            response.choices[0].message.content = json.dumps(memory_payload(messages)["content"])
        return response

    monkeypatch.setattr(version.ScriptedLLMClient, "call", keep_note)
    root = tmp_path / "unchanged"
    manifest = version.run_experiment(
        backend="scripted", completed_period_limit=3, request_limit=100, run_dir=root,
    )
    assert manifest["success"] is True and manifest["closing_memory_complete"] is True
    updates = rows(root / "private_memory_updates.jsonl")
    assert len(updates) == 12
    for update in updates:
        assert update["new_state"]["content"] == {"note": "", "evidence_periods": []}
    for path in root.glob("agents/*/private_memory.json"):
        assert read_json(path)["revision"] == 3


def test_closing_failure_is_reported_without_rewriting_completed_economics(tmp_path, monkeypatch):
    original = version.ScriptedLLMClient.call

    async def bad_closing(self, model=None, messages=None, **kwargs):
        response = await original(self, model=model, messages=messages, **kwargs)
        if self.kind == "participant":
            _, observation = self._participant_turn(messages)
            if observation["task"] == "closing":
                assert is_memory_update(messages)
                response.choices[0].message.content = "An incomplete closing note"
        return response

    monkeypatch.setattr(version.ScriptedLLMClient, "call", bad_closing)
    root = tmp_path / "closing-failure"
    manifest = version.run_experiment(
        backend="scripted", completed_period_limit=1, request_limit=100, run_dir=root,
    )
    assert manifest["success"] is True and manifest["completed_periods"] == [1]
    assert manifest["closing_memory_complete"] is False
    assert len(rows(root / "period_records.jsonl")) == 4
    actions = [row for row in rows(root / "agent_actions.jsonl") if row["action"] == "submit_decision" and row["ok"]]
    assert len(actions) == 2 and all(row["environment_receipt"]["period"] == 1 for row in actions)
    assert rows(root / "private_memory_updates.jsonl") == []
    assert all(read_json(path)["revision"] == 0 for path in root.glob("agents/*/private_memory.json"))
