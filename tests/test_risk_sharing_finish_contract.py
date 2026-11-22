"""Project finish descriptions preserve the real PersonAgent tool contract."""

from __future__ import annotations

import copy
import json
import socket
from importlib import import_module

import pytest

from agentsociety2.agent import PersonAgent
from agentsociety2.agent.person_tools import finish_ask_tool_schema, finish_step_tool_schema
from custom.agents.risk_sharing_participant import RiskSharingParticipant
from test_fixed_preferences_integration import isolated_runtime, rows  # noqa: F401


def without_descriptions(value):
    if isinstance(value, dict):
        return {key: without_descriptions(item) for key, item in value.items() if key != "description"}
    if isinstance(value, list):
        return [without_descriptions(item) for item in value]
    return value


def test_readonly_finish_keeps_the_framework_answer_contract():
    agent = RiskSharingParticipant()
    assert agent._react_tool_overrides(readonly=True)["finish"] == finish_ask_tool_schema()


@pytest.mark.parametrize("module_name,private_note", [
    ("experiments.baseline.run_smoke", False),
    ("experiments.fixed_preferences_v1.run_smoke", False),
    ("experiments.fixed_preferences_memory_v1.run_smoke", True),
])
def test_real_framework_advertises_compatible_finish_and_keeps_waiting_calls(
    tmp_path, monkeypatch, caplog, module_name, private_note,
):
    monkeypatch.setenv("PYTHON_DOTENV_DISABLED", "1")
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")

    def no_network(*args, **kwargs):
        pytest.fail("Finish contract tests must not contact a model provider")

    monkeypatch.setattr(socket.socket, "connect", no_network)
    module = import_module(module_name)
    original_call = module.ScriptedLLMClient.call
    # Registry discovery may reload custom modules on the first baseline run;
    # instrument the actual framework parser shared by every participant class.
    original_parse = PersonAgent._parse_react_turn
    advertised, waiting_calls, accepted_finishes = [], [], []
    framework_schema = finish_step_tool_schema()
    untouched_framework_schema = copy.deepcopy(framework_schema)

    async def inspect_provider(self, model=None, messages=None, **kwargs):
        if self.kind == "participant" and kwargs.get("tools"):
            agent, observation = self._participant_turn(messages)
            schema = next(tool for tool in kwargs["tools"] if tool["function"]["name"] == "finish")
            advertised.append(copy.deepcopy(schema))
            description = json.dumps(schema)
            assert "Memory generation happens" not in description
            assert "no separate memory pass" not in description
            assert "consolidation" not in description
            assert ("separate memory-update" in description) is private_note
            assert "waiting or finished" in description
            assert without_descriptions(schema) == without_descriptions(framework_schema)
            if observation["task"] == "waiting":
                waiting_calls.append(agent)
        return await original_call(self, model=model, messages=messages, **kwargs)

    def inspect_framework_parse(self, response, *, readonly=False):
        turn = original_parse(self, response, readonly=readonly)
        for decision in turn.decisions:
            if decision.action == "finish":
                assert not turn.error
                assert decision.args["memories"]
                accepted_finishes.append({
                    "agent": self.id,
                    "task": self._last_participant_observation["task"],
                    "receipt": getattr(self, "_accepted_receipt", None),
                    "memories": copy.deepcopy(decision.args["memories"]),
                })
        return turn

    monkeypatch.setattr(module.ScriptedLLMClient, "call", inspect_provider)
    monkeypatch.setattr(PersonAgent, "_parse_react_turn", inspect_framework_parse)
    root = tmp_path / module_name.split(".")[-2]
    manifest = module.run_experiment(
        backend="scripted", env_path=tmp_path / "unused.env", participants=4,
        agent_concurrency=4, completed_period_limit=1, request_limit=40, run_dir=root,
    )
    assert manifest["success"] is True
    if private_note:
        assert manifest["closing_memory_complete"] is True
    assert len(advertised) == 4
    assert len(waiting_calls) == 2 and len(set(waiting_calls)) == 2
    assert framework_schema == untouched_framework_schema == finish_step_tool_schema()
    assert "invalid ReAct decision" not in caplog.text

    normal_finishes = [entry for entry in accepted_finishes if entry["task"] != "closing"]
    assert len(normal_finishes) == 4
    assert {entry["agent"] for entry in normal_finishes if entry["task"] == "waiting"} == set(waiting_calls)
    local_receipts = [entry for entry in normal_finishes if entry["receipt"] is not None]
    assert len(local_receipts) == 2
    assert all(entry["memories"][0]["source"] == "environment_receipt" for entry in local_receipts)
    assert manifest["participant_model_requests"] == (8 if private_note else 4)
    assert manifest["router_model_requests"] == 2
    actions = [row for row in rows(root / "agent_actions.jsonl") if row.get("action") == "submit_decision"]
    assert len(actions) == 2 and all(row["ok"] for row in actions)
    assert all(row["submitted_amount"] == row["participant_model_output"]["amount"] == row["environment_receipt"]["amount"] for row in actions)
