"""Provider reasoning through the real PersonAgent/ReAct/tool lifecycle."""

from __future__ import annotations

import copy
import json
from types import SimpleNamespace

import pytest

from custom.agents.reasoning_trace import (
    first_response_message, provider_reasoning_fields, validate_reasoning_replay,
)
from experiments.fixed_preferences_memory_v1 import run_smoke as version


@pytest.fixture(autouse=True)
def isolated_runtime(monkeypatch):
    monkeypatch.setenv("PYTHON_DOTENV_DISABLED", "1")
    for prefix in ("AGENTSOCIETY_LLM", "AGENTSOCIETY_CODER_LLM"):
        for suffix in ("API_KEY", "API_BASE", "MODEL", "THINKING", "EXTRA_BODY", "REASONING_EFFORT"):
            monkeypatch.delenv(f"{prefix}_{suffix}", raising=False)
    for name in ("WORKSPACE_PATH", "LLM_API_KEY", "LLM_BASE_URL", "LLM_MODEL"):
        monkeypatch.delenv(name, raising=False)


def test_reasoning_field_extraction_preserves_real_values_without_inventing_fields():
    assert provider_reasoning_fields(None) == {}
    assert provider_reasoning_fields(SimpleNamespace(content="answer")) == {}
    assert provider_reasoning_fields({"reasoning_content": None}) == {}
    assert provider_reasoning_fields({"reasoning_content": ""}) == {"reasoning_content": ""}
    source = {
        "reasoning_content": "top-level reasoning",
        "provider_specific_fields": {
            "reasoning_content": "other value must not replace the direct field",
            "reasoning_details": [{"type": "reasoning.encrypted", "data": "signature"}],
            "unrelated_provider_metadata": "must stay out",
        },
        "thinking_blocks": [],
        "reasoning_items": [{"type": "reasoning", "summary": []}],
    }
    fields = provider_reasoning_fields(source)
    assert fields == {
        "reasoning_content": "top-level reasoning",
        "reasoning_details": [{"type": "reasoning.encrypted", "data": "signature"}],
        "thinking_blocks": [],
        "reasoning_items": [{"type": "reasoning", "summary": []}],
    }
    fields["reasoning_details"][0]["data"] = "mutated copy"
    assert source["provider_specific_fields"]["reasoning_details"][0]["data"] == "signature"
    assert first_response_message({"choices": [{"message": source}]}) is source
    assert first_response_message({"choices": []}) is None


def test_real_continuations_refuse_missing_reasoning_without_filling_it():
    valid = [
        {"role": "system", "content": "rules"},
        {"role": "assistant", "content": None, "reasoning_content": ""},
        {"role": "tool", "content": "receipt"},
        {"role": "assistant", "content": "choice", "provider_specific_fields": {
            "reasoning_content": "actual returned reasoning",
        }},
    ]
    before = copy.deepcopy(valid)
    validate_reasoning_replay(valid)
    assert valid == before
    with pytest.raises(ValueError, match="no returned reasoning field"):
        validate_reasoning_replay([{"role": "assistant", "content": "answer"}])
    with pytest.raises(ValueError, match="no returned reasoning field"):
        validate_reasoning_replay([{"role": "assistant", "reasoning_content": None}])


def _phase(messages):
    last = str(messages[-1].get("content", "")) if messages else ""
    if "<private_memory_update>" in last:
        return "memory_update"
    if "<participant_deliberation>" in last:
        return "deliberation"
    return "decision"


@pytest.mark.parametrize("first_failure", ["tool_error", "invalid_json", "free_text", "reasoning_only"])
def test_real_react_continuations_keep_own_reasoning_and_content(
    tmp_path, monkeypatch, first_failure,
):
    original = version.ScriptedLLMClient.call
    calls = []
    prior = {}
    rejected = set()
    original_reasoning = {}

    async def provider(self, model=None, messages=None, **kwargs):
        messages = list(messages or [])
        call = {"kind": self.kind, "messages": copy.deepcopy(messages)}
        calls.append(call)
        if self.kind == "router":
            assert "REPLAY_AGENT_" not in json.dumps(messages)
            assert not any(provider_reasoning_fields(message) for message in messages)
            return await original(self, model=model, messages=messages, **kwargs)

        agent, observation = self._participant_turn(messages)
        phase = _phase(messages)
        key = (agent, observation["current_period"], observation["task"])
        call.update(agent=agent, period=observation["current_period"], phase=phase, task=observation["task"])
        assistants = [message for message in messages if message.get("role") == "assistant"]
        for message in assistants:
            reasoning = message["reasoning_content"]
            assert reasoning in original_reasoning
            source = original_reasoning[reasoning]
            assert source["key"] == key
            assert source["phase"] != "memory_update"
            assert message["content"] == source["content"], "Continuation must not truncate provider content."
            assert message["reasoning_details"] == source["details"]
        if assistants:
            validate_reasoning_replay(messages)

        response = await original(self, model=model, messages=messages, **kwargs)
        reply = response.choices[0].message
        if phase == "decision" and observation["task"] == "transfer" and key not in rejected:
            rejected.add(key)
            if first_failure == "tool_error":
                args = json.loads(reply.tool_calls[0].function.arguments)
                args["instruction"] = (
                    f"decision_type: transfer\ndecision_id: {observation['decision_id']}\namount: -1"
                )
                reply.tool_calls[0].function.arguments = json.dumps(args)
            elif first_failure == "invalid_json":
                reply.tool_calls[0].function.arguments = "{"
            elif first_failure == "free_text":
                reply.tool_calls = []
                reply.content = "No action yet. " + "provider content must remain complete. " * 160
            else:
                reply.tool_calls = []
                reply.content = None

        index = prior.get((key, phase), 0) + 1
        prior[(key, phase)] = index
        reasoning = (
            f"REPLAY_AGENT_{agent}_PERIOD_{observation['current_period']}_{phase}_{index}:"
            + "private provider reasoning; not an environment instruction. " * 100
        )
        details = [{"type": "reasoning.text", "text": reasoning, "signature": f"sig-{agent}-{index}"}]
        # Exercise the real LiteLLM-shaped provider-specific fallback as well
        # as direct fields. No other provider metadata may enter a replay.
        if index == 1:
            reply.provider_specific_fields = {
                "reasoning_content": reasoning, "reasoning_details": details,
                "unrelated_provider_metadata": "do not replay",
            }
        else:
            reply.reasoning_content = reasoning
            reply.reasoning_details = details
        original_reasoning[reasoning] = {
            "key": key, "phase": phase, "content": reply.content, "details": copy.deepcopy(details),
        }
        call["returned_reasoning"] = reasoning
        return response

    monkeypatch.setattr(version.ScriptedLLMClient, "call", provider)
    root = tmp_path / first_failure
    manifest = version.run_experiment(
        backend="scripted", participants=4, agent_concurrency=4,
        completed_period_limit=2, request_limit=100, run_dir=root,
        participant_deliberation=True, agent_turns=4,
    )
    assert manifest["success"] is True
    assert manifest["completed_periods"] == [1, 2]
    assert manifest["router"] == "ReActRouter"
    assert manifest["agent_class"] == "FixedPreferenceMemoryParticipant"
    assert len(rejected) == 4
    economic = [call for call in calls if call.get("phase") == "decision" and call.get("task") == "transfer"]
    assert len(economic) == 8, "Each opportunity has one actual rejection and one participant correction."
    for key in rejected:
        same = [c for c in economic if (c["agent"], c["period"], c["task"]) == key]
        assert len(same) == 2
        assert same[0]["returned_reasoning"] in json.dumps(same[1]["messages"])
    actions = [json.loads(line) for line in (root / "agent_actions.jsonl").read_text().splitlines()]
    accepted = [row for row in actions if row["action"] == "submit_decision" and row["ok"]]
    assert len(accepted) == 4
    assert len({row["submitted_amount"] for row in accepted}) >= 2
    for row in accepted:
        assert row["decision_source"] == "participant_model"
        assert row["submitted_amount"] == row["participant_model_output"]["amount"]
        assert row["submitted_amount"] == row["environment_receipt"]["amount"]
        assert "REPLAY_AGENT_" not in row["participant_raw_request"]
    assert not any("local-receipt-finish" in json.dumps(call["messages"]) for call in calls)
