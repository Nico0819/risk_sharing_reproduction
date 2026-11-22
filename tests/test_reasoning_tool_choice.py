"""Check actual LLM request options without contacting a model provider."""

from __future__ import annotations

import asyncio
import copy
import json
import socket
from importlib import import_module
from types import SimpleNamespace

import pytest

from custom.agents.fixed_preferences_memory_v1.state import initial_state, memory_payload, write_state
from custom.agents.fixed_preferences_v1.profiles import profile_sha256


RUNNERS = (
    "experiments.baseline.run_smoke",
    "experiments.fixed_preferences_v1.run_smoke",
    "experiments.fixed_preferences_memory_v1.run_smoke",
)


class CapturingProvider:
    model_name = "offline-reasoning-fixture"

    def __init__(self):
        self.requests = []

    async def call(self, model=None, messages=None, **kwargs):
        self.requests.append(copy.deepcopy({"messages": messages, **kwargs}))
        message = SimpleNamespace(content="Offline fixture response", tool_calls=[],
                                  reasoning_content="Synthetic returned reasoning")
        return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason="stop")])


def tool(name, properties):
    return {"type": "function", "function": {
        "name": name, "parameters": {"type": "object", "properties": properties},
    }}


@pytest.mark.parametrize("module_name", RUNNERS)
def test_llm_participant_auto_preserves_tool_constraints_and_router_required(tmp_path, monkeypatch, module_name):
    monkeypatch.setenv("PYTHON_DOTENV_DISABLED", "1")
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")

    def no_network(*args, **kwargs):
        pytest.fail("Tool-choice compatibility tests must not contact a provider")

    monkeypatch.setattr(socket.socket, "connect", no_network)
    module = import_module(module_name)
    profile = {"risk_aversion": "medium", "advantageous_inequality_aversion": "low"}
    state = initial_state(1, 2, "offline-scope", profile_sha256(profile))
    memory_path = tmp_path / "private_memory.json"
    write_state(memory_path, state)
    participant_options = {"enforce_participant_progression": True, "agent_id": 1,
                           "pair_id": 1, "backend": "llm"}
    if module_name != RUNNERS[0]:
        participant_options["preference_profile"] = profile
    if module_name == RUNNERS[2]:
        participant_options.update(private_memory_path=memory_path, private_memory_scope="offline-scope")
    provider = CapturingProvider()
    budget = module.RequestBudget(max_calls=20)
    participant = module.BudgetedClient(provider, budget, 8192, **participant_options)
    schemas = [
        tool("ask_env", {"instruction": {"type": "string"}, "readonly": {"type": "boolean"}}),
        tool("finish", {"message": {"type": "string"}}),
        tool("unavailable_tool", {}),
    ]
    original_schemas = copy.deepcopy(schemas)
    cases = [
        ("initial_transfer", "transfer", None, None, {"ask_env", "finish"}),
        ("initial_wait", "wait", None, None, {"ask_env", "finish"}),
        ("observation_returned", "transfer", "ask_env", "observation", {"ask_env"}),
        ("rejected_decision", "transfer", "ask_env", "ERROR: invalid amount", {"ask_env"}),
        ("accepted_receipt", "transfer", "ask_env", "submitted", {"finish"}),
        ("rejected_tool", "transfer", "unavailable_tool", "ERROR: unavailable tool", {"ask_env", "finish"}),
    ]

    for case, task, previous_tool, result, expected_names in cases:
        observation = {"agent_id": 1, "partner_id": 2, "current_period": 1,
                       "task": task, "history": [], "decision_id": "offline-decision-token"}
        payload = {"agent_id": 1, "observations": [{"observation": observation}]}
        messages = [
            {"role": "system", "content": f"<fixed_preferences>{json.dumps(profile)}</fixed_preferences>"},
            {"role": "user", "content": (
                f"<risk_sharing_turn>{json.dumps(payload)}</risk_sharing_turn>\n"
                f"<private_memory>{json.dumps(memory_payload(state))}</private_memory>"
            )},
        ]
        if previous_tool:
            messages.extend([
                {"role": "assistant", "content": None,
                 "reasoning_content": "Synthetic preceding provider reasoning",
                 "tool_calls": [{"id": "previous-call", "type": "function", "function": {
                     "name": previous_tool,
                     "arguments": json.dumps({"instruction": "fixture", "readonly": result == "observation"}),
                 }}]},
                {"role": "tool", "tool_call_id": "previous-call", "content": result},
            ])
        asyncio.run(participant.call(
            messages=messages, tools=schemas,
            tool_choice={"type": "function", "function": {"name": "ask_env"}},
        ))
        request = provider.requests[-1]
        assert request["tool_choice"] == "auto", case
        assert request["max_tokens"] == 8192
        assert "max_output_tokens" not in request
        assert {entry["function"]["name"] for entry in request["tools"]} == expected_names, case
        if expected_names == {"ask_env"}:
            parameters = request["tools"][0]["function"]["parameters"]
            assert parameters["properties"]["readonly"] == {"type": "boolean", "enum": [False]}
            assert "readonly" in parameters["required"]
        assert request["messages"] == messages
    assert schemas == original_schemas, "Restricting one request must not mutate shared tool schemas"
    assert len(provider.requests) == budget.calls == len(cases)

    # Router thinking is disabled and may still force the explicitly supplied
    # RPC. It must retain the participant's exact economic arguments.
    arguments = {"decision_type": "transfer", "decision_id": "offline-decision-token", "amount": 73}
    rpc = {"tool": "submit_decision", "arguments": arguments}
    router_provider = CapturingProvider()
    router = module.BudgetedClient(router_provider, module.RequestBudget(max_calls=1), 8192,
                                   enforce_router_progression=True, backend="llm")
    asyncio.run(router.call(
        messages=[{"role": "user", "content": f"<risk_sharing_rpc>{json.dumps(rpc)}</risk_sharing_rpc>"}],
        tools=[tool("submit_decision", {name: {} for name in arguments}), tool("get_observation", {})],
        tool_choice="auto",
    ))
    request = router_provider.requests[0]
    assert request["tool_choice"] == "required"
    assert request["max_tokens"] == 8192
    assert "max_output_tokens" not in request
    assert [entry["function"]["name"] for entry in request["tools"]] == ["submit_decision"]
    assert request["tools"][0]["function"]["parameters"]["properties"] == {
        name: {"enum": [value]} for name, value in arguments.items()
    }
