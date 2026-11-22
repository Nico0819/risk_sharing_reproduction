"""Inspect real SDK serialization using fake credentials and blocked sockets."""

from __future__ import annotations

import asyncio
import json
import socket
import sys

import httpx
import pytest


@pytest.mark.parametrize("role", ["participant", "router"])
def test_reasoning_and_output_budget_reach_chat_completion_transport(tmp_path, monkeypatch, role):
    monkeypatch.setenv("PYTHON_DOTENV_DISABLED", "1")
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    for name, value in {
        "LLM_API_KEY": "offline-test-credential",
        "LLM_BASE_URL": "https://offline-test.invalid/v1",
        "LLM_MODEL": "deepseek-flash",
    }.items():
        monkeypatch.setenv(name, value)
    for prefix in ("AGENTSOCIETY_LLM", "AGENTSOCIETY_CODER_LLM"):
        for suffix in ("API_KEY", "API_BASE", "MODEL", "THINKING", "EXTRA_BODY", "REASONING_EFFORT"):
            alias = f"{prefix}_{suffix}"
            monkeypatch.delenv(alias, raising=False)
            cached = sys.modules.get("agentsociety2.config.config")
            if cached is not None:
                attr = alias.removeprefix("AGENTSOCIETY_")
                monkeypatch.setattr(cached.Config, attr, getattr(cached.Config, attr))

    def no_network(*args, **kwargs):
        raise AssertionError("The transport regression must never open a socket")

    monkeypatch.setattr(socket.socket, "connect", no_network)
    captured = []
    reasoning = "Provider fixture reasoning must survive the SDK and the next tool turn."
    action = "ask_env" if role == "participant" else "submit_decision"
    action_args = (
        {"instruction": "decision_type: transfer\ndecision_id: sdk-token\namount: 7", "readonly": False}
        if role == "participant" else
        {"agent_id": 1, "decision_type": "transfer", "decision_id": "sdk-token", "amount": 7}
    )

    def capture(request):
        assert request.url.host == "offline-test.invalid"
        body = json.loads(request.content)
        captured.append(body)
        message = {
            "role": "assistant", "content": "",
            "tool_calls": [{
                "id": f"sdk-call-{len(captured)}", "type": "function",
                "function": {"name": action, "arguments": json.dumps(action_args)},
            }],
        }
        if role == "participant":
            message["reasoning_content"] = reasoning
        return httpx.Response(200, request=request, json={
            "id": f"sdk-response-{len(captured)}", "object": "chat.completion",
            "created": 1, "model": "deepseek-flash",
            "choices": [{"index": 0, "message": message, "finish_reason": "tool_calls"}],
            "usage": {"prompt_tokens": 11, "completion_tokens": 17, "total_tokens": 28},
        })

    async def async_send(self, request, *args, **kwargs):
        return capture(request)

    def sync_send(self, request, *args, **kwargs):
        return capture(request)

    monkeypatch.setattr(httpx.AsyncClient, "send", async_send)
    monkeypatch.setattr(httpx.Client, "send", sync_send)

    from custom.config import load_runtime_config
    load_runtime_config("llm", env_path=tmp_path / "absent.env")
    from custom.agents.reasoning_trace import provider_reasoning_fields
    from experiments.baseline.run_smoke import BudgetedClient, RequestBudget, RunLogger
    from agentsociety2.config.llm_dispatcher import build_client_for_role

    inner = build_client_for_role("default" if role == "participant" else "coder")
    trace_path = tmp_path / "participant_reasoning.jsonl"
    client = BudgetedClient(
        inner, RequestBudget(3), 8192, backend="llm", agent_id=1, pair_id=1,
        enforce_participant_progression=role == "participant",
        enforce_router_progression=role == "router",
        reasoning_logger=RunLogger(trace_path, announce=False),
    )
    if role == "participant":
        observation = {"agent_id": 1, "current_period": 1, "task": "transfer", "decision_id": "sdk-token"}
        turn = {"agent_id": 1, "observations": [{"observation": observation}]}
        messages = [{"role": "user", "content": "<risk_sharing_turn>" + json.dumps(turn) + "</risk_sharing_turn>"}]
        properties = {"instruction": {"type": "string"}, "readonly": {"type": "boolean"}}
    else:
        rpc = {"tool": action, "arguments": action_args}
        messages = [{"role": "user", "content": "<risk_sharing_rpc>" + json.dumps(rpc) + "</risk_sharing_rpc>"}]
        properties = {
            name: {"type": "integer" if type(value) is int else "string"}
            for name, value in action_args.items()
        }
    tools = [{"type": "function", "function": {
        "name": action, "description": "Offline transport fixture",
        "parameters": {"type": "object", "properties": properties, "required": list(properties)},
    }}]

    async def exercise():
        first = await client.call(messages=messages, tools=tools)
        if role == "participant":
            reply = first.choices[0].message
            assert provider_reasoning_fields(reply)["reasoning_content"] == reasoning
            continued = [*messages, {
                "role": "assistant", "content": reply.content,
                "tool_calls": [call.model_dump() for call in reply.tool_calls],
                **provider_reasoning_fields(reply),
            }, {"role": "tool", "tool_call_id": "sdk-call-1", "content": "ERROR: correct the request"}]
            second = await client.call(messages=continued, tools=tools)
            assert provider_reasoning_fields(second.choices[0].message)["reasoning_content"] == reasoning

    asyncio.run(exercise())
    assert len(captured) == (2 if role == "participant" else 1)
    for body in captured:
        assert body["max_tokens"] == 8192
        assert "max_output_tokens" not in body
        assert body["thinking"] == {"type": "enabled" if role == "participant" else "disabled"}
        assert body["tool_choice"] == ("auto" if role == "participant" else "required")
    if role == "participant":
        prior = next(message for message in captured[1]["messages"] if message["role"] == "assistant")
        assert prior["reasoning_content"] == reasoning
        traces = [json.loads(line) for line in trace_path.read_text().splitlines()]
        assert [trace["reasoning_content"] for trace in traces] == [reasoning, reasoning]
        assert all(trace["max_output_tokens"] == 8192 for trace in traces)
    else:
        assert not trace_path.exists()
