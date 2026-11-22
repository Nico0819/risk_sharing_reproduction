"""Provider-response trace auditing, without contacting any model provider.

The integration fixtures replace only the provider boundary. PersonAgent,
workspace restoration, trusted routing and environment settlement remain real.
Synthetic reasoning is always labelled scripted, never a real provider trace.
"""

from __future__ import annotations

import asyncio
import copy
import json
import socket
from collections import Counter
from importlib import import_module
from types import SimpleNamespace

import pytest

from custom.participant_trace import ParticipantTraceRecorder, summarize_reasoning_traces


RUNNERS = (
    "experiments.baseline.run_smoke",
    "experiments.fixed_preferences_v1.run_smoke",
    "experiments.fixed_preferences_memory_v1.run_smoke",
)


@pytest.fixture(autouse=True)
def isolated_runtime(monkeypatch):
    import custom.config

    # No test reads project credentials, including scripted config loading.
    monkeypatch.setattr(custom.config, "_parse_dotenv", lambda path: {})
    monkeypatch.setenv("PYTHON_DOTENV_DISABLED", "1")
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    for prefix in ("AGENTSOCIETY_LLM", "AGENTSOCIETY_CODER_LLM"):
        for suffix in ("API_KEY", "API_BASE", "MODEL", "THINKING", "EXTRA_BODY", "REASONING_EFFORT"):
            monkeypatch.delenv(f"{prefix}_{suffix}", raising=False)
    for name in ("WORKSPACE_PATH", "LLM_API_KEY", "LLM_BASE_URL", "LLM_MODEL"):
        monkeypatch.delenv(name, raising=False)

    def unexpected_network(*args, **kwargs):
        pytest.fail("Reasoning audit regressions must not contact a provider")

    monkeypatch.setattr(socket.socket, "connect", unexpected_network)


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def fake_response(*, fields=None, finish_reason="stop"):
    message = SimpleNamespace(content="Explicit final response", tool_calls=[], **(fields or {}))
    return SimpleNamespace(
        id="fictional-response-id", model="fictional-provider-model",
        choices=[SimpleNamespace(message=message, finish_reason=finish_reason)],
        usage=SimpleNamespace(
            prompt_tokens=31, completion_tokens=47, total_tokens=78,
            completion_tokens_details=SimpleNamespace(reasoning_tokens=29),
            prompt_tokens_details=SimpleNamespace(cached_tokens=11),
        ),
    )


def record(recorder, *, response=None, error_type=None, **overrides):
    arguments = {
        "response": response, "error_type": error_type,
        "agent_id": 2, "pair_id": 1, "phase": "decision", "request_index": 7,
        "model": "requested-model", "duration_seconds": 0.125,
        "context": {
            "period": 3, "task": "transfer", "decision_id": "bound-decision-token",
            "private_memory_snapshot": {"revision": 4},
            "private_memory_sha256": "memory-fingerprint",
            "preferences_sha256": "preference-fingerprint",
        },
        "request_kwargs": {"max_output_tokens": 4096},
    }
    return recorder.record(**{**arguments, **overrides})


@pytest.mark.parametrize(
    "fields,status,readable",
    [
        ({}, "missing", False),
        ({"reasoning_content": None}, "missing", False),
        ({"reasoning_content": ""}, "empty", False),
        ({"reasoning_content": "  \n "}, "empty", False),
        ({"reasoning_details": []}, "empty", False),
        ({"reasoning_content": "A returned reasoning text."}, "available", True),
        ({"reasoning_details": [{"type": "reasoning.text", "text": "A provider explanation."}]}, "available", True),
        ({"reasoning_details": [{"type": "encrypted", "data": "opaque-fixture"}]}, "available", False),
    ],
)
def test_missing_empty_and_opaque_fields_are_not_invented_or_called_complete(fields, status, readable):
    recorder = ParticipantTraceRecorder(backend="llm", sanitize=lambda value: value)

    row = record(recorder, response=fake_response(fields=fields))

    assert row["reasoning_status"] == status
    assert row["readable_reasoning_present"] is readable
    assert row["provider_trace_complete"] is (readable and status == "available")
    if status == "missing":
        assert "reasoning_content" not in row
        assert row["reasoning_fields_returned"] == []
    assert row["response_source"] == "provider"  # A local fixture exercising classification only.
    assert row["capture_boundary"] == "framework_client_call_result"
    assert row["provider_retry_attempts_captured"] is False
    assert row["input_context_request_index"] == row["request_index"] == 7
    assert (row["agent_id"], row["pair_id"], row["period"], row["decision_id"]) == (
        2, 1, 3, "bound-decision-token",
    )
    assert row["usage"] == {
        "prompt_tokens": 31, "completion_tokens": 47, "total_tokens": 78,
        "completion_tokens_details": {"reasoning_tokens": 29},
        "prompt_tokens_details": {"cached_tokens": 11},
    }


@pytest.mark.parametrize(
    "finish_reason,truncated,complete",
    [("stop", False, True), ("tool_calls", False, True), ("length", True, False),
     ("max_output_tokens", True, False), ("content_filter", False, False), (None, False, None)],
)
def test_finish_reason_marks_partial_and_unknown_responses(finish_reason, truncated, complete):
    recorder = ParticipantTraceRecorder(backend="llm", sanitize=lambda value: value)
    row = record(recorder, response=fake_response(
        fields={"reasoning_content": "Returned partial or complete text."}, finish_reason=finish_reason,
    ))

    assert row["finish_reason"] == finish_reason
    assert row["response_truncated"] is truncated
    assert row["response_complete"] is complete
    assert row["provider_trace_complete"] is (complete is True)
    summary = summarize_reasoning_traces([recorder], backend="llm")
    assert summary["capture_boundary"] == "framework_client_call_result"
    assert summary["provider_retry_attempts_captured"] is False
    assert summary["participant_reasoning_truncated_responses"] == int(truncated)
    assert summary["participant_reasoning_capture_complete"] is (complete is True)


@pytest.mark.parametrize("error_type", ["TimeoutError", "CancelledError"])
def test_failed_or_cancelled_requests_are_recorded_without_fabricating_a_response(error_type):
    recorder = ParticipantTraceRecorder(backend="llm", sanitize=lambda value: value)
    row = record(recorder, error_type=error_type)

    assert row["reasoning_status"] == "request_failed"
    assert row["error_type"] == error_type
    assert row["ok"] is False
    assert row["response_complete"] is False
    assert row["provider_trace_complete"] is False
    assert row["reasoning_fields_returned"] == []
    assert row["usage"] == {}


def test_trace_whitelist_and_credential_redaction_preserve_long_text_and_raw_response(tmp_path):
    from experiments.fixed_preferences_memory_v1.run_smoke import RunLogger, _safe_log_value

    path = tmp_path / "participant_reasoning.jsonl"
    recorder = ParticipantTraceRecorder(
        RunLogger(path, announce=False), backend="llm", sanitize=_safe_log_value,
    )
    long_text = "BEGIN_TRACE " + "A complete retained reasoning sentence. " * 400 + " END_TRACE"
    response = fake_response(fields={"reasoning_content": long_text})
    response.headers = {"Authorization": "Bearer never-record-header-secret"}
    response.provider_extra = {"api_key": "never-record-provider-extra"}
    response.usage.unknown_provider_metadata = "never-record-usage-extra"
    response.choices[0].message.provider_specific_fields = {"api_key": "never-record-message-extra"}

    row = record(recorder, response=response, request_kwargs={
        "max_output_tokens": 32768, "extra_headers": {"Authorization": "Bearer never-record-request-secret"},
    })

    assert len(long_text) > 10000
    assert row["reasoning_content"] == long_text == rows(path)[0]["reasoning_content"]
    assert row["redaction_applied"] is False
    assert row["provider_trace_complete"] is True
    assert "never-record" not in path.read_text()
    assert "headers" not in row and "provider_extra" not in row

    secret_text = 'LLM_API_KEY="fake-body-secret"; Authorization: Bearer fake-bearer-secret; {"api_key":"fake-json-secret"}'
    response = fake_response(fields={
        "reasoning_content": secret_text,
        "reasoning_details": [{"text": "A visible explanation", "password": "fake-detail-secret"}],
    })
    response.choices[0].message.content = "password=fake-final-secret"
    response.choices[0].message.tool_calls = [SimpleNamespace(
        id="fixture-call", type="function",
        function=SimpleNamespace(name="ask_env", arguments='{"instruction":"api_key=fake-tool-secret"}'),
    )]
    before = copy.deepcopy(vars(response.choices[0].message))
    row = record(recorder, response=response, request_index=8)

    assert row["redaction_applied"] is True
    assert row["provider_trace_complete"] is False
    assert vars(response.choices[0].message) == before, "Auditing must not mutate the model's replay response"
    written = path.read_text()
    for secret in ("fake-body-secret", "fake-bearer-secret", "fake-json-secret", "fake-detail-secret", "fake-final-secret", "fake-tool-secret"):
        assert secret not in written
    assert "<redacted>" in written


@pytest.mark.parametrize("module_name", RUNNERS)
def test_concurrent_framework_agents_log_separate_reasoning_without_extra_requests(tmp_path, monkeypatch, module_name):
    module = import_module(module_name)
    with_memory = module_name == RUNNERS[2]
    original = module.ScriptedLLMClient.call
    observed = {"calls": [], "active": 0, "peak": 0}

    async def with_reasoning(self, model=None, messages=None, **kwargs):
        messages = list(messages or [])
        turn = self._participant_turn(messages) if self.kind == "participant" else None
        phase = "decision"
        last = str(messages[-1].get("content", "")) if messages else ""
        if "<private_memory_update>" in last:
            phase = "memory_update"
        elif "<participant_deliberation>" in last:
            phase = "deliberation"
        call = {"kind": self.kind, "turn": copy.deepcopy(turn), "phase": phase,
                "messages": copy.deepcopy(messages), "kwargs": copy.deepcopy(kwargs)}
        observed["calls"].append(call)
        if turn:
            observed["active"] += 1
            observed["peak"] = max(observed["peak"], observed["active"])
        try:
            await asyncio.sleep(0.003)
            response = await original(self, model=model, messages=messages, **kwargs)
        finally:
            if turn:
                observed["active"] -= 1
        reply = response.choices[0].message
        if turn:
            agent_id, observation = turn
            marker = f"REASONING_AGENT_{agent_id}_PERIOD_{observation['current_period']}_{observation['task']}_{phase}"
            reply.reasoning_content = marker + "\n" + "Complete scripted trace payload. " * 400 + "\nEND_TRACE"
            call["reasoning"] = reply.reasoning_content
        else:
            reply.reasoning_content = "ROUTER_ROUTING_ONLY"
        response.choices[0].finish_reason = "tool_calls" if reply.tool_calls else "stop"
        response.usage = SimpleNamespace(prompt_tokens=11, completion_tokens=17, total_tokens=28,
                                        completion_tokens_details=SimpleNamespace(reasoning_tokens=9))
        return response

    monkeypatch.setattr(module.ScriptedLLMClient, "call", with_reasoning)
    root = tmp_path / module_name.split(".")[-2]
    expected_participant_calls = 20 if with_memory else 12
    expected_total_calls = expected_participant_calls + 4
    manifest = module.run_experiment(
        backend="scripted", participants=4, agent_concurrency=4, completed_period_limit=2,
        participant_deliberation=True, request_limit=expected_total_calls + 2, run_dir=root,
    )

    assert manifest["success"] is True
    assert manifest["capture_boundary"] == "framework_client_call_result"
    assert manifest["provider_retry_attempts_captured"] is False
    assert manifest["router"] == "ReActRouter"
    assert observed["peak"] == 4
    assert manifest["participant_model_requests"] == expected_participant_calls
    assert manifest["router_model_requests"] == 4
    assert manifest["actual_model_requests"] == expected_total_calls == len(observed["calls"])
    traces = rows(root / "participant_reasoning.jsonl")
    # Finalization exports every saved response, including the closing memory
    # call, without adding requests or mixing one participant's text with another.
    from experiments.export_reasoning import render_text

    source_name = f"{root.name}/participant_reasoning.jsonl"
    assert (root / "reasoning_text/all_agents.txt").read_text() == render_text(traces, source_name)
    for agent_id in range(1, 5):
        selected = [row for row in traces if row["agent_id"] == agent_id]
        assert (root / f"reasoning_text/agent_{agent_id:04d}.txt").read_text() == render_text(selected, source_name)
    assert "[全部 Agent](reasoning_text/all_agents.txt)" in (root / "summary.md").read_text()
    assert rows(root / "run.log")[-1]["event"] == "reasoning_export_finished"
    contexts = {row["request_index"]: row for row in rows(root / "participant_contexts.jsonl")}
    assert len(traces) == expected_participant_calls == len(contexts)
    assert len({row["request_index"] for row in traces}) == len(traces)
    assert Counter(row["phase"] for row in traces) == {
        "decision": 8, "deliberation": 4, **({"memory_update": 8} if with_memory else {}),
    }
    for trace in traces:
        context = contexts[trace["input_context_request_index"]]
        assert trace["role"] == "participant"
        assert trace["capture_boundary"] == "framework_client_call_result"
        assert trace["provider_retry_attempts_captured"] is False
        assert trace["response_source"] == "scripted"
        assert trace["provider_trace_complete"] is False
        assert trace["response_complete"] is True and trace["response_truncated"] is False
        assert trace["reasoning_status"] == "available"
        assert len(trace["reasoning_content"]) > 10000
        assert trace["usage"]["completion_tokens_details"]["reasoning_tokens"] == 9
        assert trace["usage"]["prompt_tokens"] == 11
        for key in ("agent_id", "pair_id", "period", "task", "decision_id", "phase"):
            assert trace[key] == context[key]
        matches = [call for call in observed["calls"] if call["kind"] == "participant"
                   and call["reasoning"] == trace["reasoning_content"]]
        assert len(matches) == 1
        if with_memory:
            assert trace["private_memory_sha256"] == context["private_memory_sha256"]
            assert trace["private_memory_revision"] == context["private_memory_snapshot"]["revision"]
        if trace["phase"] != "decision":
            assert trace["request_tools"] == []
    for call in observed["calls"]:
        text = json.dumps(call["messages"])
        if call["kind"] == "router":
            assert "REASONING_AGENT_" not in text
        else:
            owner = call["turn"][0]
            for other in range(1, 5):
                if other != owner:
                    assert f"REASONING_AGENT_{other}_" not in text
    assert "ROUTER_ROUTING_ONLY" not in (root / "participant_reasoning.jsonl").read_text()
    assert manifest["participant_reasoning_capture_complete"] is None
    assert manifest["participant_reasoning_responses"] == expected_participant_calls

    actions = [row for row in rows(root / "agent_actions.jsonl") if row["action"] == "submit_decision"]
    assert len(actions) == 4
    first_period = [row for row in actions if row["environment_receipt"]["period"] == 1]
    assert len({row["submitted_amount"] for row in first_period}) == 2
    for action in actions:
        assert action["ok"] and action["decision_source"] == "participant_model"
        assert action["submitted_amount"] == action["participant_model_output"]["amount"]
        assert action["submitted_amount"] == action["environment_receipt"]["amount"]
        matched = [row for row in traces if row["decision_id"] == action["decision_id"] and row["phase"] == "decision"]
        assert len(matched) == 1 and matched[0]["agent_id"] == action["agent_id"]
        assert action["participant_raw_request"] in json.loads(matched[0]["tool_calls"][0]["function"]["arguments"])["instruction"]
