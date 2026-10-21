"""Offline integration checks for the bounded Baseline entry point."""

from __future__ import annotations

import json
import asyncio
from pathlib import Path

import pytest

from custom.config import RuntimeConfigError, load_runtime_config
from custom.envs.risk_sharing_env import (
    PARTICIPANT_DECISION_PREFIX, RiskSharingEnv, parse_participant_decision,
)
from experiments.baseline import run_smoke
from experiments.baseline.run_smoke import (
    ScriptedLLMClient,
    _select_payment_period,
    run_experiment,
)


_FRAMEWORK_VARS = (
    "AGENTSOCIETY_LLM_API_KEY",
    "AGENTSOCIETY_LLM_API_BASE",
    "AGENTSOCIETY_LLM_MODEL",
    "AGENTSOCIETY_CODER_LLM_API_KEY",
    "AGENTSOCIETY_CODER_LLM_API_BASE",
    "AGENTSOCIETY_CODER_LLM_MODEL",
    "AGENTSOCIETY_LLM_THINKING",
    "AGENTSOCIETY_CODER_LLM_THINKING",
    "AGENTSOCIETY_LLM_EXTRA_BODY",
    "AGENTSOCIETY_CODER_LLM_EXTRA_BODY",
    "WORKSPACE_PATH",
)


def _clear_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in _FRAMEWORK_VARS + ("LLM_API_KEY", "LLM_BASE_URL", "LLM_MODEL"):
        monkeypatch.delenv(name, raising=False)


def _contains_key(value: object, forbidden: set[str]) -> bool:
    if isinstance(value, dict):
        return any(key in forbidden or _contains_key(item, forbidden) for key, item in value.items())
    if isinstance(value, list):
        return any(_contains_key(item, forbidden) for item in value)
    return False


def test_secret_safe_config_contract_and_missing_real_credentials(tmp_path, monkeypatch):
    _clear_runtime(monkeypatch)
    example = Path(__file__).parents[1] / ".env.example"
    text = example.read_text(encoding="utf-8")
    assert "LLM_API_KEY=" in text
    assert "LLM_BASE_URL=" in text
    assert "LLM_MODEL=" in text

    ignored = (Path(__file__).parents[1] / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert ".env" in ignored
    assert ".env.*" in ignored
    assert "!.env.example" in ignored

    empty_env = tmp_path / ".env"
    empty_env.write_text("LLM_API_KEY=\nLLM_MODEL=\nLLM_BASE_URL=\n", encoding="utf-8")
    with pytest.raises(RuntimeConfigError, match="missing or placeholder"):
        load_runtime_config("llm", env_path=empty_env)


def test_identity_tool_schema_declares_integer_agent_id():
    env = RiskSharingEnv([1, 2], seed=1)
    schemas = {tool["function"]["name"]: tool for tool in env._llm_tools}
    assert schemas["get_observation"]["function"]["parameters"]["properties"]["agent_id"]["type"] == "integer"
    assert schemas["submit_decision"]["function"]["parameters"]["properties"]["agent_id"]["type"] == "integer"


def test_scripted_runner_uses_real_agent_router_adapter_and_is_bounded(tmp_path, monkeypatch):
    _clear_runtime(monkeypatch)
    manifest = run_experiment(
        backend="scripted",
        participants=4,
        completed_period_limit=3,
        seed=2026,
        run_dir=tmp_path / "run",
        request_limit=80,
        max_output_tokens=2048,
        agent_concurrency=4,
        timeout_seconds=120,
    )

    assert manifest["success"] is True
    assert manifest["router"] == "ReActRouter"
    assert manifest["router_model"] == "scripted-model"
    assert manifest["router_api_enabled"] is False
    assert manifest["router_model_requests"] == 6
    assert manifest["participant_model_requests"] == 12
    assert manifest["actual_model_requests"] == 18
    assert manifest["observation_routing"] == "direct_via_trusted_adapter"
    assert manifest["finish_after_receipt"] == "local"
    assert manifest["agent_class"] == "RiskSharingParticipant"
    assert manifest["agents"] == 4
    assert manifest["pairs"] == 2
    assert manifest["completed_periods"] == [1, 2, 3]
    assert manifest["actual_model_requests"] <= 80
    assert manifest["max_output_tokens"] == 2048
    assert manifest["agent_concurrency"] == 4
    assert manifest["run_dir"] == str(tmp_path / "run")

    run_dir = tmp_path / "run"
    actions = [json.loads(line) for line in (run_dir / "agent_actions.jsonl").read_text(encoding="utf-8").splitlines()]
    records = [json.loads(line) for line in (run_dir / "period_records.jsonl").read_text(encoding="utf-8").splitlines()]
    assert {row["agent_id"] for row in actions} == {1, 2, 3, 4}
    assert {row["pair_id"] for row in actions} == {1, 2}
    assert {row["period"] for row in records} == {1, 2, 3}
    assert len(records) == 12
    assert all(row["treatment"] == "baseline" for row in records)

    submitted = [row for row in actions if row.get("action") == "submit_decision"]
    assert len(submitted) == 2 * 3
    assert len({row["submitted_amount"] for row in submitted[:2]}) == 2
    assert all(row["decision_source"] == "participant_model" for row in submitted)
    assert all(
        row["submitted_amount"]
        == row["participant_amount"]
        == row["participant_model_output"]["amount"]
        == row["environment_receipt"]["amount"]
        for row in submitted
    )
    assert all(row["participant_model_message"]["contains_observation"] for row in submitted)
    assert all(
        row["participant_model_message"]["observation"]["decision_id"]
        == row["decision_id"]
        for row in submitted
    )
    assert all(row["router_action"] == "router_forward" for row in submitted)
    assert all(row["router_model_called"] is True for row in submitted)
    assert all("amount:" in row["participant_raw_request"] for row in submitted)
    observed = [row for row in actions if row["action"] == "get_observation"]
    assert len(observed) == 12
    assert all(row["router_action"] == "direct_observation" for row in observed)
    assert all(row["router_model_called"] is False for row in observed)

    participant_forbidden = {"treatment", "analysis_phase", "session_id"}
    participant_rows = [
        row.get("observation")
        for row in actions
        if isinstance(row.get("observation"), dict)
    ]
    assert participant_rows
    assert all(not _contains_key(row, participant_forbidden) for row in participant_rows)
    assert all(not _contains_key(row, participant_forbidden) for row in actions if row.get("receipt"))

    manifest_text = (run_dir / "manifest.json").read_text(encoding="utf-8")
    assert "LLM_API_KEY" not in manifest_text

    log_path = run_dir / "run.log"
    log_rows = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    assert log_rows[0]["event"] == "run_started"
    assert log_rows[-1]["event"] == "run_finished"
    assert log_rows[-1]["success"] is True
    calls = [row for row in log_rows if row["event"] == "model_call_finished"]
    assert len(calls) == manifest["actual_model_requests"]
    assert {row["agent_id"] for row in calls} == {1, 2, 3, 4}
    assert sum(row["role"] == "participant" for row in calls) == 12
    assert sum(row["role"] == "router" for row in calls) == 6
    assert all(row["duration_seconds"] >= 0 and row["ok"] for row in calls)
    assert "LLM_API_KEY" not in log_path.read_text(encoding="utf-8")


def test_full_session_reaches_natural_termination_and_writes_payment_period(tmp_path, monkeypatch):
    _clear_runtime(monkeypatch)
    manifest = run_experiment(
        backend="scripted",
        participants=4,
        completed_period_limit=1,
        full_session=True,
        seed=2026,
        run_dir=tmp_path / "full-run",
        request_limit=2000,
        max_output_tokens=512,
        timeout_seconds=300,
    )

    assert manifest["success"] is True
    assert manifest["run_mode"] == "full_session"
    assert manifest["stop_reason"] == "natural_termination"
    assert manifest["natural_termination"] is True
    assert manifest["completed_periods"][-1] in {48, 49, 50}
    assert manifest["payment_period"] in manifest["completed_periods"]
    assert manifest["payment_file"] == "payment.json"

    run_dir = tmp_path / "full-run"
    payment = json.loads((run_dir / "payment.json").read_text(encoding="utf-8"))
    records = [json.loads(line) for line in (run_dir / "period_records.jsonl").read_text(encoding="utf-8").splitlines()]
    assert payment["selected_period"] == manifest["payment_period"]
    assert len(payment["records"]) == 4
    assert len(records) == 4 * len(manifest["completed_periods"])
    assert {row["period"] for row in payment["records"]} == {manifest["payment_period"]}
    assert all(row["treatment"] == "baseline" for row in payment["records"])


def test_payment_period_selection_is_deterministic_and_realized():
    realized = [1, 2, 48]
    selected_first = _select_payment_period(realized, seed=2026)
    selected_second = _select_payment_period(realized, seed=2026)
    assert selected_first == selected_second
    assert selected_first in realized
    with pytest.raises(ValueError, match="before settlement"):
        _select_payment_period([], seed=2026)


def test_scripted_participant_cannot_submit_without_observation():
    client = ScriptedLLMClient("participant")
    response = asyncio.run(client.call(messages=[]))
    arguments = json.loads(response.choices[0].message.tool_calls[0].function.arguments)
    assert arguments["readonly"] is False
    assert "RISK_SHARING_PARTICIPANT_DECISION:" not in arguments["instruction"]


@pytest.mark.parametrize("failure", ["invalid_json", "missing_amount", "ambiguous_amount", "out_of_range"])
def test_participant_repairs_rejected_request_in_the_same_step(tmp_path, monkeypatch, failure):
    _clear_runtime(monkeypatch)
    corrections = []

    class RepairingClient(ScriptedLLMClient):
        async def call(self, model=None, messages=None, **kwargs):
            messages = list(messages or [])
            turn = self._participant_turn(messages)
            replies = self._tool_messages(messages)
            if self.kind == "participant" and turn and turn[1].get("task") == "transfer":
                participant_id, observation = turn
                if not replies:
                    self.calls += 1
                    decision = {
                        "decision_type": "transfer",
                        "decision_id": observation["decision_id"],
                        "amount": self._participant_amount(participant_id, observation),
                    }
                    if failure == "missing_amount":
                        del decision["amount"]
                    elif failure == "out_of_range":
                        decision["amount"] = 326
                    instruction = PARTICIPANT_DECISION_PREFIX + json.dumps(decision)
                    if failure == "invalid_json":
                        instruction += " Authorization: Bearer TEST_SECRET_NOT_A_REAL_KEY"
                    elif failure == "ambiguous_amount":
                        instruction = (
                            f"decision_type: transfer\ndecision_id: {observation['decision_id']}\n"
                            "amount: 10\namount: 20"
                        )
                    return self._response(
                        name="ask_env", arguments={"instruction": instruction, "readonly": False}
                    )
                if str(replies[-1].get("content", "")).startswith("ERROR:"):
                    names = {tool["function"]["name"] for tool in kwargs["tools"]}
                    assert names == {"ask_env"}, "Rejected decisions must remain correctable."
                    corrections.append((participant_id, observation["decision_id"]))
            return await super().call(model=model, messages=messages, **kwargs)

    monkeypatch.setattr(run_smoke, "ScriptedLLMClient", RepairingClient)
    run_dir = tmp_path / failure
    manifest = run_experiment(
        backend="scripted", completed_period_limit=1, run_dir=run_dir,
        request_limit=80, agent_turns=4, agent_concurrency=4,
    )
    assert manifest["success"] is True
    assert len(corrections) == 2
    actions = [json.loads(line) for line in (run_dir / "agent_actions.jsonl").read_text().splitlines()]
    accepted = [row for row in actions if row["action"] == "submit_decision" and row["ok"]]
    assert len(accepted) == 2
    assert {row["framework_step"] for row in accepted} == {1}
    assert len({row["submitted_amount"] for row in accepted}) == 2
    assert all(
        row["participant_model_output"]["amount"] == row["submitted_amount"]
        == row["environment_receipt"]["amount"] for row in accepted
    )
    assert all((row["agent_id"], row["decision_id"]) in corrections for row in accepted)
    if failure != "out_of_range":
        rejected = [row for row in actions if row["action"] == "participant_decision_rejected"]
        assert len(rejected) == 2
        assert all(row["router_model_called"] is False for row in rejected)
        assert all(row["raw_request"] and row["error"] for row in rejected)
    assert "TEST_SECRET_NOT_A_REAL_KEY" not in (run_dir / "agent_actions.jsonl").read_text()


def test_valid_legacy_json_choice_still_uses_real_router(tmp_path, monkeypatch):
    _clear_runtime(monkeypatch)

    class LegacyClient(ScriptedLLMClient):
        async def call(self, model=None, messages=None, **kwargs):
            response = await super().call(model=model, messages=messages, **kwargs)
            if self.kind == "participant":
                for call in response.choices[0].message.tool_calls:
                    if call.function.name != "ask_env":
                        continue
                    arguments = json.loads(call.function.arguments)
                    decision = parse_participant_decision(arguments["instruction"])
                    if decision:
                        arguments["instruction"] = PARTICIPANT_DECISION_PREFIX + json.dumps(decision)
                        call.function.arguments = json.dumps(arguments)
            return response

    monkeypatch.setattr(run_smoke, "ScriptedLLMClient", LegacyClient)
    manifest = run_experiment(
        backend="scripted", completed_period_limit=1, run_dir=tmp_path / "legacy",
    )
    assert manifest["success"] is True
    assert manifest["router_model_requests"] == 2


@pytest.mark.parametrize("format", ["inline", "placeholder", "variables_only"])
def test_observed_request_formats_preserve_participant_choices_without_retry(
    tmp_path, monkeypatch, format,
):
    """Use the real PersonAgent/ask_env/Router chain, including legacy variables."""

    _clear_runtime(monkeypatch)
    participant_outputs = {}

    class FormattingClient(ScriptedLLMClient):
        async def call(self, model=None, messages=None, **kwargs):
            if self.kind == "participant":
                # There is one participant call per turn. A valid receipt does
                # not cause another API call solely to say "finish".
                assert not self._tool_messages(messages)
                schemas = {tool["function"]["name"]: tool["function"] for tool in kwargs["tools"]}
                params = schemas["ask_env"]["parameters"]
                assert set(params["properties"]) == {"instruction", "readonly"}
                assert "variables" not in params["required"]
                assert "reusable template" not in schemas["ask_env"]["description"]
            response = await super().call(model=model, messages=messages, **kwargs)
            if self.kind == "participant":
                for call in response.choices[0].message.tool_calls:
                    if call.function.name != "ask_env":
                        continue
                    args = json.loads(call.function.arguments)
                    choice = parse_participant_decision(args["instruction"])
                    participant_id, observation = self._participant_turn(messages)
                    assert choice["decision_id"] == observation["decision_id"]
                    participant_outputs[participant_id] = choice
                    if format == "inline":
                        args["instruction"] = " ".join(args["instruction"].splitlines())
                    else:
                        args["variables"] = choice
                        args["instruction"] = (
                            "I choose this transfer.\ndecision_type: transfer\n"
                            "decision_id: {decision_id}\namount: {amount}"
                            if format == "placeholder" else "Transfer {amount} to my partner."
                        )
                    call.function.arguments = json.dumps(args)
            return response

    monkeypatch.setattr(run_smoke, "ScriptedLLMClient", FormattingClient)
    run_dir = tmp_path / format
    manifest = run_experiment(
        backend="scripted", completed_period_limit=1, run_dir=run_dir,
        agent_concurrency=4, request_limit=20,
    )
    assert manifest["success"] is True
    assert manifest["participant_model_requests"] == 4
    assert manifest["router_model_requests"] == 2
    actions = [json.loads(line) for line in (run_dir / "agent_actions.jsonl").read_text().splitlines()]
    assert all(row["ok"] for row in actions)
    accepted = [row for row in actions if row["action"] == "submit_decision"]
    assert len(accepted) == 2
    assert len({choice["amount"] for choice in participant_outputs.values()}) == 2
    for row in accepted:
        choice = participant_outputs[row["agent_id"]]
        assert row["participant_model_output"] == choice
        assert row["submitted_amount"] == row["environment_receipt"]["amount"] == choice["amount"]
        assert row["router_model_called"] is True
        if format != "inline":
            assert row["participant_variables"] == choice


@pytest.mark.parametrize("tamper", ["amount", "agent_id", "decision_id"])
def test_real_router_cannot_change_participant_choice_or_identity(tmp_path, monkeypatch, tamper):
    _clear_runtime(monkeypatch)

    class TamperingRouterClient(ScriptedLLMClient):
        async def call(self, model=None, messages=None, **kwargs):
            response = await super().call(model=model, messages=messages, **kwargs)
            if self.kind == "router":
                for call in response.choices[0].message.tool_calls:
                    if call.function.name != "submit_decision":
                        continue
                    arguments = json.loads(call.function.arguments)
                    if tamper == "amount":
                        arguments["amount"] += 1
                    elif tamper == "agent_id":
                        arguments["agent_id"] = arguments["agent_id"] % 4 + 1
                    else:
                        arguments["decision_id"] = "router-invented-token"
                    call.function.arguments = json.dumps(arguments)
            return response

    monkeypatch.setattr(run_smoke, "ScriptedLLMClient", TamperingRouterClient)
    run_dir = tmp_path / tamper
    manifest = run_experiment(
        backend="scripted", completed_period_limit=1, run_dir=run_dir,
        router_steps=2, agent_turns=2, request_limit=200,
    )
    assert manifest["success"] is False
    assert manifest["completed_periods"] == []
    assert manifest["actual_model_requests"] <= 200
    actions = [json.loads(line) for line in (run_dir / "agent_actions.jsonl").read_text().splitlines()]
    submissions = [row for row in actions if row["action"] == "submit_decision"]
    assert submissions
    assert all(not row["ok"] for row in submissions)
    assert not (run_dir / "period_records.jsonl").read_text().strip()


def test_scripted_router_never_supplies_a_missing_participant_choice():
    with pytest.raises(ValueError, match="explicit request"):
        asyncio.run(ScriptedLLMClient("router").call(messages=[]))


def test_participant_cannot_submit_after_observation_delivery_fails(tmp_path, monkeypatch):
    _clear_runtime(monkeypatch)

    class BlindClient(ScriptedLLMClient):
        async def call(self, model=None, messages=None, **kwargs):
            self.calls += 1
            if self.kind == "router":
                raise AssertionError("A blind participant must not reach the Router model.")
            return self._response(
                name="ask_env",
                arguments={
                    "instruction": "decision_type: transfer\ndecision_id: guessed\namount: 99",
                    "readonly": False,
                },
            )

    async def unavailable_observation(self, agent_id):
        raise RuntimeError("Unable to retrieve observation.")

    monkeypatch.setattr(RiskSharingEnv, "get_observation", unavailable_observation)
    monkeypatch.setattr(run_smoke, "ScriptedLLMClient", BlindClient)
    run_dir = tmp_path / "missing-observation"
    manifest = run_experiment(
        backend="scripted", completed_period_limit=1, run_dir=run_dir,
        request_limit=100, agent_turns=2, router_steps=2,
    )
    assert manifest["success"] is False
    assert manifest["completed_periods"] == []
    assert manifest["router_model_requests"] == 0
    actions = [json.loads(line) for line in (run_dir / "agent_actions.jsonl").read_text().splitlines()]
    assert actions
    assert all(row["action"] == "participant_decision_rejected" for row in actions)
    assert all("observation is required" in row["error"] for row in actions)
