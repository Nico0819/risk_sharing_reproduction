"""Decision reviews through real PersonAgent, Router and environment components.

Only the model boundary is scripted. These checks distinguish a participant's
private, non-binding review from its subsequent submitted economic choice.
"""

from __future__ import annotations

import asyncio
import copy
import json
from importlib import import_module
from types import SimpleNamespace

import pytest

from custom.envs.risk_sharing_env import parse_participant_decision
from experiments.baseline import run_smoke


@pytest.fixture(autouse=True)
def isolated_runtime(monkeypatch):
    for prefix in ("AGENTSOCIETY_LLM", "AGENTSOCIETY_CODER_LLM"):
        for suffix in ("API_KEY", "API_BASE", "MODEL", "THINKING", "EXTRA_BODY"):
            monkeypatch.delenv(f"{prefix}_{suffix}", raising=False)
    for name in ("WORKSPACE_PATH", "LLM_API_KEY", "LLM_BASE_URL", "LLM_MODEL"):
        monkeypatch.delenv(name, raising=False)


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def is_review(messages):
    latest_user = next(
        (message for message in reversed(messages) if message.get("role") == "user"),
        {},
    )
    return "<participant_deliberation>" in str(latest_user.get("content", ""))


def marker(agent_id, decision_id):
    return f"PRIVATE_REVIEW_FOR_AGENT_{agent_id}_TOKEN_{decision_id}"


def install_spy(monkeypatch):
    original = run_smoke.ScriptedLLMClient.call
    calls = []

    async def spy(self, model=None, messages=None, **kwargs):
        messages = list(messages or [])
        review = self.kind == "participant" and is_review(messages)
        turn = self._participant_turn(messages) if self.kind == "participant" else None
        assert "participant_phase" not in kwargs, "Private routing metadata must not reach a provider."
        calls.append({
            "kind": self.kind,
            "client": id(self),
            "review": review,
            "turn": copy.deepcopy(turn),
            "messages": copy.deepcopy(messages),
            "kwargs": copy.deepcopy(kwargs),
        })
        response = await original(self, model=model, messages=messages, **kwargs)
        if review:
            assert not kwargs.get("tools"), "A review must not be able to act on the environment."
            assert kwargs.get("tool_choice") in (None, "none")
            assert turn is not None
            assert turn[1]["task"] in {"transfer", "insurance"}
            message = response.choices[0].message
            assert message.content.strip() and not message.tool_calls
            message.content = marker(turn[0], turn[1]["decision_id"])
        return response

    monkeypatch.setattr(run_smoke.ScriptedLLMClient, "call", spy)
    return calls


def assert_private_review_exchange(calls, review_rows):
    reviews = {row["decision_id"]: row for row in review_rows}
    review_calls = [call for call in calls if call["review"]]
    assert len(review_calls) == len(review_rows)
    for call in review_calls:
        agent_id, observation = call["turn"]
        row = reviews[observation["decision_id"]]
        assert row["agent_id"] == agent_id == observation["agent_id"]
        assert row["period"] == observation["current_period"]
        assert row["task"] == observation["task"]
        assert row["decision_source"] == "participant_model"
        assert row["observation"] == observation
        assert row["model_messages"] == call["messages"]
        assert row["review_text"] == marker(agent_id, observation["decision_id"])
        members = {agent_id, observation["partner_id"]}
        assert all(
            {entry["agent_id"], entry["partner_id"]} == members
            and entry["period"] < observation["current_period"]
            for entry in observation["history"]
        )
        assert not {"treatment", "analysis_phase", "session_id"}.intersection(observation)
        assert "PRIVATE_REVIEW_FOR_AGENT_" not in json.dumps(call["messages"]), (
            "No subjective note from a different task or participant may enter a fresh review."
        )

    for call in calls:
        if call["review"]:
            continue
        message_text = json.dumps(call["messages"])
        if call["kind"] == "router":
            assert "PRIVATE_REVIEW_FOR_AGENT_" not in message_text
            assert "<participant_deliberation>" not in message_text
            continue
        agent_id, observation = call["turn"]
        if observation["task"] not in {"transfer", "insurance"}:
            assert "PRIVATE_REVIEW_FOR_AGENT_" not in message_text
            continue
        own = marker(agent_id, observation["decision_id"])
        assert own in message_text, "The final choice must receive that agent's own review."
        assert message_text.count("PRIVATE_REVIEW_FOR_AGENT_") == 1
        matching_review = next(
            review for review in review_calls
            if review["turn"][1]["decision_id"] == observation["decision_id"]
        )
        assert matching_review["client"] == call["client"]


def test_concurrent_agents_review_their_own_history_before_choosing(tmp_path, monkeypatch):
    calls = install_spy(monkeypatch)
    root = tmp_path / "review-two-periods"
    manifest = run_smoke.run_experiment(
        backend="scripted", completed_period_limit=2, run_dir=root,
        participant_deliberation=True, agent_concurrency=4, request_limit=50,
    )
    assert manifest["success"] is True
    assert manifest["router"] == "ReActRouter"
    assert manifest["agent_class"] == "RiskSharingParticipant"
    assert manifest["participant_deliberation"] is True
    assert manifest["participant_deliberation_file"] == "participant_deliberations.jsonl"
    assert manifest["participant_deliberation_requests"] == 4
    assert manifest["participant_model_requests"] == 12
    assert manifest["router_model_requests"] == 4
    assert manifest["actual_model_requests"] == 16
    reviews = rows(root / manifest["participant_deliberation_file"])
    assert len(reviews) == 4
    assert {row["pair_id"] for row in reviews} == {1, 2}
    assert_private_review_exchange(calls, reviews)
    assert sorted(len(row["observation"]["history"]) for row in reviews) == [0, 0, 2, 2]

    actions = rows(root / "agent_actions.jsonl")
    submitted = [row for row in actions if row["action"] == "submit_decision"]
    assert len(submitted) == 4
    assert len({row["submitted_amount"] for row in submitted[:2]}) == 2
    for action in submitted:
        assert action["ok"] and action["decision_source"] == "participant_model"
        assert action["submitted_amount"] == action["participant_model_output"]["amount"]
        assert action["submitted_amount"] == action["environment_receipt"]["amount"]
        assert any(row["decision_id"] == action["decision_id"] for row in reviews)
    calls_log = [row for row in rows(root / "run.log") if row["event"] == "model_call_finished"]
    assert sum(row["phase"] == "deliberation" for row in calls_log) == 4
    assert sum(row["phase"] == "decision" for row in calls_log) == 8
    assert sum(row["phase"] == "routing" for row in calls_log) == 4
    assert all(row["role"] == "participant" for row in calls_log if row["phase"] == "deliberation")


def test_25v_reviews_insurance_separately_with_sealed_pair_information(tmp_path, monkeypatch):
    calls = install_spy(monkeypatch)
    root = tmp_path / "insurance-review"
    run_25v = import_module("experiments.25v.run_smoke").run_experiment
    manifest = run_25v(
        backend="scripted", completed_period_limit=16, run_dir=root,
        participant_deliberation=True, agent_concurrency=4, request_limit=200,
    )
    assert manifest["success"] is True
    assert manifest["completed_periods"] == list(range(1, 17))
    assert manifest["participant_deliberation_requests"] == 36
    assert manifest["participant_model_requests"] == 104
    assert manifest["router_model_requests"] == 36
    assert manifest["actual_model_requests"] == 140
    reviews = rows(root / "participant_deliberations.jsonl")
    assert_private_review_exchange(calls, reviews)
    insurance = [row for row in reviews if row["task"] == "insurance"]
    assert {row["agent_id"] for row in insurance} == {1, 2, 3, 4}
    assert {row["period"] for row in insurance} == {16}
    for review in insurance:
        observation = review["observation"]
        assert len(observation["history"]) == 30
        assert "partner_purchase" not in observation
        assert "random_income" not in observation
        assert "income_before_transfer" not in observation
        assert observation["own_purchase"] is None
    accepted = rows(root / "insurance_decisions.jsonl")
    assert len(accepted) == 4
    assert all(row["purchase"] is (row["agent_id"] % 2 == 0) for row in accepted)
    assert {row["decision_id"] for row in accepted} == {row["decision_id"] for row in insurance}


@pytest.mark.parametrize("bad_response", ["empty", "tool_only", "truncated"])
def test_invalid_review_cannot_submit_or_fall_back_to_a_choice(tmp_path, monkeypatch, bad_response):
    original = run_smoke.ScriptedLLMClient.call

    async def invalid_review(self, model=None, messages=None, **kwargs):
        if self.kind == "participant" and is_review(list(messages or [])):
            if bad_response == "tool_only":
                return self._response(name="ask_env", arguments={
                    "instruction": "decision_type: transfer\ndecision_id: invented\namount: 0",
                    "readonly": False,
                })
            response = self._response(content="   " if bad_response == "empty" else "An incomplete review")
            if bad_response == "truncated":
                response.choices[0].finish_reason = "length"
            return response
        return await original(self, model=model, messages=messages, **kwargs)

    monkeypatch.setattr(run_smoke.ScriptedLLMClient, "call", invalid_review)
    root = tmp_path / bad_response
    manifest = run_smoke.run_experiment(
        backend="scripted", completed_period_limit=1, run_dir=root,
        participant_deliberation=True, agent_concurrency=4, request_limit=30,
    )
    assert manifest["success"] is False
    assert manifest["completed_periods"] == []
    assert manifest["router_model_requests"] == 0
    assert not any(row["action"] == "submit_decision" for row in rows(root / "agent_actions.jsonl"))
    assert not rows(root / "period_records.jsonl")


def test_rejected_choice_is_repaired_without_repeating_the_review(tmp_path, monkeypatch):
    calls = install_spy(monkeypatch)
    spy = run_smoke.ScriptedLLMClient.call
    corrupted = set()

    async def malformed_first_choice(self, model=None, messages=None, **kwargs):
        response = await spy(self, model=model, messages=messages, **kwargs)
        turn = self._participant_turn(messages or []) if self.kind == "participant" else None
        if turn and turn[1]["task"] == "transfer" and not is_review(messages):
            token = turn[1]["decision_id"]
            if token not in corrupted:
                corrupted.add(token)
                response.choices[0].message.tool_calls[0].function.arguments = json.dumps({
                    "instruction": f"decision_type: transfer\ndecision_id: {token}",
                    "readonly": False,
                })
        return response

    monkeypatch.setattr(run_smoke.ScriptedLLMClient, "call", malformed_first_choice)
    root = tmp_path / "repair-after-review"
    manifest = run_smoke.run_experiment(
        backend="scripted", completed_period_limit=1, run_dir=root,
        participant_deliberation=True, agent_concurrency=4, request_limit=40,
    )
    assert manifest["success"] is True
    assert len(corrupted) == 2
    assert manifest["participant_deliberation_requests"] == 2
    assert manifest["actual_model_requests"] == 10
    reviews = rows(root / "participant_deliberations.jsonl")
    assert len(reviews) == 2
    assert_private_review_exchange(calls, reviews)
    actions = rows(root / "agent_actions.jsonl")
    assert len([row for row in actions if row["action"] == "participant_decision_rejected"]) == 2
    accepted = [row for row in actions if row["action"] == "submit_decision" and row["ok"]]
    assert len(accepted) == 2
    assert all(row["framework_step"] == 1 for row in accepted)


def test_default_mode_retains_original_request_cost_and_cli_can_enable_review(tmp_path):
    default_root = tmp_path / "default"
    default = run_smoke.run_experiment(
        backend="scripted", completed_period_limit=1, run_dir=default_root,
    )
    assert default["success"] is True
    assert default["actual_model_requests"] == 6
    assert default["participant_deliberation"] is False
    assert default["participant_deliberation_requests"] == 0
    assert default["participant_deliberation_file"] is None
    assert not (default_root / "participant_deliberations.jsonl").exists()

    enabled_root = tmp_path / "cli-enabled"
    exit_code = run_smoke.main([
        "--backend", "scripted", "--completed-period-limit", "1",
        "--participant-deliberation", "--agent-concurrency", "4",
        "--run-dir", str(enabled_root),
    ])
    assert exit_code == 0
    enabled = json.loads((enabled_root / "manifest.json").read_text())
    assert enabled["participant_deliberation"] is True
    assert enabled["participant_deliberation_requests"] == 2
    assert enabled["actual_model_requests"] == 8
    choices = lambda root: {
        row["agent_id"]: row["submitted_amount"]
        for row in rows(root / "agent_actions.jsonl")
        if row["action"] == "submit_decision" and row["ok"]
    }
    assert choices(default_root) == choices(enabled_root)


@pytest.mark.parametrize("output_limit", [64, 2048])
def test_review_respects_token_and_request_budgets_without_cached_tools(output_limit):
    class CapturingClient:
        model_name = "offline-capture"

        def __init__(self):
            self.requests = []

        async def call(self, **kwargs):
            self.requests.append(kwargs)
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
                content="My tentative assessment.", tool_calls=[],
            ))])

    inner = CapturingClient()
    budget = run_smoke.RequestBudget(max_calls=1)
    wrapped = run_smoke.BudgetedClient(
        inner, budget, output_limit, enforce_participant_progression=True,
    )
    tool = {"type": "function", "function": {
        "name": "ask_env", "parameters": {"type": "object", "properties": {}},
    }}
    wrapped._participant_tool_schemas = [tool]
    asyncio.run(wrapped.call(
        messages=[], participant_phase="deliberation", max_output_tokens=8192,
        tools=[tool], tool_choice="required",
    ))
    assert len(inner.requests) == budget.calls == wrapped.actual_calls == wrapped.deliberation_calls == 1
    request = inner.requests[0]
    assert request["max_output_tokens"] == min(output_limit, 512)
    assert "participant_phase" not in request
    assert not request.get("tools")
    assert request.get("tool_choice") in (None, "none")
    with pytest.raises(run_smoke.RequestBudgetExceeded):
        asyncio.run(wrapped.call(messages=[], participant_phase="deliberation"))
    assert len(inner.requests) == budget.calls == 1


def test_router_cannot_request_a_participant_review():
    router = run_smoke.ScriptedLLMClient("router")
    budget = run_smoke.RequestBudget(max_calls=10)
    wrapped = run_smoke.BudgetedClient(
        router, budget, 512, enforce_router_progression=True,
    )
    with pytest.raises(ValueError, match="Only the participant"):
        asyncio.run(wrapped.call(messages=[], participant_phase="deliberation"))
    assert budget.calls == wrapped.actual_calls == router.calls == 0


@pytest.mark.parametrize("tentative_amount", [0, 999999])
def test_review_preference_is_not_the_submitted_economic_choice(tmp_path, monkeypatch, tentative_amount):
    original = run_smoke.ScriptedLLMClient.call
    final_choices = {}
    router_choices = {}

    async def revise_preference(self, model=None, messages=None, **kwargs):
        messages = list(messages or [])
        response = await original(self, model=model, messages=messages, **kwargs)
        message = response.choices[0].message
        if self.kind == "participant" and is_review(messages):
            message.content = (
                "This is a tentative preference for later reconsideration.\n"
                f"amount: {tentative_amount}\n"
                "My final tool submission may revise this preference."
            )
        else:
            for call in message.tool_calls:
                arguments = json.loads(call.function.arguments)
                if self.kind == "participant" and call.function.name == "ask_env":
                    decision = parse_participant_decision(arguments["instruction"])
                    assert decision is not None
                    agent_id, _observation = self._participant_turn(messages)
                    final_choices[agent_id] = decision["amount"]
                elif self.kind == "router" and call.function.name == "submit_decision":
                    router_choices[arguments["agent_id"]] = arguments["amount"]
        return response

    monkeypatch.setattr(run_smoke.ScriptedLLMClient, "call", revise_preference)
    root = tmp_path / f"revised-preference-{tentative_amount}"
    manifest = run_smoke.run_experiment(
        backend="scripted", completed_period_limit=1, run_dir=root,
        participant_deliberation=True, agent_concurrency=4, request_limit=20,
    )
    assert manifest["success"] is True
    assert manifest["actual_model_requests"] == 8
    reviews = rows(root / "participant_deliberations.jsonl")
    assert len(reviews) == 2
    assert all(f"amount: {tentative_amount}" in row["review_text"] for row in reviews)
    accepted = {
        row["agent_id"]: row["environment_receipt"]["amount"]
        for row in rows(root / "agent_actions.jsonl")
        if row["action"] == "submit_decision" and row["ok"]
    }
    assert len(accepted) == 2
    assert accepted == router_choices == final_choices
    assert all(0 < amount <= 325 and amount != tentative_amount for amount in accepted.values())
