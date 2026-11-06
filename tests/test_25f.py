"""Forced coverage mechanics and the real AgentSociety2 execution chain.

The probabilities below are explicit regression-test inputs, not estimates of
the paper's unresolved assignment probabilities. Integration tests substitute
only model responses; Agent, Router, identity binding and settlement are real.
"""

import asyncio
import copy
import json
import os
from importlib import import_module

import pytest

from custom.config import RuntimeConfigError
from custom.envs.risk_sharing_env import RiskSharingEnv, RiskSharingReActRouter, Stage
from experiments.baseline import run_smoke as shared
from protocol import paper_spec as spec


def run_experiment(**kwargs):
    return import_module("experiments.25f.run_smoke").run_experiment(**kwargs)


def forced_env(participants=(1, 2, 3, 4), probabilities=(1.0, 0.0, 0.0), **kwargs):
    return RiskSharingEnv(
        participants, treatment="25F",
        forced_assignment_probabilities=probabilities, **kwargs,
    )


def observe(env, agent_id):
    with env.caller_context(agent_id):
        return asyncio.run(env.get_observation(agent_id))


def step(env):
    asyncio.run(env.step(1, env.t))


def transfers(env, amount=0):
    assert env.stage is Stage.TRANSFER_DECISION
    for agent_id in env.participants:
        obs = observe(env, agent_id)
        if obs["task"] == "transfer":
            with env.caller_context(agent_id):
                asyncio.run(env.submit_decision(agent_id, obs["decision_id"], "transfer", amount))


def advance_to(env, period):
    while env.current_period < period:
        transfers(env)
        step(env)


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def contains_key(value, forbidden):
    if isinstance(value, dict):
        return any(key in forbidden or contains_key(item, forbidden) for key, item in value.items())
    if isinstance(value, list):
        return any(contains_key(item, forbidden) for item in value)
    return False


class SequenceRng:
    def __init__(self, values, events=None, label=None):
        self.values = iter(values)
        self.calls = 0
        self.events = events
        self.label = label

    def shuffle(self, values):
        pass

    def random(self):
        self.calls += 1
        if self.events is not None:
            self.events.append(self.label)
        return next(self.values)


@pytest.mark.parametrize("probabilities", [
    None, (), (1, 0), (1, 0, 0, 0), (-0.1, 0.5, 0.6),
    (0.2, 0.3, 0.4), (0.2, 0.3, 0.6), (float("nan"), 0, 1),
    (float("inf"), 0, 0), (True, 0, 0), ("1", 0, 0),
])
def test_assignment_probabilities_must_be_explicit_finite_joint_distribution(probabilities):
    with pytest.raises((TypeError, ValueError)):
        forced_env(probabilities=probabilities, seed=2026)
    assert spec.F25_INSURANCE_ASSIGNMENT_PROBABILITY is None


@pytest.mark.parametrize("treatment", ["baseline", "25V"])
def test_assignment_configuration_is_not_accepted_for_other_treatments(treatment):
    with pytest.raises((TypeError, ValueError)):
        RiskSharingEnv([1, 2], treatment=treatment, forced_assignment_probabilities=(1, 0, 0))


def test_income_and_assignment_cannot_share_the_same_rng_object():
    rng = SequenceRng([0.1] * 100)
    with pytest.raises(ValueError, match="independent"):
        forced_env(rng=rng, assignment_rng=rng)
    assert rng.calls == 0


@pytest.mark.parametrize("probabilities,draw,coverage", [
    ((0.1, 0.8999999999995, 0.0), 0.9999999999999999, (True, False)),
    ((0.0, 1.0, 0.0), 0.0, (True, False)),
    ((0.0, 0.0, 1.0), 0.0, (False, True)),
])
def test_float_tolerance_and_zero_probability_outcomes_do_not_change_assignment_support(probabilities, draw, coverage):
    env = forced_env((1, 2), probabilities, rng=SequenceRng([0.1] * 50),
                     assignment_rng=SequenceRng([draw]))
    advance_to(env, 16)
    assert tuple(observe(env, agent)["insurance_assigned"] for agent in (1, 2)) == coverage
    assert env.assignment_configuration["probabilities"] == dict(zip(
        ("both_insured", "first_only", "second_only"), probabilities,
    ))


def test_missing_configuration_fails_before_creating_run_or_calling_a_model(tmp_path, monkeypatch):
    def unexpected_runtime(*args, **kwargs):
        pytest.fail("missing 25F probabilities must fail before runtime/model initialization")

    monkeypatch.setattr(shared, "load_runtime_config", unexpected_runtime)
    run_dir = tmp_path / "invalid-run"
    with pytest.raises((TypeError, ValueError, RuntimeConfigError)):
        run_experiment(backend="scripted", run_dir=run_dir)
    assert not run_dir.exists()


def test_no_advance_information_and_assignment_only_after_period_15_settlement():
    events = []
    income_rng = SequenceRng([0.1] * 100, events, "income")
    assignment_rng = SequenceRng([0.1, 0.9], events, "assignment")
    env = forced_env(probabilities=(0.2, 0.3, 0.5), rng=income_rng, assignment_rng=assignment_rng)
    baseline = RiskSharingEnv([1, 2, 3, 4], rng=SequenceRng([0.1] * 100))
    for period in range(1, 16):
        assert env.current_period == baseline.current_period == period
        assert env.insurance_assignments == []
        assert assignment_rng.calls == 0
        for agent in env.participants:
            assert observe(env, agent) == observe(baseline, agent)
        router = RiskSharingReActRouter([env])
        assert "insurance" not in json.dumps(router._all_tools).lower()
        assert "insurance" not in (env.description() + env.init_description()).lower()
        transfers(env)
        transfers(baseline)
        assert assignment_rng.calls == 0  # submit_decision only records
        assert env.current_period == period
        step(env)
        step(baseline)
    assert env.current_period == 16 and env.stage is Stage.TRANSFER_DECISION
    assert len(env.history) == 60
    assert assignment_rng.calls == 2 and income_rng.calls == 32
    assert events[-4:] == ["assignment", "assignment", "income", "income"]
    assert len(env.insurance_assignments) == 4
    assert env._insurance_choices == {}
    step(env)  # cannot settle before the two explicit transfer decisions
    assert env.current_period == 16 and assignment_rng.calls == 2


@pytest.mark.parametrize("probabilities,draw,coverage", [
    ((1, 0, 0), 0.9, (True, True)),
    ((0, 1, 0), 0.9, (True, False)),
    ((0, 0, 1), 0.0, (False, True)),
    ((0.2, 0.3, 0.5), 0.199, (True, True)),
    ((0.2, 0.3, 0.5), 0.2, (True, False)),
    ((0.2, 0.3, 0.5), 0.5, (False, True)),
])
def test_supported_joint_assignments_and_income_accounting(probabilities, draw, coverage):
    env = forced_env((1, 2), probabilities, rng=SequenceRng([0.1] * 50),
                     assignment_rng=SequenceRng([draw]))
    advance_to(env, 16)
    high, low = observe(env, 1), observe(env, 2)
    high_insured, low_insured = coverage
    assert (high["insurance_assigned"], low["insurance_assigned"]) == coverage
    assert (high["insured"], low["insured"]) == coverage
    assert any(coverage)
    high_income = 300 if high_insured else 325
    low_income = 150 if low_insured else 125
    assert high["income_before_transfer"] == high_income
    assert low["income_before_transfer"] == low_income
    assert high["legal_transfer_range"] == {"min": 0, "max": high_income}
    assert low["task"] == "waiting" and low["legal_transfer_range"] is None
    assert high["partner"]["insured"] is low_insured
    assert low["partner"]["insured"] is high_insured
    for agent in env.participants:
        with env.caller_context(agent), pytest.raises(ValueError, match="not available"):
            asyncio.run(env.submit_insurance_decision(agent, "no-voluntary-choice", "insurance", True))
    assert env._insurance_choices == {}
    before = env.round_states
    with env.caller_context(1), pytest.raises(ValueError, match="current income"):
        asyncio.run(env.submit_decision(1, high["decision_id"], "transfer", high_income + 1))
    with env.caller_context(2), pytest.raises(ValueError, match="high-random-income"):
        asyncio.run(env.submit_decision(2, env._decision_id(1, 2), "transfer", 0))
    assert env.round_states == before
    transfers(env, high_income)
    assert env.current_period == 16 and len(env.history) == 30
    step(env)
    rows = {r["agent_id"]: r for r in env.history if r["period"] == 16}
    assert rows[1]["final_income"] == 0
    assert rows[2]["final_income"] == low_income + high_income
    for agent, insured in zip((1, 2), coverage):
        row = rows[agent]
        assert row["insurance_cost"] == (25 if insured else 0)
        assert row["insurance_payout"] == (50 if agent == 2 and insured else 0)
        assert "insurance_purchased" not in row


def test_assignment_persists_is_private_and_cannot_be_modified_through_report_snapshots():
    secret = "RESEARCHER_PRIVATE_SESSION_25f_7a64"
    assignment_rng = SequenceRng([0.25, 0.75])
    env = forced_env(probabilities=(0, 0.5, 0.5), seed=2026,
                     assignment_rng=assignment_rng, session_id=secret)
    pairs = copy.deepcopy(env.pair_members)
    advance_to(env, 16)
    assignments = env.insurance_assignments
    original = copy.deepcopy(assignments)
    by_agent = {r["agent_id"]: r["insured"] for r in assignments}
    assert len(assignments) == len(by_agent) == 4
    for row in assignments:
        assert row["period"] == 16 and row["treatment"] == "25F"
        assert row["assignment_source"] == "environment_random"
        assert row["partner_id"] == env.partner_by_agent[row["agent_id"]]
        assert row["pair_id"] == env.pair_by_agent[row["agent_id"]]
    assignments[0]["insured"] = not assignments[0]["insured"]
    assignments.clear()
    assert env.insurance_assignments == original
    configuration = env.assignment_configuration
    assert configuration["probabilities"] == {"both_insured": 0, "first_only": 0.5, "second_only": 0.5}
    assert configuration["probability_source"] == "explicit_run_configuration"
    assert configuration["paper_probability_resolved"] is False
    assert configuration["rng_source"] == "injected"
    configuration.clear()
    assert env.assignment_configuration
    for period in range(16, 33):
        assert env.current_period == period
        for agent in env.participants:
            obs = observe(env, agent)
            members = set(pairs[env.pair_by_agent[agent]])
            assert obs["insurance_assigned"] is by_agent[agent]
            assert obs["insured"] is (by_agent[agent] and period <= 30)
            assert obs["task"] in {"transfer", "waiting"}
            assert not contains_key(obs, {"session_id", "treatment", "analysis_phase",
                                          "assignment_seed", "assignment_probabilities",
                                          "insurance_purchased", "purchase"})
            assert secret not in json.dumps(obs)
            assert all(r["agent_id"] in members and r["partner_id"] in members for r in obs["history"])
            if period == 31:
                assert "no longer available" in obs["announcement"]
                assert obs["insurance_cost"] == obs["insurance_payout"] == 0
                assert any(r.get("insured") for r in obs["history"])
        assert env.insurance_assignments == original
        assert assignment_rng.calls == 2
        assert env.pair_members == pairs
        transfers(env)
        step(env)


def test_assignment_randomness_is_reproducible_and_independent_of_income_and_ending():
    baseline = RiskSharingEnv([1, 2, 3, 4], seed=2026)
    forced = forced_env(probabilities=(0.2, 0.3, 0.5), seed=2026, assignment_seed=93)
    repeated = forced_env(probabilities=(0.2, 0.3, 0.5), seed=2026, assignment_seed=93)
    for env in (baseline, forced, repeated):
        while not env.finished:
            transfers(env)
            step(env)
        assert env.completed_periods[-1] in {48, 49, 50}
        assert not any(value for period, value in env.ending_outcomes.items() if period < 48)
    assert forced.insurance_assignments == repeated.insurance_assignments
    for env in (forced, repeated):
        assert env.pair_members == baseline.pair_members
        assert env.completed_periods == baseline.completed_periods
        assert env.ending_outcomes == baseline.ending_outcomes
        assert [r["random_income"] for r in env.history] == [r["random_income"] for r in baseline.history]
        assert env._rng.getstate() == baseline._rng.getstate()


def test_forced_transfer_identity_strict_amount_current_token_and_pair_checks_remain():
    env = forced_env(rng=SequenceRng([0.1] * 100), assignment_seed=5)
    advance_to(env, 16)
    high, other_high = observe(env, 1), observe(env, 3)
    before = (env.round_states, copy.deepcopy(env.history), env.insurance_assignments)
    with pytest.raises(PermissionError):
        asyncio.run(env.submit_decision(1, high["decision_id"], "transfer", 10))
    with env.caller_context(2), pytest.raises(PermissionError):
        asyncio.run(env.submit_decision(1, high["decision_id"], "transfer", 10))
    for amount in (-1, 301, True, 1.0, "1"):
        with env.caller_context(1), pytest.raises(ValueError):
            asyncio.run(env.submit_decision(1, high["decision_id"], "transfer", amount))
    with env.caller_context(1), pytest.raises(ValueError, match="current valid"):
        asyncio.run(env.submit_decision(1, other_high["decision_id"], "transfer", 10))
    assert (env.round_states, env.history, env.insurance_assignments) == before
    transfers(env, 30)
    with env.caller_context(1):
        assert asyncio.run(env.submit_decision(1, high["decision_id"], "transfer", 30))["status"] == "already_recorded"
    step(env)
    with env.caller_context(1), pytest.raises(ValueError, match="stale"):
        asyncio.run(env.submit_decision(1, high["decision_id"], "transfer", 30))


@pytest.fixture
def offline_runtime(monkeypatch, tmp_path):
    for name in list(os.environ):
        if name.startswith("AGENTSOCIETY_") or name in {"LLM_API_KEY", "LLM_BASE_URL", "LLM_MODEL", "WORKSPACE_PATH"}:
            monkeypatch.delenv(name, raising=False)
    return {"backend": "scripted", "env_path": tmp_path / "nonexistent.env",
            "run_dir": tmp_path / "run", "request_limit": 400,
            "timeout_seconds": 120, "agent_concurrency": 4,
            "forced_assignment_probabilities": (0.2, 0.3, 0.5), "assignment_seed": 42}


@pytest.mark.parametrize("mode", ["before_assignment", "assignment_boundary", "default", "full"])
def test_f25_real_framework_observations_routing_source_and_reports(offline_runtime, monkeypatch, mode):
    captured = []

    class InspectingClient(shared.ScriptedLLMClient):
        async def call(self, model=None, messages=None, **kwargs):
            turn = self._participant_turn(messages or []) if self.kind == "participant" else None
            captured.append((self.kind, copy.deepcopy(turn), copy.deepcopy(messages), copy.deepcopy(kwargs.get("tools"))))
            return await super().call(model=model, messages=messages, **kwargs)

    monkeypatch.setattr(shared, "ScriptedLLMClient", InspectingClient)
    options = ({"completed_period_limit": 14} if mode == "before_assignment" else
               {"completed_period_limit": 15} if mode == "assignment_boundary" else
               {"full_session": True} if mode == "full" else {})
    manifest = run_experiment(**offline_runtime, **options)
    assert manifest["success"] is True and manifest["treatment"] == "25F"
    assert manifest["router"] == "ReActRouter"
    assert manifest["agent_class"] == "RiskSharingParticipant"
    assert manifest["router_api_enabled"] is False
    n = len(manifest["completed_periods"])
    if mode == "full":
        assert n in {48, 49, 50}
        assert manifest["stop_reason"] == "natural_termination"
    else:
        assert n == {"before_assignment": 14, "assignment_boundary": 15, "default": 31}[mode]
    assert manifest["participant_model_requests"] == 4 * n
    assert manifest["router_model_requests"] == 2 * n
    assert manifest["actual_model_requests"] == 6 * n
    assert manifest["insurance_decision_count"] == 0
    # Settling period 15 also prepares period 16. Its allocation exists even
    # when the smoke limit stops before any participant sees period 16.
    expected_assignments = 0 if n < 15 else 4
    assert manifest["insurance_assignment_count"] == expected_assignments
    assert manifest["insurance_assignments_file"] == "insurance_assignments.jsonl"
    configuration = manifest["insurance_assignment_configuration"]
    assert configuration["probabilities"] == {"both_insured": 0.2, "first_only": 0.3, "second_only": 0.5}
    assert configuration["seed"] == 42
    assert configuration["paper_probability_resolved"] is False
    root = offline_runtime["run_dir"]
    assignments = read_jsonl(root / "insurance_assignments.jsonl")
    assert len(assignments) == expected_assignments
    voluntary_file = root / "insurance_decisions.jsonl"
    assert not voluntary_file.exists() or read_jsonl(voluntary_file) == []
    records = read_jsonl(root / "period_records.jsonl")
    actions = read_jsonl(root / "agent_actions.jsonl")
    assert len(records) == 4 * n
    assert not contains_key(records, {"insurance_purchased"})
    by_agent = {r["agent_id"]: r["insured"] for r in assignments}
    assert all(r["assignment_source"] == "environment_random" for r in assignments)
    assignment_actions = [r for r in actions if r["action"] == "insurance_assignment"]
    assert len(assignment_actions) == expected_assignments
    assert {r["agent_id"]: r["insured"] for r in assignment_actions} == by_agent
    for row in assignment_actions:
        assert row["decision_source"] == "environment_random"
        assert row["router_model_called"] is False
        assert not {"participant_model_output", "participant_raw_request", "environment_receipt"} & row.keys()
    assert not any(r["action"] == "submit_insurance_decision" for r in actions)
    submitted = [r for r in actions if r["action"] == "submit_decision"]
    assert len(submitted) == 2 * n and all(r["ok"] for r in submitted)
    assert len({r["submitted_amount"] for r in submitted[:2]}) == 2
    for row in submitted:
        assert row["decision_source"] == "participant_model"
        assert row["submitted_amount"] == row["participant_model_output"]["amount"] == row["environment_receipt"]["amount"]
        assert row["participant_model_message"]["observation"]["decision_id"] == row["decision_id"]
    observed = [r["observation"] for r in actions if r["action"] == "get_observation"]
    assert len(observed) == 4 * n
    if mode == "assignment_boundary":
        assert max(obs["current_period"] for obs in observed) == 15
        assert max(row["period"] for row in records) == 15
        assert {row["period"] for row in assignments} == {16}
    for obs in observed:
        assert obs["task"] in {"transfer", "waiting"}
        assert not contains_key(obs, {"session_id", "treatment", "analysis_phase", "insurance_purchased"})
        assert all(r["agent_id"] in {obs["agent_id"], obs["partner_id"]} for r in obs["history"])
    for kind, turn, messages, tools in captured:
        if kind == "participant":
            assert turn is not None
            period = turn[1]["current_period"]
            if period < 16:
                assert "insurance" not in json.dumps([messages, tools]).lower()
            assert "purchase:" not in json.dumps(tools)
            if period == 16:
                announcement = turn[1]["announcement"].lower()
                assert "computer" in announcement and "random" in announcement
                assert "period 30" not in announcement and "period 31" not in announcement
                assert "neither participant chose" in announcement and "can change" in announcement
                assert turn[1]["announcement"] in json.dumps(messages)
        else:
            request = shared._router_request(messages)
            assert request["tool"] == "submit_decision"
            assert len(tools) == 1 and tools[0]["function"]["name"] == "submit_decision"
    for row in records:
        assigned = by_agent.get(row["agent_id"], False) and row["period"] >= 16
        insured = assigned and row["period"] <= 30
        assert row["insured"] is insured
        assert row["insurance_cost"] == (25 if insured else 0)
        assert row["insurance_payout"] == (50 if insured and row["random_income"] == 0 else 0)
        assert row["income_before_transfer"] == 125 + row["random_income"] - row["insurance_cost"] + row["insurance_payout"]
        assert row["final_income"] == row["income_before_transfer"] - row["transfer_sent"] + row["transfer_received"]
    steps = [r for r in read_jsonl(root / "run.log") if r["event"] == "scheduler_step_finished"]
    assert len(steps) == n
    assert all(r["stage_before"] == "transfer_decision" for r in steps)
    summary = (root / "summary.md").read_text()
    assert "25F" in summary
    assert "分配" in summary
    assert "概率" in summary
    assert spec.F25_INSURANCE_ASSIGNMENT_PROBABILITY is None


def test_f25_participant_cannot_replace_assignment_with_a_purchase_choice(offline_runtime, monkeypatch):
    attempted = set()

    class UnauthorizedChoiceClient(shared.ScriptedLLMClient):
        async def call(self, model=None, messages=None, **kwargs):
            turn = self._participant_turn(messages or []) if self.kind == "participant" else None
            if turn and turn[1]["current_period"] == 16 and turn[1]["task"] == "transfer" and turn[0] not in attempted:
                attempted.add(turn[0])
                return self._response(name="ask_env", arguments={
                    "instruction": "I attempt to decline the assigned coverage.\n"
                        "decision_type: insurance\n"
                        f"decision_id: {turn[1]['decision_id']}\npurchase: false",
                    "readonly": False,
                })
            return await super().call(model=model, messages=messages, **kwargs)

    monkeypatch.setattr(shared, "ScriptedLLMClient", UnauthorizedChoiceClient)
    manifest = run_experiment(**offline_runtime, completed_period_limit=16)
    assert manifest["success"] is True and len(attempted) == 2
    assert manifest["insurance_decision_count"] == 0
    assert manifest["insurance_assignment_count"] == 4
    assert manifest["participant_model_requests"] == 4 * 16 + 2
    assert manifest["router_model_requests"] == 2 * 16
    root = offline_runtime["run_dir"]
    actions = read_jsonl(root / "agent_actions.jsonl")
    assert not any(r["action"] == "submit_insurance_decision" for r in actions)
    rejected = [r for r in actions if r["action"] == "participant_decision_rejected"]
    assert len(rejected) == 2
    assert all(r["router_action"] == "not_forwarded" for r in rejected)
    assigned = {r["agent_id"]: r["insured"] for r in read_jsonl(root / "insurance_assignments.jsonl")}
    final_rows = [r for r in read_jsonl(root / "period_records.jsonl") if r["period"] == 16]
    assert len(final_rows) == 4
    assert all(r["insured"] is assigned[r["agent_id"]] for r in final_rows)


@pytest.mark.parametrize("tamper", ["amount", "identity", "decision_id"])
def test_f25_real_router_cannot_change_participant_transfer(offline_runtime, monkeypatch, tamper):
    altered = []
    observed_periods = {}

    class TamperingClient(shared.ScriptedLLMClient):
        async def call(self, model=None, messages=None, **kwargs):
            if self.kind == "participant":
                turn = self._participant_turn(messages or [])
                if turn:
                    observed_periods[turn[0]] = turn[1]["current_period"]
            response = await super().call(model=model, messages=messages, **kwargs)
            if self.kind == "router":
                request = shared._router_request(messages)
                for call in response.choices[0].message.tool_calls:
                    if call.function.name != "submit_decision":
                        continue
                    arguments = json.loads(call.function.arguments)
                    if observed_periods[arguments["agent_id"]] < 16:
                        continue
                    # Corrupt a legal field, so rejection proves exact forwarding,
                    # rather than just the broad range/type checks.
                    if tamper == "amount":
                        arguments["amount"] += 1
                    elif tamper == "identity":
                        pairing = RiskSharingEnv([1, 2, 3, 4], seed=2026)
                        caller = arguments["agent_id"]
                        arguments["agent_id"] = next(a for a in pairing.participants
                            if pairing.pair_by_agent[a] != pairing.pair_by_agent[caller])
                    else:
                        arguments["decision_id"] = "router-invented-token"
                    altered.append((request["arguments"], arguments))
                    call.function.arguments = json.dumps(arguments)
            return response

    monkeypatch.setattr(shared, "ScriptedLLMClient", TamperingClient)
    manifest = run_experiment(**offline_runtime, completed_period_limit=16, router_steps=2, agent_turns=2)
    assert altered
    assert manifest["success"] is False and manifest["completed_periods"] == list(range(1, 16))
    assert manifest["insurance_assignment_count"] == 4
    actions = read_jsonl(offline_runtime["run_dir"] / "agent_actions.jsonl")
    assert not any(r["action"] == "submit_decision" and r["ok"]
                   and r["environment_receipt"]["period"] == 16 for r in actions)
    assert len(read_jsonl(offline_runtime["run_dir"] / "period_records.jsonl")) == 60

