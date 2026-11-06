"""25V mechanics and offline integration with the installed AgentSociety2.

The integration tests replace only model responses, not Agent, Router,
identity binding, decorated tools or scheduler settlement.
"""

import asyncio
import copy
import hashlib
import html
import json
import os
from importlib import import_module

import pytest

from custom.envs.risk_sharing_env import (
    PARTICIPANT_DECISION_PREFIX, RiskSharingEnv, RiskSharingReActRouter,
    RiskSharingRouterAdapter, Stage, parse_participant_decision,
)
from experiments.baseline import run_smoke as shared
from protocol import paper_spec as spec


run_experiment = import_module("experiments.25v.run_smoke").run_experiment
report = import_module("experiments.25v.report")


def observe(env, agent_id):
    with env.caller_context(agent_id):
        return asyncio.run(env.get_observation(agent_id))


def insurance(env, agent_id, purchase, token=None, decision_type="insurance"):
    token = token or env._decision_id(env.pair_by_agent[agent_id], agent_id, "insurance")
    with env.caller_context(agent_id):
        return asyncio.run(env.submit_insurance_decision(agent_id, token, decision_type, purchase))


def step(env):
    asyncio.run(env.step(1, env.t))


def transfers(env, amount=0):
    assert env.stage is Stage.TRANSFER_DECISION
    for agent_id in env.participants:
        obs = observe(env, agent_id)
        if obs["task"] == "transfer":
            with env.caller_context(agent_id):
                asyncio.run(env.submit_decision(agent_id, obs["decision_id"], "transfer", amount))


def advance_to_offer(env):
    while env.current_period < 16:
        transfers(env)
        step(env)
    assert env.stage is Stage.INSURANCE_DECISION


def accept_choices(env, choices):
    for agent_id, purchase in choices.items():
        insurance(env, agent_id, purchase)
    step(env)
    assert env.current_period == 16
    assert env.stage is Stage.TRANSFER_DECISION


class SequenceRng:
    def __init__(self, values):
        self.values = iter(values)
        self.calls = 0

    def shuffle(self, values):
        pass

    def random(self):
        self.calls += 1
        return next(self.values)


def test_v25_is_indistinguishable_from_baseline_before_offer():
    baseline = RiskSharingEnv([1, 2, 3, 4], seed=2026)
    voluntary = RiskSharingEnv([1, 2, 3, 4], seed=2026, treatment="25V")
    for period in range(1, 16):
        assert baseline.current_period == voluntary.current_period == period
        for participant in baseline.participants:
            assert observe(baseline, participant) == observe(voluntary, participant)
        router = RiskSharingReActRouter([voluntary])
        descriptions = json.dumps(router._all_tools) + voluntary.description() + voluntary.init_description()
        assert "insurance" not in descriptions.lower()
        assert "25V" not in descriptions and "25F" not in descriptions
        for env in (baseline, voluntary):
            transfers(env)
            step(env)
    assert voluntary._rounds == {}  # no period-16 income shock before choice
    assert voluntary.completed_periods == list(range(1, 16))


def test_sealed_choices_no_income_or_rng_and_step_is_only_progression_boundary():
    rng = SequenceRng([0.1] * 100)
    env = RiskSharingEnv([1, 2, 3, 4], rng=rng, treatment="25V")
    advance_to_offer(env)
    before_second = observe(env, 2)
    before_rng = rng.calls
    assert before_rng == 30
    receipt = insurance(env, 1, True)
    assert receipt == {"status": "submitted", "decision_id": receipt["decision_id"],
                       "period": 16, "purchase": True}
    assert observe(env, 2) == before_second  # not even partner submission status
    assert observe(env, 1)["task"] == "waiting"
    assert "partner_purchase" not in observe(env, 1)
    insurance(env, 2, False)
    assert observe(env, 1)["partner_purchase"] is False
    assert observe(env, 2)["partner_purchase"] is True
    step(env)  # other pair is missing; do not default either choice to False
    assert env.stage is Stage.INSURANCE_DECISION
    insurance(env, 3, False)
    insurance(env, 4, False)  # both uninsured is legal in 25V
    for participant in env.participants:
        obs = observe(env, participant)
        assert obs["task"] == "waiting"
        assert "random_income" not in obs and "income_before_transfer" not in obs
    assert rng.calls == before_rng
    assert len(env.history) == 60
    assert 16 not in env.ending_outcomes
    step(env)
    assert rng.calls == before_rng + 2  # one income draw per pair
    assert env.current_period == 16 and len(env.history) == 60
    assert env.stage is Stage.TRANSFER_DECISION
    assert observe(env, 1)["decision_id"] != receipt["decision_id"]
    step(env)  # no transfers yet
    assert rng.calls == before_rng + 2 and len(env.history) == 60
    assert 16 not in env.ending_outcomes
    transfers(env, 0)  # explicit zero, not missing
    assert len(env.history) == 60
    step(env)
    assert env.completed_periods == list(range(1, 17))
    assert len(env.history) == 64
    assert env.current_period == 17


@pytest.mark.parametrize("high_insured,low_insured", [(False, False), (True, False), (False, True), (True, True)])
def test_all_voluntary_coverage_combinations_and_transfer_limit(high_insured, low_insured):
    rng = SequenceRng([0.1] * 100)
    env = RiskSharingEnv([1, 2], treatment="25V", rng=rng)
    advance_to_offer(env)
    accept_choices(env, {1: high_insured, 2: low_insured})
    high, low = observe(env, 1), observe(env, 2)
    high_income = 300 if high_insured else 325
    low_income = 150 if low_insured else 125
    assert high["income_before_transfer"] == high_income
    assert low["income_before_transfer"] == low_income
    assert high["legal_transfer_range"] == {"min": 0, "max": high_income}
    assert low["task"] == "waiting" and low["legal_transfer_range"] is None
    before_rng = rng.calls
    with env.caller_context(2), pytest.raises(ValueError, match="high-random-income"):
        asyncio.run(env.submit_decision(2, env._decision_id(1, 2), "transfer", 0))
    with env.caller_context(1), pytest.raises(ValueError, match="current income"):
        asyncio.run(env.submit_decision(1, high["decision_id"], "transfer", high_income + 1))
    assert env.round_states[1].transfer_amount == {} and rng.calls == before_rng
    transfers(env, high_income)
    step(env)
    rows = {row["agent_id"]: row for row in env.history if row["period"] == 16}
    assert rows[1]["final_income"] == 0
    assert rows[2]["final_income"] == low_income + high_income
    assert rows[1]["insurance_cost"] == (25 if high_insured else 0)
    assert rows[2]["insurance_cost"] == (25 if low_insured else 0)
    assert rows[1]["insurance_payout"] == 0
    assert rows[2]["insurance_payout"] == (50 if low_insured else 0)


@pytest.mark.parametrize("bad", [0, 1, 0.0, 1.0, "true", "false", None])
def test_purchase_type_checked_before_receipt_and_invalid_requests_do_not_mutate(bad):
    env = RiskSharingEnv([1, 2], treatment="25V", seed=12)
    advance_to_offer(env)
    before = (copy.deepcopy(env._decision_receipts), copy.deepcopy(env.history), env._rng.getstate())
    with pytest.raises(ValueError, match="strict boolean"):
        insurance(env, 1, bad)
    assert env._insurance_choices == {}
    assert (env._decision_receipts, env.history, env._rng.getstate()) == before
    receipt = insurance(env, 1, True)
    with pytest.raises(ValueError, match="strict boolean"):
        insurance(env, 1, bad, receipt["decision_id"])
    assert env._insurance_choices == {1: True}
    assert len(env._decision_receipts) == len(before[0]) + 1
    assert env._rng.getstate() == before[2]


def test_insurance_receipts_idempotency_identity_staleness_and_type_separation():
    env = RiskSharingEnv([1, 2, 3, 4], treatment="25V", seed=22)
    advance_to_offer(env)
    token = observe(env, 1)["decision_id"]
    with pytest.raises(PermissionError):
        asyncio.run(env.submit_insurance_decision(1, token, "insurance", True))
    with env.caller_context(2), pytest.raises(PermissionError):
        asyncio.run(env.submit_insurance_decision(1, token, "insurance", True))
    insurance(env, 1, True)
    state = (dict(env._decision_receipts), env._rng.getstate())
    assert insurance(env, 1, True)["status"] == "already_recorded"
    assert (env._decision_receipts, env._rng.getstate()) == state
    for agent_id, choice in [(1, False), (2, True), (3, True)]:
        with pytest.raises(ValueError, match="locked"):
            insurance(env, agent_id, choice, token)
    # bool == 1 in Python must not turn an insurance receipt into a transfer.
    with env.caller_context(1), pytest.raises(ValueError, match="locked"):
        asyncio.run(env.submit_decision(1, token, "insurance", 1))
    for agent_id in (2, 3, 4):
        insurance(env, agent_id, False)
    step(env)
    assert insurance(env, 1, True, token)["status"] == "already_recorded"
    with pytest.raises(ValueError, match="locked"):
        insurance(env, 1, False, token)
    transfers(env)
    step(env)
    with pytest.raises(ValueError, match="stale"):
        insurance(env, 1, True, token)
    with pytest.raises(ValueError, match="not available"):
        insurance(env, 1, True)


def test_offer_and_removal_timing_history_privacy_and_fixed_pairing():
    secret = "PRIVATE_RESEARCH_SEED_HORIZON_88c72"
    env = RiskSharingEnv([101, 102, 201, 202], treatment="25V", seed=11, session_id=secret)
    pairs = copy.deepcopy(env.pair_members)
    advance_to_offer(env)
    for agent in env.participants:
        obs = observe(env, agent)
        assert "30" not in json.dumps(obs["insurance_offer"])
        assert "31" not in json.dumps(obs["insurance_offer"])
        assert secret not in json.dumps(obs)
        first = insurance(env, agent, True)
        repeated = insurance(env, agent, True)
        assert secret not in json.dumps([first, repeated])
    step(env)
    while env.current_period <= 32:
        period = env.current_period
        for agent in env.participants:
            obs = observe(env, agent)
            members = set(pairs[env.pair_by_agent[agent]])
            assert all(row["agent_id"] in members and row["partner_id"] in members for row in obs["history"])
            assert all(not {"treatment", "analysis_phase", "session_id"} & row.keys() for row in obs["history"])
            assert secret not in json.dumps(obs)
            assert obs["insured"] is (period <= 30)
            assert obs["insurance_purchased"] is True  # decision history survives withdrawal
            assert obs["task"] in {"transfer", "waiting"}
            assert ("announcement" in obs) is (period == 31)
            if period == 31:
                assert "no longer available" in obs["announcement"]
                assert len(obs["history"]) == 60
                assert obs["insurance_cost"] == obs["insurance_payout"] == 0
                assert any(row.get("insured") for row in obs["history"])
        transfers(env)
        step(env)
        assert env.pair_members == pairs
    for row in env.history:
        assert row["treatment"] == "25V"
        assert row["analysis_phase"] == (1 if row["period"] <= 15 else 2 if row["period"] <= 30 else 3)
        assert row["insurance_cost"] == (25 if 16 <= row["period"] <= 30 else 0)


@pytest.mark.parametrize("ending_draws,last", [([0.77], 48), ([0.79, 0.77], 49), ([0.79, 0.79], 50)])
def test_v25_termination_and_phase_for_every_realized_round(ending_draws, last):
    values = [0.1] * (48 * 2)
    for index, draw in enumerate(ending_draws):
        values.append(draw)
        if draw >= 0.78:
            values.extend([0.1] * 2)
    rng = SequenceRng(values)
    env = RiskSharingEnv([1, 2, 3, 4], treatment="25V", rng=rng)
    while not env.finished:
        if env.stage is Stage.INSURANCE_DECISION:
            before = rng.calls
            for agent in env.participants:
                insurance(env, agent, agent % 2 == 0)
            assert rng.calls == before
        else:
            before = rng.calls
            transfers(env)
            assert rng.calls == before
        step(env)
        if env.current_period <= 47:
            # At period 16 income hasn't yet been drawn until choices are in.
            prepared = env.current_period - (env.stage is Stage.INSURANCE_DECISION)
            assert rng.calls == prepared * 2
    assert env.completed_periods == list(range(1, last + 1))
    assert rng.calls == 2 * last + len(ending_draws)
    for period in range(46, last + 1):
        rows = [row for row in env.history if row["period"] == period]
        assert len(rows) == 4 and all(row["analysis_phase"] is None for row in rows)
    before = copy.deepcopy(env.history), rng.calls
    step(env)
    assert (env.history, rng.calls) == before


@pytest.mark.parametrize("purchase", [True, False])
def test_insurance_choices_do_not_change_income_or_ending_random_sequence(purchase):
    envs = [RiskSharingEnv([1, 2, 3, 4], treatment=t, seed=77) for t in ("baseline", "25V")]
    for env in envs:
        while not env.finished:
            if env.stage is Stage.INSURANCE_DECISION:
                for agent in env.participants:
                    insurance(env, agent, purchase)
            else:
                transfers(env)
            step(env)
    assert envs[0].pair_members == envs[1].pair_members
    assert envs[0].completed_periods == envs[1].completed_periods
    assert envs[0].ending_outcomes == envs[1].ending_outcomes
    assert [r["random_income"] for r in envs[0].history] == [r["random_income"] for r in envs[1].history]
    assert envs[0]._rng.getstate() == envs[1]._rng.getstate()


@pytest.mark.parametrize("purchase", [True, False])
def test_insurance_parser_explicit_prose_json_and_variables(purchase):
    decision = {"decision_type": "insurance", "decision_id": "opaque", "purchase": purchase}
    prose = f"I choose independently.\ndecision_type: insurance\ndecision_id: opaque\npurchase: {str(purchase).lower()}"
    assert parse_participant_decision(prose) == decision
    assert parse_participant_decision(PARTICIPANT_DECISION_PREFIX + json.dumps(decision)) == decision
    assert parse_participant_decision("My choice is supplied explicitly.", decision) == decision


@pytest.mark.parametrize("bad", [0, 1, 0.0, "true", None])
def test_insurance_parser_does_not_coerce_typed_values(bad):
    with pytest.raises(ValueError, match="strict boolean"):
        parse_participant_decision(PARTICIPANT_DECISION_PREFIX + json.dumps(
            {"decision_type": "insurance", "decision_id": "opaque", "purchase": bad}
        ))


def test_insurance_parser_rejects_ambiguous_missing_and_extra_choices():
    for instruction in (
        "decision_type: insurance\ndecision_id: token",  # no default purchase
        "decision_type: insurance\ndecision_id: token\npurchase: true\npurchase: false",
        "decision_type: insurance\ndecision_id: token\npurchase: true\namount: 0",
        "decision_type: insurance\ndecision_id: token\npurchase: 1",
    ):
        with pytest.raises(ValueError):
            parse_participant_decision(instruction)


@pytest.mark.parametrize("treatment", ["baseline", "25V"])
def test_premature_option_is_rejected_before_router_and_without_mutation(treatment):
    env = RiskSharingEnv([1, 2, 3, 4], treatment=treatment, seed=2026)
    router = RiskSharingReActRouter([env])
    adapter = RiskSharingRouterAdapter(router, env)
    before = (env.round_states, env._rng.getstate())
    token = env._decision_id(env.pair_by_agent[1], 1, "insurance")
    request = f"decision_type: insurance\ndecision_id: {token}\npurchase: true"
    with pytest.raises(ValueError, match="not available"):
        asyncio.run(adapter.ask({"id": 1, "agent_id": 1, "person_id": 1}, request))
    with pytest.raises(ValueError, match="not available"):
        insurance(env, 1, True, token)
    assert env._insurance_choices == {} and env._decision_receipts == {}
    assert (env.round_states, env._rng.getstate()) == before
    assert env._trusted_caller.get() is None
    assert env._expected_participant_decision.get() is None


@pytest.fixture
def offline_runtime(monkeypatch, tmp_path):
    for name in list(os.environ):
        if name.startswith("AGENTSOCIETY_") or name in {"LLM_API_KEY", "LLM_BASE_URL", "LLM_MODEL", "WORKSPACE_PATH"}:
            monkeypatch.delenv(name, raising=False)
    return {"backend": "scripted", "env_path": tmp_path / "nonexistent.env",
            "run_dir": tmp_path / "run", "request_limit": 400, "timeout_seconds": 120}


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_v25_full_real_framework_offline_integration(offline_runtime, monkeypatch):
    captured = []

    class InspectingClient(shared.ScriptedLLMClient):
        async def call(self, model=None, messages=None, **kwargs):
            turn = self._participant_turn(messages or []) if self.kind == "participant" else None
            captured.append((self.kind, copy.deepcopy(turn), copy.deepcopy(messages), copy.deepcopy(kwargs.get("tools"))))
            return await super().call(model=model, messages=messages, **kwargs)

    monkeypatch.setattr(shared, "ScriptedLLMClient", InspectingClient)
    manifest = run_experiment(**offline_runtime, full_session=True, agent_concurrency=4)
    assert manifest["success"] is True and manifest["stop_reason"] == "natural_termination"
    assert manifest["treatment"] == "25V"
    assert manifest["insurance_decision_count"] == 4
    n = len(manifest["completed_periods"])
    assert n in {48, 49, 50}
    assert manifest["participant_model_requests"] == 4 * (n + 1)
    assert manifest["router_model_requests"] == 2 * n + 4
    assert manifest["actual_model_requests"] == 6 * n + 8
    assert manifest["router_api_enabled"] is False
    root = offline_runtime["run_dir"]
    records = read_jsonl(root / "period_records.jsonl")
    actions = read_jsonl(root / "agent_actions.jsonl")
    choices = read_jsonl(root / "insurance_decisions.jsonl")
    assert len(records) == 4 * n
    assert {row["agent_id"]: row["purchase"] for row in choices} == {1: False, 2: True, 3: False, 4: True}
    insurance_actions = [row for row in actions if row["action"] == "submit_insurance_decision"]
    assert len(insurance_actions) == 4 and all(row["ok"] for row in insurance_actions)
    for row in insurance_actions:
        assert row["decision_source"] == "participant_model"
        choice = row["participant_model_output"]["purchase"]
        assert type(choice) is bool
        assert row["submitted_purchase"] is row["participant_purchase"] is row["environment_receipt"]["purchase"] is choice
        assert "partner_purchase" not in row["participant_model_message"]["observation"]
    insurance_tokens = {row["decision_id"] for row in insurance_actions}
    transfer_actions = [row for row in actions if row["action"] == "submit_decision"]
    assert len(transfer_actions) == n * 2
    assert not insurance_tokens & {row["decision_id"] for row in transfer_actions}
    observations = [row["observation"] for row in actions if row["action"] == "get_observation"]
    assert len(observations) == 4 * (n + 1)
    for obs in observations:
        assert not {"session_id", "treatment", "analysis_phase"} & obs.keys()
        assert all(r["agent_id"] in {obs["agent_id"], obs["partner_id"]} for r in obs["history"])
        if obs["current_period"] == 16 and obs["stage"] == "insurance_decision":
            assert "random_income" not in obs
        if obs["current_period"] < 16:
            assert "insurance" not in json.dumps(obs).lower()
    for kind, turn, messages, tools in captured:
        if kind == "participant":
            assert turn is not None
            period = turn[1]["current_period"]
            # Check actual prompts and tool schemas at the model boundary.
            if period < 16:
                assert "insurance" not in json.dumps([messages, tools]).lower()
            if period == 16 and turn[1]["task"] == "insurance":
                assert "purchase:" in messages[0]["content"]
                assert "period 30" not in messages[0]["content"]
                assert "period 31" not in messages[0]["content"]
                decision_tool = next(tool for tool in tools if tool["function"]["name"] == "ask_env")
                assert "purchase:" in json.dumps(decision_tool)
        else:
            request = shared._router_request(messages)
            assert len(tools) == 1
            assert tools[0]["function"]["name"] == request["tool"]
    steps = [r for r in read_jsonl(root / "run.log") if r["event"] == "scheduler_step_finished"]
    within_16 = [r for r in steps if r["stage_before"] == "insurance_decision"]
    assert len(within_16) == 1
    assert within_16[0]["period_before"] == within_16[0]["period_after"] == 16
    assert within_16[0]["settled_period"] is None
    assert within_16[0]["completed_periods"] == list(range(1, 16))
    for row in records:
        insured = row["agent_id"] % 2 == 0 and 16 <= row["period"] <= 30
        assert row["insured"] is insured
        assert row["insurance_cost"] == (25 if insured else 0)
        assert row["insurance_payout"] == (50 if insured and row["random_income"] == 0 else 0)
        assert row["income_before_transfer"] == 125 + row["random_income"] - row["insurance_cost"] + row["insurance_payout"]
        assert row["final_income"] == row["income_before_transfer"] - row["transfer_sent"] + row["transfer_received"]
    payment = json.loads((root / "payment.json").read_text())
    assert payment["records"] == [r for r in records if r["period"] == manifest["payment_period"]]
    summary = (root / "summary.md").read_text()
    assert (root / "report.html").is_file()
    assert "[打开交互图表报告](report.html)" in summary
    assert "已接受投保决定：4 / 4；购买：2；明确拒保：2" in summary
    assert "离线测试策略（scripted）" in summary
    assert "真实模型（llm）" not in summary
    assert "| 16—30：保险期 | 15 | 60 | 30 | 750 |" in summary
    assert "第 16 轮收入明细" in summary
    assert summary.count("一致，已接受") == 4
    assert summary.count("#### Agent ") == 4
    assert summary.count("### 各组购买情况") == 1
    for pair_id in {row["pair_id"] for row in choices}:
        members = sorted((row for row in choices if row["pair_id"] == pair_id), key=lambda row: row["agent_id"])
        member_choices = "；".join(
            f"Agent {row['agent_id']}：{'购买' if row['purchase'] else '不购买'}" for row in members)
        purchased = sum(row["purchase"] for row in members)
        status = {0: "双方不购买", 1: "单方购买", 2: "双方购买"}[purchased]
        pair_rows = [row for row in records if row["pair_id"] == pair_id and 16 <= row["period"] <= 30]
        cost = sum(row["insurance_cost"] for row in pair_rows)
        payout = sum(row["insurance_payout"] for row in pair_rows)
        assert f"| {pair_id} | {member_choices} | {purchased}/2 | {status} | 15 | {cost} | {payout} |" in summary
    for action in insurance_actions:
        quoted_request = "\n".join("> " + html.escape(line, quote=False) for line in action["participant_raw_request"].splitlines())
        assert quoted_request in summary
        assert f"> purchase: {str(action['submitted_purchase']).lower()}" in summary


def test_default_run_directory_uses_original_25v(tmp_path, monkeypatch):
    monkeypatch.setattr(shared, "PROJECT_ROOT", tmp_path)
    folder = shared._make_run_dir(None, spec.Treatment.VOLUNTARY_INSURANCE_25)
    assert folder.parent == tmp_path / "experiments" / "25v" / "runs"
    assert not (tmp_path / "experiments" / "v25").exists()


def test_all_declined_report_and_historical_refresh_preserve_raw_evidence(offline_runtime, monkeypatch):
    class DecliningParticipant(shared.ScriptedLLMClient):
        async def call(self, model=None, messages=None, **kwargs):
            turn = self._participant_turn(messages or []) if self.kind == "participant" else None
            if turn and turn[1]["task"] == "insurance":
                return self._response(name="ask_env", arguments={
                    "instruction": (
                        "A deliberate decline in this offline regression test.\n"
                        "decision_type: insurance\n"
                        f"decision_id: {turn[1]['decision_id']}\npurchase: false"
                    ),
                    "readonly": False,
                })
            return await super().call(model=model, messages=messages, **kwargs)

    monkeypatch.setattr(shared, "ScriptedLLMClient", DecliningParticipant)
    manifest = run_experiment(**offline_runtime, completed_period_limit=31)
    assert manifest["success"] is True
    root = offline_runtime["run_dir"]
    generated = (root / "summary.md").read_text()
    assert "已接受投保决定：4 / 4；购买：0；明确拒保：4" in generated
    assert "所有 Agent 都明确选择了拒保" in generated
    assert "| 16—30：保险期 | 15 | 60 | 0 | 0 | 0 |" in generated
    assert "第 16 轮投保任务及条款的观察记录：4 个 Agent；第 31 轮撤销通知的观察记录：4 个 Agent" in generated
    assert generated.count("A deliberate decline in this offline regression test.") == 4
    assert generated.count("| 0/2 | 双方不购买 | 15 | 0 | 0 |") == 2
    # Simulate an older summary with no insurance detail and an existing user note.
    start, end = generated.index(report.START), generated.index(report.END) + len(report.END)
    old = generated[:start] + generated[end:] + "\nResearcher's existing note: preserve this.\n"
    (root / "summary.md").write_text(old)
    filenames = ("manifest.json", "agent_actions.jsonl", "insurance_decisions.jsonl", "period_records.jsonl", "run.log")
    hashes = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in filenames}
    assert report.refresh_summary(root) == root.resolve() / "summary.md"
    updated = (root / "summary.md").read_text()
    assert (root / "report.html").is_file()
    assert updated.count("[打开交互图表报告](report.html)") == 1
    assert "所有 Agent 都明确选择了拒保" in updated
    assert updated.count("### 各组购买情况") == 1
    assert updated.count("| 0/2 | 双方不购买 | 15 | 0 | 0 |") == 2
    assert "Researcher's existing note: preserve this." in updated
    assert (root / "summary.before-insurance-report.md").read_text() == old
    report.refresh_summary(root)
    assert (root / "summary.md").read_text() == updated
    assert (root / "summary.before-insurance-report.md").read_text() == old
    assert {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in filenames} == hashes


def test_missing_insurance_choices_are_not_reported_as_declined():
    summary = report.render_insurance_section({"agents": 4, "backend": "llm"}, [], [], [])
    assert "已接受投保决定：0 / 4；购买：0；明确拒保：0" in summary
    assert "不能把未提交解释为拒保" in summary
    assert "所有 Agent 都明确选择了拒保" not in summary
    assert "真实模型（llm）" in summary
    assert "暂无可用的分组记录。" in summary


def _report_choice(agent_id, partner_id, pair_id, purchase):
    return {"agent_id": agent_id, "partner_id": partner_id, "pair_id": pair_id,
            "period": 16, "decision_id": f"token-{agent_id}", "purchase": purchase}


@pytest.mark.parametrize("purchases,status", [
    ((True, True), "双方购买"),
    ((True, False), "单方购买"),
    ((False, False), "双方不购买"),
])
def test_pair_insurance_report_distinguishes_accepted_combinations(purchases, status):
    # Deliberately reverse input order and use a nonconsecutive pairing.
    decisions = [_report_choice(4, 1, 7, purchases[1]), _report_choice(1, 4, 7, purchases[0])]
    summary = report.render_insurance_section({"agents": 2, "backend": "scripted"}, decisions, [], [])
    choices = f"Agent 1：{'购买' if purchases[0] else '不购买'}；Agent 4：{'购买' if purchases[1] else '不购买'}"
    assert f"| 7 | {choices} | {sum(purchases)}/2 | {status} | 0 | 0 | 0 |" in summary


def test_pair_insurance_report_keeps_missing_choice_distinct_from_decline():
    decisions = [_report_choice(3, 4, 1, True)]
    # A rejected attempt by the partner cannot become an accepted choice.
    rejected = {"action": "submit_insurance_decision", "ok": False, "agent_id": 4,
                "pair_id": 1, "submitted_purchase": False,
                "participant_model_output": {"decision_type": "insurance", "purchase": False}}
    summary = report.render_insurance_section({"agents": 2, "backend": "llm"}, decisions, [], [rejected])
    assert "| 1 | Agent 3：购买；Agent 4：未提交 | 1/2 | 决定未收齐 | 0 | 0 | 0 |" in summary
    assert "| 单方购买 |" not in summary
    assert "Agent 4：不购买" not in summary


def test_pair_insurance_report_uses_successful_observation_when_no_decisions_or_settlements():
    actions = [{"action": "get_observation", "ok": True, "agent_id": 8, "pair_id": 5,
                "observation": {"agent_id": 8, "partner_id": 2, "current_period": 16,
                                "task": "insurance", "insurance_offer": {}}},
               {"action": "get_observation", "ok": False, "agent_id": 9, "pair_id": 99,
                "observation": {"agent_id": 9, "partner_id": 10, "current_period": 16}}]
    summary = report.render_insurance_section({"agents": 2, "backend": "llm"}, [], [], actions)
    assert "| 5 | Agent 2：未提交；Agent 8：未提交 | 0/2 | 决定未收齐 | 0 | 0 | 0 |" in summary
    assert "| 99 |" not in summary
    assert "| 双方不购买 |" not in summary


def test_pair_insurance_report_before_offer_does_not_invent_declines():
    records = [{"agent_id": 3, "partner_id": 1, "pair_id": 2, "period": 15,
                "insured": False, "insurance_cost": 0, "insurance_payout": 0}]
    summary = report.render_insurance_section({"agents": 2, "backend": "scripted"}, [], records, [])
    assert "| 2 | Agent 1：未提交；Agent 3：未提交 | 0/2 | 尚未到投保阶段 | 0 | 0 | 0 |" in summary
    assert "| 双方不购买 |" not in summary


def test_pair_insurance_report_uses_recorded_pairs_and_actual_costs_after_withdrawal():
    decisions = [_report_choice(3, 4, 1, False), _report_choice(4, 3, 1, True),
                 _report_choice(1, 2, 2, False), _report_choice(2, 1, 2, False)]
    records = []
    for period in (17, 18, 31):
        for decision in decisions:
            insured = decision["purchase"] and period < 31
            records.append({"agent_id": decision["agent_id"], "partner_id": decision["partner_id"],
                            "pair_id": decision["pair_id"], "period": period, "insured": insured,
                            "insurance_cost": 25 if insured else 0,
                            "insurance_payout": 50 if insured and period == 18 else 0})
    summary = report.render_insurance_section({"agents": 4, "backend": "llm"}, decisions, records, [])
    assert "| 1 | Agent 3：不购买；Agent 4：购买 | 1/2 | 单方购买 | 2 | 50 | 50 |" in summary
    assert "| 2 | Agent 1：不购买；Agent 2：不购买 | 0/2 | 双方不购买 | 2 | 0 | 0 |" in summary


def test_report_does_not_present_a_mismatched_receipt_as_verified():
    choice = {"agent_id": 1, "pair_id": 1, "period": 16, "decision_id": "token", "purchase": False}
    action = {"action": "submit_insurance_decision", "ok": True, "agent_id": 1,
              "decision_id": "token", "decision_source": "participant_model",
              "participant_model_output": {"decision_type": "insurance", "decision_id": "token", "purchase": False},
              "submitted_purchase": False,
              "environment_receipt": {"status": "submitted", "decision_id": "another-token", "period": 16, "purchase": False},
              "participant_raw_request": "This mismatched reason must not be attributed to the choice."}
    summary = report.render_insurance_section({"agents": 4, "backend": "scripted"}, [choice], [], [action])
    assert "未找到匹配证据，需核对" in summary
    assert "This mismatched reason" not in summary


@pytest.mark.parametrize("tamper", ["purchase", "identity", "decision_id"])
def test_real_agent_router_insurance_cannot_override_participant(offline_runtime, monkeypatch, tamper):
    observed_targets = []

    class TamperingClient(shared.ScriptedLLMClient):
        async def call(self, model=None, messages=None, **kwargs):
            response = await super().call(model=model, messages=messages, **kwargs)
            if self.kind == "router":
                for tool in response.choices[0].message.tool_calls:
                    if tool.function.name == "submit_insurance_decision":
                        args = json.loads(tool.function.arguments)
                        if tamper == "purchase":
                            args["purchase"] = not args["purchase"]
                        elif tamper == "decision_id":
                            args["decision_id"] = "router-invented-token"
                        else:
                            # Obtain another pair's legal participant from the
                            # independently seeded pairing, not caller + 1.
                            pairing = RiskSharingEnv([1, 2, 3, 4], seed=2026)
                            caller = args["agent_id"]
                            target = next(a for a in pairing.participants if pairing.pair_by_agent[a] != pairing.pair_by_agent[caller])
                            assert target in pairing.participants and target != caller
                            observed_targets.append((caller, target))
                            args["agent_id"] = target
                        tool.function.arguments = json.dumps(args)
            return response

    monkeypatch.setattr(shared, "ScriptedLLMClient", TamperingClient)
    manifest = run_experiment(**offline_runtime, completed_period_limit=16, router_steps=2, agent_turns=2)
    assert manifest["success"] is False
    assert manifest["completed_periods"] == list(range(1, 16))
    assert manifest["insurance_decision_count"] == 0
    root = offline_runtime["run_dir"]
    assert read_jsonl(root / "insurance_decisions.jsonl") == []
    assert len(read_jsonl(root / "period_records.jsonl")) == 4 * 15
    attempts = [r for r in read_jsonl(root / "agent_actions.jsonl") if r["action"] == "submit_insurance_decision"]
    assert attempts and all(not r["ok"] for r in attempts)
    if tamper == "identity":
        assert observed_targets


def test_real_agent_identity_overrides_forged_insurance_ctx(offline_runtime, monkeypatch):
    forged = []
    pairing = RiskSharingEnv([1, 2, 3, 4], seed=2026)

    class ForgingParticipant(shared.ScriptedLLMClient):
        async def call(self, model=None, messages=None, **kwargs):
            response = await super().call(model=model, messages=messages, **kwargs)
            turn = self._participant_turn(messages or []) if self.kind == "participant" else None
            if turn and turn[1]["task"] == "insurance":
                caller = turn[0]
                target = next(a for a in pairing.participants if pairing.pair_by_agent[a] != pairing.pair_by_agent[caller])
                assert target in pairing.participants and target != caller
                forged.append((caller, target))
                tool = response.choices[0].message.tool_calls[0]
                args = json.loads(tool.function.arguments)
                args["ctx"] = {"id": target, "agent_id": target, "person_id": target}
                tool.function.arguments = json.dumps(args)
            return response

    monkeypatch.setattr(shared, "ScriptedLLMClient", ForgingParticipant)
    manifest = run_experiment(**offline_runtime, completed_period_limit=16, agent_concurrency=4)
    assert manifest["success"] is True
    assert len(forged) == 4
    choices = read_jsonl(offline_runtime["run_dir"] / "insurance_decisions.jsonl")
    assert {row["agent_id"]: row["purchase"] for row in choices} == {1: False, 2: True, 3: False, 4: True}
    actions = read_jsonl(offline_runtime["run_dir"] / "agent_actions.jsonl")
    for row in actions:
        if row["action"] == "submit_insurance_decision":
            assert row["agent_id"] == row["participant_model_message"]["observation"]["agent_id"]


def test_participant_repairs_invalid_insurance_without_default_choice(offline_runtime, monkeypatch):
    rejected = set()
    corrected = set()

    class RepairingParticipant(shared.ScriptedLLMClient):
        async def call(self, model=None, messages=None, **kwargs):
            turn = self._participant_turn(messages or []) if self.kind == "participant" else None
            if turn and turn[1]["task"] == "insurance":
                agent_id, obs = turn
                if agent_id not in rejected:
                    rejected.add(agent_id)
                    return self._response(name="ask_env", arguments={
                        "instruction": PARTICIPANT_DECISION_PREFIX + json.dumps({
                            "decision_type": "insurance", "decision_id": obs["decision_id"],
                            "purchase": 1,  # numeric 1 must not become True
                        }),
                        "readonly": False,
                    })
                replies = self._tool_messages(messages)
                assert replies and "strict boolean" in replies[-1]["content"]
                corrected.add(agent_id)
            return await super().call(model=model, messages=messages, **kwargs)

    monkeypatch.setattr(shared, "ScriptedLLMClient", RepairingParticipant)
    manifest = run_experiment(**offline_runtime, completed_period_limit=16, agent_concurrency=4)
    assert manifest["success"] is True
    assert rejected == corrected == {1, 2, 3, 4}
    actions = read_jsonl(offline_runtime["run_dir"] / "agent_actions.jsonl")
    errors = [row for row in actions if row["action"] == "participant_decision_rejected"]
    assert len(errors) == 4
    assert all(not row["router_model_called"] for row in errors)
    accepted = [row for row in actions if row["action"] == "submit_insurance_decision"]
    assert len(accepted) == 4 and all(row["ok"] for row in accepted)
    assert all(row["submitted_purchase"] is (row["agent_id"] % 2 == 0) for row in accepted)
