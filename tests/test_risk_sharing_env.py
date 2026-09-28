"""Regression tests for the Baseline risk-sharing environment."""

from __future__ import annotations

import asyncio

import pytest

from agentsociety2.env import ReActRouter

from custom.envs.risk_sharing_env import (
    RiskSharingEnv,
    RiskSharingRouterAdapter,
    Stage,
)
from protocol import paper_spec as spec


def call(awaitable):
    return asyncio.run(awaitable)


def observe(env: RiskSharingEnv, agent_id):
    with env.caller_context(agent_id):
        return call(env.get_observation(agent_id))


def submit(env: RiskSharingEnv, agent_id, decision_id, amount, decision_type="transfer"):
    with env.caller_context(agent_id):
        return call(env.submit_decision(agent_id, decision_id, decision_type, amount))


def high_and_low(env: RiskSharingEnv, pair_id: int = 1):
    members = env.pair_members[pair_id]
    high = next(
        agent_id
        for agent_id in members
        if env.round_states[pair_id].random_income[agent_id] == spec.HIGH_RANDOM_INCOME
    )
    low = next(agent_id for agent_id in members if agent_id != high)
    return high, low


def submit_current(env: RiskSharingEnv, amount: int, pair_id: int = 1):
    high, _ = high_and_low(env, pair_id)
    observation = observe(env, high)
    confirmation = submit(env, high, observation["decision_id"], amount)
    return high, confirmation


def complete_current_period_decisions(env: RiskSharingEnv, amount: int = 0):
    """Submit one required decision for every pair; settlement stays in step()."""

    confirmations = []
    for pair_id in env.pair_members:
        high, _ = high_and_low(env, pair_id)
        observation = observe(env, high)
        confirmations.append(submit(env, high, observation["decision_id"], amount))
    return confirmations


def settle_step(env: RiskSharingEnv):
    call(env.step(1, env.t))


class SequenceRng:
    """A deterministic RNG whose shuffle preserves input order."""

    def __init__(self, values):
        self.values = list(values)
        self.calls = 0

    def shuffle(self, sequence):
        return None

    def random(self):
        self.calls += 1
        if not self.values:
            raise AssertionError("test RNG was asked for more draws than expected")
        return self.values.pop(0)


class ReActDispatchProbe(ReActRouter):
    """Router-dispatch component probe; this is not a full PersonAgent test."""

    def __init__(
        self,
        env: RiskSharingEnv,
        target_agent=None,
        action: str = "read",
    ):
        super().__init__([env], max_steps=1)
        self.environment = env
        self.target_agent = target_agent
        self.action = action

    async def ask(self, ctx, instruction, **kwargs):
        target = ctx["id"] if self.target_agent is None else self.target_agent
        if self.action == "read":
            result = await self._execute_tool(
                "get_observation", {"agent_id": target}, readonly=True
            )
        else:
            pair_id = self.environment.pair_by_agent[target]
            decision_id = self.environment._decision_id(pair_id, target)
            result = await self._execute_tool(
                "submit_decision",
                {
                    "agent_id": target,
                    "decision_id": decision_id,
                    "decision_type": "transfer",
                    "amount": 0,
                },
                readonly=False,
            )
        return result, "ok"


def test_initialization_pairing_and_reproducibility():
    with pytest.raises(ValueError):
        RiskSharingEnv([1, 2, 3], seed=1)

    one = RiskSharingEnv([1, 2], seed=1)
    four = RiskSharingEnv([1, 2, 3, 4], seed=1)
    same_a = RiskSharingEnv([1, 2, 3, 4], seed=77)
    same_b = RiskSharingEnv([1, 2, 3, 4], seed=77)

    assert len(one.pair_members) == 1
    assert len(four.pair_members) == 2
    assert same_a.pair_members == same_b.pair_members
    original_pairs = four.pair_members.copy()
    for _ in range(3):
        assert four.pair_members == original_pairs


def test_only_baseline_is_implemented():
    with pytest.raises(NotImplementedError, match="Baseline only"):
        RiskSharingEnv([1, 2], treatment=spec.Treatment.VOLUNTARY_INSURANCE_25)
    with pytest.raises(NotImplementedError, match="Baseline only"):
        RiskSharingEnv([1, 2], treatment=spec.Treatment.FORCED_INSURANCE_25)


def test_each_pair_has_complementary_random_income_and_observation_is_stable():
    env = RiskSharingEnv([1, 2, 3, 4], seed=3)
    first = {agent: observe(env, agent) for agent in env.participants}
    second = {agent: observe(env, agent) for agent in env.participants}

    assert first == second
    for pair_id, members in env.pair_members.items():
        values = [env.round_states[pair_id].random_income[agent] for agent in members]
        assert sorted(values) == [spec.LOW_RANDOM_INCOME, spec.HIGH_RANDOM_INCOME]
        assert [first[agent]["random_income"] for agent in members] == values


def test_participant_observation_and_submission_are_pair_isolated():
    env = RiskSharingEnv([1, 2, 3, 4], seed=2)
    first_pair = env.pair_members[1]
    second_pair = env.pair_members[2]
    first_agent = first_pair[0]
    observation = observe(env, first_agent)

    assert set(observation).isdisjoint({"session_id", "treatment", "analysis_phase"})
    assert observation["partner_id"] in first_pair
    assert set(second_pair).isdisjoint({first_agent, observation["partner_id"]})

    high, _ = high_and_low(env, 1)
    confirmation = submit(env, high, observe(env, high)["decision_id"], 0)
    assert set(confirmation) <= {"status", "decision_id", "period", "amount"}
    assert "records" not in confirmation
    assert "ending" not in confirmation
    assert env.history == []

    second_high, _ = high_and_low(env, 2)
    last_confirmation = submit(env, second_high, observe(env, second_high)["decision_id"], 0)
    assert set(last_confirmation) <= {"status", "decision_id", "period", "amount"}
    assert "pair_id" not in last_confirmation
    assert "transfer" not in last_confirmation
    assert "income" not in last_confirmation
    assert "records" not in last_confirmation
    assert "agent_id" not in last_confirmation
    settle_step(env)
    pair_observation = observe(env, first_agent)
    assert pair_observation["history"]
    assert all(record["agent_id"] in first_pair for record in pair_observation["history"])
    researcher_only_fields = {
        "treatment",
        "analysis_phase",
        "ending_probability",
        "real_maximum_period",
        "backend_termination_configuration",
    }
    assert set(observation).isdisjoint(researcher_only_fields)
    assert all(set(record).isdisjoint(researcher_only_fields) for record in pair_observation["history"])
    assert all(
        not any(key.startswith("insurance") for key in record)
        for record in pair_observation["history"]
    )
    assert all(record["agent_id"] not in second_pair for record in pair_observation["history"])
    assert env.pair_history[2]
    assert all(record["agent_id"] in second_pair for record in env.pair_history[2])
    assert len(pair_observation["history"]) < len(env.history)


def test_researcher_session_metadata_stays_out_of_all_participant_outputs():
    secret = "RESEARCHER_ONLY_TREATMENT_SEED_HORIZON_9f7e"
    env = RiskSharingEnv([1, 2], seed=4, session_id=secret)
    high, _ = high_and_low(env)
    observation = observe(env, high)
    router = ReActRouter([env], max_steps=1)

    assert secret not in env.description()
    assert all(secret not in (tool.description or "") for tool in router._tool_name_to_tool_obj.values())
    assert secret not in repr(observation)
    decision_id = observation["decision_id"]
    assert secret not in decision_id

    first = submit(env, high, decision_id, 0)
    duplicate = submit(env, high, decision_id, 0)
    assert secret not in repr(first)
    assert secret not in repr(duplicate)

    settle_step(env)
    history = observe(env, high)["history"]
    assert secret not in repr(history)

    with pytest.raises(ValueError) as error:
        submit(env, high, decision_id + "-wrong", 0)
    assert secret not in str(error.value)


def test_low_income_participant_has_no_transfer_opportunity_without_receipt():
    rng = SequenceRng([0.0])
    env = RiskSharingEnv([1, 2], rng=rng)
    high, low = high_and_low(env)
    assert high != low
    low_decision_id = env._decision_id(env.pair_by_agent[low], low)
    before_rounds = env.round_states
    before_receipts = dict(env._decision_receipts)
    before_history = list(env.history)
    before_pair_history = {pair: list(records) for pair, records in env.pair_history.items()}
    before_completed = list(env.completed_periods)
    before_period = env.current_period

    with pytest.raises(ValueError, match="high-random-income"):
        submit(env, low, low_decision_id, 0)

    assert env.round_states == before_rounds
    assert env._decision_receipts == before_receipts == {}
    assert env.history == before_history == []
    assert env.pair_history == before_pair_history
    assert env.completed_periods == before_completed == []
    assert env.current_period == before_period == 1
    assert rng.calls == 1


def test_missing_or_mismatched_caller_identity_is_rejected():
    env = RiskSharingEnv([1, 2], seed=4)
    high, _ = high_and_low(env)
    decision_id = env._decision_id(env.pair_by_agent[high], high)

    with pytest.raises(PermissionError, match="No trusted caller"):
        call(env.get_observation(high))
    with env.caller_context(high):
        with pytest.raises(PermissionError, match="does not match"):
            call(env.get_observation(env.partner_by_agent[high]))
        with pytest.raises(PermissionError, match="does not match"):
            call(env.submit_decision(env.partner_by_agent[high], decision_id, "transfer", 0))


def test_router_dispatch_rejects_cross_pair_read_and_write_for_registered_target():
    # This exercises the concrete ReActRouter dispatch component with a hand-
    # written context; it is not a complete PersonAgent-to-Router integration.
    env = RiskSharingEnv([1, 2, 3, 4], seed=10)
    caller, _ = high_and_low(env, 1)
    target, _ = high_and_low(env, 2)
    assert caller in env.participants and target in env.participants
    assert target != caller
    context = {"id": caller, "agent_id": caller, "person_id": caller}
    target_pair = env.pair_by_agent[target]
    target_before = env.round_states[target_pair]
    history_before = list(env.pair_history[target_pair])
    receipts_before = dict(env._decision_receipts)

    adapter = RiskSharingRouterAdapter(ReActDispatchProbe(env), env)
    observation, answer = call(adapter.ask(context, "observe"))
    assert answer == "ok"
    assert observation["agent_id"] == caller
    with pytest.raises(PermissionError, match="No trusted caller"):
        call(env.get_observation(caller))

    read_spoof = RiskSharingRouterAdapter(
        ReActDispatchProbe(env, target_agent=target, action="read"), env
    )
    with pytest.raises(PermissionError, match="does not match"):
        call(read_spoof.ask(context, "read target"))

    write_spoof = RiskSharingRouterAdapter(
        ReActDispatchProbe(env, target_agent=target, action="write"), env
    )
    with pytest.raises(PermissionError, match="does not match"):
        call(write_spoof.ask(context, "write target"))
    assert env.round_states[target_pair] == target_before
    assert env.pair_history[target_pair] == history_before
    assert env._decision_receipts == receipts_before

    with pytest.raises(PermissionError, match="must contain"):
        call(adapter.ask({"id": caller}, "observe"))
    with pytest.raises(PermissionError, match="agree"):
        call(adapter.ask({"id": caller, "agent_id": target, "person_id": caller}, "observe"))
    with pytest.raises(PermissionError, match="No trusted caller"):
        call(env.get_observation(caller))


def test_router_adapter_cleanup_and_concurrent_contexts_are_isolated():
    env = RiskSharingEnv([1, 2, 3, 4], seed=21)
    caller_a, _ = high_and_low(env, 1)
    caller_b, _ = high_and_low(env, 2)
    ctx_a = {"id": caller_a, "agent_id": caller_a, "person_id": caller_a}
    ctx_b = {"id": caller_b, "agent_id": caller_b, "person_id": caller_b}
    adapter_a = RiskSharingRouterAdapter(ReActDispatchProbe(env), env)
    adapter_b = RiskSharingRouterAdapter(ReActDispatchProbe(env), env)

    async def two_calls():
        return await asyncio.gather(
            adapter_a.ask(ctx_a, "observe A"),
            adapter_b.ask(ctx_b, "observe B"),
        )

    results = call(two_calls())
    assert {result[0]["agent_id"] for result in results} == {caller_a, caller_b}
    with pytest.raises(PermissionError, match="No trusted caller"):
        call(env.get_observation(caller_a))

    failing = RiskSharingRouterAdapter(
        ReActDispatchProbe(env, target_agent=caller_b, action="read"), env
    )
    with pytest.raises(PermissionError, match="does not match"):
        call(failing.ask(ctx_a, "raise"))
    with pytest.raises(PermissionError, match="No trusted caller"):
        call(env.get_observation(caller_b))


def test_transfer_validation_is_before_receipt_and_submit_does_not_advance():
    env = RiskSharingEnv([1, 2], seed=4)
    high, low = high_and_low(env)
    decision_id = observe(env, high)["decision_id"]

    for invalid in (-1, 326, 1.5, "100", None, False, True):
        with pytest.raises(ValueError):
            submit(env, high, decision_id, invalid)
        assert env.current_period == 1
        assert env.history == []

    accepted = submit(env, high, decision_id, 0)
    assert accepted == {
        "status": "submitted",
        "decision_id": decision_id,
        "period": 1,
        "amount": 0,
    }
    assert env.current_period == 1
    assert env.history == []
    assert observe(env, high)["task"] == "waiting"

    for invalid_duplicate in (0.0, False, True):
        with pytest.raises(ValueError):
            submit(env, high, decision_id, invalid_duplicate)
    duplicate = submit(env, high, decision_id, 0)
    assert duplicate["status"] == "already_recorded"
    assert env.history == []

    with pytest.raises(ValueError):
        submit(env, high, decision_id, 1)
    with pytest.raises(ValueError):
        submit(env, low, decision_id, 0)


@pytest.mark.parametrize(
    "transfer, expected_high, expected_low",
    [(0, 325, 125), (100, 225, 225), (325, 0, 450)],
)
def test_step_settlement_amounts_and_pair_total(transfer, expected_high, expected_low):
    env = RiskSharingEnv([1, 2], seed=5)
    high, _ = submit_current(env, transfer)
    assert env.current_period == 1
    settle_step(env)
    records = {record["agent_id"]: record for record in env.history}
    low = env.partner_by_agent[high]

    assert records[high]["final_income"] == expected_high
    assert records[low]["final_income"] == expected_low
    assert records[high]["final_income"] + records[low]["final_income"] == 450
    assert records[high]["analysis_phase"] == 1
    assert len(env.history) == 2
    assert env.completed_periods == [1]


def test_step_waits_for_all_pairs_and_settles_once():
    env = RiskSharingEnv([1, 2, 3, 4], seed=6)
    first_high, first_low = high_and_low(env, 1)
    second_high, _ = high_and_low(env, 2)
    first_id = observe(env, first_high)["decision_id"]
    second_id = observe(env, second_high)["decision_id"]

    submit(env, first_high, first_id, 0)
    settle_step(env)
    assert env.current_period == 1
    assert env.history == []
    assert observe(env, first_low)["task"] == "waiting"

    submit(env, second_high, second_id, 0)
    assert env.current_period == 1
    assert env.history == []
    settle_step(env)
    assert env.current_period == 2
    assert len(env.history) == 4
    settle_step(env)
    assert env.current_period == 2
    assert len(env.history) == 4


def test_researcher_analysis_phase_is_backend_only_and_46_to_50_are_none():
    # Income draws occupy the first 48 calls; .79 keeps 48 and 49 alive, and
    # period 50 ends by force without another random draw.
    rng = SequenceRng([0.0] * 48 + [0.79, 0.0, 0.79, 0.0])
    env = RiskSharingEnv([1, 2], rng=rng)
    initial_pairs = dict(env.pair_members)
    phases_seen = {}

    for _ in range(45):
        phases_seen[env.current_period] = env.analysis_phase
        high, _ = high_and_low(env)
        submit(env, high, observe(env, high)["decision_id"], 0)
        settle_step(env)
        if env.current_period in (16, 31):
            boundary_observation = observe(env, high)
            assert boundary_observation["task"] in {"transfer", "waiting"}
            assert boundary_observation["stage"] == Stage.TRANSFER_DECISION.value
            assert "insurance" not in repr(boundary_observation)
            assert "analysis_phase" not in boundary_observation
            assert boundary_observation["history"]
            assert len(boundary_observation["history"]) == (30 if env.current_period == 16 else 60)
            assert env.pair_members == initial_pairs

    assert phases_seen[1] == 1
    assert phases_seen[15] == 1
    assert phases_seen[16] == 2
    assert phases_seen[30] == 2
    assert phases_seen[31] == 3
    assert phases_seen[45] == 3
    assert env.current_period == 46
    assert env.history[-1]["period"] == 45
    assert env.history[-1]["analysis_phase"] == 3

    for expected_period in range(46, 51):
        high, _ = high_and_low(env)
        submit(env, high, observe(env, high)["decision_id"], 0)
        settle_step(env)
        assert env.history[-1]["period"] == expected_period

    records_by_period = {}
    for record in env.history:
        records_by_period.setdefault(record["period"], []).append(record)
    assert set(records_by_period) >= {46, 47, 48, 49, 50}
    for period in range(46, 51):
        assert records_by_period[period]
        assert all(record["analysis_phase"] is None for record in records_by_period[period])
    assert env.finished is True
    assert env.current_period == 50


def _run_one_period(env: RiskSharingEnv, amount: int = 0):
    high, _ = high_and_low(env)
    confirmation = submit(env, high, observe(env, high)["decision_id"], amount)
    assert env.current_period == confirmation["period"]
    settle_step(env)


def test_termination_periods_and_one_draw_per_period():
    values = [0.0] * 48 + [0.79, 0.0, 0.79, 0.0]
    rng = SequenceRng(values)
    env = RiskSharingEnv([1, 2], rng=rng)
    for _ in range(47):
        _run_one_period(env)
    assert env.current_period == 48
    assert env.finished is False
    assert rng.calls == 48

    high, _ = high_and_low(env)
    submit(env, high, observe(env, high)["decision_id"], 0)
    assert env.current_period == 48
    assert rng.calls == 48  # submit_decision does not draw the ending outcome
    settle_step(env)
    assert env.current_period == 49
    assert env.finished is False
    assert rng.calls == 50

    _run_one_period(env)
    assert env.current_period == 50
    assert env.finished is False
    assert rng.calls == 52

    _run_one_period(env)
    assert env.finished is True
    assert env.current_period == 50
    assert env.stage is Stage.FINISHED
    assert rng.calls == 52

    observe(env, env.participants[0])
    settle_step(env)
    assert rng.calls == 52


def test_period_48_can_end_when_injected_draw_is_below_probability():
    rng = SequenceRng([0.0] * 48 + [0.0])
    env = RiskSharingEnv([1, 2], rng=rng)
    for _ in range(48):
        _run_one_period(env)
        if env.finished:
            break
    assert env.current_period == 48
    assert env.finished is True
    assert env.ending_outcomes[48] is True
    assert rng.calls == 49


def test_termination_threshold_branches_at_48_and_49():
    # The implementation uses u < 0.78. These values distinguish 0.78 from
    # a weaker accidental 0.50 threshold without asserting a paper boundary.
    ending_at_48 = SequenceRng([0.0] * 48 + [0.77])
    env = RiskSharingEnv([1, 2], rng=ending_at_48)
    for _ in range(47):
        _run_one_period(env)
    _run_one_period(env)
    assert env.finished is True
    assert env.current_period == 48
    assert env.ending_outcomes[48] is True

    ending_at_49 = SequenceRng([0.0] * 48 + [0.79, 0.0, 0.77])
    env = RiskSharingEnv([1, 2], rng=ending_at_49)
    for _ in range(47):
        _run_one_period(env)
    _run_one_period(env)
    assert env.current_period == 49
    assert env.finished is False
    _run_one_period(env)
    assert env.finished is True
    assert env.current_period == 49
    assert env.ending_outcomes[48] is False
    assert env.ending_outcomes[49] is True

    continuing_at_49 = SequenceRng([0.0] * 48 + [0.79, 0.0, 0.79, 0.0])
    env = RiskSharingEnv([1, 2], rng=continuing_at_49)
    for _ in range(47):
        _run_one_period(env)
    _run_one_period(env)
    _run_one_period(env)
    assert env.current_period == 50
    assert env.finished is False
    assert env.ending_outcomes[49] is False
    _run_one_period(env)
    assert env.finished is True
    assert env.current_period == 50


def test_ending_is_session_level_with_one_draw_for_multiple_pairs():
    # Two pairs require two income draws per period. At period 48 there must be
    # one ending draw for the session, not one draw per pair.
    rng = SequenceRng([0.0] * 96 + [0.79, 0.0, 0.0])
    env = RiskSharingEnv([1, 2, 3, 4], rng=rng)
    for _ in range(47):
        complete_current_period_decisions(env)
        settle_step(env)
    assert env.current_period == 48
    assert rng.calls == 96
    complete_current_period_decisions(env)
    settle_step(env)
    assert env.ending_outcomes[48] is False
    assert env.current_period == 49
    assert rng.calls == 99



def test_submit_does_not_advance_period_before_env_step():
    env = RiskSharingEnv([1, 2, 3, 4], seed=11)
    initial_period = env.current_period
    initial_income = {
        pair_id: dict(state.random_income)
        for pair_id, state in env.round_states.items()
    }

    confirmations = complete_current_period_decisions(env)

    assert all(result["status"] == "submitted" for result in confirmations)
    assert env.current_period == initial_period == 1
    assert env.history == []
    assert env.completed_periods == []
    assert {
        pair_id: dict(state.random_income)
        for pair_id, state in env.round_states.items()
    } == initial_income

    settle_step(env)
    assert env.current_period == 2
    assert env.completed_periods == [1]
    assert len(env.history) == 4


def test_observation_after_submit_stays_in_current_period_until_step():
    env = RiskSharingEnv([1, 2], seed=12)
    high, _ = high_and_low(env)
    before = observe(env, high)
    result = submit(env, high, before["decision_id"], 0)
    after = observe(env, high)

    assert result["status"] == "submitted"
    assert after["current_period"] == before["current_period"] == 1
    assert after["random_income"] == before["random_income"]
    assert after["partner"]["random_income"] == before["partner"]["random_income"]
    assert after["history"] == []
    assert env.current_period == 1


def test_pair_and_partner_mapping_stay_fixed_after_three_periods():
    env = RiskSharingEnv([1, 2, 3, 4], seed=13)
    initial_pairs = dict(env.pair_members)
    initial_partners = dict(env.partner_by_agent)

    for _ in range(3):
        complete_current_period_decisions(env)
        settle_step(env)
        assert env.pair_members == initial_pairs
        assert env.partner_by_agent == initial_partners


def test_registered_participant_cannot_impersonate_another_registered_participant():
    # This is a caller_context unit test. A real AgentSociety2 Router identity
    # binding still requires a separate integration test.
    env = RiskSharingEnv([1, 2, 3, 4], seed=14)
    caller = env.pair_members[1][0]
    impersonated = env.pair_members[2][0]

    with env.caller_context(caller):
        with pytest.raises(PermissionError, match="does not match"):
            call(env.get_observation(impersonated))


@pytest.mark.parametrize(
    "first_amount, duplicate_amount",
    [(0, False), (1, True), (100, 100.0)],
)
def test_duplicate_amounts_require_strict_integer_type(first_amount, duplicate_amount):
    env = RiskSharingEnv([1, 2], seed=15)
    high, _ = high_and_low(env)
    decision_id = observe(env, high)["decision_id"]

    submit(env, high, decision_id, first_amount)
    with pytest.raises(ValueError):
        submit(env, high, decision_id, duplicate_amount)
    assert env.round_states[1].transfer_amount[high] == first_amount
    assert env.history == []

    settle_step(env)
    assert len(env.history) == 2


def test_same_integer_duplicate_is_idempotent_without_duplicate_settlement():
    env = RiskSharingEnv([1, 2], seed=16)
    high, _ = high_and_low(env)
    decision_id = observe(env, high)["decision_id"]

    first = submit(env, high, decision_id, 100)
    duplicate = submit(env, high, decision_id, 100)

    assert first["status"] == "submitted"
    assert duplicate["status"] == "already_recorded"
    assert env.current_period == 1
    assert env.history == []

    settle_step(env)
    assert len(env.history) == 2
    assert env.completed_periods == [1]
    settle_step(env)
    assert len(env.history) == 2
    assert env.completed_periods == [1]


def test_stale_decision_id_from_previous_period_is_rejected():
    env = RiskSharingEnv([1, 2], rng=SequenceRng([0.0, 0.0]))
    high, _ = high_and_low(env)
    old_decision_id = observe(env, high)["decision_id"]

    submit(env, high, old_decision_id, 0)
    settle_step(env)
    assert env.current_period == 2

    with pytest.raises(ValueError):
        submit(env, high, old_decision_id, 0)


def test_baseline_rejects_non_transfer_decision_type():
    env = RiskSharingEnv([1, 2], seed=17)
    high, _ = high_and_low(env)
    decision_id = observe(env, high)["decision_id"]

    with pytest.raises(ValueError) as error:
        submit(env, high, decision_id, 0, decision_type="insurance")
    assert "Baseline" not in str(error.value)
    with pytest.raises(ValueError):
        submit(env, high, decision_id, 0, decision_type="other")
    assert env.history == []
    assert env.current_period == 1


def test_decision_id_cannot_be_used_by_another_registered_participant():
    env = RiskSharingEnv([1, 2, 3, 4], seed=18)
    first_high, _ = high_and_low(env, 1)
    second_high, _ = high_and_low(env, 2)
    first_decision_id = observe(env, first_high)["decision_id"]

    with pytest.raises(ValueError):
        submit(env, second_high, first_decision_id, 0)
    assert env.history == []
    assert env.current_period == 1


def test_round_states_snapshot_cannot_modify_internal_state():
    env = RiskSharingEnv([1, 2], seed=19)
    high, _ = high_and_low(env)
    before = observe(env, high)
    snapshot = env.round_states
    state = snapshot[1]
    state.random_income[high] = spec.LOW_RANDOM_INCOME
    state.income_before_transfer[high] = 0

    after = observe(env, high)
    assert after["random_income"] == before["random_income"]
    assert after["income_before_transfer"] == before["income_before_transfer"]


def test_completed_histories_are_isolated_between_pairs():
    env = RiskSharingEnv([1, 2, 3, 4], seed=20)
    complete_current_period_decisions(env)
    settle_step(env)

    first_pair = set(env.pair_members[1])
    second_pair = set(env.pair_members[2])
    first_observation = observe(env, env.pair_members[1][0])
    second_observation = observe(env, env.pair_members[2][0])
    first_history_agents = {record["agent_id"] for record in first_observation["history"]}
    second_history_agents = {record["agent_id"] for record in second_observation["history"]}

    assert first_history_agents == first_pair
    assert second_history_agents == second_pair
    assert first_history_agents.isdisjoint(second_pair)
    assert second_history_agents.isdisjoint(first_pair)

