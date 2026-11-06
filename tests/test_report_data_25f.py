"""Computer assignments stay distinct from purchases in the visual report."""

from __future__ import annotations

import copy
import json
import random

import pytest

from experiments.report_data import build_report_data


def assigned(agent, partner, pair, insured, **extra):
    return {
        "agent_id": agent, "partner_id": partner, "pair_id": pair,
        "period": 16, "insured": insured,
        "assignment_source": "environment_random", "treatment": "25F", **extra,
    }


def settled(agent, partner, pair, period, *, insured=False, sent=0, **extra):
    return {
        "agent_id": agent, "partner_id": partner, "pair_id": pair, "period": period,
        "transfer_sent": sent, "final_income": 225, "insurance_assigned": insured,
        "insured": insured and 16 <= period <= 30,
        "insurance_cost": 25 if insured and 16 <= period <= 30 else 0,
        "insurance_payout": 0, **extra,
    }


def assert_no_purchase(pair):
    assert pair["purchased"] == pair["submitted"] == 0
    assert all(member["purchase"] is None for member in pair["members"])


def test_25f_many_pairs_use_original_assignments_and_recorded_costs_after_withdrawal():
    agents = list(range(1, 81))
    random.Random(2026).shuffle(agents)
    records, assignments, expected = [], [], {}
    for offset in range(40):
        left, right = agents[offset * 2:offset * 2 + 2]
        pair = offset + 1
        states = (True, offset % 3 == 0)
        expected[pair] = {left: states[0], right: states[1]}
        assignments.extend([assigned(left, right, pair, states[0]), assigned(right, left, pair, states[1])])
        for period in (16, 31):
            records.extend([
                settled(left, right, pair, period, insured=states[0], sent=pair,
                        insurance_payout=50 if period == 16 else 0),
                settled(right, left, pair, period, insured=states[1]),
            ])
    random.Random(7).shuffle(assignments)
    original = copy.deepcopy((records, assignments))
    data = build_report_data({"treatment": "25F", "agents": 4}, records, assignments=assignments)
    assert data["insurance"] is True and data["insurance_mode"] == "forced"
    assert data["reached_insurance"] is True
    assert data["agents"] == 80
    assert len(data["pairs"]) == 40
    assert data["periods"] == list(range(1, 32))
    for pair in data["pairs"]:
        states = expected[pair["id"]]
        assert {member["id"]: member["assigned"] for member in pair["members"]} == states
        assert all(member["assignment_status"] == "valid" for member in pair["members"])
        assert pair["assigned"] == 2 and pair["covered"] == sum(states.values())
        assert pair["status"] == ("双方获保" if pair["covered"] == 2 else "单方获保")
        assert pair["premium"] == 25 * pair["covered"]
        assert pair["payout"] == 50
        assert pair["transfer"][15] == pair["transfer"][30] == pair["id"]
        assert pair["transfer"][16] is None
        assert pair["summary_all"]["transfer_valid_periods"] == 2
        assert pair["summary_all"]["transfer_mean"] == pair["id"]
        assert_no_purchase(pair)
    assert (records, assignments) == original
    json.dumps(data, allow_nan=False)


@pytest.mark.parametrize("states,status", [
    ((True, True), "双方获保"), ((True, False), "单方获保"), ((False, True), "单方获保"),
])
def test_assignments_without_settlements_are_visible_without_inventing_money(states, status):
    data = build_report_data({"treatment": "25F"}, [], assignments=[
        assigned(9, 2, 7, states[0]), assigned(2, 9, 7, states[1]),
    ])
    pair = data["pairs"][0]
    assert data["periods"] == [] and data["completed_periods"] == 0
    assert pair["status"] == status
    assert pair["assigned"] == 2 and pair["covered"] == sum(states)
    assert pair["premium"] == pair["payout"] == pair["period_count"] == 0
    assert pair["transfer"] == []
    assert all(member["income"] == [] for member in pair["members"])
    assert_no_purchase(pair)


def test_before_assignment_is_not_reported_as_uninsured_or_declining():
    data = build_report_data({"treatment": "25F"}, [settled(1, 2, 1, 15), settled(2, 1, 1, 15)])
    pair = data["pairs"][0]
    assert data["reached_insurance"] is False
    assert pair["status"] == "尚未分配保险"
    assert pair["assigned"] == pair["covered"] == 0
    assert all(member["assigned"] is None and member["assignment_status"] == "missing" for member in pair["members"])
    assert_no_purchase(pair)


def test_missing_original_assignment_is_not_inferred_from_settlement_coverage():
    records = [settled(1, 2, 1, 31, insured=True), settled(2, 1, 1, 31, insured=False)]
    data = build_report_data({"treatment": "25F"}, records)
    pair = data["pairs"][0]
    assert data["reached_insurance"] is True
    assert pair["status"] == "分配记录未齐"
    assert pair["assigned"] == pair["covered"] == 0
    assert all(member["assigned"] is None for member in pair["members"])
    partial = build_report_data({"treatment": "25F"}, records, assignments=[assigned(1, 2, 1, True)])["pairs"][0]
    assert partial["status"] == "分配记录未齐"
    assert partial["assigned"] == partial["covered"] == 1
    assert [member["assigned"] for member in partial["members"]] == [True, None]


@pytest.mark.parametrize("invalid_fields", [
    {"insured": 1}, {"insured": "false"}, {"insured": None},
    {"assignment_source": "participant_model"}, {"period": 15}, {"period": "16"},
    {"treatment": "25V"}, {"partner_id": None}, {"partner_id": 1},
])
def test_invalid_assignment_fields_are_not_coerced_to_coverage(invalid_fields):
    bad = assigned(1, 2, 1, True)
    bad.update(invalid_fields)
    pair = build_report_data({"treatment": "25F"}, [], assignments=[bad, assigned(2, 1, 1, False)])["pairs"][0]
    assert pair["status"] == "分配记录需核对"
    first = next(member for member in pair["members"] if member["id"] == 1)
    assert first["assigned"] is None and first["assignment_status"] == "invalid"
    assert pair["assigned"] == 1 and pair["covered"] == 0
    assert_no_purchase(pair)


@pytest.mark.parametrize("duplicate_value", [True, False])
def test_duplicate_assignments_are_invalid_even_when_values_agree(duplicate_value):
    pair = build_report_data({"treatment": "25F"}, [], assignments=[
        assigned(1, 2, 1, True), assigned(2, 1, 1, False),
        assigned(1, 2, 1, duplicate_value), assigned(1, 2, 1, True),
    ])["pairs"][0]
    assert pair["status"] == "分配记录需核对"
    assert pair["assigned"] == 1 and pair["covered"] == 0
    assert pair["members"][0]["assigned"] is None
    assert pair["members"][0]["assignment_status"] == "invalid"


def test_impossible_both_uninsured_assignment_is_flagged_not_voluntary_refusal():
    pair = build_report_data({"treatment": "25F"}, [], assignments=[
        assigned(1, 2, 1, False), assigned(2, 1, 1, False),
    ])["pairs"][0]
    assert pair["status"] == "分配记录需核对"
    assert pair["assigned"] == pair["covered"] == 0
    assert all(member["assigned"] is None and member["assignment_status"] == "invalid" for member in pair["members"])
    assert_no_purchase(pair)


def test_mode_uses_only_its_own_ledger_for_membership_and_status():
    purchases = [{"agent_id": 91, "partner_id": 92, "pair_id": 99, "purchase": True}]
    assignments = [assigned(1, 2, 1, True), assigned(2, 1, 1, False)]
    forced = build_report_data({"treatment": "25F"}, [], purchases, assignments=assignments)
    assert [pair["id"] for pair in forced["pairs"]] == [1]
    voluntary = build_report_data({"treatment": "25V"}, [], purchases, assignments=assignments)
    assert [pair["id"] for pair in voluntary["pairs"]] == [99]
    assert voluntary["insurance_mode"] == "voluntary"
    assert voluntary["pairs"][0]["assigned"] == voluntary["pairs"][0]["covered"] == 0
    assert all(member["assigned"] is None for member in voluntary["pairs"][0]["members"])
    baseline = build_report_data({"treatment": "baseline"}, [], purchases, assignments=assignments)
    assert baseline["pairs"] == [] and baseline["insurance_mode"] == "none"


def test_forced_configuration_is_limited_to_assignment_settings_and_json_safe():
    configuration = {
        "probabilities": {"both_insured": .5, "first_only": .25, "second_only": .25, "unrelated": "LEAK_TEST"},
        "seed": 2026, "rng_source": "seeded", "probability_source": "explicit_run_configuration",
        "paper_probability_resolved": False, "unrelated": "LEAK_TEST",
    }
    data = build_report_data({"treatment": "25F", "insurance_assignment_configuration": configuration}, [])
    actual = data["insurance_assignment_configuration"]
    assert actual == {key: value for key, value in configuration.items() if key != "unrelated"} | {
        "probabilities": {key: configuration["probabilities"][key] for key in ["both_insured", "first_only", "second_only"]},
    }
    assert "LEAK_TEST" not in json.dumps(data, allow_nan=False)
    assert build_report_data({"treatment": "25V", "insurance_assignment_configuration": configuration}, [])["insurance_assignment_configuration"] is None
    assert build_report_data({"treatment": "25F"}, [])["insurance_assignment_configuration"] is None
