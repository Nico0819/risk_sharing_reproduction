"""Visual summaries must preserve actual membership, choices and missing data."""

import json
import random

import pytest

from experiments.report_data import build_report_data, summarize_pair


def settlement(agent, partner, pair, period, *, sent=0, income=225, cost=0, payout=0, **extra):
    return {"agent_id": agent, "partner_id": partner, "pair_id": pair, "period": period,
            "transfer_sent": sent, "final_income": income,
            "insurance_cost": cost, "insurance_payout": payout, **extra}


def choice(agent, partner, pair, purchase):
    return {"agent_id": agent, "partner_id": partner, "pair_id": pair,
            "period": 16, "purchase": purchase}


def test_many_pairs_follow_recorded_random_membership_not_agent_id_order():
    agents = list(range(1, 81))
    random.Random(2026).shuffle(agents)
    records, decisions = [], []
    expected = {}
    for offset in range(40):
        left, right = agents[2 * offset:2 * offset + 2]
        pair = offset + 1
        expected[pair] = sorted([left, right])
        records.extend([settlement(left, right, pair, 16, sent=pair),
                        settlement(right, left, pair, 16)])
        decisions.extend([choice(left, right, pair, True), choice(right, left, pair, False)])
    random.Random(7).shuffle(records)
    data = build_report_data({"treatment": "25V", "backend": "scripted", "agents": 4}, records, decisions)
    assert data["agents"] == 80
    assert [pair["id"] for pair in data["pairs"]] == list(range(1, 41))
    for pair in data["pairs"]:
        assert [member["id"] for member in pair["members"]] == expected[pair["id"]]
        assert pair["status"] == "单方购买"
        assert pair["transfer"] == [None] * 15 + [pair["id"]]
        assert pair["period_count"] == 1
    json.dumps(data, allow_nan=False)


def test_missing_periods_and_incomplete_pairs_do_not_become_zero_transfers():
    records = [settlement(3, 9, 4, 1, sent=60), settlement(9, 3, 4, 1),
               settlement(3, 9, 4, 3, sent=20, income=305),
               settlement(3, 9, 4, 4, sent=0), settlement(9, 3, 4, 4, sent=0)]
    data = build_report_data({"treatment": "Baseline"}, records)
    pair = data["pairs"][0]
    assert data["periods"] == [1, 2, 3, 4]
    assert data["completed_periods"] == 3
    assert pair["period_count"] == 2
    assert pair["transfer"] == [60, None, None, 0]
    assert pair["members"][0]["income"] == [225, None, 305, 225]
    assert pair["members"][1]["income"] == [225, None, None, 225]


def test_insurance_choices_survive_withdrawal_and_costs_use_actual_records():
    decisions = [choice(3, 4, 1, False), choice(4, 3, 1, True)]
    records = []
    for period in (15, 16, 18, 31):
        records.extend([
            settlement(3, 4, 1, period, sent=100, transfer_received=0, insured=False),
            settlement(4, 3, 1, period, transfer_received=100, insured=16 <= period <= 30,
                       cost=25 if period in (16, 18) else 0, payout=50 if period == 18 else 0),
        ])
    pair = build_report_data({"treatment": "25V"}, records, decisions)["pairs"][0]
    assert pair["status"] == "单方购买"
    assert pair["submitted"] == 2 and pair["purchased"] == 1
    assert [member["purchase"] for member in pair["members"]] == [False, True]
    assert pair["premium"] == 50 and pair["payout"] == 50
    assert pair["transfer"][15] == 100  # received money must not be counted twice
    assert pair["transfer"][16] is None
    assert pair["period_count"] == 4


@pytest.mark.parametrize("purchases,status", [
    ((True, True), "双方购买"), ((True, False), "单方购买"), ((False, False), "双方不购买"),
])
def test_accepted_choices_without_settlements_remain_visible(purchases, status):
    choices = [choice(7, 2, 8, purchases[0]), choice(2, 7, 8, purchases[1])]
    data = build_report_data({"treatment": "25V"}, [], choices)
    pair = data["pairs"][0]
    assert data["periods"] == [] and data["completed_periods"] == 0
    assert pair["status"] == status and pair["submitted"] == 2
    assert pair["purchased"] == sum(purchases)
    assert pair["period_count"] == pair["premium"] == pair["payout"] == 0
    assert pair["transfer"] == []
    assert all(member["income"] == [] for member in pair["members"])


def test_missing_choice_is_not_decline_even_if_settlement_marks_uninsured():
    data = build_report_data({"treatment": "25V"}, [
        settlement(3, 4, 1, 16, insured=False), settlement(4, 3, 1, 16, insured=False),
    ], [choice(3, 4, 1, False)])
    pair = data["pairs"][0]
    assert pair["status"] == "决定未收齐"
    assert pair["submitted"] == 1 and pair["purchased"] == 0
    assert [member["purchase"] for member in pair["members"]] == [False, None]


def test_before_offer_has_no_invented_purchases_or_refusals():
    pair = build_report_data({"treatment": "25V"}, [
        settlement(8, 5, 2, 15), settlement(5, 8, 2, 15),
    ])["pairs"][0]
    assert pair["status"] == "尚未到投保阶段"
    assert pair["submitted"] == pair["purchased"] == 0
    assert [member["purchase"] for member in pair["members"]] == [None, None]


def test_baseline_and_empty_report_use_manifest_without_inventing_pairs():
    manifest = {"treatment": "Baseline", "backend": "llm", "success": False, "agents": 40,
                "actual_model_requests": 12, "payment_period": None}
    data = build_report_data(manifest, [])
    assert data["pairs"] == [] and data["periods"] == []
    assert data["insurance"] is False and data["agents"] == 40
    assert data["success"] is False and data["model_requests"] == 12
    assert data["payment_period"] is None
    json.dumps(data, allow_nan=False)
    baseline = build_report_data(manifest, [settlement(2, 7, 1, 1), settlement(7, 2, 1, 1)])
    assert baseline["pairs"][0]["status"] == "无保险"
    assert baseline["pairs"][0]["premium"] == baseline["pairs"][0]["payout"] == 0


@pytest.mark.parametrize("bad", [1, "false", None])
def test_invalid_choice_is_never_coerced_to_purchase_or_decline(bad):
    pair = build_report_data({"treatment": "25V"}, [], [choice(3, 4, 1, bad)])["pairs"][0]
    assert pair["status"] == "分组或选择记录需核对"
    assert pair["submitted"] == pair["purchased"] == 0
    assert pair["members"][0]["purchase"] is None


def test_duplicate_settlement_is_rejected_instead_of_double_counting():
    row = settlement(3, 4, 1, 16, cost=25)
    with pytest.raises(ValueError, match="Duplicate settlement"):
        build_report_data({"treatment": "25V"}, [row, row])


def test_non_finite_or_missing_economic_values_stay_json_safe():
    data = build_report_data({"treatment": "Baseline"}, [
        settlement(1, 2, 1, 1, sent=None, income=float("nan")), settlement(2, 1, 1, 1),
    ])
    pair = data["pairs"][0]
    assert pair["transfer"] == [None]
    assert pair["members"][0]["income"] == [None]
    json.dumps(data, allow_nan=False)


def test_comparison_uses_complete_same_period_incomes_and_independent_denominators():
    records = [
        settlement(3, 9, 4, 1, sent=100, income=225),
        settlement(9, 3, 4, 1, income=225),
        settlement(3, 9, 4, 2, sent=0, income=325),
        settlement(9, 3, 4, 2, income=125),
        # Missing member: neither a pair transfer nor a pair income observation.
        settlement(3, 9, 4, 3, sent=30, income=295),
        # The transfer remains usable even when one income is unknown.
        settlement(3, 9, 4, 4, sent=40, income=None),
        settlement(9, 3, 4, 4, income=165),
        # Conversely, a missing transfer does not erase complete income data.
        settlement(3, 9, 4, 5, sent=None, income=300),
        settlement(9, 3, 4, 5, income=150),
        # Period 6 is entirely missing; period 7 is a recorded zero transfer.
        settlement(3, 9, 4, 7, sent=0, income=325),
        settlement(9, 3, 4, 7, income=125),
    ]
    data = build_report_data({"treatment": "Baseline"}, records)
    pair = data["pairs"][0]
    assert pair["income_mean"] == [225, 225, None, None, 225, None, 225]
    assert pair["income_gap"] == [0, 200, None, None, 150, None, 200]
    assert pair["transfer_positive"] == [1, 0, None, 1, None, None, 0]
    assert pair["summary_all"] == {
        "transfer_mean": 35,
        "transfer_valid_periods": 4,
        "transfer_total": 140,
        "transfer_positive_rate": 0.5,
        "income_mean": 225,
        "income_gap_mean": 137.5,
        "income_valid_periods": 4,
    }
    assert summarize_pair(pair, data["periods"], 3, 5) == {
        "transfer_mean": 40,
        "transfer_valid_periods": 1,
        "transfer_total": 40,
        "transfer_positive_rate": 1,
        "income_mean": 225,
        "income_gap_mean": 150,
        "income_valid_periods": 1,
    }
    empty = summarize_pair(pair, data["periods"], 6, 6)
    assert empty["transfer_valid_periods"] == empty["income_valid_periods"] == 0
    assert all(value is None for key, value in empty.items() if not key.endswith("valid_periods"))
    zero = summarize_pair(pair, data["periods"], 7, 7)
    assert zero["transfer_mean"] == zero["transfer_total"] == zero["transfer_positive_rate"] == 0
    assert zero["transfer_valid_periods"] == 1


@pytest.mark.parametrize("bad_income", [None, True, "225", float("nan"), float("inf")])
def test_comparison_never_averages_one_known_member_with_missing_or_invalid_income(bad_income):
    data = build_report_data({"treatment": "Baseline"}, [
        settlement(1, 2, 1, 1, income=325), settlement(2, 1, 1, 1, income=bad_income),
    ])
    pair = data["pairs"][0]
    assert pair["members"][0]["income"] == [325]
    assert pair["income_mean"] == pair["income_gap"] == [None]
    assert pair["summary_all"]["income_valid_periods"] == 0
    assert pair["summary_all"]["income_mean"] is None
    assert pair["summary_all"]["income_gap_mean"] is None
    assert pair["summary_all"]["transfer_valid_periods"] == 1
    json.dumps(data, allow_nan=False)


def test_comparison_period_windows_exclude_the_random_extension_from_periods_31_to_45():
    records = []
    for period in range(1, 49):
        amount = 10 if period <= 15 else 20 if period <= 30 else 30 if period <= 45 else 200
        records.extend([
            settlement(5, 8, 2, period, sent=amount, income=325 - amount),
            settlement(8, 5, 2, period, income=125 + amount),
        ])
    data = build_report_data({"treatment": "25V"}, records)
    pair = data["pairs"][0]
    for start, end, expected_amount, count in [(1, 15, 10, 15), (16, 30, 20, 15),
                                                (31, 45, 30, 15), (46, 48, 200, 3)]:
        stats = summarize_pair(pair, data["periods"], start, end)
        assert stats["transfer_mean"] == expected_amount
        assert stats["transfer_total"] == expected_amount * count
        assert stats["transfer_valid_periods"] == stats["income_valid_periods"] == count
        assert stats["income_gap_mean"] == abs(200 - 2 * expected_amount)
    assert summarize_pair(pair, data["periods"], 46)["transfer_mean"] == 200
    assert summarize_pair(pair, data["periods"], end=15)["transfer_mean"] == 10
    assert summarize_pair(pair, data["periods"], 49)["transfer_valid_periods"] == 0


@pytest.mark.parametrize("pair_count", [10, 40])
def test_many_pair_comparisons_follow_actual_membership_and_each_pairs_observed_length(pair_count):
    agents = list(range(1, 2 * pair_count + 1))
    random.Random(2026).shuffle(agents)
    records, expected = [], {}
    for offset in range(pair_count):
        left, right = agents[2 * offset:2 * offset + 2]
        pair_id, length = offset + 1, 1 + offset % 7
        observed_periods = [period for period in range(1, length + 1) if period != 3]
        amounts = [pair_id + period for period in observed_periods]
        expected[pair_id] = (sorted([left, right]), observed_periods, amounts)
        for period, amount in zip(observed_periods, amounts):
            records.extend([
                settlement(left, right, pair_id, period, sent=amount, income=325 - amount),
                settlement(right, left, pair_id, period, income=125 + amount),
            ])
    random.Random(7).shuffle(records)
    data = build_report_data({"treatment": "Baseline"}, records)
    assert len(data["pairs"]) == pair_count
    assert data["periods"] == list(range(1, 8))
    for pair in data["pairs"]:
        member_ids, observed, amounts = expected[pair["id"]]
        assert [member["id"] for member in pair["members"]] == member_ids
        stats = pair["summary_all"]
        assert stats["transfer_valid_periods"] == stats["income_valid_periods"] == len(observed)
        assert stats["transfer_mean"] == sum(amounts) / len(amounts)
        assert stats["transfer_total"] == sum(amounts)
        assert stats["transfer_positive_rate"] == 1
        assert stats["income_mean"] == 225
        assert stats["income_gap_mean"] == sum(abs(200 - 2 * amount) for amount in amounts) / len(amounts)
        assert [period for period, value in zip(data["periods"], pair["income_mean"]) if value is not None] == observed
    json.dumps(data, allow_nan=False)


def test_pair_with_more_than_two_recorded_members_has_no_comparison_aggregate():
    data = build_report_data({"treatment": "Baseline"}, [
        settlement(1, 2, 8, 1), settlement(2, 1, 8, 1), settlement(3, 1, 8, 1),
    ])
    pair = data["pairs"][0]
    assert pair["transfer"] == pair["income_mean"] == pair["income_gap"] == [None]
    assert pair["summary_all"]["transfer_valid_periods"] == 0
    assert pair["summary_all"]["income_valid_periods"] == 0
