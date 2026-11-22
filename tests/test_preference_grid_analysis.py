"""Statistical units, temporal ordering, and uncertainty in the offline report."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

from experiments.preference_grid.analysis import (
    build_analysis, cluster_mean, describe, difference, fit_cluster_ols, holm,
)


def session(job_id="baseline__P1__P7__r01", treatment="baseline", *, status="complete"):
    return dict(job_id=job_id, treatment=treatment, combination_id="P1__P7", repetition=1,
                status=status, attempt_count=1, completed_periods=48, observed_transfer_mean=10,
                first_agent_id=2, second_agent_id=1,
                first_personality_id="P1", second_personality_id="P7",
                first_risk_aversion="low", second_risk_aversion="high",
                first_advantageous_inequality_aversion="low", second_advantageous_inequality_aversion="low",
                first_purchase=True if treatment == "25v" else None,
                second_purchase=False if treatment == "25v" else None,
                first_assigned_insurance=True if treatment == "25f" else None,
                second_assigned_insurance=True if treatment == "25f" else None)


def records(s, choices):
    out = []
    for period, (sender, amount) in enumerate(choices, 1):
        for agent in (1, 2):
            high = agent == sender
            before = 325 if high else 125
            out.append(dict(job_id=s["job_id"], agent_id=agent, partner_id=3-agent,
                            period=period, had_transfer_opportunity=high,
                            income_before_transfer=before,
                            final_income=before-amount if high else before+amount,
                            transfer_sent=amount if high else 0))
    return out


def analyze(sessions, rows):
    return build_analysis(sessions, rows, {"treatments": list(dict.fromkeys(s["treatment"] for s in sessions)),
                                          "forced_assignment_probabilities": [.5, .25, .25]})


def test_t_intervals_missing_singleton_and_repetition_spread():
    assert describe([])["mean"] is None
    assert describe([3])["ci_low"] is None
    r = describe([20, 100])
    assert r["mean"] == 60 and r["median"] == 60
    assert r["sd"] == pytest.approx(np.std([20, 100], ddof=1))
    assert r["ci_high"] == pytest.approx(60 + stats.t.ppf(.975, 1) * 40)
    assert describe([False, None, float("inf"), 4])["n"] == 1


def test_correlated_agents_do_not_double_independent_sample_size():
    rows = [{"job_id": j, "value": value} for j, value in (("a", 0), ("b", 100)) for _ in range(2)]
    result = cluster_mean(rows, "value")
    assert result["n_agents"] == 4 and result["n_pairs"] == 2
    assert result["se"] == pytest.approx(describe([0, 100])["se"])
    assert result["ci_low"] == pytest.approx(describe([0, 100])["ci_low"])
    assert cluster_mean(rows[:2], "value")["ci_low"] is None


def test_welch_independent_difference_matches_scipy_and_degenerate_no_test():
    x, y = [1, 4, 9, 15], [2, 2, 5, 7, 7, 8]
    expected = stats.ttest_ind(x, y, equal_var=False)
    result = difference(x, y)
    assert result["estimate"] == pytest.approx(np.mean(x)-np.mean(y))
    assert result["p_value"] == pytest.approx(expected.pvalue)
    assert result["ci_low"] == pytest.approx(expected.confidence_interval().low)
    assert difference([], y)["estimate"] is None
    assert difference([1], y)["ci_low"] is None
    assert difference([0, 0], [1, 1])["p_value"] is None


def test_holm_adjustment_monotonic_and_excludes_untestable_comparisons():
    rows = [{"p_value": p} for p in (.04, .001, None, .03)]
    holm(rows)
    assert [r["p_holm"] for r in rows] == [.06, .003, None, .06]


def test_sender_opportunities_and_personality_identity_are_not_confused():
    s = session()
    a = analyze([s], records(s, [(2, 0), (2, 0), (1, 100)]))
    rows = {r["agent_id"]: r for r in a["agent_stages"] if r["stage"] == "all"}
    assert rows[1]["personality_id"] == "P7"
    assert rows[1]["sender_opportunities"] == 1
    assert rows[1]["transfer_mean"] == 100
    assert rows[1]["transfer_positive_rate"] == 1
    assert rows[2]["transfer_mean"] == 0
    pair = next(r for r in a["pair_stages"] if r["stage"] == "all")
    assert pair["transfer_mean"] == pytest.approx(100 / 3)
    assert pair["income_mean"] == 225


def test_no_opportunity_is_missing_not_zero_and_does_not_enter_regression():
    s = session()
    a = analyze([s], records(s, [(2, 30)] * 3))
    own = next(r for r in a["agent_stages"] if r["stage"] == "all" and r["agent_id"] == 1)
    assert own["transfer_mean"] is None and own["sender_opportunities"] == 0
    assert a["regressions"][0]["status"] == "insufficient_or_rank_deficient"
    assert not a["phase_changes"]


def test_reciprocity_uses_strictly_prior_partner_sender_not_bookkeeping_zero():
    s = session()
    a = analyze([s], records(s, [(2, 80), (1, 10), (1, 30), (2, 0), (1, 50)]))
    rows = {r["lag_group"]: r for r in a["reciprocity"] if r["stage"] == "all"}
    assert rows["no_previous"]["mean"] == 80
    # Agent 1 had two chances after partner 80 (mean=20), Agent 2 one after partner 30 (mean=0).
    assert rows["previous_positive"]["mean"] == 10
    assert rows["previous_positive"]["eligible_opportunities"] == 3
    assert rows["previous_zero"]["mean"] == 50
    assert rows["previous_zero"]["eligible_opportunities"] == 1


def test_failed_sessions_excluded_and_session_means_not_weighted_by_length():
    a, b, bad = session(), session("baseline__P1__P7__r02"), session("baseline__P1__P7__r03", status="failed")
    rows = records(a, [(1, 20)]*48) + records(b, [(1, 100)]*50) + records(bad, [(1, 300)]*48)
    result = analyze([a, b, bad], rows)
    item = next(r for r in result["treatment_summaries"] if r["stage"] == "all" and r["metric"] == "transfer_mean")
    assert item["mean"] == 60 and item["n"] == 2
    assert len(result["agent_stages"]) == 24  # 2 sessions * 2 agents * 6 stages
    last = next(r for r in result["temporal"] if r["period"] == 50 and r["metric"] == "transfer_mean")
    assert last["n"] == 1 and last["ci_low"] is None


def test_phase_changes_are_paired_within_session_and_independent_across_treatments():
    sessions, rows = [], []
    for t, effects in (("baseline", (10, 20)), ("25v", (40, 60))):
        for repeat, effect in enumerate(effects):
            s = session(f"{t}__P1__P7__r{repeat+1:02d}", t)
            sessions.append(s)
            # Large baseline differences disappear in the within-session changes.
            pre = 100*repeat
            rows += records(s, [(1, pre)]*15 + [(2, pre+effect)]*15 + [(1, pre+5)]*15)
    a = analyze(sessions, rows)
    changes = {r["treatment"]: r for r in a["phase_changes"] if r["transition"] == "introduction" and r["metric"] == "transfer_mean"}
    assert changes["baseline"]["mean"] == 15
    assert changes["25v"]["mean"] == 50
    c = next(r for r in a["contrasts"] if r["kind"] == "change_vs_baseline" and r["stage"] == "introduction" and r["metric"] == "transfer_mean")
    assert c["estimate"] == 35
    assert c["se"] == pytest.approx(difference([40, 60], [10, 20])["se"])


def test_voluntary_purchase_and_forced_assignment_have_different_rates():
    v, f = session("25v__P1__P7__r01", "25v"), session("25f__P1__P7__r01", "25f")
    a = analyze([v, f], records(v, [(1, 40)]*30)+records(f, [(1, 50)]*30))
    rates = {(r["treatment"], r["personality_id"]): r["mean"] for r in a["insurance_rates"]}
    assert rates[("25v", "P7")] == 0 and rates[("25f", "P7")] == 1
    assert {r["insured_count"] for r in a["insurance"] if r["treatment"] == "25v"} == {"1"}
    assert {r["insured_count"] for r in a["insurance"] if r["treatment"] == "25f"} == {"2"}


def test_cr1_intercept_agrees_with_independent_mean_and_rank_deficiency_skips():
    y = [0, 0, 100, 100, 20, 20]
    fit = fit_cluster_ols(y, [[1]]*6, ["a", "a", "b", "b", "c", "c"])
    assert fit["coefficients"][0]["estimate"] == pytest.approx(40)
    assert fit["coefficients"][0]["se"] == pytest.approx(describe([0, 100, 20])["se"])
    assert fit_cluster_ols(y, [[1, 1]]*6, ["a", "a", "b", "b", "c", "c"]) is None


def test_cr1_general_regression_matches_direct_cluster_score_calculation():
    rng = np.random.default_rng(123)
    x = np.column_stack([np.ones(40), rng.normal(size=40)])
    y = x @ np.array([4, 2]) + np.repeat(rng.normal(size=20), 2)
    groups = np.repeat(np.arange(20), 2)
    fit = fit_cluster_ols(y, x, list(groups))
    beta = np.linalg.solve(x.T@x, x.T@y)
    scores = np.vstack([x[groups==g].T@(y[groups==g]-x[groups==g]@beta) for g in range(20)])
    cov = np.linalg.inv(x.T@x)@(scores.T@scores)@np.linalg.inv(x.T@x)*(20/19)*(39/38)
    assert [r["estimate"] for r in fit["coefficients"]] == pytest.approx(beta)
    assert [r["se"] for r in fit["coefficients"]] == pytest.approx(np.sqrt(np.diag(cov)))


def test_empty_report_remains_json_serializable_without_nan():
    a = analyze([session(status="pending")], [])
    assert not a["pair_stages"] and not a["regressions"]
    json.dumps(a, allow_nan=False)
