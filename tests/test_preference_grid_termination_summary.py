import pytest

from experiments.preference_grid.termination_summary import build_termination_distribution


def session(period, treatment="baseline", **changes):
    return {
        "status": "complete", "run_mode": "full_session", "natural_termination": True,
        "stop_reason": "natural_termination", "completed_periods": period,
        "treatment": treatment, **changes,
    }


def test_ending_share_and_conditional_rate_have_distinct_denominators():
    sessions = [session(p) for p in (48, 48, 49, 49, 50)]
    sessions.extend([
        session(49, run_mode="smoke"), session(49, status="failed"),
        session(49, stop_reason="timeout"), session(48, natural_termination=False),
        session(47), session(51), session(48, "25v"),
    ])
    rows = build_termination_distribution(sessions)
    assert len(rows) == 9
    lookup = {(r["treatment"], r["period"]): r for r in rows}
    r = lookup["baseline", 49]
    assert (r["at_risk"], r["ended"], r["continued"], r["n_natural"]) == (3, 2, 1, 5)
    assert r["share_all"] == .4 and r["conditional_rate"] == pytest.approx(2/3)
    assert r["protocol_rate"] == .78
    last = lookup["baseline", 50]
    assert last["at_risk"] == 1 and last["conditional_rate"] == last["protocol_rate"] == 1
    # One eligible session ended at 48; nobody remained exposed at 49.
    assert lookup["25v", 49]["share_all"] == 0
    assert lookup["25v", 49]["conditional_rate"] is None
    assert lookup["25f", 49]["share_all"] is None


def test_ineligible_batch_has_zero_counts_without_fabricated_rates():
    rows = build_termination_distribution([session(50, run_mode="smoke")])
    assert all(row["n_natural"] == row["ended"] == row["at_risk"] == row["continued"] == 0
               for row in rows)
    assert all(row["share_all"] is None and row["conditional_rate"] is None for row in rows)
