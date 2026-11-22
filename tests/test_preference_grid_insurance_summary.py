import pytest

from experiments.preference_grid.insurance_summary import (
    build_insurance_summary,
    render_insurance_summary,
)


def session(treatment, first, second, *, status="complete", periods=48):
    field = "purchase" if treatment == "25v" else "assigned_insurance"
    return {
        "treatment": treatment, "status": status, "completed_periods": periods,
        f"first_{field}": first, f"second_{field}": second,
    }


def test_people_and_ordered_pairs_each_have_their_own_denominator():
    voluntary = [
        session("25v", True, True), session("25v", False, True),
        session("25v", False, False), session("25v", False, False),
    ]
    # A failed/retried experiment, a short test, and a missing decision must
    # never be counted as voluntary refusals.
    voluntary.extend([
        session("25v", False, False, status="failed"),
        session("25v", False, False, periods=15),
        session("25v", True, None),
    ])
    rows = build_insurance_summary(voluntary + [
        session("25f", True, False), session("25f", True, True),
        session("25f", False, False), session("baseline", None, None),
    ])
    v, f = rows
    assert v["planned_pairs"] == 7 and v["completed_pairs"] == 6
    assert v["reached_offer_pairs"] == 5 and v["incomplete_ledger_pairs"] == 1
    assert v["valid_pairs"] == 4 and v["eligible_people"] == 8
    assert v["insured_people"] == 3 and v["uninsured_people"] == 5
    assert v["insured_rate"] == pytest.approx(3 / 8)
    assert v["first_only_pairs"] == 0 and v["second_only_pairs"] == 1
    assert v["neither_rate"] == .5
    assert f["valid_pairs"] == 2 and f["incomplete_ledger_pairs"] == 1
    assert f["insured_people"] == 3 and f["neither_pairs"] == 0
    assert f["measure"] == "forced_assignment" and v["measure"] == "voluntary_purchase"
    html = render_insurance_summary(rows)
    assert "3 / 8" in html and "25.0%" in html and "75.0%" in html


def test_short_runs_and_missing_records_do_not_create_zero_coverage_rates():
    rows = build_insurance_summary([
        session("25v", None, None, periods=1),
        session("25v", False, 0, periods=16),
    ])
    assert rows[0]["eligible_people"] == 0
    assert rows[0]["insured_rate"] is None and rows[0]["neither_rate"] is None
    assert "0.0%" not in render_insurance_summary(rows)
    assert build_insurance_summary([session("baseline", False, False)]) == []
