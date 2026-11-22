"""Research denominators for recorded evidence and the equalization benchmark."""

import pytest

from experiments.preference_grid.analysis import METRICS
from experiments.preference_grid.record_coverage import manifest_counts, record_coverage
from test_preference_grid_analysis import analyze, records, session


def test_evidence_counts_use_each_available_source_and_distinguish_zero_from_unknown():
    realistic = {
        "participant_reasoning_responses": 60,
        "participant_reasoning_readable_responses": 54,
        "participant_reasoning_truncated_responses": 2,
        "private_memory_update_count": 150,  # export below is the authoritative count
        "memory_report": {"update_count": 45, "closing_update_count": 2},
    }
    known = dict(status="complete", completed_periods=22, treatment="25v", **manifest_counts(realistic))
    missing_memory = dict(known, **manifest_counts({
        "participant_reasoning_responses": 0, "participant_reasoning_readable_responses": 0,
        "memory_report": None,
    }))
    invalid_trace = dict(known, readable_responses=61, memory_updates=0, closing_updates=0)
    ignored = dict(known, status="failed")
    legacy = dict(treatment="baseline", status="complete", completed_periods=10, **manifest_counts({}))
    result = {row["treatment"]:row for row in record_coverage([known, missing_memory, invalid_trace, ignored, legacy])}
    observed = result["25v"]
    assert (observed["complete_sessions"], observed["sender_opportunities"]) == (3, 66)
    assert (observed["trace_reporting_sessions"], observed["readable_responses"], observed["trace_responses"]) == (2, 54, 60)
    assert (observed["truncation_reporting_sessions"], observed["truncated_responses"]) == (1, 2)
    assert (observed["memory_reporting_sessions"], observed["feedback_updates"], observed["closing_updates"]) == (2, 43, 2)
    assert result["baseline"]["trace_reporting_sessions"] == 0
    assert result["baseline"]["readable_responses"] is None
    assert result["baseline"]["feedback_updates"] is None


def test_eight_metrics_and_equalization_valid_counts_after_income_changes():
    base = session()
    forced = session("25f__P1__P7__r01", "25f")
    # In the insured block the sender has 300 and the receiver 150. A 75
    # transfer equalizes final incomes just as 100 did in the uninsured block.
    choices = [(1, 100)] * 15 + [(2, 75)] * 15
    ledger = records(forced, choices)
    for row in ledger:
        if row["period"] >= 16:
            row["income_before_transfer"] = 300 if row["had_transfer_opportunity"] else 150
            row["final_income"] = 225
    outcome = analyze([base, forced], records(base, [(1, 100)] * 30) + ledger)
    observed = {r["metric"]: r for r in outcome["phase_changes"]
                if r["transition"] == "introduction" and r["treatment"] == "25f"}
    assert len(observed) == len(METRICS) == 8
    assert observed["equalization_ratio"]["mean"] == pytest.approx(0)
    assert observed["equalization_ratio"]["n"] == 1 and observed["transfer_mean"]["mean"] == -25
    assert next(r for r in outcome["pair_stages"]
                if r["treatment"] == "25f" and r["stage"] == "16-30")["equalization_valid_periods"] == 15

    # Omit pre-transfer income once and make the sender gap zero in another
    # round. The level still has 13 valid observations, insufficient for a
    # paired 15-round change; other outcome metrics stay eligible.
    for row in ledger:
        if row["period"] == 16 and row["had_transfer_opportunity"]:
            row.pop("income_before_transfer")
        if row["period"] == 17:
            row["income_before_transfer"] = 225
    outcome = analyze([forced], ledger)
    insured_stage = next(r for r in outcome["pair_stages"] if r["stage"] == "16-30")
    assert insured_stage["period_count"] == 15
    assert insured_stage["equalization_valid_periods"] == 13
    assert insured_stage["equalization_ratio"] == 1
    assert {r["metric"] for r in outcome["phase_changes"]} == set(METRICS) - {"equalization_ratio"}
