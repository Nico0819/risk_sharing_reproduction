"""Observed ending rounds and risk sets for validated full sessions."""

from collections import Counter

from protocol import paper_spec


def build_termination_distribution(sessions):
    first = paper_spec.REAL_TERMINATION_FIRST_POSSIBLE_PERIOD
    last = paper_spec.REAL_TERMINATION_FORCED_PERIOD
    ends = Counter()
    for session in sessions:
        period = session.get("completed_periods")
        if (session.get("status") == "complete"
                and session.get("run_mode") == "full_session"
                and session.get("natural_termination") is True
                and session.get("stop_reason") == "natural_termination"
                and type(period) is int and first <= period <= last):
            ends[(session["treatment"], period)] += 1

    rows = []
    for treatment in paper_spec.Treatment:
        key = treatment.value.lower()
        total = sum(ends[key, period] for period in range(first, last + 1))
        at_risk = total
        for period in range(first, last + 1):
            ended = ends[key, period]
            continued = at_risk - ended
            protocol_rate = (paper_spec.REAL_ENDING_PROBABILITY_PERIOD_50 if period == last
                             else paper_spec.REAL_ENDING_PROBABILITY_PERIOD_48_49)
            rows.append({
                "treatment": key, "period": period, "at_risk": at_risk,
                "ended": ended, "continued": continued, "n_natural": total,
                "share_all": ended / total if total else None,
                "conditional_rate": ended / at_risk if at_risk else None,
                "protocol_rate": protocol_rate,
            })
            at_risk = continued
    return rows


def overall_ending_note(rows):
    total = sum(row["ended"] for row in rows)
    if not total:
        return "本批暂无自然结束的完整场次。"
    periods = sorted({row["period"] for row in rows})
    details = []
    for period in periods:
        at_risk = sum(row["at_risk"] for row in rows if row["period"] == period)
        ended = sum(row["ended"] for row in rows if row["period"] == period)
        if at_risk:
            details.append(f"第 {period} 轮有 {at_risk:,} 场进入，其中 {ended:,} 场结束（{ended / at_risk:.2%}）")
    return f"合计 {total:,} 场完整实验。" + "。".join(details) + "。"
