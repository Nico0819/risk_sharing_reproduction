"""Prepare recorded experiment results for researcher-facing visual reports.

This module reads neither environment state nor models. Missing observations or
settlements remain missing; an absent purchase is never interpreted as refusal.
Computer insurance assignments are kept separate from participant purchases.
"""

from __future__ import annotations

from collections import defaultdict
from math import isfinite
from typing import Any, Iterable, Mapping

from protocol import paper_spec as spec


def _sort_id(value: Any) -> tuple[int, Any]:
    if type(value) is int:
        return (0, value)
    return (1, str(value))


def _number(value: Any) -> int | float | None:
    # JSON booleans and numeric strings are not recorded economic amounts.
    if type(value) is int:
        return value
    if type(value) is float and isfinite(value):
        return value
    return None


def _period(row: Mapping[str, Any]) -> int | None:
    value = row.get("period")
    return value if type(value) is int and value > 0 else None


def _assignment_configuration(manifest: Mapping[str, Any]) -> dict[str, Any] | None:
    """Expose only the recorded assignment settings needed by the report."""

    configuration = manifest.get("insurance_assignment_configuration")
    if not isinstance(configuration, Mapping):
        return None
    probabilities = configuration.get("probabilities")
    probabilities = probabilities if isinstance(probabilities, Mapping) else {}
    result = {
        "probabilities": {
            name: _number(probabilities.get(name))
            for name in ("both_insured", "first_only", "second_only")
        },
        "seed": configuration.get("seed") if type(configuration.get("seed")) is int else None,
        "paper_probability_resolved": (
            configuration.get("paper_probability_resolved")
            if type(configuration.get("paper_probability_resolved")) is bool else None
        ),
    }
    for name in ("rng_source", "probability_source"):
        result[name] = configuration.get(name) if isinstance(configuration.get(name), str) else None
    return result


def summarize_pair(
    pair: Mapping[str, Any],
    periods: Iterable[int],
    start: int | None = None,
    end: int | None = None,
) -> dict[str, int | float | None]:
    """Summarize the recorded pair series within an inclusive period range.

    Transfer and income statistics use their own valid-period denominators.
    The income series contains only periods with both members' final incomes;
    their mean and absolute gap always refer to the same pair-period. Missing
    periods do not count as zero transfers or as zero incomes. An empty total
    is also ``None`` so it cannot be mistaken for an observed zero transfer.
    """

    transfers, income_means, income_gaps = [], [], []
    transfer_series = pair.get("transfer", ())
    income_series = pair.get("income_mean", ())
    gap_series = pair.get("income_gap", ())
    for index, period in enumerate(periods):
        if (start is not None and period < start) or (end is not None and period > end):
            continue
        transfer = _number(transfer_series[index]) if index < len(transfer_series) else None
        if transfer is not None:
            transfers.append(transfer)
        income = _number(income_series[index]) if index < len(income_series) else None
        gap = _number(gap_series[index]) if index < len(gap_series) else None
        if income is not None and gap is not None:
            income_means.append(income)
            income_gaps.append(gap)

    return {
        "transfer_mean": sum(transfers) / len(transfers) if transfers else None,
        "transfer_valid_periods": len(transfers),
        "transfer_total": sum(transfers) if transfers else None,
        "transfer_positive_rate": sum(value > 0 for value in transfers) / len(transfers) if transfers else None,
        "income_mean": sum(income_means) / len(income_means) if income_means else None,
        "income_gap_mean": sum(income_gaps) / len(income_gaps) if income_gaps else None,
        "income_valid_periods": len(income_means),
    }


def build_report_data(
    manifest: Mapping[str, Any],
    records: Iterable[Mapping[str, Any]],
    decisions: Iterable[Mapping[str, Any]] = (),
    *,
    assignments: Iterable[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Return JSON-ready series grouped by the pair IDs in accepted records.

    ``decisions`` must be the accepted insurance-decision records, not attempted
    tool actions. Only strict booleans count as submitted insurance choices.
    For 25F, ``assignments`` is the original environment assignment ledger;
    current settlement coverage must not overwrite that original assignment.
    Incomplete pair-periods have no aggregate transfer or pair income metric.
    Known member incomes remain visible individually, and missing member
    incomes remain ``None``. Pair comparisons use complete same-period incomes.
    """

    records, decisions, assignments = list(records), list(decisions), list(assignments)
    treatment = manifest.get("treatment", "Baseline")
    treatment = str(getattr(treatment, "value", treatment))
    voluntary = treatment == spec.Treatment.VOLUNTARY_INSURANCE_25.value
    forced = treatment == spec.Treatment.FORCED_INSURANCE_25.value
    insurance = voluntary or forced
    insurance_mode = "voluntary" if voluntary else "forced" if forced else "none"
    ledger = decisions if voluntary else assignments if forced else []
    members: dict[Any, set[Any]] = defaultdict(set)
    for row in [*records, *ledger]:
        pair_id = row.get("pair_id")
        if pair_id is not None:
            members[pair_id].update(
                row[field] for field in ("agent_id", "partner_id")
                if row.get(field) is not None
            )

    recorded_periods = {p for row in records if (p := _period(row)) is not None}
    periods = list(range(1, max(recorded_periods, default=0) + 1))
    settlements: dict[tuple[Any, int, Any], Mapping[str, Any]] = {}
    for row in records:
        period = _period(row)
        if period is None or row.get("pair_id") is None or row.get("agent_id") is None:
            continue
        key = (row["pair_id"], period, row["agent_id"])
        if key in settlements:
            raise ValueError(f"Duplicate settlement for pair {key[0]}, period {period}, Agent {key[2]}")
        settlements[key] = row

    choices: dict[tuple[Any, Any], bool] = {}
    invalid_choice_pairs = set()
    invalid_choice_members = set()
    if voluntary:
        for decision in decisions:
            pair_id, agent_id = decision.get("pair_id"), decision.get("agent_id")
            if pair_id is None or agent_id is None:
                continue
            key, purchase = (pair_id, agent_id), decision.get("purchase")
            if type(purchase) is not bool or (key in choices and choices[key] is not purchase):
                invalid_choice_pairs.add(pair_id)
                invalid_choice_members.add(key)
                choices.pop(key, None)
            elif key not in invalid_choice_members:
                choices[key] = purchase

    original_assignments: dict[tuple[Any, Any], bool] = {}
    invalid_assignment_pairs = set()
    invalid_assignment_members = set()
    seen_assignments = set()
    if forced:
        for assignment in assignments:
            pair_id, agent_id = assignment.get("pair_id"), assignment.get("agent_id")
            if pair_id is None or agent_id is None:
                if pair_id is not None:
                    invalid_assignment_pairs.add(pair_id)
                continue
            key = (pair_id, agent_id)
            insured = assignment.get("insured")
            if (
                key in seen_assignments
                or type(insured) is not bool
                or assignment.get("assignment_source") != "environment_random"
                or _period(assignment) != spec.INSURANCE_ACTIVE_START_PERIOD
                or assignment.get("treatment") != spec.Treatment.FORCED_INSURANCE_25.value
                or assignment.get("partner_id") is None
                or assignment.get("partner_id") == agent_id
            ):
                invalid_assignment_pairs.add(pair_id)
                invalid_assignment_members.add(key)
                original_assignments.pop(key, None)
            elif key not in invalid_assignment_members:
                original_assignments[key] = insured
            seen_assignments.add(key)
        for pair_id, agent_ids in members.items():
            keys = [(pair_id, agent_id) for agent_id in agent_ids]
            if len(keys) == spec.PAIR_SIZE and all(original_assignments.get(key) is False for key in keys):
                # The 25F design excludes assigning neither member insurance.
                invalid_assignment_pairs.add(pair_id)
                invalid_assignment_members.update(keys)
                for key in keys:
                    original_assignments.pop(key, None)

    premiums: dict[Any, int | float] = defaultdict(int)
    payouts: dict[Any, int | float] = defaultdict(int)
    if insurance:
        for (pair_id, period, _), row in settlements.items():
            if spec.INSURANCE_ACTIVE_START_PERIOD <= period <= spec.INSURANCE_ACTIVE_END_PERIOD:
                premiums[pair_id] += _number(row.get("insurance_cost")) or 0
                payouts[pair_id] += _number(row.get("insurance_payout")) or 0

    reached_offer = bool(ledger) or max(recorded_periods, default=0) >= spec.INSURANCE_DECISION_PERIOD
    pairs = []
    for pair_id in sorted(members, key=_sort_id):
        agent_ids = sorted(members[pair_id], key=_sort_id)
        submitted = sum((pair_id, agent_id) in choices for agent_id in agent_ids)
        purchased = sum(choices.get((pair_id, agent_id)) is True for agent_id in agent_ids)
        assigned = sum((pair_id, agent_id) in original_assignments for agent_id in agent_ids)
        covered = sum(original_assignments.get((pair_id, agent_id)) is True for agent_id in agent_ids)
        if not insurance:
            status = "无保险"
        elif forced:
            if pair_id in invalid_assignment_pairs or len(agent_ids) != spec.PAIR_SIZE:
                status = "分配记录需核对"
            elif assigned < spec.PAIR_SIZE:
                status = "分配记录未齐" if reached_offer else "尚未分配保险"
            else:
                status = "双方获保" if covered == spec.PAIR_SIZE else "单方获保"
        elif pair_id in invalid_choice_pairs or len(agent_ids) != spec.PAIR_SIZE:
            status = "分组或选择记录需核对"
        elif submitted < spec.PAIR_SIZE:
            status = "决定未收齐" if reached_offer else "尚未到投保阶段"
        else:
            status = {0: "双方不购买", 1: "单方购买", spec.PAIR_SIZE: "双方购买"}[purchased]

        member_series = [{
            "id": agent_id,
            "purchase": choices.get((pair_id, agent_id)),
            "assigned": original_assignments.get((pair_id, agent_id)),
            "assignment_status": (
                "invalid" if (pair_id, agent_id) in invalid_assignment_members else
                "valid" if (pair_id, agent_id) in original_assignments else "missing"
            ) if forced else None,
            "income": [
                _number(settlements.get((pair_id, period, agent_id), {}).get("final_income"))
                for period in periods
            ],
        } for agent_id in agent_ids]
        transfers, income_means, income_gaps, period_count = [], [], [], 0
        for period in periods:
            rows = [settlements.get((pair_id, period, agent_id)) for agent_id in agent_ids]
            complete = len(rows) == spec.PAIR_SIZE and all(row is not None for row in rows)
            if complete:
                period_count += 1
            amounts = [_number(row.get("transfer_sent")) if row is not None else None for row in rows]
            transfers.append(sum(amounts) if complete and all(a is not None for a in amounts) else None)
            incomes = [_number(row.get("final_income")) if row is not None else None for row in rows]
            if complete and all(income is not None for income in incomes):
                income_means.append(sum(incomes) / spec.PAIR_SIZE)
                income_gaps.append(abs(incomes[0] - incomes[1]))
            else:
                income_means.append(None)
                income_gaps.append(None)
        pair = {
            "id": pair_id,
            "label": f"Pair {pair_id}",
            "members": member_series,
            "purchased": purchased,
            "submitted": submitted,
            "assigned": assigned,
            "covered": covered,
            "status": status,
            "transfer": transfers,
            "transfer_positive": [int(amount > 0) if amount is not None else None for amount in transfers],
            "income_mean": income_means,
            "income_gap": income_gaps,
            "premium": premiums[pair_id],
            "payout": payouts[pair_id],
            "period_count": period_count,
        }
        pair["summary_all"] = summarize_pair(pair, periods)
        pairs.append(pair)

    actual_members = {agent_id for pair_members in members.values() for agent_id in pair_members}
    return {
        "treatment": treatment,
        "backend": str(manifest.get("backend", "unknown")),
        "success": manifest.get("success") is True,
        "periods": periods,
        "completed_periods": len(recorded_periods),
        "agents": len(actual_members) if actual_members else manifest.get("agents", 0),
        "model_requests": manifest.get("actual_model_requests", 0),
        "insurance": insurance,
        "insurance_mode": insurance_mode,
        "reached_insurance": reached_offer if insurance else False,
        "insurance_assignment_configuration": _assignment_configuration(manifest) if forced else None,
        "insurance_start": spec.INSURANCE_ACTIVE_START_PERIOD,
        "insurance_end": spec.INSURANCE_ACTIVE_END_PERIOD,
        "removal": spec.INSURANCE_REMOVAL_REVEALED_PERIOD,
        "payment_period": manifest.get("payment_period"),
        "pairs": pairs,
    }
