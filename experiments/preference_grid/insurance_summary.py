"""Overall insurance participation and ordered pair outcomes at the period-16 offer."""

from collections import Counter
from html import escape

from protocol.paper_spec import INSURANCE_DECISION_PERIOD


_FIELDS = {"25v": "purchase", "25f": "assigned_insurance"}
_OUTCOMES = {
    (False, False): "neither",
    (True, False): "first_only",
    (False, True): "second_only",
    (True, True): "both",
}


def _rate(count, total):
    return count / total if total else None


def build_insurance_summary(sessions):
    """One row per planned insurance treatment, counting each qualifying pair once.

    A denominator requires a validated complete session that reached the offer
    and two observed boolean choices/assignments in the relevant ledger.
    """
    summaries = []
    for treatment, field in _FIELDS.items():
        planned = [s for s in sessions if s["treatment"] == treatment]
        if not planned:
            continue
        completed = [s for s in planned if s["status"] == "complete"]
        offered = [s for s in completed if s["completed_periods"] >= INSURANCE_DECISION_PERIOD]
        outcomes = Counter()
        for session in offered:
            choices = (session.get(f"first_{field}"), session.get(f"second_{field}"))
            if all(type(value) is bool for value in choices):
                if treatment == "25f" and choices == (False, False):
                    continue  # This assignment is excluded by the protocol.
                outcomes[_OUTCOMES[choices]] += 1
        pairs = sum(outcomes.values())
        participants = 2 * pairs
        covered = outcomes["first_only"] + outcomes["second_only"] + 2 * outcomes["both"]
        row = {
            "treatment": treatment,
            "measure": "voluntary_purchase" if treatment == "25v" else "forced_assignment",
            "planned_pairs": len(planned),
            "completed_pairs": len(completed),
            "reached_offer_pairs": len(offered),
            "valid_pairs": pairs,
            "incomplete_ledger_pairs": len(offered) - pairs,
            "eligible_people": participants,
            "insured_people": covered,
            "uninsured_people": participants - covered,
            "insured_rate": _rate(covered, participants),
        }
        for outcome in _OUTCOMES.values():
            row[f"{outcome}_pairs"] = outcomes[outcome]
            row[f"{outcome}_rate"] = _rate(outcomes[outcome], pairs)
        summaries.append(row)
    return summaries


def render_insurance_summary(rows):
    """Return a self-contained table with people and pair denominators."""
    if not rows:
        return '<p class="note">本批计划未包含保险实验。</p>'

    def count_and_rate(count, rate):
        return f"{count}（{rate:.1%}）" if rate is not None else "—"

    def row_html(row):
        values = (
            "25V 自主购买" if row["treatment"] == "25v" else "25F 环境分配",
            f"{row['valid_pairs']} / {row['reached_offer_pairs']}",
            f"{row['insured_people']} / {row['eligible_people']}",
            f"{row['insured_rate']:.1%}" if row["insured_rate"] is not None else "—",
            *(count_and_rate(row[f"{group}_pairs"], row[f"{group}_rate"])
              for group in _OUTCOMES.values()),
        )
        return "<tr>" + "".join(f"<td>{escape(value)}</td>" for value in values) + "</tr>"

    headings = ("处理", "双人记录齐全 / 完整且到达第16轮的场次", "获保人数 / 有效人数",
                "总体比例", "双方均无", "仅第一位", "仅第二位", "双方都有")
    head = "".join(f"<th>{text}</th>" for text in headings)
    body = "".join(row_html(row) for row in rows)
    return (
        '<h3>第 16 轮的总体保险记录</h3><div class="scroll"><table id="insurance-ledger">'
        f"<thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>"
        '<p class="note">每场完整实验计一次。总体比例的分母为账本齐全的两人场次中的总人数。'
        '末四列为场数及其占有效场次的比例，第一位和第二位按计划中的配对顺序。'
        '25F 的规则规定每场至少一人获保。短程实验尚未到达第 16 轮时，比例显示为 —。</p>'
    )
