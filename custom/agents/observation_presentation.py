"""Shared, deterministic participant-local presentation of an observation.

The original observation is retained losslessly in the turn payload. The
reading guide adds only labels and accounting identities, never a policy,
chosen amount, partner preference, or information from the environment.
"""

from __future__ import annotations

import copy
import json
from collections import defaultdict
from typing import Any


PRESENTATION_VERSION = "readable_v1"


def _cell(value: Any) -> str:
    if value is None:
        return "not provided"
    if isinstance(value, bool):
        return "yes" if value else "no"
    return str(value).replace("|", "\\|").replace("\n", " ")


def _settled_history(observation: dict[str, Any], agent_id: int) -> list[str]:
    partner_id = observation["partner_id"]
    periods: dict[int, dict[int, dict[str, Any]]] = defaultdict(dict)
    for record in observation.get("history", []):
        owner = record.get("agent_id")
        period = record.get("period")
        if (
            type(owner) is not int or owner not in (agent_id, partner_id)
            or record.get("partner_id") != (partner_id if owner == agent_id else agent_id)
            or type(period) is not int or period < 1
            or period > observation["current_period"]
            or (period == observation["current_period"] and observation.get("task") != "finished")
            or owner in periods[period]
        ):
            raise ValueError("History must contain unique settled records for this participant's pair")
        periods[period][owner] = record

    lines = [
        "Completed rounds for your pair",
        "Each row is one settled round. The sender had the transfer opportunity; "
        "the receiver did not. A sender's amount of 0 is a choice; a receiver's "
        "bookkeeping transfer_sent=0 is not a transfer choice.",
        "| Period | Sender | Receiver | Amount | You sent | You received | "
        "Your income before transfer | Partner income before transfer | "
        "Your final income | Partner final income |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for period, records in sorted(periods.items()):
        if set(records) != {agent_id, partner_id}:
            raise ValueError("A settled round must contain both members of the participant's pair")
        own, partner = records[agent_id], records[partner_id]
        if any(type(row.get("had_transfer_opportunity")) is not bool for row in records.values()):
            raise ValueError("History must explicitly identify the transfer opportunity")
        senders = [row for row in records.values() if row["had_transfer_opportunity"]]
        if len(senders) != 1:
            raise ValueError("A settled round must identify exactly one sender")
        sender = senders[0]
        receiver = records[sender["partner_id"]]
        amount = sender["transfer_sent"]
        if (
            type(amount) is not int or amount < 0
            or receiver["transfer_received"] != amount
            or receiver["transfer_sent"] != 0 or sender["transfer_received"] != 0
        ):
            raise ValueError("History transfer direction and amounts must agree")
        values = [
            period, f"Agent {sender['agent_id']}", f"Agent {receiver['agent_id']}",
            amount, own["transfer_sent"], own["transfer_received"],
            own["income_before_transfer"], partner["income_before_transfer"],
            own["final_income"], partner["final_income"],
        ]
        lines.append("| " + " | ".join(_cell(value) for value in values) + " |")
    if not periods:
        lines.append("No rounds have been settled yet.")
    return lines


def _reading_guide(observation: dict[str, Any], agent_id: int) -> str:
    partner_id = observation.get("partner_id")
    if type(partner_id) is not int or partner_id == agent_id:
        raise ValueError("The observation must identify this participant's partner")
    task = observation.get("task")
    lines = [
        "Current observation — reading guide",
        f"You are Agent {agent_id}. Your partner is Agent {partner_id}.",
        f"Current period: {observation['current_period']}. Current task: {task}.",
    ]
    if "announcement" in observation:
        lines += ["Current announcement:", str(observation["announcement"])]

    for name, label in (
        ("had_transfer_opportunity", "You have the transfer opportunity"),
        ("transfer_submitted", "You have already submitted a transfer this round"),
        ("insurance_submitted", "You have already submitted this insurance choice"),
    ):
        if name in observation:
            lines.append(f"{label}: {_cell(observation[name])}.")
    if observation.get("decision_id") is not None:
        lines.append(f"Current decision_id (copy exactly): {observation['decision_id']}")

    partner = observation.get("partner", {})
    if partner and partner.get("agent_id") != partner_id:
        raise ValueError("Partner information must refer to the participant's actual partner")
    if "income_before_transfer" in observation:
        lines += [
            "Current income account",
            f"| Item | You (Agent {agent_id}) | Partner (Agent {partner_id}) |",
            "| --- | --- | --- |",
        ]
        for field, label in (
            ("fixed_income", "Fixed income"),
            ("random_income", "Random income"),
            ("insurance_cost", "Insurance premium deducted"),
            ("insurance_payout", "Insurance payout added"),
            ("income_before_transfer", "Income before transfer, after any listed deductions and payments"),
            ("insured", "Insurance currently active"),
            ("insurance_purchased", "Chose to purchase insurance when offered"),
            ("insurance_assigned", "Was assigned insurance by the computer"),
        ):
            if field in observation or field in partner:
                lines.append(
                    f"| {label} | {_cell(observation.get(field))} | {_cell(partner.get(field))} |"
                )
        if "insurance_cost" in observation or "insurance_payout" in observation:
            lines.append(
                "The income before transfer already includes the displayed premium and payout. "
                "Do not subtract the premium or add the payout a second time."
            )
    if observation.get("insurance_assignment_source") == "computer_random":
        lines.append("The insurance assignment was random; neither participant chose it.")
    if task == "transfer":
        bounds = observation["legal_transfer_range"]
        lines += [
            f"Choose your own integer amount x from {bounds['min']} through {bounds['max']} (inclusive).",
            f"For an amount x that you choose: your final income = {observation['income_before_transfer']} - x; "
            f"your partner's final income = {partner['income_before_transfer']} + x.",
            "These are accounting identities, not a suggested amount. Submit your own choice once.",
        ]
    elif task == "insurance":
        lines.append(
            "The current income draw has not happened. Choose whether to purchase using the "
            "insurance_offer and legal_insurance_choices in the complete observation below."
        )
        offer = observation.get("insurance_offer", {})
        lines.append("Current insurance offer:")
        for field, label in (
            ("cost_per_period", "Premium each period"),
            ("payout_if_low_random_income", "Payout if random income is 0"),
            ("payout_if_high_random_income", "Payout if random income is 200"),
        ):
            if field in offer:
                lines.append(f"- {label}: {offer[field]}.")
        if "announcement" in offer:
            lines.append(str(offer["announcement"]))
        lines.append("Legal purchase choices: " + json.dumps(observation["legal_insurance_choices"]) + ".")
        if "partner_purchase" not in observation:
            lines.append("Your partner's current purchase choice has not been revealed to you.")
    elif task == "waiting":
        lines.append("There is no decision for you to submit in this task. Finish and wait.")
    elif task == "finished":
        lines.append("The experiment has finished. Do not submit another decision.")
    lines += ["", *_settled_history(observation, agent_id)]
    return "\n".join(lines)


def render_readable_turn(payload: dict[str, Any], *, agent_id: int) -> str:
    """Render an existing framework turn without changing any observation value.

    ``agent_id`` comes from the restored participant, not a model argument.
    Normalizing the JSON transport is reversible: only the observation string
    is decoded. The complete history and all offer/status fields remain below
    the guide for inspection and pre-provider auditing.
    """

    if type(agent_id) is not int or type(payload.get("agent_id")) is not int or payload["agent_id"] != agent_id:
        raise ValueError("Turn presentation must match the framework participant identity")
    normalized = copy.deepcopy(payload)
    current = None
    for item in normalized.get("observations", []):
        value = item.get("observation") if isinstance(item, dict) else None
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                continue
        if isinstance(value, dict):
            if type(value.get("agent_id")) is not int or value["agent_id"] != agent_id:
                raise ValueError("Observation presentation must match the framework participant identity")
            item["observation"] = value
            current = value
    if current is None:
        raise ValueError("Readable presentation requires the participant's current observation")
    guide = _reading_guide(current, agent_id)
    return (
        guide
        + "\n\nComplete observation and turn context\n"
        + "The guide above is a factual view of this same observation. "
        + "All supplied fields and every settled history record are retained below.\n"
        + "<risk_sharing_turn>\n"
        # The guide carries the readable layout. Compact JSON retains the
        # full audit source without duplicating indentation on long histories.
        + json.dumps(normalized, ensure_ascii=False, separators=(",", ":"), default=str)
        + "\n</risk_sharing_turn>"
    )
