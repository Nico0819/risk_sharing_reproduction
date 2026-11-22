"""Bounded, identity-bound private state; no economic action selection."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any

VERSION = "fixed_preferences_memory_v1"
SCHEMA_VERSION = 2
MEMORY_UPDATE_MAX_ATTEMPTS = 3
MAX_NOTE_CHARACTERS = 2400
CONTENT_FIELDS = {"note", "evidence_periods"}
STATE_FIELDS = {
    "schema_version", "owner_agent_id", "partner_id", "scope_id", "preferences_sha256",
    "revision", "last_observation_sha256", "last_reviewed_period", "last_reviewed_task",
    "last_feedback_sha256", "last_reviewed_settled_period", "content", "evidence",
}
_HISTORY_FIELDS = {
    "period", "agent_id", "partner_id", "fixed_income", "random_income",
    "had_transfer_opportunity", "transfer_sent", "transfer_received",
    "income_before_transfer", "final_income", "insurance_purchased",
    "insurance_assigned", "insured", "insurance_cost", "insurance_payout",
}
_INSURANCE_STATUS_FIELDS = {"insurance_purchased", "insurance_assigned", "insured"}
_INSURANCE_TERM_FIELDS = {
    "cost_per_period", "payout_if_low_random_income", "payout_if_high_random_income",
    "announcement",
}


def _unique_object(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError(f"Duplicate private memory field: {key}")
        result[key] = value
    return result


def parse_memory_output(text: str) -> dict[str, Any]:
    """Read one JSON object, optionally inside one complete Markdown fence.

    Only the outer formatting is removed. Do not search prose for a usable
    object, repair JSON, or fill missing fields: the participant's content
    must still pass the same parsing and state validation as a plain reply.
    Persisted state files continue to require plain JSON via ``load_state``.
    """
    if not isinstance(text, str) or not text.strip() or len(text) > 10000:
        raise ValueError("Private memory requires a bounded, nonempty JSON object")
    payload = text.strip()
    if payload.startswith("```"):
        fenced = re.fullmatch(
            r"```(?:json)?[ \t]*\r?\n(?P<body>.*?)\r?\n[ \t]*```",
            payload,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if fenced is None:
            raise ValueError("Private memory must be one JSON object or one complete JSON code block")
        payload = fenced.group("body")
    result = json.loads(payload, object_pairs_hook=_unique_object)
    if not isinstance(result, dict):
        raise ValueError("Private memory must be one JSON object")
    return result


def _digest(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def observation_sha256(observation: dict[str, Any]) -> str:
    return _digest(observation)


def memory_sha256(state: dict[str, Any]) -> str:
    return _digest(state)


def _settled_cutoff(observation: dict[str, Any]) -> int:
    period = observation.get("current_period")
    if type(period) is not int or period < 1:
        raise ValueError("Memory observation requires a positive integer period")
    if observation.get("task") == "closing":
        cutoff = observation.get("last_settled_period")
        if type(cutoff) is not int or not 0 <= cutoff <= period:
            raise ValueError("Closing memory requires an explicit last settled period")
        return cutoff
    return period - 1


def feedback_payload(observation: dict[str, Any]) -> dict[str, Any]:
    """Fingerprint only settled facts and already disclosed insurance information.

    Current income, task changes and decision tokens are decision context, not
    newly learned outcomes. Only this allowlisted public projection is used;
    caller identity must still come from the framework-bound observation.
    """
    cutoff = _settled_cutoff(observation)
    owner, partner = observation["agent_id"], observation["partner_id"]
    if type(owner) is not int or type(partner) is not int or owner < 1 or partner < 1 or owner == partner:
        raise ValueError("Feedback requires two distinct participant identities")
    history = observation.get("history")
    if not isinstance(history, list):
        raise ValueError("Feedback requires the participant's settled history")
    by_period: dict[int, list[dict[str, Any]]] = {}
    for row in history:
        if not isinstance(row, dict):
            raise ValueError("Feedback history must contain records")
        period = row.get("period")
        if (
            type(period) is not int or not 1 <= period <= cutoff
            or {row.get("agent_id"), row.get("partner_id")} != {owner, partner}
        ):
            raise ValueError("Feedback history must belong to this pair's settled periods")
        by_period.setdefault(period, []).append(row)
    for rows in by_period.values():
        if len(rows) != 2 or {row["agent_id"] for row in rows} != {owner, partner}:
            raise ValueError("Feedback history requires both members once per settled period")
    records = [
        {key: copy.deepcopy(value) for key, value in row.items() if key in _HISTORY_FIELDS}
        for period in sorted(by_period)
        for row in sorted(by_period[period], key=lambda row: row["agent_id"])
    ]
    insurance = {
        key: copy.deepcopy(observation[key])
        for key in _INSURANCE_STATUS_FIELDS | {
            "own_purchase", "partner_purchase", "insurance_assignment_source", "announcement",
        }
        if key in observation and observation[key] is not None
    }
    for key in ("insurance_offer", "insurance_terms"):
        if key in observation:
            terms = observation[key]
            if not isinstance(terms, dict):
                raise ValueError("Public insurance terms must be an object")
            insurance[key] = {k: copy.deepcopy(v) for k, v in terms.items() if k in _INSURANCE_TERM_FIELDS}
    partner_view = observation.get("partner", {})
    if not isinstance(partner_view, dict):
        raise ValueError("Public partner information must be an object")
    partner_status = {
        key: copy.deepcopy(partner_view[key]) for key in _INSURANCE_STATUS_FIELDS
        if key in partner_view and partner_view[key] is not None
    }
    if partner_status:
        insurance["partner"] = partner_status
    return {"agent_id": owner, "partner_id": partner, "history": records, "insurance": insurance}


def feedback_sha256(observation: dict[str, Any]) -> str:
    return _digest(feedback_payload(observation))


def has_new_feedback(state: dict[str, Any], observation: dict[str, Any]) -> bool:
    if observation.get("agent_id") != state["owner_agent_id"] or observation.get("partner_id") != state["partner_id"]:
        raise ValueError("Memory feedback must use the owner's observation")
    payload = feedback_payload(observation)
    if not payload["history"] and not payload["insurance"]:
        return False
    return state["last_feedback_sha256"] != _digest(payload)


def empty_content() -> dict[str, Any]:
    return {"note": "", "evidence_periods": []}


def initial_state(
    agent_id: int, partner_id: int, scope_id: str, preferences_sha256: str,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION, "owner_agent_id": agent_id, "partner_id": partner_id,
        "scope_id": scope_id, "preferences_sha256": preferences_sha256,
        "revision": 0, "last_observation_sha256": None, "last_feedback_sha256": None,
        "last_reviewed_period": None, "last_reviewed_task": None,
        "last_reviewed_settled_period": None,
        "content": empty_content(), "evidence": [],
    }


def _validate_content_shape(content: dict[str, Any]) -> None:
    if not isinstance(content, dict) or set(content) != CONTENT_FIELDS:
        raise ValueError("Private memory must contain only note and evidence_periods")
    text = content["note"]
    if not isinstance(text, str) or len(text) > MAX_NOTE_CHARACTERS:
        raise ValueError(f"note must be text of at most {MAX_NOTE_CHARACTERS} characters")
    if any(tag in text.lower() for tag in ("<private_memory", "</private_memory", "<fixed_preferences", "</fixed_preferences", "<risk_sharing_turn", "</risk_sharing_turn")):
        raise ValueError("Private notes cannot contain message delimiters")
    periods = content["evidence_periods"]
    if (
        not isinstance(periods, list) or len(periods) > 6
        or any(type(period) is not int or period < 1 for period in periods)
        or len(set(periods)) != len(periods)
    ):
        raise ValueError("Evidence must be at most six distinct settled period numbers")


def evidence_for(content: dict[str, Any], observation: dict[str, Any]) -> list[dict[str, Any]]:
    _validate_content_shape(content)
    members = {observation["agent_id"], observation["partner_id"]}
    cutoff = _settled_cutoff(observation)
    result = []
    for period in content["evidence_periods"]:
        rows = [row for row in observation["history"] if row.get("period") == period]
        if (
            period > cutoff
            or len(rows) != 2 or {row.get("agent_id") for row in rows} != members
            or any({row.get("agent_id"), row.get("partner_id")} != members for row in rows)
        ):
            raise ValueError("Private memory evidence must reference this pair's settled history")
        result.extend(copy.deepcopy(sorted(rows, key=lambda row: row["agent_id"])))
    return result


def validate_content(content: dict[str, Any], observation: dict[str, Any]) -> dict[str, Any]:
    evidence_for(content, observation)
    return copy.deepcopy(content)


def validate_state(
    state: dict[str, Any], *, agent_id: int, partner_id: int, scope_id: str,
    preferences_sha256: str, observation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if isinstance(state, dict) and type(state.get("schema_version")) is int and state["schema_version"] == 1:
        raise ValueError("Private memory schema 1 is a historical format; start a new run with schema 2. Historical files are not migrated.")
    if not isinstance(state, dict) or set(state) != STATE_FIELDS:
        raise ValueError("Invalid private memory state schema")
    expected = initial_state(agent_id, partner_id, scope_id, preferences_sha256)
    for key in ("schema_version", "owner_agent_id", "partner_id", "scope_id", "preferences_sha256"):
        if type(state[key]) is not type(expected[key]) or state[key] != expected[key]:
            raise ValueError(f"Private memory binding mismatch: {key}")
    if observation is not None and (
        observation.get("agent_id") != agent_id or observation.get("partner_id") != partner_id
    ):
        raise ValueError("Private memory observation belongs to a different participant or pair")
    revision = state["revision"]
    if type(revision) is not int or revision < 0:
        raise ValueError("Invalid private memory revision")
    _validate_content_shape(state["content"])
    if revision == 0:
        if state != expected:
            raise ValueError("Initial private memory must contain no invented beliefs or evidence")
        return copy.deepcopy(state)
    for key in ("last_observation_sha256", "last_feedback_sha256"):
        digest = state[key]
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("Invalid private memory observation or feedback fingerprint")
    period = state["last_reviewed_period"]
    if type(period) is not int or period < 1 or state["last_reviewed_task"] not in {"transfer", "waiting", "insurance", "closing"}:
        raise ValueError("Invalid private memory review context")
    settled_period = state["last_reviewed_settled_period"]
    if type(settled_period) is not int or not 0 <= settled_period <= period:
        raise ValueError("Invalid private memory settled period")
    if state["last_reviewed_task"] != "closing" and settled_period >= period:
        raise ValueError("Private memory cannot use unsettled current-period feedback")
    if any(value > settled_period for value in state["content"]["evidence_periods"]):
        raise ValueError("Private memory evidence exceeds its reviewed settled period")
    evidence = state["evidence"]
    if (
        not isinstance(evidence, list)
        or len(evidence) != 2 * len(state["content"]["evidence_periods"])
        or any(not isinstance(record, dict) for record in evidence)
    ):
        raise ValueError("Private memory evidence does not match its referenced periods")
    structural_view = {
        "agent_id": agent_id, "partner_id": partner_id,
        "current_period": period, "history": evidence,
        "task": state["last_reviewed_task"], "last_settled_period": settled_period,
    }
    if evidence_for(state["content"], structural_view) != evidence:
        raise ValueError("Private memory evidence must preserve this pair's settled records")
    if observation is not None:
        if period > observation["current_period"]:
            raise ValueError("Private memory cannot come from a future period")
        if settled_period > max((row["period"] for row in observation["history"]), default=0):
            raise ValueError("Private memory cannot contain future settled feedback")
        if evidence_for(state["content"], observation) != evidence:
            raise ValueError("Private memory facts disagree with the environment history")
    return copy.deepcopy(state)


def load_state(
    path: Path, *, agent_id: int, partner_id: int, scope_id: str,
    preferences_sha256: str, observation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError("The participant's private memory file is missing or is a link")
    if path.stat().st_size > 64000:
        raise ValueError("Private memory file exceeds the bounded state size")
    state = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_object)
    return validate_state(state, agent_id=agent_id, partner_id=partner_id,
                          scope_id=scope_id, preferences_sha256=preferences_sha256,
                          observation=observation)


def write_state(path: Path, state: dict[str, Any]) -> None:
    """Atomically replace the local note; no model can select a file path."""
    if path.is_symlink():
        raise ValueError("Private memory cannot be written through a link")
    data = json.dumps(state, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=".private_memory-", delete=False) as stream:
            temp_path = Path(stream.name)
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_path, path)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def updated_state(previous: dict[str, Any], content: dict[str, Any], observation: dict[str, Any]) -> dict[str, Any]:
    if observation.get("agent_id") != previous["owner_agent_id"] or observation.get("partner_id") != previous["partner_id"]:
        raise ValueError("Memory update must use the owner's observation")
    result = copy.deepcopy(previous)
    result.update(
        revision=previous["revision"] + 1,
        last_observation_sha256=observation_sha256(observation),
        last_feedback_sha256=feedback_sha256(observation),
        last_reviewed_period=observation["current_period"], last_reviewed_task=observation["task"],
        last_reviewed_settled_period=max((row["period"] for row in observation["history"]), default=0),
        content=validate_content(content, observation), evidence=evidence_for(content, observation),
    )
    return validate_state(
        result, agent_id=previous["owner_agent_id"], partner_id=previous["partner_id"],
        scope_id=previous["scope_id"], preferences_sha256=previous["preferences_sha256"],
        observation=observation,
    )


def memory_payload(state: dict[str, Any]) -> dict[str, Any]:
    """Only the owner's note and its visible evidence reach the model."""
    return copy.deepcopy({
        "agent_id": state["owner_agent_id"], "partner_id": state["partner_id"],
        "revision": state["revision"], "last_reviewed_period": state["last_reviewed_period"],
        "last_reviewed_task": state["last_reviewed_task"],
        "last_reviewed_settled_period": state["last_reviewed_settled_period"],
        "content": state["content"], "evidence": state["evidence"],
    })
