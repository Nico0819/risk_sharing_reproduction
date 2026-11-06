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

from protocol import paper_spec as spec


VERSION = "fixed_preferences_memory_v1"
MEMORY_UPDATE_MAX_ATTEMPTS = 3
CONFIDENCES = {"unknown", "low", "medium", "high"}
MAX_BELIEF_AMOUNT = spec.FIXED_INCOME + spec.HIGH_RANDOM_INCOME
CONTENT_FIELDS = {
    "partner_assessment", "expected_partner_transfer", "perceived_partner_expectation",
    "uncertainties", "evidence_periods",
}
STATE_FIELDS = {
    "schema_version", "owner_agent_id", "partner_id", "scope_id", "preferences_sha256",
    "revision", "last_observation_sha256", "last_reviewed_period", "last_reviewed_task",
    "content", "evidence",
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


def empty_content() -> dict[str, Any]:
    return {
        "partner_assessment": "",
        "expected_partner_transfer": {"amount": None, "confidence": "unknown"},
        "perceived_partner_expectation": {"amount": None, "confidence": "unknown"},
        "uncertainties": "",
        "evidence_periods": [],
    }


def initial_state(
    agent_id: int, partner_id: int, scope_id: str, preferences_sha256: str,
) -> dict[str, Any]:
    return {
        "schema_version": 1, "owner_agent_id": agent_id, "partner_id": partner_id,
        "scope_id": scope_id, "preferences_sha256": preferences_sha256,
        "revision": 0, "last_observation_sha256": None,
        "last_reviewed_period": None, "last_reviewed_task": None,
        "content": empty_content(), "evidence": [],
    }


def _validate_content_shape(content: dict[str, Any]) -> None:
    if not isinstance(content, dict) or set(content) != CONTENT_FIELDS:
        raise ValueError("Private memory must contain only the specified belief and evidence fields")
    for key in ("partner_assessment", "uncertainties"):
        text = content[key]
        if not isinstance(text, str) or len(text) > 600:
            raise ValueError(f"{key} must be text of at most 600 characters")
        if any(tag in text.lower() for tag in ("<private_memory", "</private_memory", "<fixed_preferences", "</fixed_preferences", "<risk_sharing_turn", "</risk_sharing_turn")):
            raise ValueError("Private notes cannot contain message delimiters")
    for key in ("expected_partner_transfer", "perceived_partner_expectation"):
        belief = content[key]
        if not isinstance(belief, dict) or set(belief) != {"amount", "confidence"}:
            raise ValueError("A private forecast must specify amount and confidence only")
        amount, confidence = belief["amount"], belief["confidence"]
        if not isinstance(confidence, str) or confidence not in CONFIDENCES:
            raise ValueError("Invalid private forecast confidence")
        if amount is not None and (type(amount) is not int or not 0 <= amount <= MAX_BELIEF_AMOUNT):
            raise ValueError("A private forecast amount must be null or a legal-range integer")
        if (amount is None) != (confidence == "unknown"):
            raise ValueError("Unknown forecasts must use amount=null and confidence=unknown")
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
    result = []
    for period in content["evidence_periods"]:
        rows = [row for row in observation["history"] if row.get("period") == period]
        if (
            period >= observation["current_period"]
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
    digest = state["last_observation_sha256"]
    if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError("Invalid private memory observation fingerprint")
    period = state["last_reviewed_period"]
    if type(period) is not int or period < 1 or state["last_reviewed_task"] not in {"transfer", "waiting", "insurance"}:
        raise ValueError("Invalid private memory review context")
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
    }
    if evidence_for(state["content"], structural_view) != evidence:
        raise ValueError("Private memory evidence must preserve this pair's settled records")
    if observation is not None:
        if period > observation["current_period"]:
            raise ValueError("Private memory cannot come from a future period")
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
        last_reviewed_period=observation["current_period"], last_reviewed_task=observation["task"],
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
        "content": state["content"], "evidence": state["evidence"],
    })
