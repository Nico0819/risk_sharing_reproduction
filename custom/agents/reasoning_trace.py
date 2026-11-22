"""Preserve provider-returned reasoning without inventing or interpreting it.

This module has no model or framework imports, so both the participant and
the request logger can use the same field extraction at the response boundary.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any


PROVIDER_REASONING_FIELDS = (
    "reasoning_content",
    "reasoning_details",
    "thinking_blocks",
    "reasoning_items",
)


def message_value(message: Any, name: str, default: Any = None) -> Any:
    """Read either a provider message object or its serialized dictionary."""

    if isinstance(message, Mapping):
        return message.get(name, default)
    return getattr(message, name, default)


def provider_reasoning_fields(message: Any) -> dict[str, Any]:
    """Copy the reasoning fields that the provider actually returned.

    An empty string/list remains an empty value. A missing or null field stays
    absent; neither the agent nor its logger may manufacture a reasoning text.
    Structured details may include provider signatures needed for continuation.
    """

    extra = message_value(message, "provider_specific_fields")
    extra = extra if isinstance(extra, Mapping) else {}
    result = {}
    for name in PROVIDER_REASONING_FIELDS:
        value = message_value(message, name)
        if value is None:
            # Some LiteLLM adapters retain the original provider field here.
            # Promote only known reasoning keys, never unrelated metadata.
            value = extra.get(name)
        if value is not None:
            result[name] = copy.deepcopy(value)
    return result


def first_response_message(response: Any) -> Any:
    choices = message_value(response, "choices") or []
    return message_value(choices[0], "message") if choices else None


def validate_reasoning_replay(messages: list[dict[str, Any]]) -> None:
    """Refuse to continue a reasoning tool conversation with lost fields.

    Call this only at a real participant provider boundary with tools enabled.
    In particular, do not let an SDK silently invent a blank reasoning field
    for an assistant response that was never recorded. Explicitly returned
    empty fields remain distinguishable from absent fields in the audit log.
    """

    for index, message in enumerate(messages):
        if message_value(message, "role") != "assistant":
            continue
        if not provider_reasoning_fields(message):
            raise ValueError(
                "Cannot continue a participant reasoning tool conversation: "
                f"assistant message {index} has no returned reasoning field. "
                "A missing reasoning trace must not be replaced by an SDK placeholder."
            )
