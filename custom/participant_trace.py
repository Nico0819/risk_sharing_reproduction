"""Record provider-exposed participant reasoning without claiming hidden access.

Only response fields useful for research are selected. HTTP metadata, headers,
credentials and arbitrary provider response extras are never serialized here.
Each record captures the final result of one framework client call; retries
inside the installed framework or SDK are not individually observable here.
"""

from __future__ import annotations

import copy
import re
from collections import Counter
from collections.abc import Mapping
from typing import Any, Callable, Iterable

from custom.agents.reasoning_trace import message_value, provider_reasoning_fields


TRACE_FILE = "participant_reasoning.jsonl"
TRACE_SCOPE = "provider_exposed_reasoning_not_complete_internal_computation"
CAPTURE_BOUNDARY = "framework_client_call_result"


def redact_credentials(value: Any) -> Any:
    """Catch quoted JSON credential assignments as well as plain diagnostics."""
    credential = r"(?:authorization|(?:[a-z0-9_]*_)?api[_-]?key|secret|password)"
    if isinstance(value, Mapping):
        return {
            key: "<redacted>" if re.fullmatch(credential, str(key), re.I) else redact_credentials(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [redact_credentials(item) for item in value]
    if isinstance(value, str):
        return re.sub(
            rf'''(?i)(?<![a-z0-9_])(["']?{credential}["']?\s*[:=]\s*["']?)(?:bearer\s+)?[^"',;\s}}\]]+''',
            lambda match: match[1] + "<redacted>", value,
        )
    return value


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    dumper = getattr(value, "model_dump", None)
    if callable(dumper):
        return _plain(dumper())
    if hasattr(value, "__dict__"):
        return _plain(vars(value))
    return str(value)


def _has_data(value: Any) -> bool:
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, Mapping):
        return any(_has_data(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_has_data(item) for item in value)
    return value is not None


def _readable_reasoning(fields: Mapping[str, Any]) -> bool:
    if isinstance(fields.get("reasoning_content"), str) and fields["reasoning_content"].strip():
        return True

    def contains_text(value: Any) -> bool:
        if isinstance(value, Mapping):
            return any(
                isinstance(value.get(key), str) and bool(value[key].strip())
                for key in ("text", "thinking", "reasoning", "summary")
            ) or any(contains_text(item) for item in value.values() if isinstance(item, (Mapping, list)))
        if isinstance(value, list):
            return any(contains_text(item) for item in value)
        return False

    return any(contains_text(value) for value in fields.values())


def _tools(message: Any) -> list[dict[str, Any]]:
    result = []
    for call in message_value(message, "tool_calls") or []:
        function = message_value(call, "function")
        result.append({
            "id": message_value(call, "id"),
            "type": message_value(call, "type"),
            "function": {
                "name": message_value(function, "name"),
                "arguments": _plain(message_value(function, "arguments")),
            },
        })
    return result


def _usage(response: Any) -> dict[str, Any]:
    usage = message_value(response, "usage")
    selected = {}
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        value = message_value(usage, key)
        if value is not None:
            selected[key] = value
    for key, fields in (
        ("completion_tokens_details", ("reasoning_tokens", "accepted_prediction_tokens", "rejected_prediction_tokens")),
        ("prompt_tokens_details", ("cached_tokens",)),
    ):
        details = message_value(usage, key)
        values = {field: message_value(details, field) for field in fields if message_value(details, field) is not None}
        if values:
            selected[key] = values
    return selected


class ParticipantTraceRecorder:
    """One participant's response audit, including failed and empty responses."""

    def __init__(self, logger: Any = None, *, backend: str, sanitize: Callable[[Any], Any]):
        self.logger = logger
        self.backend = backend
        self.sanitize = sanitize
        self.counts: Counter[str] = Counter()

    def record(
        self, *, response: Any = None, error_type: str | None = None,
        agent_id: int | None, pair_id: int | None, phase: str,
        request_index: int, model: str | None, context: Mapping[str, Any] | None,
        request_kwargs: Mapping[str, Any], duration_seconds: float,
        memory_update_attempt: int | None = None,
        memory_update_kind: str | None = None,
    ) -> dict[str, Any]:
        choices = message_value(response, "choices") or []
        choice = choices[0] if choices else None
        message = message_value(choice, "message")
        reasoning = _plain(provider_reasoning_fields(message))
        finish_reason = message_value(choice, "finish_reason")
        truncated = finish_reason in {"length", "max_tokens", "max_output_tokens"}
        response_complete = (
            False if error_type or truncated or finish_reason in {"content_filter", "error"}
            else True if finish_reason in {"stop", "tool_calls", "function_call"}
            else None
        )
        status = (
            "request_failed" if error_type else "missing" if not reasoning
            else "available" if _has_data(reasoning) else "empty"
        )
        context = context or {}
        row: dict[str, Any] = {
            "schema_version": 1,
            "agent_id": agent_id, "pair_id": pair_id,
            "period": context.get("period"), "task": context.get("task"),
            "decision_id": context.get("decision_id"),
            "phase": phase, "role": "participant", "request_index": request_index,
            "response_source": "provider" if self.backend == "llm" else "scripted",
            "trace_scope": TRACE_SCOPE,
            "capture_boundary": CAPTURE_BOUNDARY,
            "provider_retry_attempts_captured": False,
            "model": message_value(response, "model") or model,
            "response_id": message_value(response, "id"),
            "ok": error_type is None and bool(choices),
            "error_type": error_type,
            "duration_seconds": round(duration_seconds, 6),
            "reasoning_status": status,
            "readable_reasoning_present": _readable_reasoning(reasoning),
            "reasoning_fields_returned": sorted(reasoning),
            **reasoning,
            "content": _plain(message_value(message, "content")),
            "tool_calls": _tools(message),
            "finish_reason": finish_reason,
            "response_complete": response_complete,
            "response_truncated": truncated,
            "usage": _usage(response),
            "max_output_tokens": request_kwargs.get("max_output_tokens"),
            "request_tools": _plain(request_kwargs.get("tools") or []),
            "request_tool_choice": _plain(request_kwargs.get("tool_choice")),
            "input_context_file": "participant_contexts.jsonl",
            "input_context_request_index": request_index,
            "private_memory_revision": (context.get("private_memory_snapshot") or {}).get("revision"),
            "private_memory_sha256": context.get("private_memory_sha256"),
            "preferences_sha256": context.get("preferences_sha256"),
        }
        if memory_update_attempt is not None:
            row["memory_update_attempt"] = memory_update_attempt
        if memory_update_kind is not None:
            row["memory_update_kind"] = memory_update_kind
        sanitized = self.sanitize(redact_credentials(copy.deepcopy(row)))
        sanitized["redaction_applied"] = sanitized != row
        sanitized["provider_trace_complete"] = (
            self.backend == "llm" and response_complete is True
            and sanitized["readable_reasoning_present"] and not sanitized["redaction_applied"]
        )
        self.counts["responses"] += 1
        self.counts[status] += 1
        self.counts["truncated"] += int(truncated)
        self.counts["redacted"] += int(sanitized["redaction_applied"])
        self.counts["readable"] += int(sanitized["readable_reasoning_present"])
        self.counts["complete"] += int(sanitized["provider_trace_complete"])
        if self.logger is not None:
            self.logger.write("participant_model_response", **sanitized)
        return sanitized


def summarize_reasoning_traces(recorders: Iterable[ParticipantTraceRecorder], *, backend: str) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    for recorder in recorders:
        counts.update(recorder.counts)
    return {
        "participant_reasoning_file": TRACE_FILE,
        "participant_reasoning_scope": TRACE_SCOPE,
        "capture_boundary": CAPTURE_BOUNDARY,
        "provider_retry_attempts_captured": False,
        "participant_thinking": "enabled" if backend == "llm" else "not_applicable_scripted",
        "router_thinking": "disabled" if backend == "llm" else "not_applicable_scripted",
        "participant_reasoning_responses": counts["responses"],
        "participant_reasoning_status_counts": {key: counts[key] for key in ("available", "missing", "empty", "request_failed")},
        "participant_reasoning_readable_responses": counts["readable"],
        "participant_reasoning_truncated_responses": counts["truncated"],
        "participant_reasoning_redacted_responses": counts["redacted"],
        "participant_reasoning_complete_responses": counts["complete"],
        "participant_reasoning_capture_complete": (
            counts["responses"] > 0 and counts["complete"] == counts["responses"]
            if backend == "llm" else None
        ),
    }
