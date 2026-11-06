"""Run the independent fixed_preferences_memory_v1 experiment (Baseline, 25V or 25F).

The scripted backend exercises the real PersonAgent -> participant model ->
environment adapter -> ReActRouter -> RiskSharingEnv path without network access. The llm
backend uses the same wiring after loading the project-root .env.
"""

from __future__ import annotations

import argparse
import copy
import asyncio
import hashlib
from importlib import import_module
import json
import re
import random
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from custom.config import (  # noqa: E402
    DEFAULT_ENV_PATH,
    RuntimeConfig,
    RuntimeConfigError,
    load_runtime_config,
)
from custom.insurance_assignment import validate_assignment_configuration  # noqa: E402
from protocol import paper_spec as protocol_spec  # noqa: E402
from custom.agents.fixed_preferences_v1.profiles import (  # noqa: E402
    build_profiles, profile_sha256, validate_profile,
)
from custom.agents.observation_presentation import PRESENTATION_VERSION  # noqa: E402
from custom.agents.fixed_preferences_memory_v1.state import (  # noqa: E402
    MEMORY_UPDATE_MAX_ATTEMPTS, initial_state, load_state, memory_payload, memory_sha256, observation_sha256,
    parse_memory_output, write_state,
)


VERSION = "fixed_preferences_memory_v1"
DEFAULT_EXPERIMENT_TIMEOUT_SECONDS = 6 * 60 * 60


def _validate_population(participants: int, agent_concurrency: int) -> None:
    """Keep fixed pairs complete and bound concurrent participant turns."""

    if (
        type(participants) is not int
        or participants < protocol_spec.PAIR_SIZE
        or participants % protocol_spec.PAIR_SIZE != 0
    ):
        raise ValueError("participants must be an even integer of at least 2")
    if type(agent_concurrency) is not int or not 1 <= agent_concurrency <= participants:
        raise ValueError("agent_concurrency must be between 1 and participants")


class RequestBudgetExceeded(RuntimeError):
    """Raised before a request would exceed the experiment-wide call cap."""


@dataclass
class RequestBudget:
    max_calls: int
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0

    def reserve(self) -> None:
        if self.calls >= self.max_calls:
            raise RequestBudgetExceeded("model request budget exhausted")
        self.calls += 1

    def record_usage(self, response: Any) -> None:
        usage = getattr(response, "usage", None)
        if usage is None:
            return
        self.input_tokens += int(getattr(usage, "prompt_tokens", 0) or 0)
        self.output_tokens += int(getattr(usage, "completion_tokens", 0) or 0)


_REDACTED = "<redacted>"


def _redact_text(value: Any) -> str:
    """Redact credential-like values before writing diagnostics."""

    text = str(value)
    patterns = (
        (r"(?i)(authorization\s*[:=]\s*(?:bearer\s+)?)[^\s,;]+", r"\1" + _REDACTED),
        (r"(?i)(bearer\s+)[^\s,;]+", r"\1" + _REDACTED),
        (r"(?i)((?:api[_-]?key|secret|password)\s*[:=]\s*)[^\s,;]+", r"\1" + _REDACTED),
    )
    for pattern, replacement in patterns:
        text = re.sub(pattern, replacement, text)
    return text


def _safe_log_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _safe_log_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_log_value(item) for item in value]
    if isinstance(value, str):
        return _redact_text(value)
    return value


class RunLogger:
    """Append secret-safe JSON events for one experiment run."""

    def __init__(self, path: Path, *, announce: bool = True):
        self.path = path
        if announce:
            self.write("run_started")

    def write(self, event: str, **fields: Any) -> None:
        row = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": event,
        }
        row.update({key: _safe_log_value(value) for key, value in fields.items()})
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def _append_run_log(path: Path, event: str, **fields: Any) -> None:
    RunLogger(path, announce=False).write(event, **fields)


def _router_request(messages: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Read the adapter's explicit RPC; never derive an economic choice."""

    for message in messages:
        content = message.get("content")
        if message.get("role") != "user" or not isinstance(content, str):
            continue
        if "<risk_sharing_rpc>" not in content:
            continue
        raw = content.split("<risk_sharing_rpc>", 1)[1].split("</risk_sharing_rpc>", 1)[0]
        request = json.loads(raw)
        if (
            isinstance(request, dict)
            and request.get("tool") in {"get_observation", "submit_decision", "submit_insurance_decision"}
            and isinstance(request.get("arguments"), dict)
        ):
            return request
    raise ValueError("Router requires an explicit request from the participant adapter.")


class BudgetedClient:
    """Apply run limits and bounded participant action progression.

    The guard only constrains the participant model's available tools after
    the initial observation. It never creates a decision or an amount.
    """

    def __init__(
        self,
        inner: Any,
        budget: RequestBudget,
        max_output_tokens: int,
        *,
        enforce_participant_progression: bool = False,
        enforce_router_progression: bool = False,
        call_logger: RunLogger | None = None,
        deliberation_logger: RunLogger | None = None,
        agent_id: int | None = None,
        pair_id: int | None = None,
        preference_profile: dict[str, str] | None = None,
        context_logger: RunLogger | None = None,
        private_memory_logger: RunLogger | None = None,
        private_memory_scope: str | None = None,
        private_memory_path: Path | None = None,
    ):
        self.inner = inner
        self.budget = budget
        self.max_output_tokens = max_output_tokens
        self.model_name = getattr(inner, "model_name", "")
        self.enforce_participant_progression = enforce_participant_progression
        self.enforce_router_progression = enforce_router_progression
        self.actual_calls = 0
        self.call_logger = call_logger
        self.deliberation_logger = deliberation_logger
        self.deliberation_calls = 0
        self.memory_update_calls = 0
        self.memory_update_commits = 0
        self.memory_update_rejections = 0
        self._last_memory_update_request_index = None
        self._last_memory_update_attempt = None
        self.private_decision_contexts: dict[str, dict[str, Any]] = {}
        self.agent_id = agent_id
        self.pair_id = pair_id
        self.preference_profile = (
            validate_profile(preference_profile) if preference_profile is not None else None
        )
        self.context_logger = context_logger
        self.private_memory_logger = private_memory_logger
        self.private_memory_scope = private_memory_scope
        self.private_memory_path = private_memory_path
        self._participant_tool_schemas: list[dict[str, Any]] = []
        self._router_tool_schemas: list[dict[str, Any]] = []

    def _preference_context(self, messages):
        """Check the actual participant input against its bound private profile."""
        if not self.enforce_participant_progression:
            return
        system = "\n".join(str(m.get("content", "")) for m in messages if m.get("role") == "system")
        blocks = re.findall(r"<fixed_preferences>\s*(\{.*?\})\s*</fixed_preferences>", system, re.DOTALL)
        if len(blocks) != 1 or self.preference_profile is None:
            raise ValueError("Exactly one bound participant preference profile is required")
        profile = validate_profile(json.loads(blocks[0]))
        if profile != self.preference_profile:
            raise ValueError("Participant preferences differ from the assigned profile")
        turn = ScriptedLLMClient._participant_turn(messages)
        if turn is None or turn[0] != self.agent_id or turn[1].get("agent_id") != self.agent_id:
            raise ValueError("Preference context must contain the bound participant observation")
        observation = turn[1]
        private_memory = self._bound_private_memory(messages, observation)
        return {
            "agent_id": self.agent_id, "pair_id": self.pair_id,
            "period": observation.get("current_period"), "task": observation.get("task"),
            "decision_id": observation.get("decision_id"), "preferences": profile,
            "preferences_sha256": profile_sha256(profile), "model_messages": messages,
            "observation_presentation": PRESENTATION_VERSION,
            "private_memory_snapshot": private_memory,
            "private_memory_sha256": memory_sha256(private_memory),
        }

    @staticmethod
    def _memory_block(messages):
        blocks = []
        for message in messages:
            content = message.get("content")
            if isinstance(content, str):
                blocks.extend(re.findall(r"<private_memory>\s*(.*?)\s*</private_memory>", content, re.DOTALL))
        if len(blocks) != 1:
            raise ValueError("Exactly one bound participant private memory is required")
        return json.loads(blocks[0])

    def _bound_private_memory(self, messages, observation):
        """Check the actual note against this client's own persisted state."""

        supplied = self._memory_block(messages)
        if self.private_memory_path is None or self.private_memory_scope is None:
            raise ValueError("The participant client requires its own private memory binding")
        state = load_state(
            self.private_memory_path, agent_id=self.agent_id,
            partner_id=observation["partner_id"], scope_id=self.private_memory_scope,
            preferences_sha256=profile_sha256(self.preference_profile), observation=observation,
        )
        expected = memory_payload(state)
        if supplied != expected:
            raise ValueError("Participant private memory differs from its bound workspace state")
        return expected

    def _check_memory_binding(self, state, observation):
        if (
            not isinstance(state, dict)
            or state.get("owner_agent_id") != self.agent_id
            or state.get("partner_id") != observation["partner_id"]
            or state.get("scope_id") != self.private_memory_scope
            or state.get("preferences_sha256") != profile_sha256(self.preference_profile)
        ):
            raise ValueError("Private memory does not belong to this participant, pair, run, and profile")

    def _restrict_participant_tools(self, messages: list[dict[str, Any]], kwargs: dict[str, Any]) -> None:
        """Allow a failed request to be corrected; finish after a receipt."""

        if not self.enforce_participant_progression:
            return

        advertised = kwargs.get("tools")
        if advertised:
            self._participant_tool_schemas = [
                tool for tool in advertised if isinstance(tool, dict)
            ]
        if not self._participant_tool_schemas:
            return

        latest_tool = next(
            (
                message
                for message in reversed(messages)
                if isinstance(message, dict) and message.get("role") == "tool"
            ),
            None,
        )
        if latest_tool:
            # PersonAgent native tool replies have no name field. Inspect the
            # preceding assistant tool call to distinguish its initial
            # readonly observation from the actual action request.
            previous_assistant = next(
                (
                    message
                    for message in reversed(messages[: messages.index(latest_tool)])
                    if isinstance(message, dict)
                    and message.get("role") == "assistant"
                ),
                None,
            )
            previous_call = None
            if previous_assistant:
                calls = previous_assistant.get("tool_calls") or []
                if calls:
                    previous_call = calls[-1]
            previous_function = (
                previous_call.get("function", {})
                if isinstance(previous_call, dict)
                else {}
            )
            previous_args = previous_function.get("arguments", "{}")
            try:
                previous_args = json.loads(previous_args)
            except (TypeError, json.JSONDecodeError):
                previous_args = {}
            if str(latest_tool.get("content", "")).startswith("ERROR:"):
                # A rejected unadvertised tool is not a completed action.
                # Keep the legitimate choices available so the participant
                # can correct itself instead of being forced to finish.
                allowed_name = (
                    "ask_env" if previous_function.get("name") == "ask_env" else None
                )
            elif (
                previous_function.get("name") == "ask_env"
                and bool(previous_args.get("readonly"))
            ):
                allowed_name = "ask_env"
            else:
                allowed_name = "finish"
        else:
            # PersonAgent.step() has already supplied the readonly observation
            # in the initial participant prompt.  Permit the model to choose
            # either its structured decision or a waiting finish.
            has_initial_observation = ScriptedLLMClient._participant_turn(messages) is not None
            allowed_name = "ask_env" if not has_initial_observation else None

        if allowed_name is None:
            restricted = [
                copy.deepcopy(tool)
                for tool in self._participant_tool_schemas
                if isinstance(tool.get("function"), dict)
                and tool["function"].get("name") in {"ask_env", "finish"}
            ]
        else:
            restricted = [
                copy.deepcopy(tool)
                for tool in self._participant_tool_schemas
                if isinstance(tool.get("function"), dict)
                and tool["function"].get("name") == allowed_name
            ]
        if not restricted:
            return

        if allowed_name == "ask_env":
            parameters = restricted[0]["function"].setdefault(
                "parameters", {}
            )
            properties = parameters.setdefault("properties", {})
            readonly = properties.setdefault("readonly", {})
            readonly["type"] = "boolean"
            readonly["enum"] = [False]
            required = list(parameters.get("required") or [])
            if "readonly" not in required:
                required.append("readonly")
            parameters["required"] = required

        kwargs["tools"] = restricted
        kwargs["tool_choice"] = "auto" if allowed_name is None else "required"

    def _restrict_router_tools(self, messages, kwargs) -> bool:
        """Constrain routing and end completed RPCs without creating a decision."""

        request = _router_request(messages)
        advertised = kwargs.get("tools")
        if advertised:
            self._router_tool_schemas = advertised
        for message in reversed(messages):
            if message.get("role") != "tool" or message.get("name") != request["tool"]:
                continue
            try:
                result = json.loads(message.get("content", ""))
            except (TypeError, json.JSONDecodeError):
                break
            if isinstance(result, dict) and "error" not in result:
                # This only terminates the Router loop after a real tool result.
                # No amount, decision type or id is produced by this shortcut.
                return True
            break
        tools = [
            copy.deepcopy(tool) for tool in self._router_tool_schemas
            if tool.get("function", {}).get("name") == request["tool"]
        ]
        if not tools:
            raise ValueError("Router request does not match an allowed environment tool.")
        properties = tools[0]["function"]["parameters"]["properties"]
        for name, value in request["arguments"].items():
            if name not in properties:
                raise ValueError("Router request contains an unknown tool argument.")
            properties[name]["enum"] = [value]
        kwargs["tools"] = tools
        kwargs["tool_choice"] = "required"
        return False

    async def call(self, model: str | None = None, messages=None, **kwargs: Any):
        messages = list(messages or [])
        phase = kwargs.pop("participant_phase", None)
        has_memory_attempt = "participant_memory_attempt" in kwargs
        memory_attempt = kwargs.pop("participant_memory_attempt", 1)
        if phase is not None and (
            phase not in {"deliberation", "memory_update"}
            or not self.enforce_participant_progression
            or self.enforce_router_progression
        ):
            raise ValueError("Only the participant client may request a private review or memory update.")
        is_deliberation = phase == "deliberation"
        is_memory_update = phase == "memory_update"
        if has_memory_attempt and not is_memory_update:
            raise ValueError("Only a private memory update may specify an attempt number")
        if is_memory_update:
            if type(memory_attempt) is not int or not 1 <= memory_attempt <= MEMORY_UPDATE_MAX_ATTEMPTS:
                raise ValueError("Private memory attempt exceeds the bounded correction limit")
        attempt_fields = {"memory_update_attempt": memory_attempt} if is_memory_update else {}
        is_private_call = is_deliberation or is_memory_update
        if is_private_call:
            # Never reuse tools cached by an earlier action or restored turn.
            kwargs.pop("tools", None)
            kwargs.pop("tool_choice", None)
            kwargs.pop("response_format", None)
        if self.enforce_router_progression and self._restrict_router_tools(messages, kwargs):
            message = SimpleNamespace(content="Environment request completed.", tool_calls=[])
            return SimpleNamespace(choices=[SimpleNamespace(message=message)])
        if is_private_call:
            private_limit = 1024 if is_memory_update else 512
            kwargs["max_output_tokens"] = min(
                kwargs.get("max_output_tokens", private_limit), private_limit, self.max_output_tokens,
            )
        else:
            self._restrict_participant_tools(messages, kwargs)
            kwargs.setdefault("max_output_tokens", self.max_output_tokens)
        # LLMClient clamps this to one actual attempt; no unbounded retry loop.
        kwargs["max_retries"] = 0
        context = self._preference_context(messages)
        self.budget.reserve()
        request_index = self.budget.calls
        self.actual_calls += 1
        if is_deliberation:
            self.deliberation_calls += 1
        if is_memory_update:
            self.memory_update_calls += 1
            self._last_memory_update_request_index = request_index
            self._last_memory_update_attempt = memory_attempt
        if context is not None and phase is None and context.get("decision_id") is not None:
            self.private_decision_contexts[context["decision_id"]] = {
                "private_memory_sha256": context["private_memory_sha256"],
                "private_memory_revision": context["private_memory_snapshot"]["revision"],
                "participant_request_index": request_index,
            }
        if context is not None and self.context_logger is not None:
            self.context_logger.write(
                "participant_model_input", **context, request_index=request_index,
                phase=phase or "decision", **attempt_fields,
            )
        started = time.monotonic()
        error_type = None
        try:
            response = await self.inner.call(model=model, messages=messages, **kwargs)
        except BaseException as exc:
            error_type = type(exc).__name__
            raise
        finally:
            if self.call_logger is not None:
                self.call_logger.write(
                    "model_call_finished",
                    agent_id=self.agent_id,
                    role="router" if self.enforce_router_progression else "participant",
                    phase=(phase or ("routing" if self.enforce_router_progression else "decision")),
                    request_index=request_index,
                    duration_seconds=round(time.monotonic() - started, 6),
                    ok=error_type is None,
                    error_type=error_type,
                    **attempt_fields,
                )
        self.budget.record_usage(response)
        return response

    def record_private_memory_update(
        self, *, observation, messages, raw_output, previous_state, new_state, attempt=1,
    ):
        """Audit a committed participant-generated update without routing it."""

        if not self.enforce_participant_progression or self.enforce_router_progression:
            raise ValueError("Only the participant client may record a private memory update")
        if observation.get("agent_id") != self.agent_id:
            raise ValueError("Memory observation does not match the bound participant")
        self._check_memory_binding(previous_state, observation)
        self._check_memory_binding(new_state, observation)
        request_turn = ScriptedLLMClient._participant_turn(messages)
        if (
            request_turn is None or request_turn != (self.agent_id, observation)
            or self._memory_block(messages) != memory_payload(previous_state)
            or parse_memory_output(raw_output) != new_state["content"]
            or self._last_memory_update_request_index is None
            or type(attempt) is not int or not 1 <= attempt <= MEMORY_UPDATE_MAX_ATTEMPTS
            or attempt != self._last_memory_update_attempt
            or new_state["revision"] != previous_state["revision"] + 1
            or new_state["last_observation_sha256"] != observation_sha256(observation)
            or new_state["last_reviewed_period"] != observation["current_period"]
            or new_state["last_reviewed_task"] != observation["task"]
            or self.private_memory_path is None
            or load_state(
                self.private_memory_path, agent_id=self.agent_id,
                partner_id=observation["partner_id"], scope_id=self.private_memory_scope,
                preferences_sha256=profile_sha256(self.preference_profile), observation=observation,
            ) != new_state
        ):
            raise ValueError("A private memory audit must match the committed workspace state")
        if self.private_memory_logger is not None:
            self.private_memory_logger.write(
                "private_memory_updated", agent_id=self.agent_id, pair_id=self.pair_id,
                period=observation["current_period"], task=observation["task"],
                request_index=self._last_memory_update_request_index,
                attempt=attempt, max_attempts=MEMORY_UPDATE_MAX_ATTEMPTS,
                memory_source="participant_model", observation_sha256=observation_sha256(observation),
                model_messages=messages, raw_output=raw_output,
                previous_state=previous_state, new_state=new_state,
                previous_state_sha256=memory_sha256(previous_state), new_state_sha256=memory_sha256(new_state),
            )
        self.memory_update_commits += 1

    def record_private_memory_rejection(
        self, *, observation, messages, raw_output, previous_state, error, attempt=1,
    ):
        """Keep rejected output diagnosable without changing successful state logs."""

        if not self.enforce_participant_progression or self.enforce_router_progression:
            raise ValueError("Only the participant client may record a rejected private memory update")
        if observation.get("agent_id") != self.agent_id:
            raise ValueError("Rejected memory observation does not match the bound participant")
        self._check_memory_binding(previous_state, observation)
        if self._memory_block(messages) != memory_payload(previous_state):
            raise ValueError("Rejected memory input does not match the previous state")
        if (
            type(attempt) is not int or not 1 <= attempt <= MEMORY_UPDATE_MAX_ATTEMPTS
            or attempt != self._last_memory_update_attempt
            or self._last_memory_update_request_index is None
        ):
            raise ValueError("Rejected memory attempt does not match the actual participant call")
        if self.call_logger is not None:
            self.call_logger.write(
                "private_memory_update_rejected", agent_id=self.agent_id, pair_id=self.pair_id,
                period=observation["current_period"], task=observation["task"],
                request_index=self._last_memory_update_request_index,
                attempt=attempt, max_attempts=MEMORY_UPDATE_MAX_ATTEMPTS,
                memory_source="participant_model", phase="memory_update",
                previous_revision=previous_state["revision"],
                raw_output=raw_output, error=str(error),
            )
        self.memory_update_rejections += 1

    def record_participant_deliberation(self, *, observation, messages, review_text):
        """Keep explicit review text in a researcher log, never Router context."""

        if not self.enforce_participant_progression or self.enforce_router_progression:
            raise ValueError("Only the participant client may record deliberation.")
        if observation.get("agent_id") != self.agent_id:
            raise ValueError("Deliberation observation does not match the bound participant.")
        if self.deliberation_logger is not None:
            self.deliberation_logger.write(
                "participant_deliberation",
                agent_id=self.agent_id, pair_id=self.pair_id,
                period=observation.get("current_period"),
                decision_id=observation.get("decision_id"), task=observation.get("task"),
                decision_source="participant_model", observation=observation,
                model_messages=messages, review_text=review_text,
            )

    def take_token_stats(self) -> dict[str, dict[str, int]]:
        getter = getattr(self.inner, "take_token_stats", None)
        return getter() if getter else {}

    def snapshot_token_stats(self) -> dict[str, dict[str, int]]:
        getter = getattr(self.inner, "snapshot_token_stats", None)
        return getter() if getter else {}


class ScriptedLLMClient:
    """Deterministic model boundary; it never performs network I/O."""

    def __init__(self, kind: str):
        if kind not in {"participant", "router"}:
            raise ValueError("scripted client kind must be participant or router")
        self.kind = kind
        self.model_name = "scripted-model"
        self.calls = 0

    @staticmethod
    def _tool_call(name: str, arguments: dict[str, Any], index: int) -> Any:
        function = SimpleNamespace(
            name=name,
            arguments=json.dumps(arguments, ensure_ascii=False),
        )
        return SimpleNamespace(id=f"scripted-call-{index}", type="function", function=function)

    def _response(self, *, name: str | None = None, arguments=None, content: str = "") -> Any:
        calls = [] if name is None else [self._tool_call(name, arguments or {}, self.calls)]
        message = SimpleNamespace(content=content, tool_calls=calls)
        usage = SimpleNamespace(prompt_tokens=1, completion_tokens=1)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=usage)

    @staticmethod
    def _tool_messages(messages: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
        return [m for m in messages if m.get("role") == "tool"]

    @staticmethod
    def _participant_turn(messages: Iterable[dict[str, Any]]) -> tuple[int, dict[str, Any]] | None:
        """Extract the raw observation included in a participant prompt."""

        for message in messages:
            content = message.get("content") if isinstance(message, dict) else None
            if not isinstance(content, str) or "<risk_sharing_turn>" not in content:
                continue
            try:
                raw = content.split("<risk_sharing_turn>", 1)[1].split(
                    "</risk_sharing_turn>", 1
                )[0].strip()
                payload = json.loads(raw)
            except (IndexError, TypeError, json.JSONDecodeError):
                continue
            if not isinstance(payload, dict):
                continue
            observations = payload.get("observations") or []
            if not isinstance(observations, list):
                continue
            for item in reversed(observations):
                value = item.get("observation") if isinstance(item, dict) else None
                if isinstance(value, dict):
                    try:
                        return int(payload["agent_id"]), value
                    except (KeyError, TypeError, ValueError):
                        continue
        return None

    @staticmethod
    def _participant_amount(agent_id: int, observation: dict[str, Any]) -> int:
        """Deterministic participant policy with agent-specific amounts."""

        upper = int((observation.get("legal_transfer_range") or {}).get("max", 0))
        proposed = 13 + agent_id * 17
        return min(upper, proposed)

    async def call(self, model: str | None = None, messages=None, **kwargs: Any) -> Any:
        self.calls += 1
        messages = list(messages or [])
        tool_messages = self._tool_messages(messages)
        if self.kind == "router":
            request = _router_request(messages)
            return self._response(
                name=request["tool"], arguments=copy.deepcopy(request["arguments"])
            )
        if messages and "<private_memory_update>" in str(messages[-1].get("content", "")):
            turn = self._participant_turn(messages)
            if turn is None:
                raise ValueError("A private memory update requires its own observation")
            participant_id, observation = turn
            history = observation["history"]
            partner_rows = [row for row in history if row["agent_id"] == observation["partner_id"]
                            and row["had_transfer_opportunity"]]
            own_rows = [row for row in history if row["agent_id"] == participant_id
                        and row["had_transfer_opportunity"]]
            latest_partner = max(partner_rows, key=lambda row: row["period"], default=None)
            latest_own = max(own_rows, key=lambda row: row["period"], default=None)
            def forecast(row):
                return {"amount": None if row is None else row["transfer_sent"],
                        "confidence": "unknown" if row is None else "low"}
            # These are offline fixtures for state transport. The scripted
            # economic policy below remains unchanged and ignores the note.
            note = {
                "partner_assessment": (
                    f"Participant {participant_id}: my current assessment uses only my pair's "
                    f"settled records through period {max((r['period'] for r in history), default=0)}."
                ),
                "expected_partner_transfer": forecast(latest_partner),
                "perceived_partner_expectation": forecast(latest_own),
                "uncertainties": "Past transfers do not guarantee future choices or reveal my partner's intentions.",
                "evidence_periods": sorted({row["period"] for row in (latest_partner, latest_own) if row is not None}),
            }
            return self._response(content=json.dumps(note, ensure_ascii=False))
        if messages and "<participant_deliberation>" in str(messages[-1].get("content", "")):
            turn = self._participant_turn(messages)
            if turn is None:
                raise ValueError("A participant review requires its own observation.")
            participant_id, observation = turn
            # Offline fixture, not a learned economic strategy or conclusion.
            return self._response(content=(
                f"Participant {participant_id}, period {observation['current_period']}: "
                f"I have {len(observation.get('history', []))} settled pair records. "
                f"I will use my scripted test policy for the {observation['task']} task "
                "and check the current observation's constraints before submitting."
            ))
        if tool_messages and not str(tool_messages[-1].get("content", "")).startswith("ERROR:"):
            return self._response(
                name="finish",
                arguments={
                    "final": "本轮行动已完成。",
                    "memories": [
                        {
                            "type": "observation",
                            "importance": 0.1,
                            "keywords": ["risk_sharing"],
                            "text": "完成本轮环境观察与合法行动。",
                            "source": "step_result",
                            "refs": [],
                        }
                    ],
                },
            )
        participant_turn = self._participant_turn(messages)
        if participant_turn is None:
            return self._response(
                name="ask_env",
                arguments={
                    "instruction": "没有可用 observation；请先通过环境读取当前 observation。",
                    "readonly": False,
                },
            )
        participant_id, observation = participant_turn
        if observation.get("task") == "insurance":
            # Test fixture only: distinct, deterministic yes/no choices are
            # produced on the participant side, never by the routing client.
            purchase = participant_id % 2 == 0
            return self._response(
                name="ask_env",
                arguments={
                    "instruction": (
                        "This is my choice for the current offer.\n"
                        "decision_type: insurance\n"
                        f"decision_id: {observation['decision_id']}\n"
                        f"purchase: {str(purchase).lower()}"
                    ),
                    "readonly": False,
                },
            )
        if observation.get("task") != "transfer":
            return self._response(
                name="finish",
                arguments={
                    "final": "没有转账机会，本轮等待。",
                    "memories": [{"text": "等待", "type": "observation"}],
                },
            )
        decision = {
            "decision_type": "transfer",
            "decision_id": observation.get("decision_id"),
            "amount": self._participant_amount(participant_id, observation),
        }
        return self._response(
            name="ask_env",
            arguments={
                "instruction": (
                    "I have chosen this transfer using my current observation.\n"
                    f"decision_type: {decision['decision_type']}\n"
                    f"decision_id: {decision['decision_id']}\n"
                    f"amount: {decision['amount']}"
                ),
                "readonly": False,
            },
        )


def _jsonl_append(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _select_payment_period(completed_periods: list[int], seed: int) -> int:
    """Select one realized period without consuming the environment RNG."""

    if not completed_periods:
        raise ValueError("Cannot select a payment period before settlement.")
    material = f"risk-sharing-payment-v1:{seed}".encode("utf-8")
    payment_seed = int.from_bytes(hashlib.sha256(material).digest()[:16], "big")
    payment_rng = random.Random(payment_seed)
    return completed_periods[payment_rng.randrange(len(completed_periods))]


def _make_run_dir(
    path: Path | None, treatment: protocol_spec.Treatment = protocol_spec.Treatment.BASELINE,
) -> Path:
    if path is not None:
        run_dir = path.expanduser().resolve()
        if run_dir.exists():
            raise RuntimeConfigError(f"run directory already exists: {run_dir}")
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        experiment = treatment.value.lower()
        run_dir = PROJECT_ROOT / "experiments" / VERSION / "runs" / experiment / f"{stamp}-{uuid.uuid4().hex[:8]}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _write_manifest(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def _safe_action_records(
    records: list[dict[str, Any]],
    env: Any,
    framework_step: int,
    participant_decisions: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    safe: list[dict[str, Any]] = []
    participant_decisions = participant_decisions or []
    for record in records:
        kwargs = dict(record.get("kwargs") or {})
        function_name = record.get("function_name")
        agent_id = kwargs.get("agent_id")
        row: dict[str, Any] = {
            "framework_step": framework_step,
            "agent_id": agent_id,
            "pair_id": env.pair_by_agent.get(agent_id),
            "action": function_name,
            "ok": not bool(record.get("exception_occurred")),
        }
        if function_name == "get_observation":
            row["observation"] = record.get("return_value")
            row["router_action"] = "direct_observation"
            row["router_model_called"] = False
            row["decision_source"] = "participant_observation_request"
        elif function_name in {"submit_decision", "submit_insurance_decision"}:
            value_field = "purchase" if function_name == "submit_insurance_decision" else "amount"
            row["router_action"] = "router_forward"
            row["router_model_called"] = True
            row[f"submitted_{value_field}"] = kwargs.get(value_field)
            row["decision_id"] = kwargs.get("decision_id")
            row["receipt"] = record.get("return_value")
            row["environment_receipt"] = record.get("return_value")
            participant = next(
                (
                    item
                    for item in participant_decisions
                    if item.get("agent_id") == agent_id
                    and item.get("decision_id") == kwargs.get("decision_id")
                    and item.get("decision_type") == kwargs.get("decision_type")
                    and item.get(value_field) == kwargs.get(value_field)
                    and type(item.get(value_field)) is type(kwargs.get(value_field))
                ),
                None,
            )
            if participant is not None:
                row["decision_source"] = "participant_model"
                row["participant_model_output"] = {
                    "decision_type": participant.get("decision_type"),
                    "decision_id": participant.get("decision_id"),
                    value_field: participant.get(value_field),
                }
                if isinstance(participant.get("participant_observation"), dict):
                    row["participant_model_message"] = {
                        "observation": participant["participant_observation"],
                        "contains_observation": True,
                    }
                row[f"participant_{value_field}"] = participant.get(value_field)
                row["participant_raw_request"] = _redact_text(participant.get("raw_request", ""))
                row["participant_variables"] = _safe_log_value(
                    participant.get("participant_variables", {})
                )
            else:
                row["decision_source"] = "unknown_without_participant_record"
        if record.get("exception_occurred"):
            row["error"] = record.get("exception_info")
        safe.append(row)
    return safe


async def _run_async(
    *,
    cfg: RuntimeConfig,
    participants: int,
    completed_period_limit: int,
    full_session: bool,
    seed: int,
    run_dir: Path,
    request_limit: int,
    max_output_tokens: int,
    router_steps: int,
    agent_turns: int,
    agent_concurrency: int,
    timeout_seconds: float,
    participant_deliberation: bool = False,
    treatment: protocol_spec.Treatment = protocol_spec.Treatment.BASELINE,
    forced_assignment_probabilities: tuple[float, float, float] | None = None,
    assignment_seed: int | None = None,
    profiles: dict[int, dict[str, str]],
    preference_assignment: dict[str, Any],
) -> dict[str, Any]:
    _validate_population(participants, agent_concurrency)
    run_log = RunLogger(run_dir / "run.log")
    run_log.write(
        "configuration_loaded",
        backend=cfg.backend,
        treatment=treatment.value,
        config_status=cfg.safe_status(),
        participants=participants,
        completed_period_limit=completed_period_limit,
        full_session=full_session,
        request_limit=request_limit,
        max_output_tokens=max_output_tokens,
        agent_concurrency=agent_concurrency,
        timeout_seconds=timeout_seconds,
        participant_deliberation=participant_deliberation,
        observation_presentation=PRESENTATION_VERSION,
    )

    # Imports that load AgentSociety2 configuration occur only after the
    # project-root configuration loader has mapped the environment variables.
    # The installed registry requires an allowed workspace root even for this
    # in-process runner; preserve an existing value and otherwise use the
    # project root (the per-run agent workspaces remain under ``run_dir``).
    import os

    os.environ.setdefault("WORKSPACE_PATH", str(PROJECT_ROOT))
    from agentsociety2.agent.service_proxy import LLMClients, ServiceProxy
    from agentsociety2.agent.runner import _step_agent_batch_async
    from agentsociety2.config.llm_dispatcher import build_client_for_role
    from custom.agents.fixed_preferences_memory_v1.participant import FixedPreferenceMemoryParticipant
    from custom.envs.risk_sharing_env import (
        RiskSharingEnv, RiskSharingReActRouter, RiskSharingRouterAdapter,
    )

    if completed_period_limit < 1:
        raise ValueError("completed_period_limit must be positive")

    budget = RequestBudget(max_calls=request_limit)
    private_memory_scope = uuid.uuid4().hex
    env = RiskSharingEnv(
        list(range(1, participants + 1)),
        treatment=treatment,
        seed=seed,
        session_id=f"smoke-{uuid.uuid4().hex}",
        forced_assignment_probabilities=forced_assignment_probabilities,
        assignment_seed=assignment_seed,
    )
    if env.assignment_configuration is not None:
        run_log.write("insurance_assignment_configuration", **env.assignment_configuration)
    participant_clients: list[BudgetedClient] = []
    router_clients: list[BudgetedClient] = []

    call_log = RunLogger(run_dir / "run.log", announce=False)
    context_log = RunLogger(run_dir / "participant_contexts.jsonl", announce=False)
    memory_path = run_dir / "private_memory_updates.jsonl"
    memory_path.touch()
    memory_log = RunLogger(memory_path, announce=False)
    deliberation_log = None
    if participant_deliberation:
        review_path = run_dir / "participant_deliberations.jsonl"
        review_path.touch()
        deliberation_log = RunLogger(review_path, announce=False)

    def _make_agent_services(agent_id: int) -> tuple[RiskSharingRouterAdapter, ServiceProxy]:
        """Give each participant its own model clients and trusted Router adapter."""

        router = RiskSharingReActRouter([env], max_steps=router_steps, max_llm_call_retry=1)
        if cfg.backend == "scripted":
            participant_inner = ScriptedLLMClient("participant")
            router_inner = ScriptedLLMClient("router")
        else:
            participant_inner = build_client_for_role("default")
            router_inner = build_client_for_role("coder")
        participant_client = BudgetedClient(
            participant_inner, budget, max_output_tokens,
            enforce_participant_progression=True,
            call_logger=call_log, agent_id=agent_id,
            deliberation_logger=deliberation_log,
            pair_id=env.pair_by_agent[agent_id],
            preference_profile=profiles[agent_id], context_logger=context_log,
            private_memory_logger=memory_log, private_memory_scope=private_memory_scope,
            private_memory_path=run_dir / "agents" / f"agent_{agent_id:04d}" / "private_memory.json",
        )
        router_client = BudgetedClient(
            router_inner, budget, max_output_tokens,
            enforce_router_progression=True,
            call_logger=call_log, agent_id=agent_id,
        )
        participant_clients.append(participant_client)
        router_clients.append(router_client)
        router._coder_dispatcher = router_client
        router._summary_dispatcher = router_client
        adapter = RiskSharingRouterAdapter(router, env)
        proxy = ServiceProxy(
            env=adapter,
            llm=LLMClients(coder=None, default=participant_client, embedding=None),
            trace=None,
            replay=None,
            run_dir=run_dir,
        )
        return adapter, proxy

    agent_services = {
        agent_id: _make_agent_services(agent_id)
        for agent_id in range(1, participants + 1)
    }
    # All adapters point at the same RiskSharingEnv.  Use one only for the
    # scheduler's current-time/settlement calls; participant calls use their
    # own adapter and participant model client.
    control_adapter = agent_services[1][0]
    run_log.write(
        "framework_ready",
        router="ReActRouter",
        router_model=cfg.model,
        router_api_enabled=cfg.backend == "llm",
        router_model_requests=0,
        agent_class="FixedPreferenceMemoryParticipant",
        pairs=len(env.pair_members),
        agent_concurrency=agent_concurrency,
        isolated_participant_clients=participants,
    )

    workspace_root = run_dir / "agents"
    agent_config = {
        "enable_memory": False,
        "private_memory_enabled": True,
        "private_memory_scope": private_memory_scope,
        "participant_deliberation": participant_deliberation,
        "enable_todo_list": False,
        "max_react_turns": agent_turns,
        # RiskSharingParticipant enforces an empty skill runtime; 2.9.0's
        # disabled_skill_ids accepts exact IDs, not a '*' wildcard.
    }
    for agent_id in range(1, participants + 1):
        FixedPreferenceMemoryParticipant.create(
            workspace_root / f"agent_{agent_id:04d}",
            {"id": agent_id, "name": f"participant-{agent_id}"},
            {**agent_config, "fixed_preferences": copy.deepcopy(profiles[agent_id]),
             "fixed_preferences_sha256": profile_sha256(profiles[agent_id])},
        )
        write_state(
            workspace_root / f"agent_{agent_id:04d}" / "private_memory.json",
            initial_state(agent_id, env.partner_by_agent[agent_id], private_memory_scope,
                          profile_sha256(profiles[agent_id])),
        )

    # Use the framework's real workspace batch runner and class restoration.
    from agentsociety2.registry import get_registry

    get_registry().register_agent_module(
        "FixedPreferenceMemoryParticipant", FixedPreferenceMemoryParticipant, is_custom=True
    )

    action_path = run_dir / "agent_actions.jsonl"
    records_path = run_dir / "period_records.jsonl"
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    start_history = 0
    stop_reason = "unknown"
    scheduler_steps = 0
    result_batches: list[dict[str, Any]] = []
    failure_details: list[dict[str, Any]] = []
    max_scheduler_steps = (
        max(3, protocol_spec.REAL_MAXIMUM_PERIOD_COUNT * 4)
        if full_session
        else max(3, completed_period_limit * 4)
    )
    started = time.monotonic()
    participant_decision_cursor = 0
    participant_rejection_cursor = 0
    assignment_cursor = 0
    agent_semaphore = asyncio.Semaphore(agent_concurrency)

    async def _step_one_agent(agent_id: int) -> list[dict[str, Any]]:
        async with agent_semaphore:
            _adapter, proxy = agent_services[agent_id]
            return await _step_agent_batch_async(
                [agent_id],
                str(workspace_root),
                "FixedPreferenceMemoryParticipant",
                scheduler_steps,
                now,
                proxy,
            )

    while (not env.finished) if full_session else (len(env.completed_periods) < completed_period_limit):
        if scheduler_steps >= max_scheduler_steps:
            stop_reason = "scheduler_step_limit"
            run_log.write("stop_requested", reason=stop_reason, scheduler_step=scheduler_steps)
            break
        scheduler_steps += 1
        run_log.write(
            "scheduler_step_started",
            scheduler_step=scheduler_steps,
            current_period=env.current_period,
            stage=env.stage.value,
            completed_periods=list(env.completed_periods),
        )
        for adapter, _proxy in agent_services.values():
            adapter.set_current_time(now)
        # Run PersonAgents concurrently up to the configured bound.  The
        # gather is a barrier: env.step() below still settles only after every
        # participant has finished this period's action.
        batches = await asyncio.gather(
            *(_step_one_agent(agent_id) for agent_id in range(1, participants + 1))
        )
        batch = [item for agent_batch in batches for item in agent_batch]
        result_batches.extend(batch)
        participant_decisions = control_adapter.participant_decisions[
            participant_decision_cursor:
        ]
        participant_decision_cursor = len(control_adapter.participant_decisions)
        step_actions = _safe_action_records(
            env.get_tool_call_history()[start_history:],
            env,
            scheduler_steps,
            participant_decisions,
        )
        step_actions.extend(
            _safe_log_value({"framework_step": scheduler_steps, **row})
            for row in control_adapter.participant_rejections[participant_rejection_cursor:]
        )
        participant_rejection_cursor = len(control_adapter.participant_rejections)
        for row in step_actions:
            if row.get("agent_id") in profiles:
                row["preferences_sha256"] = profile_sha256(profiles[row["agent_id"]])
                client = participant_clients[row["agent_id"] - 1]
                memory_context = client.private_decision_contexts.get(row.get("decision_id"))
                if memory_context is not None:
                    row.update(memory_context)
        _jsonl_append(action_path, step_actions)
        run_log.write(
            "tool_actions",
            scheduler_step=scheduler_steps,
            actions=[
                {
                    "agent_id": row.get("agent_id"),
                    "pair_id": row.get("pair_id"),
                    "action": row.get("action"),
                    "ok": row.get("ok"),
                    "task": (row.get("observation") or {}).get("task")
                    if isinstance(row.get("observation"), dict)
                    else None,
                    "submitted_amount": row.get("submitted_amount"),
                    "participant_amount": row.get("participant_amount"),
                    "submitted_purchase": row.get("submitted_purchase"),
                    "participant_purchase": row.get("participant_purchase"),
                    "decision_source": row.get("decision_source"),
                    "error": row.get("error"),
                }
                for row in step_actions
            ],
        )
        start_history = len(env.get_tool_call_history())
        for item in batch:
            run_log.write(
                "agent_result",
                scheduler_step=scheduler_steps,
                agent_id=item.get("id"),
                ok=bool(item.get("ok")),
                error=item.get("error"),
            )
        failures = [item for item in batch if not item.get("ok")]
        if failures:
            failure_details = [
                {"agent_id": item.get("id"), "error": _redact_text(item.get("error", ""))}
                for item in failures
            ]
            stop_reason = "agent_error"
            run_log.write("agent_error", scheduler_step=scheduler_steps, failures=failure_details)
            break
        before = len(env.completed_periods)
        period_before = env.current_period
        stage_before = env.stage.value
        await control_adapter.step(1, now)
        assignments = env.insurance_assignments
        new_assignments = assignments[assignment_cursor:]
        assignment_cursor = len(assignments)
        if new_assignments:
            # Record computer assignments separately from participant decisions.
            run_log.write(
                "insurance_assigned", scheduler_step=scheduler_steps,
                assignments=new_assignments,
            )
            _jsonl_append(action_path, [
                {**record, "framework_step": scheduler_steps,
                 "action": "insurance_assignment", "ok": True,
                 "decision_source": "environment_random",
                 "router_action": "not_applicable", "router_model_called": False}
                for record in new_assignments
            ])
        after = len(env.completed_periods)
        run_log.write(
            "scheduler_step_finished",
            scheduler_step=scheduler_steps,
            period_before=period_before,
            period_after=env.current_period,
            stage_before=stage_before,
            stage_after=env.stage.value,
            settled_period=env.completed_periods[-1] if after > before else None,
            finished=env.finished,
            completed_periods=list(env.completed_periods),
            model_requests=budget.calls,
        )
        if after == before and budget.calls >= request_limit:
            stop_reason = "budget_exhausted"
            run_log.write("stop_requested", reason=stop_reason, scheduler_step=scheduler_steps)
            break
        now += timedelta(seconds=1)
        if time.monotonic() - started > timeout_seconds:
            stop_reason = "runtime_timeout"
            run_log.write("stop_requested", reason=stop_reason, scheduler_step=scheduler_steps)
            break
    else:
        stop_reason = "natural_termination" if full_session else "smoke_period_limit"

    # Records come directly from the environment, never from model text.
    _jsonl_append(records_path, env.history)
    insurance_decisions = [
        {
            "agent_id": agent_id, "pair_id": env.pair_by_agent[agent_id],
            "partner_id": env.partner_by_agent[agent_id],
            "period": period, "decision_id": token, "purchase": value,
            "treatment": treatment.value,
        }
        for token, (agent_id, period, decision_type, value) in env._decision_receipts.items()
        if decision_type == "insurance"
    ]
    if treatment is protocol_spec.Treatment.VOLUNTARY_INSURANCE_25:
        _jsonl_append(run_dir / "insurance_decisions.jsonl", insurance_decisions)
    insurance_assignments = env.insurance_assignments
    if treatment is protocol_spec.Treatment.FORCED_INSURANCE_25:
        _jsonl_append(run_dir / "insurance_assignments.jsonl", insurance_assignments)
    payment_period: int | None = None
    payment_path: Path | None = None
    payment_records: list[dict[str, Any]] = []
    if full_session and env.finished:
        payment_period = _select_payment_period(list(env.completed_periods), seed)
        payment_records = [
            dict(record)
            for record in env.history
            if int(record["period"]) == payment_period
        ]
        payment_path = run_dir / "payment.json"
        _write_manifest(
            payment_path,
            {
                "schema_version": 1,
                "selection_rule": "uniformly select one realized period",
                "selected_period": payment_period,
                "records": payment_records,
            },
        )
        run_log.write(
            "payment_selected",
            selected_period=payment_period,
            realized_period_count=len(env.completed_periods),
            record_count=len(payment_records),
        )

    valid_decisions = [
        item for item in env.get_tool_call_history()
        if item.get("function_name") == "submit_decision" and not item.get("exception_occurred")
    ]
    unique_decisions = {str(item.get("kwargs", {}).get("decision_id")) for item in valid_decisions}
    expected_period_count = (
        len(env.completed_periods) if full_session else completed_period_limit
    )
    records_complete = (
        len(unique_decisions) == len(env.pair_members) * expected_period_count
        and len(env.history) == participants * expected_period_count
        and len(insurance_decisions) == (
            participants
            if treatment is protocol_spec.Treatment.VOLUNTARY_INSURANCE_25
            and expected_period_count >= protocol_spec.INSURANCE_DECISION_PERIOD
            else 0
        )
    )
    if treatment is protocol_spec.Treatment.FORCED_INSURANCE_25:
        # Settling period 15 already prepares period 16. Retain the actual
        # assignments even in a smoke that stops before agents act in 16.
        assignment_expected = env.current_period >= protocol_spec.INSURANCE_DECISION_PERIOD
        records_complete = records_complete and (
            len(insurance_assignments) == (participants if assignment_expected else 0)
            and (
                not assignment_expected
                or all(
                    any(row["insured"] for row in insurance_assignments if row["pair_id"] == pair_id)
                    for pair_id in env.pair_members
                )
            )
        )
    if full_session:
        successful = (
            stop_reason == "natural_termination"
            and env.finished
            and protocol_spec.REAL_TERMINATION_FIRST_POSSIBLE_PERIOD
            <= max(env.completed_periods, default=0)
            <= protocol_spec.REAL_TERMINATION_FORCED_PERIOD
            and records_complete
            and payment_period in env.completed_periods
            and len(payment_records) == 2 * len(env.pair_members)
        )
    else:
        successful = (
            stop_reason == "smoke_period_limit"
            and len(env.completed_periods) >= completed_period_limit
            and records_complete
        )
    if not successful and stop_reason in {"smoke_period_limit", "natural_termination"}:
        stop_reason = "incomplete_records"

    manifest = {
        "schema_version": 1,
        "experiment_version": VERSION,
        "observation_presentation": PRESENTATION_VERSION,
        "observation_presentation_sha256": _file_hash(PROJECT_ROOT / "custom/agents/observation_presentation.py"),
        "enable_memory": False,
        "framework_memory_enabled": False,
        "private_memory_enabled": True,
        "memory_implementation": "participant_workspace_json",
        "private_memory_update_requests": sum(client.memory_update_calls for client in participant_clients),
        "private_memory_update_count": sum(client.memory_update_commits for client in participant_clients),
        "private_memory_update_rejections": sum(client.memory_update_rejections for client in participant_clients),
        "private_memory_update_max_attempts": MEMORY_UPDATE_MAX_ATTEMPTS,
        "private_memory_updates_file": "private_memory_updates.jsonl",
        "private_memory_files": "agents/agent_XXXX/private_memory.json",
        "memory_update_prompt_sha256": _file_hash(PROJECT_ROOT / "custom" / "agents" / VERSION / "memory_update.md"),
        "memory_usage_prompt_sha256": _file_hash(PROJECT_ROOT / "custom" / "agents" / VERSION / "memory_usage.md"),
        "preferences_file": "participant_profiles.json",
        "preferences_sha256": _file_hash(run_dir / "participant_profiles.json"),
        "preference_assignment": preference_assignment,
        "participant_contexts_file": "participant_contexts.jsonl",
        "baseline_reference_file": "baseline_reference.json",
        "source_hashes": _version_source_hashes(),
        "scripted_policy_uses_preferences": False,
        "scripted_policy_uses_private_memory": False,
        "run_mode": "full_session" if full_session else "smoke",
        "backend": cfg.backend,
        "treatment": treatment.value,
        "router": "ReActRouter",
        "router_model": cfg.model,
        "router_api_enabled": cfg.backend == "llm",
        "observation_routing": "direct_via_trusted_adapter",
        "transfer_routing": "react_router",
        "insurance_decision_count": len(insurance_decisions),
        "insurance_decisions_file": (
            "insurance_decisions.jsonl"
            if treatment is protocol_spec.Treatment.VOLUNTARY_INSURANCE_25 else None
        ),
        "insurance_assignment_count": len(insurance_assignments),
        "insurance_assignments_file": (
            "insurance_assignments.jsonl"
            if treatment is protocol_spec.Treatment.FORCED_INSURANCE_25 else None
        ),
        "insurance_assignment_configuration": env.assignment_configuration,
        "finish_after_receipt": "local",
        "router_model_requests": sum(client.actual_calls for client in router_clients),
        "participant_model_requests": sum(client.actual_calls for client in participant_clients),
        "agent_class": "FixedPreferenceMemoryParticipant",
        "agents": participants,
        "pairs": len(env.pair_members),
        "seed": seed,
        "model": cfg.model,
        "base_url": cfg.base_url,
        "thinking": "disabled",
        "participant_deliberation": participant_deliberation,
        "participant_deliberation_requests": sum(client.deliberation_calls for client in participant_clients),
        "participant_deliberation_file": "participant_deliberations.jsonl" if participant_deliberation else None,
        "deliberation_prompt_sha256": (
            _file_hash(PROJECT_ROOT / "custom" / "agents" / "risk_sharing_deliberation.md")
            if participant_deliberation else None
        ),
        "request_limit": request_limit,
        "max_output_tokens": max_output_tokens,
        "router_max_steps": router_steps,
        "agent_max_turns": agent_turns,
        "agent_concurrency": agent_concurrency,
        "actual_model_requests": budget.calls,
        "actual_input_tokens": budget.input_tokens,
        "actual_output_tokens": budget.output_tokens,
        "completed_periods": list(env.completed_periods),
        "natural_termination": env.finished,
        "payment_period": payment_period,
        "payment_file": payment_path.name if payment_path else None,
        "stop_reason": stop_reason,
        "success": successful,
        "run_dir": str(run_dir),
        "failure_details": failure_details,
        "log_file": "run.log",
        "prompt_sha256": _file_hash(PROJECT_ROOT / "custom" / "agents" / VERSION / "participant.md"),
        "insurance_prompt_sha256": (
            _file_hash(PROJECT_ROOT / "custom" / "agents" / "risk_sharing_insurance.md")
            if treatment is protocol_spec.Treatment.VOLUNTARY_INSURANCE_25 else None
        ),
        "environment": "shared in-process RiskSharingEnv via RiskSharingRouterAdapter",
    }
    _write_manifest(run_dir / "manifest.json", manifest)

    summary_lines = [
        f"# {VERSION} / {treatment.value} {'full session' if full_session else 'smoke'} ({cfg.backend})",
        "",
        "- 固定偏好：风险厌恶、优势不平等厌恶；启用参与者独立的持久私人记忆。",
        "- 记忆由本人 participant model 更新；框架通用记忆关闭，不使用 embedding。",
        "- [私人记忆更新](private_memory_updates.jsonl) · 状态文件：`agents/agent_XXXX/private_memory.json`",
        f"- private_memory_update_requests: `{sum(client.memory_update_calls for client in participant_clients)}`",
        f"- private_memory_update_count: `{sum(client.memory_update_commits for client in participant_clients)}`",
        f"- private_memory_update_rejections: `{sum(client.memory_update_rejections for client in participant_clients)}`",
        f"- private_memory_update_max_attempts: `{MEMORY_UPDATE_MAX_ATTEMPTS}`（含初次请求，额外修正仍计入请求预算）",
        f"- observation_presentation: `{PRESENTATION_VERSION}`",
        "- [个人偏好配置](participant_profiles.json) · [实际 participant 输入](participant_contexts.jsonl)",
        "- low/medium/high 为研究配置，尚未经人类数据校准。",
        "- scripted policy 沿用基准工程策略，不模拟偏好或私人记忆的经济效应。",
        f"- success: `{successful}`",
        f"- stop_reason: `{stop_reason}`",
        f"- participants: `{participants}`",
        f"- pairs: `{len(env.pair_members)}`",
        f"- completed_periods: `{env.completed_periods}`",
        f"- model_requests: `{budget.calls}/{request_limit}`",
        f"- participant_model_requests: `{sum(client.actual_calls for client in participant_clients)}`",
        f"- router_model_requests: `{sum(client.actual_calls for client in router_clients)}`",
        f"- participant_deliberation: `{participant_deliberation}`",
        f"- participant_deliberation_requests: `{sum(client.deliberation_calls for client in participant_clients)}`",
        "- observation_routing: `direct_via_trusted_adapter`",
        "- finish_after_receipt: `local`",
        f"- agent_concurrency: `{agent_concurrency}`",
        f"- output_tokens: `{budget.output_tokens}`",
        f"- insurance decisions: `{len(insurance_decisions)}`",
        "",
    ]
    if participant_deliberation:
        summary_lines.append(
            "参与者的简短决策评估见 [participant_deliberations.jsonl](participant_deliberations.jsonl)；"
            "最终行动见 [agent_actions.jsonl](agent_actions.jsonl)。评估不等同于模型内部思考记录。\n"
        )
    if treatment is protocol_spec.Treatment.VOLUNTARY_INSURANCE_25:
        report = import_module("experiments.25v.report")
        summary_lines.append(report.render_insurance_section(
            manifest, insurance_decisions, env.history, report.read_jsonl(action_path),
        ))
    elif treatment is protocol_spec.Treatment.FORCED_INSURANCE_25:
        report = import_module("experiments.25f.report")
        summary_lines.append(report.render_insurance_section(
            manifest, insurance_assignments, env.history,
        ))
    summary_lines.extend(["", "## Pair-period settlements", ""])
    for period in sorted({int(r["period"]) for r in env.history}):
        for pair_id in sorted(env.pair_history):
            rows = [r for r in env.pair_history[pair_id] if int(r["period"]) == period]
            if rows:
                transfer = next((r["transfer_sent"] for r in rows if r["had_transfer_opportunity"]), 0)
                final = ", ".join(f"{r['agent_id']}={r['final_income']}" for r in rows)
                summary_lines.append(f"- period {period}, pair {pair_id}: transfer={transfer}; final=({final})")
    if full_session:
        summary_lines.extend(
            [
                "",
                "## Natural termination",
                "",
                f"- finished: `{env.finished}`",
                f"- realized periods: `{len(env.completed_periods)}`",
                f"- payment period: `{payment_period}`",
                f"- payment file: `{payment_path.name if payment_path else None}`",
            ]
        )
    if failure_details:
        summary_lines.extend(["", "## Failure details", ""])
        for item in failure_details:
            summary_lines.append(f"- agent {item.get('agent_id')}: {item.get('error')}")
    summary_lines.extend(
        [
            "",
            "Run diagnostics: `run.log` (JSONL; credential-like values redacted).",
        ]
    )
    if not full_session:
        summary_lines.append(
            "Payment selection is not performed in smoke mode; use `--full-session` "
            "for natural termination and payment-period selection."
        )
    summary_text = "\n".join(summary_lines) + "\n"
    try:
        from experiments.report import add_report_link, write_visual_report

        write_visual_report(
            run_dir, manifest, env.history, insurance_decisions,
            assignments=insurance_assignments,
        )
        summary_text = add_report_link(summary_text)
    except Exception as exc:
        # Presentation must not change an already settled experimental outcome.
        run_log.write("visual_report_failed", error=_redact_text(exc))
        summary_text += "\n交互报告未生成；文字记录完整，原因见 run.log。\n"
    run_log.write(
        "run_finished",
        success=successful,
        stop_reason=stop_reason,
        completed_periods=list(env.completed_periods),
        model_requests=budget.calls,
        router_model_requests=sum(client.actual_calls for client in router_clients),
        participant_model_requests=sum(client.actual_calls for client in participant_clients),
        failure_details=failure_details,
    )
    (run_dir / "summary.md").write_text(summary_text, encoding="utf-8")
    return manifest


def _version_source_hashes() -> dict[str, str]:
    paths = (
        "experiments/fixed_preferences_memory_v1/run_smoke.py",
        "custom/agents/fixed_preferences_v1/profiles.py",
        "custom/agents/fixed_preferences_v1/participant.py",
        "custom/agents/fixed_preferences_memory_v1/participant.py",
        "custom/agents/fixed_preferences_memory_v1/state.py",
        "custom/agents/fixed_preferences_memory_v1/memory_update.md",
        "custom/agents/fixed_preferences_memory_v1/memory_usage.md",
        "custom/agents/observation_presentation.py",
        "custom/agents/fixed_preferences_v1/participant.md",
        "custom/agents/fixed_preferences_memory_v1/participant.md",
        "custom/agents/risk_sharing_participant.py",
        "custom/agents/risk_sharing_insurance.md",
        "custom/agents/risk_sharing_deliberation.md",
        "custom/envs/risk_sharing_env.py",
        "custom/insurance_assignment.py",
        "custom/config.py",
        "protocol/paper_spec.py",
    )
    return {path: _file_hash(PROJECT_ROOT / path) for path in paths}


def run_experiment(**kwargs: Any) -> dict[str, Any]:
    """Run a bounded smoke or a natural-termination session."""

    if "observation_presentation" in kwargs:
        raise TypeError("observation_presentation is not a configurable run option")
    participants = kwargs.get("participants", 4)
    agent_concurrency = kwargs.get("agent_concurrency", 1)
    _validate_population(participants, agent_concurrency)
    profiles, preference_assignment = build_profiles(
        participants, kwargs.get("preference_seed", 2026), kwargs.get("profiles_file"),
    )
    preference_assignment = {**preference_assignment, "version": VERSION}
    treatment = protocol_spec.Treatment(kwargs.get("treatment", protocol_spec.Treatment.BASELINE))
    assignment_probabilities = validate_assignment_configuration(
        treatment, kwargs.get("forced_assignment_probabilities"), kwargs.get("assignment_seed"),
    )
    backend = str(kwargs.get("backend", "scripted"))
    cfg = load_runtime_config(
        backend,
        env_path=kwargs.get("env_path"),
        request_timeout=float(kwargs.get("request_timeout", 60.0)),
    )
    run_dir = _make_run_dir(kwargs.get("run_dir"), treatment)
    _write_manifest(run_dir / "participant_profiles.json", {
        "schema_version": 1, "profiles": profiles,
    })
    (run_dir / "baseline_reference.json").write_bytes(
        Path(__file__).with_name("baseline_reference.json").read_bytes()
    )
    try:
        timeout_seconds = float(
            kwargs.get("timeout_seconds", DEFAULT_EXPERIMENT_TIMEOUT_SECONDS)
        )
        coro = _run_async(
            cfg=cfg,
            treatment=treatment,
            participants=participants,
            completed_period_limit=int(kwargs.get(
                "completed_period_limit", 3 if treatment is protocol_spec.Treatment.BASELINE else 31,
            )),
            full_session=bool(kwargs.get("full_session", False)),
            seed=int(kwargs.get("seed", 2026)),
            run_dir=run_dir,
            request_limit=int(kwargs.get(
                "request_limit", 80 if treatment is protocol_spec.Treatment.BASELINE else 400,
            )),
            max_output_tokens=int(kwargs.get("max_output_tokens", 2048)),
            router_steps=int(kwargs.get("router_steps", 6)),
            agent_turns=int(kwargs.get("agent_turns", 4)),
            agent_concurrency=agent_concurrency,
            timeout_seconds=timeout_seconds,
            participant_deliberation=bool(kwargs.get("participant_deliberation", False)),
            forced_assignment_probabilities=assignment_probabilities,
            assignment_seed=kwargs.get("assignment_seed"),
            profiles=profiles,
            preference_assignment=preference_assignment,
        )
        return asyncio.run(asyncio.wait_for(coro, timeout=timeout_seconds))
    except Exception as exc:
        safe_error = _redact_text(exc)
        _append_run_log(
            run_dir / "run.log",
            "run_exception",
            error_type=type(exc).__name__,
            error=safe_error,
        )
        manifest = {
            "schema_version": 1,
            "experiment_version": VERSION,
            "observation_presentation": PRESENTATION_VERSION,
            "observation_presentation_sha256": _file_hash(PROJECT_ROOT / "custom/agents/observation_presentation.py"),
            "enable_memory": False,
            "framework_memory_enabled": False,
            "private_memory_enabled": True,
            "memory_implementation": "participant_workspace_json",
            "private_memory_update_max_attempts": MEMORY_UPDATE_MAX_ATTEMPTS,
            "private_memory_updates_file": "private_memory_updates.jsonl",
            "preferences_file": "participant_profiles.json",
            "preferences_sha256": _file_hash(run_dir / "participant_profiles.json"),
            "preference_assignment": preference_assignment,
            "backend": backend,
            "treatment": treatment.value,
            "success": False,
            "stop_reason": "configuration_or_runtime_error",
            "error_type": type(exc).__name__,
            "error": safe_error,
            "log_file": "run.log",
            "run_dir": str(run_dir),
        }
        _write_manifest(run_dir / "manifest.json", manifest)
        (run_dir / "summary.md").write_text(
            f"# {treatment.value} run ({backend})\n\n- observation_presentation: `{PRESENTATION_VERSION}`\n- success: `False`\n- stop_reason: `configuration_or_runtime_error`\n- error: `{type(exc).__name__}: {safe_error}`\n\n- diagnostics: `run.log`\n",
            encoding="utf-8",
        )
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("scripted", "llm"), required=True)
    parser.add_argument("--treatment", type=str.lower, choices=("baseline", "25v", "25f"), default="baseline")
    parser.add_argument("--preference-seed", type=int, default=2026,
                        help="Independent seed for the hypothetical balanced preference assignment.")
    parser.add_argument("--profiles-file", type=Path,
                        help="Explicit per-agent JSON profiles; overrides generated preferences.")
    parser.add_argument(
        "--forced-assignment-probabilities", nargs=3, type=float,
        metavar=("P_BOTH", "P_FIRST_ONLY", "P_SECOND_ONLY"),
        help="Required for 25F only; paper probabilities remain unresolved, so there is no default.",
    )
    parser.add_argument("--assignment-seed", type=int,
                        help="25F assignment RNG seed, independent of the preference assignment.")
    parser.add_argument(
        "--participants", type=int, default=4,
        help="Even participant count, at least 2 (default: 4; use 20 for ten fixed pairs).",
    )
    parser.add_argument(
        "--completed-period-limit",
        type=int,
        default=3,
        help="Smoke-mode target; ignored by --full-session.",
    )
    parser.add_argument(
        "--full-session",
        action="store_true",
        help="Run until the environment reaches the paper's natural termination.",
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--request-limit", type=int, default=80)
    parser.add_argument("--max-output-tokens", type=int, default=2048)
    parser.add_argument("--router-max-steps", type=int, default=6)
    parser.add_argument("--agent-max-turns", type=int, default=4)
    parser.add_argument(
        "--participant-deliberation", action="store_true",
        help=("Add one private participant review before each transfer or insurance choice. "
              "Uses extra requests; does not enable provider-native thinking or persistent memory."),
    )
    parser.add_argument(
        "--agent-concurrency",
        type=int,
        default=1,
        help=(
            "Maximum number of participant Agents stepped concurrently per "
            "period, from 1 to --participants (default: 1; use 20 with 20 participants)."
        ),
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=DEFAULT_EXPERIMENT_TIMEOUT_SECONDS,
        help="Whole-experiment time limit in seconds (default: 21600, or 6 hours).",
    )
    parser.add_argument("--request-timeout", type=float, default=60.0)
    return parser


def main(
    argv: list[str] | None = None,
) -> int:
    parser = _parser()
    selected, _ = parser.parse_known_args(argv)
    treatment = {"baseline": protocol_spec.Treatment.BASELINE,
                 "25v": protocol_spec.Treatment.VOLUNTARY_INSURANCE_25,
                 "25f": protocol_spec.Treatment.FORCED_INSURANCE_25}[selected.treatment]
    if treatment is protocol_spec.Treatment.VOLUNTARY_INSURANCE_25:
        parser.description = "Run the 25V voluntary-insurance experiment with fixed pairs."
        parser.set_defaults(completed_period_limit=31, request_limit=400)
    elif treatment is protocol_spec.Treatment.FORCED_INSURANCE_25:
        parser.description = "Run 25F with an explicit computer insurance-assignment distribution."
        parser.set_defaults(completed_period_limit=31, request_limit=400)
    args = parser.parse_args(argv)
    try:
        _validate_population(args.participants, args.agent_concurrency)
        build_profiles(args.participants, args.preference_seed, args.profiles_file)
        validate_assignment_configuration(
            treatment, getattr(args, "forced_assignment_probabilities", None),
            getattr(args, "assignment_seed", None),
        )
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    try:
        cfg = load_runtime_config(
            args.backend,
            env_path=DEFAULT_ENV_PATH,
            request_timeout=args.request_timeout,
        )
    except RuntimeConfigError as exc:
        print(f"配置未就绪：{exc}", file=sys.stderr)
        return 2
    print(json.dumps({"config": cfg.safe_status(), "limits": {"requests": args.request_limit, "max_output_tokens": args.max_output_tokens, "timeout_seconds": args.timeout_seconds, "full_session": args.full_session, "agent_concurrency": args.agent_concurrency}}, ensure_ascii=False))
    try:
        manifest = run_experiment(
            treatment=treatment,
            backend=args.backend,
            participants=args.participants,
            completed_period_limit=args.completed_period_limit,
            full_session=args.full_session,
            seed=args.seed,
            run_dir=args.run_dir,
            request_limit=args.request_limit,
            max_output_tokens=args.max_output_tokens,
            router_steps=args.router_max_steps,
            agent_turns=args.agent_max_turns,
            agent_concurrency=args.agent_concurrency,
            participant_deliberation=args.participant_deliberation,
            timeout_seconds=args.timeout_seconds,
            request_timeout=args.request_timeout,
            forced_assignment_probabilities=getattr(args, "forced_assignment_probabilities", None),
            assignment_seed=getattr(args, "assignment_seed", None),
            preference_seed=args.preference_seed,
            profiles_file=args.profiles_file,
        )
    except Exception as exc:
        print(f"运行失败：{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({"experiment_version": VERSION, "success": manifest.get("success"), "stop_reason": manifest.get("stop_reason"), "run_dir": manifest.get("run_dir")}, ensure_ascii=False))
    return 0 if manifest.get("success") else 1


if __name__ == "__main__":
    raise SystemExit(main())






