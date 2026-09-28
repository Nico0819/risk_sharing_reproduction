"""Run the bounded Baseline participant smoke experiment.

The scripted backend exercises the real PersonAgent -> adapter -> ReActRouter
-> tool -> RiskSharingEnv path without network access.  The deepseek backend
uses the same wiring after loading the project-root .env.
"""

from __future__ import annotations

import argparse
import copy
import asyncio
import hashlib
import json
import re
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


class BudgetedClient:
    """Apply run limits and optional bounded Router action progression.

    The progression guard is deliberately small: after the Router receives a
    current observation, it exposes only the action that is valid for that
    observation. It does not manufacture a decision, choose an amount, or
    call an environment method; the normal ReActRouter and adapter still do
    all dispatch, identity, and validation work.
    """

    def __init__(
        self,
        inner: Any,
        budget: RequestBudget,
        max_output_tokens: int,
        *,
        enforce_router_progression: bool = False,
        enforce_participant_progression: bool = False,
    ):
        self.inner = inner
        self.budget = budget
        self.max_output_tokens = max_output_tokens
        self.model_name = getattr(inner, "model_name", "")
        self.enforce_router_progression = enforce_router_progression
        self.enforce_participant_progression = enforce_participant_progression
        self._router_tool_schemas: list[dict[str, Any]] = []
        self._participant_tool_schemas: list[dict[str, Any]] = []

    @staticmethod
    def _decode_tool_content(message: Any) -> dict[str, Any]:
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str):
            return {}
        try:
            value = json.loads(content)
        except json.JSONDecodeError:
            return {}
        return value if isinstance(value, dict) else {}

    @staticmethod
    def _router_caller_id(messages: list[dict[str, Any]]) -> int | None:
        pattern = re.compile(r"id[^:]{0,3}:\s*(-?\d+)")
        for message in messages:
            content = message.get("content") if isinstance(message, dict) else None
            if isinstance(content, str):
                match = pattern.search(content)
                if match:
                    return int(match.group(1))
        return None

    @staticmethod
    def _prepared_router_tool(
        tool: dict[str, Any],
        *,
        required_tool: str,
        caller_id: int | None,
    ) -> dict[str, Any]:
        prepared = copy.deepcopy(tool)
        function = prepared.setdefault("function", {})
        parameters = function.setdefault("parameters", {})
        properties = parameters.setdefault("properties", {})
        required = list(parameters.get("required") or [])

        if required_tool in {"get_observation", "submit_decision"}:
            agent_schema = properties.setdefault("agent_id", {"type": "integer"})
            agent_schema.update({"type": "integer"})
            if caller_id is not None:
                agent_schema["enum"] = [caller_id]
            if "agent_id" not in required:
                required.append("agent_id")

        if required_tool == "submit_decision":
            decision_schema = properties.setdefault(
                "decision_type", {"type": "string"}
            )
            decision_schema.update({"type": "string", "enum": ["transfer"]})
            if "decision_type" not in required:
                required.append("decision_type")
            if "decision_id" not in required:
                required.append("decision_id")
            if "amount" not in required:
                required.append("amount")

        parameters["required"] = required
        return prepared

    def _restrict_router_tools(self, messages: list[dict[str, Any]], kwargs: dict[str, Any]) -> None:
        """Constrain Router calls to the current observation/action state."""

        if not self.enforce_router_progression:
            return

        advertised = kwargs.get("tools")
        if advertised:
            self._router_tool_schemas = [
                tool for tool in advertised if isinstance(tool, dict)
            ]
        if not self._router_tool_schemas:
            return

        caller_id = self._router_caller_id(messages)
        latest_tool = next(
            (
                message
                for message in reversed(messages)
                if isinstance(message, dict) and message.get("role") == "tool"
            ),
            None,
        )

        if latest_tool is None:
            first_tool = [
                self._prepared_router_tool(
                    tool,
                    required_tool="get_observation",
                    caller_id=caller_id,
                )
                for tool in self._router_tool_schemas
                if isinstance(tool.get("function"), dict)
                and tool["function"].get("name") == "get_observation"
            ]
            if first_tool:
                kwargs["tools"] = first_tool
                kwargs["tool_choice"] = "required"
            return

        latest_name = latest_tool.get("name")
        observation = self._decode_tool_content(latest_tool)
        if latest_name == "get_observation":
            readonly = any(
                isinstance(message.get("content"), str)
                and "READONLY MODE" in message["content"]
                for message in messages
                if isinstance(message, dict)
            )
            if observation.get("error"):
                required_tool = "get_observation"
            elif readonly:
                required_tool = "set_status"
            elif observation.get("task") == "transfer":
                required_tool = "submit_decision"
            else:
                required_tool = "set_status"
        elif latest_name == "submit_decision":
            # A failed submit may be corrected once with the same observation;
            # a successful submit only needs the Router's normal completion.
            required_tool = "submit_decision" if observation.get("error") else "set_status"
        else:
            return

        restricted = [
            self._prepared_router_tool(
                tool,
                required_tool=required_tool,
                caller_id=caller_id,
            )
            for tool in self._router_tool_schemas
            if isinstance(tool.get("function"), dict)
            and tool["function"].get("name") == required_tool
        ]
        if restricted:
            kwargs["tools"] = restricted
            kwargs["tool_choice"] = "required"

    def _restrict_participant_tools(self, messages: list[dict[str, Any]], kwargs: dict[str, Any]) -> None:
        """Give each PersonAgent turn one environment action, then finish."""

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
            if (
                previous_function.get("name") == "ask_env"
                and bool(previous_args.get("readonly"))
            ):
                allowed_name = "ask_env"
            else:
                allowed_name = "finish"
        else:
            allowed_name = "ask_env"

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
        kwargs["tool_choice"] = "required"

    @staticmethod
    def _normalize_participant_response(response: Any) -> Any:
        """Prevent an extra readonly ask_env turn in this bounded runner."""

        choices = getattr(response, "choices", None) or []
        for choice in choices:
            message = getattr(choice, "message", None)
            for tool_call in getattr(message, "tool_calls", None) or []:
                function = getattr(tool_call, "function", None)
                if getattr(function, "name", None) != "ask_env":
                    continue
                raw_arguments = getattr(function, "arguments", "{}") or "{}"
                try:
                    arguments = json.loads(raw_arguments)
                except (TypeError, json.JSONDecodeError):
                    continue
                if isinstance(arguments, dict) and arguments.get("readonly") is True:
                    arguments["readonly"] = False
                    function.arguments = json.dumps(
                        arguments, ensure_ascii=False
                    )
        return response

    async def call(self, model: str | None = None, messages=None, **kwargs: Any):
        self.budget.reserve()
        messages = list(messages or [])
        self._restrict_router_tools(messages, kwargs)
        self._restrict_participant_tools(messages, kwargs)
        kwargs.setdefault("max_output_tokens", self.max_output_tokens)
        # LLMClient clamps this to one actual attempt; no unbounded retry loop.
        kwargs["max_retries"] = 0
        response = await self.inner.call(model=model, messages=messages, **kwargs)
        if self.enforce_participant_progression:
            response = self._normalize_participant_response(response)
        self.budget.record_usage(response)
        return response

    def take_token_stats(self) -> dict[str, dict[str, int]]:
        getter = getattr(self.inner, "take_token_stats", None)
        return getter() if getter else {}

    def snapshot_token_stats(self) -> dict[str, dict[str, int]]:
        getter = getattr(self.inner, "snapshot_token_stats", None)
        return getter() if getter else {}


class ScriptedLLMClient:
    """Deterministic model boundary; it never performs network I/O."""

    def __init__(self, kind: str):
        self.kind = kind
        self.model_name = "scripted-model"
        self.calls = 0

    @staticmethod
    def _tool_names(tools: Any) -> set[str]:
        names: set[str] = set()
        for tool in tools or []:
            try:
                names.add(str(tool["function"]["name"]))
            except (KeyError, TypeError):
                continue
        return names

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
    def _decode_tool_content(message: dict[str, Any]) -> dict[str, Any]:
        raw = message.get("content") or "{}"
        try:
            value = json.loads(raw)
            return value if isinstance(value, dict) else {}
        except (TypeError, json.JSONDecodeError):
            return {}

    @staticmethod
    def _caller_id(messages: Iterable[dict[str, Any]]) -> int:
        """Read the framework identity from ReActRouter's initial ctx text.

        The adapter remains the authority: this value is only used to make a
        deterministic model response, while the adapter validates it against
        PersonAgent's trusted context before dispatch.
        """

        pattern = re.compile(r"['\"]id['\"]\s*[:=]\s*(-?\d+)")
        for message in messages:
            content = message.get("content") if isinstance(message, dict) else None
            if not isinstance(content, str):
                continue
            match = pattern.search(content)
            if match:
                return int(match.group(1))
        raise RuntimeError("scripted router could not recover the framework caller id")

    async def call(self, model: str | None = None, messages=None, **kwargs: Any) -> Any:
        self.calls += 1
        messages = list(messages or [])
        tools = kwargs.get("tools")
        names = self._tool_names(tools)
        tool_messages = self._tool_messages(messages)
        if self.kind == "summary":
            return self._response(
                content=json.dumps(
                    {"status": "success", "summary": "已完成环境请求。"},
                    ensure_ascii=False,
                )
            )
        caller_id = self._caller_id(messages) if self.kind == "router" else None
        readonly_request = any(
            isinstance(m.get("content"), str) and "READONLY MODE" in m["content"]
            for m in messages
            if isinstance(m, dict)
        )

        if self.kind == "participant":
            if tool_messages:
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
            return self._response(
                name="ask_env",
                arguments={
                    "instruction": "读取当前 pair observation；若我有转账机会，选择合法整数并提交 transfer，否则等待。",
                    "readonly": False,
                },
            )

        # ReActRouter's first request advertises environment tools.  Subsequent
        # requests omit schemas but retain named tool result messages.
        if not tool_messages and "get_observation" in names:
            return self._response(name="get_observation", arguments={"agent_id": caller_id})
        if tool_messages:
            last = tool_messages[-1]
            last_name = str(last.get("name") or "")
            value = self._decode_tool_content(last)
            if last_name == "get_observation":
                if (not readonly_request) and value.get("task") == "transfer" and value.get("decision_id"):
                    return self._response(
                        name="submit_decision",
                        arguments={
                            "agent_id": caller_id,
                            "decision_id": value["decision_id"],
                            "decision_type": "transfer",
                            "amount": 0,
                        },
                    )
                return self._response(
                    name="set_status",
                    arguments={"status": "success", "reason": "waiting"},
                )
            if last_name in {"submit_decision", "set_status"}:
                if last_name == "submit_decision":
                    return self._response(
                        name="set_status",
                        arguments={"status": "success", "reason": "decision recorded"},
                    )
                return self._response(content="environment request completed")
            if value.get("error"):
                return self._response(content="environment request failed")
        return self._response(content="environment request completed")


def _jsonl_append(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _make_run_dir(path: Path | None) -> Path:
    if path is not None:
        run_dir = path.expanduser().resolve()
        if run_dir.exists():
            raise RuntimeConfigError(f"run directory already exists: {run_dir}")
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_dir = PROJECT_ROOT / "experiments" / "baseline" / "runs" / f"{stamp}-{uuid.uuid4().hex[:8]}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _write_manifest(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def _safe_action_records(records: list[dict[str, Any]], env: Any, framework_step: int) -> list[dict[str, Any]]:
    safe: list[dict[str, Any]] = []
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
        elif function_name == "submit_decision":
            row["model_choice"] = {"decision_type": kwargs.get("decision_type"), "amount": kwargs.get("amount")}
            row["submitted_amount"] = kwargs.get("amount")
            row["decision_id"] = kwargs.get("decision_id")
            row["receipt"] = record.get("return_value")
        if record.get("exception_occurred"):
            row["error"] = record.get("exception_info")
        safe.append(row)
    return safe


async def _run_async(
    *,
    cfg: RuntimeConfig,
    participants: int,
    completed_period_limit: int,
    seed: int,
    run_dir: Path,
    request_limit: int,
    max_output_tokens: int,
    router_steps: int,
    agent_turns: int,
    timeout_seconds: float,
) -> dict[str, Any]:
    run_log = RunLogger(run_dir / "run.log")
    run_log.write(
        "configuration_loaded",
        backend=cfg.backend,
        config_status=cfg.safe_status(),
        participants=participants,
        completed_period_limit=completed_period_limit,
        request_limit=request_limit,
        max_output_tokens=max_output_tokens,
        timeout_seconds=timeout_seconds,
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
    from agentsociety2.env import ReActRouter
    from custom.agents.risk_sharing_participant import RiskSharingParticipant
    from custom.envs.risk_sharing_env import RiskSharingEnv, RiskSharingRouterAdapter

    if participants != 4:
        raise ValueError("this smoke entry is fixed to four participants / two pairs")
    if completed_period_limit < 1:
        raise ValueError("completed_period_limit must be positive")

    budget = RequestBudget(max_calls=request_limit)
    env = RiskSharingEnv(
        list(range(1, participants + 1)),
        seed=seed,
        session_id=f"smoke-{uuid.uuid4().hex}",
    )
    router = ReActRouter([env], max_steps=router_steps, max_llm_call_retry=1)
    if cfg.backend == "scripted":
        router_client = BudgetedClient(ScriptedLLMClient("router"), budget, max_output_tokens, enforce_router_progression=True)
        summary_client = BudgetedClient(ScriptedLLMClient("summary"), budget, max_output_tokens)
        participant_client = BudgetedClient(ScriptedLLMClient("participant"), budget, max_output_tokens, enforce_participant_progression=True)
    else:
        router_client = BudgetedClient(build_client_for_role("coder"), budget, max_output_tokens, enforce_router_progression=True)
        summary_client = BudgetedClient(build_client_for_role("default"), budget, max_output_tokens)
        participant_client = BudgetedClient(build_client_for_role("default"), budget, max_output_tokens, enforce_participant_progression=True)
    router._coder_dispatcher = router_client
    router._summary_dispatcher = summary_client
    adapter = RiskSharingRouterAdapter(router, env)
    proxy = ServiceProxy(
        env=adapter,
        llm=LLMClients(coder=router_client, default=participant_client, embedding=None),
        trace=None,
        replay=None,
        run_dir=run_dir,
    )
    run_log.write(
        "framework_ready",
        router="ReActRouter",
        agent_class="RiskSharingParticipant",
        pairs=len(env.pair_members),
    )

    workspace_root = run_dir / "agents"
    agent_config = {
        "enable_memory": False,
        "enable_todo_list": False,
        "max_react_turns": agent_turns,
        "disabled_skill_ids": ["*"],
    }
    for agent_id in range(1, participants + 1):
        RiskSharingParticipant.create(
            workspace_root / f"agent_{agent_id:04d}",
            {"id": agent_id, "name": f"participant-{agent_id}"},
            agent_config,
        )

    # Use the framework's real workspace batch runner and class restoration.
    from agentsociety2.registry import get_registry

    get_registry().register_agent_module(
        "RiskSharingParticipant", RiskSharingParticipant, is_custom=True
    )

    action_path = run_dir / "agent_actions.jsonl"
    records_path = run_dir / "period_records.jsonl"
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    start_history = 0
    stop_reason = "unknown"
    scheduler_steps = 0
    result_batches: list[dict[str, Any]] = []
    failure_details: list[dict[str, Any]] = []
    max_scheduler_steps = max(3, completed_period_limit * 4)
    started = time.monotonic()
    while len(env.completed_periods) < completed_period_limit:
        if scheduler_steps >= max_scheduler_steps:
            stop_reason = "scheduler_step_limit"
            run_log.write("stop_requested", reason=stop_reason, scheduler_step=scheduler_steps)
            break
        scheduler_steps += 1
        run_log.write(
            "scheduler_step_started",
            scheduler_step=scheduler_steps,
            current_period=env.current_period,
            completed_periods=list(env.completed_periods),
        )
        adapter.set_current_time(now)
        # The framework runner is still used for every PersonAgent; the
        # scripted model boundary is kept sequential so its deterministic
        # response cannot mix two concurrent ReAct conversations.
        batch = []
        for agent_id in range(1, participants + 1):
            batch.extend(
                await _step_agent_batch_async(
                    [agent_id],
                    str(workspace_root),
                    "RiskSharingParticipant",
                    scheduler_steps,
                    now,
                    proxy,
                )
            )
        result_batches.extend(batch)
        step_actions = _safe_action_records(
            env.get_tool_call_history()[start_history:], env, scheduler_steps
        )
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
        await adapter.step(1, now)
        after = len(env.completed_periods)
        run_log.write(
            "scheduler_step_finished",
            scheduler_step=scheduler_steps,
            period_before=env.current_period if after == before else env.current_period - 1,
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
        stop_reason = "smoke_period_limit"

    # Records come directly from the environment, never from model text.
    _jsonl_append(records_path, env.history)
    valid_decisions = [
        item for item in env.get_tool_call_history()
        if item.get("function_name") == "submit_decision" and not item.get("exception_occurred")
    ]
    unique_decisions = {str(item.get("kwargs", {}).get("decision_id")) for item in valid_decisions}
    successful = (
        stop_reason == "smoke_period_limit"
        and len(env.completed_periods) >= completed_period_limit
        and len(unique_decisions) == 2 * completed_period_limit
        and len(env.history) == 4 * completed_period_limit
    )
    if not successful and stop_reason == "smoke_period_limit":
        stop_reason = "incomplete_records"

    manifest = {
        "schema_version": 1,
        "backend": cfg.backend,
        "router": "ReActRouter",
        "agent_class": "RiskSharingParticipant",
        "agents": participants,
        "pairs": len(env.pair_members),
        "seed": seed,
        "model": cfg.model,
        "base_url": cfg.base_url,
        "thinking": "disabled",
        "request_limit": request_limit,
        "max_output_tokens": max_output_tokens,
        "router_max_steps": router_steps,
        "agent_max_turns": agent_turns,
        "actual_model_requests": budget.calls,
        "actual_input_tokens": budget.input_tokens,
        "actual_output_tokens": budget.output_tokens,
        "completed_periods": list(env.completed_periods),
        "stop_reason": stop_reason,
        "success": successful,
        "run_dir": str(run_dir),
        "failure_details": failure_details,
        "log_file": "run.log",
        "prompt_sha256": _file_hash(PROJECT_ROOT / "custom" / "agents" / "risk_sharing_participant.md"),
        "environment": "shared in-process RiskSharingEnv via RiskSharingRouterAdapter",
    }
    _write_manifest(run_dir / "manifest.json", manifest)

    summary_lines = [
        f"# Baseline smoke ({cfg.backend})",
        "",
        f"- success: `{successful}`",
        f"- stop_reason: `{stop_reason}`",
        f"- completed_periods: `{env.completed_periods}`",
        f"- model_requests: `{budget.calls}/{request_limit}`",
        f"- output_tokens: `{budget.output_tokens}`",
        "",
        "## Pair-period settlements",
        "",
    ]
    for period in sorted({int(r["period"]) for r in env.history}):
        for pair_id in sorted(env.pair_history):
            rows = [r for r in env.pair_history[pair_id] if int(r["period"]) == period]
            if rows:
                transfer = next((r["transfer_sent"] for r in rows if r["had_transfer_opportunity"]), 0)
                final = ", ".join(f"{r['agent_id']}={r['final_income']}" for r in rows)
                summary_lines.append(f"- period {period}, pair {pair_id}: transfer={transfer}; final=({final})")
    if failure_details:
        summary_lines.extend(["", "## Failure details", ""])
        for item in failure_details:
            summary_lines.append(f"- agent {item.get('agent_id')}: {item.get('error')}")
    summary_lines.extend([
        "",
        "Run diagnostics: `run.log` (JSONL; credential-like values redacted).",
        "Payment selection is not implemented; this is an externally truncated engineering smoke run, not a completed paper payment experiment.",
    ])
    run_log.write(
        "run_finished",
        success=successful,
        stop_reason=stop_reason,
        completed_periods=list(env.completed_periods),
        model_requests=budget.calls,
        failure_details=failure_details,
    )
    (run_dir / "summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    return manifest


def run_experiment(**kwargs: Any) -> dict[str, Any]:
    """Run one bounded smoke experiment and return the non-secret manifest."""

    backend = str(kwargs.get("backend", "scripted"))
    cfg = load_runtime_config(
        backend,
        env_path=kwargs.get("env_path"),
        request_timeout=float(kwargs.get("request_timeout", 60.0)),
    )
    run_dir = _make_run_dir(kwargs.get("run_dir"))
    try:
        coro = _run_async(
            cfg=cfg,
            participants=int(kwargs.get("participants", 4)),
            completed_period_limit=int(kwargs.get("completed_period_limit", 3)),
            seed=int(kwargs.get("seed", 2026)),
            run_dir=run_dir,
            request_limit=int(kwargs.get("request_limit", 80)),
            max_output_tokens=int(kwargs.get("max_output_tokens", 2048)),
            router_steps=int(kwargs.get("router_steps", 6)),
            agent_turns=int(kwargs.get("agent_turns", 4)),
            timeout_seconds=float(kwargs.get("timeout_seconds", 900.0)),
        )
        return asyncio.run(asyncio.wait_for(coro, timeout=float(kwargs.get("timeout_seconds", 900.0))))
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
            "backend": backend,
            "success": False,
            "stop_reason": "configuration_or_runtime_error",
            "error_type": type(exc).__name__,
            "error": safe_error,
            "log_file": "run.log",
            "run_dir": str(run_dir),
        }
        _write_manifest(run_dir / "manifest.json", manifest)
        (run_dir / "summary.md").write_text(
            f"# Baseline smoke ({backend})\n\n- success: `False`\n- stop_reason: `configuration_or_runtime_error`\n- error: `{type(exc).__name__}: {safe_error}`\n\n- diagnostics: `run.log`\n",
            encoding="utf-8",
        )
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("scripted", "deepseek"), required=True)
    parser.add_argument("--participants", type=int, default=4)
    parser.add_argument("--completed-period-limit", type=int, default=3)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--request-limit", type=int, default=80)
    parser.add_argument("--max-output-tokens", type=int, default=2048)
    parser.add_argument("--router-max-steps", type=int, default=6)
    parser.add_argument("--agent-max-turns", type=int, default=4)
    parser.add_argument("--timeout-seconds", type=float, default=900.0)
    parser.add_argument("--request-timeout", type=float, default=60.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        cfg = load_runtime_config(
            args.backend,
            env_path=DEFAULT_ENV_PATH,
            request_timeout=args.request_timeout,
        )
    except RuntimeConfigError as exc:
        print(f"配置未就绪：{exc}", file=sys.stderr)
        return 2
    print(json.dumps({"config": cfg.safe_status(), "limits": {"requests": args.request_limit, "max_output_tokens": args.max_output_tokens, "timeout_seconds": args.timeout_seconds}}, ensure_ascii=False))
    try:
        manifest = run_experiment(
            backend=args.backend,
            participants=args.participants,
            completed_period_limit=args.completed_period_limit,
            seed=args.seed,
            run_dir=args.run_dir,
            request_limit=args.request_limit,
            max_output_tokens=args.max_output_tokens,
            router_steps=args.router_max_steps,
            agent_turns=args.agent_max_turns,
            timeout_seconds=args.timeout_seconds,
            request_timeout=args.request_timeout,
        )
    except Exception as exc:
        print(f"运行失败：{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({"success": manifest.get("success"), "stop_reason": manifest.get("stop_reason"), "run_dir": manifest.get("run_dir", str(args.run_dir) if args.run_dir else "generated under experiments/baseline/runs")}, ensure_ascii=False))
    return 0 if manifest.get("success") else 1


if __name__ == "__main__":
    raise SystemExit(main())






