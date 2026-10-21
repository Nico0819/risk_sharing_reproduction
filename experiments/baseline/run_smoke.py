"""Run the Baseline participant smoke or full-session experiment.

The scripted backend exercises the real PersonAgent -> participant model ->
environment adapter -> ReActRouter -> RiskSharingEnv path without network access. The llm
backend uses the same wiring after loading the project-root .env.
"""

from __future__ import annotations

import argparse
import copy
import asyncio
import hashlib
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
from protocol import paper_spec as protocol_spec  # noqa: E402


DEFAULT_EXPERIMENT_TIMEOUT_SECONDS = 6 * 60 * 60


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
            and request.get("tool") in {"get_observation", "submit_decision"}
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
        agent_id: int | None = None,
    ):
        self.inner = inner
        self.budget = budget
        self.max_output_tokens = max_output_tokens
        self.model_name = getattr(inner, "model_name", "")
        self.enforce_participant_progression = enforce_participant_progression
        self.enforce_router_progression = enforce_router_progression
        self.actual_calls = 0
        self.call_logger = call_logger
        self.agent_id = agent_id
        self._participant_tool_schemas: list[dict[str, Any]] = []
        self._router_tool_schemas: list[dict[str, Any]] = []

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
            if (
                previous_function.get("name") == "ask_env"
                and (
                    bool(previous_args.get("readonly"))
                    or str(latest_tool.get("content", "")).startswith("ERROR:")
                )
            ):
                allowed_name = "ask_env"
            else:
                allowed_name = "finish"
        else:
            # PersonAgent.step() has already supplied the readonly observation
            # in the initial participant prompt.  Permit the model to choose
            # either its structured decision or a waiting finish.
            has_initial_observation = any(
                isinstance(message.get("content"), str)
                and '"observations": [' in message["content"]
                and "<risk_sharing_turn>" in message["content"]
                for message in messages
                if isinstance(message, dict)
            )
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
        if self.enforce_router_progression and self._restrict_router_tools(messages, kwargs):
            message = SimpleNamespace(content="Environment request completed.", tool_calls=[])
            return SimpleNamespace(choices=[SimpleNamespace(message=message)])
        self.budget.reserve()
        request_index = self.budget.calls
        self.actual_calls += 1
        self._restrict_participant_tools(messages, kwargs)
        kwargs.setdefault("max_output_tokens", self.max_output_tokens)
        # LLMClient clamps this to one actual attempt; no unbounded retry loop.
        kwargs["max_retries"] = 0
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
                    request_index=request_index,
                    duration_seconds=round(time.monotonic() - started, 6),
                    ok=error_type is None,
                    error_type=error_type,
                )
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
            for item in reversed(observations):
                value = item.get("observation") if isinstance(item, dict) else None
                if isinstance(value, str):
                    try:
                        observation = json.loads(value)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(observation, dict):
                        return int(payload["agent_id"]), observation
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
        elif function_name == "submit_decision":
            row["router_action"] = "router_forward"
            row["router_model_called"] = True
            row["submitted_amount"] = kwargs.get("amount")
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
                    and item.get("amount") == kwargs.get("amount")
                ),
                None,
            )
            if participant is not None:
                row["decision_source"] = "participant_model"
                row["participant_model_output"] = {
                    "decision_type": participant.get("decision_type"),
                    "decision_id": participant.get("decision_id"),
                    "amount": participant.get("amount"),
                }
                if isinstance(participant.get("participant_observation"), dict):
                    row["participant_model_message"] = {
                        "observation": participant["participant_observation"],
                        "contains_observation": True,
                    }
                row["participant_amount"] = participant.get("amount")
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
) -> dict[str, Any]:
    run_log = RunLogger(run_dir / "run.log")
    run_log.write(
        "configuration_loaded",
        backend=cfg.backend,
        config_status=cfg.safe_status(),
        participants=participants,
        completed_period_limit=completed_period_limit,
        full_session=full_session,
        request_limit=request_limit,
        max_output_tokens=max_output_tokens,
        agent_concurrency=agent_concurrency,
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
    from custom.agents.risk_sharing_participant import RiskSharingParticipant
    from custom.envs.risk_sharing_env import (
        RiskSharingEnv, RiskSharingReActRouter, RiskSharingRouterAdapter,
    )

    if participants != 4:
        raise ValueError("this smoke entry is fixed to four participants / two pairs")
    if completed_period_limit < 1:
        raise ValueError("completed_period_limit must be positive")
    if not 1 <= agent_concurrency <= participants:
        raise ValueError("agent_concurrency must be between 1 and participants")

    budget = RequestBudget(max_calls=request_limit)
    env = RiskSharingEnv(
        list(range(1, participants + 1)),
        seed=seed,
        session_id=f"smoke-{uuid.uuid4().hex}",
    )
    participant_clients: list[BudgetedClient] = []
    router_clients: list[BudgetedClient] = []

    call_log = RunLogger(run_dir / "run.log", announce=False)

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
        agent_class="RiskSharingParticipant",
        pairs=len(env.pair_members),
        agent_concurrency=agent_concurrency,
        isolated_participant_clients=participants,
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
    max_scheduler_steps = (
        max(3, protocol_spec.REAL_MAXIMUM_PERIOD_COUNT * 4)
        if full_session
        else max(3, completed_period_limit * 4)
    )
    started = time.monotonic()
    participant_decision_cursor = 0
    participant_rejection_cursor = 0
    agent_semaphore = asyncio.Semaphore(agent_concurrency)

    async def _step_one_agent(agent_id: int) -> list[dict[str, Any]]:
        async with agent_semaphore:
            _adapter, proxy = agent_services[agent_id]
            return await _step_agent_batch_async(
                [agent_id],
                str(workspace_root),
                "RiskSharingParticipant",
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
        await control_adapter.step(1, now)
        after = len(env.completed_periods)
        run_log.write(
            "scheduler_step_finished",
            scheduler_step=scheduler_steps,
            period_before=period_before,
            period_after=env.current_period,
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
        len(unique_decisions) == 2 * expected_period_count
        and len(env.history) == 4 * expected_period_count
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
        "run_mode": "full_session" if full_session else "smoke",
        "backend": cfg.backend,
        "router": "ReActRouter",
        "router_model": cfg.model,
        "router_api_enabled": cfg.backend == "llm",
        "observation_routing": "direct_via_trusted_adapter",
        "transfer_routing": "react_router",
        "finish_after_receipt": "local",
        "router_model_requests": sum(client.actual_calls for client in router_clients),
        "participant_model_requests": sum(client.actual_calls for client in participant_clients),
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
        "prompt_sha256": _file_hash(PROJECT_ROOT / "custom" / "agents" / "risk_sharing_participant.md"),
        "environment": "shared in-process RiskSharingEnv via RiskSharingRouterAdapter",
    }
    _write_manifest(run_dir / "manifest.json", manifest)

    summary_lines = [
        f"# Baseline {'full session' if full_session else 'smoke'} ({cfg.backend})",
        "",
        f"- success: `{successful}`",
        f"- stop_reason: `{stop_reason}`",
        f"- completed_periods: `{env.completed_periods}`",
        f"- model_requests: `{budget.calls}/{request_limit}`",
        f"- participant_model_requests: `{sum(client.actual_calls for client in participant_clients)}`",
        f"- router_model_requests: `{sum(client.actual_calls for client in router_clients)}`",
        "- observation_routing: `direct_via_trusted_adapter`",
        "- finish_after_receipt: `local`",
        f"- agent_concurrency: `{agent_concurrency}`",
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
    (run_dir / "summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    return manifest


def run_experiment(**kwargs: Any) -> dict[str, Any]:
    """Run a bounded smoke or a natural-termination full Baseline session."""

    backend = str(kwargs.get("backend", "scripted"))
    cfg = load_runtime_config(
        backend,
        env_path=kwargs.get("env_path"),
        request_timeout=float(kwargs.get("request_timeout", 60.0)),
    )
    run_dir = _make_run_dir(kwargs.get("run_dir"))
    try:
        timeout_seconds = float(
            kwargs.get("timeout_seconds", DEFAULT_EXPERIMENT_TIMEOUT_SECONDS)
        )
        coro = _run_async(
            cfg=cfg,
            participants=int(kwargs.get("participants", 4)),
            completed_period_limit=int(kwargs.get("completed_period_limit", 3)),
            full_session=bool(kwargs.get("full_session", False)),
            seed=int(kwargs.get("seed", 2026)),
            run_dir=run_dir,
            request_limit=int(kwargs.get("request_limit", 80)),
            max_output_tokens=int(kwargs.get("max_output_tokens", 2048)),
            router_steps=int(kwargs.get("router_steps", 6)),
            agent_turns=int(kwargs.get("agent_turns", 4)),
            agent_concurrency=int(kwargs.get("agent_concurrency", 1)),
            timeout_seconds=timeout_seconds,
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
    parser.add_argument("--backend", choices=("scripted", "llm"), required=True)
    parser.add_argument("--participants", type=int, default=4)
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
        "--agent-concurrency",
        type=int,
        default=1,
        help=(
            "Maximum number of participant Agents stepped concurrently per "
            "period (default: 1; use 4 for all four Agents)."
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
    print(json.dumps({"config": cfg.safe_status(), "limits": {"requests": args.request_limit, "max_output_tokens": args.max_output_tokens, "timeout_seconds": args.timeout_seconds, "full_session": args.full_session, "agent_concurrency": args.agent_concurrency}}, ensure_ascii=False))
    try:
        manifest = run_experiment(
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






