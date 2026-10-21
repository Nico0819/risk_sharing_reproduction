"""Minimal PersonAgent for the risk-sharing smoke experiment."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from agentsociety2.agent import PersonAgent
from agentsociety2.agent.base.react import ReactToolResult

from custom.envs.risk_sharing_env import parse_participant_decision


PROMPT_PATH = Path(__file__).with_name("risk_sharing_participant.md")


class RiskSharingParticipant(PersonAgent):
    """A neutral participant that acts only through the risk-sharing tools."""

    @classmethod
    def description(cls) -> str:
        return "Risk-sharing participant with a bounded observation-and-transfer task."

    @classmethod
    def init_description(cls) -> str:
        return (
            "A PersonAgent subclass for the two-person transfer task. "
            "It uses only the environment observation and decision tools."
        )

    async def step(self, tick: int, t: datetime) -> str:
        """Use PersonAgent's real observe-then-react lifecycle."""

        return await super().step(tick, t)

    def _react_tool_overrides(self, *, readonly: bool) -> dict[str, Any]:
        """Describe our explicit-choice protocol instead of generic templates."""

        overrides = super()._react_tool_overrides(readonly=readonly)
        overrides["ask_env"] = {
            "type": "function",
            "function": {
                "name": "ask_env",
                "description": (
                    "Submit your own transfer choice using the observation already "
                    "provided. Explain your reasoning in ordinary language if useful, "
                    "then give decision_type, decision_id, and amount as labelled "
                    "lines in instruction. Use literal values, not template placeholders. "
                    "The Router only forwards your choice. For a waiting task use finish."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "instruction": {
                            "type": "string",
                            "description": (
                                "Your explanation followed by decision_type: transfer, "
                                "decision_id: the exact current observation token, and "
                                "amount: your chosen integer, each on its own line."
                            ),
                        },
                        "readonly": {"type": "boolean", "enum": [False]},
                    },
                    "required": ["instruction", "readonly"],
                    "additionalProperties": False,
                },
            },
        }
        return overrides

    async def _complete_react_once(self, messages, *, readonly: bool = False):
        """End a verified submission locally when subjective memory is disabled.

        This supplies only framework termination, never an economic decision.
        Failed requests still go back to the participant model for correction.
        """

        receipt = getattr(self, "_accepted_receipt", None)
        if receipt is not None and not readonly and not self._enable_memory:
            function = SimpleNamespace(
                name="finish",
                arguments=json.dumps({"memories": [{
                    "type": "observation", "importance": 0.0,
                    "text": "Environment accepted my submitted choice.",
                    "source": "environment_receipt",
                }]}),
            )
            tool_call = SimpleNamespace(
                id=f"local-receipt-finish-{self.id}", type="function", function=function,
            )
            message = SimpleNamespace(content="", tool_calls=[tool_call])
            return SimpleNamespace(choices=[SimpleNamespace(message=message)])
        return await super()._complete_react_once(messages, readonly=readonly)

    async def dispatch_react_tool(
        self,
        action: str,
        args: dict[str, Any],
        *,
        readonly: bool = False,
    ):
        """Record a participant decision before sending it to ``ask_env``."""

        decision = None
        if action == "ask_env":
            instruction = str(args.get("instruction") or "")
            observation = getattr(self, "_last_participant_observation", None)
            # Match the framework's actual merging of ctx.variables/variables.
            variables = self._build_env_tool_context(args).get("variables")
            try:
                if observation is None:
                    raise ValueError("A current observation is required before deciding.")
                decision = parse_participant_decision(instruction, variables)
                if decision is None:
                    raise ValueError(
                        "No explicit decision fields were found in your request. "
                        "Use decision_type, decision_id, and amount; if your task "
                        "is waiting, use finish."
                    )
                if readonly or args.get("readonly", False) is not False:
                    raise ValueError("A transfer must use readonly=false.")
            except (TypeError, ValueError) as exc:
                recorder = getattr(self._env, "record_participant_rejection", None)
                if recorder is not None:
                    recorder(
                        self.id, instruction, str(exc),
                        variables=variables if isinstance(variables, dict) else None,
                    )
                return ReactToolResult(
                    False,
                    f"{exc} No transfer was submitted. You may explain your choice "
                    "in prose, then give one line each for decision_type: transfer, "
                    "decision_id: the current observation token, and amount: your "
                    "chosen integer. Correct the request yourself; no JSON is needed.",
                    {"error": str(exc)},
                )
            if decision is not None and self._env is not None:
                recorder = getattr(self._env, "record_participant_decision", None)
                if recorder is not None:
                    recorder(
                        self.id,
                        decision,
                        raw_request=instruction,
                        observation=getattr(
                            self, "_last_participant_observation", None
                        ),
                        variables=variables,
                    )
        result = await super().dispatch_react_tool(action, args, readonly=readonly)
        if decision is not None and result.ok:
            try:
                receipt = json.loads(result.observation)
            except (TypeError, json.JSONDecodeError):
                receipt = None
            if (
                isinstance(receipt, dict)
                and receipt.get("status") in {"submitted", "already_recorded"}
                and receipt.get("decision_id") == decision["decision_id"]
                and receipt.get("amount") == decision["amount"]
                and receipt.get("period") == observation["current_period"]
            ):
                self._accepted_receipt = receipt
        return result

    def build_react_messages(
        self,
        *,
        tick: int,
        t: datetime,
        observations: list[dict[str, Any]],
        question: str | None = None,
        readonly: bool = False,
        skill_hooks: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, str]]:
        """Use the project prompt without injecting paper/research metadata."""

        rules = PROMPT_PATH.read_text(encoding="utf-8")
        payload = {
            "agent_id": self.id,
            "observations": observations,
            "readonly": readonly,
            "question": question,
        }
        self._last_participant_observation = None
        self._accepted_receipt = None
        for item in reversed(observations):
            value = item.get("observation") if isinstance(item, dict) else None
            if isinstance(value, str):
                try:
                    parsed = json.loads(value)
                except json.JSONDecodeError:
                    continue
                if isinstance(parsed, dict):
                    self._last_participant_observation = parsed
                    break
        return [
            {"role": "system", "content": rules},
            {
                "role": "user",
                "content": "<risk_sharing_turn>\n"
                + json.dumps(payload, ensure_ascii=False, default=str)
                + "\n</risk_sharing_turn>",
            },
        ]
