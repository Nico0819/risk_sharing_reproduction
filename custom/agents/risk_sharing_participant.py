"""Minimal PersonAgent for the risk-sharing smoke experiment."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
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

    async def dispatch_react_tool(
        self,
        action: str,
        args: dict[str, Any],
        *,
        readonly: bool = False,
    ):
        """Record a participant decision before sending it to ``ask_env``."""

        if action == "ask_env":
            instruction = str(args.get("instruction") or "")
            decision = parse_participant_decision(instruction)
            if (
                getattr(self, "_last_participant_observation", None) is not None
                and decision is None
            ):
                return ReactToolResult(
                    False,
                    "The current observation was already provided; submit your own structured decision or finish waiting.",
                    {"error": "observation_already_read"},
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
                    )
        return await super().dispatch_react_tool(action, args, readonly=readonly)

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
