"""Minimal PersonAgent for the risk-sharing smoke experiment."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from agentsociety2.agent import PersonAgent


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
        """Run one model-driven action without a second pre-observation call.

        The base PersonAgent performs a readonly ``<observe>`` request before
        every action loop. This project environment already exposes the
        current observation through the participant's real ``ask_env`` tool
        call, so omitting that redundant call keeps the bounded smoke within
        its request budget while retaining the PersonAgent/ReAct lifecycle.
        """

        await self._ensure_initialized()
        self._step_count += 1
        self._current_time = t
        self._refresh_visible_skills()
        self.persist_agent_json(tick=tick, t=t)
        result = await self.run_react_loop(
            tick=tick,
            t=t,
            observations=[],
            skill_hooks=None,
        )
        await self._memory_runtime.after_step(
            tick=tick,
            t=t,
            step_count=self._step_count,
            finish_memories=self._last_finish_memories,
        )
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
        return [
            {"role": "system", "content": rules},
            {
                "role": "user",
                "content": "<risk_sharing_turn>\n"
                + json.dumps(payload, ensure_ascii=False, default=str)
                + "\n</risk_sharing_turn>",
            },
        ]
