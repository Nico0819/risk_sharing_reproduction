"""Participant with private fixed preferences and the original tool boundary."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from custom.agents.risk_sharing_participant import (
    INSURANCE_PROMPT_PATH,
    RiskSharingParticipant,
)
from .profiles import profile_sha256, render_profile, validate_profile


PROMPT_PATH = Path(__file__).with_name("participant.md")


class FixedPreferenceParticipant(RiskSharingParticipant):
    """Keep individual preferences in the framework's static workspace config."""

    @classmethod
    def description(cls) -> str:
        return "Risk-sharing participant with fixed risk and advantageous inequality preferences."

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
        if self._config.get("enable_memory", False):
            raise ValueError("fixed_preferences_v1 requires enable_memory=False")
        profile = validate_profile(self._config.get("fixed_preferences"))
        expected_hash = self._config.get("fixed_preferences_sha256")
        if not isinstance(expected_hash, str) or expected_hash != profile_sha256(profile):
            raise ValueError("fixed preference configuration does not match its recorded hash")

        # Reuse shared observation presentation and receipt/deliberation reset.
        # Replace the complete system message with this version's own prompt;
        # the baseline's monetary-only objective must not leak into this one.
        messages = super().build_react_messages(
            tick=tick, t=t, observations=observations, question=question,
            readonly=readonly, skill_hooks=skill_hooks,
        )
        rules = PROMPT_PATH.read_text(encoding="utf-8")
        if (self._last_participant_observation or {}).get("task") == "insurance":
            rules += "\n\n" + INSURANCE_PROMPT_PATH.read_text(encoding="utf-8")
        messages[0] = {"role": "system", "content": rules + "\n\n" + render_profile(profile)}
        return messages
