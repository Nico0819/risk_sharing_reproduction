"""Minimal PersonAgent for the risk-sharing smoke experiment."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from agentsociety2.agent import PersonAgent
from agentsociety2.agent.base.react import ReactToolResult

from custom.agents.observation_presentation import render_readable_turn
from custom.envs.risk_sharing_env import parse_participant_decision


PROMPT_PATH = Path(__file__).with_name("risk_sharing_participant.md")
INSURANCE_PROMPT_PATH = Path(__file__).with_name("risk_sharing_insurance.md")
DELIBERATION_PROMPT_PATH = Path(__file__).with_name("risk_sharing_deliberation.md")


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

    def _decision_format(self) -> str:
        observation = getattr(self, "_last_participant_observation", None) or {}
        if observation.get("task") == "insurance":
            return (
                "decision_type: insurance\ndecision_id: <the observation decision_id>\n"
                "purchase: <true or false, chosen by you>"
            )
        return (
            "decision_type: transfer\ndecision_id: <the observation decision_id>\n"
            "amount: <your chosen integer>"
        )

    async def step(self, tick: int, t: datetime) -> str:
        """Use PersonAgent's real observe-then-react lifecycle."""

        return await super().step(tick, t)

    def _refresh_visible_skills(self) -> None:
        """Keep generic skills and their lifecycle hooks outside the experiment.

        AgentSociety2 2.9.0 treats disabled_skill_ids as exact IDs, so ['*']
        is not a wildcard. Clear both sets on restore and before every step.
        """

        self.skill_runtime.set_visible_skills(set())
        self.skill_runtime.set_activated_skills(set())

    def _react_tool_overrides(self, *, readonly: bool) -> dict[str, Any]:
        """Describe our explicit-choice protocol instead of generic templates."""

        overrides = super()._react_tool_overrides(readonly=readonly)
        overrides["ask_env"] = {
            "type": "function",
            "function": {
                "name": "ask_env",
                "description": (
                    "Submit your own current choice using the observation already "
                    "provided. Explain your reasoning in ordinary language if useful, "
                    "then give the explicit decision fields as labelled "
                    "lines in instruction. Use literal values, not template placeholders. "
                    "The Router only forwards your choice. For a waiting task use finish."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "instruction": {
                            "type": "string",
                            "description": (
                                "Your explanation followed by literal values in this format:\n"
                                + self._decision_format()
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
        observation = getattr(self, "_last_participant_observation", None) or {}
        if (
            self._config.get("participant_deliberation", False)
            and not readonly
            and observation.get("task") in {"transfer", "insurance"}
            and not getattr(self, "_deliberation_completed", False)
        ):
            await self._review_decision(messages, observation)
        return await super()._complete_react_once(messages, readonly=readonly)

    async def _review_decision(self, messages, observation):
        """Ask this participant for a brief assessment before its tool call.

        This is an explicit extra completion, not provider-native reasoning or
        persistent memory. Only this turn's participant context receives it.
        """

        review_request = {
            "role": "user",
            "content": "<participant_deliberation>\n"
            + DELIBERATION_PROMPT_PATH.read_text(encoding="utf-8")
            + "\n</participant_deliberation>",
        }
        review_messages = [*messages, review_request]
        response = await self.acompletion(
            review_messages, participant_phase="deliberation",
            max_output_tokens=512,
        )
        choice = response.choices[0]
        reply = choice.message
        summary = getattr(reply, "content", None)
        if (
            getattr(reply, "tool_calls", None)
            or getattr(reply, "function_call", None)
            or not isinstance(summary, str)
            or not summary.strip()
            or getattr(choice, "finish_reason", None) not in {None, "stop"}
        ):
            raise ValueError("Participant deliberation must return a complete text summary without tool calls.")
        recorder = getattr(self._dispatcher, "record_participant_deliberation", None)
        if recorder is not None:
            recorder(observation=observation, messages=review_messages, review_text=summary)
        messages.extend([
            review_request,
            {"role": "assistant", "content": summary},
            {
                "role": "user",
                "content": (
                    "Your private review is complete. It is a tentative assessment, "
                    "not new environment information. Now make your final choice "
                    "using the current observation and the permitted action tool. "
                    "You may revise the preference expressed in your review."
                ),
            },
        ])
        self._deliberation_completed = True

    async def dispatch_react_tool(
        self,
        action: str,
        args: dict[str, Any],
        *,
        readonly: bool = False,
    ):
        """Enforce the experiment's tool boundary, then record explicit choices.

        The framework parses unadvertised native and textual tool names too.
        Hiding its generic workspace/skill tools in the model schema does not
        prevent execution, so reject them before delegating to PersonAgent.
        """

        if action not in {"ask_env", "finish"}:
            return ReactToolResult(
                False,
                "This participant may only use ask_env or finish; "
                "the requested tool is not permitted.",
                {"error": "participant_tool_not_permitted", "action": action},
            )

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
                        "Use the specified decision fields; if your task "
                        "is waiting, use finish."
                    )
                if decision["decision_type"] != observation.get("task"):
                    raise ValueError("Your decision_type must match the current observation task.")
                if readonly or args.get("readonly", False) is not False:
                    raise ValueError("A decision must use readonly=false.")
            except (TypeError, ValueError) as exc:
                recorder = getattr(self._env, "record_participant_rejection", None)
                if recorder is not None:
                    recorder(
                        self.id, instruction, str(exc),
                        variables=variables if isinstance(variables, dict) else None,
                    )
                return ReactToolResult(
                    False,
                    f"{exc} No decision was submitted. You may explain your choice "
                    "in prose, then provide literal values in this format:\n"
                    + self._decision_format()
                    + "\nCorrect the request yourself; no JSON is needed.",
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
            value_field = "purchase" if decision["decision_type"] == "insurance" else "amount"
            try:
                receipt = json.loads(result.observation)
            except (TypeError, json.JSONDecodeError):
                receipt = None
            if (
                isinstance(receipt, dict)
                and receipt.get("status") in {"submitted", "already_recorded"}
                and receipt.get("decision_id") == decision["decision_id"]
                and receipt.get(value_field) == decision[value_field]
                and type(receipt.get(value_field)) is type(decision[value_field])
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
        self._deliberation_completed = False
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
        if (self._last_participant_observation or {}).get("task") == "insurance":
            # Never load the option's instructions into earlier turns. The
            # offer and its prices arrive only in the permitted observation.
            rules += "\n\n" + INSURANCE_PROMPT_PATH.read_text(encoding="utf-8")
        return [
            {"role": "system", "content": rules},
            {
                "role": "user",
                "content": render_readable_turn(payload, agent_id=self.id),
            },
        ]
