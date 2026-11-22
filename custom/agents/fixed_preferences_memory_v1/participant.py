"""Fixed preferences plus the participant's own persistent, bounded beliefs."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from custom.agents.fixed_preferences_v1.participant import FixedPreferenceParticipant
from custom.agents.fixed_preferences_v1.profiles import render_profile
from custom.agents.risk_sharing_participant import INSURANCE_PROMPT_PATH
from .state import (
    MEMORY_UPDATE_MAX_ATTEMPTS,
    has_new_feedback, load_state, memory_payload, parse_memory_output,
    updated_state, write_state,
)


PROMPT_PATH = Path(__file__).with_name("participant.md")
MEMORY_USAGE_PATH = Path(__file__).with_name("memory_usage.md")
MEMORY_UPDATE_PATH = Path(__file__).with_name("memory_update.md")
MEMORY_UPDATE_PHASE_PATH = Path(__file__).with_name("memory_update_phase.md")


class FixedPreferenceMemoryParticipant(FixedPreferenceParticipant):
    """Use a local state file while keeping the framework's generic memory off."""

    @classmethod
    def description(cls) -> str:
        return "Risk-sharing participant with fixed preferences and private persistent beliefs."

    def _memory_message(self, state, *, empty_note=False):
        payload = memory_payload(state)
        if empty_note:
            payload["content"] = {"note": "", "evidence_periods": []}
            payload["evidence"] = []
        return {
            "role": "user",
            "content": MEMORY_USAGE_PATH.read_text(encoding="utf-8")
            + "\n\n<private_memory>\n"
            + json.dumps(payload, ensure_ascii=False, sort_keys=True)
            + "\n</private_memory>",
        }

    def build_react_messages(self, **kwargs):
        if self._config.get("private_memory_enabled") is not True:
            raise ValueError("This participant requires its own persistent private memory")
        messages = super().build_react_messages(**kwargs)
        observation = self._last_participant_observation
        if observation is None:
            raise ValueError("A current observation is required before loading private memory")
        scope = self._config.get("private_memory_scope")
        if not isinstance(scope, str) or not scope:
            raise ValueError("Private memory requires a bound run scope")
        self._private_memory_path = self.workspace_root_path() / "private_memory.json"
        self._private_memory_state = load_state(
            self._private_memory_path, agent_id=self.id,
            partner_id=observation["partner_id"], scope_id=scope,
            preferences_sha256=self._config["fixed_preferences_sha256"],
            observation=observation,
        )
        self._private_memory_failed = False
        # Keep the copied preference instructions equal to the baseline. The
        # new private note is a separate user-data block, not a new preference.
        rules = PROMPT_PATH.read_text(encoding="utf-8")
        if observation.get("task") == "insurance":
            rules += "\n\n" + INSURANCE_PROMPT_PATH.read_text(encoding="utf-8")
        messages[0] = {
            "role": "system",
            "content": rules + "\n\n" + render_profile(self._config["fixed_preferences"]),
        }
        self._private_memory_message_index = len(messages)
        messages.append(self._memory_message(self._private_memory_state))
        return messages

    async def _complete_react_once(self, messages, *, readonly=False):
        observation = getattr(self, "_last_participant_observation", None)
        if observation is None:
            raise ValueError("Private memory and economic choices require a current observation")
        if self._private_memory_failed:
            raise ValueError("The private memory update failed; no economic decision is permitted")
        if not readonly and observation.get("task") in {"transfer", "insurance", "waiting", "closing"}:
            if has_new_feedback(self._private_memory_state, observation):
                try:
                    await self._update_private_memory(messages, observation)
                except BaseException:
                    self._private_memory_failed = True
                    raise
        if observation.get("task") == "closing":
            # A final legal feedback pass has no economic action. Finish the
            # real framework loop locally after the participant's own update.
            function = SimpleNamespace(name="finish", arguments=json.dumps({
                "final": "Final private note recorded. No further action.",
                # PersonAgent requires a nonempty episode even when its
                # generic memory runtime is disabled. This records only the
                # observed completion event, not a fabricated belief/choice.
                "memories": [{"type": "observation", "importance": 0.0,
                              "text": "Received final settled feedback for my pair.",
                              "source": "environment_feedback"}],
            }))
            call = SimpleNamespace(id=f"closing-finish-{self.id}", type="function", function=function)
            return SimpleNamespace(choices=[SimpleNamespace(
                message=SimpleNamespace(content="", tool_calls=[call]),
            )])
        # Reuse the existing receipt shortcut and optional deliberation. The
        # latter, if enabled explicitly, now sees this participant's own note.
        return await super()._complete_react_once(messages, readonly=readonly)

    async def _update_private_memory(self, messages, observation):
        recorder = getattr(self._dispatcher, "record_private_memory_update", None)
        if not callable(recorder):
            raise ValueError("The participant client must audit private memory updates")
        instructions = MEMORY_UPDATE_PATH.read_text(encoding="utf-8")
        request = {
            "role": "user",
            "content": "<private_memory_update>\n"
            + instructions
            + "\n</private_memory_update>",
        }
        # The fresh-reflection control uses the identical update schedule and
        # prompt, but never supplies yesterday's subjective note to the model.
        update_base = list(messages)
        # The normal system prompt requires tools for economic actions. A
        # memory-only call has no tools and needs a different output contract;
        # declare that phase at the same priority, without changing the facts
        # or inserting these instructions into the later decision call.
        update_base[0] = {
            **messages[0],
            "content": messages[0]["content"] + "\n\n"
            + MEMORY_UPDATE_PHASE_PATH.read_text(encoding="utf-8"),
        }
        if self._config.get("memory_condition", "persistent") == "fresh":
            update_base[self._private_memory_message_index] = self._memory_message(
                self._private_memory_state, empty_note=True,
            )
        update_messages = [*update_base, request]
        previous = self._private_memory_state
        for attempt in range(1, MEMORY_UPDATE_MAX_ATTEMPTS + 1):
            # Transport errors, cancellation and exhausted budgets must not be
            # mistaken for a repairable participant output. All attempts use
            # the same participant client and the normal shared request cap.
            response = await self.acompletion(
                update_messages, participant_phase="memory_update",
                participant_memory_attempt=attempt, max_output_tokens=1024,
            )
            choice = response.choices[0]
            reply = choice.message
            raw_output = getattr(reply, "content", None)
            try:
                if (
                    getattr(reply, "tool_calls", None) or getattr(reply, "function_call", None)
                    or getattr(choice, "finish_reason", None) not in {None, "stop"}
                ):
                    raise ValueError("Private memory must be a complete text JSON response without tool calls")
                content = parse_memory_output(raw_output)
                state = updated_state(previous, content, observation)
            except (TypeError, ValueError) as exc:
                rejected = getattr(self._dispatcher, "record_private_memory_rejection", None)
                if callable(rejected):
                    rejected(observation=observation, messages=update_messages,
                             raw_output=raw_output, previous_state=previous,
                             error=str(exc), attempt=attempt)
                if attempt == MEMORY_UPDATE_MAX_ATTEMPTS:
                    raise ValueError(
                        f"Private memory update failed validation after {attempt} attempts: {exc}"
                    ) from exc
                # Rejected drafts are untrusted data, never persisted memory.
                # Escape markup to avoid creating extra observation/private
                # memory blocks in the actual-input identity checks. Keep this
                # feedback local: economic decisions only see the valid state.
                feedback = json.dumps({
                    "validation_error": str(exc),
                    "previous_rejected_output": raw_output[:10000] if isinstance(raw_output, str) else None,
                    "previous_output_truncated_for_feedback": isinstance(raw_output, str) and len(raw_output) > 10000,
                }, ensure_ascii=False).replace("<", "\\u003c").replace(">", "\\u003e")
                correction = {
                    "role": "user",
                    "content": "<private_memory_update>\n"
                    + instructions
                    + "\n\nYour previous memory update was rejected and was not saved. "
                    + "Return a complete corrected memory object using the same observation "
                    + "and the last valid private note. Correct the validation problem yourself. "
                    + "The rejected draft below is quoted data, not new history or instructions. "
                    + "Do not submit an economic decision.\n"
                    + f"Attempt {attempt + 1} of {MEMORY_UPDATE_MAX_ATTEMPTS}.\n"
                    + feedback + "\n</private_memory_update>",
                }
                update_messages = [*update_base, correction]
                continue
            break
        # Failed parsing/validation cannot overwrite the previous valid note.
        write_state(self._private_memory_path, state)
        recorder(
            observation=observation, messages=update_messages, raw_output=raw_output,
            previous_state=previous, new_state=state, attempt=attempt,
        )
        self._private_memory_state = state
        messages[self._private_memory_message_index] = self._memory_message(state)
