"""Memory maintenance has its own task contract, without changing decisions."""

import copy

import pytest

from custom.agents.fixed_preferences_v1.profiles import render_profile
from custom.agents.fixed_preferences_memory_v1.participant import (
    MEMORY_UPDATE_PHASE_PATH, PROMPT_PATH,
)
from experiments.fixed_preferences_memory_v1 import run_smoke as version
from test_fixed_preferences_integration import isolated_runtime, read_json, rows  # noqa: F401
from test_fixed_preferences_memory_integration import is_memory_update, memory_payload


@pytest.mark.parametrize("condition", ["persistent", "fresh"])
def test_real_memory_and_decision_calls_have_distinct_system_contracts(tmp_path, monkeypatch, condition):
    original = version.ScriptedLLMClient.call
    phase_prompt = MEMORY_UPDATE_PHASE_PATH.read_text(encoding="utf-8")
    calls = []
    rejected_once = False

    async def capture(self, model=None, messages=None, **kwargs):
        nonlocal rejected_once
        before = copy.deepcopy(messages)
        response = await original(self, model=model, messages=messages, **kwargs)
        if self.kind != "participant":
            return response
        agent, observation = self._participant_turn(messages)
        update = is_memory_update(messages)
        calls.append((agent, observation, update, copy.deepcopy(messages)))
        system = messages[0]["content"]
        assert kwargs["max_output_tokens"] == 8192
        if update:
            assert system.endswith(phase_prompt)
            assert not kwargs.get("tools") and "tool_choice" not in kwargs
            assert "Current invocation: private memory update only." in system
            assert "insurance" not in str(messages).lower()  # Baseline never reveals an offer.
            if condition == "fresh":
                assert memory_payload(messages)["content"] == {"note": "", "evidence_periods": []}
            if agent == 1 and not rejected_once:
                # The observed provider failure was an economic answer during
                # maintenance. It must still be rejected, never auto-submitted.
                response.choices[0].message.content = "decision_type: transfer\namount: 100"
                rejected_once = True
        else:
            assert phase_prompt not in system
            assert "Current invocation: private memory update only." not in str(messages)
            assert kwargs.get("tools")
        assert messages == before
        return response

    monkeypatch.setattr(version.ScriptedLLMClient, "call", capture)
    root = tmp_path / condition
    result = version.run_experiment(
        backend="scripted", memory_condition=condition, participants=4,
        agent_concurrency=4, completed_period_limit=3, request_limit=100,
        max_output_tokens=8192, run_dir=root,
    )
    assert result["memory_experiment_complete"] is True
    assert result["private_memory_update_rejections"] == 1
    assert result["private_memory_update_count"] == 12
    profiles = read_json(root / "participant_profiles.json")["profiles"]
    for agent, observation, update, messages in calls:
        original_system = PROMPT_PATH.read_text(encoding="utf-8") + "\n\n" + render_profile(profiles[str(agent)])
        assert messages[0]["content"] == original_system + ("\n\n" + phase_prompt if update else "")
        assert observation["agent_id"] == agent
    assert len([c for c in calls if not c[2]]) == 12  # Waiting choices still use participant model.
    transfers = [r for r in rows(root / "agent_actions.jsonl") if r.get("action") == "submit_decision" and r.get("ok")]
    assert len(transfers) == 6
    assert all(r["submitted_amount"] == r["participant_model_output"]["amount"] == r["environment_receipt"]["amount"] for r in transfers)
    assert len(result["memory_update_phase_prompt_sha256"]) == 64
