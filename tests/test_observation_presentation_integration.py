"""Readable observations through the installed framework and unchanged decisions.

These tests validate lossless delivery and routing, not an improvement in LLM
comprehension. Only the provider boundary is scripted; agents and Router are real.
"""

from __future__ import annotations

import json
import hashlib
from pathlib import Path

import pytest

from experiments.fixed_preferences_v1 import run_smoke as version
from experiments.baseline import run_smoke as baseline
from test_fixed_preferences_integration import (
    check_contexts,
    check_transfers,
    contains_key,
    isolated_runtime,  # noqa: F401 -- import the autouse fixture for this module
    read_json,
    rows,
    spy_models,
)


def turn_payload(messages):
    matching = [
        message["content"]
        for message in messages
        if isinstance(message.get("content"), str)
        and "<risk_sharing_turn>" in message["content"]
    ]
    assert len(matching) == 1
    return json.loads(
        matching[0].split("<risk_sharing_turn>", 1)[1].split("</risk_sharing_turn>", 1)[0]
    )


def check_readable_delivery(root, manifest, calls, *, with_preferences=True):
    assert manifest["observation_presentation"] == "readable_v1"
    assert manifest["router"] == "ReActRouter"
    source = Path(__file__).parents[1] / "custom/agents/observation_presentation.py"
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if with_preferences:
        profiles = read_json(root / "participant_profiles.json")["profiles"]
        check_contexts(root, manifest, calls, profiles)
        assert manifest["source_hashes"]["custom/agents/observation_presentation.py"] == digest
        assert manifest["agent_class"] == "FixedPreferenceParticipant"
        assert manifest["enable_memory"] is False
        contexts = rows(root / "participant_contexts.jsonl")
        assert all(row["observation_presentation"] == "readable_v1" for row in contexts)
    else:
        assert manifest["agent_class"] == "RiskSharingParticipant"
        assert manifest["observation_presentation_sha256"] == digest
        assert not (root / "participant_profiles.json").exists()
    for call in calls:
        if call["kind"] != "participant":
            continue
        agent_id, observation = call["turn"]
        payload = turn_payload(call["messages"])
        assert payload["agent_id"] == agent_id
        observed = [item["observation"] for item in payload["observations"]]
        # The observation is an object rather than an escaped JSON string.
        assert observed and all(isinstance(item, dict) for item in observed)
        assert observed[-1] == observation
        user = next(message["content"] for message in call["messages"] if message.get("role") == "user")
        assert "Current observation — reading guide" in user
        assert "Completed rounds for your pair" in user
        assert payload["readonly"] is False
        assert not contains_key(observation, {
            "treatment", "analysis_phase", "session_id", "profiles", "fixed_preferences",
            "risk_aversion", "advantageous_inequality_aversion",
        })
        members = {agent_id, observation["partner_id"]}
        assert all(
            {record["agent_id"], record["partner_id"]} == members
            and record["period"] < observation["current_period"]
            for record in observation["history"]
        )
        config = read_json(root / "agents" / f"agent_{agent_id:04d}" / "config.json")
        assert "observation_presentation" not in config
        assert config["enable_memory"] is False
        if not with_preferences:
            assert "fixed_preferences" not in config
            assert "<fixed_preferences>" not in json.dumps(call["messages"])
    if with_preferences:
        return check_transfers(root, profiles)
    transfers = [row for row in rows(root / "agent_actions.jsonl") if row["action"] == "submit_decision"]
    assert transfers and all(row["ok"] for row in transfers)
    for row in transfers:
        observation = row["participant_model_message"]["observation"]
        assert observation["agent_id"] == row["agent_id"]
        assert observation["decision_id"] == row["decision_id"]
        assert row["decision_source"] == "participant_model"
        assert type(row["submitted_amount"]) is int
        assert 0 <= row["submitted_amount"] <= observation["legal_transfer_range"]["max"]
        assert row["submitted_amount"] == row["participant_model_output"]["amount"] == row["environment_receipt"]["amount"]
    return transfers


def normalized_observations(calls):
    """Compare complete legal observations, excluding only fresh opaque tokens."""
    observations = {}
    for call in calls:
        if call["kind"] == "participant":
            agent_id, observation = call["turn"]
            key = (agent_id, observation["current_period"], observation["task"])
            assert key not in observations
            observations[key] = {
                field: value for field, value in observation.items() if field != "decision_id"
            }
    return observations


def economic_records(root):
    # A new run has a different session ID, while every economic field must match.
    return [
        {key: value for key, value in record.items() if key != "session_id"}
        for record in rows(root / "period_records.jsonl")
    ]


@pytest.mark.parametrize("module", [baseline, version], ids=["original", "fixed_preferences"])
def test_readable_twenty_agent_contexts_remain_private_and_concurrent(tmp_path, monkeypatch, module):
    spy = spy_models(monkeypatch, module=module)
    root = tmp_path / "twenty-readable"
    manifest = module.run_experiment(
        backend="scripted",
        participants=20, agent_concurrency=20, completed_period_limit=2,
        request_limit=100, run_dir=root, preference_seed=19,
    )
    assert manifest["success"] is True
    assert spy["peak"] == 20
    assert manifest["participant_model_requests"] == 40
    assert manifest["router_model_requests"] == 20
    transfers = check_readable_delivery(root, manifest, spy["calls"], with_preferences=module is version)
    assert len(transfers) == 20
    waiting = [
        call for call in spy["calls"]
        if call["kind"] == "participant" and call["turn"][1]["task"] == "waiting"
    ]
    assert len(waiting) == 20
    assert all([tool["name"] for tool in call["tool_calls"]] == ["finish"] for call in waiting)


@pytest.mark.parametrize("treatment", ["baseline", "25V", "25F"])
def test_readable_preserves_all_three_treatments_through_insurance_withdrawal(
    tmp_path, monkeypatch, treatment,
):
    spies = {"original": spy_models(monkeypatch, module=baseline), "fixed_preferences": spy_models(monkeypatch)}
    kwargs = {"forced_assignment_probabilities": (0.5, 0.25, 0.25)} if treatment == "25F" else {}
    roots, manifests, recorded_calls = {}, {}, {}
    for name, module in (("original", baseline), ("fixed_preferences", version)):
        roots[name] = tmp_path / name
        manifests[name] = module.run_experiment(
            backend="scripted", treatment=treatment,
            participants=4, agent_concurrency=4, completed_period_limit=31,
            request_limit=300, run_dir=roots[name], **kwargs,
        )
        recorded_calls[name] = spies[name]["calls"]
        assert manifests[name]["success"] is True
        assert manifests[name]["completed_periods"] == list(range(1, 32))
        assert len(check_readable_delivery(
            roots[name], manifests[name], recorded_calls[name], with_preferences=module is version,
        )) == 62

    assert normalized_observations(recorded_calls["original"]) == normalized_observations(
        recorded_calls["fixed_preferences"]
    )
    assert economic_records(roots["original"]) == economic_records(roots["fixed_preferences"])
    assert manifests["original"]["prompt_sha256"] != manifests["fixed_preferences"]["prompt_sha256"]
    assert manifests["original"]["actual_model_requests"] == manifests["fixed_preferences"]["actual_model_requests"]

    participant = [call for calls in recorded_calls.values() for call in calls if call["kind"] == "participant"]
    insurance = [call for call in participant if call["turn"][1]["task"] == "insurance"]
    assert len(insurance) == (8 if treatment == "25V" else 0)
    for call in participant:
        observation = call["turn"][1]
        if observation["current_period"] < 16:
            assert "Your current observation offers an insurance purchase decision" not in json.dumps(call["messages"])
        if treatment != "baseline" and observation["current_period"] == 31:
            assert observation["insured"] is False
            assert "announcement" in observation
            if treatment == "25F":
                assert observation["insurance_assignment_source"] == "computer_random"
                assert "insurance_assigned" in observation
            else:
                assert "insurance_purchased" in observation


@pytest.mark.parametrize("module", [baseline, version], ids=["original", "fixed_preferences"])
def test_presentation_is_not_a_runtime_or_cli_selection(tmp_path, monkeypatch, module):
    def unexpected_runtime(*args, **kwargs):
        pytest.fail("Removed arguments must fail before loading credentials or calling models.")

    monkeypatch.setattr(module, "load_runtime_config", unexpected_runtime)
    defaults = module._parser().parse_args(["--backend", "scripted"])
    assert not hasattr(defaults, "observation_presentation")
    with pytest.raises(SystemExit):
        module._parser().parse_args([
            "--backend", "scripted", "--observation-presentation", "readable_v1",
        ])
    root = tmp_path / "invalid"
    with pytest.raises((TypeError, ValueError), match="presentation"):
        module.run_experiment(
            backend="scripted", observation_presentation="readable_v1", run_dir=root,
        )
    assert not root.exists()


def test_readable_observation_is_available_in_deliberation_and_decision(tmp_path, monkeypatch):
    spy = spy_models(monkeypatch)
    root = tmp_path / "deliberation-readable"
    manifest = version.run_experiment(
        backend="scripted",
        participant_deliberation=True, agent_concurrency=4,
        completed_period_limit=1, run_dir=root,
    )
    assert manifest["success"] is True
    assert manifest["participant_deliberation_requests"] == 2
    assert manifest["participant_model_requests"] == 6
    assert manifest["actual_model_requests"] == 8
    assert len(check_readable_delivery(root, manifest, spy["calls"])) == 2
    contexts = rows(root / "participant_contexts.jsonl")
    assert sum(row["phase"] == "deliberation" for row in contexts) == 2
    assert sum(row["phase"] == "decision" for row in contexts) == 4


def test_readable_mode_still_rejects_amount_created_by_router(tmp_path, monkeypatch):
    original = version.ScriptedLLMClient.call

    async def tampering_call(self, model=None, messages=None, **kwargs):
        response = await original(self, model=model, messages=messages, **kwargs)
        if self.kind == "router":
            for tool in response.choices[0].message.tool_calls:
                if tool.function.name == "submit_decision":
                    arguments = json.loads(tool.function.arguments)
                    arguments["amount"] += 1
                    tool.function.arguments = json.dumps(arguments)
        return response

    monkeypatch.setattr(version.ScriptedLLMClient, "call", tampering_call)
    root = tmp_path / "tampering"
    manifest = version.run_experiment(
        backend="scripted",
        completed_period_limit=1, run_dir=root,
        router_steps=2, agent_turns=2, request_limit=100,
    )
    assert manifest["success"] is False
    assert manifest["completed_periods"] == []
    assert rows(root / "period_records.jsonl") == []
    submissions = [row for row in rows(root / "agent_actions.jsonl") if row["action"] == "submit_decision"]
    assert submissions and all(not row["ok"] for row in submissions)
