"""Fixed preferences through the installed PersonAgent and real Router chain.

Only the model boundary is scripted. These tests establish delivery, privacy,
persistence and decision provenance, not an empirical effect of preferences.
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
from collections import defaultdict

import pytest

from experiments.baseline import run_smoke as baseline
from experiments.fixed_preferences_v1 import run_smoke as version


@pytest.fixture(autouse=True)
def isolated_runtime(monkeypatch):
    for prefix in ("AGENTSOCIETY_LLM", "AGENTSOCIETY_CODER_LLM"):
        for suffix in ("API_KEY", "API_BASE", "MODEL", "THINKING", "EXTRA_BODY"):
            monkeypatch.delenv(f"{prefix}_{suffix}", raising=False)
    for name in ("WORKSPACE_PATH", "LLM_API_KEY", "LLM_BASE_URL", "LLM_MODEL"):
        monkeypatch.delenv(name, raising=False)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def contains_key(value, forbidden):
    if isinstance(value, dict):
        return any(key in forbidden or contains_key(item, forbidden) for key, item in value.items())
    if isinstance(value, list):
        return any(contains_key(item, forbidden) for item in value)
    return False


def supplied_profile(messages):
    texts = [message.get("content", "") for message in messages if isinstance(message.get("content"), str)]
    matching = [text for text in texts if "<fixed_preferences>" in text]
    assert len(matching) == 1, "Exactly one personal profile must reach a participant call."
    assert matching[0].count("<fixed_preferences>") == 1
    payload = matching[0].split("<fixed_preferences>", 1)[1].split("</fixed_preferences>", 1)[0]
    profile = json.loads(payload)
    assert set(profile) == {"risk_aversion", "advantageous_inequality_aversion"}
    return profile


def spy_models(monkeypatch, module=version):
    original = module.ScriptedLLMClient.call
    state = {"calls": [], "active": 0, "peak": 0}

    async def call(self, model=None, messages=None, **kwargs):
        messages = list(messages or [])
        turn = self._participant_turn(messages) if self.kind == "participant" else None
        record = {
            "kind": self.kind, "client": id(self), "turn": copy.deepcopy(turn),
            "messages": copy.deepcopy(messages),
        }
        state["calls"].append(record)
        if self.kind == "participant":
            state["active"] += 1
            state["peak"] = max(state["peak"], state["active"])
        try:
            # Yield at the real model boundary so simultaneous agents can be observed.
            await asyncio.sleep(0.003)
            response = await original(self, model=model, messages=messages, **kwargs)
        finally:
            if self.kind == "participant":
                state["active"] -= 1
        record["tool_calls"] = [
            {"name": item.function.name, "arguments": json.loads(item.function.arguments)}
            for item in (response.choices[0].message.tool_calls or [])
        ]
        return response

    monkeypatch.setattr(module.ScriptedLLMClient, "call", call)
    return state


def make_profiles(path, participants=4):
    levels = ("low", "medium", "high")
    profiles = {
        str(agent): {
            "risk_aversion": levels[(agent - 1) % 3],
            "advantageous_inequality_aversion": levels[((agent - 1) // 3) % 3],
        }
        for agent in range(1, participants + 1)
    }
    path.write_text(json.dumps({"schema_version": 1, "profiles": profiles}), encoding="utf-8")
    return profiles


def check_contexts(root, manifest, calls, profiles):
    from custom.agents.fixed_preferences_v1.profiles import profile_sha256

    participant_calls = [item for item in calls if item["kind"] == "participant"]
    contexts = rows(root / "participant_contexts.jsonl")
    assert len(contexts) == len(participant_calls) == manifest["participant_model_requests"]
    clients = defaultdict(set)
    for call in participant_calls:
        agent, obs = call["turn"]
        profile = profiles[str(agent)]
        assert supplied_profile(call["messages"]) == profile
        clients[call["client"]].add(agent)
        assert obs["agent_id"] == agent
        assert not contains_key(obs, {
            "fixed_preferences", "risk_aversion", "advantageous_inequality_aversion",
            "treatment", "analysis_phase", "session_id", "preference_seed", "profiles",
        })
        members = {agent, obs["partner_id"]}
        assert all(
            {record["agent_id"], record["partner_id"]} == members
            and record["period"] < obs["current_period"]
            for record in obs["history"]
        )
        # Every real provider-bound message list must have a matching audit entry.
        matches = [
            row for row in contexts
            if row["agent_id"] == agent and row["model_messages"] == call["messages"]
        ]
        assert len(matches) == 1
        row = matches[0]
        assert row["preferences"] == profile
        assert row["preferences_sha256"] == profile_sha256(profile)
        assert row["period"] == obs["current_period"]
        assert row["task"] == obs["task"]
    assert len(clients) == manifest["agents"]
    assert all(len(agents) == 1 for agents in clients.values())
    for call in calls:
        if call["kind"] == "router":
            # This scripted fixture does not repeat preferences in its reason.
            # A real participant may mention its own preferences voluntarily;
            # the invariant is that no private profile is injected by routing.
            assert "<fixed_preferences>" not in json.dumps(call["messages"])


def check_transfers(root, profiles):
    from custom.agents.fixed_preferences_v1.profiles import profile_sha256

    actions = rows(root / "agent_actions.jsonl")
    transfers = [row for row in actions if row["action"] == "submit_decision"]
    assert transfers and all(row["ok"] for row in transfers)
    assert len({row["decision_id"] for row in transfers}) == len(transfers)
    for row in transfers:
        obs = row["participant_model_message"]["observation"]
        assert obs["agent_id"] == row["agent_id"]
        assert obs["task"] == "transfer" and obs["had_transfer_opportunity"] is True
        assert obs["decision_id"] == row["decision_id"]
        assert row["preferences_sha256"] == profile_sha256(profiles[str(row["agent_id"])])
        assert row["decision_source"] == "participant_model"
        amount = row["submitted_amount"]
        assert type(amount) is int
        assert 0 <= amount <= obs["legal_transfer_range"]["max"]
        assert amount == row["participant_model_output"]["amount"] == row["environment_receipt"]["amount"]
    return transfers


def test_twenty_agents_restore_only_their_fixed_preferences_without_memory(tmp_path, monkeypatch):
    spy = spy_models(monkeypatch)
    root = tmp_path / "twenty"
    manifest = version.run_experiment(
        backend="scripted", participants=20, agent_concurrency=20,
        completed_period_limit=2, request_limit=100, run_dir=root,
        preference_seed=19,
    )
    assert manifest["success"] is True
    assert manifest["experiment_version"] == "fixed_preferences_v1"
    assert manifest["agent_class"] == "FixedPreferenceParticipant"
    assert manifest["router"] == "ReActRouter"
    assert manifest["enable_memory"] is False
    assert manifest["participant_deliberation"] is False
    assert manifest["participant_model_requests"] == 40
    assert manifest["router_model_requests"] == 20
    assert manifest["actual_model_requests"] == 60
    assert spy["peak"] == 20
    assert manifest["preferences_file"] == "participant_profiles.json"
    profile_file = root / manifest["preferences_file"]
    assert manifest["preferences_sha256"] == hashlib.sha256(profile_file.read_bytes()).hexdigest()
    document = read_json(profile_file)
    profiles = document["profiles"]
    assert set(profiles) == {str(agent) for agent in range(1, 21)}
    assert len({tuple(sorted(profile.items())) for profile in profiles.values()}) == 9
    check_contexts(root, manifest, spy["calls"], profiles)
    transfers = check_transfers(root, profiles)
    assert len(transfers) == 20
    for agent in range(1, 21):
        workspace = root / "agents" / f"agent_{agent:04d}"
        config = read_json(workspace / "config.json")
        assert config["fixed_preferences"] == profiles[str(agent)]
        assert config["enable_memory"] is False
        assert read_json(workspace / "AGENT.json")["agent_class"] == "FixedPreferenceParticipant"


@pytest.mark.parametrize("treatment", ["25V", "25F"])
def test_insurance_tasks_and_withdrawal_keep_the_same_private_profiles(tmp_path, monkeypatch, treatment):
    spy = spy_models(monkeypatch)
    source = tmp_path / "profiles.json"
    expected = make_profiles(source)
    root = tmp_path / treatment
    kwargs = {"forced_assignment_probabilities": (0.5, 0.25, 0.25)} if treatment == "25F" else {}
    manifest = version.run_experiment(
        backend="scripted", treatment=treatment, completed_period_limit=31,
        agent_concurrency=4, request_limit=300, run_dir=root, profiles_file=source,
        **kwargs,
    )
    assert manifest["success"] is True
    assert read_json(root / "participant_profiles.json")["profiles"] == expected
    assert manifest["completed_periods"] == list(range(1, 32))
    check_contexts(root, manifest, spy["calls"], expected)
    assert len(check_transfers(root, expected)) == 62
    participant = [item for item in spy["calls"] if item["kind"] == "participant"]
    insurance = [item for item in participant if item["turn"][1]["task"] == "insurance"]
    assert len(insurance) == (4 if treatment == "25V" else 0)
    assert manifest["participant_model_requests"] == (128 if treatment == "25V" else 124)
    assert manifest["router_model_requests"] == (66 if treatment == "25V" else 62)
    for call in participant:
        obs = call["turn"][1]
        if obs["current_period"] < 16:
            assert "Your current observation offers an insurance purchase decision" not in json.dumps(call["messages"])
        if obs["current_period"] == 31:
            assert obs["insured"] is False and "announcement" in obs


def test_baseline_and_new_entry_points_do_not_change_each_other(tmp_path):
    first = baseline.run_experiment(backend="scripted", completed_period_limit=2, run_dir=tmp_path / "old-first")
    middle = version.run_experiment(backend="scripted", completed_period_limit=2, run_dir=tmp_path / "new")
    last = baseline.run_experiment(backend="scripted", completed_period_limit=2, run_dir=tmp_path / "old-last")
    for manifest in (first, middle, last):
        assert manifest["success"] is True
    assert first["agent_class"] == last["agent_class"] == "RiskSharingParticipant"
    assert middle["agent_class"] == "FixedPreferenceParticipant"
    assert first["prompt_sha256"] == last["prompt_sha256"] != middle["prompt_sha256"]
    assert first["actual_model_requests"] == middle["actual_model_requests"] == last["actual_model_requests"]
    # The unchanged scripted fixture deliberately ignores preferences. Equal
    # outcomes here check the unchanged mechanism, not behavioral calibration.
    def outcomes(directory):
        return [
            {key: row[key] for key in ("agent_id", "pair_id", "period", "random_income", "transfer_sent", "final_income")}
            for row in rows(directory / "period_records.jsonl")
        ]
    assert outcomes(tmp_path / "old-first") == outcomes(tmp_path / "new") == outcomes(tmp_path / "old-last")
    assert not (tmp_path / "old-last" / "participant_profiles.json").exists()
    assert "fixed_preferences" not in read_json(tmp_path / "old-last" / "agents" / "agent_0001" / "config.json")


def test_two_high_income_participant_outputs_reach_environment_unchanged(tmp_path, monkeypatch):
    spy = spy_models(monkeypatch)
    root = tmp_path / "provenance"
    manifest = version.run_experiment(backend="scripted", completed_period_limit=1, run_dir=root)
    profiles = read_json(root / "participant_profiles.json")["profiles"]
    transfers = check_transfers(root, profiles)
    assert manifest["success"] is True and len(transfers) == 2
    assert len({row["submitted_amount"] for row in transfers}) == 2
    for row in transfers:
        outputs = [
            tool["arguments"]["instruction"]
            for call in spy["calls"]
            if call["kind"] == "participant" and call["turn"][0] == row["agent_id"]
            for tool in call["tool_calls"] if tool["name"] == "ask_env"
        ]
        assert len(outputs) == 1
        assert f"amount: {row['submitted_amount']}" in outputs[0]
        assert row["decision_id"] in outputs[0]
    waiting = [call for call in spy["calls"] if call["kind"] == "participant" and call["turn"][1]["task"] == "waiting"]
    assert len(waiting) == 2
    assert all([tool["name"] for tool in call["tool_calls"]] == ["finish"] for call in waiting)


@pytest.mark.parametrize("invalid", ["missing_agent", "unknown_level", "extra_field"])
def test_invalid_profiles_fail_before_loading_runtime_or_creating_run(tmp_path, monkeypatch, invalid):
    source = tmp_path / "invalid.json"
    profiles = make_profiles(source)
    if invalid == "missing_agent":
        del profiles["4"]
    elif invalid == "unknown_level":
        profiles["1"]["risk_aversion"] = "always buy insurance"
    else:
        profiles["1"]["amount"] = 100
    source.write_text(json.dumps({"schema_version": 1, "profiles": profiles}), encoding="utf-8")

    def unexpected_runtime(*args, **kwargs):
        pytest.fail("Invalid preferences must fail before credentials or model initialization.")

    monkeypatch.setattr(version, "load_runtime_config", unexpected_runtime)
    with pytest.raises((TypeError, ValueError)):
        version.run_experiment(backend="scripted", profiles_file=source, run_dir=tmp_path / "run")
    assert not (tmp_path / "run").exists()


def test_25f_still_requires_explicit_assignment_probabilities(tmp_path, monkeypatch):
    def unexpected_runtime(*args, **kwargs):
        pytest.fail("Missing 25F probabilities must fail before runtime initialization.")

    monkeypatch.setattr(version, "load_runtime_config", unexpected_runtime)
    with pytest.raises((TypeError, ValueError)):
        version.run_experiment(backend="scripted", treatment="25F", run_dir=tmp_path / "run")
    assert not (tmp_path / "run").exists()


@pytest.mark.parametrize("field", ["amount", "agent_id", "decision_id"])
def test_preference_version_still_rejects_router_changes(tmp_path, monkeypatch, field):
    class TamperingRouter(version.ScriptedLLMClient):
        async def call(self, model=None, messages=None, **kwargs):
            response = await super().call(model=model, messages=messages, **kwargs)
            if self.kind == "router":
                for tool in response.choices[0].message.tool_calls:
                    if tool.function.name == "submit_decision":
                        args = json.loads(tool.function.arguments)
                        if field == "amount":
                            args[field] += 1
                        elif field == "agent_id":
                            args[field] = args[field] % 4 + 1
                        else:
                            args[field] = "router-generated-token"
                        tool.function.arguments = json.dumps(args)
            return response

    monkeypatch.setattr(version, "ScriptedLLMClient", TamperingRouter)
    root = tmp_path / field
    manifest = version.run_experiment(
        backend="scripted", completed_period_limit=1, run_dir=root,
        router_steps=2, agent_turns=2, request_limit=200,
    )
    assert manifest["success"] is False
    assert manifest["completed_periods"] == []
    assert manifest["actual_model_requests"] <= 200
    submissions = [row for row in rows(root / "agent_actions.jsonl") if row["action"] == "submit_decision"]
    assert submissions and all(not row["ok"] for row in submissions)
    assert rows(root / "period_records.jsonl") == []


@pytest.mark.parametrize("tamper", ["change", "delete"])
def test_invalid_actual_preference_input_stops_before_any_model_request(tmp_path, monkeypatch, tamper):
    from custom.agents.fixed_preferences_v1.participant import FixedPreferenceParticipant

    original = FixedPreferenceParticipant.build_react_messages
    inner_calls = []

    def corrupted_messages(self, **kwargs):
        messages = original(self, **kwargs)
        profile = supplied_profile(messages)
        system = messages[0]["content"]
        before, remainder = system.split("<fixed_preferences>", 1)
        _payload, after = remainder.split("</fixed_preferences>", 1)
        if tamper == "delete":
            messages[0]["content"] = before + after
        else:
            profile["risk_aversion"] = "high" if profile["risk_aversion"] != "high" else "low"
            messages[0]["content"] = before + "<fixed_preferences>" + json.dumps(profile) + "</fixed_preferences>" + after
        return messages

    async def unexpected_inner(self, **kwargs):
        inner_calls.append(self.kind)
        raise AssertionError("A mismatched or missing profile must not reach a model provider.")

    monkeypatch.setattr(FixedPreferenceParticipant, "build_react_messages", corrupted_messages)
    monkeypatch.setattr(version.ScriptedLLMClient, "call", unexpected_inner)
    root = tmp_path / tamper
    manifest = version.run_experiment(
        backend="scripted", completed_period_limit=1, run_dir=root, agent_concurrency=4,
    )
    assert manifest["success"] is False
    assert manifest["completed_periods"] == []
    assert manifest["actual_model_requests"] == 0
    assert manifest["participant_model_requests"] == 0
    assert manifest["router_model_requests"] == 0
    assert inner_calls == []
    assert rows(root / "period_records.jsonl") == []
    assert not any(row["action"] == "submit_decision" for row in rows(root / "agent_actions.jsonl"))


def test_optional_deliberation_preserves_profile_in_review_and_choice(tmp_path, monkeypatch):
    spy = spy_models(monkeypatch)
    root = tmp_path / "with-review"
    manifest = version.run_experiment(
        backend="scripted", completed_period_limit=1, run_dir=root,
        participant_deliberation=True, agent_concurrency=4,
    )
    assert manifest["success"] is True
    assert manifest["enable_memory"] is False
    assert manifest["participant_deliberation"] is True
    assert manifest["participant_deliberation_requests"] == 2
    assert manifest["participant_model_requests"] == 6
    assert manifest["router_model_requests"] == 2
    assert manifest["actual_model_requests"] == 8
    profiles = read_json(root / "participant_profiles.json")["profiles"]
    check_contexts(root, manifest, spy["calls"], profiles)
    assert len(check_transfers(root, profiles)) == 2
    contexts = rows(root / "participant_contexts.jsonl")
    assert sum(row["phase"] == "deliberation" for row in contexts) == 2
    assert sum(row["phase"] == "decision" for row in contexts) == 4
    reviews = rows(root / "participant_deliberations.jsonl")
    assert len(reviews) == 2
    for review in reviews:
        assert supplied_profile(review["model_messages"]) == profiles[str(review["agent_id"])]
        choices = [
            row for row in contexts
            if row["phase"] == "decision" and row["agent_id"] == review["agent_id"]
            and row["decision_id"] == review["decision_id"]
        ]
        assert len(choices) == 1
        assert any(
            message.get("role") == "assistant" and message.get("content") == review["review_text"]
            for message in choices[0]["model_messages"]
        )
    for agent in range(1, 5):
        config = read_json(root / "agents" / f"agent_{agent:04d}" / "config.json")
        assert config["enable_memory"] is False
        assert config["fixed_preferences"] == profiles[str(agent)]


def test_exported_population_can_be_reused_across_treatments(tmp_path):
    first_root = tmp_path / "baseline"
    first = version.run_experiment(
        backend="scripted", completed_period_limit=1, run_dir=first_root,
        preference_seed=93,
    )
    exported = first_root / "participant_profiles.json"
    document = read_json(exported)
    assert set(document) == {"schema_version", "profiles"}
    second_root = tmp_path / "25v"
    second = version.run_experiment(
        backend="scripted", treatment="25V", completed_period_limit=1,
        run_dir=second_root, profiles_file=exported, preference_seed=999,
    )
    assert first["success"] is True and second["success"] is True
    assert first["treatment"] == "baseline" and second["treatment"] == "25V"
    assert first["preferences_sha256"] == second["preferences_sha256"]
    assert read_json(second_root / "participant_profiles.json") == document
    assert second["preference_assignment"]["source"] == "file"
    assert second["preference_assignment"]["preference_seed"] is None
    for agent in range(1, 5):
        relative = f"agents/agent_{agent:04d}/config.json"
        first_config, second_config = read_json(first_root / relative), read_json(second_root / relative)
        assert first_config["fixed_preferences"] == second_config["fixed_preferences"]
        assert first_config["fixed_preferences_sha256"] == second_config["fixed_preferences_sha256"]
