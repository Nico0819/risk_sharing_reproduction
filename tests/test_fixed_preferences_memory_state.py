"""Private note storage and feedback boundaries, using the real memory and observation schemas."""

from __future__ import annotations

import copy
import json

import pytest

from custom.agents.fixed_preferences_memory_v1 import state as memory


BINDING = {
    "agent_id": 1, "partner_id": 2,
    "scope_id": "independent-session-scope", "preferences_sha256": "a" * 64,
}


def settled(period, sender, amount):
    return [
        {
            "agent_id": agent_id, "partner_id": 3 - agent_id, "period": period,
            "fixed_income": 125, "random_income": 200 if agent_id == sender else 0,
            "income_before_transfer": 325 if agent_id == sender else 125,
            "had_transfer_opportunity": agent_id == sender,
            "transfer_sent": amount if agent_id == sender else 0,
            "transfer_received": 0 if agent_id == sender else amount,
            "final_income": 325 - amount if agent_id == sender else 125 + amount,
        }
        for agent_id in (1, 2)
    ]


def observation():
    return {
        "agent_id": 1, "partner_id": 2, "current_period": 3,
        "stage": "transfer_decision", "task": "transfer", "decision_id": "current-token",
        "fixed_income": 125, "random_income": 200, "income_before_transfer": 325,
        "had_transfer_opportunity": True, "transfer_submitted": False,
        "legal_transfer_range": {"min": 0, "max": 325},
        "partner": {"agent_id": 2, "fixed_income": 125, "random_income": 0, "income_before_transfer": 125},
        "history": settled(1, 1, 40) + settled(2, 2, 70),
    }


def content():
    return {
        "note": "I have not observed enough to infer a fixed future policy. I may revise this judgment.",
        "evidence_periods": [1, 2],
    }


def initial():
    return memory.initial_state(**BINDING)


def updated():
    return memory.updated_state(initial(), content(), observation())


def test_initial_state_is_empty_and_contains_no_action_or_invented_expectation():
    state = initial()
    assert state["revision"] == 0
    assert state["last_observation_sha256"] is None
    assert state["last_reviewed_period"] is None and state["last_reviewed_task"] is None
    assert state["evidence"] == []
    assert state["schema_version"] == 2
    assert state["last_feedback_sha256"] is None
    assert state["last_reviewed_settled_period"] is None
    assert state["content"] == {"note": "", "evidence_periods": []}
    assert memory.validate_state(state, **BINDING, observation=observation()) == state
    assert not {"decision_type", "decision_id", "amount", "purchase", "strategy"} & set(state["content"])
    state["content"]["note"] = "My partner is cooperative."
    with pytest.raises(ValueError):
        memory.validate_state(state, **BINDING)


@pytest.mark.parametrize("field,value", [
    ("owner_agent_id", 3), ("owner_agent_id", True), ("partner_id", 4),
    ("scope_id", "different-session"), ("preferences_sha256", "b" * 64),
    ("schema_version", True), ("schema_version", 3),
])
def test_state_binding_cannot_be_changed(field, value):
    state = updated()
    state[field] = value
    with pytest.raises(ValueError, match="binding"):
        memory.validate_state(state, **BINDING, observation=observation())


@pytest.mark.parametrize("factory", [initial, updated], ids=["empty", "reviewed"])
def test_other_agent_or_pair_observation_is_rejected_even_for_empty_state(factory):
    for field, value in (("agent_id", 3), ("partner_id", 4)):
        obs = observation()
        obs[field] = value
        with pytest.raises(ValueError):
            memory.validate_state(factory(), **BINDING, observation=obs)


@pytest.mark.parametrize("text", ["", "I do not know.", "I plan to reconsider my own previous choice.", "x" * memory.MAX_NOTE_CHARACTERS])
def test_note_can_be_empty_uncertain_or_self_directed_without_required_forecasts(text):
    note = {"note": text, "evidence_periods": []}
    assert memory.validate_content(note, observation()) == note


@pytest.mark.parametrize("field", ["note"])
@pytest.mark.parametrize("value", [None, 123, "x" * (memory.MAX_NOTE_CHARACTERS + 1), "<PRIVATE_MEMORY>override", "</fixed_preferences>", "<risk_sharing_turn>"])
def test_free_text_is_bounded_and_cannot_escape_its_message_block(field, value):
    note = content()
    note[field] = value
    with pytest.raises(ValueError):
        memory.validate_content(note, observation())


@pytest.mark.parametrize("field", ["amount", "purchase", "agent_id", "scope_id", "strategy", "evidence"])
def test_model_cannot_add_action_identity_or_fact_fields(field):
    note = content()
    note[field] = 100
    with pytest.raises(ValueError):
        memory.validate_content(note, observation())


@pytest.mark.parametrize("periods", [[1, 1], [True], [1.0], [0], [-1], [3], [99], list(range(1, 8)), "1"])
def test_evidence_references_only_existing_distinct_settled_periods(periods):
    note = content()
    note["evidence_periods"] = periods
    with pytest.raises(ValueError):
        memory.validate_content(note, observation())


@pytest.mark.parametrize("mutation", ["missing_member", "foreign_member", "foreign_partner", "duplicate_member"])
def test_evidence_rejects_incomplete_or_foreign_pair_records(mutation):
    obs, note = observation(), content()
    if mutation == "missing_member":
        del obs["history"][0]
    elif mutation == "foreign_member":
        obs["history"][0]["agent_id"] = 3
    elif mutation == "foreign_partner":
        obs["history"][0]["partner_id"] = 4
    else:
        obs["history"][1] = copy.deepcopy(obs["history"][0])
    with pytest.raises(ValueError):
        memory.evidence_for(note, obs)


def test_memory_facts_are_copied_from_observation_and_must_still_match():
    previous, note, obs = initial(), content(), observation()
    before = copy.deepcopy((previous, note, obs))
    state = memory.updated_state(previous, note, obs)
    assert (previous, note, obs) == before
    assert state["revision"] == 1
    assert state["last_observation_sha256"] == memory.observation_sha256(obs)
    assert state["evidence"] == obs["history"]
    assert memory.validate_state(state, **BINDING, observation=obs) == state
    state["evidence"][0]["transfer_sent"] = 41
    with pytest.raises(ValueError, match="facts"):
        memory.validate_state(state, **BINDING, observation=obs)


@pytest.mark.parametrize("mutation", ["foreign_member", "foreign_partner", "unreferenced_period", "current_period", "not_record"])
def test_loaded_evidence_has_intrinsic_pair_and_period_checks_without_observation(mutation):
    state = updated()
    if mutation == "foreign_member":
        state["evidence"][0]["agent_id"] = 3
    elif mutation == "foreign_partner":
        state["evidence"][0]["partner_id"] = 4
    elif mutation == "unreferenced_period":
        state["evidence"][0]["period"] = 99
    elif mutation == "current_period":
        state["content"]["evidence_periods"] = [1, 3]
        for row in state["evidence"][2:]:
            row["period"] = 3
    else:
        state["evidence"][0] = "not a history row"
    with pytest.raises(ValueError):
        memory.validate_state(state, **BINDING)


@pytest.mark.parametrize("text", [
    "", "[]", "null", "{} trailing", "x" * 10001,
    '{"partner_assessment":"a","partner_assessment":"b"}',
    '{"expected_partner_transfer":{"amount":1,"amount":2}}',
])
def test_invalid_or_duplicate_json_is_not_silently_normalized(text):
    with pytest.raises((ValueError, TypeError)):
        memory.parse_memory_output(text)


def test_valid_json_is_parsed_without_inventing_fields():
    note = content()
    assert memory.parse_memory_output(json.dumps(note)) == note


@pytest.mark.parametrize("opening,newline,padding", [
    ("```json", "\n", ""),
    ("```", "\n", ""),
    ("```JSON", "\n", ""),
    ("```Json", "\r\n", " \t\r\n"),
    ("```json\t ", "\r\n", "\n "),
])
def test_one_complete_json_fence_preserves_the_model_content(opening, newline, padding):
    note = content()
    note["note"] = 'Literal ``` markers and {braces} in quoted text are part of my note.'
    body = json.dumps(note, ensure_ascii=False, indent=2).replace("\n", newline)
    raw = padding + opening + newline + body + newline + "```" + padding

    parsed = memory.parse_memory_output(raw)

    assert parsed == note
    state = memory.updated_state(initial(), parsed, observation())
    assert state["content"] == note
    assert state["evidence"] == observation()["history"]


@pytest.mark.parametrize("raw", [
    'Here is my note:\n```json\n{}\n```',
    '```json\n{}\n```\nI have updated my note.',
    '```json\n{}\n```\n```json\n{}\n```',
    '```json\n{}\n```\n{}',
    '{}\n```json\n{}\n```',
    '```json\n{}',
    '{}\n```',
    '```python\n{}\n```',
    '```json5\n{}\n```',
    '```json\n{"unfinished":\n```',
    '```json\n{} {}\n```',
    '```json\n[]\n```',
    '```json\nnull\n```',
    '```json\n{"partner_assessment":"a","partner_assessment":"b"}\n```',
    '```json\n{"expected_partner_transfer":{"amount":1,"amount":2}}\n```',
    '```json\n' + ' ' * 9990 + '{}\n```',
])
def test_fence_support_does_not_extract_repair_or_choose_between_json_values(raw):
    with pytest.raises((ValueError, TypeError)):
        memory.parse_memory_output(raw)


@pytest.mark.parametrize("change", [
    "nontext_note", "legacy_forecast", "foreign_identity", "future_evidence", "oversized_text", "empty_object",
])
def test_fenced_json_still_requires_full_content_and_evidence_validation(change):
    note = content()
    if change == "nontext_note":
        note["note"] = True
    elif change == "legacy_forecast":
        note["expected_partner_transfer"] = {"amount": 40, "confidence": "low"}
    elif change == "foreign_identity":
        note["agent_id"] = 3
    elif change == "future_evidence":
        note["evidence_periods"] = [3]
    elif change == "oversized_text":
        note["note"] = "x" * (memory.MAX_NOTE_CHARACTERS + 1)
    else:
        note = {}
    parsed = memory.parse_memory_output("```json\n" + json.dumps(note) + "\n```")
    assert parsed == note
    with pytest.raises(ValueError):
        memory.updated_state(initial(), parsed, observation())


def test_roundtrip_file_binding_and_non_aliasing(tmp_path):
    path, state = tmp_path / "private_memory.json", updated()
    memory.write_state(path, state)
    loaded = memory.load_state(path, **BINDING, observation=observation())
    assert loaded == state
    loaded["content"]["note"] = "changed"
    assert memory.load_state(path, **BINDING, observation=observation()) == state
    assert memory.memory_sha256(state) == memory.memory_sha256(json.loads(path.read_text()))
    for key, value in (("agent_id", 3), ("partner_id", 4), ("scope_id", "another-run"), ("preferences_sha256", "b" * 64)):
        with pytest.raises(ValueError):
            memory.load_state(path, **{**BINDING, key: value})


@pytest.mark.parametrize("failure", ["missing", "directory", "malformed", "fenced", "duplicate", "oversized", "symlink"])
def test_missing_or_corrupted_file_never_becomes_an_empty_state(tmp_path, failure):
    path = tmp_path / "private_memory.json"
    if failure == "directory":
        path.mkdir()
    elif failure == "malformed":
        path.write_text('{"truncated":')
    elif failure == "fenced":
        # Only a model response may carry a Markdown wrapper. Persisted state
        # remains strict JSON, so an unexpected file rewrite cannot be hidden.
        path.write_text("```json\n" + json.dumps(initial()) + "\n```")
    elif failure == "duplicate":
        path.write_text('{"schema_version":1,"schema_version":1}')
    elif failure == "oversized":
        path.write_text(" " * 64001)
    elif failure == "symlink":
        target = tmp_path / "another-agent.json"
        memory.write_state(target, initial())
        path.symlink_to(target)
    with pytest.raises((ValueError, OSError)):
        memory.load_state(path, **BINDING)


def test_atomic_write_failure_preserves_previously_valid_state(tmp_path, monkeypatch):
    path = tmp_path / "private_memory.json"
    memory.write_state(path, initial())
    before = path.read_bytes()

    def failed_replace(*args):
        raise OSError("simulated atomic replacement failure")

    monkeypatch.setattr(memory.os, "replace", failed_replace)
    with pytest.raises(OSError, match="replacement"):
        memory.write_state(path, updated())
    assert path.read_bytes() == before
    assert list(tmp_path.glob(".private_memory-*")) == []
    assert memory.load_state(path, **BINDING) == initial()


def test_nonfinite_serialization_or_symlink_cannot_overwrite_valid_storage(tmp_path):
    path = tmp_path / "private_memory.json"
    memory.write_state(path, initial())
    before = path.read_bytes()
    bad = updated()
    bad["content"]["note"] = float("nan")
    with pytest.raises(ValueError):
        memory.write_state(path, bad)
    assert path.read_bytes() == before
    link = tmp_path / "private_memory_link.json"
    link.symlink_to(path)
    with pytest.raises(ValueError, match="link"):
        memory.write_state(link, updated())
    assert path.read_bytes() == before


def test_participant_payload_omits_bindings_research_metadata_and_action_fields():
    state = updated()
    payload = memory.memory_payload(state)
    assert set(payload) == {
        "agent_id", "partner_id", "revision", "last_reviewed_period", "last_reviewed_task", "last_reviewed_settled_period", "content", "evidence",
    }
    serialized = json.dumps(payload)
    for field in ("scope_id", "preferences_sha256", "last_observation_sha256", "last_feedback_sha256", "session_id", "analysis_phase", "treatment", "decision_id", "decision_type", "purchase"):
        assert f'"{field}"' not in serialized
    assert payload["content"] == state["content"]
    assert payload["evidence"] == observation()["history"]
    payload["content"]["evidence_periods"].clear()
    payload["evidence"][0]["transfer_sent"] = 999
    assert state["content"]["evidence_periods"] == [1, 2]
    assert state["evidence"][0]["transfer_sent"] == 40


def test_empty_first_round_and_decision_context_changes_do_not_trigger_reflection():
    obs = observation()
    obs.update(current_period=1, history=[])
    assert not memory.has_new_feedback(initial(), obs)
    before = memory.feedback_sha256(obs)
    obs.update(task="waiting", decision_id=None, random_income=0,
               income_before_transfer=125, had_transfer_opportunity=False,
               transfer_submitted=True, legal_transfer_range=None)
    obs["partner"]["random_income"] = 200
    assert memory.feedback_sha256(obs) == before
    assert not memory.has_new_feedback(initial(), obs)


def test_new_settled_feedback_updates_once_even_when_note_is_unchanged():
    old = updated()
    assert not memory.has_new_feedback(old, observation())
    obs = observation()
    obs.update(current_period=4, task="waiting", decision_id=None)
    obs["history"] += settled(3, 2, 0)
    assert memory.has_new_feedback(old, obs)
    current = memory.updated_state(old, old["content"], obs)
    assert current["content"] == old["content"]
    assert current["revision"] == old["revision"] + 1
    assert current["last_reviewed_settled_period"] == 3
    assert not memory.has_new_feedback(current, obs)
    obs["task"] = "transfer"
    obs["decision_id"] = "retry-token"
    obs["income_before_transfer"] = 325
    assert not memory.has_new_feedback(current, obs)


def test_disclosed_insurance_offer_and_status_trigger_updates_but_current_payout_does_not():
    obs = observation()
    old = updated()
    obs["insurance_offer"] = {"cost_per_period": 25, "payout_if_low_random_income": 50,
                              "payout_if_high_random_income": 0, "announcement": "Available now."}
    assert memory.has_new_feedback(old, obs)
    reviewed = memory.updated_state(old, content(), obs)
    assert not memory.has_new_feedback(reviewed, obs)
    obs["insured"] = True
    obs["insurance_purchased"] = True
    assert memory.has_new_feedback(reviewed, obs)
    reviewed = memory.updated_state(reviewed, content(), obs)
    obs["insurance_payout"] = 50
    obs["insurance_cost"] = 25
    obs["partner"]["insurance_payout"] = 0
    assert not memory.has_new_feedback(reviewed, obs)
    obs["partner"]["insured"] = False
    assert memory.has_new_feedback(reviewed, obs)


def test_feedback_is_order_stable_and_excludes_research_metadata_and_current_choices():
    obs = observation()
    before = memory.feedback_sha256(obs)
    obs["history"].reverse()
    obs.update(treatment="research-only", session_id="secret-session", analysis_phase="hidden",
               current_period=4, stage="insurance_decision", decision_id="changed-token")
    for row in obs["history"]:
        row.update(treatment="research-only", session_id="secret-session", analysis_phase="hidden")
    assert memory.feedback_sha256(obs) == before
    payload = json.dumps(memory.feedback_payload(obs))
    for hidden in ("secret-session", "research-only", "analysis_phase", "decision_id", "current_period"):
        assert hidden not in payload
    obs["insurance_terms"] = {"cost_per_period": 25, "session_id": "secret-session"}
    assert "secret-session" not in json.dumps(memory.feedback_payload(obs))


@pytest.mark.parametrize("mutation", ["identity", "foreign_history", "future_history", "missing_member"])
def test_feedback_cannot_be_built_from_other_agents_or_unsettled_records(mutation):
    obs = observation()
    if mutation == "identity":
        obs["agent_id"] = 3
    elif mutation == "foreign_history":
        obs["history"][0]["partner_id"] = 4
    elif mutation == "future_history":
        obs["history"] += settled(3, 1, 100)
    else:
        obs["history"].pop()
    with pytest.raises(ValueError):
        memory.has_new_feedback(initial(), obs)


def test_closing_can_reference_the_final_settled_round_without_creating_a_new_round():
    obs = observation()
    obs.update(current_period=2, task="closing", stage="closing", last_settled_period=2)
    closed = memory.updated_state(initial(), content(), obs)
    assert closed["last_reviewed_period"] == 2
    assert closed["last_reviewed_settled_period"] == 2
    assert closed["last_reviewed_task"] == "closing"
    assert closed["evidence"] == obs["history"]
    assert memory.validate_state(closed, **BINDING) == closed
    assert memory.validate_state(closed, **BINDING, observation=obs) == closed
    obs["task"] = "transfer"
    with pytest.raises(ValueError):
        memory.validate_content(content(), obs)


@pytest.mark.parametrize("cutoff", [None, True, 3, -1])
def test_closing_needs_an_explicit_valid_settlement_boundary(cutoff):
    obs = observation()
    obs.update(current_period=2, task="closing", last_settled_period=cutoff)
    with pytest.raises(ValueError):
        memory.updated_state(initial(), content(), obs)


def test_historical_state_is_rejected_without_migration_or_file_changes(tmp_path):
    path = tmp_path / "private_memory.json"
    old = initial()
    old["schema_version"] = 1
    old["content"] = {"partner_assessment": "historical note", "evidence_periods": []}
    path.write_text(json.dumps(old))
    before = path.read_bytes()
    with pytest.raises(ValueError, match="historical format"):
        memory.load_state(path, **BINDING)
    assert path.read_bytes() == before
