"""Information preservation and factual attribution in the shared reading guide."""

import copy
import json

import pytest

from custom.agents.observation_presentation import render_readable_turn


def settled(period, *, sender=10, amount=40):
    return [
        {
            "agent_id": agent, "partner_id": 16 if agent == 10 else 10,
            "period": period, "fixed_income": 125,
            "random_income": 200 if agent == sender else 0,
            "had_transfer_opportunity": agent == sender,
            "income_before_transfer": 325 if agent == sender else 125,
            "transfer_sent": amount if agent == sender else 0,
            "transfer_received": 0 if agent == sender else amount,
            "final_income": 325 - amount if agent == sender else 125 + amount,
        }
        for agent in (10, 16)
    ]


def observation():
    return {
        "agent_id": 10, "partner_id": 16, "current_period": 2,
        "stage": "transfer_decision", "task": "transfer", "decision_id": "opaque-current-token",
        "fixed_income": 125, "random_income": 200, "income_before_transfer": 325,
        "had_transfer_opportunity": True, "transfer_submitted": False,
        "legal_transfer_range": {"min": 0, "max": 325},
        "partner": {"agent_id": 16, "fixed_income": 125, "random_income": 0, "income_before_transfer": 125},
        "history": settled(1),
    }


def turn(obs):
    return {
        "agent_id": obs["agent_id"], "readonly": False, "question": None,
        "observations": [{"observation": json.dumps(obs), "source": "ask_env"}],
    }


def unpack(text):
    return json.loads(text.split("<risk_sharing_turn>\n", 1)[1].split("\n</risk_sharing_turn>", 1)[0])


def test_readable_turn_is_lossless_and_does_not_mutate_transport_or_observation():
    obs = observation()
    payload = turn(obs)
    before = copy.deepcopy(payload)
    text = render_readable_turn(payload, agent_id=10)
    restored = unpack(text)
    assert payload == before
    assert restored["observations"][0]["observation"] == obs
    assert restored["observations"][0]["source"] == "ask_env"
    assert {k: v for k, v in restored.items() if k != "observations"} == {
        k: v for k, v in payload.items() if k != "observations"
    }
    assert '\\"history\\"' not in text
    assert "Current decision_id (copy exactly): opaque-current-token" in text


def test_history_identifies_actual_sender_when_same_agent_is_high_again():
    text = render_readable_turn(turn(observation()), agent_id=10)
    assert "| 1 | Agent 10 | Agent 16 | 40 | 40 | 0 | 325 | 125 | 285 | 165 |" in text
    assert "your final income = 325 - x; your partner's final income = 125 + x" in text
    assert "Choose your own integer amount x from 0 through 325" in text


def test_history_uses_the_other_participants_perspective_and_marks_waiting():
    obs = observation()
    obs.update(
        agent_id=16, partner_id=10, task="waiting", decision_id=None,
        random_income=0, income_before_transfer=125, had_transfer_opportunity=False,
        legal_transfer_range=None,
        partner={"agent_id": 10, "fixed_income": 125, "random_income": 200, "income_before_transfer": 325},
    )
    text = render_readable_turn(turn(obs), agent_id=16)
    assert "| 1 | Agent 10 | Agent 16 | 40 | 0 | 40 | 125 | 325 | 165 | 285 |" in text
    assert "There is no decision for you to submit" in text
    assert "Choose your own integer amount" not in text
    assert "Current decision_id (copy exactly)" not in text


def test_zero_transfer_still_has_a_sender_and_does_not_become_a_receiver_choice():
    obs = observation()
    obs["history"] = settled(1, sender=16, amount=0)
    text = render_readable_turn(turn(obs), agent_id=10)
    assert "| 1 | Agent 16 | Agent 10 | 0 | 0 | 0 | 125 | 325 | 125 | 325 |" in text
    assert "a receiver's bookkeeping transfer_sent=0 is not a transfer choice" in text


def test_insurance_adjusted_income_is_used_without_reapplying_premium():
    obs = observation()
    obs.update(current_period=30, income_before_transfer=300, insurance_cost=25,
               insurance_payout=0, insured=True, insurance_purchased=True,
               legal_transfer_range={"min": 0, "max": 300})
    obs["partner"].update(insurance_cost=0, insurance_payout=0, insured=False, insurance_purchased=False)
    text = render_readable_turn(turn(obs), agent_id=10)
    assert "your final income = 300 - x; your partner's final income = 125 + x" in text
    assert "Do not subtract the premium or add the payout a second time" in text
    assert "| Insurance premium deducted | 25 | 0 |" in text
    assert "both end at 225" not in text
    assert unpack(text)["observations"][0]["observation"] == obs


def test_offer_is_visible_without_inventing_current_income_or_partner_choice():
    obs = {
        "agent_id": 10, "partner_id": 16, "current_period": 16,
        "stage": "insurance_decision", "task": "insurance", "decision_id": "insurance-token",
        "insurance_submitted": False, "own_purchase": None,
        "legal_insurance_choices": [True, False],
        "insurance_offer": {"cost_per_period": 25, "payout_if_low_random_income": 50,
                            "payout_if_high_random_income": 0, "announcement": "Choose once now."},
        "history": settled(1),
    }
    text = render_readable_turn(turn(obs), agent_id=10)
    assert "The current income draw has not happened" in text
    assert "Premium each period: 25" in text
    assert "Payout if random income is 0: 50" in text
    assert "Your partner's current purchase choice has not been revealed" in text
    assert "Current income account" not in text
    assert unpack(text)["observations"][0]["observation"] == obs
    assert "partner_purchase" not in text


def test_no_future_insurance_information_is_introduced_before_offer():
    text = render_readable_turn(turn(observation()), agent_id=10)
    assert "insurance" not in text.lower()
    assert "treatment" not in text
    assert "session_id" not in text
    assert "analysis_phase" not in text


def test_withdrawal_keeps_assignment_distinct_from_current_coverage():
    obs = observation()
    obs.update(current_period=31, insurance_assigned=True, insured=False,
               insurance_assignment_source="computer_random", insurance_cost=0, insurance_payout=0,
               announcement="Insurance is no longer available from this period.")
    obs["partner"].update(insurance_assigned=False, insured=False, insurance_cost=0, insurance_payout=0)
    text = render_readable_turn(turn(obs), agent_id=10)
    assert "| Insurance currently active | no | no |" in text
    assert "| Was assigned insurance by the computer | yes | no |" in text
    assert "The insurance assignment was random; neither participant chose it" in text
    assert "Insurance is no longer available from this period" in text
    assert "Chose to purchase" not in text


def test_all_history_is_preserved_and_finished_round_is_supported():
    obs = {"agent_id": 10, "partner_id": 16, "current_period": 48, "task": "finished",
           "stage": "finished", "history": [r for p in range(1, 49) for r in settled(p)]}
    text = render_readable_turn(turn(obs), agent_id=10)
    assert "| 1 | Agent 10" in text and "| 48 | Agent 10" in text
    assert text.count("| Agent 10 | Agent 16 |") == 48
    assert unpack(text)["observations"][0]["observation"] == obs


@pytest.mark.parametrize("case", ["caller", "observation", "partner", "foreign_history", "future", "duplicate"])
def test_mismatched_identity_or_history_cannot_be_presented_as_this_agents_facts(case):
    obs = observation()
    payload = turn(obs)
    if case == "caller":
        payload["agent_id"] = 16
    elif case == "observation":
        obs["agent_id"] = 16
    elif case == "partner":
        obs["partner"]["agent_id"] = 99
    elif case == "foreign_history":
        obs["history"][1]["agent_id"] = 99
    elif case == "future":
        obs["history"][0]["period"] = 2
    else:
        obs["history"].append(copy.deepcopy(obs["history"][0]))
    payload["observations"][0]["observation"] = json.dumps(obs)
    with pytest.raises(ValueError):
        render_readable_turn(payload, agent_id=10)
