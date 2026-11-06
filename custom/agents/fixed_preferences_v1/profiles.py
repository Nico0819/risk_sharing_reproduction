"""Two experimental preference dimensions, without an action-selection policy.

Levels are engineering hypotheses, not estimated economic parameters.  Profile
assignment uses its own RNG so it cannot alter income, pairing or termination.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Mapping
from pathlib import Path
from typing import Any


VERSION = "fixed_preferences_v1"
LEVELS = ("low", "medium", "high")
FIELDS = ("risk_aversion", "advantageous_inequality_aversion")

_DESCRIPTIONS = {
    "risk_aversion": {
        "low": (
            "You place relatively little weight on uncertainty in your own final "
            "payoff. You are relatively willing to accept uncertain outcomes "
            "when their expected payoff is attractive."
        ),
        "medium": (
            "You place moderate weight on uncertainty in your own final payoff. "
            "Balance expected payoff against the possibility of worse outcomes "
            "and the cost of reducing that uncertainty."
        ),
        "high": (
            "You place substantial weight on uncertainty in your own final payoff. "
            "Avoiding worse possible outcomes has substantial value to you, "
            "while the cost of reducing uncertainty still matters."
        ),
    },
    "advantageous_inequality_aversion": {
        "low": (
            "You feel relatively little discomfort when your own income after "
            "transfers exceeds your partner's. Reducing an income advantage "
            "has relatively little value to you compared with keeping income."
        ),
        "medium": (
            "You feel moderate discomfort when your own income after transfers "
            "exceeds your partner's. Reducing an income advantage has some value "
            "to you, which you weigh against the income you would give up."
        ),
        "high": (
            "You feel substantial discomfort when your own income after transfers "
            "exceeds your partner's. Reducing an income advantage has substantial "
            "value to you, while the income you would give up still matters."
        ),
    },
}


def validate_profile(profile: Mapping[str, Any]) -> dict[str, str]:
    """Return a fresh, canonical profile; reject extra fields and free text."""

    if not isinstance(profile, Mapping) or set(profile) != set(FIELDS):
        raise ValueError("fixed preferences must contain exactly: " + ", ".join(FIELDS))
    result = {}
    for field in FIELDS:
        value = profile[field]
        if not isinstance(value, str) or value not in LEVELS:
            raise ValueError(f"{field} must be one of: {', '.join(LEVELS)}")
        result[field] = value
    return result


def profile_sha256(profile: Mapping[str, Any]) -> str:
    canonical = json.dumps(validate_profile(profile), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def render_profile(profile: Mapping[str, Any]) -> str:
    """Render only this participant's stable preferences, never study metadata."""

    profile = validate_profile(profile)
    return (
        "Your personal preferences remain the same throughout this experiment.\n"
        "<fixed_preferences>\n"
        + json.dumps(profile, sort_keys=True)
        + "\n</fixed_preferences>\n"
        + "\n".join(_DESCRIPTIONS[field][profile[field]] for field in FIELDS)
        + "\nThese preferences describe what matters to you; they do not specify "
        "an action, a transfer amount, or a requirement to equalize incomes. "
        "Make your own choices using the current observation and your pair's "
        "completed history. Your partner's preferences are unknown to you."
    )


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def build_profiles(
    participants: int,
    preference_seed: int,
    profiles_file: Path | None = None,
) -> tuple[dict[int, dict[str, str]], dict[str, Any]]:
    """Create or validate a complete population before any model calls.

    Generated populations balance all nine combinations (count difference at
    most one). Latin ordering also balances each dimension's three levels for
    partial blocks. The population is then shuffled onto Agent IDs. These are
    synthetic experimental conditions, not a claimed human population sample.
    """

    if type(participants) is not int or participants < 2 or participants % 2:
        raise ValueError("participants must be an even integer of at least 2")
    if type(preference_seed) is not int:
        raise ValueError("preference_seed must be an integer")
    if profiles_file is not None:
        path = Path(profiles_file).expanduser().resolve()
        raw = path.read_bytes()
        payload = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_object)
        if not isinstance(payload, dict) or set(payload) != {"schema_version", "profiles"}:
            raise ValueError("profiles file must contain exactly schema_version and profiles")
        if type(payload["schema_version"]) is not int or payload["schema_version"] != 1:
            raise ValueError("profiles file schema_version must be 1")
        supplied = payload["profiles"]
        expected_ids = {str(agent_id) for agent_id in range(1, participants + 1)}
        if not isinstance(supplied, dict) or set(supplied) != expected_ids:
            raise ValueError("profiles file must specify exactly one profile for every Agent ID")
        profiles = {
            agent_id: validate_profile(supplied[str(agent_id)])
            for agent_id in range(1, participants + 1)
        }
        metadata = {
            "source": "file",
            "design": "explicit_per_agent",
            "preference_seed": None,
            "source_file": str(path),
            "source_sha256": hashlib.sha256(raw).hexdigest(),
        }
    else:
        rng = random.Random(preference_seed)
        risk_levels = list(LEVELS)
        inequality_levels = list(LEVELS)
        offsets = list(range(3))
        rng.shuffle(risk_levels)
        rng.shuffle(inequality_levels)
        rng.shuffle(offsets)
        combinations = [
            {FIELDS[0]: risk_levels[index], FIELDS[1]: inequality_levels[(index + offset) % 3]}
            for offset in offsets
            for index in range(3)
        ]
        population = [dict(combinations[index % len(combinations)]) for index in range(participants)]
        rng.shuffle(population)
        profiles = {agent_id: profile for agent_id, profile in enumerate(population, 1)}
        metadata = {
            "source": "generated",
            "design": "balanced_3x3_synthetic",
            "preference_seed": preference_seed,
        }
    return profiles, {"version": VERSION, "schema_version": 1, **metadata}
