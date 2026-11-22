"""Build and validate a reproducible 9 x 9 ordered preference-pair design.

Each job is a fresh two-participant session. Research labels live in the batch
plan, never in participant observations or in the strict preference profiles.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from custom.agents.fixed_preferences_v1.profiles import FIELDS, LEVELS
from custom.insurance_assignment import validate_assignment_configuration
from protocol.paper_spec import Treatment


PLAN_FILE = "plan.json"
EXPERIMENT_VERSION = "fixed_preferences_memory_v1"
TREATMENTS = ("baseline", "25v", "25f")
SEED_STRATEGY = "sha256_independent_treatment_combination_repetition_v1"
_JOB_ID = re.compile(r"(?:baseline|25v|25f)__P[1-9]__P[1-9]__r[0-9]{2,}\Z")
_LEVEL_LABELS = {"low": "低", "medium": "中", "high": "高"}


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def _json_value(value: Any, name: str) -> Any:
    """Copy pure JSON data without silently coercing keys or Python objects."""
    def check(item: Any) -> None:
        if isinstance(item, dict):
            if any(type(key) is not str for key in item):
                raise ValueError(f"{name} must use string JSON object keys")
            for child in item.values():
                check(child)
        elif isinstance(item, list):
            for child in item:
                check(child)
        elif item is not None and type(item) not in (str, int, float, bool):
            raise ValueError(f"{name} must contain only JSON values")
    check(value)
    try:
        return json.loads(_json_bytes(value))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain only finite JSON values") from exc


def _seed(master_seed: int, treatment: str, combination: str, repetition: int, domain: str) -> int:
    material = [SEED_STRATEGY, master_seed, treatment, combination, repetition, domain]
    return int.from_bytes(hashlib.sha256(_json_bytes(material)).digest(), "big")


def _personalities() -> list[dict[str, str]]:
    return [
        {"personality_id": f"P{index + 1}", FIELDS[0]: risk, FIELDS[1]: inequality,
         "label": f"风险厌恶：{_LEVEL_LABELS[risk]}；优势不平等厌恶：{_LEVEL_LABELS[inequality]}"}
        for index, (risk, inequality) in enumerate(
            (risk, inequality) for risk in LEVELS for inequality in LEVELS
        )
    ]


def build_plan(
    *, repetitions: int = 10,
    treatments: Sequence[str] = TREATMENTS,
    master_seed: int = 2026,
    forced_assignment_probabilities: Sequence[float] | None = None,
    run_settings: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Enumerate all ordered pairs; assign A/B to the environment's real slots.

    Seeds include treatment, so repeated runs and treatments have independent
    pseudorandom income/termination streams. No shared-shock matching is implied.
    """
    if type(repetitions) is not int or repetitions < 1:
        raise ValueError("repetitions must be a positive integer")
    if type(master_seed) is not int:
        raise ValueError("master_seed must be an integer")
    if (not isinstance(treatments, Sequence) or isinstance(treatments, (str, bytes))
            or not treatments or any(type(item) is not str or item not in TREATMENTS for item in treatments)):
        raise ValueError("treatments must be a nonempty sequence of baseline, 25v, 25f")
    if len(set(treatments)) != len(treatments):
        raise ValueError("treatments must not contain duplicates")
    probabilities = validate_assignment_configuration(
        Treatment.FORCED_INSURANCE_25 if "25f" in treatments else Treatment.BASELINE,
        forced_assignment_probabilities,
    )
    if run_settings is None:
        run_settings = {}
    if not isinstance(run_settings, dict):
        raise ValueError("run_settings must be a JSON object")
    settings = _json_value(run_settings, "run_settings")
    personalities = _personalities()
    jobs = []
    seeds: set[int] = set()
    for treatment in treatments:
        for first in personalities:
            for second in personalities:
                combination = f"{first['personality_id']}__{second['personality_id']}"
                for repetition in range(1, repetitions + 1):
                    seed = _seed(master_seed, treatment, combination, repetition, "environment")
                    assignment_seed = (_seed(master_seed, treatment, combination, repetition, "assignment")
                                       if treatment == "25f" else None)
                    for value in (seed, assignment_seed):
                        if value is None:
                            continue
                        if value in seeds:
                            raise ValueError("Derived seed collision; change master_seed")
                        seeds.add(value)
                    # RiskSharingEnv shuffles IDs before it fixes the pair. Keep
                    # the design's first/second slots aligned with 25F allocation.
                    members = [1, 2]
                    random.Random(seed).shuffle(members)
                    profiles = {
                        str(agent_id): {field: profile[field] for field in FIELDS}
                        for agent_id, profile in zip(members, (first, second))
                    }
                    jobs.append({
                        "job_id": f"{treatment}__{combination}__r{repetition:02d}",
                        "treatment": treatment, "combination_id": combination,
                        "repetition": repetition, "seed": seed,
                        "assignment_seed": assignment_seed, "pair_members": members,
                        "first": {"agent_id": members[0], **first},
                        "second": {"agent_id": members[1], **second},
                        "profiles": profiles,
                    })
    plan = {
        "schema_version": 1, "experiment_version": EXPERIMENT_VERSION,
        "repetitions": repetitions, "treatments": list(treatments),
        "master_seed": master_seed, "seed_strategy": SEED_STRATEGY,
        "forced_assignment_probabilities": list(probabilities) if probabilities is not None else None,
        "run_settings": settings, "personalities": personalities, "jobs": jobs,
    }
    return {**plan, "plan_sha256": hashlib.sha256(_json_bytes(plan)).hexdigest()}


def _validate_plan(plan: Any) -> dict[str, Any]:
    if not isinstance(plan, dict):
        raise ValueError("plan must be a JSON object")
    required = {"schema_version", "experiment_version", "repetitions", "treatments", "master_seed",
                "seed_strategy", "forced_assignment_probabilities", "run_settings",
                "personalities", "jobs", "plan_sha256"}
    if set(plan) != required:
        raise ValueError("plan has missing or unknown fields")
    canonical = build_plan(
        repetitions=plan["repetitions"], treatments=plan["treatments"],
        master_seed=plan["master_seed"],
        forced_assignment_probabilities=plan["forced_assignment_probabilities"],
        run_settings=plan["run_settings"],
    )
    # Comparing canonical JSON also distinguishes booleans from integer IDs.
    if _json_bytes(plan) != _json_bytes(canonical):
        raise ValueError("plan differs from its canonical design or checksum")
    return canonical


def _atomic_json(path: Path, value: Any) -> None:
    """Publish a complete JSON document without replacing an existing file."""
    descriptor, temporary = tempfile.mkstemp(prefix=".plan-", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(_json_bytes(value) + b"\n")
        os.link(temporary, path)
    finally:
        os.unlink(temporary)


def write_plan(batch_dir: Path, plan: Mapping[str, Any]) -> None:
    """Create immutable case inputs in a new directory; do not create run dirs."""
    canonical = _validate_plan(plan)
    batch_dir = Path(batch_dir).expanduser()
    batch_dir.mkdir(parents=True, exist_ok=False)
    cases = batch_dir / "cases"
    cases.mkdir()
    for job in canonical["jobs"]:
        if not _JOB_ID.fullmatch(job["job_id"]):
            raise ValueError("unsafe job_id")
        case = cases / job["job_id"]
        case.mkdir()
        _atomic_json(case / "profiles.json", {"schema_version": 1, "profiles": job["profiles"]})
        _atomic_json(case / "job.json", job)
    # The plan is the completion marker: interrupted generation cannot load as
    # a complete batch merely because some case inputs have been written.
    _atomic_json(batch_dir / PLAN_FILE, canonical)


def _unique_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _read_json(path: Path) -> Any:
    if path.is_symlink():
        raise ValueError(f"batch inputs must not be symlinks: {path.name}")
    with path.open(encoding="utf-8") as handle:
        return json.load(handle, object_pairs_hook=_unique_keys,
                         parse_constant=_reject_nonfinite)


def _reject_nonfinite(value: str) -> Any:
    raise ValueError(f"nonfinite JSON value: {value}")


def load_plan(batch_dir: Path, *, job_id: str | None = None) -> dict[str, Any]:
    """Validate the full design and all case inputs, or one worker's inputs.

    A worker may select its job to avoid rereading thousands of unrelated case
    files. The complete plan and checksum are still canonically validated.
    """
    if job_id is not None and (not isinstance(job_id, str) or not _JOB_ID.fullmatch(job_id)):
        raise ValueError("unsafe job_id")
    batch_dir = Path(batch_dir).expanduser()
    if batch_dir.is_symlink():
        raise ValueError("batch directory must not be a symlink")
    plan = _validate_plan(_read_json(batch_dir / PLAN_FILE))
    selected = plan["jobs"] if job_id is None else [job for job in plan["jobs"] if job["job_id"] == job_id]
    if not selected:
        raise ValueError(f"unknown job_id: {job_id}")
    cases = batch_dir / "cases"
    if cases.is_symlink():
        raise ValueError("cases directory must not be a symlink")
    for job in selected:
        # Design validation already guarantees this, but keep the filesystem
        # boundary explicit if canonical generation is extended in the future.
        if not _JOB_ID.fullmatch(job["job_id"]):
            raise ValueError("unsafe job_id")
        case = cases / job["job_id"]
        if case.is_symlink():
            raise ValueError("case directory must not be a symlink")
        expected_profiles = {"schema_version": 1, "profiles": job["profiles"]}
        if _json_bytes(_read_json(case / "job.json")) != _json_bytes(job):
            raise ValueError(f"case job differs from plan: {job['job_id']}")
        if _json_bytes(_read_json(case / "profiles.json")) != _json_bytes(expected_profiles):
            raise ValueError(f"case profiles differ from plan: {job['job_id']}")
    return plan
