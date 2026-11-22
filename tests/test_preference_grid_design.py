"""Factorial plans enumerate ordered pairs without leaking research labels."""

from __future__ import annotations

import copy
import hashlib
import json
import socket
from collections import Counter

import pytest

from custom.agents.fixed_preferences_v1.profiles import FIELDS, build_profiles
from experiments.preference_grid.design import build_plan, load_plan, write_plan


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    import custom.config

    monkeypatch.setattr(custom.config, "_parse_dotenv", lambda path: {})
    monkeypatch.setenv("PYTHON_DOTENV_DISABLED", "1")
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")

    def no_network(*args, **kwargs):
        pytest.fail("Preference design tests must not use the network")

    monkeypatch.setattr(socket.socket, "connect", no_network)


def test_default_design_has_2430_jobs_and_ten_repeats_of_all_81_ordered_pairs():
    plan = build_plan(forced_assignment_probabilities=(0.5, 0.25, 0.25))
    assert plan["experiment_version"] == "fixed_preferences_memory_v1"
    assert len(plan["personalities"]) == 9
    assert len(plan["jobs"]) == 2430
    assert len({job["job_id"] for job in plan["jobs"]}) == 2430
    assert Counter(job["treatment"] for job in plan["jobs"]) == {"baseline": 810, "25v": 810, "25f": 810}
    counts = Counter((job["treatment"], job["combination_id"]) for job in plan["jobs"])
    assert len(counts) == 243 and set(counts.values()) == {10}
    pairs = {(job["first"]["personality_id"], job["second"]["personality_id"]) for job in plan["jobs"]}
    assert len(pairs) == 81
    assert ("P1", "P2") in pairs and ("P2", "P1") in pairs
    assert len([pair for pair in pairs if pair[0] == pair[1]]) == 9
    assert [(item[FIELDS[0]], item[FIELDS[1]]) for item in plan["personalities"]] == [
        (risk, inequality) for risk in ("low", "medium", "high")
        for inequality in ("low", "medium", "high")
    ]
    seeds = [value for job in plan["jobs"] for value in (job["seed"], job["assignment_seed"]) if value is not None]
    assert len(seeds) == len(set(seeds)) == 3240
    assert all(job["assignment_seed"] is None for job in plan["jobs"] if job["treatment"] != "25f")


def test_plan_and_seeds_are_reproducible_without_mutating_global_rng():
    import random

    state = random.getstate()
    settings = {"backend": "scripted", "nested": {"keep": [1, True, None]}}
    arguments = dict(treatments=("baseline", "25v"), repetitions=2, master_seed=123,
                     run_settings=settings)
    one = build_plan(**arguments)
    two = build_plan(**arguments)
    assert one == two and random.getstate() == state
    settings["nested"]["keep"].append("not copied")
    assert one["run_settings"]["nested"]["keep"] == [1, True, None]
    changed = build_plan(**{**arguments, "master_seed": 124})
    assert {job["seed"] for job in one["jobs"]}.isdisjoint(job["seed"] for job in changed["jobs"])
    # Treatment selection/order does not perturb the seeds of retained jobs.
    subset = build_plan(treatments=("25v",), repetitions=2, master_seed=123)
    by_id = {job["job_id"]: job for job in one["jobs"]}
    assert all(job == by_id[job["job_id"]] for job in subset["jobs"])


def test_planned_slots_match_actual_environment_and_strict_participant_profiles():
    from custom.envs.risk_sharing_env import RiskSharingEnv
    from protocol.paper_spec import Treatment

    plan = build_plan(repetitions=1, forced_assignment_probabilities=(0.5, 0.25, 0.25))
    sampled = [job for index, job in enumerate(plan["jobs"]) if index % 4 == 0]
    assert len(sampled) > 50
    orders = set()
    for job in sampled:
        treatments = {"baseline": Treatment.BASELINE, "25v": Treatment.VOLUNTARY_INSURANCE_25,
                      "25f": Treatment.FORCED_INSURANCE_25}
        arguments = {"treatment": treatments[job["treatment"]], "seed": job["seed"]}
        if job["treatment"] == "25f":
            arguments.update(forced_assignment_probabilities=plan["forced_assignment_probabilities"],
                             assignment_seed=job["assignment_seed"])
        env = RiskSharingEnv([1, 2], **arguments)
        assert list(env.pair_members[1]) == job["pair_members"]
        orders.add(tuple(env.pair_members[1]))
        for slot, agent_id in zip(("first", "second"), env.pair_members[1]):
            assert job[slot]["agent_id"] == agent_id
            assert job["profiles"][str(agent_id)] == {field: job[slot][field] for field in FIELDS}
            assert set(job["profiles"][str(agent_id)]) == set(FIELDS)
    assert orders == {(1, 2), (2, 1)}


@pytest.mark.parametrize("arguments", [
    {"repetitions": 0}, {"repetitions": -1}, {"repetitions": True}, {"repetitions": 1.5},
    {"master_seed": True}, {"master_seed": "2026"},
    {"treatments": []}, {"treatments": "baseline"}, {"treatments": ["baseline", "baseline"]},
    {"treatments": ["../baseline"]}, {"treatments": ["BASELINE"]},
    {"run_settings": []}, {"run_settings": {1: "not a string key"}},
    {"run_settings": {"timeout": float("inf")}},
])
def test_invalid_design_inputs_are_rejected(arguments):
    with pytest.raises(ValueError):
        build_plan(**{"treatments": ("baseline",), **arguments})


def test_25f_requires_explicit_valid_probabilities_and_other_treatments_reject_them():
    with pytest.raises(ValueError, match="probabilities"):
        build_plan()
    for probabilities in ((0.3, 0.3, 0.3), (True, 0, 0), (0.5, 0.5), (0.5, 0.5, float("nan"))):
        with pytest.raises(ValueError):
            build_plan(forced_assignment_probabilities=probabilities)
    with pytest.raises(ValueError, match="only allowed for 25F"):
        build_plan(treatments=("baseline",), forced_assignment_probabilities=(0.5, 0.25, 0.25))


def test_write_load_roundtrip_strict_profiles_and_no_run_directories(tmp_path):
    plan = build_plan(treatments=("baseline",), repetitions=1)
    root = tmp_path / "batch"
    write_plan(root, plan)
    assert load_plan(root) == plan
    assert {path.name for path in root.iterdir()} == {"plan.json", "cases"}
    assert len(list((root / "cases").iterdir())) == 81
    assert not list(root.rglob("*.tmp"))
    for job in plan["jobs"]:
        case = root / "cases" / job["job_id"]
        assert {path.name for path in case.iterdir()} == {"profiles.json", "job.json"}
        profiles, metadata = build_profiles(2, 2026, case / "profiles.json")
        assert profiles == {int(agent): profile for agent, profile in job["profiles"].items()}
        assert metadata["source"] == "file"
    before = (root / "plan.json").read_bytes()
    with pytest.raises(FileExistsError):
        write_plan(root, plan)
    assert (root / "plan.json").read_bytes() == before


@pytest.mark.parametrize("mutation", ["path", "seed", "agent_bool", "checksum", "unknown"])
def test_tampered_plan_rejected_even_if_job_edit_is_rehashed(tmp_path, mutation):
    plan = build_plan(treatments=("baseline",), repetitions=1)
    root = tmp_path / "batch"
    write_plan(root, plan)
    altered = copy.deepcopy(plan)
    if mutation == "path":
        altered["jobs"][0]["job_id"] = "../../outside"
    elif mutation == "seed":
        altered["jobs"][0]["seed"] += 1
    elif mutation == "agent_bool":
        altered["jobs"][0]["first"]["agent_id"] = True
    elif mutation == "checksum":
        altered["plan_sha256"] = "0" * 64
    else:
        altered["extra"] = "unexpected"
    if mutation != "checksum":
        content = {key: value for key, value in altered.items() if key != "plan_sha256"}
        altered["plan_sha256"] = hashlib.sha256(json.dumps(
            content, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False,
        ).encode()).hexdigest()
    (root / "plan.json").write_text(json.dumps(altered), encoding="utf-8")
    with pytest.raises(ValueError):
        load_plan(root)


@pytest.mark.parametrize("target", ["profiles.json", "job.json"])
def test_modified_case_input_is_rejected(tmp_path, target):
    plan = build_plan(treatments=("baseline",), repetitions=1)
    root = tmp_path / "batch"
    write_plan(root, plan)
    path = root / "cases" / plan["jobs"][0]["job_id"] / target
    path.write_text('{"changed": true}', encoding="utf-8")
    with pytest.raises(ValueError, match="differ.*from plan"):
        load_plan(root)


def test_symlink_case_and_duplicate_json_keys_are_rejected(tmp_path):
    plan = build_plan(treatments=("baseline",), repetitions=1)
    root = tmp_path / "batch"
    write_plan(root, plan)
    case = root / "cases" / plan["jobs"][0]["job_id"]
    saved = case.with_name("saved")
    case.rename(saved)
    case.symlink_to(saved, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        load_plan(root)
    case.unlink()
    saved.rename(case)
    path = root / "plan.json"
    raw = path.read_text(encoding="utf-8")
    path.write_text('{"schema_version": 1,' + raw[1:], encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate JSON key"):
        load_plan(root)


def test_worker_load_validates_full_plan_but_reads_only_selected_case(tmp_path, monkeypatch):
    from experiments.preference_grid import design

    plan = build_plan(treatments=("baseline",), repetitions=1)
    root = tmp_path / "batch"
    write_plan(root, plan)
    chosen, unrelated = plan["jobs"][:2]
    unrelated_path = root / "cases" / unrelated["job_id"] / "profiles.json"
    unrelated_path.write_text('{"unrelated": "invalid"}', encoding="utf-8")
    opened = []
    original_read = design._read_json

    def record_read(path):
        opened.append(path.relative_to(root).as_posix())
        return original_read(path)

    monkeypatch.setattr(design, "_read_json", record_read)
    assert load_plan(root, job_id=chosen["job_id"]) == plan
    assert opened == ["plan.json", f"cases/{chosen['job_id']}/job.json",
                      f"cases/{chosen['job_id']}/profiles.json"]
    with pytest.raises(ValueError, match="differ.*from plan"):
        load_plan(root)

    # The optimization must never waive validation of other jobs in plan.json.
    altered = copy.deepcopy(plan)
    altered["jobs"][1]["seed"] += 1
    (root / "plan.json").write_text(json.dumps(altered), encoding="utf-8")
    with pytest.raises(ValueError, match="canonical design"):
        load_plan(root, job_id=chosen["job_id"])


@pytest.mark.parametrize("job_id", ["baseline__P1__P1__r02", "../../outside", "", 1])
def test_worker_load_rejects_unknown_or_unsafe_job(tmp_path, job_id):
    plan = build_plan(treatments=("baseline",), repetitions=1)
    root = tmp_path / "batch"
    write_plan(root, plan)
    with pytest.raises(ValueError, match="job_id"):
        load_plan(root, job_id=job_id)


def test_worker_load_still_rejects_symlink_cases_directory(tmp_path):
    plan = build_plan(treatments=("baseline",), repetitions=1)
    root = tmp_path / "batch"
    write_plan(root, plan)
    (root / "cases").rename(root / "saved_cases")
    (root / "cases").symlink_to(root / "saved_cases", target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        load_plan(root, job_id=plan["jobs"][0]["job_id"])
