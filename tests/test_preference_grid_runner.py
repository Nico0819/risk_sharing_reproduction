"""Batch orchestration preserves independent real framework pair-sessions."""

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.preference_grid.design import build_plan, load_plan, write_plan
from experiments.preference_grid import run_batch as batch


def create_plan(root, *, treatment="baseline", periods=3, repetitions=1):
    settings = batch.validate_settings({"backend": "scripted", "full_session": False,
                                        "completed_period_limit": periods, "max_output_tokens": 512})
    plan = build_plan(repetitions=repetitions, treatments=[treatment], run_settings=settings,
                      forced_assignment_probabilities=[0.5, 0.25, 0.25] if treatment == "25f" else None)
    write_plan(root, plan)
    return plan


@pytest.mark.parametrize("treatment", ["baseline", "25v", "25f"])
def test_real_subprocess_pair_session_has_exact_profiles_labels_and_exports(tmp_path, treatment):
    root = tmp_path / treatment
    repetitions = 2 if treatment == "baseline" else 1
    plan = create_plan(root, treatment=treatment, periods=31, repetitions=repetitions)
    result = batch.run_batch(root, concurrency=2, combinations=["P2__P8"])
    assert result["scheduled"] == result["completed_this_invocation"] == repetitions
    assert result["report"]["complete"] == repetitions
    assert result["failed_this_invocation"] == 0
    job = next(j for j in plan["jobs"] if j["combination_id"] == "P2__P8")
    run = root / "cases" / job["job_id"] / "attempts/0001/run"
    manifest = json.loads((run / "manifest.json").read_text())
    assert manifest["memory_experiment_complete"] is True
    assert manifest["router"] == "ReActRouter"
    assert manifest["completed_periods"] == list(range(1, 32))
    actual = json.loads((run / "participant_profiles.json").read_text())["profiles"]
    assert actual == job["profiles"]
    assert json.loads((run / "pairing_metadata.json").read_text())["first"] == job["first"]
    rows = [json.loads(line) for line in (run / "period_records.jsonl").read_text().splitlines()]
    assert {r["agent_id"] for r in rows} == set(job["pair_members"])
    assert all(r["pair_id"] == 1 and r["partner_id"] != r["agent_id"] for r in rows)
    if treatment == "25f":
        assignments = [json.loads(line) for line in (run / "insurance_assignments.jsonl").read_text().splitlines()]
        assert all(row["pair_members"] == job["pair_members"] for row in assignments)
    contexts = [json.loads(line) for line in (run / "participant_contexts.jsonl").read_text().splitlines()]
    for context in contexts:
        assert context["preferences"] == job["profiles"][str(context["agent_id"])]
        assert "combination_id" not in json.dumps(context)
        assert "personality_id" not in json.dumps(context)
    assert (run / "reasoning_text/all_agents.txt").is_file()
    assert (run / "memory_text/all_agents.txt").is_file()
    assert job["combination_id"] in (run / "summary.md").read_text()
    assert (root / "report.html").is_file()
    csv_text = (root / "period_records.csv").read_text()
    assert "P2__P8" in csv_text and "advantageous_inequality_aversion" in csv_text
    if repetitions == 2:
        other_job = next(j for j in plan["jobs"] if j["combination_id"] == "P2__P8" and j["repetition"] == 2)
        other_run = root / "cases" / other_job["job_id"] / "attempts/0001/run"
        first_state = json.loads((run / "agents/agent_0001/private_memory.json").read_text())
        other_state = json.loads((other_run / "agents/agent_0001/private_memory.json").read_text())
        assert first_state["scope_id"] != other_state["scope_id"]
        assert job["seed"] != other_job["seed"]
        for selected_run in (run, other_run):
            first_contexts = [json.loads(line) for line in (selected_run / "participant_contexts.jsonl").read_text().splitlines()]
            initial = [row for row in first_contexts if row["period"] == 1 and row["phase"] == "decision"]
            assert len(initial) == 2
            assert all(row["private_memory_snapshot"]["revision"] == 0 for row in initial)
            assert all(row["private_memory_snapshot"]["content"]["note"] == "" for row in initial)
    before = (run / "private_memory_updates.jsonl").read_bytes()
    resumed = batch.run_batch(root, combinations=["P2__P8"])
    assert resumed["scheduled"] == 0
    assert (run / "private_memory_updates.jsonl").read_bytes() == before


def test_scheduler_limits_live_subprocesses_and_retains_failed_attempts(tmp_path, monkeypatch):
    root = tmp_path / "batch"
    plan = create_plan(root)
    observed = {"active": 0, "peak": 0, "calls": []}

    async def launch(*cmd, **kwargs):
        observed["active"] += 1
        observed["peak"] = max(observed["peak"], observed["active"])
        observed["calls"].append(cmd)
        process = SimpleNamespace(pid=99999999, returncode=None)

        async def wait():
            await asyncio.sleep(0.01)
            observed["active"] -= 1
            process.returncode = 1
            return 1
        process.wait = wait
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", launch)
    monkeypatch.setattr(batch, "freeze_execution", lambda *args: None)
    result = batch.run_batch(root, concurrency=3, limit_jobs=7)
    assert observed["peak"] == 3 and result["failed_this_invocation"] == 7
    assert len(observed["calls"]) == 7
    first_job = plan["jobs"][0]
    first_attempt = batch.attempts_for(root, first_job)[0]
    before = (first_attempt / "status.json").read_bytes()
    assert batch.run_batch(root, combinations=[first_job["combination_id"]])["scheduled"] == 0
    retried = batch.run_batch(root, combinations=[first_job["combination_id"]], retry_failed=True)
    assert retried["scheduled"] == 1
    assert (first_attempt / "status.json").read_bytes() == before
    assert [p.name for p in batch.attempts_for(root, first_job)] == ["0001", "0002"]
    assert load_plan(root)["jobs"][0] == first_job


def test_plan_cli_does_not_resolve_credentials_or_start_models(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Planning must not initialize any provider")
    monkeypatch.setattr(batch, "execution_identity", forbidden)
    root = tmp_path / "plan"
    assert batch.main(["plan", "--batch-dir", str(root), "--forced-assignment-probabilities", "0.5", "0.25", "0.25"]) == 0
    plan = load_plan(root)
    assert len(plan["jobs"]) == 2430
    assert plan["run_settings"]["full_session"] is True
    assert not (root / "execution.json").exists()
    assert not list(root.glob("cases/*/attempts"))


def test_running_batch_lock_and_changed_configuration_are_rejected(tmp_path, monkeypatch):
    root = tmp_path / "locked"
    create_plan(root)
    with batch.batch_lock(root), pytest.raises(ValueError, match="already running"):
        batch.run_batch(root)
    monkeypatch.setattr(batch, "execution_identity", lambda settings: {"model": "first"})
    batch.freeze_execution(root, {})
    monkeypatch.setattr(batch, "execution_identity", lambda settings: {"model": "different"})
    with pytest.raises(ValueError, match="changed"):
        batch.freeze_execution(root, {})


def test_retry_includes_success_manifest_with_missing_result_files(tmp_path, monkeypatch):
    root = tmp_path / "invalid-result"
    plan = create_plan(root)
    job = plan["jobs"][0]
    attempt = root / "cases" / job["job_id"] / "attempts/0001"
    run = attempt / "run"
    run.mkdir(parents=True)
    batch.write_json(attempt / "status.json", {"status": "completed", "exit_code": 0, "run_dir": "run"})
    batch.write_json(run / "manifest.json", {"success": True, "memory_experiment_complete": True,
                     "closing_memory_complete": True, "stop_reason": "smoke_period_limit",
                     "treatment": "Baseline", "seed": job["seed"]})
    chosen = []

    async def execute(root, jobs, settings, concurrency):
        chosen.extend(jobs)
        return [{"status": "failed"}]

    monkeypatch.setattr(batch, "_execute", execute)
    monkeypatch.setattr(batch, "freeze_execution", lambda *args: None)
    result = batch.run_batch(root, combinations=[job["combination_id"]], retry_failed=True)
    assert chosen == [job] and result["scheduled"] == 1
    assert result["report"]["invalid"] == 1


def test_default_python_plan_can_be_reported_without_run_settings(tmp_path):
    from experiments.preference_grid.report import write_batch_report

    root = tmp_path / "default-settings"
    write_plan(root, build_plan(treatments=["baseline"], repetitions=1))
    assert write_batch_report(root) == {"planned": 81, "pending": 81}


@pytest.mark.parametrize("settings", [{"backend": "unknown"}, {"timeout_seconds": float("nan")},
                                      {"request_limit": True}, {"memory_condition": "fresh"}, {"unknown": 1}])
def test_invalid_execution_settings_rejected(settings):
    with pytest.raises(ValueError):
        batch.validate_settings(settings)
