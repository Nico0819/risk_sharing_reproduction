"""One isolated process for one pair-session; never chooses economic actions."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from custom.participant_trace import redact_credentials
from experiments.preference_grid.design import load_plan
from experiments.preference_grid.run_batch import execution_identity, validate_settings, write_json


def run_job(batch_dir: Path, job_id: str, attempt: int) -> dict:
    root = Path(batch_dir).resolve()
    plan = load_plan(root, job_id=job_id)
    job = next((job for job in plan["jobs"] if job["job_id"] == job_id), None)
    if job is None or type(attempt) is not int or attempt < 1:
        raise ValueError("Unknown job or invalid attempt")
    settings = validate_settings(plan["run_settings"])
    if execution_identity(settings) != json.loads((root / "execution.json").read_text()):
        raise ValueError("Execution conditions changed since batch start")
    case_dir = root / "cases" / job_id
    run_dir = case_dir / "attempts" / f"{attempt:04d}" / "run"
    from experiments.fixed_preferences_memory_v1.run_smoke import run_experiment
    from protocol.paper_spec import Treatment

    treatment = {"baseline": Treatment.BASELINE, "25v": Treatment.VOLUNTARY_INSURANCE_25,
                 "25f": Treatment.FORCED_INSURANCE_25}[job["treatment"]]
    try:
        return run_experiment(
            **settings, treatment=treatment, participants=2, agent_concurrency=2,
            seed=job["seed"], assignment_seed=job["assignment_seed"],
            forced_assignment_probabilities=plan["forced_assignment_probabilities"] if job["treatment"] == "25f" else None,
            profiles_file=case_dir / "profiles.json", run_dir=run_dir,
            env_path=Path(os.devnull) if settings["backend"] == "scripted" else None,
        )
    finally:
        if run_dir.is_dir():
            # Research labels stay in files, never in observations or prompts.
            metadata = {"plan_sha256": plan["plan_sha256"], "attempt": attempt, **job}
            write_json(run_dir / "pairing_metadata.json", metadata)
            summary = run_dir / "summary.md"
            if summary.exists():
                with summary.open("a", encoding="utf-8") as handle:
                    handle.write(f"\n配对组合：`{job['combination_id']}`；重复：`{job['repetition']}`；尝试：`{attempt}`。\n\n")
                    for slot in ("first", "second"):
                        member = job[slot]
                        handle.write(f"- {slot}：Agent {member['agent_id']}，{member['personality_id']}，{member['label']}。\n")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-dir", type=Path, required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--attempt", type=int, required=True)
    args = parser.parse_args(argv)
    try:
        result = run_job(args.batch_dir, args.job_id, args.attempt)
        complete = result.get("success") is True and result.get("memory_experiment_complete") is True
        print(json.dumps({"job_id": args.job_id, "success": complete, "stop_reason": result.get("stop_reason")}, ensure_ascii=False))
        return 0 if complete else 1
    except Exception as exc:
        print(json.dumps({"job_id": args.job_id, "success": False, "error_type": type(exc).__name__,
                          "error": redact_credentials(str(exc))}, ensure_ascii=False))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
