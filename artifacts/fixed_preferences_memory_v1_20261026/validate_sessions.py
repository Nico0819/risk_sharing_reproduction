"""Run full offline sessions and verify the private-memory/decision audit chain."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from custom.agents.fixed_preferences_memory_v1.state import parse_memory_output


def rows(path):
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def verify_baseline():
    reference = json.loads((ROOT / "experiments/fixed_preferences_memory_v1/baseline_reference.json").read_text())
    changes = [name for name, digest in reference["sha256"].items()
               if not (ROOT / name).is_file()
               or hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest]
    assert not changes, changes
    return len(reference["sha256"])


def main():
    batch = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    output_dir = HERE / "runs" / batch
    output_dir.mkdir(parents=True)
    environment = dict(os.environ)
    for prefix in ("AGENTSOCIETY_LLM", "AGENTSOCIETY_CODER_LLM"):
        for suffix in ("API_KEY", "API_BASE", "MODEL", "THINKING", "EXTRA_BODY"):
            environment.pop(f"{prefix}_{suffix}", None)
    results = []
    for treatment in ("baseline", "25v", "25f"):
        run_dir = ROOT / "experiments/fixed_preferences_memory_v1/runs" / treatment / batch
        output = output_dir / f"{treatment}.log"
        command = [
            str(ROOT / ".venv/bin/python"), "-m", "experiments.fixed_preferences_memory_v1.run_smoke",
            "--backend", "scripted", "--treatment", treatment,
            "--participants", "20", "--agent-concurrency", "20", "--full-session",
            "--seed", "2026", "--preference-seed", "2026", "--request-limit", "5000",
            "--max-output-tokens", "512", "--router-max-steps", "6", "--agent-max-turns", "4",
            "--timeout-seconds", "300", "--run-dir", str(run_dir),
        ]
        if treatment == "25f":
            command += ["--forced-assignment-probabilities", "0.5", "0.25", "0.25"]
        with output.open("w", encoding="utf-8") as stream:
            process = subprocess.run(command, cwd=ROOT, env=environment, stdout=stream,
                                     stderr=subprocess.STDOUT, timeout=360, check=False)
        manifest = json.loads((run_dir / "manifest.json").read_text())
        assert process.returncode == 0 and manifest["success"], (treatment, str(output))
        assert manifest["backend"] == "scripted" and manifest["stop_reason"] == "natural_termination"
        contexts = {row["request_index"]: row for row in rows(run_dir / "participant_contexts.jsonl")}
        latest, commits = {}, 0
        for event in rows(run_dir / "private_memory_updates.jsonl"):
            agent = event["agent_id"]
            before, after = event["previous_state"], event["new_state"]
            if agent in latest:
                assert latest[agent] == before
            else:
                assert before["revision"] == 0
            assert after["revision"] == before["revision"] + 1
            assert after["owner_agent_id"] == agent
            assert parse_memory_output(event["raw_output"]) == after["content"]
            request = contexts[event["request_index"]]
            assert request["phase"] == "memory_update" and request["agent_id"] == agent
            assert request["private_memory_snapshot"]["revision"] == before["revision"]
            latest[agent] = after
            commits += 1
        assert len(latest) == 20
        assert commits == manifest["private_memory_update_count"] == manifest["private_memory_update_requests"]
        for agent, state in latest.items():
            persisted = json.loads((run_dir / "agents" / f"agent_{agent:04d}" / "private_memory.json").read_text())
            assert persisted == state
        transfers = 0
        for action in rows(run_dir / "agent_actions.jsonl"):
            if action["action"] != "submit_decision" or not action["ok"]:
                continue
            decision = contexts[action["participant_request_index"]]
            assert decision["phase"] == "decision" and decision["agent_id"] == action["agent_id"]
            assert decision["private_memory_sha256"] == action["private_memory_sha256"]
            assert decision["private_memory_snapshot"]["revision"] == action["private_memory_revision"]
            assert action["submitted_amount"] == action["participant_model_output"]["amount"] == action["environment_receipt"]["amount"]
            transfers += 1
        assert transfers == 10 * len(manifest["completed_periods"])
        result = {
            "treatment": treatment, "command": command, "success": True,
            "backend": manifest["backend"], "agents": 20, "pairs": 10,
            "completed_periods": len(manifest["completed_periods"]),
            "memory_updates": commits, "accepted_transfers": transfers,
            "participant_requests": manifest["participant_model_requests"],
            "router_requests": manifest["router_model_requests"],
            "actual_model_requests": manifest["actual_model_requests"],
            "memory_and_decision_audit_matches": True,
            "baseline_files_unchanged": verify_baseline(),
            "run_dir": str(run_dir.relative_to(ROOT)),
            "console_output": str(output.relative_to(ROOT)),
        }
        results.append(result)
        (HERE / "scripted_runs.json").write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps({key: value for key, value in result.items() if key != "command"}), flush=True)


if __name__ == "__main__":
    main()
