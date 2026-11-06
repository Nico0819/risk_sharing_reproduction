"""Validate all six public experiment entry points offline, without format options."""

from __future__ import annotations

import json
import os
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def main():
    batch = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    batch_dir = HERE / "runs" / batch
    batch_dir.mkdir(parents=True)
    environment = dict(os.environ)
    for prefix in ("AGENTSOCIETY_LLM", "AGENTSOCIETY_CODER_LLM"):
        for suffix in ("API_KEY", "API_BASE", "MODEL", "THINKING", "EXTRA_BODY"):
            environment.pop(f"{prefix}_{suffix}", None)
    results = []
    for variant in ("original", "fixed_preferences_v1"):
        for treatment in ("baseline", "25v", "25f"):
            name = f"{variant}_{treatment}"
            run_dir, output = batch_dir / name, batch_dir / f"{name}.log"
            module = (
                f"experiments.{treatment}.run_smoke" if variant == "original"
                else "experiments.fixed_preferences_v1.run_smoke"
            )
            command = [
                str(ROOT / ".venv/bin/python"), "-m", module,
                "--backend", "scripted", "--participants", "20", "--agent-concurrency", "20",
                "--full-session", "--seed", "2026", "--request-limit", "5000",
                "--max-output-tokens", "512", "--router-max-steps", "6", "--agent-max-turns", "4",
                "--timeout-seconds", "300", "--run-dir", str(run_dir),
            ]
            if variant == "fixed_preferences_v1":
                command += ["--treatment", treatment, "--preference-seed", "2026"]
            if treatment == "25f":
                command += ["--forced-assignment-probabilities", "0.5", "0.25", "0.25"]
            with output.open("w", encoding="utf-8") as stream:
                process = subprocess.run(command, cwd=ROOT, env=environment, stdout=stream,
                                         stderr=subprocess.STDOUT, timeout=360, check=False)
            manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
            actions = [json.loads(line) for line in (run_dir / "agent_actions.jsonl").read_text().splitlines()]
            transfers = [a for a in actions if a["action"] == "submit_decision" and a["ok"]]
            assert len(transfers) == len(manifest["completed_periods"]) * manifest["pairs"]
            for action in transfers:
                assert action["decision_source"] == "participant_model"
                assert action["submitted_amount"] == action["participant_model_output"]["amount"]
                assert action["submitted_amount"] == action["environment_receipt"]["amount"]
            result = {
                "variant": variant, "treatment": treatment, "command": command,
                "exit_code": process.returncode, "success": manifest["success"],
                "stop_reason": manifest["stop_reason"], "backend": manifest["backend"],
                "observation_presentation": manifest["observation_presentation"],
                "agents": manifest["agents"], "pairs": manifest["pairs"],
                "completed_periods": manifest["completed_periods"], "accepted_transfers": len(transfers),
                "participant_amounts_match_receipts": True,
                "run_dir": str(run_dir.relative_to(ROOT)), "console_output": str(output.relative_to(ROOT)),
            }
            results.append(result)
            (HERE / "scripted_runs.json").write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print(json.dumps({key: value for key, value in result.items() if key != "command"}), flush=True)
            assert process.returncode == 0 and manifest["success"] and manifest["backend"] == "scripted"
            assert manifest["observation_presentation"] == "readable_v1"
            assert manifest["stop_reason"] == "natural_termination"


if __name__ == "__main__":
    main()
