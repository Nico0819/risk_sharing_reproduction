"""Run three full, offline 20-person sessions using readable observations."""

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
    environment = dict(os.environ)
    for prefix in ("AGENTSOCIETY_LLM", "AGENTSOCIETY_CODER_LLM"):
        for suffix in ("API_KEY", "API_BASE", "MODEL", "THINKING", "EXTRA_BODY"):
            environment.pop(f"{prefix}_{suffix}", None)
    results = []
    for treatment in ("baseline", "25v", "25f"):
        run_dir = HERE / "runs" / batch / treatment
        output = HERE / f"{treatment}_scripted.log"
        command = [
            str(ROOT / ".venv/bin/python"), "-m", "experiments.fixed_preferences_v1.run_smoke",
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
        manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
        result = {
            "treatment": treatment, "command": command, "exit_code": process.returncode,
            "success": manifest["success"], "stop_reason": manifest["stop_reason"],
            "backend": manifest["backend"], "observation_presentation": manifest["observation_presentation"],
            "agents": manifest["agents"], "pairs": manifest["pairs"],
            "completed_periods": manifest["completed_periods"],
            "run_dir": str(run_dir), "console_output": str(output),
        }
        results.append(result)
        (HERE / "scripted_runs.json").write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({key: value for key, value in result.items() if key != "command"}), flush=True)
        assert process.returncode == 0 and result["success"] and result["backend"] == "scripted"
        assert result["stop_reason"] == "natural_termination"


if __name__ == "__main__":
    main()
