"""Offline integration checks for the bounded Baseline entry point."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from custom.config import RuntimeConfigError, load_runtime_config
from custom.envs.risk_sharing_env import RiskSharingEnv
from experiments.baseline.run_smoke import run_experiment


_FRAMEWORK_VARS = (
    "AGENTSOCIETY_LLM_API_KEY",
    "AGENTSOCIETY_LLM_API_BASE",
    "AGENTSOCIETY_LLM_MODEL",
    "AGENTSOCIETY_CODER_LLM_API_KEY",
    "AGENTSOCIETY_CODER_LLM_API_BASE",
    "AGENTSOCIETY_CODER_LLM_MODEL",
    "AGENTSOCIETY_LLM_THINKING",
    "AGENTSOCIETY_CODER_LLM_THINKING",
    "AGENTSOCIETY_LLM_EXTRA_BODY",
    "AGENTSOCIETY_CODER_LLM_EXTRA_BODY",
    "WORKSPACE_PATH",
)


def _clear_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in _FRAMEWORK_VARS + ("DEEPSEEK_API_KEY", "DEEPSEEK_BASE_URL", "DEEPSEEK_MODEL"):
        monkeypatch.delenv(name, raising=False)


def _contains_key(value: object, forbidden: set[str]) -> bool:
    if isinstance(value, dict):
        return any(key in forbidden or _contains_key(item, forbidden) for key, item in value.items())
    if isinstance(value, list):
        return any(_contains_key(item, forbidden) for item in value)
    return False


def test_secret_safe_config_contract_and_missing_real_credentials(tmp_path, monkeypatch):
    _clear_runtime(monkeypatch)
    example = Path(__file__).parents[1] / ".env.example"
    text = example.read_text(encoding="utf-8")
    assert "DEEPSEEK_API_KEY=" in text
    assert "DEEPSEEK_BASE_URL=https://api.deepseek.com" in text
    assert "DEEPSEEK_MODEL=deepseek-flash" in text

    ignored = (Path(__file__).parents[1] / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert ".env" in ignored
    assert ".env.*" in ignored
    assert "!.env.example" in ignored

    empty_env = tmp_path / ".env"
    empty_env.write_text("DEEPSEEK_API_KEY=\nDEEPSEEK_MODEL=\n", encoding="utf-8")
    with pytest.raises(RuntimeConfigError, match="missing or placeholder"):
        load_runtime_config("deepseek", env_path=empty_env)


def test_identity_tool_schema_declares_integer_agent_id():
    env = RiskSharingEnv([1, 2], seed=1)
    schemas = {tool["function"]["name"]: tool for tool in env._llm_tools}
    assert schemas["get_observation"]["function"]["parameters"]["properties"]["agent_id"]["type"] == "integer"
    assert schemas["submit_decision"]["function"]["parameters"]["properties"]["agent_id"]["type"] == "integer"


def test_scripted_runner_uses_real_agent_router_adapter_and_is_bounded(tmp_path, monkeypatch):
    _clear_runtime(monkeypatch)
    manifest = run_experiment(
        backend="scripted",
        participants=4,
        completed_period_limit=3,
        seed=2026,
        run_dir=tmp_path / "run",
        request_limit=80,
        max_output_tokens=2048,
        timeout_seconds=120,
    )

    assert manifest["success"] is True
    assert manifest["router"] == "ReActRouter"
    assert manifest["agent_class"] == "RiskSharingParticipant"
    assert manifest["agents"] == 4
    assert manifest["pairs"] == 2
    assert manifest["completed_periods"] == [1, 2, 3]
    assert manifest["actual_model_requests"] <= 80
    assert manifest["max_output_tokens"] == 2048
    assert manifest["run_dir"] == str(tmp_path / "run")

    run_dir = tmp_path / "run"
    actions = [json.loads(line) for line in (run_dir / "agent_actions.jsonl").read_text(encoding="utf-8").splitlines()]
    records = [json.loads(line) for line in (run_dir / "period_records.jsonl").read_text(encoding="utf-8").splitlines()]
    assert {row["agent_id"] for row in actions} == {1, 2, 3, 4}
    assert {row["pair_id"] for row in actions} == {1, 2}
    assert {row["period"] for row in records} == {1, 2, 3}
    assert len(records) == 12
    assert all(row["treatment"] == "baseline" for row in records)

    participant_forbidden = {"treatment", "analysis_phase", "session_id"}
    participant_rows = [
        row.get("observation")
        for row in actions
        if isinstance(row.get("observation"), dict)
    ]
    assert participant_rows
    assert all(not _contains_key(row, participant_forbidden) for row in participant_rows)
    assert all(not _contains_key(row, participant_forbidden) for row in actions if row.get("receipt"))

    manifest_text = (run_dir / "manifest.json").read_text(encoding="utf-8")
    assert "DEEPSEEK_API_KEY" not in manifest_text

    log_path = run_dir / "run.log"
    log_rows = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    assert log_rows[0]["event"] == "run_started"
    assert log_rows[-1]["event"] == "run_finished"
    assert log_rows[-1]["success"] is True
    assert "DEEPSEEK_API_KEY" not in log_path.read_text(encoding="utf-8")
