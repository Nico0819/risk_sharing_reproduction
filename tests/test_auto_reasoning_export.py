"""Readable reasoning exports survive normal and exceptional runner shutdown."""

from __future__ import annotations

import asyncio
import json
import socket
from importlib import import_module
from pathlib import Path

import pytest

from experiments import export_reasoning as exporter


RUNNERS = (
    "experiments.baseline.run_smoke",
    "experiments.fixed_preferences_v1.run_smoke",
    "experiments.fixed_preferences_memory_v1.run_smoke",
)


@pytest.fixture(autouse=True)
def isolated_runtime(monkeypatch):
    import custom.config

    monkeypatch.setattr(custom.config, "_parse_dotenv", lambda path: {})
    monkeypatch.setenv("PYTHON_DOTENV_DISABLED", "1")
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    for prefix in ("AGENTSOCIETY_LLM", "AGENTSOCIETY_CODER_LLM"):
        for suffix in ("API_KEY", "API_BASE", "MODEL", "THINKING", "EXTRA_BODY", "REASONING_EFFORT"):
            monkeypatch.delenv(f"{prefix}_{suffix}", raising=False)
    for name in ("WORKSPACE_PATH", "LLM_API_KEY", "LLM_BASE_URL", "LLM_MODEL"):
        monkeypatch.delenv(name, raising=False)

    def unexpected_network(*args, **kwargs):
        pytest.fail("Automatic export tests must not contact a provider")

    monkeypatch.setattr(socket.socket, "connect", unexpected_network)


def write_trace(root: Path) -> Path:
    source = root / "participant_reasoning.jsonl"
    row = {
        "agent_id": 1, "period": 1, "request_index": 1,
        "phase": "decision", "task": "transfer", "response_source": "scripted",
        "reasoning_status": "available", "reasoning_content": "LOCAL_FIXTURE_REASONING",
    }
    source.write_text(json.dumps(row) + "\n", encoding="utf-8")
    return source


def events(root: Path) -> list[dict]:
    return [json.loads(line) for line in (root / "run.log").read_text(encoding="utf-8").splitlines()
            if line.strip()]


def assert_export(root: Path) -> None:
    assert "LOCAL_FIXTURE_REASONING" in (root / "reasoning_text/agent_0001.txt").read_text(encoding="utf-8")
    assert "LOCAL_FIXTURE_REASONING" in (root / "reasoning_text/all_agents.txt").read_text(encoding="utf-8")
    finished = [row for row in events(root) if row["event"] == "reasoning_export_finished"]
    assert len(finished) == 1
    assert set(finished[0]["output_files"]) == {
        "reasoning_text/all_agents.txt", "reasoning_text/agent_0001.txt",
    }
    assert "reasoning_text/all_agents.txt" in (root / "summary.md").read_text(encoding="utf-8")


def test_finalizer_exports_and_links_saved_trace_without_changing_source(tmp_path):
    source = write_trace(tmp_path)
    before = source.read_bytes()
    summary = "# Existing result\n\n- success: True\n"
    (tmp_path / "summary.md").write_text(summary, encoding="utf-8")

    exporter.finalize_reasoning_export(tmp_path)

    assert_export(tmp_path)
    assert source.read_bytes() == before
    assert (tmp_path / "summary.md").read_text(encoding="utf-8").startswith(summary)


@pytest.mark.parametrize("contents", [None, "", " \n\n"])
def test_no_trace_records_are_skipped_without_inventing_text(tmp_path, contents):
    if contents is not None:
        (tmp_path / "participant_reasoning.jsonl").write_text(contents, encoding="utf-8")
    (tmp_path / "summary.md").write_text("# No requests\n", encoding="utf-8")

    exporter.finalize_reasoning_export(tmp_path)

    assert not (tmp_path / "reasoning_text").exists()
    assert [row["event"] for row in events(tmp_path)] == ["reasoning_export_skipped"]
    if contents is not None:
        assert (tmp_path / "participant_reasoning.jsonl").read_text(encoding="utf-8") == contents


def test_malformed_trace_reports_failure_without_overwriting_source(tmp_path):
    source = write_trace(tmp_path)
    with source.open("a", encoding="utf-8") as handle:
        handle.write('{"incomplete":\n')
    before = source.read_bytes()
    (tmp_path / "summary.md").write_text("# Original failed experiment\n", encoding="utf-8")

    exporter.finalize_reasoning_export(tmp_path)

    assert source.read_bytes() == before
    assert not (tmp_path / "reasoning_text").exists()
    failed = [row for row in events(tmp_path) if row["event"] == "reasoning_export_failed"]
    assert len(failed) == 1 and failed[0]["error"]
    summary = (tmp_path / "summary.md").read_text(encoding="utf-8")
    assert summary.startswith("# Original failed experiment\n") and "失败" in summary


def test_export_error_is_redacted_and_does_not_escape(tmp_path, monkeypatch):
    source = write_trace(tmp_path)
    before = source.read_bytes()
    (tmp_path / "summary.md").write_text("# Result\n", encoding="utf-8")

    def failing_export(*args, **kwargs):
        raise RuntimeError("api_key=fake-export-credential; Authorization: Bearer fake-header-credential")

    monkeypatch.setattr(exporter, "export_reasoning", failing_export)
    exporter.finalize_reasoning_export(tmp_path)

    assert source.read_bytes() == before
    failed = [row for row in events(tmp_path) if row["event"] == "reasoning_export_failed"]
    assert len(failed) == 1 and failed[0]["error"]
    diagnostics = (tmp_path / "summary.md").read_text(encoding="utf-8") + (tmp_path / "run.log").read_text(encoding="utf-8")
    assert "fake-export-credential" not in diagnostics
    assert "fake-header-credential" not in diagnostics


@pytest.mark.parametrize("export_fails", [False, True])
def test_diagnostic_write_errors_do_not_escape(tmp_path, monkeypatch, export_fails):
    source = write_trace(tmp_path)
    before = source.read_bytes()
    (tmp_path / "summary.md").write_text("# Result\n", encoding="utf-8")
    original_open = Path.open

    def blocked_diagnostics(path, *args, **kwargs):
        if path.name in {"run.log", "summary.md"}:
            raise OSError("diagnostic destination unavailable")
        return original_open(path, *args, **kwargs)

    def failing_export(*args, **kwargs):
        raise OSError("export destination unavailable")

    monkeypatch.setattr(Path, "open", blocked_diagnostics)
    if export_fails:
        monkeypatch.setattr(exporter, "export_reasoning", failing_export)

    exporter.finalize_reasoning_export(tmp_path)

    assert source.read_bytes() == before
    if not export_fails:
        assert (tmp_path / "reasoning_text/agent_0001.txt").is_file()


@pytest.mark.parametrize("module_name", RUNNERS)
def test_timeout_runs_finalizer_after_cancellation_and_preserves_error(tmp_path, monkeypatch, module_name):
    module = import_module(module_name)
    cancelled = []

    async def slow_run(**kwargs):
        write_trace(kwargs["run_dir"])
        try:
            await asyncio.sleep(60)
        finally:
            cancelled.append(True)

    monkeypatch.setattr(module, "_run_async", slow_run)
    root = tmp_path / "timeout"
    with pytest.raises(TimeoutError):
        module.run_experiment(
            backend="scripted", participants=2, completed_period_limit=1,
            run_dir=root, timeout_seconds=0.03,
        )

    assert cancelled == [True]
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["success"] is False
    assert manifest["stop_reason"] == "configuration_or_runtime_error"
    assert manifest["error_type"] == "TimeoutError"
    assert_export(root)


@pytest.mark.parametrize("module_name", RUNNERS)
def test_agent_error_return_still_exports_saved_trace(tmp_path, monkeypatch, module_name):
    module = import_module(module_name)
    failure = {"success": False, "stop_reason": "agent_error", "completed_periods": [1]}

    async def failed_agent_run(**kwargs):
        root = kwargs["run_dir"]
        write_trace(root)
        (root / "manifest.json").write_text(json.dumps(failure), encoding="utf-8")
        (root / "summary.md").write_text("# Agent error\n", encoding="utf-8")
        return failure

    monkeypatch.setattr(module, "_run_async", failed_agent_run)
    root = tmp_path / "agent-error"
    result = module.run_experiment(
        backend="scripted", participants=2, completed_period_limit=2, run_dir=root,
    )

    assert result is failure
    assert json.loads((root / "manifest.json").read_text(encoding="utf-8")) == failure
    assert_export(root)
