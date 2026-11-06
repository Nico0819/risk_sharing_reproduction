"""25F HTML reports consume computer assignments and preserve experiment records."""

import hashlib
from importlib import import_module
import json
import random
import re

from experiments.report import REPORT_LINK, main, write_visual_report


def embedded_data(path):
    payload = re.search(
        r'<script id="report-data" type="application/json">(.*?)</script>',
        path.read_text(encoding="utf-8"), re.S,
    )
    assert payload is not None
    return json.loads(payload[1])


def forced_records(pair_count=10):
    agents = list(range(101, 101 + 2 * pair_count))
    random.Random(2026).shuffle(agents)
    records, assignments, expected = [], [], {}
    for offset in range(pair_count):
        left, right = agents[2 * offset:2 * offset + 2]
        pair = 1000 + offset * 7
        coverage = {left: True, right: offset % 2 == 0}
        expected[pair] = coverage
        for agent, partner in ((left, right), (right, left)):
            assignments.append({
                "agent_id": agent, "partner_id": partner, "pair_id": pair,
                "period": 16, "insured": coverage[agent],
                "assignment_source": "environment_random", "treatment": "25F",
            })
        for period in (1, 16, 31):
            amount = offset + period
            for agent, partner in ((left, right), (right, left)):
                insured = period == 16 and coverage[agent]
                random_income = 200 if agent == left else 0
                cost = 25 if insured else 0
                payout = 50 if insured and agent == right else 0
                records.append({
                    "agent_id": agent, "partner_id": partner, "pair_id": pair,
                    "period": period, "treatment": "25F", "random_income": random_income,
                    "insurance_assigned": coverage[agent] if period >= 16 else False,
                    "insured": insured, "insurance_cost": cost, "insurance_payout": payout,
                    "transfer_sent": amount if agent == left else 0,
                    "final_income": 125 + random_income - cost + payout
                    + (-amount if agent == left else amount),
                })
    random.Random(7).shuffle(records)
    random.Random(8).shuffle(assignments)
    return records, assignments, expected


def assert_forced_groups(data, expected):
    assert data["treatment"] == "25F"
    assert data["insurance"] is True
    assert data["insurance_mode"] == "forced"
    assert data["agents"] == 2 * len(expected)
    assert len(data["pairs"]) == len(expected)
    for pair in data["pairs"]:
        coverage = expected[pair["id"]]
        assert {member["id"]: member["assigned"] for member in pair["members"]} == coverage
        assert all(member["purchase"] is None for member in pair["members"])
        assert pair["submitted"] == pair["purchased"] == 0
        assert pair["assigned"] == 2
        assert pair["covered"] == sum(coverage.values())
        assert pair["status"] == ("双方获保" if pair["covered"] == 2 else "单方获保")
        assert pair["premium"] == 25 * pair["covered"]
        assert pair["period_count"] == 3
        assert pair["transfer"][0] is not None
        assert pair["transfer"][15] is not None
        assert pair["transfer"][30] is not None
        assert pair["transfer"][1:15] == [None] * 14


def test_forced_visual_report_handles_many_pairs_and_retains_original_assignments(tmp_path):
    records, assignments, expected = forced_records(40)
    originals = json.dumps([records, assignments], sort_keys=True)
    # A stray voluntary ledger must not create a fictitious extra pair.
    decisions = [{"agent_id": 901, "partner_id": 902, "pair_id": 9999,
                  "period": 16, "purchase": True}]
    path = write_visual_report(
        tmp_path, {"treatment": "25F", "agents": 4, "backend": "scripted"},
        records, decisions, assignments=assignments,
    )
    data = embedded_data(path)
    assert_forced_groups(data, expected)
    assert data["periods"] == list(range(1, 32))
    assert data["completed_periods"] == 3
    assert data["insurance_start"] == 16 and data["removal"] == 31
    assert not re.search(r'<script[^>]+src=', path.read_text())
    assert json.dumps([records, assignments], sort_keys=True) == originals


def test_historical_forced_report_refresh_reads_assignment_ledger_without_replaying(tmp_path):
    records, assignments, expected = forced_records(10)
    configuration = {
        "probabilities": {"both_insured": 0.5, "first_only": 0.25, "second_only": 0.25},
        "seed": 2026, "probability_source": "explicit_run_configuration",
        "paper_probability_resolved": False,
    }
    raw = {
        "manifest.json": json.dumps({"treatment": "25F", "backend": "llm", "agents": 20,
                                     "insurance_assignment_configuration": configuration}),
        "period_records.jsonl": "\n".join(json.dumps(row) for row in records) + "\n",
        "insurance_assignments.jsonl": "\n".join(json.dumps(row) for row in assignments) + "\n",
        "insurance_decisions.jsonl": json.dumps({"pair_id": 9999, "agent_id": 901,
                                                "partner_id": 902, "purchase": True}) + "\n",
        "agent_actions.jsonl": '{"action":"submit_decision","submitted_amount":73}\n',
        "run.log": "original run diagnostics\n",
        "payment.json": '{"selected_period":16,"records":[]}\n',
    }
    for name, text in raw.items():
        (tmp_path / name).write_text(text, encoding="utf-8")
    original = "# 25F full session\n\nOriginal assignment table and researcher notes.\n"
    (tmp_path / "summary.md").write_text(original)
    hashes = {name: hashlib.sha256((tmp_path / name).read_bytes()).hexdigest() for name in raw}

    assert main(["--run-dir", str(tmp_path)]) == 0
    report = embedded_data(tmp_path / "report.html")
    assert_forced_groups(report, expected)
    assert report["insurance_assignment_configuration"]["probabilities"] == configuration["probabilities"]
    assert report["insurance_assignment_configuration"]["paper_probability_resolved"] is False
    assert (tmp_path / "summary.before-visual-report.md").read_text() == original
    updated = (tmp_path / "summary.md").read_text()
    assert updated.count(REPORT_LINK) == 1
    assert "Original assignment table and researcher notes." in updated

    assert main(["--run-dir", str(tmp_path)]) == 0
    assert (tmp_path / "summary.md").read_text() == updated
    assert (tmp_path / "summary.before-visual-report.md").read_text() == original
    assert {name: hashlib.sha256((tmp_path / name).read_bytes()).hexdigest() for name in raw} == hashes


def test_forced_history_without_assignment_file_keeps_unknown_coverage(tmp_path):
    records, _, _ = forced_records(1)
    (tmp_path / "manifest.json").write_text(json.dumps({"treatment": "25F", "agents": 2}))
    (tmp_path / "period_records.jsonl").write_text("\n".join(json.dumps(row) for row in records))
    (tmp_path / "summary.md").write_text("# Incomplete archived files\n")
    assert main(["--run-dir", str(tmp_path)]) == 0
    pair = embedded_data(tmp_path / "report.html")["pairs"][0]
    assert pair["assigned"] == pair["covered"] == 0
    assert pair["status"] == "分配记录未齐"
    assert all(member["assigned"] is None and member["purchase"] is None for member in pair["members"])


def test_new_forced_experiment_automatically_writes_html_through_withdrawal(tmp_path, monkeypatch):
    for prefix in ("AGENTSOCIETY_LLM", "AGENTSOCIETY_CODER_LLM"):
        for suffix in ("API_KEY", "API_BASE", "MODEL", "THINKING", "EXTRA_BODY"):
            monkeypatch.delenv(f"{prefix}_{suffix}", raising=False)
    for name in ("WORKSPACE_PATH", "LLM_API_KEY", "LLM_BASE_URL", "LLM_MODEL"):
        monkeypatch.delenv(name, raising=False)
    run = import_module("experiments.25f.run_smoke").run_experiment
    root = tmp_path / "fresh-25f"
    manifest = run(
        backend="scripted", participants=4, completed_period_limit=31,
        run_dir=root, seed=2026, request_limit=250, max_output_tokens=512,
        forced_assignment_probabilities=(0.5, 0.25, 0.25), assignment_seed=2026,
        agent_concurrency=4, timeout_seconds=120,
    )
    assert manifest["success"] is True
    assert manifest["completed_periods"] == list(range(1, 32))
    assert manifest["insurance_decision_count"] == 0
    assert manifest["actual_model_requests"] == 31 * 6
    assert not (root / "insurance_decisions.jsonl").exists()
    assert REPORT_LINK in (root / "summary.md").read_text()
    report = embedded_data(root / "report.html")
    assert report["insurance_mode"] == "forced" and report["insurance"] is True
    assignments = [json.loads(line) for line in (root / "insurance_assignments.jsonl").read_text().splitlines()]
    original_coverage = {row["agent_id"]: row["insured"] for row in assignments}
    assert len(assignments) == 4
    assert {member["id"]: member["assigned"] for pair in report["pairs"] for member in pair["members"]} == original_coverage
    records = [json.loads(line) for line in (root / "period_records.jsonl").read_text().splitlines()]
    assert all(row["insured"] is False for row in records if row["period"] == 31)
    assert sum(pair["premium"] for pair in report["pairs"]) == 15 * 25 * sum(original_coverage.values())
    assert sum(pair["payout"] for pair in report["pairs"]) == sum(row["insurance_payout"] for row in records)
