"""Batch comparisons preserve personality assignments and independent repeats."""

from __future__ import annotations

import copy
import csv
import json
from pathlib import Path

import pytest

from experiments.preference_grid.report import write_batch_report


def personality(number, risk="low", inequality="low"):
    return {"personality_id": f"P{number}", "risk_aversion": risk,
            "advantageous_inequality_aversion": inequality, "label": f"人格 {number}"}


def job(repetition=1, treatment="baseline", first_id=2):
    first = {"agent_id": first_id, **personality(1)}
    second = {"agent_id": 3 - first_id, **personality(2, "high", "medium")}
    return {"job_id": f"{treatment}__P1__P2__r{repetition:02d}", "combination_id": "P1__P2",
            "treatment": treatment, "repetition": repetition, "seed": repetition,
            "assignment_seed": repetition + 100,
            "pair_members": [first_id, 3 - first_id], "first": first, "second": second,
            "profiles": {str(p["agent_id"]): {k: p[k] for k in ("risk_aversion", "advantageous_inequality_aversion")}
                         for p in (first, second)}}


def plan_for(monkeypatch, jobs, *, full=False, completed_period_limit=3):
    from experiments.preference_grid import design

    plan = {"schema_version": 1, "experiment_version": "fixed_preferences_memory_v1",
            "repetitions": 10, "master_seed": 1, "run_settings": {
                "full_session": full, "completed_period_limit": completed_period_limit},
            "treatments": list(dict.fromkeys(j["treatment"] for j in jobs)),
            "personalities": [personality(i) for i in range(1, 10)], "jobs": jobs}
    monkeypatch.setattr(design, "load_plan", lambda path: copy.deepcopy(plan))
    return plan


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def save_rows(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def result(root, item, *, periods=3, amount=100, attempt=1, status="completed", complete=True, full=False):
    attempt_dir = root / "cases" / item["job_id"] / "attempts" / f"{attempt:04d}"
    run = attempt_dir / "run"
    run.mkdir(parents=True)
    save_json(attempt_dir / "status.json", {"status": status, "exit_code": 0 if complete else 1, "run_dir": "run"})
    manifest = {"treatment": item["treatment"].upper() if item["treatment"] != "baseline" else "Baseline",
                "success": complete, "memory_experiment_complete": complete, "closing_memory_complete": complete,
                "natural_termination": full, "completed_periods": list(range(1, periods + 1)),
                "run_mode": "full_session" if full else "smoke",
                "stop_reason": "natural_termination" if full else "smoke_period_limit",
                "actual_model_requests": periods * 3, "backend": "scripted", "seed": item["seed"]}
    save_json(run / "manifest.json", manifest)
    save_json(run / "participant_profiles.json", {"schema_version": 1, "profiles": item["profiles"]})
    rows = []
    first, second = item["pair_members"]
    for period in range(1, periods + 1):
        rows.extend([{"agent_id": first, "partner_id": second, "pair_id": 1, "period": period,
                      "transfer_sent": amount, "final_income": 325 - amount, "had_transfer_opportunity": True},
                     {"agent_id": second, "partner_id": first, "pair_id": 1, "period": period,
                      "transfer_sent": 0, "final_income": 125 + amount, "had_transfer_opportunity": False}])
    save_rows(run / "period_records.jsonl", rows)
    return run


def table(root, name):
    with (root / name).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def test_equal_weight_repeats_label_profiles_and_keep_raw_data_unchanged(tmp_path, monkeypatch):
    jobs = [job(1), job(2)]
    plan_for(monkeypatch, jobs, full=True)
    first = result(tmp_path, jobs[0], periods=48, amount=20, full=True)
    second = result(tmp_path, jobs[1], periods=50, amount=100, full=True)
    original = {p: p.read_bytes() for root in (first, second) for p in root.iterdir()}

    assert write_batch_report(tmp_path) == {"planned": 2, "complete": 2}

    all_stage = next(row for row in table(tmp_path, "combination_summary.csv") if row["stage"] == "all")
    assert float(all_stage["transfer_mean_mean"]) == 60  # not weighted toward the longer session
    assert float(all_stage["transfer_mean_sd"]) == pytest.approx(56.5685424949)
    assert all_stage["transfer_mean_n"] == all_stage["n_completed"] == "2"
    assert all_stage["run_mode"] == "full_session"
    assert "first_agent_id" not in all_stage  # Agent IDs can swap between independent repetitions.
    tail = next(row for row in table(tmp_path, "combination_summary.csv") if row["stage"] == "46+")
    assert float(tail["transfer_mean_mean"]) == 60 and tail["transfer_mean_n"] == "2"
    records = table(tmp_path, "period_records.csv")
    assert len(records) == 196
    assert {row["personality_id"] for row in records if row["agent_id"] == "2"} == {"P1"}
    assert {row["partner_personality_id"] for row in records if row["agent_id"] == "2"} == {"P2"}
    assert {row["first_agent_id"] for row in records} == {"2"}
    assert all(p.read_bytes() == raw for p, raw in original.items())
    html = (tmp_path / "report.html").read_text()
    assert 'id="matrix"' in html and 'id="jobs"' in html
    assert 'id="preferences"' in html
    assert "regression" not in html and "回归" not in (tmp_path / "analysis_summary.md").read_text()
    assert "https://" not in html and "http://" not in html
    assert "推理" in html and "memory_text/all_agents.txt" in html


def test_latest_attempt_failure_does_not_fall_back_to_older_success(tmp_path, monkeypatch):
    jobs = [job(1), job(2), job(3)]
    plan_for(monkeypatch, jobs)
    result(tmp_path, jobs[0], amount=50)
    result(tmp_path, jobs[0], amount=150, attempt=2, status="failed", complete=False)
    result(tmp_path, jobs[1], amount=90)

    counts = write_batch_report(tmp_path)

    assert counts == {"planned": 3, "failed": 1, "complete": 1, "pending": 1}
    sessions = table(tmp_path, "pair_sessions.csv")
    assert [s["status"] for s in sessions] == ["failed", "complete", "pending"]
    assert "attempt_count" not in sessions[0] and "attempt" not in sessions[0]
    assert sessions[0]["observed_transfer_mean"] == ""
    summary = next(row for row in table(tmp_path, "combination_summary.csv") if row["stage"] == "all")
    assert summary["n_planned"] == "3" and summary["n_completed"] == "1"
    assert float(summary["transfer_mean_mean"]) == 90 and summary["transfer_mean_sd"] == ""
    records = table(tmp_path, "period_records.csv")
    assert {r["repetition"] for r in records} == {"2"}
    assert {r["transfer_sent"] for r in records} == {"90", "0"}
    assert "attempt" not in records[0]
    assert "重跑" not in (tmp_path / "report.html").read_text()


@pytest.mark.parametrize("corruption", ["profile", "agent", "partner", "pair", "duplicate", "treatment"])
def test_mismatched_records_are_invalid_and_excluded(tmp_path, monkeypatch, corruption):
    item = job()
    plan_for(monkeypatch, [item])
    run = result(tmp_path, item)
    if corruption == "profile":
        wrong = copy.deepcopy(item["profiles"])
        wrong["2"]["risk_aversion"] = "high"
        save_json(run / "participant_profiles.json", {"schema_version": 1, "profiles": wrong})
    else:
        rows = [json.loads(line) for line in (run / "period_records.jsonl").read_text().splitlines()]
        if corruption == "duplicate":
            rows.append(rows[0])
        else:
            key = {"agent": "agent_id", "partner": "partner_id", "pair": "pair_id", "treatment": "treatment"}[corruption]
            rows[0][key] = "25V" if corruption == "treatment" else 9
        save_rows(run / "period_records.jsonl", rows)

    assert write_batch_report(tmp_path) == {"planned": 1, "invalid": 1}
    assert table(tmp_path, "period_records.csv") == []
    assert all(row["transfer_mean_mean"] == "" for row in table(tmp_path, "combination_summary.csv"))


def test_missing_settlement_and_closing_memory_are_incomplete(tmp_path, monkeypatch):
    jobs = [job(1), job(2)]
    plan_for(monkeypatch, jobs)
    first = result(tmp_path, jobs[0])
    second = result(tmp_path, jobs[1])
    rows = (first / "period_records.jsonl").read_text().splitlines()
    (first / "period_records.jsonl").write_text("\n".join(rows[:-1]) + "\n")
    manifest = json.loads((second / "manifest.json").read_text())
    manifest["closing_memory_complete"] = False
    save_json(second / "manifest.json", manifest)

    assert write_batch_report(tmp_path) == {"planned": 2, "incomplete": 2}
    assert all(row["transfer_mean_n"] == "0" for row in table(tmp_path, "combination_summary.csv"))


def test_full_sessions_require_natural_termination_and_keep_tail_separate(tmp_path, monkeypatch):
    jobs = [job(1), job(2)]
    plan_for(monkeypatch, jobs, full=True)
    run = result(tmp_path, jobs[0], periods=48, amount=25, full=True)
    result(tmp_path, jobs[1], periods=3, amount=100, full=False)
    rows = [json.loads(line) for line in (run / "period_records.jsonl").read_text().splitlines()]
    for row in rows:
        if row["period"] >= 46 and row["had_transfer_opportunity"]:
            row["transfer_sent"] = 200
    save_rows(run / "period_records.jsonl", rows)

    assert write_batch_report(tmp_path) == {"planned": 2, "complete": 1, "incomplete": 1}
    summaries = {row["stage"]: row for row in table(tmp_path, "combination_summary.csv")}
    assert float(summaries["31-45"]["transfer_mean_mean"]) == 25
    assert float(summaries["46+"]["transfer_mean_mean"]) == 200
    termination = [r for r in table(tmp_path, "termination_distribution.csv") if r["treatment"] == "baseline"]
    assert [r["ended"] for r in termination] == ["1", "0", "0"]
    assert [r["conditional_rate"] for r in termination] == ["1.0", "", ""]
    note = "合计 1 场完整实验。第 48 轮有 1 场进入，其中 1 场结束（100.00%）"
    for name in ("analysis_summary.md", "report.html"):
        assert note in (tmp_path / name).read_text()


def test_running_or_missing_manifest_remains_visible_without_invented_result(tmp_path, monkeypatch):
    items = [job(1), job(2)]
    plan_for(monkeypatch, items)
    for index, item in enumerate(items):
        run = result(tmp_path, item, status="running")
        (run / "manifest.json").unlink()
        if index:
            (run.parent / "status.json").unlink()
    assert write_batch_report(tmp_path) == {"planned": 2, "running": 1, "incomplete": 1}
    assert all(row["transfer_mean_n"] == "0" for row in table(tmp_path, "combination_summary.csv"))


def test_voluntary_purchase_and_forced_assignment_remain_distinct(tmp_path, monkeypatch):
    jobs = [job(1, "25v"), job(1, "25f")]
    plan_for(monkeypatch, jobs, completed_period_limit=16)
    for item in jobs:
        run = result(tmp_path, item, periods=16)
        first, second = item["pair_members"]
        ledger = [{"pair_id": 1, "agent_id": a, "partner_id": b, "period": 16,
                   "purchase": a == first} for a, b in ((first, second), (second, first))]
        if item["treatment"] == "25v":
            save_rows(run / "insurance_decisions.jsonl", ledger)
        else:
            for row in ledger:
                row["insured"] = row.pop("purchase")
                row.update(treatment="25F", assignment_source="environment_random", pair_members=item["pair_members"])
            save_rows(run / "insurance_assignments.jsonl", ledger)
    assert write_batch_report(tmp_path) == {"planned": 2, "complete": 2}
    voluntary, forced = table(tmp_path, "pair_sessions.csv")
    assert voluntary["first_purchase"] == "True" and voluntary["first_assigned_insurance"] == ""
    assert forced["first_purchase"] == "" and forced["first_assigned_insurance"] == "True"


def test_html_escapes_untrusted_labels(tmp_path, monkeypatch):
    item = job()
    plan = plan_for(monkeypatch, [item])
    plan["personalities"][0]["label"] = "</script><script>alert('bad')</script>"
    write_batch_report(tmp_path)
    html = (tmp_path / "report.html").read_text()
    assert "</script><script>alert('bad')</script>" not in html
    assert "\\u003c/script\\u003e" in html


@pytest.mark.parametrize("seed", [999, None, True])
def test_manifest_seed_must_match_this_repetition(tmp_path, monkeypatch, seed):
    item = job()
    plan_for(monkeypatch, [item])
    run = result(tmp_path, item)
    manifest = json.loads((run / "manifest.json").read_text())
    manifest["seed"] = seed
    save_json(run / "manifest.json", manifest)

    assert write_batch_report(tmp_path) == {"planned": 1, "invalid": 1}
    assert "seed" in table(tmp_path, "pair_sessions.csv")[0]["error"]


def test_forced_insurance_order_must_match_design_slots(tmp_path, monkeypatch):
    item = job(1, "25f")
    plan_for(monkeypatch, [item], completed_period_limit=16)
    run = result(tmp_path, item, periods=16)
    first, second = item["pair_members"]
    ledger = [{"pair_id": 1, "agent_id": a, "partner_id": b, "period": 16,
               "insured": a == first, "treatment": "25F", "assignment_source": "environment_random",
               "pair_members": list(reversed(item["pair_members"]))}
              for a, b in ((first, second), (second, first))]
    save_rows(run / "insurance_assignments.jsonl", ledger)

    assert write_batch_report(tmp_path) == {"planned": 1, "invalid": 1}
    assert "slots" in table(tmp_path, "pair_sessions.csv")[0]["error"]


def test_smoke_length_must_match_plan_and_absent_stage_is_not_zero(tmp_path, monkeypatch):
    items = [job(1), job(2)]
    plan_for(monkeypatch, items, completed_period_limit=3)
    result(tmp_path, items[0], periods=1)
    result(tmp_path, items[1], periods=3)

    assert write_batch_report(tmp_path) == {"planned": 2, "incomplete": 1, "complete": 1}
    tail = next(row for row in table(tmp_path, "combination_summary.csv") if row["stage"] == "46+")
    assert tail["transfer_mean_mean"] == "" and tail["transfer_mean_n"] == "0"


@pytest.mark.parametrize("field", ["job_id", "repetition", "seed", "first", "second"])
def test_pairing_metadata_must_match_job_when_present(tmp_path, monkeypatch, field):
    item = job()
    plan_for(monkeypatch, [item])
    run = result(tmp_path, item)
    metadata = copy.deepcopy(item)
    metadata[field] = "wrong"
    save_json(run / "pairing_metadata.json", metadata)

    assert write_batch_report(tmp_path) == {"planned": 1, "invalid": 1}
    assert "metadata" in table(tmp_path, "pair_sessions.csv")[0]["error"]


def test_inspection_diagnostics_redact_credentials(tmp_path, monkeypatch):
    from experiments.preference_grid import report

    item = job()
    plan_for(monkeypatch, [item])
    result(tmp_path, item)

    def broken_json(path):
        raise ValueError("api_key=fake-report-secret; Authorization: Bearer fake-header-secret")

    monkeypatch.setattr(report, "_json", broken_json)
    assert write_batch_report(tmp_path) == {"planned": 1, "invalid": 1}
    for filename in ("pair_sessions.csv", "report.html"):
        text = (tmp_path / filename).read_text(encoding="utf-8-sig")
        assert "fake-report-secret" not in text and "fake-header-secret" not in text
