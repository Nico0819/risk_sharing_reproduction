"""Portable visual reports are derived artifacts, never a new experiment."""

import hashlib
import json
import random
import re

import pytest

from experiments.report import REPORT_LINK, add_report_link, main, write_visual_report


def embedded_report_data(path):
    html = path.read_text(encoding="utf-8")
    payload = re.search(r'<script id="report-data" type="application/json">(.*?)</script>', html, re.S)
    assert payload is not None
    return json.loads(payload[1])


def many_group_records(pair_count):
    """Recorded random pairs with different lengths and deliberately missing data."""

    agents = list(range(101, 101 + 2 * pair_count))
    random.Random(2026).shuffle(agents)
    records, decisions, expected = [], [], {}
    for offset in range(pair_count):
        left, right = agents[2 * offset:2 * offset + 2]
        pair_id = 1001 + 7 * offset  # IDs do not imply sequential members or pairs.
        periods = (1, 16, 17, 31, 46)[:2 + offset % 4]
        expected[pair_id] = {"members": sorted([left, right]), "transfer": {},
                             "income_mean": {}, "income_gap": {}, "purchase": {left: offset % 2 == 0}}
        decisions.append({"pair_id": pair_id, "agent_id": left, "partner_id": right,
                          "period": 16, "purchase": offset % 2 == 0})
        if offset % 5:
            decisions.append({"pair_id": pair_id, "agent_id": right, "partner_id": left,
                              "period": 16, "purchase": False})
            expected[pair_id]["purchase"][right] = False
        for period in periods:
            if period == 17 and offset % 3 == 0:
                continue
            amount = offset + period
            missing_transfer = period == 16 and offset % 4 == 1
            missing_income = period == 16 and offset % 4 == 2
            missing_partner = period == 31 and offset % 3 == 1
            records.append({"pair_id": pair_id, "agent_id": left, "partner_id": right,
                            "period": period, "transfer_sent": None if missing_transfer else amount,
                            "final_income": None if missing_income else 325 - amount})
            if not missing_partner:
                records.append({"pair_id": pair_id, "agent_id": right, "partner_id": left,
                                "period": period, "transfer_sent": 0, "final_income": 125 + amount})
            expected[pair_id]["transfer"][period] = None if missing_transfer or missing_partner else amount
            expected[pair_id]["income_mean"][period] = None if missing_income or missing_partner else 225
            expected[pair_id]["income_gap"][period] = None if missing_income or missing_partner else abs(200 - 2 * amount)
    random.Random(7).shuffle(records)
    random.Random(8).shuffle(decisions)
    return records, decisions, expected


def assert_recorded_groups(data, expected):
    assert len(data["pairs"]) == len(expected)
    assert data["agents"] == 2 * len(expected)
    assert {pair["id"] for pair in data["pairs"]} == set(expected)
    for pair in data["pairs"]:
        source = expected[pair["id"]]
        assert [member["id"] for member in pair["members"]] == source["members"]
        for member in pair["members"]:
            assert member["purchase"] is source["purchase"].get(member["id"])
        assert pair["submitted"] == len(source["purchase"])
        for metric in ("transfer", "income_mean", "income_gap"):
            assert pair[metric] == [source[metric].get(period) for period in data["periods"]]
        transfer_values = [value for value in source["transfer"].values() if value is not None]
        income_values = [value for value in source["income_mean"].values() if value is not None]
        income_gaps = [value for value in source["income_gap"].values() if value is not None]
        assert pair["summary_all"]["transfer_mean"] == sum(transfer_values) / len(transfer_values)
        assert pair["summary_all"]["transfer_total"] == sum(transfer_values)
        assert pair["summary_all"]["transfer_valid_periods"] == len(transfer_values)
        assert pair["summary_all"]["income_valid_periods"] == len(income_values)
        assert pair["summary_all"]["income_mean"] == sum(income_values) / len(income_values)
        assert pair["summary_all"]["income_gap_mean"] == sum(income_gaps) / len(income_gaps)


def test_html_embeds_chart_library_and_escapes_data_not_interpreted_as_markup(tmp_path):
    unsafe_id = '</script><img src=x onerror="alert(1)">__PLOTLY_JS__'
    records = [
        {"pair_id": 1, "agent_id": unsafe_id, "partner_id": 2, "period": 1,
         "transfer_sent": 73, "final_income": 252},
        {"pair_id": 1, "agent_id": 2, "partner_id": unsafe_id, "period": 1,
         "transfer_sent": 0, "final_income": 198},
    ]
    path = write_visual_report(tmp_path, {"treatment": "baseline", "backend": "scripted"}, records)
    html = path.read_text()
    raw_data = re.search(r'<script id="report-data" type="application/json">(.*?)</script>', html, re.S)[1]
    assert "</script>" not in raw_data
    data = json.loads(raw_data)
    assert data["pairs"][0]["members"][1]["id"] == unsafe_id
    assert data["pairs"][0]["transfer"] == [73]
    # External library URLs must not be needed when opening the file locally.
    assert not re.search(r'<script[^>]+src=', html)
    assert "plotly.js" in html


@pytest.mark.parametrize("pair_count", [10, 40])
def test_generated_comparison_embeds_only_recorded_pairs_and_valid_observations(tmp_path, pair_count):
    records, decisions, expected = many_group_records(pair_count)
    original_records = json.dumps(records, sort_keys=True)
    original_decisions = json.dumps(decisions, sort_keys=True)
    manifest = {"treatment": "25V", "backend": "scripted", "agents": 4}
    path = write_visual_report(tmp_path, manifest, records, decisions)
    data = embedded_report_data(path)
    assert data["run_id"] == tmp_path.name
    assert data["periods"] == list(range(1, 47))
    assert data["completed_periods"] == 5
    assert_recorded_groups(data, expected)
    assert json.dumps(records, sort_keys=True) == original_records
    assert json.dumps(decisions, sort_keys=True) == original_decisions


def test_historical_visual_refresh_is_idempotent_and_preserves_raw_files(tmp_path):
    records, decisions, expected = many_group_records(10)
    raw = {
        "manifest.json": json.dumps({"treatment": "25V", "backend": "scripted", "agents": 20}),
        "period_records.jsonl": "\n".join(json.dumps(row) for row in records) + "\n",
        "insurance_decisions.jsonl": "\n".join(json.dumps(row) for row in decisions) + "\n",
        "agent_actions.jsonl": '{"action":"submit_decision","submitted_amount":73}\n',
        "participant_deliberations.jsonl": '{"agent_id":101,"reason":"original rationale"}\n',
        "run.log": "original log\n",
    }
    for name, content in raw.items():
        (tmp_path / name).write_text(content)
    original = "# Original result\n\nResearcher's note.\n"
    (tmp_path / "summary.md").write_text(original)
    hashes = {name: hashlib.sha256((tmp_path / name).read_bytes()).hexdigest() for name in raw}
    assert main(["--run-dir", str(tmp_path)]) == 0
    assert (tmp_path / "report.html").exists()
    assert_recorded_groups(embedded_report_data(tmp_path / "report.html"), expected)
    assert (tmp_path / "summary.before-visual-report.md").read_text() == original
    summary = (tmp_path / "summary.md").read_text()
    assert summary.count(REPORT_LINK) == 1
    assert "Researcher's note." in summary
    assert main(["--run-dir", str(tmp_path)]) == 0
    assert_recorded_groups(embedded_report_data(tmp_path / "report.html"), expected)
    assert (tmp_path / "summary.md").read_text() == summary
    assert {name: hashlib.sha256((tmp_path / name).read_bytes()).hexdigest() for name in raw} == hashes


def test_empty_historical_run_does_not_invent_pairs_from_the_manifest_headcount(tmp_path):
    (tmp_path / "manifest.json").write_text(json.dumps({"treatment": "25V", "agents": 20}))
    (tmp_path / "period_records.jsonl").write_text("")
    (tmp_path / "summary.md").write_text("# Not settled yet\n")
    assert main(["--run-dir", str(tmp_path)]) == 0
    data = embedded_report_data(tmp_path / "report.html")
    assert data["agents"] == 20
    assert data["pairs"] == data["periods"] == []
    assert data["completed_periods"] == 0


def test_report_link_does_not_duplicate_or_remove_summary_content():
    original = "# Run\n\n## Data\nExisting records.\n"
    updated = add_report_link(original)
    assert updated.startswith("# Run\n\n" + REPORT_LINK)
    assert original.partition("\n")[2] in updated
    assert add_report_link(updated) == updated
