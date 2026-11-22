"""Offline, pair-session summaries for the planned preference grid.

Research labels are joined to saved results here; they never enter participant
observations. Each completed pair-session has one vote in a cell's mean.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Any

from custom.participant_trace import redact_credentials
from experiments.preference_grid.insurance_summary import build_insurance_summary, render_insurance_summary
from experiments.preference_grid import record_coverage
from experiments.preference_grid.termination_summary import build_termination_distribution, overall_ending_note
from experiments.report_data import build_report_data, summarize_pair
from protocol import paper_spec as spec


STAGES = {"all": (None, None), "1-45": (1, 45), "1-15": (1, 15), "16-30": (16, 30),
          "31-45": (31, 45), "46+": (46, None)}
DIMENSIONS = ("risk_aversion", "advantageous_inequality_aversion")


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"{path.name}: records must be JSON objects")
    return rows


def _metadata(job: dict[str, Any]) -> dict[str, Any]:
    result = {name: job[name] for name in ("job_id", "combination_id", "treatment", "repetition", "seed", "assignment_seed")}
    for side in ("first", "second"):
        result.update({f"{side}_{key}": job[side][key] for key in ("agent_id", "personality_id", *DIMENSIONS, "label")})
    return result


def _validate_rows(rows: list[dict[str, Any]], members: list[int], *, settlements: bool) -> None:
    seen = set()
    for row in rows:
        agent, partner = row.get("agent_id"), row.get("partner_id")
        if (type(agent) is not int or type(partner) is not int
                or agent not in members or partner not in members or agent == partner
                or type(row.get("pair_id")) is not int or row["pair_id"] != 1):
            raise ValueError("recorded pair membership does not match the planned two participants")
        if settlements:
            period = row.get("period")
            if type(period) is not int or period < 1:
                raise ValueError("settlement period must be a positive integer")
            key = (agent, period)
            if key in seen:
                raise ValueError("duplicate participant-period settlement")
            seen.add(key)
        else:
            if agent in seen or row.get("period") != spec.INSURANCE_DECISION_PERIOD:
                raise ValueError("duplicate or out-of-period insurance record")
            seen.add(agent)


def inspect_job(batch_dir: Path, job: dict[str, Any], full_session: bool,
                 completed_period_limit: int) -> tuple[dict, list[dict], dict]:
    session = {**_metadata(job), "run_mode": "full_session" if full_session else "smoke",
               "status": "pending", "attempt_count": 0, "attempt": None, "run_dir": None,
               "success": None, "closing_memory_complete": None, "memory_experiment_complete": None,
               "natural_termination": None, "stop_reason": None, "completed_periods": 0,
               "actual_model_requests": None, "error": None,
               "first_purchase": None, "second_purchase": None,
               "first_assigned_insurance": None, "second_assigned_insurance": None}
    stats, records = {}, []
    attempts_root = batch_dir / "cases" / job["job_id"] / "attempts"
    attempts = sorted((p for p in attempts_root.iterdir() if p.is_dir() and p.name.isdigit()),
                      key=lambda p: (int(p.name), p.name)) if attempts_root.is_dir() else []
    if not attempts:
        return session, records, stats
    attempt = attempts[-1]
    session.update(attempt_count=len(attempts), attempt=attempt.name)
    try:
        if not attempt.resolve().is_relative_to(batch_dir.resolve()):
            raise ValueError("attempt must stay inside the batch directory")
        status_path = attempt / "status.json"
        status = _json(status_path) if status_path.is_file() else {}
        if not isinstance(status, dict) or status.get("run_dir", "run") != "run":
            raise ValueError("invalid attempt status or run directory")
        run_dir = attempt / "run"
        if not run_dir.resolve().is_relative_to(batch_dir.resolve()):
            raise ValueError("run must stay inside the batch directory")
        session["run_dir"] = run_dir.relative_to(batch_dir).as_posix()
        session["status"] = {"running": "running", "failed": "failed", "interrupted": "failed"}.get(status.get("status"), "incomplete")
        manifest_path = run_dir / "manifest.json"
        manifest = _json(manifest_path) if manifest_path.is_file() else {}
        if not isinstance(manifest, dict):
            raise ValueError("manifest must be a JSON object")
        if manifest and str(manifest.get("treatment", "")).lower() != job["treatment"]:
            raise ValueError("manifest treatment does not match the planned treatment")
        session.update(record_coverage.manifest_counts(manifest))
        for key in ("success", "closing_memory_complete", "memory_experiment_complete", "natural_termination",
                    "stop_reason", "actual_model_requests"):
            session[key] = manifest.get(key)
        records = _rows(run_dir / "period_records.jsonl")
        choices = _rows(run_dir / "insurance_decisions.jsonl") if job["treatment"] == "25v" else []
        assignments = _rows(run_dir / "insurance_assignments.jsonl") if job["treatment"] == "25f" else []
        if manifest and ("seed" in manifest or records or choices or assignments or manifest.get("success") is True):
            if type(manifest.get("seed")) is not int or manifest["seed"] != job["seed"]:
                raise ValueError("manifest seed does not match the planned repetition")
        metadata_path = run_dir / "pairing_metadata.json"
        if metadata_path.exists():
            metadata = _json(metadata_path)
            if not isinstance(metadata, dict) or any(metadata.get(key) != job[key]
                    for key in ("job_id", "combination_id", "treatment", "repetition", "seed", "assignment_seed", "pair_members", "first", "second", "profiles")):
                raise ValueError("saved pairing metadata does not match the planned repetition")
        profile_path = run_dir / "participant_profiles.json"
        if profile_path.exists():
            profile = _json(profile_path)
            if profile != {"schema_version": 1, "profiles": job["profiles"]}:
                raise ValueError("saved participant profiles do not match the planned personalities")
        elif records or choices or assignments or manifest.get("success") is True:
            raise ValueError("saved participant profiles are missing")
        members = job["pair_members"]
        _validate_rows(records, members, settlements=True)
        _validate_rows(choices, members, settlements=False)
        _validate_rows(assignments, members, settlements=False)
        for assignment in assignments:
            slots = assignment.get("pair_members")
            if (not isinstance(slots, list) or not all(type(member) is int for member in slots)
                    or slots != members):
                raise ValueError("insurance assignment first/second slots do not match the plan")
        for row in records:
            if "treatment" in row and str(row["treatment"]).lower() != job["treatment"]:
                raise ValueError("settlement treatment does not match the plan")
        data = build_report_data({**manifest, "treatment": "Baseline" if job["treatment"] == "baseline" else job["treatment"].upper()},
                                 records, choices, assignments=assignments)
        pair = data["pairs"][0] if data["pairs"] else None
        periods = sorted({row["period"] for row in records})
        session["completed_periods"] = pair["period_count"] if pair else 0
        if pair:
            stats = {stage: summarize_pair(pair, data["periods"], start, end)
                     for stage, (start, end) in STAGES.items()}
            session.update({f"observed_{key}": value for key, value in pair["summary_all"].items()})
            for side in ("first", "second"):
                member = next((m for m in pair["members"] if m["id"] == job[side]["agent_id"]), {})
                session[f"{side}_purchase"] = member.get("purchase")
                session[f"{side}_assigned_insurance"] = member.get("assigned")
        complete_records = bool(pair and periods and pair["period_count"] == len(periods)
                                and periods == list(range(1, periods[-1] + 1))
                                and manifest.get("completed_periods") == periods
                                and pair["summary_all"]["transfer_valid_periods"] == len(periods)
                                and pair["summary_all"]["income_valid_periods"] == len(periods))
        complete = (status.get("status") == "completed" and status.get("exit_code") == 0
                    and manifest.get("success") is True and manifest.get("memory_experiment_complete") is True
                    and manifest.get("closing_memory_complete") is True and complete_records)
        if full_session:
            complete = (complete and manifest.get("run_mode") == "full_session"
                        and manifest.get("natural_termination") is True
                        and manifest.get("stop_reason") == "natural_termination"
                        and spec.REAL_TERMINATION_FIRST_POSSIBLE_PERIOD <= periods[-1] <= spec.REAL_TERMINATION_FORCED_PERIOD)
        else:
            complete = (complete and manifest.get("run_mode") == "smoke"
                        and manifest.get("stop_reason") == "smoke_period_limit"
                        and len(periods) == completed_period_limit)
        if complete:
            session["status"] = "complete"
        elif session["status"] not in {"running", "failed"}:
            session["status"] = "incomplete"
        if session["status"] == "complete" and job["treatment"] != "baseline" and periods[-1] >= spec.INSURANCE_DECISION_PERIOD:
            valid_ledger = pair["submitted"] == 2 if job["treatment"] == "25v" else pair["assigned"] == 2
            if not valid_ledger or "核对" in pair["status"]:
                raise ValueError("insurance choice or assignment ledger is incomplete or invalid")
    except (OSError, ValueError, KeyError, TypeError, IndexError) as exc:
        session["status"] = "invalid"
        session["error"] = redact_credentials(f"{type(exc).__name__}: {exc}")
        return session, [], {}
    annotated = []
    for row in records:
        side = "first" if row["agent_id"] == job["first"]["agent_id"] else "second"
        partner_side = "second" if side == "first" else "first"
        annotated.append({**row, **_metadata(job),
                          "treatment": row.get("treatment", job["treatment"]),
                          "planned_treatment": job["treatment"], "run_mode": session["run_mode"],
                          "session_status": session["status"], "attempt": session["attempt"],
                          "personality_id": job[side]["personality_id"],
                          "partner_personality_id": job[partner_side]["personality_id"],
                          **{name: job[side][name] for name in DIMENSIONS},
                          **{f"partner_{name}": job[partner_side][name] for name in DIMENSIONS}})
    return session, annotated, stats


def _write_csv(path: Path, rows: list[dict[str, Any]], fallback: tuple[str, ...]) -> None:
    columns = list(dict.fromkeys(key for row in rows for key in row)) or list(fallback)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def write_batch_report(batch_dir: Path) -> dict[str, int]:
    """Rebuild derived tables/HTML without modifying attempts or calling models."""
    from experiments.preference_grid.design import load_plan

    batch_dir = Path(batch_dir).expanduser().resolve()
    plan = load_plan(batch_dir)
    full_session = plan["run_settings"].get("full_session", True)
    sessions, records, grouped = [], [], defaultdict(list)
    for job in plan["jobs"]:
        session, annotated, _ = inspect_job(batch_dir, job, full_session,
                                            plan["run_settings"].get("completed_period_limit", 3))
        # Source selection metadata is used by snapshot/runner tools only.
        session.pop("attempt")
        session.pop("attempt_count")
        if session["status"] == "complete":
            for row in annotated:
                row.pop("attempt")
            records.extend(annotated)
        else:
            # Keep the plan denominator and diagnostic, omit partial economic outcomes.
            session = {**_metadata(job), **{key: session[key] for key in
                       ("run_mode", "status", "error", "run_dir")}}
        sessions.append(session)
        grouped[(job["treatment"], job["combination_id"])].append(session)
    from experiments.preference_grid.analysis import METRICS, build_analysis

    analysis = build_analysis(sessions, records, plan)
    analysis["insurance_summary"] = build_insurance_summary(sessions)
    analysis["termination_distribution"] = build_termination_distribution(sessions)
    analysis["record_coverage"] = record_coverage.record_coverage(sessions)
    cells = {(r["treatment"], r["combination_id"], r["stage"], r["metric"]): r
             for r in analysis["cell_summaries"]}
    summaries = []
    for (treatment, combination), items in grouped.items():
        counts = Counter(session["status"] for session in items)
        first = items[0]
        for stage in STAGES:
            row = {"treatment": treatment, "combination_id": combination, "stage": stage,
                   "run_mode": first["run_mode"],
                   **{key: value for key, value in first.items() if key.startswith(("first_", "second_")) and key not in {
                       "first_agent_id", "second_agent_id", "first_purchase", "second_purchase",
                       "first_assigned_insurance", "second_assigned_insurance"}},
                   "n_planned": len(items), "n_completed": counts["complete"],
                   **{f"n_{status}": counts[status] for status in ("pending", "running", "failed", "incomplete", "invalid")}}
            for metric in METRICS:
                cell = cells.get((treatment, combination, stage, metric), {})
                row[f"{metric}_n"] = cell.get("n", 0)
                row[f"{metric}_mean"] = cell.get("mean")
                row[f"{metric}_sd"] = cell.get("sd")
            summaries.append(row)
    _write_csv(batch_dir / "pair_sessions.csv", sessions, ("job_id", "status"))
    _write_csv(batch_dir / "period_records.csv", records, ("job_id", "agent_id", "period", "personality_id", "partner_personality_id"))
    _write_csv(batch_dir / "combination_summary.csv", summaries, ("treatment", "combination_id", "stage", "n_completed"))
    # Hash only the economic ledgers and identity files actually used here;
    # reasoning logs, credentials and private model settings are not read.
    source_paths = [Path(__file__), Path(__file__).with_name("analysis.py"),
                    Path(__file__).with_name("record_coverage.py"), Path(__file__).with_name("insurance_summary.py"),
                    Path(__file__).with_name("termination_summary.py"),
                    Path(__file__).with_name("report_template.html"),
                    Path(__file__).resolve().parents[1] / "report_data.py"]
    provenance = {"generated_at": datetime.now(timezone.utc).isoformat(),
                  "plan_sha256": plan.get("plan_sha256"),
                  "analysis_packages": {name: version(name) for name in ("numpy", "scipy")},
                  "analysis_sources_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths},
                  "input_sha256": {}}
    for session in sessions:
        if not session.get("run_dir"):
            continue
        run = batch_dir / session["run_dir"]
        if not run.resolve().is_relative_to(batch_dir):
            continue
        for name in ("manifest.json", "period_records.jsonl", "insurance_decisions.jsonl",
                     "insurance_assignments.jsonl", "participant_profiles.json", "pairing_metadata.json"):
            path = run / name
            if path.is_file() and path.resolve().is_relative_to(batch_dir):
                provenance["input_sha256"][str(path.relative_to(batch_dir))] = hashlib.sha256(path.read_bytes()).hexdigest()
        status_path = run.parent / "status.json"
        if status_path.is_file() and status_path.resolve().is_relative_to(batch_dir):
            provenance["input_sha256"][str(status_path.relative_to(batch_dir))] = hashlib.sha256(status_path.read_bytes()).hexdigest()
    (batch_dir / "analysis_provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8")
    exports = {
        "insurance_summary.csv": analysis["insurance_summary"],
        "termination_distribution.csv": analysis["termination_distribution"],
        "record_coverage.csv": analysis["record_coverage"],
        "pair_stage_metrics.csv": analysis["pair_stages"],
        "agent_stage_metrics.csv": analysis["agent_stages"],
        "treatment_statistics.csv": analysis["treatment_summaries"],
        "personality_statistics.csv": analysis["preferences"],
        "period_statistics.csv": analysis["temporal"],
        "phase_changes.csv": analysis["phase_changes"],
        "treatment_contrasts.csv": analysis["contrasts"],
        "insurance_statistics.csv": analysis["insurance"],
        "insurance_rates.csv": analysis["insurance_rates"],
        "reciprocity_statistics.csv": analysis["reciprocity"],
        "orientation_comparisons.csv": analysis["orientation"],
        "preference_regressions.csv": [{**{k: v for k, v in model.items() if k != "coefficients"}, **coef}
                                       for model in analysis["regressions"] for coef in model.get("coefficients", [])],
    }
    for filename, rows in exports.items():
        _write_csv(batch_dir / filename, rows, ("treatment", "stage"))
    # Full reusable analysis data; the browser only needs aggregate and pair data.
    (batch_dir / "analysis.json").write_text(json.dumps(analysis, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    _write_analysis_summary(batch_dir, analysis)
    browser_analysis = {k: v for k, v in analysis.items() if k not in {"agent_stages", "regressions"}}
    browser_analysis["methods"] = {k: v for k, v in analysis["methods"].items() if k != "regression"}
    payload = json.dumps({"personalities": plan["personalities"], "treatments": plan["treatments"],
                          "run_mode": "full_session" if full_session else "smoke", "sessions": sessions,
                          "summaries": summaries, "stages": list(STAGES), "analysis": browser_analysis}, ensure_ascii=False, allow_nan=False)
    payload = payload.replace("&", "\\u0026").replace("<", "\\u003c").replace(">", "\\u003e")
    template = Path(__file__).with_name("report_template.html").read_text(encoding="utf-8")
    html = template.replace("__INSURANCE_SUMMARY__", render_insurance_summary(analysis["insurance_summary"]))
    html = html.replace("__ENDING_NOTE__", overall_ending_note(analysis["termination_distribution"]))
    html = html.replace("__RECORD_COVERAGE__", record_coverage.render_coverage(analysis["record_coverage"]))
    (batch_dir / "report.html").write_text(html.replace("__DATA__", payload), encoding="utf-8")
    return {"planned": len(sessions), **dict(Counter(session["status"] for session in sessions))}


def _write_analysis_summary(root: Path, analysis: dict) -> None:
    def fmt(value):
        return "—" if value is None else f"{value:.2f}"

    def pformat(value):
        return "—" if value is None else "<0.0001" if value < .0001 else f"{value:.4f}"

    def percent(value):
        return "—" if value is None else f"{value:.2%}"

    def treatment_label(treatment):
        return "Baseline" if treatment == "baseline" else treatment.upper()

    lines = ["# 人格配对实验结果", "", "每场完整的配对实验计一次。同场的两名参与者以及他们经历的各轮属于同一个样本。", "",
             "## 完成情况", "", "| 实验 | 完成场次 / 计划场次 | 已结算的配对轮数 |",
             "| --- | ---: | ---: |"]
    for row in analysis["coverage"]:
        lines.append(f"| {treatment_label(row['treatment'])} | {row['completed']} / {row['planned']} | {row['settled_pair_periods']} |")
    lines += ["", "## 自然结束轮次", "",
              "第 1 至 47 轮均会继续。第 48、49 轮各有 78% 的条件概率结束，到第 50 轮一定结束。"
              "全组占比以该实验全部自然结束的场次为分母，条件结束率以实际进入该轮的场次为分母。", "",
              "| 实验 | 轮次 | 自然结束总场次 | 进入该轮 | 本轮结束 | 继续下轮 | 全组占比 | 条件结束率 | 协议概率 |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in analysis["termination_distribution"]:
        lines.append(f"| {treatment_label(row['treatment'])} | {row['period']} | {row['n_natural']} | {row['at_risk']} | "
                     f"{row['ended']} | {row['continued']} | {percent(row['share_all'])} | "
                     f"{percent(row['conditional_rate'])} | {percent(row['protocol_rate'])} |")
    lines += ["", overall_ending_note(analysis["termination_distribution"])]
    lines += ["", "## 全程和分阶段转账", "", "平均转账先在每场内计算，再对场次等权平均。"
              "全程包含该场实际结算的所有轮次。95% 区间为场次均值的 Student t 区间。", "",
              "| 实验 | 阶段 | 场次 | 平均转账 | 95% 区间 | 场次间标准差 |",
              "| --- | --- | ---: | ---: | --- | ---: |"]
    for row in analysis["treatment_summaries"]:
        if row["metric"] == "transfer_mean":
            stage = "全程" if row["stage"] == "all" else row["stage"]
            lines.append(f"| {treatment_label(row['treatment'])} | {stage} | {row['n']} | {fmt(row['mean'])} | "
                         f"[{fmt(row['ci_low'])}, {fmt(row['ci_high'])}] | {fmt(row['sd'])} |")
    lines += ["", "## 阶段变化相对 Baseline 的差异", "", "先为每场计算后阶段减前阶段，再比较不同实验的场次变化量。"
              "相同重复编号不代表相同随机路径，因此跨实验使用独立样本 Welch 比较。"
              "这些是描述性差异，不直接解释为保险的因果效果。", "",
              "| 实验 − Baseline | 阶段变化 | 转账变化差 | 95% 区间 | Holm 校正 p |",
              "| --- | --- | ---: | --- | ---: |"]
    for row in analysis["contrasts"]:
        if row["kind"] == "change_vs_baseline" and row["metric"] == "transfer_mean":
            lines.append(f"| {row['left']} − {row['right']} | {row['stage']} | {fmt(row['estimate'])} | "
                         f"[{fmt(row['ci_low'])}, {fmt(row['ci_high'])}] | {pformat(row['p_holm'])} |")
    lines += ["", "## 第 16 轮的购买与分配", "",
              "25V 统计参与者自己选择购买的人数，25F 统计环境随机分配到保险的人数。"
              "后四列以配对场数计，第一位和第二位对应实验计划中的两个席位。", "",
              "| 实验 | 购买或获配人数 / 有效人数 | 比例 | 有效场次 | 双方均无 | 仅第一位 | 仅第二位 | 双方都有 |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in analysis["insurance_summary"]:
        rate = row["insured_rate"]
        label = "25V 自主购买" if row["treatment"] == "25v" else "25F 环境分配"
        lines.append(f"| {label} | {row['insured_people']} / {row['eligible_people']} | "
                     + (f"{rate:.2%}" if rate is not None else "—")
                     + " | " + " | ".join(str(row[key]) for key in
                        ("valid_pairs", "neither_pairs", "first_only_pairs", "second_only_pairs", "both_pairs")) + " |")
    lines += ["", "## 转账是否足以平分当轮收入", "",
              "每次发送者机会按实际转账除以平分所需金额计算，后者等于转账前发送者与接收者的收入差的一半。"
              "100% 对应两人最终收入相等，超过 100% 表示转账后接收者收入更高。"
              "每场先取可计算轮次的均值，再在场次间等权平均。精确平分率按实际结算后的两人收入相等统计。", "",
              "| 实验 | 阶段 | 平均平分比例（有效场次） | 精确平分率（有效场次） |",
              "| --- | --- | ---: | ---: |"]
    fixed = {(r["treatment"], r["stage"], r["metric"]): r for r in analysis["treatment_summaries"]}
    def show_stat(treatment, stage, metric):
        stat = fixed.get((treatment, stage, metric), {})
        return f"{stat['mean']:.2%}（{stat['n']}）" if stat.get("n") else "—（0）"

    for item in analysis["coverage"]:
        treatment = item["treatment"]
        for stage in ("all", "1-15", "16-30", "31-45"):
            ratio = show_stat(treatment, stage, "equalization_ratio")
            equal = show_stat(treatment, stage, "equal_income_rate")
            lines.append(f"| {treatment_label(treatment)} | {'全程' if stage == 'all' else stage} | {ratio} | {equal} |")
    lines += ["", "## 决策、推理文本与记忆的覆盖情况", "", record_coverage.NOTE, "",
              "| " + " | ".join(record_coverage.COLUMNS) + " |",
              "| " + " | ".join("---" for _ in record_coverage.COLUMNS) + " |"]
    for values in record_coverage.display_rows(analysis["record_coverage"]):
        lines.append("| " + " | ".join(values) + " |")
    lines += ["", "## 进一步阅读", "",
              "打开 report.html 可查看轮次曲线、重复分布、人格热图、偏好分组、投保与获保、互惠条件行为和方向对比。"
              "点击热图单元格可查看该组合的每次重复与对应原始报告。", "",
              "- 没有转账机会的轮次不计入 Agent 的零转账选择。互惠条件只使用此前已结算的搭档发送者选择，未曾观察到与观察到零转账分开。",
              "- 25V 投保率是自主购买行为；25F 获保率是环境分配结果。分组对比不将自选择或随机分配混为一谈。",
              f"- 本批 25F 分配概率：{analysis['assignment_probabilities']}（双方、仅第一位、仅第二位）。这是运行配置，论文概率仍未确认。",
              "- Baseline 的每人平均收入恒为 225 是转账守恒的结果，不能说明风险降低或合作程度相同。收入波动是 Agent 的阶段内实际收入总体标准差，不是效用或风险厌恶参数估计。",
              "- 46+ 为随机延长阶段；49、50 轮只包含存续场次，不应把其均值变化全归于学习。",
              "- 全期每人平均收入是对该场已实现收入逐轮等权计算的值，与随机抽轮支付相对应；不是所有轮次收入的累计支付。",
              "- 小样本、全零或完全相同的重复不能证明行为无不确定性。n<2 不给区间；零方差差异不生成显著性检验。",
              "- 总报告未自动阅读或归纳推理/记忆正文，因而不能据此声称已识别信任、动机或私人记忆的因果作用。",
              "- 所有跨实验比较共享一个 Holm 校正族，方向对比另用一个校正族；图中的 95% 区间仍是逐项区间，不是同时置信区间。", "",
              "方法参考：[SciPy Welch 检验](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_ind.html)、"
              "[CR1 聚类协方差](https://www.statsmodels.org/stable/generated/statsmodels.stats.sandwich_covariance.cov_cluster.html)。", ""]
    (root / "analysis_summary.md").write_text("\n".join(lines), encoding="utf-8")

