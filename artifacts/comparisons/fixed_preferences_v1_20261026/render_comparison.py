"""Render local descriptive outputs; never change an experiment or call a model.

Run summarize_runs.py first, then this script with the project virtualenv.
Paper means and SDs are transcribed from Table 2, printed page 841.
"""

from pathlib import Path
from statistics import mean
import csv
import hashlib
import json

import plotly.graph_objects as go
from plotly.subplots import make_subplots


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
STATS = json.loads((HERE / "statistics.json").read_text())
PHASES = ("pre", "insurance", "post")
LABELS = ("1–15", "16–30", "31–45")
COLORS = {"baseline": "#276FBF", "25V": "#D95F02", "25F": "#368B61"}
PAPER = {
    "baseline": {"n": 32, "means": [64, 59, 52], "sd": [35, 41, 42]},
    "25V": {"n": 36, "means": [65, 48, 59], "sd": [44, 39, 43]},
    "25F": {"n": 40, "means": [78, 60, 72], "sd": [34, 36, 39]},
}


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_csv(name, rows):
    with (HERE / name).open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


phase_rows, pair_rows, preference_rows, fingerprints = [], [], [], {}
for treatment, record in STATS.items():
    run = ROOT / record["run"]
    raw = read_rows(run / "period_records.jsonl")
    donors = [r for r in raw if r["had_transfer_opportunity"]]
    assert len(raw) == 20 * 48 and len(donors) == 10 * 48
    for phase, (lo, hi) in zip(PHASES, ((1, 15), (16, 30), (31, 45))):
        assert record["phases"][phase]["n"] == 150
        assert mean(r["transfer_sent"] for r in donors if lo <= r["period"] <= hi) == record["phases"][phase]["mean"]
    for index, phase in enumerate(PHASES):
        for source in ("paper_table_2", "LLM_run"):
            phase_rows.append({
                "source": source, "treatment": treatment, "periods": LABELS[index],
                "participants": PAPER[treatment]["n"] if source == "paper_table_2" else 20,
                "mean_transfer": PAPER[treatment]["means"][index] if source == "paper_table_2" else record["phases"][phase]["mean"],
                "sd_transfer": PAPER[treatment]["sd"][index] if source == "paper_table_2" else record["phases"][phase]["sd"],
            })
    for pair_id, pair in record["by_pair"].items():
        pair_rows.append({
            "treatment": treatment, "pair_id": int(pair_id),
            "members": "/".join(map(str, pair["members"])),
            "insurance_category": "not_applicable" if treatment == "baseline" else pair["insurance_category"],
            **{phase + "_mean": pair["phase_means"][phase] for phase in PHASES},
            "post_minus_pre": pair["post_minus_pre"],
        })
    for name in ("manifest.json", "period_records.jsonl", "participant_profiles.json"):
        fingerprints[str((run / name).relative_to(ROOT))] = hashlib.sha256((run / name).read_bytes()).hexdigest()
    if treatment == "25V":
        profiles = json.loads((run / "participant_profiles.json").read_text())["profiles"]
        decisions = read_rows(run / "insurance_decisions.jsonl")
        for dimension in ("risk_aversion", "advantageous_inequality_aversion"):
            for level in ("low", "medium", "high"):
                ids = {int(agent) for agent, profile in profiles.items() if profile[dimension] == level}
                observed = [d for d in decisions if d["agent_id"] in ids]
                transfers = [r["transfer_sent"] for r in donors if r["agent_id"] in ids and r["period"] <= 15]
                preference_rows.append({
                    "dimension": dimension, "level": level, "agents": len(ids),
                    "purchases": sum(d["purchase"] for d in observed),
                    "purchase_rate": mean(d["purchase"] for d in observed),
                    "pre_transfer_mean_opportunity_weighted": mean(transfers),
                    "pre_transfer_opportunities": len(transfers),
                })

write_csv("phase_comparison.csv", phase_rows)
write_csv("pair_comparison.csv", pair_rows)
write_csv("preference_associations.csv", preference_rows)
(HERE / "input_fingerprints.json").write_text(json.dumps(fingerprints, indent=2) + "\n")

fig = make_subplots(
    rows=2, cols=2, vertical_spacing=0.18, horizontal_spacing=0.1,
    subplot_titles=("论文 Table 2：阶段平均转账", "本次 LLM：同口径阶段平均转账",
                    "本次 LLM：逐轮平均转账（每轮 10 pair）", "各 pair 的后期减前期：31–45 减 1–15"),
)
for treatment, color in COLORS.items():
    record = STATS[treatment]
    for col, values in ((1, PAPER[treatment]["means"]), (2, [record["phases"][p]["mean"] for p in PHASES])):
        fig.add_trace(go.Scatter(
            x=list(LABELS), y=values, mode="lines+markers", name=treatment,
            legendgroup=treatment, showlegend=col == 1,
            line={"color": color, "width": 3, "dash": "dash" if col == 1 else "solid"},
            hovertemplate="阶段 %{x}<br>平均转账 %{y:.2f}<extra>" + treatment + "</extra>",
        ), row=1, col=col)
    donors = [r for r in read_rows(ROOT / record["run"] / "period_records.jsonl") if r["had_transfer_opportunity"]]
    fig.add_trace(go.Scatter(
        x=list(range(1, 49)), y=[mean(r["transfer_sent"] for r in donors if r["period"] == p) for p in range(1, 49)],
        name=treatment, legendgroup=treatment, showlegend=False,
        mode="lines", line={"color": color, "width": 2},
        hovertemplate="第 %{x} 轮<br>平均转账 %{y:.2f}<extra>" + treatment + "</extra>",
    ), row=2, col=1)
    pairs = sorted(record["by_pair"], key=int)
    fig.add_trace(go.Bar(
        x=[int(p) for p in pairs], y=[record["by_pair"][p]["post_minus_pre"] for p in pairs],
        name=treatment, legendgroup=treatment, showlegend=False, marker_color=color,
        hovertemplate="Pair %{x}<br>后期减前期 %{y:.2f}<extra>" + treatment + "</extra>",
    ), row=2, col=2)
fig.add_vrect(x0=15.5, x1=30.5, fillcolor="#D95F02", opacity=0.08, line_width=0, row=2, col=1)
fig.add_vrect(x0=45.5, x1=48.5, fillcolor="#888888", opacity=0.12, line_width=0, row=2, col=1)
fig.add_hline(y=0, line_color="#555555", line_width=1, row=2, col=2)
for row, col in ((1, 1), (1, 2), (2, 1)):
    fig.update_yaxes(range=[0, 120], title_text="转账点数", row=row, col=col)
fig.update_yaxes(title_text="变化点数", row=2, col=2)
fig.update_xaxes(title_text="轮次", row=2, col=1)
fig.update_xaxes(title_text="固定 pair", dtick=1, row=2, col=2)
fig.update_layout(
    title="固定偏好 LLM 实验与原论文：描述性比较", template="plotly_white",
    height=850, margin={"t": 100, "b": 80}, barmode="group",
    font={"family": "Arial, sans-serif", "size": 13},
    legend={"orientation": "h", "y": -0.11},
)
html = fig.to_html(full_html=True, include_plotlyjs=True, config={"responsive": True, "displaylogo": False})
caption = (
    '<div style="max-width:1200px;margin:20px auto;font:15px/1.6 sans-serif;padding:0 20px">'
    '<p>每轮只计有转账机会者，保留零选择；论文比较窗口是 1–45 轮。'
    '橙色区域为保险期，灰色区域 46–48 轮不进入阶段比较。'
    '每种模拟包含 10 个 pair，重复轮次不是独立受试者。此图不表示显著性检验。</p>'
    '<p>论文阶段均值来自 Lin, Meng &amp; Weng (2020), Table 2，已取整。'
    '点击图例可隐藏/显示 treatment，鼠标悬停可查看数值。'
    '<a href="comparison.md">详细说明</a> · '
    '<a href="phase_comparison.csv">阶段数据 CSV</a> · '
    '<a href="pair_comparison.csv">pair 数据 CSV</a></p></div>'
)
html = html.replace("<body>", "<body>" + caption, 1)
(HERE / "comparison.html").write_text(html, encoding="utf-8")
print(HERE / "comparison.html")
