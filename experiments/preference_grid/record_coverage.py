"""Report the size and availability of the evidence for each treatment."""

from html import escape


def manifest_counts(manifest):
    memory = manifest.get("memory_report")
    memory = memory if isinstance(memory, dict) else {}
    return {
        "trace_responses": manifest.get("participant_reasoning_responses"),
        "readable_responses": manifest.get("participant_reasoning_readable_responses"),
        "truncated_responses": manifest.get("participant_reasoning_truncated_responses"),
        "memory_updates": memory.get("update_count"),
        "closing_updates": memory.get("closing_update_count"),
    }


def _observed_pair(row, total, subset):
    return (type(row.get(total)) is int and type(row.get(subset)) is int
            and 0 <= row[subset] <= row[total])


def record_coverage(sessions):
    result = []
    for treatment in dict.fromkeys(row["treatment"] for row in sessions):
        complete = [s for s in sessions if s["treatment"] == treatment and s["status"] == "complete"]
        traces = [s for s in complete if _observed_pair(s, "trace_responses", "readable_responses")]
        memory = [s for s in complete if _observed_pair(s, "memory_updates", "closing_updates")]
        truncated = [s for s in traces if _observed_pair(s, "trace_responses", "truncated_responses")]
        def count(rows, key):
            return sum(s[key] for s in rows) if rows else None

        result.append({
            "treatment": treatment, "complete_sessions": len(complete),
            "sender_opportunities": sum(s["completed_periods"] for s in complete),
            "trace_reporting_sessions": len(traces), "trace_responses": count(traces, "trace_responses"),
            "readable_responses": count(traces, "readable_responses"),
            "truncation_reporting_sessions": len(truncated), "truncated_responses": count(truncated, "truncated_responses"),
            "memory_reporting_sessions": len(memory),
            "feedback_updates": sum(s["memory_updates"] - s["closing_updates"] for s in memory) if memory else None,
            "closing_updates": count(memory, "closing_updates"),
        })
    return result


COLUMNS = ("实验", "完整场次", "转账机会", "推理可读条数 / 已记录响应", "推理计数覆盖场次",
           "常规记忆更新", "终局补记", "记忆计数覆盖场次")
NOTE = ("每个结算轮次提供一次转账机会。模型响应计数取自参与者运行清单，包含决策、格式修正及记忆调用。"
        "可读条数以有文字返回为准。反馈更新和终局补记取自记忆导出清单。分母仅含提供对应计数的完整场次，缺项显示为 —。")


def display_rows(rows):
    def fmt(value):
        return "—" if value is None else f"{value:,}"
    for r in rows:
        yield (r["treatment"], fmt(r["complete_sessions"]), fmt(r["sender_opportunities"]),
               f'{fmt(r["readable_responses"])} / {fmt(r["trace_responses"])}', fmt(r["trace_reporting_sessions"]),
               fmt(r["feedback_updates"]), fmt(r["closing_updates"]), fmt(r["memory_reporting_sessions"]))


def render_coverage(rows):
    headers = "".join(f"<th>{x}</th>" for x in COLUMNS)
    body = "".join("<tr>" + "".join(f"<td>{escape(x)}</td>" for x in row) + "</tr>" for row in display_rows(rows))
    return (f'<h3>决策、推理文本和私人记忆的记录量</h3><div class="scroll"><table id="record-coverage">'
            f'<thead><tr>{headers}</tr></thead><tbody>{body}</tbody></table></div><p class="note">{NOTE}</p>')
