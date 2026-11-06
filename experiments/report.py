"""Standalone, offline charts for experiment records; no model calls."""

from __future__ import annotations

import json
import re
from pathlib import Path

from experiments.report_data import build_report_data


REPORT_LINK = "[打开交互图表报告](report.html) · 可筛选组别、查看逐轮数值和下载图表"
TEMPLATE = Path(__file__).with_name("report_template.html")


def add_report_link(markdown: str) -> str:
    """Put one report entry below the summary title, preserving other content."""

    if REPORT_LINK in markdown:
        return markdown
    title, separator, rest = markdown.partition("\n")
    return title + "\n\n" + REPORT_LINK + "\n" + separator + rest


def write_visual_report(run_dir: Path, manifest, records, decisions=(), *, assignments=()) -> Path:
    """Embed Plotly and only the chart data so the HTML works without a server."""

    from plotly.offline import get_plotlyjs

    data = build_report_data(manifest, records, decisions, assignments=assignments)
    data["run_id"] = run_dir.name
    # Records are data, including string identifiers. Never let them terminate
    # an HTML script element or turn into markup.
    payload = json.dumps(data, ensure_ascii=False, allow_nan=False)
    payload = payload.replace("&", "\\u0026").replace("<", "\\u003c").replace(">", "\\u003e")
    template = TEMPLATE.read_text(encoding="utf-8")
    values = {"__REPORT_DATA__": payload, "__PLOTLY_JS__": get_plotlyjs()}
    rendered = re.sub(r"__REPORT_DATA__|__PLOTLY_JS__", lambda match: values[match[0]], template)
    path = run_dir / "report.html"
    path.write_text(rendered, encoding="utf-8")
    return path


def main(argv=None) -> int:
    """Rebuild a historical run's derived charts without replaying the run."""

    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    run_dir = parser.parse_args(argv).run_dir.expanduser().resolve()

    def read_rows(name):
        return [json.loads(line) for line in (run_dir / name).read_text(encoding="utf-8").splitlines() if line.strip()]

    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    # A forced assignment is not a participant purchase. Select the ledger
    # by treatment, even if an unrelated file is present in an old directory.
    forced = manifest.get("treatment") == "25F"
    choices = (
        read_rows("insurance_decisions.jsonl")
        if manifest.get("treatment") == "25V" and (run_dir / "insurance_decisions.jsonl").is_file()
        else []
    )
    assignments = (
        read_rows("insurance_assignments.jsonl")
        if forced and (run_dir / "insurance_assignments.jsonl").is_file()
        else []
    )
    path = write_visual_report(
        run_dir, manifest, read_rows("period_records.jsonl"), choices,
        assignments=assignments,
    )
    summary = run_dir / "summary.md"
    original = summary.read_text(encoding="utf-8")
    updated = add_report_link(original)
    if updated != original:
        try:
            with (run_dir / "summary.before-visual-report.md").open("x", encoding="utf-8") as handle:
                handle.write(original)
        except FileExistsError:
            pass
        summary.write_text(updated, encoding="utf-8")
    print(f"Visual report: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
