"""Offline, lossless re-rendering of recorded inputs; never calls a model."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

from custom.agents.observation_presentation import render_readable_turn


def extract(content):
    return json.loads(content.split("<risk_sharing_turn>", 1)[1].split("</risk_sharing_turn>", 1)[0])


def main():
    runs = {
        "baseline": "20261025T134921Z-99290f7f",
        "25v": "20261025T145836Z-65a997b1",
        "25f": "20261026T032217Z-54514d0b",
    }
    results = []
    for treatment, run_id in runs.items():
        source = ROOT / "experiments/fixed_preferences_v1/runs" / treatment / run_id / "participant_contexts.jsonl"
        count, lengths = 0, []
        with source.open(encoding="utf-8") as stream:
            for line in stream:
                context = json.loads(line)
                for message in context["model_messages"]:
                    content = message.get("content")
                    if message.get("role") != "user" or not isinstance(content, str) or "<risk_sharing_turn>" not in content:
                        continue
                    original = extract(content)
                    readable = render_readable_turn(original, agent_id=context["agent_id"])
                    expected = json.loads(json.dumps(original))
                    for item in expected["observations"]:
                        if isinstance(item.get("observation"), str):
                            item["observation"] = json.loads(item["observation"])
                    assert extract(readable) == expected, (source, context["agent_id"], context["period"])
                    count += 1
                    if context["period"] == 45:
                        lengths.append((len(content), len(readable)))
                    if (treatment, context["agent_id"], context["period"]) in {
                        ("25f", 10, 2), ("25v", 6, 30),
                    }:
                        name = f"{treatment}_agent_{context['agent_id']}_period_{context['period']}_preview.md"
                        (HERE / name).write_text(readable + "\n", encoding="utf-8")
        results.append({
            "treatment": treatment, "run_id": run_id,
            "source": str(source.relative_to(ROOT)), "inputs_checked": count,
            "all_fields_equal_after_decoding": True,
            "period_45_mean_characters": {
                "legacy": sum(old for old, new in lengths) / len(lengths),
                "readable_v1": sum(new for old, new in lengths) / len(lengths),
            },
        })
    result = {
        "mode": "offline_reformat_only", "external_model_calls": 0,
        "renderer_sha256": hashlib.sha256((ROOT / "custom/agents/observation_presentation.py").read_bytes()).hexdigest(),
        "runs": results,
        "total_inputs_checked": sum(item["inputs_checked"] for item in results),
    }
    (HERE / "historical_input_audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
