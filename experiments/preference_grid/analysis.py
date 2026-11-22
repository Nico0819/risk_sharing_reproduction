"""Offline exploratory analyses. A pair-session is the independent cluster.

No model calls, fitting to paper targets, or changes to experiment records.
Only sessions accepted by report.inspect_job enter these calculations.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from itertools import combinations
import math

import numpy as np
from scipy import stats


WINDOWS = {"all": (1, None), "1-45": (1, 45), "1-15": (1, 15),
           "16-30": (16, 30), "31-45": (31, 45), "46+": (46, None)}
METRICS = ("transfer_mean", "transfer_positive_rate", "income_mean", "income_gap_mean",
           "income_sd", "zero_transfer_rate", "equal_income_rate", "equalization_ratio")
LEVELS = ("low", "medium", "high")
DIMENSIONS = ("risk_aversion", "advantageous_inequality_aversion")


def number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def average(values):
    values = [v for v in values if number(v)]
    return float(np.mean(values)) if values else None


def describe(values):
    """Student-t interval across independent session summaries; n=1 has no CI."""
    x = np.asarray([v for v in values if number(v)], dtype=float)
    result = dict(n=len(x), mean=None, sd=None, median=None, q25=None, q75=None,
                  minimum=None, maximum=None, se=None, ci_low=None, ci_high=None)
    if not len(x):
        return result
    mean = float(x.mean())
    result.update(mean=mean, median=float(np.median(x)), q25=float(np.quantile(x, .25)),
                  q75=float(np.quantile(x, .75)), minimum=float(x.min()), maximum=float(x.max()))
    if len(x) > 1:
        sd = float(x.std(ddof=1))
        se = sd / math.sqrt(len(x))
        margin = float(stats.t.ppf(.975, len(x) - 1)) * se
        result.update(sd=sd, se=se, ci_low=mean - margin, ci_high=mean + margin)
    return result


def cluster_mean(rows, metric):
    """Equal agent weights, with uncertainty clustered by independent job_id."""
    valid = [r for r in rows if number(r.get(metric))]
    result = describe([r[metric] for r in valid])
    clusters = defaultdict(list)
    for row in valid:
        clusters[row["job_id"]].append(row[metric])
    g, n = len(clusters), len(valid)
    result.update(n_agents=n, n_pairs=g, se=None, ci_low=None, ci_high=None)
    if g > 1:
        mean = result["mean"]
        scores = [sum(v - mean for v in values) for values in clusters.values()]
        se = math.sqrt(g / (g - 1) * sum(v * v for v in scores)) / n
        margin = float(stats.t.ppf(.975, g - 1)) * se
        result.update(se=se, ci_low=mean - margin, ci_high=mean + margin)
    return result


def difference(left, right):
    """Independent samples: left minus right; repetition labels are not pairs."""
    a, b = describe(left), describe(right)
    out = dict(estimate=None, se=None, ci_low=None, ci_high=None, p_value=None,
               n_left=a["n"], n_right=b["n"], mean_left=a["mean"], mean_right=b["mean"])
    if not a["n"] or not b["n"]:
        return out
    delta = a["mean"] - b["mean"]
    out["estimate"] = delta
    if min(a["n"], b["n"]) < 2:
        return out
    va, vb = a["sd"] ** 2 / a["n"], b["sd"] ** 2 / b["n"]
    se = math.sqrt(va + vb)
    # No test for a degenerate estimated sampling distribution.
    if se == 0:
        return out
    df = (va + vb) ** 2 / (va * va / (a["n"] - 1) + vb * vb / (b["n"] - 1))
    margin = float(stats.t.ppf(.975, df)) * se
    out.update(se=se, ci_low=delta - margin, ci_high=delta + margin,
               p_value=float(2 * stats.t.sf(abs(delta / se), df)))
    return out


def holm(rows):
    """Holm correction within the explicitly reported family of comparisons."""
    ranked = sorted((r for r in rows if number(r.get("p_value"))), key=lambda r: r["p_value"])
    previous = 0.0
    for index, row in enumerate(ranked):
        previous = max(previous, min(1.0, (len(ranked) - index) * row["p_value"]))
        row["p_holm"] = previous
    for row in rows:
        row.setdefault("p_holm", None)


def fit_cluster_ols(y, x, groups):
    """OLS + CR1 covariance, t(G-1). Rank-deficient designs are not fitted."""
    y, x = np.asarray(y, dtype=float), np.asarray(x, dtype=float)
    if x.ndim != 2 or not len(y):
        return None
    n, k = x.shape
    keys = list(dict.fromkeys(groups))
    g = len(keys)
    if n <= k or g <= k or np.linalg.matrix_rank(x) != k:
        return None
    beta, _, _, _ = np.linalg.lstsq(x, y, rcond=None)
    residual = y - x @ beta
    scores = defaultdict(lambda: np.zeros(k))
    for group, score in zip(groups, x * residual[:, None]):
        scores[group] += score
    meat = sum((np.outer(v, v) for v in scores.values()), start=np.zeros((k, k)))
    bread = np.linalg.inv(x.T @ x)
    covariance = bread @ meat @ bread * (g / (g - 1)) * ((n - 1) / (n - k))
    se = np.sqrt(np.maximum(np.diag(covariance), 0))
    critical = float(stats.t.ppf(.975, g - 1))
    centered = float(np.sum((y - y.mean()) ** 2))
    return {"n_agents": n, "n_pairs": g, "r_squared": 1 - float(residual @ residual) / centered if centered else None,
            "coefficients": [{"estimate": float(b), "se": float(s),
                              "ci_low": float(b - critical * s), "ci_high": float(b + critical * s)}
                             for b, s in zip(beta, se)]}


def _in_window(row, window):
    start, end = window
    return row["period"] >= start and (end is None or row["period"] <= end)


def _insurance_group(session):
    field = "purchase" if session["treatment"] == "25v" else "assigned_insurance"
    values = [session.get(f"{side}_{field}") for side in ("first", "second")]
    return str(sum(values)) if all(type(v) is bool for v in values) else "unknown"


def build_analysis(sessions, records, plan):
    complete = {s["job_id"]: s for s in sessions if s["status"] == "complete"}
    by_job = defaultdict(list)
    for row in records:
        if row["job_id"] in complete:
            by_job[row["job_id"]].append(row)
    pair_stages, agent_stages, trajectory, reciprocity_agents = [], [], [], []
    for job_id, session in complete.items():
        raw = sorted(by_job[job_id], key=lambda r: (r["period"], r["agent_id"]))
        by_period, by_agent = defaultdict(list), defaultdict(list)
        common = {key: session[key] for key in ("job_id", "treatment", "combination_id", "repetition")}
        common["insurance_group"] = _insurance_group(session) if session["treatment"] != "baseline" else "none"
        for row in raw:
            by_period[row["period"]].append(row)
            by_agent[row["agent_id"]].append(row)
        periods, lag_events, previous_sender = [], [], {}
        for period, members in sorted(by_period.items()):
            sender = next(r for r in members if r.get("had_transfer_opportunity") is True)
            incomes = [r["final_income"] for r in members]
            amount = sender["transfer_sent"]
            receiver = next(r for r in members if r["agent_id"] != sender["agent_id"])
            before = [r.get("income_before_transfer") for r in (sender, receiver)]
            initial_gap = before[0] - before[1] if all(number(v) for v in before) else None
            gap = abs(incomes[0] - incomes[1])
            pair = {**common, "period": period, "transfer_mean": amount,
                    "transfer_positive_rate": int(amount > 0), "zero_transfer_rate": int(amount == 0),
                    "income_mean": average(incomes), "income_gap_mean": gap,
                    "equal_income_rate": int(gap == 0),
                    "before_gap": abs(initial_gap) if initial_gap is not None else None,
                    "equalization_ratio": 2 * amount / initial_gap if initial_gap is not None and initial_gap > 0 else None}
            periods.append(pair)
            trajectory.append(pair)
            # Strictly earlier settled sender opportunity, never receiver bookkeeping zero.
            previous = previous_sender.get(sender["partner_id"])
            lag_events.append({**sender, "lag_group": "no_previous" if previous is None else
                               ("previous_zero" if previous == 0 else "previous_positive")})
            previous_sender[sender["agent_id"]] = amount
        for stage, window in WINDOWS.items():
            stage_periods = [p for p in periods if _in_window(p, window)]
            if not stage_periods:
                continue
            agents = []
            for agent_id, history in by_agent.items():
                subset = [r for r in history if _in_window(r, window)]
                if not subset:
                    continue
                sender_rows = [r for r in subset if r.get("had_transfer_opportunity") is True]
                incomes = [r["final_income"] for r in subset]
                side = "first" if agent_id == session["first_agent_id"] else "second"
                partner_side = "second" if side == "first" else "first"
                a = {**common, "stage": stage, "agent_id": agent_id, "slot": side,
                     "personality_id": session[f"{side}_personality_id"],
                     "partner_personality_id": session[f"{partner_side}_personality_id"],
                     **{dim: session[f"{side}_{dim}"] for dim in DIMENSIONS},
                     **{f"partner_{dim}": session[f"{partner_side}_{dim}"] for dim in DIMENSIONS},
                     "period_count": len(subset), "sender_opportunities": len(sender_rows),
                     "transfer_mean": average([r["transfer_sent"] for r in sender_rows]),
                     "transfer_positive_rate": average([int(r["transfer_sent"] > 0) for r in sender_rows]),
                     "income_mean": average(incomes), "income_sd": float(np.std(incomes)),
                     "income_minimum": min(incomes),
                     "insurance_choice": session.get(f"{side}_purchase") if session["treatment"] == "25v" else
                                         session.get(f"{side}_assigned_insurance") if session["treatment"] == "25f" else None}
                agent_stages.append(a)
                agents.append(a)
                for lag in ("no_previous", "previous_zero", "previous_positive"):
                    eligible = [r for r in lag_events if r["agent_id"] == agent_id and
                                _in_window(r, window) and r["lag_group"] == lag]
                    if eligible:
                        reciprocity_agents.append({**a, "lag_group": lag, "eligible_opportunities": len(eligible),
                                                   "transfer_mean": average([r["transfer_sent"] for r in eligible])})
            pair_stages.append({**common, "stage": stage, "period_count": len(stage_periods),
                                "equalization_valid_periods": sum(p["equalization_ratio"] is not None for p in stage_periods),
                                **{metric: average([p[metric] for p in stage_periods]) for metric in METRICS if metric != "income_sd"},
                                "income_sd": average([a["income_sd"] for a in agents]),
                                "before_gap": average([p["before_gap"] for p in stage_periods])})

    treatment_summaries, cell_summaries, distributions = [], [], []
    for keys, output in ((("treatment", "stage"), treatment_summaries),
                         (("treatment", "stage", "combination_id"), cell_summaries)):
        groups = defaultdict(list)
        for row in pair_stages:
            groups[tuple(row[k] for k in keys)].append(row)
        for key, rows in groups.items():
            for metric in METRICS:
                output.append({**dict(zip(keys, key)), "metric": metric, **describe([r[metric] for r in rows])})
    for key, rows in _groups(pair_stages, ("treatment", "stage")).items():
        for low, high in ((0, 1e-12), (1e-12, 25), (25, 50), (50, 75), (75, 100), (100, 125), (125, float("inf"))):
            count = sum(low <= r["transfer_mean"] < high for r in rows)
            distributions.append({"treatment": key[0], "stage": key[1], "bin": "0" if low == 0 else
                                  "(0,25)" if low == 1e-12 else f"[{low},{high if math.isfinite(high) else '∞'})",
                                  "count": count, "share": count / len(rows)})
    temporal = []
    for (treatment, period), rows in _groups(trajectory, ("treatment", "period")).items():
        for metric in ("transfer_mean", "transfer_positive_rate", "income_gap_mean"):
            temporal.append({"treatment": treatment, "period": period, "metric": metric,
                             **describe([r[metric] for r in rows])})

    preference, reciprocity, insurance = [], [], []
    for (treatment, stage), rows in _groups(agent_stages, ("treatment", "stage")).items():
        for dimension in (*DIMENSIONS, *(f"partner_{d}" for d in DIMENSIONS), "personality_id"):
            for (level,), subset in _groups(rows, (dimension,)).items():
                for metric in ("transfer_mean", "transfer_positive_rate", "income_mean", "income_sd"):
                    preference.append({"treatment": treatment, "stage": stage, "dimension": dimension,
                                       "level": level, "metric": metric, **cluster_mean(subset, metric)})
    for (treatment, stage, lag), rows in _groups(reciprocity_agents, ("treatment", "stage", "lag_group")).items():
        reciprocity.append({"treatment": treatment, "stage": stage, "lag_group": lag,
                            "eligible_opportunities": sum(r["eligible_opportunities"] for r in rows),
                            **cluster_mean(rows, "transfer_mean")})
    for (treatment, stage, group), rows in _groups(pair_stages, ("treatment", "stage", "insurance_group")).items():
        if treatment == "baseline":
            continue
        for metric in METRICS:
            insurance.append({"treatment": treatment, "stage": stage, "insured_count": group,
                              "metric": metric, **describe([r[metric] for r in rows])})
    insurance_rates = []
    for (treatment, personality), rows in _groups([a for a in agent_stages if a["stage"] == "16-30" and
                                                 type(a["insurance_choice"]) is bool], ("treatment", "personality_id")).items():
        insurance_rates.append({"treatment": treatment, "personality_id": personality,
                                **cluster_mean([{**r, "choice": int(r["insurance_choice"])} for r in rows], "choice")})

    transitions = (("introduction", "1-15", "16-30"), ("removal", "16-30", "31-45"),
                   ("post_vs_pre", "1-15", "31-45"))
    deltas = []
    for job_id, rows in _groups(pair_stages, ("job_id",)).items():
        stages = {r["stage"]: r for r in rows}
        for transition, pre, post in transitions:
            # Require two complete 15-round windows, including in smoke reports.
            if pre not in stages or post not in stages or any(stages[s]["period_count"] != 15 for s in (pre, post)):
                continue
            for metric in METRICS:
                if metric == "equalization_ratio" and any(stages[s]["equalization_valid_periods"] != 15 for s in (pre, post)):
                    continue
                a, b = stages[pre][metric], stages[post][metric]
                if number(a) and number(b):
                    deltas.append({"job_id": job_id[0], "treatment": stages[pre]["treatment"],
                                   "transition": transition, "metric": metric, "delta": b - a})
    changes = [{"treatment": t, "transition": transition, "metric": metric, **describe([r["delta"] for r in rows])}
               for (t, transition, metric), rows in _groups(deltas, ("treatment", "transition", "metric")).items()]
    contrasts = []
    for stage in WINDOWS:
        for metric in METRICS:
            for left, right in combinations(plan["treatments"], 2):
                contrasts.append({"kind": "level", "stage": stage, "metric": metric, "left": right, "right": left,
                                  **difference([r[metric] for r in pair_stages if r["stage"] == stage and r["treatment"] == right],
                                               [r[metric] for r in pair_stages if r["stage"] == stage and r["treatment"] == left])})
    for transition, _, _ in transitions:
        for metric in METRICS:
            for treatment in plan["treatments"]:
                if treatment == "baseline" or "baseline" not in plan["treatments"]:
                    continue
                contrasts.append({"kind": "change_vs_baseline", "stage": transition, "metric": metric,
                                  "left": treatment, "right": "baseline",
                                  **difference([r["delta"] for r in deltas if r["transition"] == transition and r["metric"] == metric and r["treatment"] == treatment],
                                               [r["delta"] for r in deltas if r["transition"] == transition and r["metric"] == metric and r["treatment"] == "baseline"])})
    holm(contrasts)
    orientation = []
    for (treatment, stage), rows in _groups(pair_stages, ("treatment", "stage")).items():
        combos = _groups(rows, ("combination_id",))
        for (combo,), forward in combos.items():
            a, b = combo.split("__")
            if a >= b or (f"{b}__{a}",) not in combos:
                continue
            reverse = combos[(f"{b}__{a}",)]
            orientation.append({"treatment": treatment, "stage": stage, "forward": combo, "reverse": f"{b}__{a}",
                                **difference([r["transfer_mean"] for r in forward], [r["transfer_mean"] for r in reverse])})
    holm(orientation)

    regressions = []
    terms = ["intercept"] + [f"{prefix}{dim}:{level}" for prefix in ("", "partner_") for dim in DIMENSIONS for level in LEVELS[1:]]
    for (treatment, stage), rows in _groups(agent_stages, ("treatment", "stage")).items():
        if stage not in ("1-15", "16-30", "31-45"):
            continue
        for metric in ("transfer_mean", "transfer_positive_rate"):
            valid = [r for r in rows if number(r.get(metric))]
            x = [[1] + [int(r[term.split(":")[0]] == term.split(":")[1]) for term in terms[1:]] for r in valid]
            fit = fit_cluster_ols([r[metric] for r in valid], x, [r["job_id"] for r in valid])
            regressions.append({"treatment": treatment, "stage": stage, "metric": metric,
                                "status": "estimated" if fit else "insufficient_or_rank_deficient",
                                **({**fit, "coefficients": [{"term": term, **coef} for term, coef in zip(terms, fit["coefficients"])]} if fit else {})})

    coverage = []
    for (treatment,), subset in _groups(sessions, ("treatment",)).items():
        valid = [s for s in subset if s["status"] == "complete"]
        coverage.append({"treatment": treatment, "planned": len(subset), "completed": len(valid),
                         "status_counts": dict(Counter(s["status"] for s in subset)),
                         "settled_pair_periods": sum(s["completed_periods"] for s in valid)})
    return {"schema_version": 1, "confidence_level": .95,
            "methods": {"mean_intervals": "Student t on independent pair-session means",
                        "agent_intervals": "Equal agent weights; CR1 intercept-only variance clustered by job_id",
                        "contrasts": "Independent Welch intervals; within-session changes computed before cross-treatment contrasts",
                        "multiplicity": "Holm over all reported treatment contrasts; separate Holm family for orientation contrasts; CIs are pointwise",
                        "regression": "OLS, categorical own/partner preferences, reference=low; CR1 job clusters, t(G-1); exploratory additive model",
                        "interpretation": "Descriptive simulation outcomes, not validated human behavior or identification of memory effects"},
            "coverage": coverage, "treatment_summaries": treatment_summaries, "cell_summaries": cell_summaries,
            "pair_stages": pair_stages, "agent_stages": agent_stages, "temporal": temporal,
            "distributions": distributions, "preferences": preference, "reciprocity": reciprocity,
            "insurance": insurance, "insurance_rates": insurance_rates, "phase_changes": changes,
            "contrasts": contrasts, "orientation": orientation, "regressions": regressions,
            "assignment_probabilities": plan.get("forced_assignment_probabilities"),
            "complete_agent_instances": 2 * len(complete)}


def _groups(rows, keys):
    result = defaultdict(list)
    for row in rows:
        result[tuple(row[k] for k in keys)].append(row)
    return result
