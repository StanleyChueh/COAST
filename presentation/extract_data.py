"""Extract every measured number used by the deck from the research records.

Reads   research/reproduction/experiments/*.{csv,yaml,json}   (never writes there)
Writes  presentation/generated_figures/chart_data.json         (values + provenance)
        presentation/generated_figures/data_checks.txt         (cross-check log)

Each value is recomputed from per-episode records where they exist and then
asserted against the summary value the phase report states. If a record
changes and the deck would silently drift, this script fails instead.

Run from the repository root:
    uv run --no-sync python presentation/extract_data.py
"""

from __future__ import annotations

import csv
import json
import math
import pathlib
from collections import defaultdict

import yaml
from scipy.stats import fisher_exact

ROOT = pathlib.Path(__file__).resolve().parents[1]
EXP = ROOT / "research" / "reproduction" / "experiments"
OUT = ROOT / "presentation" / "generated_figures"

checks: list[str] = []


def rel(p: pathlib.Path) -> str:
    return str(p.relative_to(ROOT))


def rows(name: str) -> list[dict]:
    with open(EXP / name, newline="") as f:
        return list(csv.DictReader(f))


def load_yaml(name: str) -> dict:
    with open(EXP / name) as f:
        return yaml.safe_load(f)


def check(label: str, got, expected, tol: float = 0.0) -> None:
    ok = abs(got - expected) <= tol if isinstance(got, float) or isinstance(expected, float) else got == expected
    checks.append(f"{'OK  ' if ok else 'FAIL'} {label}: computed={got!r} reported={expected!r}")
    if not ok:
        raise AssertionError(checks[-1])


def wilson(k: int, n: int, z: float = 1.959964) -> list[float]:
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [round(c - h, 3), round(c + h, 3)]


def frac(k: int, n: int, source: str, **extra) -> dict:
    return {"k": k, "n": n, "rate": round(k / n, 3), "wilson95": wilson(k, n), "label": f"{k}/{n}", "source": source, **extra}


def fisher_p(k1: int, n1: int, k2: int, n2: int) -> float:
    return float(fisher_exact([[k1, n1 - k1], [k2, n2 - k2]])[1])


def count(rs: list[dict], col: str = "success") -> int:
    return sum(int(r[col]) for r in rs)


def agreement(a: list[int], b: list[int]) -> int:
    return sum(int(x == y) for x, y in zip(a, b, strict=True))


data: dict = {"_generated_by": "presentation/extract_data.py", "_source_root": rel(EXP)}

# ---------------------------------------------------------------- Phase 0 / 1A
p0 = load_yaml("phase0_smoke_test.yaml")
data["phase0"] = {"smoke": frac(p0["baseline_successes"], p0["num_episodes"], "phase0_smoke_test.yaml")}
p1a = load_yaml("phase1a_activation_collection.yaml")
data["phase1a"] = {"fit": frac(p1a["successes"], 15, "phase1a_activation_collection.yaml (states 0-14, unseeded)")}
check("1A fit successes", p1a["successes"], 8)

# ---------------------------------------------------------------- Phase 1C / 1D
r1c = rows("phase1c_episode_results.csv")
r1d = rows("phase1d_episode_results.csv")
src1d = "phase1d_episode_results.csv"
nf = {s: [int(r[f"nf{s}_success"]) for r in r1d] for s in (100, 200, 300)}
mixed = sum(1 for i in range(15) if len({nf[100][i], nf[200][i], nf[300][i]}) > 1)
data["phase1c"] = {
    "baseline": frac(count(r1c, "baseline_success"), 15, "phase1c_episode_results.csv (states 15-29, unseeded)"),
    "v0": frac(count(r1c, "repo_success"), 15, "phase1c_episode_results.csv"),
    "v3": frac(count(r1c, "paper_success"), 15, "phase1c_episode_results.csv"),
    "agree_v0_v3": agreement([int(r["repo_success"]) for r in r1c], [int(r["paper_success"]) for r in r1c]),
}
data["phase1d"] = {
    "noise_floor": {str(s): frac(sum(v), 15, f"{src1d} nf{s}_success (states 15-29, master seed {s})") for s, v in nf.items()},
    "states_mixed_across_seeds": mixed,
    "baseline": frac(count(r1d, "baseline_success"), 15, src1d),
    "v0": frac(count(r1d, "repo_success"), 15, src1d),
    "v3": frac(count(r1d, "paper_success"), 15, src1d),
    "agree_v0_v3": agreement([int(r["repo_success"]) for r in r1d], [int(r["paper_success"]) for r in r1d]),
}
check("1D noise floor", [data["phase1d"]["noise_floor"][s]["k"] for s in ("100", "200", "300")], [13, 7, 10])
check("1D states mixed", mixed, 11)
check("1C baseline", data["phase1c"]["baseline"]["k"], 8)
check("1C V0/V3 agreement", data["phase1c"]["agree_v0_v3"], 9)
check("1D V0/V3 agreement", data["phase1d"]["agree_v0_v3"], 15)

# ---------------------------------------------------------------- Phase 1E
p1e = load_yaml("phase1e_intervention_diagnostics.yaml")
res = p1e["results"]
dist = res["distribution_per_application"]
data["phase1e"] = {
    "source": "phase1e_intervention_diagnostics.yaml (V0, L5 a0.5 b0.1 global, states 15-29, seed 100)",
    "hook_applications": res["counts"]["hook_applications"],
    "rel_delta_mean": dist["mean_relative_delta"]["mean"],
    "rel_delta_std": dist["mean_relative_delta"]["std"],
    "cos_mean": dist["mean_cosine_delta_hidden"]["mean"],
    "cos_min": dist["mean_cosine_delta_hidden"]["min"],
    "cos_max": dist["mean_cosine_delta_hidden"]["max"],
    "hidden_norm_mean": dist["mean_hidden_norm"]["mean"],
    "effective_scale": res["active_layer_5"]["effective_scale_if_pure_shrink"],
    "rayleigh_text": "≈ 0.006",
    "lambda_max": p1e["conceptor"]["spectrum_offline"]["eig_max"],
    "lambda_median": p1e["conceptor"]["spectrum_offline"]["eig_median"],
    "trace_over_d": p1e["conceptor"]["spectrum_offline"]["trace_over_d"],
    "by_step_delta": res["by_denoising_step"]["mean_delta_norm"],
    "by_step_hidden": res["by_denoising_step"]["mean_hidden_norm"],
}
check("1E rel delta", data["phase1e"]["rel_delta_mean"], 0.09942)
check("1E cosine", data["phase1e"]["cos_mean"], -0.99962)
assert "0.006" in res["active_layer_5"]["implied_rayleigh_quotient_hCh_over_hh"]

# ---------------------------------------------------------------- Phase 2A / 2B / 3A
r2a = rows("phase2a_episode_results.csv")
r2b = rows("phase2b_episode_results.csv")
v2b = {c: [int(r[f"{c}_success"]) for r in r2b] for c in ("baseline", "shrinkage", "coast", "random_matched")}
ratio2b = {c: round(sum(float(r[f"{c}_mean_norm_ratio"]) for r in r2b) / 15, 4) for c in ("shrinkage", "coast", "random_matched")}
data["phase2a"] = {
    "baseline": frac(count(r2a, "baseline_success"), 15, "phase2a_episode_results.csv"),
    "coast": frac(count(r2a, "coast_success"), 15, "phase2a_episode_results.csv"),
    "shrinkage": frac(count(r2a, "shrinkage_success"), 15, "phase2a_episode_results.csv"),
    "agree_coast_shrink": agreement([int(r["coast_success"]) for r in r2a], [int(r["shrinkage_success"]) for r in r2a]),
}
data["phase2b"] = {
    **{c: frac(sum(v), 15, "phase2b_episode_results.csv (states 15-29, paired noise seed 100)") for c, v in v2b.items()},
    "agree_coast_random": agreement(v2b["coast"], v2b["random_matched"]),
    "agree_coast_shrink": agreement(v2b["coast"], v2b["shrinkage"]),
    "norm_ratio": ratio2b,
}
check("2B counts", [data["phase2b"][c]["k"] for c in v2b], [13, 10, 11, 11])
check("2B agree coast/random", data["phase2b"]["agree_coast_random"], 15)
check("2B agree coast/shrink", data["phase2b"]["agree_coast_shrink"], 14)
check("2A agree coast/shrink", data["phase2a"]["agree_coast_shrink"], 14)

r3a = rows("phase3a_episode_results.csv")
by3 = defaultdict(dict)
ratio3 = defaultdict(list)
for r in r3a:
    by3[r["condition"]][int(r["init_state"])] = int(r["success"])
    if r["mean_norm_ratio"]:
        ratio3[r["condition"]].append(float(r["mean_norm_ratio"]))
vec3 = {c: [by3[c][s] for s in range(30, 45)] for c in by3}
data["phase3a"] = {
    **{c: frac(sum(v), 15, "phase3a_episode_results.csv (held-out states 30-44, paired noise seed 100)") for c, v in vec3.items()},
    "agree_coast_random": agreement(vec3["coast"], vec3["random_matched"]),
    "agree_coast_shrink": agreement(vec3["coast"], vec3["shrinkage"]),
    "mcnemar_base_coast_p": 0.5,
}
check("3A counts", [data["phase3a"][c]["k"] for c in ("baseline", "shrinkage", "coast", "random_matched")], [9, 10, 11, 11])
check("3A agree coast/random", data["phase3a"]["agree_coast_random"], 13)
check("3A agree coast/shrink", data["phase3a"]["agree_coast_shrink"], 14)

# ---------------------------------------------------------------- Phase 4A
r4a = rows("phase4a_episode_results.csv")


def c4(state_set: str, cond: str, lo: int | None = None, hi: int | None = None) -> tuple[int, int]:
    sel = [r for r in r4a if r["state_set"] == state_set and r["condition"] == cond]
    if lo is not None:
        sel = [r for r in sel if lo <= int(r["init_state"]) <= hi]
    return count(sel), len(sel)


src4 = "phase4a_episode_results.csv (states 15-44, paired noise seed 100)"
data["phase4a"] = {
    "S_baseline": frac(*c4("S", "baseline"), src4),
    "S_coast_b01": frac(*c4("S", "coast_b01"), src4),
    "S_shrink_b01": frac(*c4("S", "shrink_b01"), src4),
    "all50": {
        c: frac(c4("S", c)[0] + c4("P", c)[0], 50, "phase4a_episode_results.csv (P + S, all 50 states)")
        for c in ("baseline", "coast_b01", "coast_b02", "coast_b03", "shrink_b01", "shrink_b02", "shrink_b03")
    },
}
check("4A S baseline", data["phase4a"]["S_baseline"]["k"], 21)
check("4A S coast b0.1", data["phase4a"]["S_coast_b01"]["k"], 20)
check("4A all-50 COAST b0.1/0.2/0.3", [data["phase4a"]["all50"][c]["k"] for c in ("coast_b01", "coast_b02", "coast_b03")], [32, 33, 27])

# ---------------------------------------------------------------- Phase 5A / 5C
r5a = rows("phase5a_ks3_episode_results.csv")
src5a = "phase5a_ks3_episode_results.csv (states 15-44, unseeded)"
g5a = frac(count([r for r in r5a if r["condition"].startswith("coast_global")]), 30, src5a)
b5a = frac(count([r for r in r5a if r["condition"] == "baseline"]), 30, src5a)
r5c = rows("phase5c_test_episode_results.csv")
src5c = "phase5c_test_episode_results.csv (states 15-44, unseeded)"
t5c = {c: frac(count([r for r in r5c if r["condition"] == c]), 30, src5c) for c in ("baseline", "global", "positive_only", "per_step")}
sel5c = {r["condition"]: r for r in rows("phase5c_test_results.csv")}
for c in ("global", "positive_only", "per_step"):
    t5c[c]["fit"] = f"{sel5c[c]['fit_success']}/{sel5c[c]['fit_n']}"
    t5c[c]["config"] = f"L{sel5c[c]['layer']} α{sel5c[c]['alpha']} β{sel5c[c]['beta']}"
    t5c[c]["p_vs_paper"] = float(sel5c[c]["fisher_p_vs_paper"])
    check(f"5C {c} test k", t5c[c]["k"], int(sel5c[c]["test_success"]))
p5 = load_yaml("phase5c_ks3_oracle_sweep.yaml")
fitb = p5["fit_results"]["fit_state_baselines"]
fit_baselines = [fitb[k] for k in ("global_L0_5", "global_L11_17", "positive_only_L0_5", "positive_only_L11_17", "per_step")]
data["phase5a"] = {"global": g5a, "baseline": b5a, "p_vs_paper": fisher_p(19, 30, 28, 30)}
data["phase5c"] = {"test": t5c, "fit_baselines": fit_baselines, "fit_baselines_source": "phase5c_ks3_oracle_sweep.yaml fit_state_baselines"}
check("5A global/baseline", [g5a["k"], b5a["k"]], [19, 19])
check("5A Fisher vs 28/30", round(data["phase5a"]["p_vs_paper"], 3), 0.010, 0.0005)
check("5C Fisher global vs 28/30", round(fisher_p(21, 30, 28, 30), 3), round(t5c["global"]["p_vs_paper"], 3))
check("5C fit baselines", fit_baselines, [10, 5, 4, 11, 6])
pool_g = g5a["k"] + data["phase4a"]["S_coast_b01"]["k"] + t5c["global"]["k"]
pool_b = b5a["k"] + data["phase4a"]["S_baseline"]["k"] + t5c["baseline"]["k"]
data["ks3_pooled_15_44"] = {
    "global": frac(pool_g, 90, "5A + 4A(S, b0.1) + 5C global runs"),
    "baseline": frac(pool_b, 90, "5A + 4A(S) + 5C baseline runs"),
    "p_global_vs_baseline": fisher_p(pool_g, 90, pool_b, 90),
    "p_global_vs_paper": fisher_p(pool_g, 90, 28, 30),
}
check("pooled global 60/90", pool_g, 60)
check("pooled baseline 57/90", pool_b, 57)
check("pooled vs paper p", round(data["ks3_pooled_15_44"]["p_global_vs_paper"], 3), 0.004, 0.0005)

# ---------------------------------------------------------------- Phase 6C
r6c = [r for r in rows("phase6c_candidate_test_results.csv") if r["record"] == "summary"]
src6c = "phase6c_candidate_test_results.csv (miranda-v2 @ 29059a5, states 0-14, unseeded)"


def runs6(cond: str) -> list[dict]:
    return [frac(int(r["successes"]), int(r["n_episodes"]), src6c, run=r["run"]) for r in r6c if r["condition"] == cond]


c6 = {
    "baseline": runs6("baseline"),
    "global": runs6("global_L5_a0.5_b0.1"),
    "random_b01": runs6("random_L5_b0.1"),
    "random_b03": runs6("random_L5_b0.3"),
    "per_step_0": runs6("per_step_0_L5_a10.0_b0.3"),
    "per_step_9": runs6("per_step_9_L5_a10.0_b0.3"),
    "pos_only": runs6("pos_only_L11_a1.0_b0.1"),
}
gk = sum(x["k"] for x in c6["global"])
bk = sum(x["k"] for x in c6["baseline"])
rk = sum(x["k"] for x in c6["random_b01"])
data["phase6c"] = {
    **c6,
    "paper_native": {"global": "14/15", "per_step": "13/15", "pos_only": "12/15", "base": "8/15"},
    "pooled": {"global": f"{gk}/30", "baseline": f"{bk}/45", "random_b01": f"{rk}/30"},
    "p_global_vs_baseline": fisher_p(gk, 30, bk, 45),
    "p_global_vs_random": fisher_p(gk, 30, rk, 30),
    "p_global_vs_paper_native": fisher_p(gk, 30, 14, 15),
}
check("6C global runs", [x["k"] for x in c6["global"]], [13, 12])
check("6C baseline runs", [x["k"] for x in c6["baseline"]], [11, 10, 12])
check("6C random b0.1 runs", [x["k"] for x in c6["random_b01"]], [11, 12])
check("6C p global vs baseline", round(data["phase6c"]["p_global_vs_baseline"], 2), 0.40)
check("6C p global vs random", round(data["phase6c"]["p_global_vs_random"], 2), 0.75)
check("6C p global vs paper", round(data["phase6c"]["p_global_vs_paper_native"], 2), 0.65)

bm = rows("phase6c_branch_matrix.csv")
branch_rows = [r for r in bm if r["branch"].startswith("upstream/")]
pr_row = next(r for r in bm if r["branch"].startswith("refs/pull"))
data["phase6c"]["branches"] = {
    "public_branches": len(branch_rows),
    "pr_heads": int(pr_row["branch"].split("(")[1].split()[0]),
    "deep_inspected": sum(1 for r in branch_rows if r["inspected_deeply"].strip().lower() == "yes"),
    "source": "phase6c_branch_matrix.csv",
}
check("6C public branches", data["phase6c"]["branches"]["public_branches"], 22)
check("6C PR heads", data["phase6c"]["branches"]["pr_heads"], 50)

# ---------------------------------------------------------------- Phase 7A
r7 = rows("phase7a_summary.csv")
a7 = json.load(open(EXP / "phase7a_analysis.json"))
y7 = load_yaml("phase7a_cross_task_validation.yaml")
tasks7 = ["LR2a", "KS4", "LR1"]
cond7 = ["BASELINE", "PAPER_GLOBAL", "RANDOM_CONTROL"]
s7: dict = {}
for t in tasks7:
    s7[t] = {"paper_base": a7["tasks"][t]["paper_base"], "paper_global": a7["tasks"][t]["paper_global"], "config": a7["tasks"][t]["config"]}
    for arm in ("H", "T"):
        s7[t][arm] = {}
        for c in cond7:
            sel = {r["repeat"]: r for r in r7 if r["task"] == t and r["arm"] == arm and r["condition"] == c}
            pooled = sel["pooled"]
            reps = [int(sel["1"]["successes"]), int(sel["2"]["successes"])]
            k, n = int(pooled["successes"]), int(pooled["n"])
            check(f"7A {t} {arm} {c} reps sum", sum(reps), k)
            s7[t][arm][c] = frac(k, n, "phase7a_summary.csv (pooled over 2 repeats; same init states)", reps=reps, rep_n=n // 2)
    for key in ("G_H", "G_T", "PGG"):
        s7[t][key] = a7["tasks"][t][key]
    gh = s7[t]["H"]["PAPER_GLOBAL"]["rate"] - s7[t]["H"]["BASELINE"]["rate"]
    gt = s7[t]["T"]["PAPER_GLOBAL"]["rate"] - s7[t]["T"]["BASELINE"]["rate"]
    check(f"7A {t} PGG recomputed", round(gh - gt, 2), round(a7["tasks"][t]["PGG"], 2), 0.011)
    s7[t]["p_T_global_vs_base"] = a7["tasks"][t]["arms"]["T"]["fisher_global_vs_base"]
    s7[t]["p_H_global_vs_base"] = a7["tasks"][t]["arms"]["H"]["fisher_global_vs_base"]
ctp = y7["results"]["cross_task_pooled"]
data["phase7a"] = {
    "tasks": s7,
    "task_order": tasks7,
    "mean_PGG": a7["mean_PGG"],
    "n_episodes": a7["n_episode_rows"],
    "pooled": ctp,
    "case": "A",
    "operators": [
        {k: (v if k in ("task", "operator") else (float(v) if v else None)) for k, v in r.items()}
        for r in rows("phase7a_operator_diagnostics.csv")
    ],
}
tot = {c: sum(s7[t]["T"][c]["k"] for t in tasks7) for c in cond7}
check("7A pooled T global", f"{tot['PAPER_GLOBAL']}/180", ctp["T"]["PAPER_GLOBAL"])
check("7A pooled T random", f"{tot['RANDOM_CONTROL']}/180", ctp["T"]["RANDOM_CONTROL"])
check("7A pooled T baseline", f"{tot['BASELINE']}/180", ctp["T"]["BASELINE"])
check("7A episodes", a7["n_episode_rows"], 810)
cosines = [o["cos_dh_h"] for o in data["phase7a"]["operators"] if o["operator"] == "PAPER_GLOBAL"]
data["phase7a"]["global_cos_range"] = [min(cosines), max(cosines)]

OUT.mkdir(parents=True, exist_ok=True)
(OUT / "chart_data.json").write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
(OUT / "data_checks.txt").write_text("\n".join(checks) + f"\n\n{len(checks)} checks passed.\n")
print(f"wrote {rel(OUT / 'chart_data.json')}; {len(checks)} cross-checks passed")
