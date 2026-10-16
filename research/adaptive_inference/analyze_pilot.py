"""Stage-5 pilot analysis: same-state intervention matrix and the preregistered gate.

Implements exactly the estimands of PILOT_PREREG.md (numbered as there). Written and
hashed before any pilot outcome was inspected.

    uv run python research/adaptive_inference/analyze_pilot.py \
        --run results/adaptive_inference/pilot
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import pathlib

import numpy as np
from scipy import stats

ROUTE = ("I0", "IL", "IC", "IH")  # routing set; tie order = cost order
INTERV = ("IH", "IL", "IC")
PLACEBO = "I0P"
ALL_ARMS = ("I0", "I0P", "IH", "IL", "IC")
N_BOOT = 4000
GATE = {"A_pp": 5.0, "A_pareto_pp": 2.5, "B_unique": 0.05, "B_min_arms": 2, "C_share": 0.90, "D_pp": 2.5}
LAMBDAS = (0.0, 0.01, 0.02, 0.05, 0.1)


# --------------------------------------------------------------------------- load


def load(run: pathlib.Path):
    rows = []
    for f in sorted((run / "units").glob("*.jsonl")):
        rows.extend(json.loads(line) for line in f.read_text().splitlines() if line.strip())
    return rows


def validity(rows) -> dict:
    ident = [r for r in rows if r["kind"] == "identity"]
    br = [r for r in rows if r["kind"] == "branch"]
    exc = [r for r in br + ident if r.get("exception")]
    restore_ok = [r for r in br + ident if r.get("restore_state_ok") and r.get("restore_obs_ok")]
    return {
        "units": len({(r["task_id"], r["init_state"]) for r in rows if r["kind"] == "reference"}),
        "decision_states": len([r for r in rows if r["kind"] == "decision"]),
        "identity_branches": len(ident),
        "identity_ok": sum(bool(r.get("identity_ok")) for r in ident),
        "identity_failures": [(r["group_id"], r.get("exception")) for r in ident if not r.get("identity_ok")],
        "branches": len(br),
        "restore_ok": len(restore_ok),
        "restore_checked": len(br) + len(ident),
        "exceptions": len(exc),
        "exception_messages": collections.Counter((r["exception"] or "")[:100] for r in exc).most_common(10),
    }


def build(rows, n_rep: int):
    """Y[state] -> {arm: np.array(R)} for states with every arm x replicate valid."""
    by_state = collections.defaultdict(dict)
    meta = {}
    for r in rows:
        if r["kind"] == "decision":
            meta[r["group_id"]] = r
        if r["kind"] != "branch":
            continue
        by_state[r["group_id"]][(r["intervention_id"], r["replicate"])] = r
    states, dropped = [], []
    for g, d in sorted(by_state.items()):
        ok = all(
            (a, k) in d and not d[(a, k)].get("exception") and d[(a, k)].get("success") is not None
            for a in ALL_ARMS
            for k in range(n_rep)
        )
        (states if ok else dropped).append(g)
    y = {
        g: {a: np.array([int(by_state[g][(a, k)]["success"]) for k in range(n_rep)]) for a in ALL_ARMS} for g in states
    }
    cost = {
        g: {a: np.array([float(by_state[g][(a, k)]["server_infer_ms_total"]) for k in range(n_rep)]) for a in ALL_ARMS}
        for g in states
    }
    prog = {
        g: {
            a: np.array([int(by_state[g][(a, k)]["goal_predicates_satisfied"]) for k in range(n_rep)]) for a in ALL_ARMS
        }
        for g in states
    }
    task = {g: by_state[g][("I0", 0)]["task_id"] for g in states}
    unit = {g: (by_state[g][("I0", 0)]["task_id"], by_state[g][("I0", 0)]["init_state"]) for g in states}
    step = {g: by_state[g][("I0", 0)]["decision_step"] for g in states}
    return states, dropped, y, cost, prog, task, unit, step, meta


# ------------------------------------------------------------------ estimators


def _choose(means: dict, arms) -> str:
    best = max(means[a] for a in arms)
    return next(a for a in arms if means[a] == best)  # tie -> earliest (cheapest) in `arms`


def always_on(states, y):
    return {a: float(np.mean([y[g][a].mean() for g in states])) for a in ALL_ARMS}


def loo_state_oracle(states, y, arms=ROUTE, cost=None, lam=0.0, cost_ref=None):
    vals, costs = [], []
    for g in states:
        n = len(y[g][arms[0]])
        for r in range(n):
            m = {a: (y[g][a].sum() - y[g][a][r]) / (n - 1) for a in arms}
            if lam and cost_ref is not None:
                m = {a: m[a] - lam * cost_ref[a] / cost_ref["I0"] for a in arms}
            a = _choose(m, arms)
            vals.append(y[g][a][r])
            if cost is not None:
                costs.append(cost[g][a][r])
    return float(np.mean(vals)), (float(np.mean(costs)) if costs else None)


def naive_oracle(states, y, arms=ROUTE):
    return float(np.mean([np.max(np.stack([y[g][a] for a in arms]), 0).mean() for g in states]))


def loo_task_oracle(states, y, task, arms=ROUTE):
    vals = []
    tasks = sorted({task[g] for g in states})
    n = len(y[states[0]][arms[0]])
    for t in tasks:
        ts = [g for g in states if task[g] == t]
        for r in range(n):
            m = {a: np.mean([(y[g][a].sum() - y[g][a][r]) / (n - 1) for g in ts]) for a in arms}
            a = _choose(m, arms)
            vals.extend(y[g][a][r] for g in ts)
    return float(np.mean(vals))


def loo_risk_gate(states, y):
    """RL^2-style: if LOO base success <= 0.5, use the LOO globally best intervention, else I0."""
    n = len(y[states[0]]["I0"])
    vals = []
    for r in range(n):
        glob = {a: np.mean([(y[g][a].sum() - y[g][a][r]) / (n - 1) for g in states]) for a in INTERV}
        best_i = _choose(glob, ("IL", "IC", "IH"))
        for g in states:
            risk_ok = (y[g]["I0"].sum() - y[g]["I0"][r]) / (n - 1) > 0.5
            vals.append(y[g]["I0" if risk_ok else best_i][r])
    return float(np.mean(vals))


def rescue_stats(states, y, competitors):
    salv, uniq, resc, harm, base_succ = 0, collections.Counter(), collections.Counter(), collections.Counter(), 0
    n = len(y[states[0]]["I0"])
    for g in states:
        for r in range(n):
            b = y[g]["I0"][r]
            if b == 1:
                base_succ += 1
                for a in competitors:
                    harm[a] += int(y[g][a][r] == 0)
                continue
            succ = [a for a in competitors if y[g][a][r] == 1]
            for a in succ:
                resc[a] += 1
            if succ:
                salv += 1
                if len(succ) == 1:
                    uniq[succ[0]] += 1
    tot = sum(resc.values())
    return {
        "salvageable": salv,
        "unique": {a: uniq[a] for a in competitors},
        "unique_rate": {a: uniq[a] / salv if salv else float("nan") for a in competitors},
        "rescues": {a: resc[a] for a in competitors},
        "rescue_share": {a: resc[a] / tot if tot else float("nan") for a in competitors},
        "harm_rate": {a: harm[a] / base_succ if base_succ else float("nan") for a in competitors},
        "base_success_pairs": base_succ,
    }


def complementarity(states, y, arms=ALL_ARMS):
    n = len(y[states[0]]["I0"])
    tot = len(states) * n
    out = {}
    for a in arms:
        for b in arms:
            if a != b:
                out[f"{a}>{b}"] = sum(int(((y[g][a] == 1) & (y[g][b] == 0)).sum()) for g in states) / tot
    return out


def risk_utility(states, y):
    bins = collections.defaultdict(list)
    for g in states:
        bins[float(y[g]["I0"].mean())].append(g)
    table, cont = {}, []
    for b in sorted(bins):
        gs = bins[b]
        winners = collections.Counter()
        for g in gs:
            p = {a: y[g][a].mean() for a in ROUTE}
            mx = max(p.values())
            w = [a for a in ROUTE if p[a] == mx]
            winners[w[0] if len(w) == 1 else "tie"] += 1
        deltas = {a: [float(y[g][a].mean() - y[g]["I0"].mean()) for g in gs] for a in INTERV}
        table[b] = {
            "n_states": len(gs),
            "winner_counts": dict(winners),
            "delta_mean": {a: float(np.mean(v)) for a, v in deltas.items()},
            "delta_var": {a: float(np.var(v)) for a, v in deltas.items()},
            "frac_states_any_positive_delta": float(
                np.mean([max(deltas[a][i] for a in INTERV) > 0 for i in range(len(gs))])
            ),
        }
        cont.append([winners.get(a, 0) for a in ROUTE])
    cont = np.array(cont)
    keep_r = cont.sum(1) > 0
    keep_c = cont.sum(0) > 0
    sub = cont[keep_r][:, keep_c]
    test = None
    if sub.shape[0] >= 2 and sub.shape[1] >= 2:
        chi2, p, dof, _ = stats.chi2_contingency(sub)
        test = {
            "chi2": float(chi2),
            "p": float(p),
            "dof": int(dof),
            "table_rows_bins": [b for b, k in zip(sorted(bins), keep_r, strict=True) if k],
            "table_cols": [a for a, k in zip(ROUTE, keep_c, strict=True) if k],
            "table": sub.tolist(),
        }
    return {"bins": table, "bin_x_winner_test": test}


def feature_conditioning(states, y, step, meta, run: pathlib.Path):
    feats = collections.defaultdict(dict)
    for g in states:
        m = meta.get(g, {})
        fs = m.get("feature_scalars") or {}
        feats[g]["cag_contrast"] = fs.get("cag_contrast_model_norm")
        feats[g]["il_displacement"] = fs.get("il_displacement_model_norm")
        feats[g]["decision_step"] = step[g]
        feats[g]["goal_progress"] = m.get("goal_predicates_at_state")
        fp = run / "features" / f"{g}.npz"
        if fp.exists():
            z = np.load(fp)
            if "base_samples" in z.files:
                feats[g]["base_sample_spread"] = float(np.mean(np.std(z["base_samples"], axis=0)))
    out = {}
    for f in ("base_sample_spread", "cag_contrast", "il_displacement", "decision_step", "goal_progress"):
        gs = [g for g in states if feats[g].get(f) is not None]
        if len(gs) < 10:
            continue
        x = np.array([feats[g][f] for g in gs], float)
        row = {}
        for a in (*INTERV, "base_success"):
            v = np.array([y[g]["I0"].mean() if a == "base_success" else y[g][a].mean() - y[g]["I0"].mean() for g in gs])
            rho, p = stats.spearmanr(x, v)
            row[a] = {"spearman": float(rho), "p": float(p), "n": len(gs)}
        out[f] = row
    return out


# -------------------------------------------------------------------- bootstrap


def bootstrap(states, y, task, unit, rng):
    units_by_task = collections.defaultdict(list)
    for u in sorted(set(unit.values())):
        units_by_task[u[0]].append(u)
    states_by_unit = collections.defaultdict(list)
    for g in states:
        states_by_unit[unit[g]].append(g)
    keys = ["O_state", "B", "O_task", "gainA", "gainD", "O_risk", "S_I0"]
    acc = {k: [] for k in keys}
    for _ in range(N_BOOT):
        ss, yy, tt, cnt = [], {}, {}, 0
        for us in units_by_task.values():
            for u in (us[i] for i in rng.integers(0, len(us), len(us))):
                for g in states_by_unit[u]:
                    k = f"{g}#{cnt}"
                    cnt += 1
                    ss.append(k)
                    yy[k] = y[g]
                    tt[k] = task[g]
        s_on = {a: float(np.mean([yy[g][a].mean() for g in ss])) for a in ROUTE}
        o_state, _ = loo_state_oracle(ss, yy)
        b = max(s_on.values())
        o_task = loo_task_oracle(ss, yy, tt)
        acc["O_state"].append(o_state)
        acc["B"].append(b)
        acc["O_task"].append(o_task)
        acc["gainA"].append(o_state - b)
        acc["gainD"].append(o_state - o_task)
        acc["O_risk"].append(loo_risk_gate(ss, yy))
        acc["S_I0"].append(s_on["I0"])
    return {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in acc.items()}


def mcnemar_exact(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2**n)


# ------------------------------------------------------------------------- main


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="results/adaptive_inference/pilot")
    ap.add_argument("--n_rep", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    run = pathlib.Path(args.run)
    out = run / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    rows = load(run)
    val = validity(rows)
    states, dropped, y, cost, prog, task, unit, step, meta = build(rows, args.n_rep)
    s = {"script_sha256": hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(), "validity": val}
    s["n_states_complete"] = len(states)
    s["n_states_dropped"] = len(dropped)
    s["states_per_task"] = dict(collections.Counter(task[g] for g in states))
    s_on = always_on(states, y)
    s["always_on"] = s_on
    s["always_on_by_task"] = {
        t: {a: float(np.mean([y[g][a].mean() for g in states if task[g] == t])) for a in ALL_ARMS}
        for t in sorted(set(task.values()))
    }
    cost_ref = {a: float(np.mean([cost[g][a].mean() for g in states])) for a in ALL_ARMS}
    s["mean_server_ms_per_branch"] = cost_ref
    s["mean_goal_progress"] = {a: float(np.mean([prog[g][a].mean() for g in states])) for a in ALL_ARMS}
    b_arm = max(ROUTE, key=lambda a: s_on[a])
    o_state, o_state_cost = loo_state_oracle(states, y, cost=cost)
    o_task = loo_task_oracle(states, y, task)
    o_risk = loo_risk_gate(states, y)
    s["oracle"] = {
        "best_always_on_arm": b_arm,
        "B": s_on[b_arm],
        "O_state_loo": o_state,
        "O_state_loo_mean_cost_ms": o_state_cost,
        "O_task_loo": o_task,
        "O_risk_gate_loo": o_risk,
        "naive_oracle_route": naive_oracle(states, y),
        "naive_oracle_route_plus_placebo": naive_oracle(states, y, (*ROUTE, PLACEBO)),
        "naive_oracle_I0_I0P": naive_oracle(states, y, ("I0", PLACEBO)),
        "O_state_loo_I0_I0P": loo_state_oracle(states, y, arms=("I0", PLACEBO))[0],
        "O_state_loo_route_plus_placebo": loo_state_oracle(states, y, arms=("I0", PLACEBO, "IL", "IC", "IH"))[0],
        "captured_fraction_risk_gate": (o_risk - s_on["I0"]) / (o_state - s_on["I0"])
        if o_state != s_on["I0"]
        else float("nan"),
    }
    s["cost_frontier"] = {
        str(lam): dict(
            zip(
                ("O_state", "mean_cost_ms"),
                loo_state_oracle(states, y, cost=cost, lam=lam, cost_ref=cost_ref),
                strict=True,
            )
        )
        for lam in LAMBDAS
    }
    s["rescue_interventions"] = rescue_stats(states, y, INTERV)
    s["rescue_with_placebo"] = rescue_stats(states, y, (*INTERV, PLACEBO))
    s["complementarity"] = complementarity(states, y)
    s["delta_hist"] = {
        a: dict(collections.Counter(float(y[g][a].mean() - y[g]["I0"].mean()) for g in states))
        for a in (*INTERV, PLACEBO)
    }
    s["frac_states_hurt"] = {
        a: float(np.mean([y[g][a].mean() < y[g]["I0"].mean() for g in states])) for a in (*INTERV, PLACEBO)
    }
    s["frac_states_help"] = {
        a: float(np.mean([y[g][a].mean() > y[g]["I0"].mean() for g in states])) for a in (*INTERV, PLACEBO)
    }
    pairs = {}
    for a in (*INTERV, PLACEBO):
        b = sum(int(((y[g][a] == 1) & (y[g]["I0"] == 0)).sum()) for g in states)
        c = sum(int(((y[g][a] == 0) & (y[g]["I0"] == 1)).sum()) for g in states)
        pairs[a] = {"a_only": b, "I0_only": c, "mcnemar_p_descriptive": mcnemar_exact(b, c)}
    s["paired_vs_I0"] = pairs
    s["risk_utility"] = risk_utility(states, y)
    s["feature_conditioning"] = feature_conditioning(states, y, step, meta, run)
    s["bootstrap95"] = bootstrap(states, y, task, unit, rng)

    # gate
    ci = s["bootstrap95"]
    gain_a = 100 * (o_state - s_on[b_arm])
    pareto = any(
        v["O_state"] >= s_on[b_arm] + GATE["A_pareto_pp"] / 100 and v["mean_cost_ms"] <= cost_ref[b_arm]
        for v in s["cost_frontier"].values()
    )
    gate_a = (gain_a >= GATE["A_pp"] and ci["gainA"][0] > 0) or pareto
    ur, urp = s["rescue_interventions"]["unique_rate"], s["rescue_with_placebo"]["unique_rate"]
    b_arms = [a for a in INTERV if ur[a] >= GATE["B_unique"] and urp[a] > urp[PLACEBO]]
    gate_b = len(b_arms) >= GATE["B_min_arms"]
    share = s["rescue_interventions"]["rescue_share"]
    gate_c = max(share.values()) <= GATE["C_share"]
    gain_d = 100 * (o_state - o_task)
    gate_d = gain_d >= GATE["D_pp"] and ci["gainD"][0] > 0
    nstate = max(1, val["decision_states"])
    gate_e = (
        val["identity_ok"] >= 0.99 * val["identity_branches"]
        and val["restore_ok"] == val["restore_checked"] - val["exceptions"]
        and val["exceptions"] < 0.01 * max(1, val["branches"] + val["identity_branches"])
        and len(states) >= 0.9 * nstate
    )
    s["gate"] = {
        "A": {
            "pass": bool(gate_a),
            "gain_pp": gain_a,
            "ci_pp": [100 * x for x in ci["gainA"]],
            "pareto_route": bool(pareto),
        },
        "B": {"pass": bool(gate_b), "unique_rate": ur, "unique_rate_with_placebo": urp, "arms_passing": b_arms},
        "C": {"pass": bool(gate_c), "max_share": max(share.values()), "share": share},
        "D": {"pass": bool(gate_d), "gain_pp": gain_d, "ci_pp": [100 * x for x in ci["gainD"]]},
        "E": {"pass": bool(gate_e)},
    }
    verdict = "BLOCKED" if not gate_e else ("GO" if all(s["gate"][k]["pass"] for k in "ABCD") else "NO-GO")
    s["verdict"] = verdict

    # intervention matrix (one row per state x replicate)
    with (out / "intervention_matrix.csv").open("w") as f:
        f.write("group_id,task_id,init_state,decision_step,replicate," + ",".join(ALL_ARMS) + "\n")
        for g in states:
            for r in range(args.n_rep):
                f.write(
                    f"{g},{unit[g][0]},{unit[g][1]},{step[g]},{r},"
                    + ",".join(str(int(y[g][a][r])) for a in ALL_ARMS)
                    + "\n"
                )
    (out / "summary.json").write_text(json.dumps(s, indent=1, default=str))
    print(
        json.dumps(
            {k: s[k] for k in ("validity", "n_states_complete", "always_on", "oracle", "gate", "verdict")},
            indent=1,
            default=str,
        )
    )


if __name__ == "__main__":
    main()
