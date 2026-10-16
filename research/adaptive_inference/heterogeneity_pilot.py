"""Post-hoc variance-components check of state-level effect heterogeneity (pilot data).

For a contrast c between two arms (paired by replicate under CRN), per state s:
    d_{s,r} = Y_{a,r}(s) - Y_{b,r}(s),  est_s = mean_r d,  se2_s = var_r(d) / R
True between-state variance tau^2 = Var_s(est_s) - mean_s(se2_s) (method of moments),
computed overall and after removing task means (within-task heterogeneity).
The placebo contrast I0P - I0 has true tau^2 = 0 and calibrates the estimator.

    uv run python research/adaptive_inference/heterogeneity_pilot.py
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import analyze_pilot as ap

CONTRASTS = (
    ("I0P", "I0"),
    ("IL", "I0"),
    ("IC", "I0"),
    ("IH", "I0"),
    ("IL", "IC"),
    ("IL", "IH"),
    ("IC", "IH"),
)


def tau2(states, y, task, a, b, rng, n_boot=2000):
    def stat(ss):
        d = np.stack([y[g][a] - y[g][b] for g in ss]).astype(float)  # (S, R)
        est = d.mean(1)
        se2 = d.var(1, ddof=1) / d.shape[1]
        t = np.array([task[g] for g in ss])
        within = est.copy()
        for k in np.unique(t):
            within[t == k] -= est[t == k].mean()
        n_t = len(np.unique(t))
        dof_corr = (len(ss) - n_t) / max(1, len(ss) - 1)  # removing task means removes dof
        return (
            est.var(ddof=1) - se2.mean(),
            within.var(ddof=1) - se2.mean() * dof_corr * (len(ss) - 1) / max(1, len(ss) - n_t),
        )

    point = stat(states)
    units = sorted({(task[g], g.split("_")[1]) for g in states})
    by_unit = {u: [g for g in states if (task[g], g.split("_")[1]) == u] for u in units}
    boots = []
    for _ in range(n_boot):
        pick = [units[i] for i in rng.integers(0, len(units), len(units))]
        boots.append(stat([g for u in pick for g in by_unit[u]]))
    boots = np.array(boots)
    return {
        "tau2": point[0],
        "tau": float(np.sqrt(max(point[0], 0))),
        "tau2_ci": np.percentile(boots[:, 0], [2.5, 97.5]).tolist(),
        "tau2_within_task": point[1],
        "tau_within_task": float(np.sqrt(max(point[1], 0))),
        "tau2_within_ci": np.percentile(boots[:, 1], [2.5, 97.5]).tolist(),
    }


def main() -> None:
    run = pathlib.Path("results/adaptive_inference/pilot")
    states, _, y, _, _, task, _, _, _ = ap.build(ap.load(run), 4)
    rng = np.random.default_rng(0)
    res = {f"{a}-{b}": tau2(states, y, task, a, b, rng) for a, b in CONTRASTS}
    (run / "analysis" / "heterogeneity.json").write_text(json.dumps(res, indent=1))
    for k, v in res.items():
        print(
            f"{k:8s} tau={v['tau']:.3f} tau2={v['tau2']:+.4f} CI[{v['tau2_ci'][0]:+.4f},{v['tau2_ci'][1]:+.4f}]"
            f"  within-task tau={v['tau_within_task']:.3f} tau2={v['tau2_within_task']:+.4f}"
            f" CI[{v['tau2_within_ci'][0]:+.4f},{v['tau2_within_ci'][1]:+.4f}]"
        )


def implied_oracle_gain(states, y, task, n_mc=200_000, seed=0):
    """Gaussian method-of-moments model of true per-state contrasts D_k = p_k - p_IL
    (k in I0, IC, IH): mean mu (per task) and between-state covariance Sigma (pooled,
    within-task) = cov(estimated contrasts) - mean sampling covariance. Returns the
    implied gain of a perfect-knowledge state router over always-IL, and over the
    per-task best arm. Rough: ignores the [0,1] bounds of probabilities."""
    rng = np.random.default_rng(seed)
    arms = ("I0", "IC", "IH")
    d = np.stack([np.stack([y[g][k] - y[g]["IL"] for k in arms], 0) for g in states]).astype(float)  # S,K,R
    r = d.shape[2]
    est = d.mean(2)
    t = np.array([task[g] for g in states])
    resid = est.copy()
    mu_t = {}
    for k in np.unique(t):
        mu_t[k] = est[t == k].mean(0)
        resid[t == k] -= mu_t[k]
    samp = np.mean([np.cov(d[s]) / r for s in range(len(states))], 0)
    sigma = np.cov(resid.T) - samp
    w, v = np.linalg.eigh(sigma)
    sigma_psd = (v * np.clip(w, 0, None)) @ v.T
    gains_il, gains_task = [], []
    for k in np.unique(t):
        n_k = int((t == k).sum())
        z = rng.multivariate_normal(mu_t[k], sigma_psd, size=n_mc)  # true contrasts vs IL
        best_task = max(0.0, float(mu_t[k].max()))  # per-task best arm relative to IL
        oracle = np.maximum(0.0, z.max(1)).mean()
        gains_il.append((n_k, oracle))
        gains_task.append((n_k, oracle - best_task))
    n = sum(a for a, _ in gains_il)
    return {
        "sigma_eigenvalues": w.tolist(),
        "implied_oracle_gain_over_always_IL_pp": 100 * sum(a * b for a, b in gains_il) / n,
        "implied_oracle_gain_over_task_best_pp": 100 * sum(a * b for a, b in gains_task) / n,
    }


if __name__ == "__main__":
    main()
    _states, _, _y, _, _, _task, _, _, _ = ap.build(ap.load(pathlib.Path("results/adaptive_inference/pilot")), 4)
    implied = implied_oracle_gain(_states, _y, _task)
    print(json.dumps(implied, indent=1))
    p = pathlib.Path("results/adaptive_inference/pilot/analysis/heterogeneity.json")
    p.write_text(json.dumps({**json.loads(p.read_text()), "implied_oracle": implied}, indent=1))
