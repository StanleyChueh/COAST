"""Stage 1: zero-GPU horizon-oracle headroom analysis on the Phase-B AAC data.

Uses only existing rollouts (results/aac/phaseB_coast, plus the Phase A-U replicate
arms in results/aac/phaseA_unseeded for the noise calibration). Nothing is trained.

Scenes are matched on (task, episode_seed) and accepted only if every arm involved
logged the same scene_hash (initial MuJoCo state) and model_hash (static geometry).

Every rollout is ONE stochastic draw: GR00T's flow noise uses a running RNG
(aac_seed_per_call=None), so two arms at the same scene differ by horizon AND by
noise. A per-scene "any arm succeeds" oracle therefore also harvests rollout noise.
The replicate section measures that noise directly: Phase A-U re-ran fixed8, fixed16
and AAC on 4 of the 22 tasks (same seeds, same scenes, same server protocol, other
noise), so "oracle over two replicates of the same h" can be compared with "oracle
over two different h" on identical scenes.

Usage (repo root, root venv):
    uv run python research/adaptive_inference/analyze_horizon_oracle.py
"""

from __future__ import annotations

import argparse
import collections
import dataclasses
import hashlib
import itertools
import json
import math
import pathlib
import re
import subprocess

import numpy as np

FIXED = ("fixed2", "fixed4", "fixed8", "fixed12", "fixed16")
H_OF = {a: int(a.removeprefix("fixed")) for a in FIXED}
N_BOOT = 4000

# Pre-specified gate (research brief, Stage 1). Not to be changed after results.
GATE_LOW_PP = 3.0
GATE_HIGH_PP = 5.0


# --------------------------------------------------------------------------- io


def load_arm(run: pathlib.Path, arm: str) -> dict[tuple[str, int], dict]:
    out = {}
    for f in sorted((run / "arms" / arm / "logs").glob("*/episodes.jsonl")):
        for line in f.read_text().splitlines():
            if not line.strip():
                continue
            e = json.loads(line)
            key = (e["task"], int(e["episode_seed"]))
            if key in out:
                raise SystemExit(f"duplicate episode {arm} {key}")
            out[key] = e
    return out


def match(arms: dict[str, dict], names: tuple[str, ...]):
    """Scenes present in every arm in `names` with identical scene/model hashes."""
    keys = set.intersection(*(set(arms[n]) for n in names))
    ok, bad = [], collections.Counter()
    for k in sorted(keys):
        recs = [arms[n][k] for n in names]
        sh = {r["scene_hash"] for r in recs}
        mh = {r.get("model_hash") for r in recs} - {None}
        if len(sh) == 1 and len(mh) <= 1:
            ok.append(k)
        else:
            bad[k[0]] += 1
    return ok, bad


def audit_exceptions(run: pathlib.Path, arm: str) -> dict:
    """Separate aborted infrastructure attempts from completed episodes."""
    arm_dir = run / "arms" / arm
    lines = (arm_dir / "attempts.log").read_text().splitlines() if (arm_dir / "attempts.log").exists() else []
    kinds = collections.Counter()
    gave_up = []
    for ln in lines:
        m = re.match(r"(\S+) (start|resume|GAVE UP) (\S+) at (?:episode )?(\d+)/(\d+)", ln)
        if not m:
            kinds["unparsed"] += 1
            continue
        kinds[m.group(2)] += 1
        if m.group(2) == "GAVE UP":
            gave_up.append((m.group(1)[:19], m.group(3), int(m.group(4))))
    tb_total, oom_total, other = 0, 0, collections.Counter()
    for log in sorted((arm_dir / "task_logs").glob("*.log")):
        text = log.read_text(errors="replace")
        for chunk in text.split("Traceback (most recent call last):")[1:]:
            tb_total += 1
            errs = [ln for ln in chunk.splitlines() if ln and not ln.startswith(" ")]
            msg = next((e for e in errs if "Error" in e), errs[0] if errs else "?")
            if "out of memory" in chunk:
                oom_total += 1
            else:
                other[msg[:120]] += 1
    eps = load_arm(run, arm)
    run_ids = collections.Counter(e["run_id"] for e in eps.values())
    inf_ids = collections.Counter()
    for f in (arm_dir / "logs").glob("*/inference.jsonl"):
        with f.open() as fh:
            for line in fh:
                i = line.find('"run_id": "')
                inf_ids[line[i + 11 : i + 23]] += 1
    return {
        "attempt_lines": dict(kinds),
        "gave_up_tasks": len(gave_up),
        "gave_up_at_episode": sorted({g[2] for g in gave_up}),
        "gave_up_dates": sorted({g[0][:10] for g in gave_up}),
        "traceback_blocks": tb_total,
        "traceback_blocks_cuda_oom": oom_total,
        "traceback_blocks_other": dict(other),
        "completed_episodes": len(eps),
        "episode_run_ids": len(run_ids),
        "inference_run_ids": len(inf_ids),
        "orphan_inference_run_ids": sorted(set(inf_ids) - set(run_ids)),
    }


# ------------------------------------------------------------------- statistics


def mcnemar_exact(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * p)


@dataclasses.dataclass
class Matrix:
    tasks: np.ndarray  # (S,) task names
    seeds: np.ndarray  # (S,)
    y: np.ndarray  # (S, A) 0/1
    arms: tuple[str, ...]

    def by_task(self) -> dict[str, np.ndarray]:
        return {t: self.y[self.tasks == t] for t in np.unique(self.tasks)}


def build_matrix(arms: dict[str, dict], names: tuple[str, ...], keys) -> Matrix:
    y = np.array([[int(arms[n][k]["success"]) for n in names] for k in keys], dtype=np.int8)
    return Matrix(np.array([k[0] for k in keys]), np.array([k[1] for k in keys]), y, names)


def per_task_best_crossfit(m: Matrix) -> float:
    """Choose each task's best arm on even seeds, score it on odd seeds, and vice versa."""
    hits, n = 0, 0
    for t in np.unique(m.tasks):
        sel = m.tasks == t
        y, s = m.y[sel], m.seeds[sel]
        for fit, ev in ((s % 2 == 0, s % 2 == 1), (s % 2 == 1, s % 2 == 0)):
            j = int(np.argmax(y[fit].mean(0)))  # ties -> first arm in order
            hits += int(y[ev, j].sum())
            n += int(ev.sum())
    return hits / n


def headroom_stats(m: Matrix, rng) -> dict:
    y = m.y.astype(float)
    n_s, n_a = y.shape
    rates = y.mean(0)
    best = int(np.argmax(rates))
    oracle = y.max(1)
    bt = m.by_task()
    task_best_insample = np.concatenate([v[:, int(np.argmax(v.mean(0)))] for v in bt.values()]).mean()

    # bootstrap the headroom over the best single arm and over the in-sample per-task best
    tasks = sorted(bt)
    hb, ht, ob = [], [], []
    for _ in range(N_BOOT):
        parts = [bt[t][rng.integers(0, len(bt[t]), len(bt[t]))] for t in tasks]
        allp = np.concatenate(parts)
        o = allp.max(1).mean()
        ob.append(o)
        hb.append(o - allp.mean(0).max())
        ht.append(o - np.concatenate([p[:, int(np.argmax(p.mean(0)))] for p in parts]).mean())
    ci = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]  # noqa: E731

    unique = {m.arms[j]: int(((y[:, j] == 1) & (np.delete(y, j, 1).sum(1) == 0)).sum()) for j in range(n_a)}
    comp = {}
    for i, j in itertools.permutations(range(n_a), 2):
        comp[f"{m.arms[i]}>{m.arms[j]}"] = float(((y[:, i] == 1) & (y[:, j] == 0)).mean())
    k = y.sum(1)
    return {
        "n_scenes": n_s,
        "n_tasks": len(bt),
        "rates": {m.arms[j]: float(rates[j]) for j in range(n_a)},
        "best_single": m.arms[best],
        "best_single_rate": float(rates[best]),
        "per_task_best_insample": float(task_best_insample),
        "per_task_best_crossfit": per_task_best_crossfit(m),
        "scene_oracle": float(oracle.mean()),
        "scene_oracle_ci": ci(ob),
        "headroom_vs_best_single_pp": 100 * float(oracle.mean() - rates[best]),
        "headroom_vs_best_single_ci_pp": [100 * x for x in ci(hb)],
        "headroom_vs_task_best_pp": 100 * float(oracle.mean() - task_best_insample),
        "headroom_vs_task_best_ci_pp": [100 * x for x in ci(ht)],
        "regret_pp": {m.arms[j]: 100 * float(oracle.mean() - rates[j]) for j in range(n_a)},
        "unique_success": unique,
        "unique_success_frac_of_salvageable": {a: v / max(1, int((oracle == 1).sum())) for a, v in unique.items()},
        "k_success_hist": {int(i): int((k == i).sum()) for i in range(n_a + 1)},
        "all_fail": int((k == 0).sum()),
        "all_succeed": int((k == n_a).sum()),
        "mixed": int(((k > 0) & (k < n_a)).sum()),
        "complementarity": comp,
    }


def within_task_heterogeneity(m: Matrix) -> dict:
    """How often the scene-level winner differs from the task-level winner."""
    out = {}
    tot_mixed, tot_taskbest_fails = 0, 0
    for t, y in m.by_task().items():
        j = int(np.argmax(y.mean(0)))
        k = y.sum(1)
        mixed = (k > 0) & (k < y.shape[1])
        fails_but_other = (y[:, j] == 0) & (k > 0)
        patterns = collections.Counter("".join(map(str, r)) for r in y[mixed])
        out[t] = {
            "task_best": m.arms[j],
            "rates": {a: float(v) for a, v in zip(m.arms, y.mean(0), strict=True)},
            "n": len(y),
            "mixed": int(mixed.sum()),
            "task_best_fails_other_succeeds": int(fails_but_other.sum()),
            "distinct_success_patterns_in_mixed": len(patterns),
        }
        tot_mixed += int(mixed.sum())
        tot_taskbest_fails += int(fails_but_other.sum())
    return {
        "per_task": out,
        "mixed_scenes": tot_mixed,
        "scenes_where_task_best_fails_but_another_h_succeeds": tot_taskbest_fails,
        "frac_of_all_scenes": tot_taskbest_fails / len(m.y),
    }


def cochran_q(y: np.ndarray) -> tuple[float, float]:
    """Cochran's Q for k related binary samples (marginal h effect within scenes)."""
    n, k = y.shape
    col = y.sum(0)
    row = y.sum(1)
    tot = row.sum()
    denom = k * tot - (row**2).sum()
    if denom == 0:
        return 0.0, 1.0
    q = (k - 1) * (k * (col**2).sum() - tot**2) / denom
    from scipy.stats import chi2

    p = float(chi2.sf(q, k - 1))
    return float(q), p


# ------------------------------------------------------------------ replicates


def replicate_calibration(b: dict[str, dict], u: dict[str, dict], rng) -> dict:
    """Same-h replicate pairs (Phase B vs Phase A-U) vs different-h pairs on identical scenes."""
    arms = {f"{a}_B": b[a] for a in ("fixed8", "fixed16", "aac")}
    arms.update({f"{a}_U": u[a] for a in ("fixed8", "fixed16", "aac")})
    names = tuple(arms)
    keys, bad = match(arms, names)
    m = build_matrix(arms, names, keys)
    idx = {n: i for i, n in enumerate(names)}
    bt = m.by_task()
    pairs = {
        # same intervention, two noise draws
        "same:fixed16_B|fixed16_U": ("fixed16_B", "fixed16_U"),
        "same:fixed8_B|fixed8_U": ("fixed8_B", "fixed8_U"),
        "same:aac_B|aac_U": ("aac_B", "aac_U"),
        # different intervention, both from phase B (what the 22-task oracle uses)
        "diff:fixed16_B|fixed8_B": ("fixed16_B", "fixed8_B"),
        "diff:fixed16_B|aac_B": ("fixed16_B", "aac_B"),
        "diff:fixed8_B|aac_B": ("fixed8_B", "aac_B"),
        # different intervention, different runs (no shared run-level factor)
        "diff:fixed16_B|fixed8_U": ("fixed16_B", "fixed8_U"),
        "diff:fixed16_U|fixed8_B": ("fixed16_U", "fixed8_B"),
        "diff:fixed16_U|fixed8_U": ("fixed16_U", "fixed8_U"),
    }
    res = {}
    for name, (a1, a2) in pairs.items():
        i, j = idx[a1], idx[a2]

        def two(s, i=i, j=j):
            return np.stack([s[:, i], s[:, j]], 1)

        y2 = two(m.y)
        disc = float((y2[:, 0] != y2[:, 1]).mean())
        orc = float(y2.max(1).mean())
        best = float(y2.mean(0).max())
        boots_o, boots_g = [], []
        for _ in range(N_BOOT):
            s = np.concatenate([two(bt[t])[rng.integers(0, len(bt[t]), len(bt[t]))] for t in sorted(bt)])
            boots_o.append(s.max(1).mean())
            boots_g.append(s.max(1).mean() - s.mean(0).max())
        b_cnt = int(((y2[:, 0] == 1) & (y2[:, 1] == 0)).sum())
        c_cnt = int(((y2[:, 0] == 0) & (y2[:, 1] == 1)).sum())
        res[name] = {
            "rates": [float(y2[:, 0].mean()), float(y2[:, 1].mean())],
            "discordance": disc,
            "a_only": b_cnt,
            "b_only": c_cnt,
            "mcnemar_p": mcnemar_exact(b_cnt, c_cnt),
            "oracle": orc,
            "oracle_ci": [float(np.percentile(boots_o, 2.5)), float(np.percentile(boots_o, 97.5))],
            "oracle_gain_over_better_pp": 100 * (orc - best),
            "oracle_gain_ci_pp": [100 * float(np.percentile(boots_g, 2.5)), 100 * float(np.percentile(boots_g, 97.5))],
        }
    # Paired difference of 2-arm oracle gains: diff-h pair vs same-h pair on the same scenes.
    contrasts = {}
    for diff, same in (
        ("diff:fixed16_B|fixed8_B", "same:fixed16_B|fixed16_U"),
        ("diff:fixed16_B|fixed8_U", "same:fixed16_B|fixed16_U"),
        ("diff:fixed16_U|fixed8_B", "same:fixed16_B|fixed16_U"),
        ("diff:fixed16_B|aac_B", "same:fixed16_B|fixed16_U"),
    ):
        (a1, a2), (s1, s2) = pairs[diff], pairs[same]
        cols = [idx[a1], idx[a2], idx[s1], idx[s2]]

        def g(s, cols=cols):
            d = s[:, cols[:2]]
            r = s[:, cols[2:]]
            return (d.max(1).mean() - d.mean(0).max()) - (r.max(1).mean() - r.mean(0).max())

        boots = [
            g(np.concatenate([bt[t][rng.integers(0, len(bt[t]), len(bt[t]))] for t in sorted(bt)]))
            for _ in range(N_BOOT)
        ]
        contrasts[f"{diff} minus {same}"] = {
            "pp": 100 * float(g(m.y)),
            "ci_pp": [100 * float(np.percentile(boots, 2.5)), 100 * float(np.percentile(boots, 97.5))],
        }
    return {
        "tasks": sorted(bt),
        "n_scenes": len(keys),
        "hash_mismatch_by_task": dict(bad),
        "pairs": res,
        "oracle_gain_contrasts": contrasts,
    }


def noise_adjusted_oracle(m: Matrix, flip_same: float) -> dict:
    """Simple sensitivity bound: how large would a 5-way oracle be from noise alone?

    Model: each scene has a latent success probability p_s shared by all horizons
    (no scene x horizon interaction), and arms are 5 independent draws. With two
    replicates, P(disagree) = 2 E[p(1-p)] = flip_same. Under a Beta(a, b) for p_s
    with the observed mean, matching E[p(1-p)] fixes the dispersion. Oracle(5) under
    this null is 1 - E[(1-p)^5]. Pure illustration of scale, reported as such.
    """
    mean = float(m.y.mean())
    target = flip_same / 2.0  # E[p(1-p)]
    # E[p(1-p)] = mean(1-mean) * (a+b)/(a+b+1) for Beta(a,b) -> solve for kappa=a+b
    base = mean * (1 - mean)
    if target >= base or target <= 0:
        return {"note": "flip rate outside Beta-null range", "mean": mean, "flip_same": flip_same}
    r = target / base  # = kappa/(kappa+1)
    kappa = r / (1 - r)
    a, b = mean * kappa, (1 - mean) * kappa
    # E[(1-p)^k] for Beta(a,b) = B(a, b+k)/B(a,b) = prod_{i=0}^{k-1} (b+i)/(a+b+i)
    out = {"mean_success": mean, "flip_same": flip_same, "beta_a": a, "beta_b": b}
    for k in (1, 2, 5, 6):
        e = 1.0
        for i in range(k):
            e *= (b + i) / (a + b + i)
        out[f"null_oracle_{k}"] = 1 - e
    return out


# ------------------------------------------------------------------------ main


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase_b", default="results/aac/phaseB_coast")
    ap.add_argument("--replicate", default="results/aac/phaseA_unseeded")
    ap.add_argument("--out", default="research/adaptive_inference/results/horizon_oracle")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    run_b, run_u, out = pathlib.Path(args.phase_b), pathlib.Path(args.replicate), pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    b = {a: load_arm(run_b, a) for a in (*FIXED, "aac")}
    u = {a: load_arm(run_u, a) for a in ("fixed8", "fixed16", "aac")}

    summary: dict = {
        "git_sha": git_sha(),
        "script_sha256": hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),
        "inputs": {"phase_b": str(run_b), "replicate": str(run_u)},
        "gate": {"low_pp": GATE_LOW_PP, "high_pp": GATE_HIGH_PP},
    }

    # 0. exception audit
    summary["exception_audit"] = {a: audit_exceptions(run_b, a) for a in (*FIXED, "aac")}

    # 1. raw (all logged) per-arm rates, to check the reported aggregate numbers
    summary["raw_rates"] = {
        a: {"n": len(v), "successes": int(sum(e["success"] for e in v.values()))} for a, v in b.items()
    }

    # 2. matched 5-way fixed-h set
    keys5, bad5 = match(b, FIXED)
    m5 = build_matrix(b, FIXED, keys5)
    summary["matched_fixed"] = {"n_scenes": len(keys5), "hash_mismatch_by_task": dict(bad5)}
    summary["fixed_oracle"] = headroom_stats(m5, rng)
    summary["fixed_heterogeneity"] = within_task_heterogeneity(m5)
    q, p = cochran_q(m5.y.astype(float))
    summary["fixed_cochran_q"] = {"Q": q, "p": p}

    # 3. matched 6-way (fixed + AAC)
    keys6, bad6 = match(b, (*FIXED, "aac"))
    m6 = build_matrix(b, (*FIXED, "aac"), keys6)
    summary["fixed_plus_aac_oracle"] = headroom_stats(m6, rng)

    # 4. replicate-based noise calibration
    rep = replicate_calibration(b, u, rng)
    summary["replicate"] = rep
    flip16 = rep["pairs"]["same:fixed16_B|fixed16_U"]["discordance"]
    flip8 = rep["pairs"]["same:fixed8_B|fixed8_U"]["discordance"]
    summary["noise_null_5way"] = noise_adjusted_oracle(m5, (flip16 + flip8) / 2)

    # 5. gate (raw, as specified) + calibrated reading
    hd = summary["fixed_oracle"]["headroom_vs_best_single_pp"]
    raw_gate = "LOW" if hd < GATE_LOW_PP else ("HIGH" if hd >= GATE_HIGH_PP else "BORDERLINE")
    summary["gate_raw"] = raw_gate

    # matrices for the report
    with (out / "fixed_matrix.csv").open("w") as f:
        f.write("task,episode_seed," + ",".join(FIXED) + "\n")
        for t, s, row in zip(m5.tasks, m5.seeds, m5.y, strict=True):
            f.write(f"{t},{s}," + ",".join(map(str, row)) + "\n")
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: summary[k] for k in ("raw_rates", "matched_fixed", "gate_raw")}, indent=1))
    fo = summary["fixed_oracle"]
    print(
        f"best single {fo['best_single']} {fo['best_single_rate']:.3f}; scene oracle {fo['scene_oracle']:.3f};"
        f" headroom {fo['headroom_vs_best_single_pp']:.1f} pp {fo['headroom_vs_best_single_ci_pp']}"
    )
    print(json.dumps(rep["pairs"], indent=1))
    print(json.dumps(rep["oracle_gain_contrasts"], indent=1))
    print(json.dumps(summary["noise_null_5way"], indent=1))


if __name__ == "__main__":
    main()
