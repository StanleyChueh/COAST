"""Analyze an AAC run directory produced by experiments/aac/run_arms.sh.

    uv run python experiments/aac/analyze_aac.py results/aac/<run> [--reference fixed16]

Reads   <run>/arms/<arm>/logs/<task>/{episodes,inference}.jsonl
Writes  <run>/episodes.jsonl, <run>/inference.jsonl   (merged, with "arm")
        <run>/summary.json, <run>/report.md, <run>/plots/*.png

Statistics: Wilson 95% CIs for success rates; paired comparisons on
(task, episode) pairs whose initial-scene hashes match, with an exact McNemar
test on discordant pairs and a task-stratified bootstrap CI of the difference.
"""

from __future__ import annotations

import argparse
import collections
import itertools
import json
import math
import pathlib

import numpy as np

# Reference categorical palette (dataviz skill, light mode), fixed order.
C_BLUE, C_ORANGE, C_AQUA, C_YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, MUTED, GRID = "#1a1a19", "#6b6a63", "#e4e3dc"


def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, c - h), min(1.0, c + h))


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value on discordant counts b, c."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * p)


def _arm_sort_key(arm: str):
    if arm.startswith("fixed"):
        try:
            return (0, int(arm[5:]), arm)
        except ValueError:
            return (0, 99, arm)
    return (1, 0, arm)


def load(run: pathlib.Path):
    episodes, inference = [], []
    for arm_dir in sorted((run / "arms").iterdir()):
        if not arm_dir.is_dir():
            continue
        arm = arm_dir.name
        for f in sorted((arm_dir / "logs").glob("*/episodes.jsonl")):
            episodes.extend(dict(json.loads(line), arm=arm) for line in f.read_text().splitlines() if line.strip())
        for f in sorted((arm_dir / "logs").glob("*/inference.jsonl")):
            inference.extend(dict(json.loads(line), arm=arm) for line in f.read_text().splitlines() if line.strip())
    # Drop calls from episodes that never finished (client crash / OOM kill): with a
    # run_id they are identified exactly; legacy records without one are kept only if
    # their episode index was logged.
    logged = {(e["arm"], e["task"], e["episode"], e.get("run_id")) for e in episodes}
    logged_legacy = {(e["arm"], e["task"], e["episode"]) for e in episodes}
    inference = [
        r
        for r in inference
        if (r["arm"], r["task"], r["episode"], r.get("run_id")) in logged
        or (r.get("run_id") is None and (r["arm"], r["task"], r["episode"]) in logged_legacy)
    ]
    return episodes, inference


def eval_all_status(run: pathlib.Path) -> dict:
    """Simulator exceptions: Python tracebacks in any client log of the arm, plus
    resume attempts recorded by run_arms.sh."""
    out = {}
    for arm_dir in sorted((run / "arms").iterdir()):
        if not arm_dir.is_dir():
            continue
        tracebacks = []
        for log in [*arm_dir.glob("eval/parallel_logs/*.log"), *arm_dir.glob("task_logs/*.log")]:
            text = log.read_text(errors="replace")
            for chunk in text.split("Traceback (most recent call last):")[1:]:
                err = [ln for ln in chunk.splitlines() if ln and not ln.startswith(" ")]
                tracebacks.append(f"{log.name}: {err[0] if err else '?'}")
        resumes = 0
        if (arm_dir / "attempts.log").exists():
            resumes = sum("resume" in ln for ln in (arm_dir / "attempts.log").read_text().splitlines())
        out[arm_dir.name] = {"nan_tasks": [], "tracebacks": tracebacks, "resumes": resumes}
    return out


def summarize(episodes, inference, run, reference):
    arms = sorted({e["arm"] for e in episodes}, key=_arm_sort_key)
    tasks = sorted({e["task"] for e in episodes})
    inf_by_arm = collections.defaultdict(list)
    for r in inference:
        inf_by_arm[r["arm"]].append(r)
    status = {}
    for r in run if isinstance(run, list) else [run]:
        for arm, st in eval_all_status(r).items():
            agg = status.setdefault(arm, {"nan_tasks": [], "tracebacks": [], "resumes": 0})
            agg["tracebacks"] += st["tracebacks"]
            agg["resumes"] += st["resumes"]

    summary = {"arms": {}, "per_task": {}, "paired": {}, "tasks": tasks}
    for arm in arms:
        eps = [e for e in episodes if e["arm"] == arm]
        k = sum(e["success"] for e in eps)
        n = len(eps)
        infs = inf_by_arm[arm]
        infer_ms = [r["server_timing"].get("infer_ms") for r in infs if r.get("server_timing")]
        sample_ms = [r["aac"]["sample_ms"] for r in infs if "aac" in r and "sample_ms" in r["aac"]]
        select_ms = [r["aac"]["select_ms"] for r in infs if "aac" in r and "select_ms" in r["aac"]]
        peak = [r["aac"].get("peak_mem_mb") for r in infs if "aac" in r and r["aac"].get("peak_mem_mb")]
        exec_h = [r["executed_h"] for r in infs]
        total_infer_ms = float(np.nansum([x for x in infer_ms if x is not None]))
        summary["arms"][arm] = {
            "episodes": n,
            "successes": int(k),
            "success_rate": k / n if n else float("nan"),
            "wilson95": wilson(int(k), n),
            "mean_executed_h": float(np.mean(exec_h)) if exec_h else None,
            "policy_calls_per_episode": float(np.mean([e["num_policy_calls"] for e in eps])) if eps else None,
            "env_steps_per_episode": float(np.mean([e["env_steps"] for e in eps])) if eps else None,
            "infer_ms_mean": float(np.mean(infer_ms)) if infer_ms else None,
            "infer_ms_p50": float(np.median(infer_ms)) if infer_ms else None,
            "sample_ms_mean": float(np.mean(sample_ms)) if sample_ms else None,
            "select_ms_mean": float(np.mean(select_ms)) if select_ms else None,
            "peak_vram_mb_max": float(np.max(peak)) if peak else None,
            "policy_compute_s_per_success": (total_infer_ms / 1000.0 / k) if k else None,
            "policy_compute_s_per_episode": total_infer_ms / 1000.0 / n if n else None,
            "num_inference_calls": len(infs),
            "simulator_exceptions": len(status.get(arm, {}).get("tracebacks", [])),
            "simulator_exception_messages": status.get(arm, {}).get("tracebacks", []),
            "resumed_task_runs": status.get(arm, {}).get("resumes", 0),
        }
        for t in tasks:
            te = [e for e in eps if e["task"] == t]
            tk = sum(e["success"] for e in te)
            summary["per_task"].setdefault(t, {})[arm] = {
                "episodes": len(te),
                "successes": int(tk),
                "success_rate": tk / len(te) if te else float("nan"),
                "wilson95": wilson(int(tk), len(te)),
                "mean_executed_h": float(np.mean([h for e in te for h in e["executed_horizons"]])) if te else None,
            }

    # Paired comparisons.
    # Pair on the scene seed when available (lets several runs with different
    # seed ranges be merged); fall back to the episode index.
    def _ep_id(e):
        return e["episode_seed"] if e.get("episode_seed") is not None else e["episode"]

    by_key = {}
    for e in episodes:
        key = (e["arm"], e["task"], _ep_id(e))
        if key in by_key:
            raise SystemExit(f"duplicate episode {key}: overlapping runs?")
        by_key[key] = e
    rng = np.random.default_rng(0)
    comparisons = []
    if "aac" in arms:
        comparisons += [("aac", a) for a in arms if a != "aac"]
    for a, b in comparisons:
        pairs, hash_mismatch = [], 0
        for (arm, t, ep), e in by_key.items():
            if arm != a or (b, t, ep) not in by_key:
                continue
            f = by_key[(b, t, ep)]
            model_differs = e.get("model_hash") and f.get("model_hash") and e["model_hash"] != f["model_hash"]
            if e.get("scene_hash") != f.get("scene_hash") or model_differs:
                hash_mismatch += 1
                continue
            pairs.append((t, int(e["success"]), int(f["success"])))
        if not pairs:
            continue
        b_cnt = sum(1 for _, x, y in pairs if x == 1 and y == 0)
        c_cnt = sum(1 for _, x, y in pairs if x == 0 and y == 1)
        diffs = np.array([x - y for _, x, y in pairs], float)
        by_task = collections.defaultdict(list)
        for t, x, y in pairs:
            by_task[t].append(x - y)
        boots = []
        for _ in range(4000):
            s = []
            for vals in by_task.values():
                v = np.asarray(vals, float)
                s.append(v[rng.integers(0, len(v), len(v))])
            boots.append(np.concatenate(s).mean())
        summary["paired"][f"{a}_vs_{b}"] = {
            "n_pairs": len(pairs),
            "scene_hash_mismatches": hash_mismatch,
            f"{a}_only_success": b_cnt,
            f"{b}_only_success": c_cnt,
            "mean_diff": float(diffs.mean()),
            "bootstrap95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
            "mcnemar_exact_p": mcnemar_exact(b_cnt, c_cnt),
        }

    summary["mechanism"] = mechanism(inference)
    summary["reference_arm"] = reference
    return summary


def mechanism(inference):
    """AAC decision statistics, from every call that carried AAC diagnostics."""
    out = {}
    for arm in sorted({r["arm"] for r in inference}, key=_arm_sort_key):
        recs = [r for r in inference if r["arm"] == arm and "aac" in r and "H_total" in r["aac"]]
        if not recs:
            continue
        hs = np.array([r["aac"]["h_star"] for r in recs])
        he = np.array([r["aac"]["h_entropy"] for r in recs])
        hm = np.array([r["aac"]["h_min_motion"] for r in recs])
        tot = np.array([r["aac"]["H_total"] for r in recs], float)
        comps = {k: np.array([r["aac"][k] for r in recs], float) for k in ("H_pos", "H_rot", "H_grip")}
        spread = np.array([r["aac"]["candidate_spread"] for r in recs], float)
        nonfinite = int(
            sum(
                (~np.isfinite(np.array(r["aac"][k], float))).sum()
                for r in recs
                for k in ("H_pos", "H_rot", "H_grip", "H_total", "chunk_mean")
            )
        )
        by_task = collections.defaultdict(list)
        for r in recs:
            by_task[r["task"]].append(r["aac"]["h_star"])
        out[arm] = {
            "num_calls": len(recs),
            "h_star_hist": {int(h): int((hs == h).sum()) for h in range(1, 17)},
            "h_star_distinct": len(np.unique(hs)),
            "h_star_mean": float(hs.mean()),
            "h_star_by_task": {t: float(np.mean(v)) for t, v in sorted(by_task.items())},
            "h_entropy_hist": {int(h): int((he == h).sum()) for h in range(1, 17)},
            "frac_min_motion_binding": float(np.mean(hm > he)),
            "frac_min_motion_eq_16": float(np.mean(hm == 16)),
            "frac_h_eq_16": float(np.mean(hs == 16)),
            "mean_H_total_by_index": tot.mean(0).tolist(),
            "mean_H_pos_by_index": comps["H_pos"].mean(0).tolist(),
            "mean_H_rot_by_index": comps["H_rot"].mean(0).tolist(),
            "mean_H_grip_by_index": comps["H_grip"].mean(0).tolist(),
            "H_total_last_minus_first_mean": float((tot[:, -1] - tot[:, 0]).mean()),
            "frac_calls_H_total_increasing_overall": float(np.mean(tot[:, -1] > tot[:, 0])),
            "mean_candidate_spread_by_index": spread.mean(0).tolist(),
            "nonfinite_entropy_values": nonfinite,
            "num_samples": sorted({r["aac"]["num_samples"] for r in recs}),
        }
    return out


def _style(ax, title, xlabel, ylabel):
    ax.set_title(title, loc="left", fontsize=11, color=INK)
    ax.set_xlabel(xlabel, color=MUTED)
    ax.set_ylabel(ylabel, color=MUTED)
    ax.grid(visible=True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED)


def plots(summary, inference, episodes, out: pathlib.Path):
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt

    out.mkdir(parents=True, exist_ok=True)
    mech = summary["mechanism"]
    aac_arm = "aac" if "aac" in mech else (next(iter(mech)) if mech else None)
    if aac_arm:
        m = mech[aac_arm]
        # 1. histogram of selected horizons
        fig, ax = plt.subplots(figsize=(6.4, 3.6))
        xs = list(range(1, 17))
        ax.bar(xs, [m["h_star_hist"][h] for h in xs], color=C_BLUE, width=0.8)
        _style(ax, f"Selected horizon h* ({aac_arm}, {m['num_calls']} calls)", "h*", "calls")
        ax.set_xticks(xs)
        fig.tight_layout()
        fig.savefig(out / "h_star_hist.png", dpi=150)
        plt.close(fig)

        # 2. mean selected horizon by task
        bt = m["h_star_by_task"]
        fig, ax = plt.subplots(figsize=(6.4, 0.45 * len(bt) + 1.2))
        names = list(bt)
        ax.barh(names, [bt[t] for t in names], color=C_BLUE, height=0.6)
        for i, t in enumerate(names):
            ax.text(bt[t] + 0.15, i, f"{bt[t]:.1f}", va="center", fontsize=8, color=INK)
        _style(ax, "Mean h* by task", "mean h*", "")
        ax.set_xlim(0, 16.5)
        fig.tight_layout()
        fig.savefig(out / "h_star_by_task.png", dpi=150)
        plt.close(fig)

        # 3. entropy vs future action index
        fig, ax = plt.subplots(figsize=(6.4, 3.8))
        idx = np.arange(1, 17)
        for key, label, col in (
            ("mean_H_pos_by_index", "translation E_t", C_BLUE),
            ("mean_H_rot_by_index", "rotation E_r", C_ORANGE),
            ("mean_H_grip_by_index", "gripper E_g", C_AQUA),
            ("mean_H_total_by_index", "total", C_YELLOW),
        ):
            ax.plot(idx, m[key], color=col, linewidth=2, marker="o", markersize=4, label=label)
        _style(ax, "Mean action entropy by future step (nats)", "future action index i", "entropy (nats)")
        ax.legend(frameon=False, fontsize=8, loc="best")
        ax.set_xticks(idx)
        fig.tight_layout()
        fig.savefig(out / "entropy_vs_index.png", dpi=150)
        plt.close(fig)

        # 4. selected horizon over episode progress
        recs = [r for r in inference if r["arm"] == aac_arm and "aac_horizon" in r]
        ep_len = {(e["task"], e["episode"]): e["env_steps"] for e in episodes if e["arm"] == aac_arm}
        prog = np.array([r["step"] / max(1, ep_len.get((r["task"], r["episode"]), 1)) for r in recs])
        hh = np.array([r["aac_horizon"] for r in recs])
        if len(prog):
            bins = np.linspace(0, 1, 11)
            centers = (bins[:-1] + bins[1:]) / 2
            means = [
                hh[(prog >= lo) & (prog < hi)].mean() if ((prog >= lo) & (prog < hi)).any() else np.nan
                for lo, hi in itertools.pairwise(bins)
            ]
            fig, ax = plt.subplots(figsize=(6.4, 3.6))
            jitter = np.random.default_rng(0).uniform(-0.2, 0.2, len(hh))
            ax.scatter(prog, hh + jitter, s=8, color=C_BLUE, alpha=0.15, linewidths=0)
            ax.plot(centers, means, color=C_ORANGE, linewidth=2, marker="o", markersize=5, label="decile mean")
            _style(ax, "h* over episode progress", "fraction of episode elapsed", "h*")
            ax.legend(frameon=False, fontsize=8)
            fig.tight_layout()
            fig.savefig(out / "h_star_vs_progress.png", dpi=150)
            plt.close(fig)

        # 5. candidate disagreement over future horizon
        fig, ax = plt.subplots(figsize=(6.4, 3.6))
        ax.plot(idx, m["mean_candidate_spread_by_index"], color=C_BLUE, linewidth=2, marker="o", markersize=4)
        _style(
            ax,
            "Candidate disagreement by future step",
            "future action index i",
            "mean std across candidates\n(normalized dims 0-6)",
        )
        fig.tight_layout()
        fig.savefig(out / "candidate_spread_vs_index.png", dpi=150)
        plt.close(fig)

    # 6. success vs fixed h, AAC as a reference band
    arms = summary["arms"]
    fixed = sorted([a for a in arms if a.startswith("fixed")], key=_arm_sort_key)
    if fixed:
        fig, ax = plt.subplots(figsize=(6.4, 3.8))
        hs = [int(a[5:]) for a in fixed]
        rates = [arms[a]["success_rate"] * 100 for a in fixed]
        lo = [arms[a]["wilson95"][0] * 100 for a in fixed]
        hi = [arms[a]["wilson95"][1] * 100 for a in fixed]
        ax.errorbar(
            hs,
            rates,
            yerr=[np.subtract(rates, lo), np.subtract(hi, rates)],
            color=C_BLUE,
            linewidth=2,
            marker="o",
            markersize=6,
            capsize=3,
            label="fixed h (Wilson 95%)",
        )
        if "aac" in arms:
            a = arms["aac"]
            ax.axhspan(a["wilson95"][0] * 100, a["wilson95"][1] * 100, color=C_ORANGE, alpha=0.12, linewidth=0)
            ax.axhline(
                a["success_rate"] * 100, color=C_ORANGE, linewidth=2, label=f"AAC (mean h {a['mean_executed_h']:.1f})"
            )
        _style(ax, "Success rate vs execution horizon", "fixed execution horizon h", "success (%)")
        ax.set_xticks(hs)
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        fig.savefig(out / "success_vs_h.png", dpi=150)
        plt.close(fig)


def report(summary) -> str:
    lines = ["# AAC run report", ""]
    arms = summary["arms"]
    lines += [
        "| Method | Success | Wilson 95% CI | Mean h | Policy calls / ep | Latency / query (ms, mean) | "
        "Policy compute / success (s) | Peak VRAM (MB) | Sim exceptions (resumes) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for a, s in arms.items():
        name = "AAC" if a == "aac" else a.replace("fixed", "Fixed-")
        lines.append(
            f"| {name} | {s['successes']}/{s['episodes']} ({100 * s['success_rate']:.1f}%) | "
            f"[{100 * s['wilson95'][0]:.1f}, {100 * s['wilson95'][1]:.1f}] | "
            f"{s['mean_executed_h']:.2f} | {s['policy_calls_per_episode']:.1f} | "
            f"{(s['infer_ms_mean'] or float('nan')):.0f} | "
            f"{(s['policy_compute_s_per_success'] or float('nan')):.1f} | "
            f"{(s['peak_vram_mb_max'] or float('nan')):.0f} | "
            f"{s['simulator_exceptions']} ({s['resumed_task_runs']}) |"
        )
    lines += ["", "## Per task (successes / episodes)", ""]
    an = list(arms)
    lines.append("| Task | " + " | ".join(an) + " |")
    lines.append("|---|" + "---:|" * len(an))
    for t, row in summary["per_task"].items():
        lines.append(
            f"| {t} | "
            + " | ".join(f"{row[a]['successes']}/{row[a]['episodes']}" if a in row else "-" for a in an)
            + " |"
        )
    if summary["paired"]:
        lines += [
            "",
            "## Paired comparisons (same task, episode, initial scene)",
            "",
            "| Comparison | Pairs | A-only | B-only | Mean diff | Bootstrap 95% | McNemar p | Hash mismatches |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for k, p in summary["paired"].items():
            a, b = k.split("_vs_")
            lines.append(
                f"| {a} vs {b} | {p['n_pairs']} | {p[f'{a}_only_success']} | {p[f'{b}_only_success']} | "
                f"{100 * p['mean_diff']:+.1f} pp | [{100 * p['bootstrap95'][0]:+.1f}, "
                f"{100 * p['bootstrap95'][1]:+.1f}] | {p['mcnemar_exact_p']:.3f} | "
                f"{p['scene_hash_mismatches']} |"
            )
    if summary["mechanism"]:
        lines += [
            "",
            "## AAC decisions (every call that carried AAC diagnostics)",
            "",
            "| Arm | Calls | distinct h* | mean h* | h*=16 | min-motion binding | non-finite | h* histogram (2..16) |",
            "|---|---:|---:|---:|---:|---:|---:|---|",
        ]
        for a, m in summary["mechanism"].items():
            hist = " ".join(str(m["h_star_hist"][h]) for h in range(2, 17))
            lines.append(
                f"| {a} | {m['num_calls']} | {m['h_star_distinct']} | {m['h_star_mean']:.2f} | "
                f"{100 * m['frac_h_eq_16']:.0f}% | {100 * m['frac_min_motion_binding']:.0f}% | "
                f"{m['nonfinite_entropy_values']} | {hist} |"
            )
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=pathlib.Path, help="output dir (and first run dir)")
    ap.add_argument("--also", type=pathlib.Path, nargs="*", default=[], help="extra run dirs to merge")
    ap.add_argument("--reference", default="fixed16")
    args = ap.parse_args()
    episodes, inference = [], []
    for run in [args.run, *args.also]:
        if not (run / "arms").exists():
            continue
        e, i = load(run)
        episodes += e
        inference += i
    if not episodes:
        raise SystemExit(f"no episodes under {args.run}/arms/*/logs")
    args.run.mkdir(parents=True, exist_ok=True)
    with open(args.run / "episodes.jsonl", "w") as f:
        for e in episodes:
            f.write(json.dumps(e) + "\n")
    with open(args.run / "inference.jsonl", "w") as f:
        for r in inference:
            f.write(json.dumps(r) + "\n")
    runs = [r for r in [args.run, *args.also] if (r / "arms").exists()]
    summary = summarize(episodes, inference, runs, args.reference)
    summary["runs"] = [str(r) for r in runs]
    (args.run / "summary.json").write_text(json.dumps(summary, indent=2))
    (args.run / "report.md").write_text(report(summary))
    plots(summary, inference, episodes, args.run / "plots")
    print(report(summary))


if __name__ == "__main__":
    main()
