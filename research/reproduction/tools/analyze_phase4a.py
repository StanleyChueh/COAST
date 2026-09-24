"""Phase 4A analysis: COAST reproduction sanity study (analysis only; no rollouts, no steering code).

Part 1: libero_10 task 2, 7 conditions x 2 state sets, from
  examples/libero_env/output/phase4a_<cond>_task02_<set>/     (set P = states 45-49 + 0-14, set S = states 15-44)
Part 2: pilot on one harder task, from
  examples/libero_env/output/phase4a_pilot_<cond>_task<TT>_pilot/

For every run the script
  1. checks the client exited cleanly and every episode is present,
  2. recomputes every noise fingerprint from its key and checks pairing across all conditions of the set,
  3. validates the steering diagnostics (10 records per call, layer 5, right strategy / beta, finite),
  4. computes success, transitions, beta sensitivity, COAST-vs-shrinkage, intervention statistics,
  5. writes the long-format per-episode CSV and prints a JSON summary.

Descriptive only. Usage (repo root, CPU only):
    CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu uv run --no-sync python research/reproduction/tools/analyze_phase4a.py \
        [--pilot-task 3] [--json-out PATH]
"""

from __future__ import annotations

import argparse
import collections
import csv
import itertools
import json
import math
import pathlib
import re
import sys

import imageio.v2 as iio
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from analyze_phase1d import EXP
from analyze_phase1d import OUT
from analyze_phase1d import mcnemar_exact
from analyze_phase1d import wilson
from analyze_phase2a import FIELDS

from openpi.serving.noise_control import derive_noise
from openpi.serving.noise_control import noise_fingerprint

MASTER_SEED = 100
NUM_STEPS_WAIT = 10
MAX_STEPS = 520
NUM_INIT_STATES = 50
# name -> (strategy, beta); baseline has no steering payload
COND = {
    "baseline": (None, None),
    "coast_b01": ("global", 0.1),
    "coast_b02": ("global", 0.2),
    "coast_b03": ("global", 0.3),
    "shrink_b01": ("shrinkage", 0.1),
    "shrink_b02": ("shrinkage", 0.2),
    "shrink_b03": ("shrinkage", 0.3),
}
PILOT_COND = ("baseline", "coast_b01", "coast_b02")
SETS = {  # name -> (client seed, episodes, run-dir suffix)
    "P": (45, 20, "P"),
    "S": (15, 30, "S"),
}
SUBSETS = {  # analysis subsets by init state
    "P_fresh": ("P", set(range(45, 50))),
    "P_fit": ("P", set(range(15))),
    "S_dev15_29": ("S", set(range(15, 30))),
    "S_heldout30_44": ("S", set(range(30, 45))),
}
_EP_RE = re.compile(r"Episode (\d+)/(\d+): success=(True|False)")


def find_task_dir(run: pathlib.Path) -> pathlib.Path:
    dirs = [p for p in run.iterdir() if p.is_dir() and re.match(r"\d\d-", p.name)]
    assert len(dirs) == 1, f"{run}: expected exactly one task dir, found {dirs}"
    return dirs[0]


def load_run(dirname: str, n: int, seed: int, task_id: int, cond: str) -> dict:
    run = OUT / dirname
    log = (run / "client.log").read_text()
    assert "CLIENT_EXIT=0" in log, f"{dirname}: client did not exit cleanly"
    succ = {int(m.group(1)) - 1: m.group(3) == "True" for m in _EP_RE.finditer(log)}
    assert sorted(succ) == list(range(n)), f"{dirname}: missing episodes"
    tdir = find_task_dir(run)
    episodes = []
    for ep in range(n):
        with iio.get_reader(tdir / f"episode_{ep:03d}.mp4") as reader:
            steps = reader.count_frames() - NUM_STEPS_WAIT
        ok = succ[ep]
        assert ok or steps == MAX_STEPS, f"{dirname} ep {ep}: failure that is not a {MAX_STEPS}-step timeout"
        episodes.append(
            {"episode": ep, "init_state": (seed + ep) % NUM_INIT_STATES, "success": int(ok), "steps": steps}
        )
    fps = [json.loads(line) for line in (tdir / "noise_fingerprints.jsonl").read_text().splitlines()]
    strategy, beta = COND[cond]
    diags = []
    if strategy is not None:
        diags = [json.loads(line) for line in (tdir / "steering_diagnostics.jsonl").read_text().splitlines()]
    else:
        assert not (tdir / "steering_diagnostics.jsonl").exists(), f"{dirname}: baseline has diagnostics"
    return {"episodes": episodes, "fps": fps, "diags": diags, "seed": seed, "n": n, "task_id": task_id, "cond": cond}


def verify_noise(runs: dict[str, dict]) -> dict:
    recomputed = 0
    coords: dict[str, dict] = {}
    counts: dict[str, list[int]] = {}
    for name, r in runs.items():
        by_ep = collections.defaultdict(list)
        for f in r["fps"]:
            key = {k: f[k] for k in ("master_seed", "task_id", "init_state", "rollout_step")}
            assert key["master_seed"] == MASTER_SEED, (name, key)
            assert key["task_id"] == r["task_id"], (name, key)
            assert key["init_state"] == r["episodes"][f["episode"]]["init_state"], f"{name}: init_state mismatch"
            assert noise_fingerprint(derive_noise(key, 10, 32)) == f["sha256"], f"{name}: bad fingerprint {key}"
            recomputed += 1
            by_ep[f["episode"]].append(f["rollout_step"])
        for ep, steps in by_ep.items():
            assert steps == list(range(0, r["episodes"][ep]["steps"], 5)), f"{name} ep {ep}: schedule"
        assert sorted(by_ep) == list(range(r["n"])), f"{name}: missing episodes in fingerprint log"
        counts[name] = [len(by_ep[ep]) for ep in range(r["n"])]
        coords[name] = {(f["init_state"], f["rollout_step"]): f["sha256"] for f in r["fps"]}
    pairs = {}
    for a, b in itertools.combinations(coords, 2):
        shared = coords[a].keys() & coords[b].keys()
        identical = sum(coords[a][c] == coords[b][c] for c in shared)
        assert identical == len(shared), f"{a} vs {b}: noise not paired"
        pairs[f"{a}|{b}"] = [len(shared), identical]
    first = next(iter(runs.values()))["episodes"][0]["init_state"]
    return {
        "fingerprints_recomputed_and_matched": recomputed,
        "requests": {n: len(r["fps"]) for n, r in runs.items()},
        "n_condition_pairs": len(pairs),
        "all_pairs_identical_at_every_shared_coordinate": all(v[0] == v[1] for v in pairs.values()),
        "shared_coordinates_min_max": [min(v[0] for v in pairs.values()), max(v[0] for v in pairs.values())],
        "sample_step0": {n: coords[n][(first, 0)][:16] for n in coords},
        "per_episode_counts": counts,
    }


def validate_diagnostics(runs: dict[str, dict]) -> None:
    for name, r in runs.items():
        strategy, beta = COND[r["cond"]]
        if strategy is None:
            continue
        calls = collections.defaultdict(list)
        for d in r["diags"]:
            calls[(d["episode"], d["rollout_step"])].append(d)
            assert (d["layer"], d["token_count"], d["hidden_dim"]) == (5, 10, 1024), (name, d)
            assert d["beta"] == beta, (name, d["beta"])
            assert d["strategy"] == strategy, (name, d["strategy"])
            assert d["init_state"] == r["episodes"][d["episode"]]["init_state"]
            assert all(math.isfinite(d[f]) for f in FIELDS)
        for rs in calls.values():
            assert [d["denoising_step"] for d in rs] == list(range(10)), f"{name}: denoising steps"
        fp_sched = {(f["episode"], f["rollout_step"]) for f in r["fps"]}
        assert set(calls) == fp_sched, f"{name}: diagnostics schedule != fingerprint schedule"


def episode_intervention(r: dict) -> dict[int, dict]:
    out = {}
    for ep in range(r["n"]):
        recs = [d for d in r["diags"] if d["episode"] == ep]
        out[ep] = {f: float(np.mean([d[f] for d in recs])) for f in FIELDS} | {"hook_applications": len(recs)}
    return out


def pooled_intervention(r: dict, states: set[int] | None = None) -> dict:
    recs = [d for d in r["diags"] if states is None or d["init_state"] in states]
    out = {"n_applications": len(recs)}
    for f in (
        "mean_delta_norm",
        "mean_relative_delta",
        "mean_norm_ratio",
        "mean_cosine_delta_hidden",
        "mean_hidden_norm",
    ):
        out[f] = round(float(np.mean([d[f] for d in recs])), 5)
    out["norm_ratio_sd"] = round(float(np.std([d["mean_norm_ratio"] for d in recs])), 5)
    return out


def sel(r: dict, states: set[int] | None) -> list[dict]:
    return [e for e in r["episodes"] if states is None or e["init_state"] in states]


def transition(x: list[dict], y: list[dict]) -> dict:
    """x -> y over paired episodes (same order / init states)."""
    assert [e["init_state"] for e in x] == [e["init_state"] for e in y]
    gains = [a["init_state"] for a, b in zip(x, y, strict=True) if not a["success"] and b["success"]]
    losses = [a["init_state"] for a, b in zip(x, y, strict=True) if a["success"] and not b["success"]]
    both = [(a["steps"], b["steps"]) for a, b in zip(x, y, strict=True) if a["success"] and b["success"]]
    return {
        "fail_to_success": len(gains),
        "success_to_fail": len(losses),
        "gain_states": gains,
        "loss_states": losses,
        "agreement": len(x) - len(gains) - len(losses),
        "n": len(x),
        "mcnemar_exact_p_reference_only": mcnemar_exact(len(gains), len(losses)),
        "both_success": len(both),
        "mean_steps_when_both_succeed": [
            round(float(np.mean([s[0] for s in both])), 1),
            round(float(np.mean([s[1] for s in both])), 1),
        ]
        if both
        else None,
    }


def analyze_set(runs: dict[str, dict], states: set[int] | None) -> dict:
    """Success, transitions, beta sensitivity and baseline-strength stratification over an init-state subset."""
    eps = {n: sel(r, states) for n, r in runs.items()}
    n = len(next(iter(eps.values())))
    succ = {
        c: {"successes": sum(e["success"] for e in v), "n": n, "wilson95": wilson(sum(e["success"] for e in v), n)}
        for c, v in eps.items()
    }
    base = eps["baseline"]
    vs_base = {c: transition(base, v) for c, v in eps.items() if c != "baseline"}
    pairs = {}
    for b in ("01", "02", "03"):
        if f"coast_b{b}" in eps and f"shrink_b{b}" in eps:
            pairs[f"coast_b{b}_vs_shrink_b{b}"] = transition(eps[f"shrink_b{b}"], eps[f"coast_b{b}"])
    beta_pairs = {}
    for a, b in (
        ("coast_b01", "coast_b02"),
        ("coast_b02", "coast_b03"),
        ("coast_b01", "coast_b03"),
        ("shrink_b01", "shrink_b02"),
        ("shrink_b02", "shrink_b03"),
        ("shrink_b01", "shrink_b03"),
    ):
        if a in eps and b in eps:
            beta_pairs[f"{a}->{b}"] = transition(eps[a], eps[b])
    # requested comparison E: COAST at every beta against shrinkage beta=0.1 (M=0.9I)
    vs_shrink01 = {}
    if "shrink_b01" in eps:
        for c in ("coast_b01", "coast_b02", "coast_b03"):
            if c in eps:
                vs_shrink01[f"{c}_vs_shrink_b01"] = transition(eps["shrink_b01"], eps[c])
    # baseline-strength stratification: rescue rate on baseline-failed states, retention on baseline-success states
    strat = {}
    bfail = [i for i, e in enumerate(base) if not e["success"]]
    bsucc = [i for i, e in enumerate(base) if e["success"]]
    for c, v in eps.items():
        if c == "baseline":
            continue
        strat[c] = {
            "baseline_failed_states": len(bfail),
            "rescued": sum(v[i]["success"] for i in bfail),
            "baseline_success_states": len(bsucc),
            "retained": sum(v[i]["success"] for i in bsucc),
        }
    return {
        "n_states": n,
        "success": succ,
        "vs_baseline": vs_base,
        "coast_vs_matched_beta_shrinkage": pairs,
        "coast_vs_shrink_b01": vs_shrink01,
        "beta_pairs": beta_pairs,
        "baseline_strength_stratification": strat,
    }


def replicate_check(sets_runs: dict[str, dict[str, dict]]) -> dict:
    """Determinism / reproducibility check of the identical-config conditions against Phases 2B and 3A."""
    out = {}
    mapping = {"baseline": "baseline", "coast_b01": "coast", "shrink_b01": "shrinkage"}
    if "S" not in sets_runs:
        return out
    with (EXP / "phase2b_episode_results.csv").open() as f:
        wide = {int(r["init_state"]): r for r in csv.DictReader(f)}
    ref = {
        "phase2b_dev15_29": {
            c: {s: (int(wide[s][f"{m}_success"]), int(wide[s][f"{m}_rollout_steps"])) for s in range(15, 30)}
            for c, m in mapping.items()
        }
    }
    with (EXP / "phase3a_episode_results.csv").open() as f:
        rows = list(csv.DictReader(f))
    ref["phase3a_heldout30_44"] = {
        c: {int(r["init_state"]): (int(r["success"]), int(r["rollout_steps"])) for r in rows if r["condition"] == m}
        for c, m in mapping.items()
    }
    for refname, table in ref.items():
        out[refname] = {}
        for c in mapping:
            cur = {e["init_state"]: (e["success"], e["steps"]) for e in sets_runs["S"][c]["episodes"]}
            states = sorted(table[c])
            same_outcome = sum(cur[s][0] == table[c][s][0] for s in states)
            same_len = sum(cur[s][1] == table[c][s][1] for s in states)
            out[refname][c] = {
                "n": len(states),
                "same_outcome": same_outcome,
                "same_rollout_length": same_len,
                "successes_now": sum(cur[s][0] for s in states),
                "successes_ref": sum(table[c][s][0] for s in states),
            }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot-task", type=int, default=None)
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    sets_runs: dict[str, dict[str, dict]] = {}
    summary: dict = {"part1": {}}
    for sname, (seed, n, suffix) in SETS.items():
        d = OUT / f"phase4a_baseline_task02_{suffix}"
        if not d.exists():
            continue
        try:
            sets_runs[sname] = {c: load_run(f"phase4a_{c}_task02_{suffix}", n, seed, 2, c) for c in COND}
        except (FileNotFoundError, AssertionError) as e:
            print(f"# set {sname} incomplete: {e}", file=sys.stderr)
            sets_runs.pop(sname, None)
    for sname, runs in sets_runs.items():
        validate_diagnostics(runs)
        block = {"noise_pairing": verify_noise(runs), "all_states": analyze_set(runs, None)}
        block["intervention_by_condition"] = {c: pooled_intervention(r) for c, r in runs.items() if r["diags"]}
        for subname, (parent, states) in SUBSETS.items():
            if parent == sname:
                block[subname] = analyze_set(runs, states)
                block.setdefault("intervention_by_subset", {})[subname] = {
                    c: pooled_intervention(r, states) for c, r in runs.items() if r["diags"]
                }
        summary["part1"][sname] = block
    if "P" in sets_runs and "S" in sets_runs:
        pooled = {c: {"episodes": sets_runs["P"][c]["episodes"] + sets_runs["S"][c]["episodes"]} for c in COND}
        for c in COND:
            pooled[c]["episodes"] = sorted(pooled[c]["episodes"], key=lambda e: e["init_state"])
        assert [e["init_state"] for e in pooled["baseline"]["episodes"]] == list(range(50))
        summary["part1"]["ALL50_pooled_descriptive"] = analyze_set(pooled, None)
    summary["replicate_vs_earlier_phases"] = replicate_check(sets_runs)

    # long-format CSV: Part 1
    rows = []
    for sname, runs in sets_runs.items():
        counts = summary["part1"][sname]["noise_pairing"]["per_episode_counts"]
        for c, r in runs.items():
            em = episode_intervention(r) if r["diags"] else None
            for e in r["episodes"]:
                strategy, beta = COND[c]
                row = {
                    "part": "1",
                    "task": "libero_10_task02",
                    "state_set": sname,
                    "condition": c,
                    "strategy": strategy or "none",
                    "beta": beta if beta is not None else "",
                    "episode": e["episode"],
                    "init_state": e["init_state"],
                    "success": e["success"],
                    "rollout_steps": e["steps"],
                    "noise_fingerprints": counts[c][e["episode"]],
                }
                if em:
                    m = em[e["episode"]]
                    row |= {
                        "hook_applications": m["hook_applications"],
                        "mean_delta_norm": f"{m['mean_delta_norm']:.6f}",
                        "mean_relative_delta": f"{m['mean_relative_delta']:.6f}",
                        "mean_norm_ratio": f"{m['mean_norm_ratio']:.6f}",
                        "mean_cosine_delta_hidden": f"{m['mean_cosine_delta_hidden']:.6f}",
                        "mean_hidden_norm": f"{m['mean_hidden_norm']:.6f}",
                    }
                rows.append(row)

    # Part 2 pilot
    if args.pilot_task is not None:
        tid = args.pilot_task
        pruns = {c: load_run(f"phase4a_pilot_{c}_task{tid:02d}_pilot", 10, 15, tid, c) for c in PILOT_COND}
        validate_diagnostics(pruns)
        summary["part2"] = {
            "task_id": tid,
            "noise_pairing": verify_noise(pruns),
            "analysis": analyze_set(pruns, None),
            "intervention_by_condition": {c: pooled_intervention(r) for c, r in pruns.items() if r["diags"]},
        }
        summary["part2"]["per_state"] = {
            c: [(e["init_state"], e["success"], e["steps"]) for e in r["episodes"]] for c, r in pruns.items()
        }
        counts = summary["part2"]["noise_pairing"]["per_episode_counts"]
        for c, r in pruns.items():
            em = episode_intervention(r) if r["diags"] else None
            for e in r["episodes"]:
                strategy, beta = COND[c]
                row = {
                    "part": "2",
                    "task": f"libero_10_task{tid:02d}",
                    "state_set": "pilot",
                    "condition": c,
                    "strategy": strategy or "none",
                    "beta": beta if beta is not None else "",
                    "episode": e["episode"],
                    "init_state": e["init_state"],
                    "success": e["success"],
                    "rollout_steps": e["steps"],
                    "noise_fingerprints": counts[c][e["episode"]],
                }
                if em:
                    m = em[e["episode"]]
                    row |= {
                        "hook_applications": m["hook_applications"],
                        "mean_delta_norm": f"{m['mean_delta_norm']:.6f}",
                        "mean_relative_delta": f"{m['mean_relative_delta']:.6f}",
                        "mean_norm_ratio": f"{m['mean_norm_ratio']:.6f}",
                        "mean_cosine_delta_hidden": f"{m['mean_cosine_delta_hidden']:.6f}",
                        "mean_hidden_norm": f"{m['mean_hidden_norm']:.6f}",
                    }
                rows.append(row)

    cols = [
        "part",
        "task",
        "state_set",
        "condition",
        "strategy",
        "beta",
        "episode",
        "init_state",
        "success",
        "rollout_steps",
        "noise_fingerprints",
        "hook_applications",
        "mean_delta_norm",
        "mean_relative_delta",
        "mean_norm_ratio",
        "mean_cosine_delta_hidden",
        "mean_hidden_norm",
    ]
    with (EXP / "phase4a_episode_results.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in cols})
    text = json.dumps(summary, indent=1)
    if args.json_out:
        pathlib.Path(args.json_out).write_text(text)
    print(text)


if __name__ == "__main__":
    main()
