"""Phase 7B exploratory analysis (NOT pre-registered as gating; run after the primary records were frozen).

1. Action-level comparison at identical observations and identical noise (phase7b_action_probe.py output):
   per task, over held-out states 15-44 x master seeds {100, 200} (the first request of each primary episode):
     ||a_coast - a_base||, ||a_shrink - a_base||, ||a_rand - a_base||, ||a_coast - a_shrink||, ||a_coast - a_rand||,
     cos(a_coast - a_base, a_shrink - a_base), cos(a_rand - a_base, a_shrink - a_base),
     ||a_coast - a_shrink|| / ||a_shrink - a_base||                      (action chunks flattened, 10 x 7)
   Cross-check: the probe's rollout_step-0 noise fingerprints equal the primary episodes' noise_verification_hash.
2. PURE_SHRINKAGE vs BASELINE and RANDOM_CONTROL vs PURE_SHRINKAGE paired tables (same machinery as the primary
   analysis; exploratory context for the primary comparison).

Writes experiments/phase7b_action_probe.csv and experiments/phase7b_exploratory.json.

Usage (repo root, CPU): .venv/bin/python research/reproduction/tools/analyze_phase7b_exploratory.py <probe_dir>
"""

from __future__ import annotations

import csv
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from analyze_phase1d import EXP
from analyze_phase7b import BOOT_SEED
from analyze_phase7b import N_BOOT
from analyze_phase7b import N_STATES
from analyze_phase7b import TASKS
from analyze_phase7b import paired

CONDS = ("BASELINE", "ORIGINAL_COAST", "PURE_SHRINKAGE", "RANDOM_CONTROL")


def cos(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.sum(a * b, 1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-12)


def main() -> None:
    probe = pathlib.Path(sys.argv[1])
    with open(EXP / "phase7b_episode_results.csv") as f:
        episodes = list(csv.DictReader(f))
    first_hash = {
        (r["task"], int(r["master_seed"]), int(r["init_state"])): r["noise_verification_hash"] for r in episodes
    }
    rows, out = [], {"action_probe": {}, "extra_paired": {}}
    for task in TASKS:
        z = np.load(probe / f"{task}_probe.npz")
        assert tuple(str(c) for c in z["conditions"]) == CONDS
        a = z["actions"].reshape(2, N_STATES, 4, -1)  # seed, state, condition, 70
        # noise cross-check against the primary episodes
        log = [json.loads(line) for line in (probe / f"{task}_probe_noise_log.jsonl").read_text().splitlines()]
        n_checked = 0
        for rec in log:
            assert rec["sha256"] == first_hash[(task, rec["master_seed"], rec["init_state"])], rec
            n_checked += 1
        flat = a.reshape(-1, 4, a.shape[-1])
        base, coast, shrink, rand = (flat[:, i] for i in range(4))
        m = {
            "task": task,
            "n_observations": len(flat),
            "noise_hashes_matching_primary": n_checked,
            "a_base_norm": float(np.mean(np.linalg.norm(base, axis=1))),
            "coast_minus_base": float(np.mean(np.linalg.norm(coast - base, axis=1))),
            "shrink_minus_base": float(np.mean(np.linalg.norm(shrink - base, axis=1))),
            "random_minus_base": float(np.mean(np.linalg.norm(rand - base, axis=1))),
            "coast_minus_shrink": float(np.mean(np.linalg.norm(coast - shrink, axis=1))),
            "coast_minus_random": float(np.mean(np.linalg.norm(coast - rand, axis=1))),
            "cos_coastdelta_shrinkdelta": float(np.mean(cos(coast - base, shrink - base))),
            "cos_coastdelta_shrinkdelta_min": float(np.min(cos(coast - base, shrink - base))),
            "cos_randdelta_shrinkdelta": float(np.mean(cos(rand - base, shrink - base))),
            "ratio_coast_shrink_over_shrink_base": float(
                np.mean(np.linalg.norm(coast - shrink, axis=1) / np.linalg.norm(shrink - base, axis=1))
            ),
        }
        rows.append({k: (round(v, 5) if isinstance(v, float) else v) for k, v in m.items()})
        out["action_probe"][task] = m

    with open(EXP / "phase7b_action_probe.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    succ = {}
    for r in episodes:
        succ.setdefault((r["task"], int(r["repeat"]), r["condition"]), {})[int(r["init_state"])] = int(r["success"])
    rng = np.random.default_rng(BOOT_SEED)
    for task in TASKS:
        for x, y in (("PURE_SHRINKAGE", "BASELINE"), ("RANDOM_CONTROL", "PURE_SHRINKAGE")):
            vec = {
                (rep, c): [succ[(task, rep, c)][s] for s in range(15, 15 + N_STATES)] for rep in (1, 2) for c in (x, y)
            }
            t1, t2 = paired(vec[(1, x)], vec[(1, y)]), paired(vec[(2, x)], vec[(2, y)])
            per_state = (
                np.array(vec[(1, x)]) - np.array(vec[(1, y)]) + np.array(vec[(2, x)]) - np.array(vec[(2, y)])
            ) / 2.0
            draws = per_state[rng.integers(0, N_STATES, size=(N_BOOT, N_STATES))].mean(axis=1)
            lo, hi = np.percentile(draws, [2.5, 97.5])
            out["extra_paired"][f"{task}|{x}_vs_{y}"] = {
                "rep1": t1,
                "rep2": t2,
                "pooled_rd": float(per_state.mean()),
                "boot95": [float(lo), float(hi)],
            }
    (EXP / "phase7b_exploratory.json").write_text(json.dumps(out, indent=2))
    for r in rows:
        print(r)
    for k, v in out["extra_paired"].items():
        print(
            k,
            f"rep1 b/c={v['rep1']['coast_only_b']}/{v['rep1']['other_only_c']} p={v['rep1']['mcnemar_p']:.3f}",
            f"rep2 b/c={v['rep2']['coast_only_b']}/{v['rep2']['other_only_c']} p={v['rep2']['mcnemar_p']:.3f}",
            f"RD={v['pooled_rd']:+.3f} CI=[{v['boot95'][0]:+.3f},{v['boot95'][1]:+.3f}]",
        )


if __name__ == "__main__":
    main()
