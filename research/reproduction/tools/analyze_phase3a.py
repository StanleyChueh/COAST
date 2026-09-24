"""Phase 3A analysis: held-out validation (init states 30-44) of baseline / shrinkage / real COAST / random matched.

Inputs (gitignored run outputs under
examples/libero_env/output/phase3a_{baseline,shrinkage,coast,random_matched}_task02_seed30/):
  client.log (per-episode success, CLIENT_EXIT=0), episode_NNN.mp4 (rollout length = frames - 10),
  noise_fingerprints.jsonl, steering_diagnostics.jsonl (steered conditions only).
Development reference: the Phase 2B runs (init states 15-29, same configuration and master seed).

The script
  1. recomputes every held-out noise fingerprint from its key and checks pairing across the four conditions,
  2. validates the diagnostics logs (10 records per call, steps 0..9, layer 5, finite, right strategy),
  3. computes success, transitions and intervention statistics, and the same quantities for Phase 2B,
  4. writes research/reproduction/experiments/phase3a_episode_results.csv (long format) and prints a JSON summary.

Descriptive only. Usage (repo root, CPU only):
    CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu uv run python research/reproduction/tools/analyze_phase3a.py
"""

from __future__ import annotations

import collections
import csv
import itertools
import json
import math
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from analyze_phase1d import EXP
from analyze_phase1d import OUT
from analyze_phase1d import TASK_DIR
from analyze_phase1d import load_episodes
from analyze_phase1d import load_fingerprints
from analyze_phase1d import pair_stats
from analyze_phase1d import wilson
from analyze_phase2a import FIELDS

from openpi.serving.noise_control import derive_noise
from openpi.serving.noise_control import noise_fingerprint

MASTER_SEED = 100
NAMES = ("baseline", "shrinkage", "coast", "random_matched")
STRATEGY = {"shrinkage": "shrinkage", "coast": "global", "random_matched": "random_matched"}
STEERED = tuple(STRATEGY)
SPLITS = {
    "heldout": {"states": list(range(30, 45)), "dir": "phase3a_{}_task02_seed30"},
    "dev_phase2b": {"states": list(range(15, 30)), "dir": "phase2b_{}_task02_seed15"},
}


def run_dir(split: str, name: str) -> str:
    return SPLITS[split]["dir"].format(name)


def load_diagnostics(split: str, name: str) -> list[dict]:
    path = OUT / run_dir(split, name) / TASK_DIR / "steering_diagnostics.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()]


def verify_noise(split: str, fps: dict[str, list[dict]], episodes: dict[str, list[dict]]) -> dict:
    states = SPLITS[split]["states"]
    recomputed = 0
    per_episode_counts = {}
    for name, records in fps.items():
        by_ep = collections.defaultdict(list)
        for r in records:
            key = {k: r[k] for k in ("master_seed", "task_id", "init_state", "rollout_step")}
            assert key["master_seed"] == MASTER_SEED
            assert key["task_id"] == 2
            assert key["init_state"] == states[r["episode"]], f"{name}: episode/init_state mismatch"
            assert noise_fingerprint(derive_noise(key, 10, 32)) == r["sha256"], f"{name}: bad fingerprint {key}"
            recomputed += 1
            by_ep[r["episode"]].append(r["rollout_step"])
        for ep, steps in by_ep.items():
            assert steps == list(range(0, episodes[name][ep]["steps"], 5)), f"{name} ep {ep}: schedule"
        assert sorted(by_ep) == list(range(15)), f"{name}: missing episodes in fingerprint log"
        per_episode_counts[name] = [len(by_ep[ep]) for ep in range(15)]
    coords = {name: {(r["init_state"], r["rollout_step"]): r["sha256"] for r in recs} for name, recs in fps.items()}
    pairs = {}
    for a, b in itertools.combinations(coords, 2):
        shared = coords[a].keys() & coords[b].keys()
        identical = sum(coords[a][c] == coords[b][c] for c in shared)
        assert identical == len(shared), f"{a} vs {b}: unpaired noise"
        pairs[f"{a}_vs_{b}"] = {"shared_coordinates": len(shared), "identical": identical}
    first = (states[0], 0)
    return {
        "recomputed_and_matched": recomputed,
        "requests": {n: len(r) for n, r in fps.items()},
        "pairs": pairs,
        f"sample_init{states[0]}_step0": {n: coords[n][first][:16] for n in coords},
        "per_episode_counts": per_episode_counts,
        "init_states_seen": sorted({r["init_state"] for recs in fps.values() for r in recs}),
    }


def validate_diagnostics(name: str, records: list[dict], fps: list[dict], states: list[int]) -> None:
    calls = collections.defaultdict(list)
    for r in records:
        calls[(r["episode"], r["rollout_step"])].append(r)
        assert (r["layer"], r["token_count"], r["hidden_dim"], r["beta"]) == (5, 10, 1024, 0.1)
        assert r["strategy"] == STRATEGY[name], (name, r["strategy"])
        assert r["init_state"] == states[r["episode"]]
        assert all(math.isfinite(r[f]) for f in FIELDS)
    for rs in calls.values():
        assert [r["denoising_step"] for r in rs] == list(range(10))
    assert sorted(calls) == sorted((f["episode"], f["rollout_step"]) for f in fps), f"{name}: schedule mismatch"


def summarize(values: list[float]) -> dict:
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def analyze_split(split: str) -> tuple[dict, dict, dict, dict]:
    states = SPLITS[split]["states"]
    episodes = {n: load_episodes(run_dir(split, n)) for n in NAMES}
    fps = {n: load_fingerprints(run_dir(split, n)) for n in NAMES}
    noise = verify_noise(split, fps, episodes)
    diags = {n: load_diagnostics(split, n) for n in STEERED}
    for n in STEERED:
        validate_diagnostics(n, diags[n], fps[n], states)
    succ = {n: [e["success"] for e in eps] for n, eps in episodes.items()}
    out = {
        "noise_pairing": noise,
        "init_states": states,
        "success": {
            n: {"successes": sum(v), "rate": round(sum(v) / 15, 3), "wilson95": wilson(sum(v), 15)}
            for n, v in succ.items()
        },
        "success_vectors": succ,
        "rollout_steps": {n: [e["steps"] for e in eps] for n, eps in episodes.items()},
        "transitions": {
            f"{a}_vs_{b}": pair_stats(episodes[a], episodes[b]) for a, b in itertools.combinations(NAMES, 2)
        },
        "disagreement_states": {
            f"{a}_vs_{b}": [states[i] for i in range(15) if succ[a][i] != succ[b][i]]
            for a, b in itertools.combinations(NAMES, 2)
        },
        "all_four_success": sum(all(succ[n][i] for n in NAMES) for i in range(15)),
        "all_four_failure": sum(not any(succ[n][i] for n in NAMES) for i in range(15)),
        "steered_all_agree": sum(len({succ[n][i] for n in STEERED}) == 1 for i in range(15)),
        "hook_applications": {n: len(diags[n]) for n in STEERED},
        "per_application": {n: {f: summarize([r[f] for r in diags[n]]) for f in FIELDS} for n in STEERED},
        "by_denoising_step": {
            n: {
                f: [float(np.mean([r[f] for r in diags[n] if r["denoising_step"] == t])) for t in range(10)]
                for f in ("mean_delta_norm", "mean_norm_ratio", "mean_hidden_norm")
            }
            for n in STEERED
        },
    }
    # Paired rollout step 0: identical observation + noise across conditions -> within-call effect of M only.
    key = lambda r: (r["episode"], r["rollout_step"], r["denoising_step"])  # noqa: E731
    by = {n: {key(r): r for r in diags[n]} for n in STEERED}
    paired = {}
    for a, b in itertools.combinations(STEERED, 2):
        step0 = [k for k in by[a].keys() & by[b].keys() if k[1] == 0]
        shared = by[a].keys() & by[b].keys()
        paired[f"{a}_vs_{b}"] = {
            "rollout_step0_identical_hidden_norm_at_denoising_step0": sum(
                by[a][k]["mean_hidden_norm"] == by[b][k]["mean_hidden_norm"] for k in step0 if k[2] == 0
            ),
            "rollout_step0_max_abs_rel_hidden_norm_diff": float(
                max(abs(by[a][k]["mean_hidden_norm"] / by[b][k]["mean_hidden_norm"] - 1) for k in step0)
            ),
            "shared_applications": len(shared),
            "delta_norm_ratio": float(
                np.mean([by[a][k]["mean_delta_norm"] / by[b][k]["mean_delta_norm"] for k in shared])
            ),
            "norm_ratio_diff": float(
                np.mean([by[a][k]["mean_norm_ratio"] - by[b][k]["mean_norm_ratio"] for k in shared])
            ),
        }
    out["paired"] = paired
    return out, episodes, fps, diags


def main() -> None:
    held, episodes, _, diags = analyze_split("heldout")
    dev, _, _, _ = analyze_split("dev_phase2b")
    summary = {"heldout": held, "dev_phase2b": dev}
    # Held-out and development noise keys differ by init_state, so no noise coordinate is shared between splits.
    summary["heldout_dev_state_overlap"] = sorted(set(held["init_states"]) & set(dev["init_states"]))

    def key_stats(s: dict) -> dict:
        return {
            "successes": {n: s["success"][n]["successes"] for n in NAMES},
            "baseline_vs_coast": s["transitions"]["baseline_vs_coast"],
            "coast_vs_shrinkage_agreement": s["transitions"]["shrinkage_vs_coast"]["agreement"],
            "coast_vs_random_agreement": s["transitions"]["coast_vs_random_matched"]["agreement"],
            "steered_all_agree": s["steered_all_agree"],
            "norm_ratio": {n: s["per_application"][n]["mean_norm_ratio"]["mean"] for n in STEERED},
        }

    summary["dev_vs_heldout"] = {"dev_phase2b": key_stats(dev), "heldout": key_stats(held)}

    # Long-format per-episode CSV (held-out only).
    ep_means = {
        n: {
            ep: {f: float(np.mean([r[f] for r in diags[n] if r["episode"] == ep])) for f in FIELDS}
            | {"hook_applications": sum(r["episode"] == ep for r in diags[n])}
            for ep in range(15)
        }
        for n in STEERED
    }
    counts = held["noise_pairing"]["per_episode_counts"]
    with (EXP / "phase3a_episode_results.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "condition",
                "strategy",
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
        )
        for n in NAMES:
            for ep in range(15):
                row = [
                    n,
                    STRATEGY.get(n, "none"),
                    ep,
                    held["init_states"][ep],
                    episodes[n][ep]["success"],
                    episodes[n][ep]["steps"],
                    counts[n][ep],
                ]
                if n in STEERED:
                    m = ep_means[n][ep]
                    row += [m["hook_applications"]] + [
                        f"{m[k]:.6f}"
                        for k in (
                            "mean_delta_norm",
                            "mean_relative_delta",
                            "mean_norm_ratio",
                            "mean_cosine_delta_hidden",
                            "mean_hidden_norm",
                        )
                    ]
                else:
                    row += [0, "", "", "", "", ""]
                writer.writerow(row)

    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
