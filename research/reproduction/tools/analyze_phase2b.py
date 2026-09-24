"""Phase 2B analysis: baseline vs shrinkage vs real COAST (global) vs random matched conceptor, paired noise seed 100.

Inputs (gitignored run outputs under
examples/libero_env/output/phase2b_{baseline,shrinkage,coast,random_matched}_task02_seed15/):
  client.log (per-episode success, CLIENT_EXIT=0), episode_NNN.mp4 (rollout length = frames - 10),
  noise_fingerprints.jsonl, steering_diagnostics.jsonl (steered conditions only).

The script
  1. recomputes every noise fingerprint and checks pairing across the four conditions,
  2. validates the diagnostics logs (10 records per call, steps 0..9, layer 5, finite, right strategy),
  3. replays baseline / shrinkage / COAST against the Phase 2A runs with the same configuration and seed,
  4. writes research/reproduction/experiments/phase2b_episode_results.csv and prints a JSON summary.

Descriptive only. Usage (repo root, CPU only):
    CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu uv run python research/reproduction/tools/analyze_phase2b.py
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
from analyze_phase1d import INIT_STATES
from analyze_phase1d import OUT
from analyze_phase1d import TASK_DIR
from analyze_phase1d import load_episodes
from analyze_phase1d import load_fingerprints
from analyze_phase1d import pair_stats
from analyze_phase1d import wilson
from analyze_phase2a import FIELDS
from analyze_phase2a import verify_noise

CONDITIONS = {
    "baseline": "phase2b_baseline_task02_seed15",
    "shrinkage": "phase2b_shrinkage_task02_seed15",
    "coast": "phase2b_coast_task02_seed15",
    "random_matched": "phase2b_random_matched_task02_seed15",
}
STRATEGY = {"shrinkage": "shrinkage", "coast": "global", "random_matched": "random_matched"}
STEERED = tuple(STRATEGY)
PHASE2A = {  # same configuration and master seed, earlier server process (same GPU 1)
    "baseline": "phase2a_baseline_task02_seed15",
    "shrinkage": "phase2a_shrinkage_task02_seed15",
    "coast": "phase2a_coast_task02_seed15",
}
EPISODE_FIELDS = ("mean_delta_norm", "mean_relative_delta", "mean_hidden_norm", "mean_norm_ratio")


def load_diagnostics(name: str) -> list[dict]:
    path = OUT / CONDITIONS[name] / TASK_DIR / "steering_diagnostics.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()]


def validate_diagnostics(name: str, records: list[dict], fps: list[dict]) -> None:
    calls = collections.defaultdict(list)
    for r in records:
        calls[(r["episode"], r["rollout_step"])].append(r)
        assert (r["layer"], r["token_count"], r["hidden_dim"], r["beta"]) == (5, 10, 1024, 0.1)
        assert r["strategy"] == STRATEGY[name], (name, r["strategy"])
        assert all(math.isfinite(r[f]) for f in FIELDS)
    for rs in calls.values():
        assert [r["denoising_step"] for r in rs] == list(range(10))
    assert sorted(calls) == sorted((f["episode"], f["rollout_step"]) for f in fps), f"{name}: schedule mismatch"


def replay(episodes: dict[str, list[dict]]) -> dict:
    out = {}
    for cond, run_dir in PHASE2A.items():
        ref = load_episodes(run_dir)
        stats = pair_stats(ref, episodes[cond])
        out[f"phase2a_{cond}_vs_phase2b_{cond}"] = {
            "identical_outcomes": stats["agreement"],
            "identical_rollout_lengths": stats["identical_rollout_length"],
            "differing": [
                {
                    "init_state": INIT_STATES[i],
                    "phase2a": [a["success"], a["steps"]],
                    "phase2b": [b["success"], b["steps"]],
                }
                for i, (a, b) in enumerate(zip(ref, episodes[cond], strict=True))
                if a != b
            ],
        }
    return out


def episode_means(records: list[dict], field: str) -> list[float]:
    by_ep = collections.defaultdict(list)
    for r in records:
        by_ep[r["episode"]].append(r[field])
    return [float(np.mean(by_ep[ep])) for ep in range(15)]


def stats(values: list[float]) -> dict:
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def main() -> None:
    episodes = {name: load_episodes(run_dir) for name, run_dir in CONDITIONS.items()}
    fps = {name: load_fingerprints(run_dir) for name, run_dir in CONDITIONS.items()}
    noise = verify_noise(fps, episodes)
    diags = {name: load_diagnostics(name) for name in STEERED}
    for name in STEERED:
        validate_diagnostics(name, diags[name], fps[name])

    successes = {name: [e["success"] for e in eps] for name, eps in episodes.items()}
    summary: dict = {
        "noise_pairing": noise,
        "success": {
            name: {"successes": sum(v), "rate": round(sum(v) / 15, 3), "wilson95": wilson(sum(v), 15)}
            for name, v in successes.items()
        },
        "success_vectors": successes,
        "rollout_steps": {name: [e["steps"] for e in eps] for name, eps in episodes.items()},
        "transitions": {
            f"{a}_vs_{b}": pair_stats(episodes[a], episodes[b]) for a, b in itertools.combinations(CONDITIONS, 2)
        },
        "disagreement_states": {
            f"{a}_vs_{b}": [INIT_STATES[i] for i in range(15) if successes[a][i] != successes[b][i]]
            for a, b in itertools.combinations(CONDITIONS, 2)
        },
        "all_four_success": sum(all(successes[n][i] for n in CONDITIONS) for i in range(15)),
        "all_four_failure": sum(not any(successes[n][i] for n in CONDITIONS) for i in range(15)),
        "steered_all_agree": sum(len({successes[n][i] for n in STEERED}) == 1 for i in range(15)),
        "replay_vs_phase2a": replay(episodes),
        "counts": {
            name: {"inference_calls": len(fps[name]), "hook_applications": len(diags.get(name, []))}
            for name in CONDITIONS
        },
    }

    # Intervention magnitude and hidden-norm change: per application, per denoising step, per episode.
    summary["per_application"] = {name: {f: stats([r[f] for r in diags[name]]) for f in FIELDS} for name in STEERED}
    summary["by_denoising_step"] = {
        name: {
            f: [float(np.mean([r[f] for r in diags[name] if r["denoising_step"] == t])) for t in range(10)]
            for f in ("mean_delta_norm", "mean_norm_ratio", "mean_cosine_delta_hidden", "mean_hidden_norm")
        }
        for name in STEERED
    }
    per_episode = {name: {f: episode_means(diags[name], f) for f in EPISODE_FIELDS} for name in STEERED}

    # Paired comparisons. At rollout_step 0 all conditions see an identical observation and identical noise,
    # so h at denoising step 0 is identical; later denoising steps differ only by the within-call effect of M.
    key = lambda r: (r["episode"], r["rollout_step"], r["denoising_step"])  # noqa: E731
    by = {name: {key(r): r for r in diags[name]} for name in STEERED}
    paired = {}
    for a, b in itertools.combinations(STEERED, 2):
        step0 = sorted(k for k in by[a].keys() & by[b].keys() if k[1] == 0)
        shared = sorted(by[a].keys() & by[b].keys())
        paired[f"{a}_vs_{b}"] = {
            "rollout_step0": {
                "n_applications": len(step0),
                "hidden_norm_identical_at_denoising_step0": sum(
                    by[a][k]["mean_hidden_norm"] == by[b][k]["mean_hidden_norm"] for k in step0 if k[2] == 0
                ),
                "hidden_norm_rel_diff_by_step": [
                    float(
                        np.mean(
                            [
                                (by[a][k]["mean_hidden_norm"] - by[b][k]["mean_hidden_norm"])
                                / by[b][k]["mean_hidden_norm"]
                                for k in step0
                                if k[2] == t
                            ]
                        )
                    )
                    for t in range(10)
                ],
            },
            "shared_coordinates": {
                "n_applications": len(shared),
                "mean_delta_norm_diff": float(
                    np.mean([by[a][k]["mean_delta_norm"] - by[b][k]["mean_delta_norm"] for k in shared])
                ),
                "delta_norm_ratio": float(
                    np.mean([by[a][k]["mean_delta_norm"] / by[b][k]["mean_delta_norm"] for k in shared])
                ),
                "norm_ratio_diff": float(
                    np.mean([by[a][k]["mean_norm_ratio"] - by[b][k]["mean_norm_ratio"] for k in shared])
                ),
            },
        }
    summary["paired"] = paired

    with (EXP / "phase2b_episode_results.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        header = ["episode", "init_state"]
        for name in CONDITIONS:
            header += [f"{name}_success", f"{name}_rollout_steps"]
        for name in STEERED:
            header += [f"{name}_{fld}" for fld in EPISODE_FIELDS]
        writer.writerow(header)
        for ep in range(15):
            row = [ep, INIT_STATES[ep]]
            for name in CONDITIONS:
                row += [episodes[name][ep]["success"], episodes[name][ep]["steps"]]
            for name in STEERED:
                row += [f"{per_episode[name][fld][ep]:.6f}" for fld in EPISODE_FIELDS]
            writer.writerow(row)

    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
