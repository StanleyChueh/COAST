"""Phase 1E analysis: steering intervention diagnostics (layer 5, alpha 0.5, beta 0.1, global, V0 NPZ).

Inputs (gitignored run outputs under examples/libero_env/output/phase1e_repo_diag_task02_seed15/):
  - client.log                  per-episode success, clean exit ("exit=0")
  - <task>/episode_NNN.mp4      rollout length (= frames - num_steps_wait, Phase 1C/1D convention)
  - <task>/steering_diagnostics.jsonl   one record per steering-hook application
  - <task>/noise_fingerprints.jsonl     one record per inference call (paired-noise schedule)

The script
  1. validates the diagnostics log (10 records per inference call, denoising steps 0..9, one layer,
     finite values, request schedule identical to the noise-fingerprint log),
  2. checks the run against the Phase 1D repository-faithful condition (same config, same noise),
  3. writes two small CSVs and prints a JSON summary (distributions, per-denoising-step means,
     success vs failure, correlations, extreme timesteps).

Descriptive only: no causal claims. Usage (repo root, CPU only):
    CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu uv run python research/reproduction/tools/analyze_phase1e.py
"""

from __future__ import annotations

import collections
import csv
import json
import math
import pathlib
import re

import imageio.v2 as iio
import numpy as np
from scipy import stats

ROOT = pathlib.Path(__file__).resolve().parents[3]
OUT = ROOT / "examples/libero_env/output"
RUN = "phase1e_repo_diag_task02_seed15"
TASK_DIR = "02-kitchen-scene3-turn-on-the-stove-and-put-the-moka-pot-on-it"
EXP = ROOT / "research/reproduction/experiments"
PHASE1D_CSV = EXP / "phase1d_episode_results.csv"
PHASE1D_REPO_FINGERPRINTS = OUT / "phase1d_repo_task02_seed15" / TASK_DIR / "noise_fingerprints.jsonl"
NUM_STEPS_WAIT = 10
NUM_EPISODES = 15
INIT_STATES = list(range(15, 30))
NUM_DENOISING_STEPS = 10
LAYER, BETA, ALPHA, STRATEGY = 5, 0.1, 0.5, "global"
TOKENS, HIDDEN = 10, 1024
FIELDS = (
    "mean_delta_norm",
    "max_delta_norm",
    "mean_relative_delta",
    "max_relative_delta",
    "mean_hidden_norm",
    "mean_cosine_delta_hidden",
)
_EPISODE_RE = re.compile(r"Episode (\d+)/15: success=(True|False)")


def load_episodes() -> list[dict]:
    log = (OUT / RUN / "client.log").read_text()
    assert "exit=0" in log, "client did not exit cleanly"
    successes = {int(m.group(1)) - 1: m.group(2) == "True" for m in _EPISODE_RE.finditer(log)}
    assert sorted(successes) == list(range(NUM_EPISODES)), "missing episodes"
    episodes = []
    for ep in range(NUM_EPISODES):
        with iio.get_reader(OUT / RUN / TASK_DIR / f"episode_{ep:03d}.mp4") as reader:
            steps = reader.count_frames() - NUM_STEPS_WAIT
        episodes.append({"episode": ep, "init_state": INIT_STATES[ep], "success": int(successes[ep]), "steps": steps})
    return episodes


def load_jsonl(path: pathlib.Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def validate(records: list[dict], episodes: list[dict]) -> dict:
    calls = collections.defaultdict(list)
    for r in records:
        calls[(r["episode"], r["rollout_step"])].append(r)
    for (ep, step), rs in calls.items():
        assert [r["denoising_step"] for r in rs] == list(range(NUM_DENOISING_STEPS)), (ep, step)
        for r in rs:
            assert r["init_state"] == INIT_STATES[ep]
            assert (r["layer"], r["token_count"], r["hidden_dim"]) == (LAYER, TOKENS, HIDDEN)
            assert (r["beta"], r["alpha"], r["strategy"]) == (BETA, ALPHA, STRATEGY)
            assert all(math.isfinite(r[f]) for f in FIELDS)
    for e in episodes:
        steps = sorted(s for ep, s in calls if ep == e["episode"])
        assert steps == list(range(0, e["steps"], 5)), f"episode {e['episode']}: unexpected request schedule"
    fingerprints = load_jsonl(OUT / RUN / TASK_DIR / "noise_fingerprints.jsonl")
    fp_coords = sorted((f["episode"], f["rollout_step"]) for f in fingerprints)
    assert fp_coords == sorted(calls), "diagnostics and noise-fingerprint request schedules differ"
    return {
        "inference_calls": len(calls),
        "hook_applications": len(records),
        "records_per_call": NUM_DENOISING_STEPS,
        "layers_seen": sorted({r["layer"] for r in records}),
        "all_values_finite": True,
        "schedule_matches_noise_fingerprints": True,
        "fingerprints": fingerprints,
    }


def phase1d_replay(episodes: list[dict], fingerprints: list[dict]) -> dict:
    with PHASE1D_CSV.open() as f:
        rows = list(csv.DictReader(f))
    success_1d = [int(r["repo_success"]) for r in rows]
    steps_1d = [int(r["repo_rollout_steps"]) for r in rows]
    result = {
        "identical_outcomes": sum(a == e["success"] for a, e in zip(success_1d, episodes, strict=True)),
        "identical_rollout_lengths": sum(a == e["steps"] for a, e in zip(steps_1d, episodes, strict=True)),
    }
    if PHASE1D_REPO_FINGERPRINTS.exists():
        old = {(f["init_state"], f["rollout_step"]): f["sha256"] for f in load_jsonl(PHASE1D_REPO_FINGERPRINTS)}
        new = {(f["init_state"], f["rollout_step"]): f["sha256"] for f in fingerprints}
        shared = old.keys() & new.keys()
        result["noise_fingerprints_shared"] = len(shared)
        result["noise_fingerprints_identical"] = sum(old[k] == new[k] for k in shared)
        result["phase1d_requests"] = len(old)
    return result


def describe(values) -> dict:
    v = np.asarray(values, dtype=np.float64)
    q = np.percentile(v, [0, 5, 25, 50, 75, 95, 100])
    return {
        "n": int(v.size),
        "mean": float(v.mean()),
        "std": float(v.std(ddof=1)),
        "min": float(q[0]),
        "p05": float(q[1]),
        "p25": float(q[2]),
        "median": float(q[3]),
        "p75": float(q[4]),
        "p95": float(q[5]),
        "max": float(q[6]),
    }


def episode_means(records: list[dict], field: str, max_rollout_step: int | None = None) -> list[float]:
    out = []
    for ep in range(NUM_EPISODES):
        vals = [
            r[field]
            for r in records
            if r["episode"] == ep and (max_rollout_step is None or r["rollout_step"] <= max_rollout_step)
        ]
        out.append(float(np.mean(vals)))
    return out


def group_compare(values: list[float], success: list[int]) -> dict:
    s = [v for v, y in zip(values, success, strict=True) if y]
    f = [v for v, y in zip(values, success, strict=True) if not y]
    r, r_p = stats.pearsonr(success, values)  # point-biserial
    rho, rho_p = stats.spearmanr(success, values)
    u = stats.mannwhitneyu(s, f, alternative="two-sided", method="exact")
    return {
        "success_n": len(s),
        "failure_n": len(f),
        "success_mean": float(np.mean(s)),
        "success_std": float(np.std(s, ddof=1)),
        "failure_mean": float(np.mean(f)),
        "failure_std": float(np.std(f, ddof=1)) if len(f) > 1 else None,
        "failure_minus_success": float(np.mean(f) - np.mean(s)),
        "point_biserial_r": float(r),
        "point_biserial_p_reference_only": float(r_p),
        "spearman_rho": float(rho),
        "spearman_p_reference_only": float(rho_p),
        "mann_whitney_U": float(u.statistic),
        "mann_whitney_exact_p_reference_only": float(u.pvalue),
    }


def extreme(record: dict) -> dict:
    keys = ("episode", "init_state", "rollout_step", "denoising_step")
    return {**{k: record[k] for k in keys}, **{f: record[f] for f in FIELDS}}


def main() -> None:
    episodes = load_episodes()
    records = load_jsonl(OUT / RUN / TASK_DIR / "steering_diagnostics.jsonl")
    checks = validate(records, episodes)
    fingerprints = checks.pop("fingerprints")
    success = [e["success"] for e in episodes]

    summary: dict = {
        "episodes": NUM_EPISODES,
        "successes": sum(success),
        "rollout_steps": [e["steps"] for e in episodes],
        "success_vector": success,
        **checks,
        "phase1d_repo_replay": phase1d_replay(episodes, fingerprints),
        "distribution_per_application": {f: describe([r[f] for r in records]) for f in FIELDS},
    }

    by_step = {}
    for t in range(NUM_DENOISING_STEPS):
        rs = [r for r in records if r["denoising_step"] == t]
        by_step[t] = {f: describe([r[f] for r in rs]) for f in FIELDS}
    summary["by_denoising_step"] = {
        t: {
            "n": v["mean_delta_norm"]["n"],
            **{f"{f}_mean": v[f]["mean"] for f in FIELDS},
            "mean_delta_norm_std": v["mean_delta_norm"]["std"],
            "mean_delta_norm_min": v["mean_delta_norm"]["min"],
            "mean_delta_norm_max": v["mean_delta_norm"]["max"],
        }
        for t, v in by_step.items()
    }
    summary["active_layer"] = {
        "layer": LAYER,
        **{f"{f}_mean": float(np.mean([r[f] for r in records])) for f in FIELDS},
    }

    # Success vs failure. Failed episodes run to the 520-step timeout, so whole-episode means mix in
    # task phases that successful episodes never reach. The common window restricts every episode to
    # rollout steps all 15 episodes queried.
    common_window = min(max(r["rollout_step"] for r in records if r["episode"] == ep) for ep in range(NUM_EPISODES))
    comparison = {"common_window_max_rollout_step": common_window}
    per_episode = {}
    for f in ("mean_delta_norm", "mean_relative_delta", "mean_hidden_norm", "mean_cosine_delta_hidden"):
        whole = episode_means(records, f)
        window = episode_means(records, f, common_window)
        per_episode[f] = whole
        per_episode[f"{f}_window"] = window
        comparison[f] = {
            "whole_episode": group_compare(whole, success),
            "common_window": group_compare(window, success),
        }
    app_s = [r["mean_delta_norm"] for r in records if success[r["episode"]]]
    app_f = [r["mean_delta_norm"] for r in records if not success[r["episode"]]]
    comparison["application_weighted_mean_delta_norm"] = {
        "success": float(np.mean(app_s)),
        "success_n": len(app_s),
        "failure": float(np.mean(app_f)),
        "failure_n": len(app_f),
    }
    summary["success_vs_failure"] = comparison

    # Time course: 50-step rollout bins, split by outcome.
    bins = {}
    for lo in range(0, 520, 50):
        row = {}
        for label, flag in (("success", 1), ("failure", 0)):
            vals = [
                r["mean_delta_norm"]
                for r in records
                if success[r["episode"]] == flag and lo <= r["rollout_step"] < lo + 50
            ]
            row[label] = {"n_applications": len(vals), "mean_delta_norm": float(np.mean(vals)) if vals else None}
        bins[f"{lo}-{lo + 49}"] = row
    summary["rollout_bins_mean_delta_norm"] = bins

    # Extremes: single application, denoising step (averaged), rollout step (common window, all 15 episodes).
    largest = max(records, key=lambda r: r["mean_delta_norm"])
    smallest = min(records, key=lambda r: r["mean_delta_norm"])
    largest_token = max(records, key=lambda r: r["max_delta_norm"])
    step_means = {t: v["mean_delta_norm_mean"] for t, v in summary["by_denoising_step"].items()}
    rollout_means = {}
    for step in range(0, common_window + 1, 5):
        rollout_means[step] = float(np.mean([r["mean_delta_norm"] for r in records if r["rollout_step"] == step]))
    summary["extremes"] = {
        "largest_application": extreme(largest),
        "smallest_application": extreme(smallest),
        "largest_single_token_delta": extreme(largest_token),
        "largest_denoising_step": {
            "denoising_step": max(step_means, key=step_means.get),
            "mean": max(step_means.values()),
        },
        "smallest_denoising_step": {
            "denoising_step": min(step_means, key=step_means.get),
            "mean": min(step_means.values()),
        },
        "largest_rollout_step_common_window": {
            "rollout_step": max(rollout_means, key=rollout_means.get),
            "mean": max(rollout_means.values()),
        },
        "smallest_rollout_step_common_window": {
            "rollout_step": min(rollout_means, key=rollout_means.get),
            "mean": min(rollout_means.values()),
        },
    }
    # Effective scale: for the conceptor hook delta = beta (C - I) h. With cos(delta, h) ~ -1 the
    # intervention is ~ h' = (1 - s) h with s = relative delta.
    rel = summary["active_layer"]["mean_relative_delta_mean"]
    summary["effective_scale_if_pure_shrink"] = 1.0 - rel

    with (EXP / "phase1e_episode_intervention.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "episode",
                "init_state",
                "success",
                "rollout_steps",
                "inference_calls",
                "mean_delta_norm",
                "mean_relative_delta",
                "mean_hidden_norm",
                "mean_cosine_delta_hidden",
                "mean_delta_norm_window",
                "mean_relative_delta_window",
            ]
        )
        for e in episodes:
            ep = e["episode"]
            writer.writerow(
                [
                    ep,
                    e["init_state"],
                    e["success"],
                    e["steps"],
                    len({r["rollout_step"] for r in records if r["episode"] == ep}),
                    f"{per_episode['mean_delta_norm'][ep]:.5f}",
                    f"{per_episode['mean_relative_delta'][ep]:.6f}",
                    f"{per_episode['mean_hidden_norm'][ep]:.4f}",
                    f"{per_episode['mean_cosine_delta_hidden'][ep]:.6f}",
                    f"{per_episode['mean_delta_norm_window'][ep]:.5f}",
                    f"{per_episode['mean_relative_delta_window'][ep]:.6f}",
                ]
            )
    with (EXP / "phase1e_denoising_step_intervention.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        cols = [f"{fld}_mean" for fld in FIELDS]
        writer.writerow(
            ["denoising_step", "n", *cols, "mean_delta_norm_std", "mean_delta_norm_min", "mean_delta_norm_max"]
        )
        for t, v in summary["by_denoising_step"].items():
            writer.writerow(
                [
                    t,
                    v["n"],
                    *(f"{v[c]:.6f}" for c in cols),
                    f"{v['mean_delta_norm_std']:.5f}",
                    f"{v['mean_delta_norm_min']:.5f}",
                    f"{v['mean_delta_norm_max']:.5f}",
                ]
            )

    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
