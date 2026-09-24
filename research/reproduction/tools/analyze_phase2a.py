"""Phase 2A analysis: baseline vs COAST (global, V0) vs pure shrinkage (M = 0.9 I), paired noise seed 100.

Inputs (gitignored run outputs under examples/libero_env/output/phase2a_{baseline,coast,shrinkage}_task02_seed15/):
  client.log (per-episode success, CLIENT_EXIT=0), episode_NNN.mp4 (rollout length = frames - 10),
  noise_fingerprints.jsonl, steering_diagnostics.jsonl (COAST and shrinkage only).

The script
  1. recomputes every noise fingerprint and checks pairing across the three conditions,
  2. validates the diagnostics logs (10 records per call, steps 0..9, layer 5, finite),
  3. replays A against Phase 1D (baseline, noise-floor seed 100) and B against Phase 1D repo / Phase 1E,
  4. writes research/reproduction/experiments/phase2a_episode_results.csv and prints a JSON summary.

Descriptive only. Usage (repo root, CPU only):
    CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu uv run python research/reproduction/tools/analyze_phase2a.py
"""

from __future__ import annotations

import collections
import csv
import itertools
import json
import math
import pathlib
import sys

import imageio.v2 as iio
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from analyze_phase1d import _EPISODE_RE
from analyze_phase1d import EXP
from analyze_phase1d import INIT_STATES
from analyze_phase1d import NUM_STEPS_WAIT
from analyze_phase1d import OUT
from analyze_phase1d import TASK_DIR
from analyze_phase1d import load_episodes
from analyze_phase1d import load_fingerprints
from analyze_phase1d import pair_stats
from analyze_phase1d import wilson

from openpi.serving.noise_control import derive_noise
from openpi.serving.noise_control import noise_fingerprint

MASTER_SEED = 100
CONDITIONS = {
    "baseline": "phase2a_baseline_task02_seed15",
    "coast": "phase2a_coast_task02_seed15",
    "shrinkage": "phase2a_shrinkage_task02_seed15",
}
STEERED = ("coast", "shrinkage")
REFERENCE_RUNS = {  # earlier runs with the same configuration and master seed 100
    "phase1d_baseline": ("baseline", "phase1d_baseline_task02_seed15"),
    "phase1d_noisefloor_seed100": ("baseline", "phase1d_noisefloor_seed100_task02_seed15"),
    "phase1d_repo": ("coast", "phase1d_repo_task02_seed15"),
    "phase1e_repo_diag": ("coast", "phase1e_repo_diag_task02_seed15"),
}
FIELDS = (
    "mean_delta_norm",
    "max_delta_norm",
    "mean_relative_delta",
    "mean_hidden_norm",
    "mean_cosine_delta_hidden",
    "mean_norm_ratio",
)
EPISODE_FIELDS = ("mean_delta_norm", "mean_relative_delta", "mean_hidden_norm", "mean_norm_ratio")


def load_diagnostics(name: str) -> list[dict]:
    path = OUT / CONDITIONS[name] / TASK_DIR / "steering_diagnostics.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()]


def verify_noise(fps: dict[str, list[dict]], episodes: dict[str, list[dict]]) -> dict:
    recomputed = 0
    for name, records in fps.items():
        by_ep = collections.defaultdict(list)
        for r in records:
            key = {k: r[k] for k in ("master_seed", "task_id", "init_state", "rollout_step")}
            assert key["master_seed"] == MASTER_SEED
            assert key["task_id"] == 2
            assert key["init_state"] == INIT_STATES[r["episode"]]
            assert noise_fingerprint(derive_noise(key, 10, 32)) == r["sha256"], f"{name}: bad fingerprint {key}"
            recomputed += 1
            by_ep[r["episode"]].append(r["rollout_step"])
        for ep, steps in by_ep.items():
            assert steps == list(range(0, episodes[name][ep]["steps"], 5)), f"{name} ep {ep}: schedule"
    coords = {name: {(r["init_state"], r["rollout_step"]): r["sha256"] for r in recs} for name, recs in fps.items()}
    pairs = {}
    for a, b in itertools.combinations(coords, 2):
        shared = coords[a].keys() & coords[b].keys()
        identical = sum(coords[a][c] == coords[b][c] for c in shared)
        assert identical == len(shared), f"{a} vs {b}: unpaired noise"
        pairs[f"{a}_vs_{b}"] = {"shared_coordinates": len(shared), "identical": identical}
    return {
        "recomputed_and_matched": recomputed,
        "requests": {n: len(r) for n, r in fps.items()},
        "pairs": pairs,
        "sample_init15_step0": {n: coords[n][(15, 0)][:16] for n in coords},
    }


def validate_diagnostics(name: str, records: list[dict], fps: list[dict]) -> None:
    calls = collections.defaultdict(list)
    for r in records:
        calls[(r["episode"], r["rollout_step"])].append(r)
        assert (r["layer"], r["token_count"], r["hidden_dim"], r["beta"]) == (5, 10, 1024, 0.1)
        assert r["strategy"] == {"coast": "global", "shrinkage": "shrinkage"}[name]
        assert all(math.isfinite(r[f]) for f in FIELDS)
    for rs in calls.values():
        assert [r["denoising_step"] for r in rs] == list(range(10))
    assert sorted(calls) == sorted((f["episode"], f["rollout_step"]) for f in fps), f"{name}: schedule mismatch"


def replay(episodes: dict[str, list[dict]]) -> dict:
    out = {}
    for ref, (cond, run_dir) in REFERENCE_RUNS.items():
        if not (OUT / run_dir / "client.log").exists():
            out[ref] = "missing"
            continue
        try:
            ref_eps = load_episodes(run_dir)
        except AssertionError:  # Phase 1E's client log marks a clean exit with "exit=0"
            ref_eps = _load_episodes_exit_marker(run_dir)
        stats = pair_stats(ref_eps, episodes[cond])
        out[f"{ref}_vs_{cond}"] = {
            "identical_outcomes": stats["agreement"],
            "identical_rollout_lengths": stats["identical_rollout_length"],
            "differing_lengths": [
                {"init_state": INIT_STATES[i], "reference": a["steps"], "phase2a": b["steps"]}
                for i, (a, b) in enumerate(zip(ref_eps, episodes[cond], strict=True))
                if a["steps"] != b["steps"]
            ],
        }
    return out


def _load_episodes_exit_marker(run_dir: str) -> list[dict]:
    """Same as analyze_phase1d.load_episodes, for Phase 1E's client log (clean exit marked "exit=0")."""
    text = (OUT / run_dir / "client.log").read_text()
    assert "exit=0" in text, f"{run_dir}: client did not exit cleanly"
    successes = {int(m.group(1)) - 1: m.group(2) == "True" for m in _EPISODE_RE.finditer(text)}
    assert sorted(successes) == list(range(15)), f"{run_dir}: missing episodes"
    eps = []
    for ep in range(15):
        with iio.get_reader(OUT / run_dir / TASK_DIR / f"episode_{ep:03d}.mp4") as reader:
            steps = reader.count_frames() - NUM_STEPS_WAIT
        eps.append({"success": int(successes[ep]), "steps": steps})
    return eps


def episode_means(records: list[dict], field: str) -> list[float]:
    by_ep = collections.defaultdict(list)
    for r in records:
        by_ep[r["episode"]].append(r[field])
    return [float(np.mean(by_ep[ep])) for ep in range(15)]


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
        "all_three_success": sum(all(successes[n][i] for n in CONDITIONS) for i in range(15)),
        "all_three_failure": sum(not any(successes[n][i] for n in CONDITIONS) for i in range(15)),
        "replay": replay(episodes),
        "counts": {
            name: {"inference_calls": len(fps[name]), "hook_applications": len(diags.get(name, []))}
            for name in CONDITIONS
        },
    }

    # Intervention magnitude and hidden-norm change: per application, per denoising step, per episode.
    summary["per_application"] = {
        name: {
            f: {
                "mean": float(np.mean([r[f] for r in diags[name]])),
                "std": float(np.std([r[f] for r in diags[name]], ddof=1)),
                "min": float(np.min([r[f] for r in diags[name]])),
                "max": float(np.max([r[f] for r in diags[name]])),
            }
            for f in FIELDS
        }
        for name in STEERED
    }
    summary["by_denoising_step"] = {
        name: {
            f: [float(np.mean([r[f] for r in diags[name] if r["denoising_step"] == t])) for t in range(10)]
            for f in FIELDS
        }
        for name in STEERED
    }
    per_episode = {name: {f: episode_means(diags[name], f) for f in EPISODE_FIELDS} for name in STEERED}

    # Paired comparison at rollout_step 0: identical observation and identical noise in both conditions,
    # so h at denoising step 0 is identical; differences at later denoising steps reflect only the
    # within-call effect of the conceptor term (plus the bf16 magnitude mismatch).
    key = lambda r: (r["episode"], r["rollout_step"], r["denoising_step"])  # noqa: E731
    coast_by = {key(r): r for r in diags["coast"]}
    shrink_by = {key(r): r for r in diags["shrinkage"]}
    step0 = [k for k in coast_by if k[1] == 0]
    summary["paired_rollout_step0"] = {
        "n_applications": len(step0),
        "hidden_norm_identical_at_denoising_step0": sum(
            coast_by[k]["mean_hidden_norm"] == shrink_by[k]["mean_hidden_norm"] for k in step0 if k[2] == 0
        ),
        "by_denoising_step": {
            t: {
                "coast_mean_delta_norm": float(np.mean([coast_by[k]["mean_delta_norm"] for k in step0 if k[2] == t])),
                "shrink_mean_delta_norm": float(np.mean([shrink_by[k]["mean_delta_norm"] for k in step0 if k[2] == t])),
                "coast_norm_ratio": float(np.mean([coast_by[k]["mean_norm_ratio"] for k in step0 if k[2] == t])),
                "shrink_norm_ratio": float(np.mean([shrink_by[k]["mean_norm_ratio"] for k in step0 if k[2] == t])),
                "hidden_norm_rel_diff": float(
                    np.mean(
                        [
                            (coast_by[k]["mean_hidden_norm"] - shrink_by[k]["mean_hidden_norm"])
                            / shrink_by[k]["mean_hidden_norm"]
                            for k in step0
                            if k[2] == t
                        ]
                    )
                ),
            }
            for t in range(10)
        },
    }
    # All shared coordinates (trajectories may have diverged): per-application paired differences.
    shared = sorted(coast_by.keys() & shrink_by.keys())
    summary["paired_shared_coordinates"] = {
        "n_applications": len(shared),
        "coast_minus_shrink_mean_delta_norm": float(
            np.mean([coast_by[k]["mean_delta_norm"] - shrink_by[k]["mean_delta_norm"] for k in shared])
        ),
        "coast_minus_shrink_norm_ratio": float(
            np.mean([coast_by[k]["mean_norm_ratio"] - shrink_by[k]["mean_norm_ratio"] for k in shared])
        ),
        "coast_over_shrink_delta_norm_ratio": float(
            np.mean([coast_by[k]["mean_delta_norm"] / shrink_by[k]["mean_delta_norm"] for k in shared])
        ),
    }

    with (EXP / "phase2a_episode_results.csv").open("w", newline="") as f:
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
