"""Phase 1D: analyze the paired-noise LIBERO rollouts (development states 15-29 only).

Reads the LIBERO client outputs under examples/libero_env/output/phase1d_*:
  - client.log                   per-episode success
  - episode_NNN.mp4              rollout length (= frames - num_steps_wait, same convention as Phase 1C)
  - noise_fingerprints.jsonl     per-request noise key + server-echoed SHA-256

and
  1. independently re-derives the noise for every logged key and checks the echoed SHA-256,
  2. checks that matching (task_id, init_state, rollout_step) coordinates carry identical
     fingerprints across conditions sharing a master seed (and differ across seeds),
  3. writes the per-episode CSV and prints a JSON summary (counts, transitions, agreement,
     Phase 1C comparison).

Usage (repo root, CPU only):
    CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu uv run python research/reproduction/tools/analyze_phase1d.py
"""

from __future__ import annotations

import csv
import itertools
import json
import math
import pathlib
import re

import imageio.v2 as iio

from openpi.serving.noise_control import derive_noise
from openpi.serving.noise_control import noise_fingerprint

ROOT = pathlib.Path(__file__).resolve().parents[3]
OUT = ROOT / "examples/libero_env/output"
TASK_DIR = "02-kitchen-scene3-turn-on-the-stove-and-put-the-moka-pot-on-it"
EXP = ROOT / "research/reproduction/experiments"
NUM_STEPS_WAIT = 10
INIT_STATES = list(range(15, 30))
CONDITIONS = {  # name -> (output dir, master noise seed)
    "baseline": ("phase1d_baseline_task02_seed15", 100),
    "repo": ("phase1d_repo_task02_seed15", 100),
    "paper": ("phase1d_paper_task02_seed15", 100),
    "nf100": ("phase1d_noisefloor_seed100_task02_seed15", 100),
    "nf200": ("phase1d_noisefloor_seed200_task02_seed15", 200),
    "nf300": ("phase1d_noisefloor_seed300_task02_seed15", 300),
}
PHASE1C = {
    "baseline": "phase1c_baseline_task02_seed15",
    "repo": "phase1c_repo_task02_seed15",
    "paper": "phase1c_paper_task02_seed15",
}
_EPISODE_RE = re.compile(r"Episode (\d+)/15: success=(True|False)")


def load_episodes(out_dir: str) -> list[dict]:
    log = (OUT / out_dir / "client.log").read_text()
    assert "CLIENT_EXIT=0" in log, f"{out_dir}: client did not exit cleanly"
    successes = {int(m.group(1)) - 1: m.group(2) == "True" for m in _EPISODE_RE.finditer(log)}
    assert sorted(successes) == list(range(15)), f"{out_dir}: missing episodes"
    episodes = []
    for ep in range(15):
        with iio.get_reader(OUT / out_dir / TASK_DIR / f"episode_{ep:03d}.mp4") as reader:
            steps = reader.count_frames() - NUM_STEPS_WAIT
        episodes.append(
            {"success": int(successes[ep]), "steps": steps, "steps_to_success": steps - 1 if successes[ep] else None}
        )
    return episodes


def load_fingerprints(out_dir: str) -> list[dict]:
    path = OUT / out_dir / TASK_DIR / "noise_fingerprints.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()]


def wilson(k: int, n: int, z: float = 1.959964) -> list[float]:
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [round(centre - half, 3), round(centre + half, 3)]


def mcnemar_exact(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2**n
    return round(min(1.0, 2 * tail), 3)


def pair_stats(x: list[dict], y: list[dict]) -> dict:
    gains = sum(1 for a, b in zip(x, y, strict=True) if not a["success"] and b["success"])
    losses = sum(1 for a, b in zip(x, y, strict=True) if a["success"] and not b["success"])
    identical_steps = sum(1 for a, b in zip(x, y, strict=True) if a["steps"] == b["steps"])
    return {
        "fail_to_success": gains,
        "success_to_fail": losses,
        "agreement": 15 - gains - losses,
        "identical_rollout_length": identical_steps,
        "mcnemar_exact_p_reference_only": mcnemar_exact(gains, losses),
    }


def verify_fingerprints(fps: dict[str, list[dict]], episodes: dict[str, list[dict]]) -> dict:
    report: dict = {}
    # (1) independent recomputation of every logged fingerprint, and per-episode request schedule
    recomputed = 0
    for name, records in fps.items():
        seed = CONDITIONS[name][1]
        by_ep: dict[int, list[int]] = {}
        for r in records:
            key = {k: r[k] for k in ("master_seed", "task_id", "init_state", "rollout_step")}
            assert key["master_seed"] == seed
            assert key["task_id"] == 2
            assert key["init_state"] == INIT_STATES[r["episode"]], f"{name}: episode/init_state mismatch"
            assert noise_fingerprint(derive_noise(key, 10, 32)) == r["sha256"], f"{name}: bad fingerprint {key}"
            recomputed += 1
            by_ep.setdefault(r["episode"], []).append(r["rollout_step"])
        for ep, steps in by_ep.items():
            n_steps = episodes[name][ep]["steps"]
            assert steps == list(range(0, n_steps, 5)), f"{name} ep {ep}: unexpected request schedule"
        report[name] = {"requests": len(records)}
    report["recomputed_and_matched"] = recomputed

    # (2) cross-condition pairing at matching coordinates
    coords = {name: {(r["init_state"], r["rollout_step"]): r["sha256"] for r in recs} for name, recs in fps.items()}
    pairs = {}
    for a, b in itertools.combinations(coords, 2):
        shared = coords[a].keys() & coords[b].keys()
        equal = sum(1 for c in shared if coords[a][c] == coords[b][c])
        same_seed = CONDITIONS[a][1] == CONDITIONS[b][1]
        pairs[f"{a}_vs_{b}"] = {"same_master_seed": same_seed, "shared_coordinates": len(shared), "identical": equal}
        if same_seed:
            assert equal == len(shared), f"{a} vs {b}: unpaired noise at a shared coordinate"
        else:
            assert equal == 0, f"{a} vs {b}: different seeds produced identical noise"
    report["pairs"] = pairs
    # Sample of coordinates for the written record.
    sample_coords = [(15, 0), (15, 5), (22, 100), (29, 200)]
    report["sample"] = {
        f"init{s}_step{t}": {name: coords[name].get((s, t), "n/a")[:16] for name in coords} for s, t in sample_coords
    }
    return report


def main() -> None:
    episodes = {name: load_episodes(d) for name, (d, _) in CONDITIONS.items()}
    fps = {name: load_fingerprints(d) for name, (d, _) in CONDITIONS.items()}
    fp_report = verify_fingerprints(fps, episodes)

    # Phase 1C: re-derive rollout lengths with the same video convention and check against the committed CSV.
    p1c = {name: load_episodes(d) for name, d in PHASE1C.items()}
    with open(EXP / "phase1c_episode_results.csv") as f:
        p1c_csv = list(csv.DictReader(f))
    for name in PHASE1C:
        for row, ep in zip(p1c_csv, p1c[name], strict=True):
            assert int(row[f"{name}_success"]) == ep["success"]
            assert int(row[f"{name}_rollout_steps"]) == ep["steps"]

    main_names = ["baseline", "repo", "paper"]
    nf_names = ["nf100", "nf200", "nf300"]
    with open(EXP / "phase1d_episode_results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        header = ["episode", "init_state"]
        for name in main_names + nf_names:
            header += [f"{name}_success", f"{name}_rollout_steps", f"{name}_steps_to_success"]
        writer.writerow(header)
        for ep, state in enumerate(INIT_STATES):
            row = [ep, state]
            for name in main_names + nf_names:
                e = episodes[name][ep]
                row += [e["success"], e["steps"], "" if e["steps_to_success"] is None else e["steps_to_success"]]
            writer.writerow(row)

    def summary(eps: list[dict]) -> dict:
        k = sum(e["success"] for e in eps)
        tts = sorted(e["steps_to_success"] for e in eps if e["success"])
        return {
            "successes": k,
            "rate": round(k / 15, 3),
            "wilson95": wilson(k, 15),
            "success_vector": [e["success"] for e in eps],
            "rollout_steps": [e["steps"] for e in eps],
            "time_to_success_median": (tts[len(tts) // 2] + tts[(len(tts) - 1) // 2]) / 2 if tts else None,
        }

    def pairwise(group: dict[str, list[dict]]) -> dict:
        return {f"{a}_vs_{b}": pair_stats(group[a], group[b]) for a, b in itertools.combinations(group, 2)}

    result = {
        "phase1d": {name: summary(episodes[name]) for name in main_names + nf_names},
        "phase1d_pairs": pairwise({n: episodes[n] for n in main_names}),
        "noise_floor_pairs": pairwise({n: episodes[n] for n in nf_names}),
        "replicate_baseline_vs_nf100": pair_stats(episodes["baseline"], episodes["nf100"]),
        "phase1c": {name: summary(p1c[name]) for name in main_names},
        "phase1c_pairs": pairwise(p1c),
        "phase1c_vs_phase1d_same_condition": {n: pair_stats(p1c[n], episodes[n]) for n in main_names},
        "all_three_success_1d": sum(all(episodes[n][i]["success"] for n in main_names) for i in range(15)),
        "all_three_fail_1d": sum(not any(episodes[n][i]["success"] for n in main_names) for i in range(15)),
        "fingerprints": fp_report,
    }
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
