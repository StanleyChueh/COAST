"""Phase 7B analysis: paired mechanism decomposition (analysis only; no rollouts). Plan: phase7b_preregistration.md.

Reads the run tree written by the Phase 7B queue runner:
    <runs>/<TASK_KEY>/rep<r>/{provenance.json, summary.json, driver.log, server_noise_log.jsonl,
                              <CONDITION>/{client.log, client_noise_log.jsonl, server_noise_log.jsonl,
                                           NN-<task>/episode_XXX.mp4}}
and the queue logs <logs>/queue_gpu<g>.log (GPU/port provenance).

Per episode: success from the client's own "Episode k/N: success=..." line; init_state from the
"phase7b_state_map" lines; rollout_steps = env_steps - 10 from "phase7b_episode_end" (cross-checked against video
frames - 10, as in Phase 7A).

Noise verification (all asserted):
  - every client-logged and server-logged fingerprint is recomputed independently from its key;
  - client and server logs carry the same request set per condition; sampler path matches the condition;
  - per episode the schedule is rollout_step = 0, 5, ... < rollout_steps and init_state = 15 + episode;
  - within a (task, repeat), all four conditions carry identical noise at every shared (init_state, rollout_step);
  - no fingerprint is shared between the two repeats.

Writes experiments/phase7b_episode_results.csv, phase7b_summary.csv, phase7b_paired_tables.csv and
phase7b_analysis.json, and prints the analysis.

Usage (repo root, CPU only):
    .venv/bin/python research/reproduction/tools/analyze_phase7b.py <runs_dir> <logs_dir>
"""

from __future__ import annotations

import csv
import hashlib
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
from analyze_phase1d import wilson

from openpi.serving.noise_control import derive_noise
from openpi.serving.noise_control import noise_fingerprint

NUM_STEPS_WAIT = 10
MAX_STEPS = 520
REPLAN = 5
OFFSET, N_STATES = 15, 30
CANDIDATE_SHA = "29059a537ce123abdab7a4e164db0c72eeff8024"
CHECKPOINT = "checkpoints/openpi-libero-2000 (HF brandonyang/openpi-libero-2000 rev aaeeabc72f8a)"
TASKS = {  # key -> (task_id, layer, alpha, beta)
    "LR2a": (0, 11, 0.5, 0.3),
    "KS4": (3, 5, 1.0, 0.1),
    "LR1": (7, 11, 0.5, 0.3),
}
CONDITIONS = ("BASELINE", "ORIGINAL_COAST", "PURE_SHRINKAGE", "RANDOM_CONTROL")
REPEATS = {1: 100, 2: 200}  # repeat -> master seed (preregistered)
COMPARISONS = ("PURE_SHRINKAGE", "RANDOM_CONTROL", "BASELINE")  # ORIGINAL_COAST vs each; first is primary
N_BOOT, BOOT_SEED = 10_000, 7070

_EP_RE = re.compile(r"Episode (\d+)/(\d+): success=(True|False)")
_MAP_RE = re.compile(r"phase7b_state_map episode=(\d+) init_state=(\d+) sha1=([0-9a-f]+)")
_END_RE = re.compile(r"phase7b_episode_end episode=(\d+) init_state=(\d+) env_steps=(\d+) requests=(\d+)")
_START_RE = re.compile(r"START gpu=(\d+) port=(\d+) task=(\S+) rep=(\d+) seed=(\d+) out=(\S+)")


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar: 2 * P[X <= min(b, c)], X ~ Bin(b + c, 1/2), capped at 1."""
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2**n)


def load_jsonl(path: pathlib.Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load_condition(cond_dir: pathlib.Path) -> dict:
    log = (cond_dir / "client.log").read_text()
    succ = {int(m.group(1)) - 1: m.group(3) == "True" for m in _EP_RE.finditer(log)}
    assert sorted(succ) == list(range(N_STATES)), f"{cond_dir}: episodes {sorted(succ)}"
    smap = {int(m.group(1)): (int(m.group(2)), m.group(3)) for m in _MAP_RE.finditer(log)}
    ends = {int(m.group(1)): (int(m.group(2)), int(m.group(3)), int(m.group(4))) for m in _END_RE.finditer(log)}
    assert sorted(smap) == list(range(N_STATES)), cond_dir
    assert sorted(ends) == list(range(N_STATES)), cond_dir
    video_dir = next(d for d in cond_dir.iterdir() if d.is_dir())
    eps = []
    for ep in range(N_STATES):
        init_state, env_steps, requests = ends[ep]
        assert smap[ep][0] == init_state == OFFSET + ep, (cond_dir, ep)
        with iio.get_reader(video_dir / f"episode_{ep:03d}.mp4") as r:
            frames = r.count_frames()
        assert frames == env_steps, (cond_dir, ep, frames, env_steps)
        steps = env_steps - NUM_STEPS_WAIT
        assert steps <= MAX_STEPS, (cond_dir, ep, steps)
        assert succ[ep] or steps == MAX_STEPS, f"{cond_dir} ep {ep}: failure before timeout ({steps})"
        assert requests == math.ceil(steps / REPLAN), (cond_dir, ep, requests, steps)
        eps.append(
            {
                "episode": ep,
                "init_state": init_state,
                "success": int(succ[ep]),
                "rollout_steps": steps,
                "sha1": smap[ep][1],
            }
        )
    return {
        "episodes": eps,
        "client_noise": load_jsonl(cond_dir / "client_noise_log.jsonl"),
        "server_noise": load_jsonl(cond_dir / "server_noise_log.jsonl"),
        "log_rate": float(np.mean([e["success"] for e in eps])),
        "video_dir": str(video_dir),
    }


def verify_noise(data: dict) -> dict:
    report: dict = {"recomputed": 0, "pairs": {}, "cross_repeat_shared": None, "samples": []}
    cache: dict[tuple, str] = {}

    def fp(key: tuple) -> str:
        if key not in cache:
            payload = dict(zip(("master_seed", "task_id", "init_state", "rollout_step"), key, strict=True))
            cache[key] = noise_fingerprint(derive_noise(payload, 10, 32))
        return cache[key]

    by_rep_sha: dict[int, set] = {1: set(), 2: set()}
    coord: dict = {}
    for (task, rep, cond), d in data.items():
        task_id, seed = TASKS[task][0], REPEATS[rep]
        cl, sv = d["client_noise"], d["server_noise"]
        assert len(cl) == len(sv), (task, rep, cond)
        expect_path = "sample_actions" if cond == "BASELINE" else "sample_actions_with_steering"
        by_ep: dict[int, list[int]] = {}
        for c, s in zip(cl, sv, strict=True):
            key = (c["master_seed"], c["task_id"], c["init_state"], c["rollout_step"])
            assert key == (s["master_seed"], s["task_id"], s["init_state"], s["rollout_step"]), (task, rep, cond)
            assert key[:2] == (seed, task_id), (task, rep, cond, key)
            ref = fp(key)
            assert c["sha256"] == s["sha256"] == s["sampler_sha256"] == ref, (task, rep, cond, key)
            assert c["sampler"] == s["sampler"] == expect_path, (task, rep, cond)
            assert c["condition"] == s["condition"] == cond, (task, rep, cond)
            assert c["init_state"] == OFFSET + c["episode"]
            report["recomputed"] += 2
            by_ep.setdefault(c["episode"], []).append(c["rollout_step"])
            by_rep_sha[rep].add(ref)
            coord.setdefault((task, rep), {}).setdefault(cond, {})[(key[2], key[3])] = ref
        for e in d["episodes"]:
            assert by_ep[e["episode"]] == list(range(0, e["rollout_steps"], REPLAN)), (task, rep, cond, e["episode"])
    for (task, rep), per_cond in sorted(coord.items()):
        for a, b in itertools.combinations(CONDITIONS, 2):
            shared = set(per_cond[a]) & set(per_cond[b])
            same = sum(per_cond[a][k] == per_cond[b][k] for k in shared)
            assert same == len(shared), (task, rep, a, b)
            report["pairs"][f"{task}/rep{rep}/{a}~{b}"] = [len(shared), same]
    report["cross_repeat_shared"] = len(by_rep_sha[1] & by_rep_sha[2])
    assert report["cross_repeat_shared"] == 0
    for task in TASKS:
        for state, step in ((15, 0), (22, 100), (29, 200), (44, 50)):
            row = {"task": task, "init_state": state, "rollout_step": step}
            for rep in REPEATS:
                for cond in CONDITIONS:
                    row[f"rep{rep}_{cond}"] = coord[(task, rep)][cond].get((state, step), "")[:16]
            report["samples"].append(row)
    report["unique_keys"] = len(cache)
    return report


def paired(x: list[int], y: list[int]) -> dict:
    both = sum(1 for a, b in zip(x, y, strict=True) if a and b)
    b_ = sum(1 for a, b in zip(x, y, strict=True) if a and not b)
    c_ = sum(1 for a, b in zip(x, y, strict=True) if not a and b)
    neither = sum(1 for a, b in zip(x, y, strict=True) if not a and not b)
    return {
        "n": len(x),
        "both_succeed": both,
        "coast_only_b": b_,
        "other_only_c": c_,
        "both_fail": neither,
        "rd": (b_ - c_) / len(x),
        "mcnemar_p": mcnemar_exact(b_, c_),
    }


def main() -> None:
    runs, logs = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
    gpu_port = {}
    for qlog in sorted(logs.glob("queue_gpu*.log")):
        for m in _START_RE.finditer(qlog.read_text()):
            gpu_port[(m.group(3), int(m.group(4)))] = (int(m.group(1)), int(m.group(2)), int(m.group(5)), m.group(6))
    data, prov = {}, {}
    for task in TASKS:
        for rep, seed in REPEATS.items():
            rdir = runs / task / f"rep{rep}"
            assert (rdir / "DONE").exists(), rdir
            prov[(task, rep)] = json.loads((rdir / "provenance.json").read_text())
            summ = {
                c["condition"]: c["success_rate"] for c in json.loads((rdir / "summary.json").read_text())["conditions"]
            }
            assert list(summ) == list(CONDITIONS)
            for cond in CONDITIONS:
                d = load_condition(rdir / cond)
                assert round(d["log_rate"], 2) == summ[cond], (task, rep, cond)  # client prints %.2f
                data[(task, rep, cond)] = d
            p = prov[(task, rep)]
            assert (p["master_seed"], p["state_offset"], p["num_episodes"]) == (seed, OFFSET, N_STATES), (task, rep)
            assert (p["max_steps"], p["torch_compile_disable"]) == (None, "1"), (task, rep)
            assert (p["layer"], p["alpha"], p["beta"]) == TASKS[task][1:]
            assert gpu_port[(task, rep)][2] == seed
            # the same state content in every condition of a (task, repeat)
            sha1s = {tuple(e["sha1"] for e in data[(task, rep, c)]["episodes"]) for c in CONDITIONS}
            assert len(sha1s) == 1, (task, rep)

    # state content identical across repeats too
    for task in TASKS:
        assert data[(task, 1, "BASELINE")]["episodes"][0]["sha1"] == data[(task, 2, "BASELINE")]["episodes"][0]["sha1"]

    noise = verify_noise(data)

    # ---------------- episode CSV ----------------
    rows = []
    for task, (task_id, layer, alpha, beta) in TASKS.items():
        for rep, seed in REPEATS.items():
            gpu, port, _, _ = gpu_port[(task, rep)]
            for cond in CONDITIONS:
                d = data[(task, rep, cond)]
                first = {c["episode"]: c["sha256"] for c in d["client_noise"] if c["rollout_step"] == 0}
                rows.extend(
                    {
                        "task": task,
                        "task_id": task_id,
                        "condition": cond,
                        "repeat": rep,
                        "master_seed": seed,
                        "episode": e["episode"],
                        "init_state": e["init_state"],
                        "success": e["success"],
                        "rollout_steps": e["rollout_steps"],
                        "layer": layer,
                        "alpha": alpha,
                        "beta": beta,
                        "noise_key": f"({seed},{task_id},{e['init_state']},rollout_step)",
                        "noise_verification_hash": first[e["episode"]],
                        "checkpoint": CHECKPOINT,
                        "candidate_sha": CANDIDATE_SHA,
                        "gpu": gpu,
                        "port": port,
                        "output_dir": str(runs / task / f"rep{rep}" / cond),
                    }
                    for e in d["episodes"]
                )
    assert len(rows) == 720
    ep_csv = EXP / "phase7b_episode_results.csv"
    with open(ep_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    succ = {
        (t, r, c): [e["success"] for e in data[(t, r, c)]["episodes"]]
        for t in TASKS
        for r in REPEATS
        for c in CONDITIONS
    }
    steps = {
        (t, r, c): [e["rollout_steps"] for e in data[(t, r, c)]["episodes"]]
        for t in TASKS
        for r in REPEATS
        for c in CONDITIONS
    }

    # ---------------- summary CSV ----------------
    summary_rows = []
    for task in TASKS:
        for cond in CONDITIONS:
            r1, r2 = sum(succ[(task, 1, cond)]), sum(succ[(task, 2, cond)])
            per_state = [a + b for a, b in zip(succ[(task, 1, cond)], succ[(task, 2, cond)], strict=True)]
            lo, hi = wilson(r1 + r2, 60)
            summary_rows.append(
                {
                    "task": task,
                    "condition": cond,
                    "rep1_successes": r1,
                    "rep2_successes": r2,
                    "pooled_successes": r1 + r2,
                    "pooled_n": 60,
                    "pooled_rate": round((r1 + r2) / 60, 4),
                    "wilson95_lo_descriptive": lo,
                    "wilson95_hi_descriptive": hi,
                    "states_2of2": per_state.count(2),
                    "states_1of2": per_state.count(1),
                    "states_0of2": per_state.count(0),
                }
            )
    with open(EXP / "phase7b_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0]))
        w.writeheader()
        w.writerows(summary_rows)

    # ---------------- paired tables + bootstrap ----------------
    rng = np.random.default_rng(BOOT_SEED)
    paired_rows, boot_draws, results = [], {}, {}
    for task in TASKS:
        for other in COMPARISONS:
            tabs = {}
            for rep in REPEATS:
                tabs[rep] = paired(succ[(task, rep, "ORIGINAL_COAST")], succ[(task, rep, other)])
            pooled = paired(
                succ[(task, 1, "ORIGINAL_COAST")] + succ[(task, 2, "ORIGINAL_COAST")],
                succ[(task, 1, other)] + succ[(task, 2, other)],
            )
            per_state = (
                np.array(
                    [
                        (succ[(task, 1, "ORIGINAL_COAST")][s] - succ[(task, 1, other)][s])
                        + (succ[(task, 2, "ORIGINAL_COAST")][s] - succ[(task, 2, other)][s])
                        for s in range(N_STATES)
                    ],
                    dtype=float,
                )
                / 2.0
            )
            idx = rng.integers(0, N_STATES, size=(N_BOOT, N_STATES))
            draws = per_state[idx].mean(axis=1)
            boot_draws[(task, other)] = draws
            lo, hi = np.percentile(draws, [2.5, 97.5])
            rd1, rd2 = tabs[1]["rd"], tabs[2]["rd"]
            ci_excludes_0 = bool(lo > 0 or hi < 0)
            same_sign = (rd1 > 0 and rd2 > 0) or (rd1 < 0 and rd2 < 0)
            mcn_rep = (
                tabs[1]["mcnemar_p"] < 0.05
                and tabs[2]["mcnemar_p"] < 0.05
                and np.sign(rd1) == np.sign(rd2)
                and rd1 != 0
            )
            clear = (ci_excludes_0 and same_sign) or mcn_rep
            direction = int(np.sign(pooled["rd"])) if clear else 0
            identical_len = {
                rep: sum(
                    a == b for a, b in zip(steps[(task, rep, "ORIGINAL_COAST")], steps[(task, rep, other)], strict=True)
                )
                for rep in REPEATS
            }
            results[(task, other)] = {
                "rep1": tabs[1],
                "rep2": tabs[2],
                "pooled_descriptive": pooled,
                "pooled_rd": pooled["rd"],
                "boot95": [float(lo), float(hi)],
                "ci_excludes_0": ci_excludes_0,
                "same_nonzero_sign_both_repeats": same_sign,
                "repeatable_mcnemar": bool(mcn_rep),
                "clear_difference": bool(clear),
                "clear_direction": direction,
                "identical_rollout_length": identical_len,
            }
            for label, t in (("rep1", tabs[1]), ("rep2", tabs[2]), ("pooled_descriptive", pooled)):
                paired_rows.append(
                    {
                        "task": task,
                        "comparison": f"ORIGINAL_COAST_vs_{other}",
                        "table": label,
                        **{k: (round(v, 6) if isinstance(v, float) else v) for k, v in t.items()},
                        "state_cluster_boot95_lo": round(float(lo), 4) if label == "pooled_descriptive" else "",
                        "state_cluster_boot95_hi": round(float(hi), 4) if label == "pooled_descriptive" else "",
                        "clear_difference": bool(clear) if label == "pooled_descriptive" else "",
                    }
                )
    with open(EXP / "phase7b_paired_tables.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(paired_rows[0]))
        w.writeheader()
        w.writerows(paired_rows)

    cross = {}
    for other in COMPARISONS:
        mean_rd = float(np.mean([results[(t, other)]["pooled_rd"] for t in TASKS]))
        draws = np.mean([boot_draws[(t, other)] for t in TASKS], axis=0)
        lo, hi = np.percentile(draws, [2.5, 97.5])
        signs = [int(np.sign(results[(t, other)]["pooled_rd"])) for t in TASKS]
        cross[other] = {"mean_task_rd": mean_rd, "boot95": [float(lo), float(hi)], "task_rd_signs": signs}

    # ---------------- preregistered case rule (primary: vs PURE_SHRINKAGE) ----------------
    dirs = [results[(t, "PURE_SHRINKAGE")]["clear_direction"] for t in TASKS]
    if not any(dirs):
        case = "A"
    elif sum(d > 0 for d in dirs) >= 2 and not any(d < 0 for d in dirs):
        case = "B"
    else:
        case = "C"

    analysis = {
        "episodes": len(rows),
        "episode_csv_sha256": hashlib.sha256(ep_csv.read_bytes()).hexdigest(),
        "noise_verification": noise,
        "results": {f"{t}|{o}": v for (t, o), v in results.items()},
        "cross_task": cross,
        "clear_directions_vs_shrinkage": dict(zip(TASKS, dirs, strict=True)),
        "case": case,
        "gpu_port": {f"{t}|rep{r}": v for (t, r), v in gpu_port.items()},
        "wall": {
            f"{t}|rep{r}": json.loads((runs / t / f"rep{r}" / "summary.json").read_text())["conditions"]
            for t in TASKS
            for r in REPEATS
        },
    }
    (EXP / "phase7b_analysis.json").write_text(json.dumps(analysis, indent=2, default=str))

    # ---------------- print ----------------
    print(
        f"episodes: {len(rows)}; noise fingerprints recomputed: {noise['recomputed']}; unique keys {noise['unique_keys']}"
    )
    print("pairing (shared coords, identical):")
    for k, v in noise["pairs"].items():
        print(f"  {k}: {v}")
    print(f"cross-repeat shared fingerprints: {noise['cross_repeat_shared']}")
    for s in noise["samples"]:
        print("  sample", s)
    print()
    for r in summary_rows:
        print(r)
    print()
    for (task, other), v in results.items():
        print(
            f"{task:5s} COAST vs {other:15s} rep1 b/c={v['rep1']['coast_only_b']}/{v['rep1']['other_only_c']} "
            f"p={v['rep1']['mcnemar_p']:.3f} | rep2 b/c={v['rep2']['coast_only_b']}/{v['rep2']['other_only_c']} "
            f"p={v['rep2']['mcnemar_p']:.3f} | pooled RD={v['pooled_rd']:+.3f} CI=[{v['boot95'][0]:+.3f},"
            f"{v['boot95'][1]:+.3f}] clear={v['clear_difference']} identical_len={v['identical_rollout_length']}"
        )
    print()
    for k, v in cross.items():
        print("cross-task COAST vs", k, v)
    print("CASE:", case, dirs)


if __name__ == "__main__":
    main()
