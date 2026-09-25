"""Phase 7A analysis: cross-task historical-vs-held-out protocol validation (analysis only; no rollouts).

Reads the run tree written by the Phase 7A queue runner:
    <runs>/<TASK_KEY>/<H|T>/rep<r>/<task_name[:60]>/<condition>/{client.log, client_cmd.json, NN-<task>/episode_XXX.mp4}
and the queue logs (<logs>/queue_gpu<g>.log) for GPU/port provenance.

Per episode:
- success: the client's own log line "Episode k/N: success=..."
- init_state: "phase7a_state_map" lines for arm T; k for arm H, where the unmodified client uses initial_states[k]
- rollout_steps: video frames - NUM_STEPS_WAIT (one frame per env step, 10 settling steps)
Cross-checks: the log-derived rate equals the driver's summary.json success_rate, and success agrees with video
length (< 530 frames), barring success on the very last step.

Writes experiments/phase7a_episode_results.csv, experiments/phase7a_summary.csv and prints the analysis
(stdout, plus experiments/phase7a_analysis.json).

Usage (repo root, CPU only):
    .venv/bin/python research/reproduction/tools/analyze_phase7a.py <runs_dir> <logs_dir>
"""

from __future__ import annotations

import csv
import json
import pathlib
import re
import statistics
import sys

import imageio.v2 as iio

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from analyze_phase1d import EXP
from analyze_phase1d import wilson
from analyze_phase5a_ks3 import fisher_two_sided

NUM_STEPS_WAIT = 10
MAX_STEPS = 520
CANDIDATE_SHA = "29059a537ce123abdab7a4e164db0c72eeff8024"
CHECKPOINT = "checkpoints/openpi-libero-2000 (HF brandonyang/openpi-libero-2000 rev aaeeabc72f8a)"
CONCEPTOR_SOURCE = (
    "candidate for_subin/build_conceptors.py (defaults) on brandonyang/pi05-libero-activations-v1-2000-15env@3e3a8fe2"
)
TASKS = {  # key -> (task_id, full name, layer, alpha, beta, paper base, paper global)
    "LR2a": (
        0,
        "LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket",
        11,
        0.5,
        0.3,
        0.40,
        0.80,
    ),
    "KS4": (
        3,
        "KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it",
        5,
        1.0,
        0.1,
        0.40,
        0.73,
    ),
    "LR1": (
        7,
        "LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket",
        11,
        0.5,
        0.3,
        0.80,
        0.93,
    ),
}
ARM_N = {"H": 15, "T": 30}
EP_RE = re.compile(r"Episode (\d+)/(\d+): success=(True|False)")
MAP_RE = re.compile(r"phase7a_state_map episode=(\d+) init_state=(\d+)")
START_RE = re.compile(r"START gpu=(\d+) port=(\d+) task=(\S+) arm=(\S) rep=(\d+)")


def cond_label(name: str) -> str:
    if name == "baseline":
        return "BASELINE"
    if name.startswith("global_"):
        return "PAPER_GLOBAL"
    if name.startswith("random_"):
        return "RANDOM_CONTROL"
    raise ValueError(name)


def load(runs: pathlib.Path, logs: pathlib.Path) -> list[dict]:
    prov = {}
    for q in sorted(logs.glob("queue_gpu*.log")):
        for m in START_RE.finditer(q.read_text()):
            prov[(m.group(3), m.group(4), int(m.group(5)))] = (int(m.group(1)), int(m.group(2)))
    rows = []
    for key, (task_id, name, layer, alpha, beta, _, _) in TASKS.items():
        for arm in ("H", "T"):
            for rep in (1, 2):
                base = runs / key / arm / f"rep{rep}" / name[:60]
                summary = json.loads((base / "summary.json").read_text())
                logged = {c["condition"]: c["success_rate"] for c in summary["conditions"]}
                expected = {"baseline", f"global_L{layer}_a{alpha}_b{beta}", f"random_L{layer}_b{beta}"}
                assert set(logged) == expected, (key, arm, rep, sorted(logged))
                gpu, port = prov[(key, arm, rep)]
                for cond in sorted(expected):
                    cdir = base / cond
                    cmd = json.loads((cdir / "client_cmd.json").read_text())
                    offset = cmd["state_offset"]
                    assert offset == (0 if arm == "H" else 15), (key, arm, rep, cond, offset)
                    log = (cdir / "client.log").read_text()
                    eps = [(int(a), int(b), c == "True") for a, b, c in EP_RE.findall(log)]
                    n = ARM_N[arm]
                    assert [e[0] for e in eps] == list(range(1, n + 1)), (key, arm, rep, cond)
                    assert all(e[1] == n for e in eps), (key, arm, rep, cond)
                    smap = {int(a): int(b) for a, b in MAP_RE.findall(log)}
                    if arm == "T":
                        assert all(smap[k] == 15 + k for k in range(n)), (key, rep, cond)
                    else:
                        assert not smap
                    k_succ = sum(e[2] for e in eps)
                    assert abs(round(k_succ / n, 2) - logged[cond]) < 1e-9, (key, arm, rep, cond, k_succ, logged[cond])
                    vid_dir = next(cdir.glob(f"{task_id:02d}-*"))
                    for k, (_, _, succ) in enumerate(eps):
                        reader = iio.get_reader(vid_dir / f"episode_{k:03d}.mp4")
                        frames = reader.count_frames()
                        reader.close()
                        steps = frames - NUM_STEPS_WAIT
                        if (steps < MAX_STEPS) != succ:
                            assert succ, (key, arm, rep, cond, k, frames, succ)
                            assert steps == MAX_STEPS, (key, arm, rep, cond, k, frames, succ)
                        rows.append(
                            {
                                "task": key,
                                "task_id": task_id,
                                "arm": arm,
                                "condition": cond_label(cond),
                                "repeat": rep,
                                "layer": layer,
                                "alpha": alpha,
                                "beta": beta,
                                "episode": k,
                                "init_state": smap.get(k, k),
                                "success": int(succ),
                                "rollout_steps": steps,
                                "checkpoint": CHECKPOINT,
                                "candidate_sha": CANDIDATE_SHA,
                                "conceptor_source": CONCEPTOR_SOURCE
                                if cond != "baseline" and not cond.startswith("random_")
                                else (
                                    "candidate compute_random_conceptor(seed=layer*100+int(beta*10))"
                                    if cond.startswith("random_")
                                    else "none"
                                ),
                                "gpu": gpu,
                                "output_dir": str(cdir),
                            }
                        )
    return rows


def counts(rows, **kw):
    sel = [r for r in rows if all(r[k] == v for k, v in kw.items())]
    return sum(r["success"] for r in sel), len(sel)


def main() -> None:
    runs, logs = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
    rows = load(runs, logs)
    fields = list(rows[0])
    with open(EXP / "phase7a_episode_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    summ = []
    for key in TASKS:
        for arm in ("H", "T"):
            for cond in ("BASELINE", "PAPER_GLOBAL", "RANDOM_CONTROL"):
                for rep in (1, 2):
                    k, n = counts(rows, task=key, arm=arm, condition=cond, repeat=rep)
                    summ.append(
                        {
                            "task": key,
                            "arm": arm,
                            "condition": cond,
                            "repeat": rep,
                            "successes": k,
                            "n": n,
                            "success_rate": round(k / n, 4),
                        }
                    )
                k, n = counts(rows, task=key, arm=arm, condition=cond)
                summ.append(
                    {
                        "task": key,
                        "arm": arm,
                        "condition": cond,
                        "repeat": "pooled",
                        "successes": k,
                        "n": n,
                        "success_rate": round(k / n, 4),
                    }
                )
    with open(EXP / "phase7a_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summ[0]))
        w.writeheader()
        w.writerows(summ)

    out = {"n_episode_rows": len(rows), "tasks": {}}
    for key, (_, _, layer, alpha, beta, pbase, pglob) in TASKS.items():
        t = {"config": f"L{layer} a{alpha} b{beta}", "paper_base": pbase, "paper_global": pglob, "arms": {}}
        for arm in ("H", "T"):
            a = {}
            for cond in ("BASELINE", "PAPER_GLOBAL", "RANDOM_CONTROL"):
                reps = [counts(rows, task=key, arm=arm, condition=cond, repeat=r) for r in (1, 2)]
                k, n = counts(rows, task=key, arm=arm, condition=cond)
                # State-level view: successes per init state across the two repeats (0, 1 or 2).
                per_state = {}
                for r in rows:
                    if r["task"] == key and r["arm"] == arm and r["condition"] == cond:
                        per_state[r["init_state"]] = per_state.get(r["init_state"], 0) + r["success"]
                a[cond] = {
                    "rep": [f"{x}/{y}" for x, y in reps],
                    "pooled": f"{k}/{n}",
                    "rate": round(k / n, 3),
                    "wilson95": wilson(k, n),
                    "rep_diff": reps[0][0] - reps[1][0],
                    "states_2of2": sum(v == 2 for v in per_state.values()),
                    "states_1of2": sum(v == 1 for v in per_state.values()),
                    "states_0of2": sum(v == 0 for v in per_state.values()),
                    "per_state": per_state,
                }
            kg, ng = counts(rows, task=key, arm=arm, condition="PAPER_GLOBAL")
            kb, nb = counts(rows, task=key, arm=arm, condition="BASELINE")
            kr, nr = counts(rows, task=key, arm=arm, condition="RANDOM_CONTROL")
            a["fisher_global_vs_base"] = round(fisher_two_sided(kg, ng, kb, nb), 4)
            a["fisher_global_vs_random"] = round(fisher_two_sided(kg, ng, kr, nr), 4)
            a["fisher_global_vs_base_per_rep"] = [
                round(
                    fisher_two_sided(
                        *counts(rows, task=key, arm=arm, condition="PAPER_GLOBAL", repeat=r),
                        *counts(rows, task=key, arm=arm, condition="BASELINE", repeat=r),
                    ),
                    4,
                )
                for r in (1, 2)
            ]
            a["gain_vs_base"] = round(kg / ng - kb / nb, 4)
            a["gain_vs_random"] = round(kg / ng - kr / nr, 4)
            a["gain_vs_base_per_rep"] = [
                round(
                    counts(rows, task=key, arm=arm, condition="PAPER_GLOBAL", repeat=r)[0] / ARM_N[arm]
                    - counts(rows, task=key, arm=arm, condition="BASELINE", repeat=r)[0] / ARM_N[arm],
                    4,
                )
                for r in (1, 2)
            ]
            t["arms"][arm] = a
        t["G_H"] = t["arms"]["H"]["gain_vs_base"]
        t["G_T"] = t["arms"]["T"]["gain_vs_base"]
        t["PGG"] = round(t["G_H"] - t["G_T"], 4)
        t["G_H_vs_random"] = t["arms"]["H"]["gain_vs_random"]
        t["G_T_vs_random"] = t["arms"]["T"]["gain_vs_random"]
        t["PGG_vs_random"] = round(t["G_H_vs_random"] - t["G_T_vs_random"], 4)
        out["tasks"][key] = t
    out["mean_PGG"] = round(statistics.mean(t["PGG"] for t in out["tasks"].values()), 4)
    (EXP / "phase7a_analysis.json").write_text(json.dumps(out, indent=2))
    for key, t in out["tasks"].items():
        print(f"\n== {key} {t['config']} paper Base {t['paper_base']} Global {t['paper_global']}")
        for arm in ("H", "T"):
            a = t["arms"][arm]
            for cond in ("BASELINE", "PAPER_GLOBAL", "RANDOM_CONTROL"):
                c = a[cond]
                print(
                    f"  {arm} {cond:15s} reps {c['rep']} pooled {c['pooled']} ({c['rate']:.3f}, CI {c['wilson95']}) "
                    f"states 2/1/0 = {c['states_2of2']}/{c['states_1of2']}/{c['states_0of2']}"
                )
            print(
                f"  {arm} gain vs base {a['gain_vs_base']:+.3f} (per rep {a['gain_vs_base_per_rep']}) Fisher {a['fisher_global_vs_base']} "
                f"per-rep {a['fisher_global_vs_base_per_rep']}; vs random {a['gain_vs_random']:+.3f} Fisher {a['fisher_global_vs_random']}"
            )
        print(
            f"  G_H {t['G_H']:+.3f}  G_T {t['G_T']:+.3f}  PGG {t['PGG']:+.3f}  (vs random: PGG {t['PGG_vs_random']:+.3f})"
        )
    print(f"\nmean PGG {out['mean_PGG']:+.3f}; rows {len(rows)}")


if __name__ == "__main__":
    main()
