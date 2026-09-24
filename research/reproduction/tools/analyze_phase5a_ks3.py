"""Phase 5A Step 1 analysis: KS3 global reproduction (analysis only; no rollouts).

Inputs (gitignored run outputs):
  examples/libero_env/output/phase5a_ks3_baseline_test_seed15/
  examples/libero_env/output/phase5a_ks3_coast_global_L5_a0.5_b0.1_test_seed15/
  examples/libero_env/output/phase5a_ks3_servers/server.log

Checks: clean client exit, 30 episodes, init state = (15 + episode) % 50, failures are 520-step timeouts,
no noise-control / diagnostics artifacts, and the server log shows the intended hook. Reports success, Wilson
intervals, the gap to the paper's 28/30, and a descriptive (unpaired) baseline -> COAST transition table.
Writes research/reproduction/experiments/phase5a_ks3_episode_results.csv and prints a JSON summary.

Usage (repo root, CPU only):
    CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu uv run --no-sync python research/reproduction/tools/analyze_phase5a_ks3.py
"""

from __future__ import annotations

import csv
import json
import math
import pathlib
import re
import sys

import imageio.v2 as iio

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from analyze_phase1d import EXP
from analyze_phase1d import OUT
from analyze_phase1d import mcnemar_exact
from analyze_phase1d import wilson

SEED = 15
N = 30
NUM_STEPS_WAIT = 10
MAX_STEPS = 520
RUNS = {
    "baseline": "phase5a_ks3_baseline_test_seed15",
    "coast_global_L5_a0.5_b0.1": "phase5a_ks3_coast_global_L5_a0.5_b0.1_test_seed15",
}
SERVER_LOG = OUT / "phase5a_ks3_servers" / "server.log"
_EP_RE = re.compile(r"Episode (\d+)/(\d+): success=(True|False)")
PAPER_SUCCESSES, PAPER_N = 28, 30


def fisher_two_sided(a: int, n1: int, b: int, n2: int) -> float:
    tot, total_n = a + b, n1 + n2

    def p(x: int) -> float:
        return math.comb(n1, x) * math.comb(n2, tot - x) / math.comb(total_n, tot)

    obs = p(a)
    return sum(p(x) for x in range(max(0, tot - n2), min(n1, tot) + 1) if p(x) <= obs + 1e-12)


def load(name: str) -> list[dict]:
    run = OUT / RUNS[name]
    log = (run / "client.log").read_text()
    assert "CLIENT_EXIT=0" in log, f"{name}: client did not exit cleanly"
    succ = {int(m.group(1)) - 1: m.group(3) == "True" for m in _EP_RE.finditer(log)}
    assert sorted(succ) == list(range(N)), f"{name}: missing episodes"
    dirs = [d for d in run.iterdir() if d.is_dir()]
    assert len(dirs) == 1
    assert not (dirs[0] / "noise_fingerprints.jsonl").exists(), f"{name}: noise control artifacts present"
    assert not (dirs[0] / "steering_diagnostics.jsonl").exists(), f"{name}: diagnostics artifacts present"
    eps = []
    for ep in range(N):
        with iio.get_reader(dirs[0] / f"episode_{ep:03d}.mp4") as r:
            steps = r.count_frames() - NUM_STEPS_WAIT
        assert succ[ep] or steps == MAX_STEPS, f"{name} ep {ep}: failure that is not a timeout"
        eps.append({"episode": ep, "init_state": (SEED + ep) % 50, "success": int(succ[ep]), "steps": steps})
    return eps


def main() -> None:
    eps = {n: load(n) for n in RUNS}
    log = SERVER_LOG.read_text()
    server = {
        "traceback_or_error_lines": len(re.findall(r"Traceback|ERROR", log)),
        "steering_enabled": "Steering enabled" in log,
        "noise_control_enabled": "Noise control enabled" in log,
        "diagnostics_enabled": "diagnostics enabled" in log.lower(),
        "hook_built": re.findall(r"Built steering hook \((.*?)\) \[(\w+)\]", log),
    }
    assert not server["noise_control_enabled"]
    assert not server["diagnostics_enabled"]
    b, c = eps["baseline"], eps["coast_global_L5_a0.5_b0.1"]
    kb, kc = sum(e["success"] for e in b), sum(e["success"] for e in c)
    gains = [x["init_state"] for x, y in zip(b, c, strict=True) if not x["success"] and y["success"]]
    losses = [x["init_state"] for x, y in zip(b, c, strict=True) if x["success"] and not y["success"]]
    both = [(x["steps"], y["steps"]) for x, y in zip(b, c, strict=True) if x["success"] and y["success"]]
    summary = {
        "states": [e["init_state"] for e in b],
        "baseline": {"successes": kb, "n": N, "rate": round(kb / N, 3), "wilson95": wilson(kb, N)},
        "coast": {"successes": kc, "n": N, "rate": round(kc / N, 3), "wilson95": wilson(kc, N)},
        "paper_global": {
            "successes": PAPER_SUCCESSES,
            "n": PAPER_N,
            "rate": round(PAPER_SUCCESSES / PAPER_N, 3),
            "wilson95": wilson(PAPER_SUCCESSES, PAPER_N),
        },
        "gap_to_paper_successes": kc - PAPER_SUCCESSES,
        "fisher_exact_two_sided_ours_vs_paper_28_of_30": round(fisher_two_sided(kc, N, PAPER_SUCCESSES, PAPER_N), 4),
        "fisher_exact_baseline_vs_coast": round(fisher_two_sided(kb, N, kc, N), 4),
        "unpaired_transitions_baseline_to_coast": {
            "fail_to_success": len(gains),
            "success_to_fail": len(losses),
            "gain_states": gains,
            "loss_states": losses,
            "mcnemar_exact_p_reference_only_NOT_PAIRED_NOISE": mcnemar_exact(len(gains), len(losses)),
        },
        "mean_steps_when_both_succeed": [
            round(sum(s[0] for s in both) / len(both), 1),
            round(sum(s[1] for s in both) / len(both), 1),
        ]
        if both
        else None,
        "server": server,
    }
    with (EXP / "phase5a_ks3_episode_results.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["condition", "episode", "init_state", "success", "rollout_steps"])
        for name, rows in eps.items():
            for e in rows:
                w.writerow([name, e["episode"], e["init_state"], e["success"], e["steps"]])
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
