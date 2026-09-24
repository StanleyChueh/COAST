"""Phase 5C analysis: KS3 paper-style oracle selection (analysis only; no rollouts).

Modes
  select  Read the find_best_configs.py sweep outputs (fit states 0-14), recover per-episode outcomes from the saved
          videos, cross-check them against the logged success rates, and select one configuration PER STRATEGY with
          the preregistered rule (highest fit successes; ties -> first in repository iteration order = layer asc,
          alpha asc, beta asc). Writes experiments/phase5c_fit_results.csv and merges the selection of every
          completed strategy into experiments/phase5c_selected_configs.json.
  freeze  Stamp phase5c_selected_configs.json as frozen (before any test rollout) and print its sha256.
  test    Read the 30-episode test runs (states 15-44), write experiments/phase5c_test_results.csv and
          experiments/phase5c_test_episode_results.csv, and print the comparison with paper Table 4.

Inputs (gitignored): examples/libero_env/output/phase5c_sweep_*/<timestamp>/, examples/libero_env/output/phase5c_test_*/

Usage (repo root, CPU only):
    CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu uv run --no-sync python research/reproduction/tools/analyze_phase5c_oracle.py select
"""

from __future__ import annotations

import csv
import datetime
import hashlib
import json
import pathlib
import re
import sys

import imageio.v2 as iio

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from analyze_phase1d import EXP
from analyze_phase1d import OUT
from analyze_phase1d import wilson
from analyze_phase5a_ks3 import fisher_two_sided

TASK = "KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it"
TASK_ID = 2
NUM_STEPS_WAIT = 10
MAX_STEPS = 520
FIT_SEED, FIT_N = 0, 15
TEST_SEED, TEST_N = 15, 30
LAYERS = (0, 5, 11, 17)
ALPHAS = (0.1, 0.5, 1.0, 2.0, 10.0)
BETAS = (0.1, 0.3, 0.5)
NPZ_SHA256 = "1ac8fceb11c71d8bc6c190b0563a82dc8dc572b269833502e26ffd7d8e82709c"

# Shards in layer order, so concatenation preserves repository iteration order.
SWEEPS = {
    "global": ["phase5c_sweep_global_L0_5", "phase5c_sweep_global_L11_17"],
    "positive_only": ["phase5c_sweep_positive_only_L0_5", "phase5c_sweep_positive_only_L11_17"],
    "per_step": ["phase5c_sweep_per_step"],
    "linear": ["phase5c_sweep_linear"],
}
# Paper Table 4, KS3 (pi0.5), 30 test rollouts; linear is not a Table 4 column.
PAPER = {
    "global": {"successes": 28, "config": {"layer": 5, "alpha": 0.5, "beta": 0.1}},
    "per_step": {"successes": 26, "config": {"layer": 5, "alpha": 10.0, "beta": 0.3}},
    "positive_only": {"successes": 24, "config": {"layer": 11, "alpha": 1.0, "beta": 0.1}},
}
SELECTED_JSON = EXP / "phase5c_selected_configs.json"
_EP_RE = re.compile(r"Episode (\d+)/(\d+): success=(True|False)")


def expected_conditions(strategy: str) -> list[tuple[int, float, float, str]]:
    """Replicates find_best_configs.py's strategy-gated expansion and condition naming (lines 228-261)."""
    alphas = (1.0,) if strategy == "per_step" else ALPHAS
    betas = (0.0,) if strategy == "linear" else BETAS
    return [(layer, a, b, f"{strategy}_L{layer}_a{a}_b{b}") for layer in LAYERS for a in alphas for b in betas]


def video_steps(ep_dir: pathlib.Path, n: int) -> list[int]:
    steps = []
    for ep in range(n):
        with iio.get_reader(ep_dir / f"episode_{ep:03d}.mp4") as r:
            steps.append(r.count_frames() - NUM_STEPS_WAIT)
    return steps


def load_condition(run: pathlib.Path, cond: str, logged_rate: float) -> dict:
    task_dirs = [d for d in (run / TASK / cond).iterdir() if d.is_dir()]
    assert len(task_dirs) == 1, f"{cond}: expected one task dir"
    steps = video_steps(task_dirs[0], FIT_N)
    assert all(s <= MAX_STEPS for s in steps), f"{cond}: episode longer than max steps"
    k_logged = round(logged_rate * FIT_N)
    k_video = sum(s < MAX_STEPS for s in steps)
    # A success on exactly the last step is indistinguishable from a timeout by length; the logged count decides.
    assert k_video <= k_logged <= k_video + sum(s == MAX_STEPS for s in steps), f"{cond}: video/log mismatch"
    return {
        "fit_success": k_logged,
        "fit_rate": round(k_logged / FIT_N, 3),
        "steps": steps,
        "video_agrees": k_video == k_logged,
    }


def load_sweep(strategy: str) -> tuple[list[dict], list[dict]]:
    rows, baselines = [], []
    by_cond: dict[str, tuple[pathlib.Path, float]] = {}
    for shard in SWEEPS[strategy]:
        runs = sorted(p for p in (OUT / shard).iterdir() if p.is_dir())
        assert len(runs) == 1, f"{shard}: expected one timestamped run dir, found {len(runs)}"
        run = runs[0]
        args = json.loads((run / "args.json").read_text())
        assert args["seed"] == FIT_SEED
        assert args["num_episodes"] == FIT_N
        assert list(args["strategies"]) == [strategy]
        assert "SWEEP_EXIT=0" in (OUT / shard / "sweep.log").read_text(), f"{shard}: sweep did not exit cleanly"
        for line in (run / "partial_results.jsonl").read_text().splitlines():
            rec = json.loads(line)
            assert rec["task"] == TASK
            rate = rec["success_rate"]
            assert rate == rate, f"{shard}: NaN (crashed) condition {rec['condition']}"
            if rec["condition"] == "baseline":
                b = load_condition(run, "baseline", rate)
                baselines.append({"strategy_shard": shard, **{k: b[k] for k in ("fit_success", "fit_rate")}})
            else:
                assert rec["condition"] not in by_cond, f"duplicate condition {rec['condition']}"
                by_cond[rec["condition"]] = (run, rate)
    expected = expected_conditions(strategy)
    assert sorted(by_cond) == sorted(c[3] for c in expected), f"{strategy}: missing or extra conditions"
    for order, (layer, a, b, cond) in enumerate(expected):
        run, rate = by_cond[cond]
        r = load_condition(run, cond, rate)
        rows.append(
            {
                "strategy": strategy,
                "order": order,
                "layer": layer,
                "alpha": a,
                "beta": b,
                "condition": cond,
                "fit_success": r["fit_success"],
                "fit_n": FIT_N,
                "fit_rate": r["fit_rate"],
                "mean_steps": round(sum(r["steps"]) / FIT_N, 1),
                "video_agrees": r["video_agrees"],
            }
        )
    return rows, baselines


def select(rows: list[dict]) -> dict:
    best = max(r["fit_success"] for r in rows)
    ties = [r for r in rows if r["fit_success"] == best]  # rows are in repository iteration order
    sel = ties[0]
    return {
        "selected": {"layer": sel["layer"], "alpha": sel["alpha"], "beta": sel["beta"], "condition": sel["condition"]},
        "fit_success": best,
        "fit_n": FIT_N,
        "fit_rate": round(best / FIT_N, 3),
        "tie_set_size": len(ties),
        "tie_set": [r["condition"] for r in ties],
        "n_configurations": len(rows),
        "fit_success_distribution": {str(k): sum(r["fit_success"] == k for r in rows) for k in range(FIT_N + 1)},
    }


def cmd_select(strategies: list[str]) -> None:
    fit_csv = EXP / "phase5c_fit_results.csv"
    existing = {}
    if fit_csv.exists():
        with fit_csv.open() as f:
            for r in csv.DictReader(f):
                existing.setdefault(r["strategy"], []).append(r)
    out = json.loads(SELECTED_JSON.read_text()) if SELECTED_JSON.exists() else None
    if out is None:
        out = {
            "phase": "5C",
            "task": TASK,
            "task_id": TASK_ID,
            "conceptor_npz_sha256": NPZ_SHA256,
            "fit_states": "0-14 (--seed 0 --num_episodes 15)",
            "selection_rule": "per strategy: max fit successes; ties -> first in repository iteration order "
            "(layer asc, alpha asc, beta asc); no test data used",
            "frozen": False,
            "strategies": {},
            "fit_baselines": [],
        }
    assert not out["frozen"], "selection already frozen"
    summary = {}
    for s in strategies:
        rows, baselines = load_sweep(s)
        sel = select(rows)
        if s == "per_step":
            sel["note"] = (
                "repository per_step: alpha fixed at 1.0; paper per-step alpha axis (KS3 oracle alpha 10) not expressible"
            )
        if s == "linear":
            sel["note"] = "optional; not a paper Table 4 column; beta inert (h' = h + alpha * v)"
        out["strategies"][s] = sel
        out["fit_baselines"] = [b for b in out["fit_baselines"] if b["strategy_shard"] not in SWEEPS[s]] + baselines
        existing[s] = rows
        summary[s] = {
            **sel,
            "fit_baselines": baselines,
            "video_mismatch_conditions": [r["condition"] for r in rows if not r["video_agrees"]],
        }
    SELECTED_JSON.write_text(json.dumps(out, indent=2) + "\n")
    cols = [
        "strategy",
        "order",
        "layer",
        "alpha",
        "beta",
        "condition",
        "fit_success",
        "fit_n",
        "fit_rate",
        "mean_steps",
        "video_agrees",
    ]
    with fit_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for s in SWEEPS:
            for r in existing.get(s, []):
                w.writerow({c: r[c] for c in cols})
    print(json.dumps(summary, indent=1))


def cmd_freeze() -> None:
    out = json.loads(SELECTED_JSON.read_text())
    assert not out["frozen"]
    out["frozen"] = True
    out["frozen_at"] = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
    SELECTED_JSON.write_text(json.dumps(out, indent=2) + "\n")
    print(out["frozen_at"], hashlib.sha256(SELECTED_JSON.read_bytes()).hexdigest())


def load_test(name: str) -> list[dict]:
    run = OUT / f"phase5c_test_{name}"
    log = (run / "client.log").read_text()
    assert "CLIENT_EXIT=0" in log, f"{name}: client did not exit cleanly"
    succ = {int(m.group(1)) - 1: m.group(3) == "True" for m in _EP_RE.finditer(log)}
    assert sorted(succ) == list(range(TEST_N)), f"{name}: missing episodes"
    dirs = [d for d in run.iterdir() if d.is_dir()]
    assert len(dirs) == 1
    assert not (dirs[0] / "noise_fingerprints.jsonl").exists()
    assert not (dirs[0] / "steering_diagnostics.jsonl").exists()
    steps = video_steps(dirs[0], TEST_N)
    eps = []
    for ep in range(TEST_N):
        assert succ[ep] or steps[ep] == MAX_STEPS, f"{name} ep {ep}: failure that is not a timeout"
        eps.append({"episode": ep, "init_state": (TEST_SEED + ep) % 50, "success": int(succ[ep]), "steps": steps[ep]})
    return eps


def reading(k: int, k_paper: int) -> str:
    lo, hi = wilson(k_paper, TEST_N)
    if lo <= k / TEST_N <= hi:
        return "consistent"
    if fisher_two_sided(k, TEST_N, k_paper, TEST_N) < 0.05:
        return "inconsistent"
    return "inconclusive"


def cmd_test() -> None:
    sel = json.loads(SELECTED_JSON.read_text())
    assert sel["frozen"], "selection must be frozen before test analysis"
    conds = ["baseline"] + [s for s in ("global", "positive_only", "per_step", "linear") if s in sel["strategies"]]
    eps = {c: load_test(c) for c in conds}
    kb = sum(e["success"] for e in eps["baseline"])
    rows, summary = [], {}
    for c in conds:
        k = sum(e["success"] for e in eps[c])
        s = sel["strategies"].get(c)
        row = {
            "condition": c,
            "selected_configuration": "none (unsteered)" if s is None else s["selected"]["condition"],
            "layer": "" if s is None else s["selected"]["layer"],
            "alpha": "" if s is None else s["selected"]["alpha"],
            "beta": "" if s is None else s["selected"]["beta"],
            "fit_success": "" if s is None else s["fit_success"],
            "fit_n": "" if s is None else FIT_N,
            "test_success": k,
            "test_n": TEST_N,
            "test_rate": round(k / TEST_N, 3),
            "wilson95_lo": wilson(k, TEST_N)[0],
            "wilson95_hi": wilson(k, TEST_N)[1],
            "paper_test_success": PAPER[c]["successes"] if c in PAPER else "",
            "gap_to_paper": k - PAPER[c]["successes"] if c in PAPER else "",
            "fisher_p_vs_paper": round(fisher_two_sided(k, TEST_N, PAPER[c]["successes"], TEST_N), 4)
            if c in PAPER
            else "",
            "reading_vs_paper": reading(k, PAPER[c]["successes"]) if c in PAPER else "",
            "delta_vs_our_test_baseline": "" if c == "baseline" else k - kb,
            "fisher_p_vs_our_test_baseline": ""
            if c == "baseline"
            else round(fisher_two_sided(k, TEST_N, kb, TEST_N), 4),
        }
        rows.append(row)
        summary[c] = dict(row)  # copy: the extra summary fields below must not leak into the CSV row
        if s is not None:
            summary[c]["fit_to_test_rate_drop"] = round(s["fit_success"] / FIT_N - k / TEST_N, 3)
            b_states = {e["init_state"] for e in eps["baseline"] if e["success"]}
            c_states = {e["init_state"] for e in eps[c] if e["success"]}
            summary[c]["unpaired_gain_states_vs_baseline"] = sorted(c_states - b_states)
            summary[c]["unpaired_loss_states_vs_baseline"] = sorted(b_states - c_states)
    with (EXP / "phase5c_test_results.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with (EXP / "phase5c_test_episode_results.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["condition", "episode", "init_state", "success", "rollout_steps"])
        for c in conds:
            for e in eps[c]:
                w.writerow([c, e["episode"], e["init_state"], e["success"], e["steps"]])
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "select":
        cmd_select(sys.argv[2:] or ["global"])
    elif mode == "freeze":
        cmd_freeze()
    elif mode == "test":
        cmd_test()
    else:
        raise SystemExit(f"unknown mode {mode}")
