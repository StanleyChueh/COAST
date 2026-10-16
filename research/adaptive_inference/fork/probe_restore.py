"""Restore-identity probe for LIBERO (no policy, no GPU model).

Runs a reference trajectory with deterministic pseudo-random actions, snapshots at
several steps, restores each snapshot, replays the same actions and compares the
per-step state hash and rendered observations bitwise. It does this for:
  * full   - research/adaptive_inference/fork/sim_snapshot (complete mjData + python state)
  * libero - LIBERO's own get_sim_state()/set_init_state() (time, qpos, qvel only)

Usage (examples/libero_env venv):
    cd examples/libero_env
    MUJOCO_GL=egl .venv/bin/python ../../research/adaptive_inference/fork/probe_restore.py --task_id 2
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "..", "..", "..", "examples", "libero_env"))
from libero.libero import benchmark  # noqa: E402
from main import LIBERO_DUMMY_ACTION  # noqa: E402  (examples/libero_env/main.py)
from main import make_env  # noqa: E402  (examples/libero_env/main.py)
from main import prepare_policy_inputs  # noqa: E402  (examples/libero_env/main.py)
import sim_snapshot as ss  # noqa: E402


def run(task_id: int, init_idx: int, n_steps: int, snap_steps, mode: str, seed: int = 0):
    suite = benchmark.get_benchmark_dict()["libero_10"]()
    task = suite.get_task(task_id)
    env = make_env(task, 256, seed=7)
    env.reset()
    obs = env.set_init_state(suite.get_task_init_states(task_id)[init_idx])
    for _ in range(10):
        obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)
    rng = np.random.default_rng(seed)
    actions = rng.uniform(-0.5, 0.5, size=(n_steps, 7))
    actions[:, 6] = np.sign(np.sin(np.arange(n_steps) / 7.0))  # gripper toggles
    ref_hash, ref_obs, snaps = [], [], {}
    for t in range(n_steps):
        if t in snap_steps:
            snaps[t] = ss.capture(env, obs) if mode == "full" else env.get_sim_state().copy()
        ref_hash.append(ss.state_hash(env))
        ref_obs.append(ss.obs_hash(prepare_policy_inputs(obs, 224)))
        obs, _, _, _ = env.step(actions[t].tolist())
    ref_hash.append(ss.state_hash(env))
    results = {}
    for t0, snap in snaps.items():
        obs = ss.restore(env, snap) if mode == "full" else env.set_init_state(snap)
        first_state_ok = ss.state_hash(env) == ref_hash[t0]
        first_obs_ok = ss.obs_hash(prepare_policy_inputs(obs, 224)) == ref_obs[t0]
        first_div, first_obs_div, error = None, None, None
        for t in range(t0, n_steps):
            try:
                obs, _, _, _ = env.step(actions[t].tolist())
            except ValueError as e:  # LIBERO's restore keeps the old timestep -> horizon hit
                error = f"step {t}: {e}"
                first_div = first_div or t + 1
                break
            if ss.state_hash(env) != ref_hash[t + 1] and first_div is None:
                first_div = t + 1
            if t + 1 < n_steps and first_obs_div is None:
                if ss.obs_hash(prepare_policy_inputs(obs, 224)) != ref_obs[t + 1]:
                    first_obs_div = t + 1
        results[t0] = {
            "restored_state_identical": first_state_ok,
            "restored_obs_identical": first_obs_ok,
            "first_divergent_step": first_div,
            "first_divergent_obs_step": first_obs_div,
            "replayed_steps": n_steps - t0,
            "error": error,
        }
    env.close()
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task_id", type=int, default=2)
    ap.add_argument("--init_idx", type=int, default=15)
    ap.add_argument("--n_steps", type=int, default=200)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    snap_steps = {0, 37, 100, 150}
    out = {m: run(args.task_id, args.init_idx, args.n_steps, snap_steps, m) for m in ("full", "libero")}
    print(json.dumps(out, indent=1, default=str))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(out, f, indent=1, default=str)


if __name__ == "__main__":
    main()
