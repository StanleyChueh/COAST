"""Reproduce a pilot identity failure with per-step diagnostics (needs one intervention server).

Reference rollout (I0, master 1000) from reset -> snapshots at the decision steps; then
a contaminating branch (same as the pilot: the reference itself has already run), then
restore the snapshot at `--step` and re-run I0 with master 1000, recording per step:
state hash, policy-input hash, burner/site alpha values of MjModel. Reports the first
divergence and which component differed.

    cd examples/libero_env && MUJOCO_GL=egl LIBERO_CONFIG_PATH=... .venv/bin/python \
        ../../research/adaptive_inference/fork/debug_identity.py --task_id 2 --init 21 --step 0 --port 8600
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "..", "..", "..", "examples", "libero_env"))
import sim_snapshot as ss  # noqa: E402

from libero.libero import benchmark  # noqa: E402
from main import LIBERO_DUMMY_ACTION, make_env, prepare_policy_inputs  # noqa: E402
from openpi_client import websocket_client_policy as _wcp  # noqa: E402


IMAGES = {}


def site_alpha(env):
    renv = ss._robosuite_env(env)
    return np.asarray(renv.sim.model._model.site_rgba[:, 3]).copy()


def run(
    env, obs, start, policy, task, prompt, task_id, init, master, snaps_at=(), trace=None, max_steps=520, feats_at=()
):
    plan = collections.deque()
    step = start
    snaps = {}
    while step < max_steps:
        if not plan:
            if step in snaps_at:
                snaps[step] = (ss.capture(env, obs), site_alpha(env))
            el = prepare_policy_inputs(obs, 224)
            el["prompt"] = prompt
            el["__aiv__"] = {
                "op": "I0",
                "key": [master, task_id, init, step],
                "task": task.name,
                "features": step in feats_at,
            }
            if trace is not None:
                trace.append(("query", step, ss.obs_hash({k: v for k, v in el.items() if isinstance(v, np.ndarray)})))
                IMAGES.setdefault(id(trace), {})[step] = {
                    k: v.copy() for k, v in el.items() if isinstance(v, np.ndarray)
                }
            plan.extend(np.asarray(policy.infer(el)["actions"])[:5])
        obs, _, done, _ = env.step(plan.popleft().tolist())
        step += 1
        if trace is not None:
            trace.append(("state", step, ss.state_hash(env), site_alpha(env).tolist()))
        if done:
            return True, step, snaps
    return False, step, snaps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task_id", type=int, default=2)
    ap.add_argument("--init", type=int, default=21)
    ap.add_argument("--step", type=int, default=0)
    ap.add_argument("--port", type=int, default=8600)
    ap.add_argument("--out", default="")
    ap.add_argument("--features", action="store_true", help="request features at decision steps like the pilot")
    args = ap.parse_args()
    suite = benchmark.get_benchmark_dict()["libero_10"]()
    task = suite.get_task(args.task_id)
    prompt = str(task.language)
    policy = _wcp.WebsocketClientPolicy("127.0.0.1", args.port)
    env = make_env(task, 256, 7)
    env.reset()
    obs = env.set_init_state(suite.get_task_init_states(args.task_id)[args.init])
    for _ in range(10):
        obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)
    ref_trace = []
    feats_at = set(range(0, 421, 50)) if args.features else set()
    succ, end, snaps = run(
        env, obs, 0, policy, task, prompt, args.task_id, args.init, 1000, {args.step}, ref_trace, feats_at=feats_at
    )
    print("reference", succ, end, "site alpha at end", [a for a in site_alpha(env) if a > 0][:8])
    alpha_ref_end = site_alpha(env)
    out = {"reference": {"success": succ, "end": end}}
    for label, fix_model in (("identity_plain", False), ("identity_with_site_rgba_reset", True)):
        snap, snap_alpha = snaps[args.step]
        o = ss.restore(env, snap)
        if fix_model:
            ss._robosuite_env(env).sim.model._model.site_rgba[:, 3] = snap_alpha
        tr = []
        s2, e2, _ = run(env, o, args.step, policy, task, prompt, args.task_id, args.init, 1000, (), tr)
        ref_after = [x for x in ref_trace if x[1] > args.step or (x[0] == "query" and x[1] >= args.step)]
        first = None
        for a, b in zip(ref_after, tr, strict=False):
            if a[:3] != b[:3]:
                first = {
                    "ref": a[:3],
                    "branch": b[:3],
                    "ref_alpha": a[3] if len(a) > 3 else None,
                    "branch_alpha": b[3] if len(b) > 3 else None,
                }
                if first["ref_alpha"] is not None:
                    diff = [
                        i
                        for i, (p, q) in enumerate(zip(first["ref_alpha"], first["branch_alpha"], strict=True))
                        if p != q
                    ]
                    first["alpha_diff_sites"] = diff
                    first.pop("ref_alpha"), first.pop("branch_alpha")
                if a[0] == "query":
                    ia, ib = IMAGES[id(ref_trace)][a[1]], IMAGES[id(tr)][b[1]]
                    det = {}
                    for k in ia:
                        d = (
                            np.abs(ia[k].astype(np.int32) - ib[k].astype(np.int32))
                            if ia[k].dtype == np.uint8
                            else np.abs(ia[k] - ib[k])
                        )
                        nz = np.argwhere(d > 0)
                        det[k] = {
                            "n_diff": int((d > 0).sum()),
                            "max": float(d.max()),
                            "bbox": [nz.min(0).tolist(), nz.max(0).tolist()] if len(nz) else None,
                        }
                    first["image_diff"] = det
                break
        out[label] = {"success": s2, "end": e2, "first_divergence": first}
        print(label, json.dumps(out[label]))
    print(
        "alpha differs between reference end and snapshot:",
        np.flatnonzero(alpha_ref_end != snaps[args.step][1]).tolist(),
    )
    if args.out:
        with open(args.out, "w") as f:
            json.dump(out, f, indent=1, default=str)
    env.close()


if __name__ == "__main__":
    main()
