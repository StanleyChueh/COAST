"""Same-state counterfactual fork evaluator for LIBERO + the intervention server.

For every (task, init_state) unit:
  1. Reference rollout: base operator I0, replan every 5 steps, noise master M_ref.
     At preregistered decision steps (every `decision_every_steps` post-settle steps,
     only if at least `min_remaining` steps remain) the environment is snapshotted
     (complete mjData + robosuite python state + the observation) and the server
     returns pre-intervention features for that state.
  2. For every decision state and every branch b = (operator, replicate r):
       restore the snapshot, verify state hash and policy-input hash equal the
       recorded ones, then run the rest of the episode with that operator.
     Noise for every query is derived on the server from (M_r, task_id, init, rollout_step),
     so all operators of replicate r share common random numbers.
  3. An identity branch (I0, M_ref) per decision state must reproduce the reference
     continuation exactly (outcome, length and final state hash); it detects any
     nondeterminism instead of silently treating branches as paired.

Branch identity is the operator id, never an action candidate. Outcomes of branches
are never sent to the server and never used as features.

Output (one directory per run):
  manifest.json, units/<task_id>_<init>.jsonl (one JSON row per reference / branch),
  features/<task_id>_<init>_<step>.npz (pre-intervention features)

Run from examples/libero_env (Python 3.8 venv):
    MUJOCO_GL=egl LIBERO_CONFIG_PATH=... .venv/bin/python \
        ../../research/adaptive_inference/fork/libero_fork.py --out <dir> --port 8600 --units 0:15,0:16
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import socket
import sys
import time
import traceback
import uuid
from typing import Dict, List, Optional, Tuple

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "..", "..", "..", "examples", "libero_env"))
import sim_snapshot as ss  # noqa: E402

from libero.libero import benchmark  # noqa: E402
from main import LIBERO_DUMMY_ACTION, make_env, prepare_policy_inputs  # noqa: E402
from openpi_client import websocket_client_policy as _wcp  # noqa: E402

AIV_KEY = "__aiv__"
SUITE = "libero_10"
MAX_STEPS = 520
NUM_STEPS_WAIT = 10
RESIZE = 224
ENV_SEED = 7

# Arm -> (server op, replan steps, noise-master offset). IH is the temporal operator
# (client-side). I0P is the resampling placebo: the base operator with independent
# noise (master + 1000), i.e. "re-roll the base policy" - the negative control for
# complementarity and the Reuse-Before-You-Retrieve-style resampling arm.
ARMS = {
    "I0": ("I0", 5, 0),
    "I0P": ("I0", 5, 1000),
    "IH": ("I0", 2, 0),
    "IL": ("IL", 5, 0),
    "IC": ("IC", 5, 0),
}


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    return x


def goal_progress(env) -> Tuple[int, int]:
    renv = ss._robosuite_env(env)
    goal = renv.parsed_problem["goal_state"]
    sat = sum(bool(renv._eval_predicate(s)) for s in goal)
    return int(sat), len(goal)


class Runner:
    def __init__(self, args):
        self.args = args
        self.policy = _wcp.WebsocketClientPolicy(args.host, args.port)
        meta = self.policy.get_server_metadata()
        if meta.get("aiv_impl") is None:
            raise RuntimeError("server is not an intervention server (no aiv_impl in metadata)")
        self.server_meta = meta
        self.suite = benchmark.get_benchmark_dict()[SUITE]()
        self.run_id = uuid.uuid4().hex[:12]

    def query(self, obs, prompt, op, key, task_name, features=False):
        el = prepare_policy_inputs(obs, RESIZE)
        el["prompt"] = prompt
        el[AIV_KEY] = {"op": op, "key": [int(k) for k in key], "task": task_name, "features": bool(features)}
        ih = ss.obs_hash({k: v for k, v in el.items() if isinstance(v, np.ndarray)})
        t0 = time.perf_counter()
        out = self.policy.infer(el)
        rt = (time.perf_counter() - t0) * 1000
        aiv = out["aiv"]
        if aiv["op"] != op or list(aiv["key"]) != list(key):
            raise RuntimeError("server echo mismatch: sent {} {} got {} {}".format(op, key, aiv["op"], aiv["key"]))
        return np.asarray(out["actions"], dtype=np.float32), aiv, ih, rt

    def rollout(
        self,
        env,
        obs,
        start_step: int,
        prompt: str,
        task_id: int,
        init: int,
        task_name: str,
        arm: str,
        master: int,
        decision_steps=(),
        on_decision=None,
    ) -> Dict:
        """Run from post-settle rollout step `start_step` until success or MAX_STEPS."""
        op, replan, offset = ARMS[arm]
        master = master + offset
        plan = collections.deque()
        calls, lat, infer_ms = 0, 0.0, 0.0
        first_chunk, executed_h, first_hash = None, [], None
        immediate = None
        success = False
        step = start_step
        while step < MAX_STEPS:
            if not plan:
                is_decision = on_decision is not None and step in decision_steps
                if is_decision:
                    on_decision(step, obs)  # snapshot before the query
                chunk, aiv, ih, rt = self.query(
                    obs, prompt, op, (master, task_id, init, step), task_name, features=is_decision
                )
                if is_decision:
                    on_decision(step, obs, aiv=aiv, input_hash=ih)
                calls += 1
                lat += rt
                infer_ms += float(aiv["infer_ms"])
                if first_chunk is None:
                    first_chunk = chunk
                    first_hash = ih
                if chunk.shape[0] < replan:
                    raise ValueError("chunk shorter than replan")
                plan.extend(chunk[:replan])
                executed_h.append(replan)
            action = plan.popleft()
            obs, reward, done, info = env.step(action.tolist())
            step += 1
            if immediate is None and not plan:
                immediate = {
                    "eef_pos": np.asarray(obs["robot0_eef_pos"]).tolist(),
                    "gripper_qpos": np.asarray(obs["robot0_gripper_qpos"]).tolist(),
                    "step": step,
                }
            if done:
                success = True
                break
        sat, n_goal = goal_progress(env)
        return {
            "success": bool(success),
            "end_step": step,
            "steps_executed": step - start_step,
            "policy_calls": calls,
            "roundtrip_ms_total": lat,
            "server_infer_ms_total": infer_ms,
            "executed_horizon": ARMS[arm][1],
            "first_chunk": first_chunk,
            "first_input_hash": first_hash,
            "immediate": immediate,
            "goal_predicates_satisfied": sat,
            "goal_predicates_total": n_goal,
            "final_state_hash": ss.state_hash(env),
        }

    def run_unit(self, task_id: int, init: int, writer, feat_dir: str) -> None:
        a = self.args
        task = self.suite.get_task(task_id)
        task_name = task.name
        prompt = str(task.language)
        init_states = self.suite.get_task_init_states(task_id)
        env = make_env(task, 256, ENV_SEED)
        try:
            env.reset()
            obs = env.set_init_state(init_states[init])
            for _ in range(NUM_STEPS_WAIT):
                obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)
            decision_steps = set(range(0, MAX_STEPS - a.min_remaining + 1, a.decision_every_steps))
            snaps: Dict[int, Dict] = {}

            def on_decision(step, obs_now, aiv=None, input_hash=None):
                if aiv is None:  # before the query: snapshot, request features
                    snaps[step] = {
                        "snap": ss.capture(env, obs_now),
                        "state_hash": ss.state_hash(env),
                        "goal_progress": goal_progress(env),  # identity branch verifies this is side-effect free
                    }
                    return None
                snaps[step]["input_hash"] = input_hash
                f = aiv.get("features") or {}
                arrays = {k: np.asarray(v) for k, v in f.items() if v is not None and not np.isscalar(v)}
                scalars = {k: v for k, v in f.items() if v is None or np.isscalar(v)}
                np.savez_compressed(os.path.join(feat_dir, "{}_{}_{}.npz".format(task_id, init, step)), **arrays)
                snaps[step]["feature_scalars"] = scalars
                snaps[step]["features_ms"] = aiv.get("features_ms")
                return None

            t0 = time.time()
            ref = self.rollout(
                env, obs, 0, prompt, task_id, init, task_name, "I0", a.master_ref, decision_steps, on_decision
            )
            ref_row = {
                "kind": "reference",
                "run_id": self.run_id,
                "task_id": task_id,
                "task": task_name,
                "instruction": prompt,
                "init_state": init,
                "arm": "I0",
                "master": a.master_ref,
                "decision_steps": sorted(snaps),
                "wall_s": time.time() - t0,
            }
            ref_row.update(ref)
            writer(ref_row)
            for step in sorted(snaps):
                rec = snaps[step]
                group = "{}_{}_{}".format(task_id, init, step)
                writer(
                    {
                        "kind": "decision",
                        "run_id": self.run_id,
                        "group_id": group,
                        "task_id": task_id,
                        "task": task_name,
                        "init_state": init,
                        "decision_step": step,
                        "state_hash": rec["state_hash"],
                        "observation_hash": rec.get("input_hash"),
                        "feature_scalars": rec.get("feature_scalars"),
                        "goal_predicates_at_state": rec["goal_progress"][0],
                        "goal_predicates_total": rec["goal_progress"][1],
                        "features_ms": rec.get("features_ms"),
                        "features_file": "features/{}.npz".format(group),
                    }
                )
                branches = [("I0", a.master_ref, -1)]  # identity branch
                for r, m in enumerate(a.branch_masters):
                    for arm in a.arms:
                        branches.append((arm, m, r))
                for arm, master, r in branches:
                    row = {
                        "kind": "identity" if r < 0 else "branch",
                        "run_id": self.run_id,
                        "group_id": group,
                        "task_id": task_id,
                        "task": task_name,
                        "instruction": prompt,
                        "init_state": init,
                        "decision_step": step,
                        "state_hash": rec["state_hash"],
                        "observation_hash": rec["input_hash"],
                        "intervention_id": arm,
                        "replicate": r,
                        "master": master,
                        "rng": {
                            "noise_key": [master + ARMS[arm][2], task_id, init, "rollout_step"],
                            "env_seed": ENV_SEED,
                        },
                    }
                    t1 = time.time()
                    try:
                        obs_r = ss.restore(env, rec["snap"])
                        row["restore_state_ok"] = ss.state_hash(env) == rec["state_hash"]
                        el = prepare_policy_inputs(obs_r, RESIZE)
                        el["prompt"] = prompt
                        row["restore_obs_ok"] = (
                            ss.obs_hash({k: v for k, v in el.items() if isinstance(v, np.ndarray)})
                            == rec["input_hash"]
                        )
                        if not (row["restore_state_ok"] and row["restore_obs_ok"]):
                            raise RuntimeError("restore identity failed")
                        res = self.rollout(env, obs_r, step, prompt, task_id, init, task_name, arm, master)
                        row.update(res)
                        row["exception"] = None
                        if r < 0:
                            row["identity_ok"] = (
                                res["success"] == ref["success"]
                                and res["end_step"] == ref["end_step"]
                                and res["final_state_hash"] == ref["final_state_hash"]
                            )
                    except Exception as e:  # noqa: BLE001
                        row["exception"] = "{}: {}".format(type(e).__name__, e)
                        row["traceback"] = traceback.format_exc()[-2000:]
                    row["wall_s"] = time.time() - t1
                    writer(row)
        finally:
            env.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8600)
    ap.add_argument("--units", required=True, help="comma list of task_id:init_state")
    ap.add_argument("--master_ref", type=int, default=1000)
    ap.add_argument("--branch_masters", default="2001,2002,2003,2004")
    ap.add_argument("--arms", default="I0,I0P,IH,IL,IC")
    ap.add_argument("--decision_every_steps", type=int, default=50)
    ap.add_argument("--min_remaining", type=int, default=100)
    args = ap.parse_args()
    args.branch_masters = [int(x) for x in args.branch_masters.split(",")]
    args.arms = [x for x in args.arms.split(",") if x]
    for arm in args.arms:
        if arm not in ARMS:
            raise SystemExit("unknown arm {}".format(arm))
    units = [tuple(int(x) for x in u.split(":")) for u in args.units.split(",") if u]
    os.makedirs(os.path.join(args.out, "units"), exist_ok=True)
    os.makedirs(os.path.join(args.out, "features"), exist_ok=True)
    runner = Runner(args)
    man = {
        "run_id": runner.run_id,
        "host": socket.gethostname(),
        "argv": sys.argv,
        "args": {k: v for k, v in vars(args).items()},
        "server_metadata": {k: v for k, v in runner.server_meta.items() if k.startswith("aiv_")},
        "arms": ARMS,
        "suite": SUITE,
        "max_steps": MAX_STEPS,
        "num_steps_wait": NUM_STEPS_WAIT,
        "resize": RESIZE,
        "env_seed": ENV_SEED,
        "fork_code_sha256": {
            f: hashlib.sha256(open(os.path.join(_HERE, f), "rb").read()).hexdigest()
            for f in ("libero_fork.py", "sim_snapshot.py")
        },
    }
    with open(os.path.join(args.out, "manifest.{}.json".format(runner.run_id)), "w") as f:
        json.dump(_jsonable(man), f, indent=1)
    for task_id, init in units:
        path = os.path.join(args.out, "units", "{}_{}.jsonl".format(task_id, init))
        if os.path.exists(path + ".done"):
            print("skip finished unit", task_id, init, flush=True)
            continue
        if os.path.exists(path):
            os.rename(path, path + ".partial.{}".format(int(time.time())))
        with open(path, "w") as fh:

            def writer(row, fh=fh):
                fh.write(json.dumps(_jsonable(row)) + "\n")
                fh.flush()

            t0 = time.time()
            runner.run_unit(task_id, init, writer, os.path.join(args.out, "features"))
        open(path + ".done", "w").write(str(time.time() - t0))
        print("finished unit", task_id, init, "in", round(time.time() - t0), "s", flush=True)


if __name__ == "__main__":
    main()
