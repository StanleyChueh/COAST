"""Dump real LIBERO policy inputs (after 10 settle steps) for server-side operator checks.

    cd examples/libero_env && MUJOCO_GL=egl LIBERO_CONFIG_PATH=... \
        .venv/bin/python ../../research/adaptive_inference/fork/dump_obs.py --out <file.npz>
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "..", "..", "examples", "libero_env"))
from libero.libero import benchmark  # noqa: E402
from main import LIBERO_DUMMY_ACTION  # noqa: E402
from main import make_env  # noqa: E402
from main import prepare_policy_inputs  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--task_ids", default="0,2,3,7")
    ap.add_argument("--init_idx", type=int, default=15)
    args = ap.parse_args()
    suite = benchmark.get_benchmark_dict()["libero_10"]()
    arrays = {}
    for tid in [int(x) for x in args.task_ids.split(",")]:
        task = suite.get_task(tid)
        env = make_env(task, 256, seed=7)
        env.reset()
        obs = env.set_init_state(suite.get_task_init_states(tid)[args.init_idx])
        for _ in range(10):
            obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)
        el = prepare_policy_inputs(obs, 224)
        for k, v in el.items():
            arrays[f"t{tid}|{k}"] = v
        arrays[f"t{tid}|prompt"] = np.array(str(task.language))
        arrays[f"t{tid}|name"] = np.array(task.name)
        env.close()
    np.savez(args.out, **arrays)
    print("wrote", args.out, sorted(arrays))


if __name__ == "__main__":
    main()
