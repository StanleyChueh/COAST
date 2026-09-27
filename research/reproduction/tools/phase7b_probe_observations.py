"""Phase 7B (exploratory): save the first policy observation of each held-out init state.

Python 3.8 libero venv, cwd = <candidate>/examples/libero_env. Uses the unmodified candidate client's own
functions (get_task_suite, make_env with seed 7, LIBERO_DUMMY_ACTION settle steps, prepare_policy_inputs), so the
saved element is what main.py sends at rollout_step 0 of the episode for that init state. No policy is queried.

Order invariance check: state 15 is rendered first and again last; the two elements must be byte-identical.

Usage: python phase7b_probe_observations.py <task_id> <first_state> <n_states> <out_npz>
"""

import hashlib
import os
import sys

sys.path.insert(0, os.getcwd())

import main as client
import numpy as np


def element_for(env, init_state, args):
    env.reset()
    obs = env.set_init_state(init_state)
    for _ in range(args["num_steps_wait"]):
        obs, _, _, _ = env.step(client.LIBERO_DUMMY_ACTION)
    return client.prepare_policy_inputs(obs, args["resize_size"])


def digest(el):
    h = hashlib.sha256()
    for k in sorted(el):
        h.update(np.ascontiguousarray(el[k]).tobytes())
    return h.hexdigest()


def main():
    task_id, first, n, out = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
    args = {"num_steps_wait": 10, "resize_size": 224, "seed": 7}
    np.random.seed(args["seed"])
    suite = client.get_task_suite("libero_10")
    task = suite.get_task(task_id)
    states = suite.get_task_init_states(task_id)
    env = client.make_env(task, client.LIBERO_ENV_RESOLUTION, args["seed"])
    try:
        elements = [element_for(env, states[s], args) for s in range(first, first + n)]
        again = element_for(env, states[first], args)
    finally:
        env.close()
    order_invariant = digest(again) == digest(elements[0])
    np.savez_compressed(
        out,
        image=np.stack([e["observation/image"] for e in elements]),
        wrist_image=np.stack([e["observation/wrist_image"] for e in elements]),
        state=np.stack([e["observation/state"] for e in elements]),
        init_states=np.arange(first, first + n),
        prompt=np.array(str(task.language)),
        order_invariant=np.array(order_invariant),
    )
    print("saved", out, "order_invariant", order_invariant, "prompt", task.language)


if __name__ == "__main__":
    main()
