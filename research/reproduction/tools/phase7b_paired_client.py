"""Phase 7B: historical LIBERO client with a shifted init-state window and paired policy noise.

Evaluation-only helper (Python 3.8 libero venv, cwd = <candidate>/examples/libero_env). It runs the
**unmodified** candidate client `examples/libero_env/main.py` @ 29059a5 (miranda-v2) and changes two things:

1. State selection (identical to Phase 7A's `phase7a_state_window_client.py`): the suite's
   `get_task_init_states(task_id)` returns `states[offset:]`, so episode k runs init state offset + k.
2. Paired noise key (Phase 1D protocol): every `policy.infer(element)` call gets
   element["__noise_control__"] = {master_seed, task_id, init_state, rollout_step}.
   - init_state = offset + (number of env.set_init_state calls - 1); the state passed to set_init_state is
     checked against the original suite's states[init_state] (SHA-1 of its bytes).
   - rollout_step = (env.step calls since set_init_state) - num_steps_wait, i.e. main.py's own
     `rollout_step` at the moment it requests a chunk (0, 5, 10, ...). It is asserted to equal
     replan_steps * (index of the request within the episode).
   Every response must echo the same key; the echoed derived-noise SHA-256 must equal the sampler-side SHA-256.
   One JSON line per request goes to --noise_log.

Env construction, env.seed(args.seed), np.random.seed(args.seed), prompts, replanning, action execution,
success check and the websocket client are the candidate's.

Usage:
    python phase7b_paired_client.py --state_offset 15 --policy_noise_seed 100 --noise_log <path> <main.py args>
"""

import hashlib
import importlib.util
import json
import logging
import os
import pathlib
import sys

sys.path.insert(0, os.getcwd())  # the candidate client lives in the current working directory

import main as client  # noqa: E402  (candidate examples/libero_env/main.py)
import numpy as np  # noqa: E402
import tyro  # noqa: E402

logger = logging.getLogger("phase7b_client")
MAIN_ROOT = pathlib.Path(__file__).resolve().parents[3]


def _load_wire():
    # Phase 1D wire protocol (stdlib only), loaded from the main repo; the candidate openpi_client lacks it.
    path = MAIN_ROOT / "packages" / "openpi-client" / "src" / "openpi_client" / "noise_control.py"
    spec = importlib.util.spec_from_file_location("phase1d_noise_wire", str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


WIRE = _load_wire()


def _pop(argv, name, cast):
    rest, value, i = [], None, 0
    while i < len(argv):
        if argv[i] == name:
            value = cast(argv[i + 1])
            i += 2
            continue
        rest.append(argv[i])
        i += 1
    return value, rest


def _sha1(state):
    return hashlib.sha1(np.asarray(state).tobytes()).hexdigest()[:12]


class Tracker:
    def __init__(self, offset, original_states_by_task):
        self.offset = offset
        self.original = original_states_by_task
        self.task_id = None
        self.episode = -1
        self.steps = 0
        self.requests = 0

    def log_episode_end(self):
        if self.episode >= 0:
            logger.info(
                "phase7b_episode_end episode=%d init_state=%d env_steps=%d requests=%d",
                self.episode, self.init_state, self.steps, self.requests,
            )  # fmt: skip

    def on_set_init_state(self, state):
        self.log_episode_end()
        self.episode += 1
        self.steps = 0
        self.requests = 0
        init_state = self.offset + self.episode
        expected = self.original[self.task_id][init_state]
        if _sha1(state) != _sha1(expected):
            raise RuntimeError("init state mismatch at episode {}".format(self.episode))
        logger.info("phase7b_state_map episode=%d init_state=%d sha1=%s", self.episode, init_state, _sha1(state))

    @property
    def init_state(self):
        return self.offset + self.episode


class _TrackedEnv:
    def __init__(self, env, tracker):
        self._env = env
        self._tracker = tracker

    def set_init_state(self, state):
        self._tracker.on_set_init_state(state)
        return self._env.set_init_state(state)

    def step(self, action):
        self._tracker.steps += 1
        return self._env.step(action)

    def close(self):
        self._tracker.log_episode_end()
        return self._env.close()

    def __getattr__(self, name):
        return getattr(self._env, name)


class _ShiftedSuite:
    def __init__(self, suite, offset, tracker):
        self._suite = suite
        self._offset = offset
        self._tracker = tracker

    def get_task_init_states(self, task_id):
        states = self._suite.get_task_init_states(task_id)
        self._tracker.original[task_id] = states
        self._tracker.task_id = task_id
        return states[self._offset :]

    def __getattr__(self, name):
        return getattr(self._suite, name)


def main():
    argv = sys.argv[1:]
    offset, argv = _pop(argv, "--state_offset", int)
    master_seed, argv = _pop(argv, "--policy_noise_seed", int)
    noise_log, argv = _pop(argv, "--noise_log", str)
    if offset is None or master_seed is None or noise_log is None:
        raise SystemExit("--state_offset, --policy_noise_seed and --noise_log are required")
    args = tyro.cli(client.Args, args=argv)
    tracker = Tracker(offset, {})

    original_get_task_suite = client.get_task_suite
    original_make_env = client.make_env
    base_policy_cls = client._websocket_client_policy.WebsocketClientPolicy

    class PairedNoisePolicy(base_policy_cls):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            if not self.get_server_metadata().get("phase7b_paired_noise"):
                raise RuntimeError("server does not advertise phase7b_paired_noise")

        def infer(self, obs):
            rollout_step = tracker.steps - args.num_steps_wait
            if rollout_step != args.replan_steps * tracker.requests:
                raise RuntimeError("unexpected request schedule: {} vs {}".format(rollout_step, tracker.requests))
            key = WIRE.build_noise_control_payload(master_seed, args.task_id, tracker.init_state, rollout_step)
            obs = dict(obs)
            obs[WIRE.NOISE_CONTROL_KEY] = key
            response = super().infer(obs)
            echo = response.get(WIRE.NOISE_CONTROL_ECHO_KEY)
            if echo is None or any(int(echo[f]) != key[f] for f in WIRE.NOISE_KEY_FIELDS):
                raise RuntimeError("noise echo mismatch: {} vs {}".format(echo, key))
            if echo["sha256"] != echo["sampler_sha256"]:
                raise RuntimeError("sampler noise differs from derived noise")
            tracker.requests += 1
            with open(noise_log, "a") as f:
                rec = dict(key)
                rec.update(
                    {
                        "episode": tracker.episode,
                        "sha256": echo["sha256"],
                        "sampler": echo["sampler"],
                        "condition": echo["condition"],
                    }
                )
                f.write(json.dumps(rec) + "\n")
            return response

    client.get_task_suite = lambda name: _ShiftedSuite(original_get_task_suite(name), offset, tracker)
    client.make_env = lambda task, resolution, seed: _TrackedEnv(original_make_env(task, resolution, seed), tracker)
    client._websocket_client_policy.WebsocketClientPolicy = PairedNoisePolicy

    logger.info("phase7b_paired offset=%d master_seed=%d", offset, master_seed)
    client.main(args)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
