"""Phase 7A: run the historical LIBERO client with a shifted init-state window.

Evaluation-only helper. It changes **state selection only**. Everything else is the
unmodified candidate client `examples/libero_env/main.py` @ 29059a5 (miranda-v2):
env construction, env.seed(args.seed), np.random.seed(args.seed), prompts, replanning,
success check and the websocket policy client.

The candidate client evaluates `initial_states[episode]` for episode = 0..N-1. This
wrapper swaps the module-level `get_task_suite` for a proxy whose
`get_task_init_states(task_id)` returns `states[offset:]`. Episode k therefore runs
init state `offset + k`. With `--state_offset 0` it is an exact pass-through.

Usage (Python 3.8 libero venv, cwd = <candidate>/examples/libero_env):
    python phase7a_state_window_client.py --state_offset 15 <all normal main.py args>
"""

import hashlib
import logging
import os
import sys

# The candidate client lives in the current working directory.
sys.path.insert(0, os.getcwd())

import main as client  # noqa: E402  (candidate examples/libero_env/main.py)
import numpy as np  # noqa: E402
import tyro  # noqa: E402

logger = logging.getLogger("phase7a_state_window")


def _pop_state_offset(argv):
    """Remove '--state_offset N' from argv; return (offset, remaining argv)."""
    rest, offset = [], 0
    i = 0
    while i < len(argv):
        if argv[i] == "--state_offset":
            offset = int(argv[i + 1])
            i += 2
            continue
        rest.append(argv[i])
        i += 1
    return offset, rest


class _ShiftedSuite:
    """Delegates to the real LIBERO suite; only get_task_init_states is shifted."""

    def __init__(self, suite, offset):
        self._suite = suite
        self._offset = offset

    def get_task_init_states(self, task_id):
        states = self._suite.get_task_init_states(task_id)
        shifted = states[self._offset:]
        # Log the index -> state mapping (with a content hash) for every state the
        # client could draw; only the first --num_episodes entries are used.
        for k in range(len(shifted)):
            digest = hashlib.sha1(np.asarray(shifted[k]).tobytes()).hexdigest()[:12]
            logger.info("phase7a_state_map episode=%d init_state=%d sha1=%s", k, self._offset + k, digest)
        return shifted

    def __getattr__(self, name):
        return getattr(self._suite, name)


def main():
    offset, argv = _pop_state_offset(sys.argv[1:])
    original_get_task_suite = client.get_task_suite

    def shifted_get_task_suite(task_suite_name):
        return _ShiftedSuite(original_get_task_suite(task_suite_name), offset)

    client.get_task_suite = shifted_get_task_suite
    logger.info("phase7a_state_window offset=%d", offset)
    client.main(tyro.cli(client.Args, args=argv))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
