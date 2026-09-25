"""Phase 7A: run the historical candidate steering driver with a chosen init-state window.

Evaluation-only helper. It imports the **unmodified** candidate driver
`experiments/pi05_libero/src/conceptor_steering.py` @ 29059a5 (miranda-v2) and
replaces exactly one function, `run_single_task_eval` (which launches the LIBERO
client subprocess). The replacement differs from the original in two ways only:

1. With `--state-offset 0` it launches the same `examples/libero_env/main.py` with
   the same arguments as the original. With an offset > 0 it launches
   `phase7a_state_window_client.py`, which changes init-state selection only.
2. It writes the client's full stdout/stderr to `<condition dir>/client.log` and the
   exact command to `<condition dir>/client_cmd.json`. The original discards the log
   on success. We keep it to recover per-episode outcomes.

Untouched (all from the candidate module): model loading, `load_npz`, conceptor
lookup, `ConceptorSteeringHook` (h' = h @ ((1-b)I + bC).T), `SteeredPolicyWrapper`,
the websocket server, `compute_random_conceptor`, the condition loop/order, success
parsing, `summary.json`.

Usage (root venv; PYTHONPATH=<candidate>/src:<candidate>/packages/openpi-client/src):
    python phase7a_candidate_driver.py --state-offset 15 <all normal conceptor_steering.py args>
"""

import json
import logging
import os
import pathlib
import subprocess
import sys

import tyro

CANDIDATE_ROOT = pathlib.Path(
    os.environ.get("PHASE7A_CANDIDATE_ROOT", "/home/stanley/Stanley_ws/COAST-research/COAST-paper-candidate")
)
CLIENT_WRAPPER = pathlib.Path(__file__).resolve().parent / "phase7a_state_window_client.py"

sys.path.insert(0, str(CANDIDATE_ROOT / "experiments" / "pi05_libero" / "src"))
import conceptor_steering as cs  # noqa: E402  (unmodified candidate driver)

logger = logging.getLogger("phase7a_driver")
STATE_OFFSET = 0


def run_single_task_eval(task_name, task_suite_name, num_episodes, port, output_dir):
    """Same contract as cs.run_single_task_eval; see the module docstring for the two differences."""
    task_id = cs.LIBERO_TASK_IDS[task_name]
    libero_env_dir = cs.REPO_ROOT / "examples" / "libero_env"
    abs_output_dir = str(pathlib.Path(output_dir).resolve())
    python = str(libero_env_dir / ".venv" / "bin" / "python")
    if STATE_OFFSET == 0:
        entry = [str(libero_env_dir / "main.py")]
    else:
        entry = [str(CLIENT_WRAPPER), "--state_offset", str(STATE_OFFSET)]
    cmd = [
        python,
        *entry,
        "--task_suite_name", task_suite_name,
        "--task_id", str(task_id),
        "--num_episodes", str(num_episodes),
        "--port", str(port),
        "--output_dir", abs_output_dir,
    ]
    env = os.environ.copy()
    env["MUJOCO_GL"] = "egl"
    pathlib.Path(abs_output_dir).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(abs_output_dir, "client_cmd.json"), "w") as f:
        json.dump({"cmd": cmd, "cwd": str(libero_env_dir), "state_offset": STATE_OFFSET}, f, indent=2)
    logger.info("Eval: %s", " ".join(cmd))
    proc = subprocess.run(cmd, cwd=str(libero_env_dir), env=env, capture_output=True, text=True, timeout=7200)
    log_text = (proc.stdout or "") + (proc.stderr or "")
    with open(os.path.join(abs_output_dir, "client.log"), "w") as f:
        f.write(log_text)
    if proc.returncode != 0:
        logger.error("Eval failed (rc=%s):\n%s", proc.returncode, log_text[-3000:])
        return None
    matches = cs.SUCCESS_RATE_RE.findall(log_text)
    if matches:
        return float(matches[-1])
    logger.error("No success_rate found in main.py output:\n%s", log_text[-2000:])
    return None


def _pop_state_offset(argv):
    rest, offset, i = [], 0, 0
    while i < len(argv):
        if argv[i] == "--state-offset":
            offset = int(argv[i + 1])
            i += 2
            continue
        rest.append(argv[i])
        i += 1
    return offset, rest


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    STATE_OFFSET, argv = _pop_state_offset(sys.argv[1:])
    cs.run_single_task_eval = run_single_task_eval
    logger.info("phase7a driver: candidate=%s state_offset=%d", cs.__file__, STATE_OFFSET)
    cs.main(tyro.cli(cs.Args, args=argv))
