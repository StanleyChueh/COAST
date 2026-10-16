"""Post-hoc sensitivity analyses of the pilot (labelled as such in PILOT_GATE.md).

Re-runs the frozen estimators of analyze_pilot.py on subsets, without changing them:
  * tasks with exact identity (0, 3, 7): KS3 (task 2) showed rare renderer nondeterminism
  * per task
    uv run python research/adaptive_inference/sensitivity_pilot.py
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import analyze_pilot as ap


def gate_numbers(states, y, task, unit, rng, *, boot=True):
    s_on = ap.always_on(states, y)
    b_arm = max(ap.ROUTE, key=lambda a: s_on[a])
    o_state, _ = ap.loo_state_oracle(states, y)
    o_task = ap.loo_task_oracle(states, y, task)
    o_risk = ap.loo_risk_gate(states, y)
    ri = ap.rescue_stats(states, y, ap.INTERV)
    rp = ap.rescue_stats(states, y, (*ap.INTERV, ap.PLACEBO))
    out = {
        "n_states": len(states),
        "always_on": s_on,
        "best_arm": b_arm,
        "gainA_pp": 100 * (o_state - s_on[b_arm]),
        "gainD_pp": 100 * (o_state - o_task),
        "O_state": o_state,
        "O_task": o_task,
        "O_risk": o_risk,
        "naive_oracle": ap.naive_oracle(states, y),
        "naive_oracle_I0_I0P": ap.naive_oracle(states, y, ("I0", ap.PLACEBO)),
        "unique_rate": ri["unique_rate"],
        "unique_rate_with_placebo": rp["unique_rate"],
        "rescue_share": ri["rescue_share"],
    }
    if boot:
        ci = ap.bootstrap(states, y, task, unit, rng)
        out["ciA_pp"] = [100 * x for x in ci["gainA"]]
        out["ciD_pp"] = [100 * x for x in ci["gainD"]]
    return out


def main() -> None:
    run = pathlib.Path("results/adaptive_inference/pilot")
    rows = ap.load(run)
    states, _, y, _, _, task, unit, _, _ = ap.build(rows, 4)
    rng = np.random.default_rng(0)
    res = {"exact_tasks_0_3_7": gate_numbers([g for g in states if task[g] != 2], y, task, unit, rng)}
    for t in sorted(set(task.values())):
        res[f"task_{t}"] = gate_numbers([g for g in states if task[g] == t], y, task, unit, rng, boot=False)
    out = run / "analysis" / "sensitivity.json"
    out.write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main()
