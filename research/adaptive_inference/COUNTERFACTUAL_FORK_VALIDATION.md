# Stage 4: same-state counterfactual fork harness — validation

Harness:
- [`fork/sim_snapshot.py`](fork/sim_snapshot.py) — exact snapshot and restore;
- [`fork/libero_fork.py`](fork/libero_fork.py) — reference rollout, decision states, branches and identity branches;
- [`intervention_server.py`](intervention_server.py) — operators and keyed noise.

Validation tools:
- [`fork/probe_restore.py`](fork/probe_restore.py) (simulator only);
- [`validate_operators.py`](validate_operators.py) (policy only);
- the identity branch inside every pilot decision state (end to end).

## 1. What is serialized at a decision state

| state component | how | why it matters |
|---|---|---|
| MuJoCo physics | `mj_getState(mjSTATE_INTEGRATION)`: time, qpos, qvel, act, **qacc_warmstart**, ctrl, qfrc_applied, xfrc_applied, mocap pos/quat, eq_active, userdata, plugin state | LIBERO's `get_sim_state()` stores only (time, qpos, qvel). Without the warm start and ctrl the solver path differs |
| robosuite env counters | `timestep`, `cur_time`, `done` | LIBERO's restore keeps the *old* timestep. Repeated restores then hit the 1000-step horizon (`executing action in terminated episode`; observed) |
| robot, controller, interpolators, **grippers** | numeric attributes (arrays, scalars, ring/delta buffers) of each object | the Panda gripper keeps a stateful `current_action`. Without it the trajectory diverged on the first step even with identical MuJoCo state (observed) |
| observables | numeric attributes of every robosuite observable | sensor caches |
| observation | the observation returned by the env at that state, stored verbatim | robosuite computes step-time observations inside the substep loop. A forced recomputation after restore is not bitwise equal (observed), so the decision observation is part of the state |
| observation history | none needed | π0.5 conditions on the current frame and state only; the action queue is empty at every decision state by construction |
| task state | derived from sim state (BDDL predicates) | `goal_progress` at the state is recorded; the identity branch confirms the evaluation has no side effects |
| environment RNG | not used after reset in LIBERO stepping | nothing to restore; checked by the identity branch |
| NumPy / Python / Torch RNG | **not used for policy noise**. The server derives flow noise from an integer key (`SeedSequence(key) → PCG64 → N(0,1)`, float32, shape (10, 32)) | branches are reproducible and common random numbers are exact |
| policy noise / inference seed | key = (master, task_id, init_state, rollout_step); the server echoes the key and noise sha256 on every response | CRN across operators; independent masters across replicates |

Every branch first verifies `state_hash` (SHA-256 over time, qpos, qvel, act, ctrl, qacc_warmstart, mocap, applied forces and the env counters) and the SHA-256 of the exact policy-input arrays against the values recorded at the decision state. A mismatch raises and the branch is recorded as an exception, never silently paired.

## 2. Simulator-only restore identity (`probe_restore.py`)

Tasks 0, 2, 3 and 7 (init 15). 300 steps of deterministic pseudo-random actions with gripper toggles. Snapshots at steps 0, 37, 100 and 150, then restore and replay the same actions.

| method | restored state identical | restored obs identical | first divergent state step | first divergent obs step |
|---|---|---|---|---|
| **this harness** | 16/16 | 16/16 | **none** (all replays to step 300) | **none** |
| LIBERO `get_sim_state` / `set_init_state` | 0/16 | 0/16 | **step t0+1** in every case | t0+1 (or a horizon exception) |

Outputs: `results/adaptive_inference/validation/probe_restore_task{0,2,3,7}.json`.

**A methodological note for the field.** Any "same-state" comparison built on LIBERO's own `set_init_state(get_sim_state())` mid-episode is not a same-state comparison. It diverges on the first step.

## 3. Policy-side identity (`validate_operators.py`)

See [`INTERVENTION_SPEC.md`](INTERVENTION_SPEC.md):
- noise golden fingerprint reproduced;
- I0 is bitwise equal to stock `Policy.infer(noise)`;
- disabled IL (β = 0) is bitwise equal to I0;
- IC at ω = 1 matches I0 to ≤ 6e-8;
- the read-only feature hook is inert;
- features are deterministic.

## 4. End-to-end identity (smoke and pilot)

Each decision state gets an **identity branch**: restore, then run I0 with the reference noise master. It must reproduce the reference rollout's continuation, meaning the same success, the same end step and the same final `state_hash`.

| run | identity branches | exact | restore checks passed |
|---|---|---|---|
| smoke2 (task 2, init 15, steps 0 / 200 / 400) | 3 | 3/3 | 15/15 |
| smoke3 (task 0, init 15, steps 0 / 300, with I0P + goal progress) | 2 | 2/2 | 12/12 |
| **pilot** (306 decision states, 4 tasks) | 306 | **303/306** (tasks 0/3/7: 235/235; KS3: 68/71) | **6,426/6,426** state + observation hash checks; 0 exceptions |

These include decision states reached after 200–400 policy-driven steps with contacts, grasps and articulated objects. They also include a closed loop of up to 104 policy calls from the restored state. Across processes, Phase 1D already showed closed-loop determinism given the noise.

**Residual non-determinism found by the pilot (KS3 only).**
- The 3 failing identity branches were reproduced with `fork/debug_identity.py`. In 2 of 4 identical repeats, the wrist-camera image differs in **one pixel by one intensity level** at a step where the MuJoCo state is bitwise identical, just after the stove flame appears. The reference rollout itself shows the same two-valued pixel, so this is EGL rendering non-determinism, not a restore defect.
- Separately, LIBERO stores flame visibility in `MjModel.site_rgba`, which v1 does not snapshot. It can be stale for the first simulated step after a restore. That render is never a policy input for h ≥ 2: alpha 1 and 2 render identically, and visibility is corrected at the end of the step.
- A hardened v2 (`results/adaptive_inference/validation/sim_snapshot.v2_proposal.py`: restores `site_rgba`, `geom_rgba` and `vis_site_names`; adds them to `state_hash`) is required before any h = 1 operator.
- Details: `PILOT_GATE.md` §4.

Outputs `results/adaptive_inference/smoke{1,2,3}` are excluded from analysis. smoke1 was the run that exposed the variable-size `efc_*` arrays in the first, full-mjData-copy implementation, which is why the harness moved to `mjSTATE_INTEGRATION`.

## 5. Data schema (one JSON row per record, `units/<task>_<init>.jsonl`)

- `reference`: run_id, task, instruction, init_state, master, decision_steps, success, end_step, policy_calls, latency, goal predicates, final_state_hash.
- `decision`: group_id = `task_init_step`, decision_step, state_hash, observation_hash, feature scalars (CAG contrast, IL displacement, IL intervention norm), goal predicates at the state, and the path to `features/<group>.npz`. The npz holds the latent at denoising steps 0 and 9, the first chunks of I0, IL and IC, and K = 8 base samples.
- `identity` / `branch`: group_id, task, instruction, init_state, decision_step, state_hash, observation_hash, **intervention_id**, replicate, master, rng (noise key, env seed), restore_state_ok / restore_obs_ok, success, end_step, steps_executed, policy_calls, roundtrip and server ms, executed_horizon, first action chunk, first input hash, immediate outcome (eef / gripper after the first executed chunk), goal predicates at the end, final_state_hash, exception and traceback, wall_s, identity_ok (identity rows only).

Router inputs come only from `decision` rows and `features/`. Branch outcomes are labels only.

## 6. Tests

| test | where | result |
|---|---|---|
| synthetic estimator tests (null → no LOO gain while the naive oracle is inflated; state-specific vs task-specific heterogeneity; tie order; risk gate) | `research/adaptive_inference/tests/test_analyze_pilot.py` (`uv run pytest research/adaptive_inference/tests`) | 5 passed |
| simulator restore identity | `fork/probe_restore.py` (LIBERO venv, EGL) | 16/16 exact (§2) |
| operator identity | `validate_operators.py` (GPU) | all checks pass (§3) |
| end-to-end identity | identity branch in every decision state | §4 |
