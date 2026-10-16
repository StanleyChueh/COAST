# Pilot preregistration, amendment 1 (2026-10-17, before any rollout)

Amends [`PILOT_PREREG.md`](PILOT_PREREG.md) (frozen at tag `bimanual-utility-prereg-v1` = `f127e6d`).

Written before any RoboTwin rollout, planner run or VLA inference has been executed or observed. No simulator has run yet; asset and checkpoint downloads are still in progress.

**Unchanged:** every threshold, N, R, task, condition and estimand of gates A–F. **Added:** three-layer reporting, Gate G, Gate-0 checks, logging fields and outcome labels, all requested by the second research brief.

## 1. Three layers (reporting structure)

Layer A is a scene's physical opportunity, Layer B the demonstration's utilization of it, and Layer C the VLA's realization of Layer B. All are computed per analysis scene and aggregated per task with scene-cluster bootstrap CIs (`PILOT_PREREG.md` §7).

### Layer A: physical opportunity (planner / expert; physics steps at 250 Hz)

- `P_success(s) = 1[E2 succeeds] − 1[E1best succeeds]`. Every analysis scene has E2 = 1 by selection. Reported with the FEASIBLE / UNREACHABLE / UNDETERMINED split; UNREACHABLE is the lower bound for necessity.
- `P_eff(s) = (T_E1best − T_E2) / T_E1best` on scenes where E1best succeeds (= `BU_time`, prereg §7A).
- **Dependency-aware parallel bound** `T_crit(s)`, new and optimistic:
  - E2 is instrumented to log each primitive (`grasp_actor`, `place_actor`, `move_by_displacement`, `back_to_origin`, gripper open/close, `delay`) with its arm, its duration in physics steps, and the objects it involves. An object is involved if it is named in the call, or held by that arm at the time.
  - `T_crit` is the critical path through the DAG whose edges are (same arm → sequential) and (same object → sequential).
  - It ignores collisions and reachability, so it is an **optimistic, non-constructive** bound on what a dual strategy with E2's arm assignment could achieve.
  - `P_eff_ub(s) = (T_E1best − T_crit) / T_E1best`.
- E2ser is kept as defined (each simultaneous `move(a, b)` split into `move(a); move(b)`).

### Layer B: demonstration utilization (the scripted demonstrator E2 = the training-data generator)

E2 is recorded at the dataset frame rate (`save_freq` = 15 physics steps per frame, `demo_clean.yml`), on the same analysis scenes. Per scene:
- arms used;
- `SMP_E2`;
- frames;
- per-arm TCP path length;
- idle-arm path length;
- per-primitive arm roles (phase table).

**Demonstration overlap utilization** `DOU(s) = (T_E2ser − T_E2) / (T_E2ser − T_crit)`. It is defined when `T_E2ser − T_crit ≥ 0.10 · T_E2ser`, and is the share of the dependency-allowed parallel saving that the demonstrator realizes.

**Training-set check** (secondary; bandwidth permitting):
- Per-task SMP, frames and arms used from `lerobot/robotwin_unified` episodes of the 8 tasks (parquet only, no video), compared with E2 on our seeds.
- If it cannot be downloaded, this is recorded. E2 is the generator of those episodes by construction (RoboTwin `collect_data`).

### Layer C: VLA realization (π0.5)

- C1 (idle-arm clamp) and C2 (parallel realization) are unchanged from `PILOT_PREREG.md` §7.
- **C2 evaluability (new):** C2 is evaluable on a task only if median `SMP_E2 ≥ 0.20` on its FEASIBLE scenes, i.e. the demonstrations contain parallelism to realize. A task that fails this cannot pass C2.
- **C3 (phase-dependent utilization; descriptive, not gated).** L1 = stack_blocks_three.
  - Scenes are typed by block layout (one-sided vs mixed).
  - Per stage (block k), compare the arm E2 uses with the VLA's dominant arm (the arm with the larger TCP path during that stage), as stage-wise arm agreement.
  - Also compare SMP in a ±30-frame window around each E2 arm switch.

## 2. Gate G: source of the utilization gap (new; frozen now)

**DG, data gap meaningful:** Gate A passes **and**, on ≥ 2 tasks:
- the median over eligible scenes of `DOU ≤ 0.50`;
- the median `(T_E2ser − T_crit) / T_E2ser ≥ 0.20` (the opportunity is substantial);
- the bootstrap CI upper bound of the DOU median is `< 0.75`.

Because `T_crit` is non-constructive, DG supports at most a recommendation to test constructive (planner-generated) parallel demonstrations. It never supports a claim that a faster dual strategy exists.

**MG, model gap meaningful:** the original pilot pass, A ∧ B ∧ C ∧ D ∧ E (`PILOT_PREREG.md` §9), with the C2 evaluability rule above.

| G | condition | pilot decision line |
|---|---|---|
| G4 neither | ¬DG ∧ ¬MG | `PILOT NO-GO — NO MEANINGFUL BIMANUAL UTILIZATION GAP` |
| G1 data | DG ∧ ¬MG | `PILOT DATA-GAP — PHYSICAL OPPORTUNITY EXISTS BUT DEMONSTRATIONS DO NOT CONTAIN IT` |
| G2 model | ¬DG ∧ MG | `PILOT MODEL-GAP — DEMONSTRATIONS CONTAIN BIMANUAL UTILITY THAT π0.5 FAILS TO REALIZE` |
| G3 both | DG ∧ MG | `PILOT MIXED-GAP — BOTH DATA AND MODEL LIMITATIONS ARE MEANINGFUL` |

Only G2 and G3 lead to the cross-model gate F (X-VLA, small confirmatory subset on fresh seeds). A positive C1 without A remains NO-GO, as before: a harmful idle arm alone is TAO-DA territory.

## 3. Gate 0 additions

The `PILOT_PREREG.md` §4 checks stay. Mapping to the brief: G0.1 = scene reproducibility, G0.5 = success-checker validity, G0.6 = single-arm feasibility, G0.4 = fallback timing, G0.9 = throughput.

| id | check | pass criterion |
|---|---|---|
| G0.1 (extended) | hash of object poses, robot qpos, task state and **initial observation** (images, state, prompt) for `setup_demo(seed)` run twice | identical |
| G0.2b | π0.5 same scene + same noise key twice ⇒ identical first action chunk (bitwise); **different** key ⇒ first chunks differ (max abs diff > 1e-3) | 10/10 and 10/10 |
| G0.3b | clamp semantics: the clamped arm's commanded targets equal its hold target at every action; the active arm's commanded slice is bitwise equal to the policy's output; the number of `take_action` calls per policy chunk is unchanged | exact |
| G0.4b | per-arm TOPP-fallback counts logged for V2 and VCI | reported (VLA time = executed actions, never physics steps) |
| G0.9b | **no silent distribution change**: the observation dict sent to the policy (images, state, prompt) is the environment's unmodified observation in every condition. On VCI vs V2 with the same key, the first observation hashes are identical | exact |
| G0.10 | throughput: episodes/h per condition, policy-server VRAM, simulator RAM/VRAM | reported; drives the G0.9 reductions only |

**Gate-0 outcome** is exactly one of:
- `GATE0 PASS`;
- `GATE0 FAIL — FIXABLE` (fix, then rerun the failed check on the calibration block only);
- `GATE0 FAIL — SCIENTIFICALLY INVALID`, which is the prereg's BLOCKED.

## 4. Logging (every rollout; in addition to prereg §6)

- **Identity:** task, scene seed, noise master and per-call keys, condition, checkpoint id + revision, git SHAs (experiment repo, RoboTwin, LeRobot), initial-state hash, initial-observation hash.
- **Outcome:** success at the official and at 2× budget (object-centric and official predicates).
- **Counts:** policy queries, executed actions, environment physics steps.
- **Per arm, per action:** TCP pose, gripper state, joint displacement, active flag.
- **Derived:**
  - fractions left-active, right-active and both-active;
  - longest simultaneous-motion run;
  - TCP path length per arm;
  - arm–arm and idle-arm–object contacts.
- **Diagnostics:** TOPP fallbacks per arm, planner failures, exceptions with traceback, wall clock.

## 5. Analysis freeze

The analysis code lives in the experiment repository `~/Stanley_ws/bimanual-utility`. It is frozen (SHA + test output in `research/ANALYSIS_FREEZE.md` there) **before** any main-pilot rollout. Gate-0 calibration data are used only for the Gate-0 checks above.
