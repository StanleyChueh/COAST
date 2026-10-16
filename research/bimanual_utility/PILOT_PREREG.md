# Pilot preregistration: Bimanual Utilization Gap (design only, NOT yet frozen)

Status (2026-10-16): **design**. Nothing has been run, and no RoboTwin rollout of any kind has been observed by us.

**Freeze procedure.** Before Gate 0 is executed:
- record `sha256` of this file and of the (to-be-written) analysis script `analyze_bimanual_pilot.py` and its synthetic tests in `PREREG_FROZEN.txt`.

**What Gate 0 may still change** (fixed rules, §4):
- task substitution, in the fixed reserve order;
- one revision of an object-centric predicate, using calibration data only;
- compute reductions, in the fixed order.

Nothing else may change after freezing: no thresholds, estimands, N, R or tasks.

Lesson carried over from `research/adaptive_inference` (preregistered NO-GO): **re-sampling the same policy manufactures apparent effects.** Therefore:
- every VLA condition has noise replicates;
- every contrast is tested against same-condition replicate disagreement;
- no per-scene max over noisy conditions (naive oracle) is ever used as an estimate.

## 1. Questions

- **Q1 (physical potential, planner-anchored).** In matched scenes, how often is one arm physically sufficient (certified), and how much faster is the demonstrated dual strategy than the best certified single-arm strategy?
- **Q2 (learning gap: harmful / redundant arm).** On scenes where the demonstrator uses one arm, does clamping the arm it leaves idle change a VLA's success? (Cases 3 and 5)
- **Q3 (learning gap: unrealized parallelism).** On scenes where the demonstrator moves both arms simultaneously, does the VLA realize that parallelism? Measured as simultaneous-movement share and frames-to-success relative to the demonstrator, net of the VLA's generic slow-down. (Case 2)
- **Q4 (scene-level structure).** Does necessity / advantage vary across scenes *within* a task?

Not addressed by this pilot:
- Case 1, which needs a VLA trained on single-arm demonstrations;
- Case 4, phase adaptation, which is exploratory only.

See `NOVELTY_BOUNDARY.md` §3 for why a masked dual-arm VLA never serves as single-arm evidence.

## 2. Stack (pinned at freeze time)

- **RoboTwin 2.0** @ `ea8b211` (MIT), `demo_clean` setting, Aloha-AgileX, official cameras, official instruction generator. Assets: HF `TianxingChen/RoboTwin2.0` (MIT), revision hash recorded at freeze.
- **LeRobot** (local fork `1e5f2049`) `envs/robotwin.py`, in a separate workspace `~/Stanley_ws/bimanual-utility/` (see `INFRASTRUCTURE_AUDIT.md` §6).
- **Primary VLA:** `SidneyXie/pi05_robotwin` (Apache-2.0); HF commit hash recorded at freeze. Settings: chunk 50, all 50 actions executed per call (model-card default), 10 flow steps.
- **Second VLA (gate F only):** `2toINF/X-VLA-RoboTwin2` (Apache-2.0), with per-call keyed `torch.Generator` noise (patch verified bitwise before use).
- **Policy noise:**
  - key = (master m, task index, seed, policy-call index) → `SeedSequence` → `PCG64` → N(0,1) float32 of the model's noise shape;
  - passed to `predict_action_chunk(noise=...)`;
  - the server logs key and noise sha256 per call.
- **Common random numbers:** within a scene and replicate m, all conditions use the same key sequence.

## 3. Scenes

- For each task, iterate RoboTwin evaluation seeds from 100000 upward. A seed is valid iff the official dual expert succeeds (the standard `expert_check`).
- The first **10 valid seeds** form the **calibration block** (Gate 0 only, never analysed).
- The next valid seeds form the **analysis set**:
  - **N = 60** for R1 (place_phone_stand) and R2 (move_pillbottle_pad);
  - **N = 30** for P1 (pick_dual_bottles), P2 (place_dual_shoes), S1 (place_bread_skillet), H1 (handover_block), C1 (lift_pot) and L1 (stack_blocks_three).
- **The population is "scenes the dual expert solves"**, and conclusions are restricted to it.

N and R were fixed from the power calculation in §8, which uses only stated assumptions and no pilot data.

## 4. Gate 0: infrastructure validity (calibration block; all must pass before the main run)

| id | check | pass criterion | if it fails |
|---|---|---|---|
| G0.1 | scene determinism: `setup_demo(seed)` twice | identical initial actor poses and robot qpos (exact) on 10/10 seeds per task | **BLOCKED** (matched scenes impossible) |
| G0.2 | planner determinism: E2 run twice | identical success and physics-step count on ≥ 95% of seeds | run all planner conditions 3× and use majority success / median time (fallback fixed now) |
| G0.3 | closed-loop determinism: π0.5 dual, same key, twice (10 seeds × R1, P1) | identical action-trajectory hash on ≥ 90% | if success disagrees on > 10% of same-key pairs, CRN pairing is void and analyses switch to the unpaired versions in §7 (fixed now). Report the divergence rate either way |
| G0.4 | time accounting: TOPP-fallback count per arm per `take_action`, per condition | reported | VLA physics time is never a primary metric. If fallback frequency differs between V2 and VCI by > 5 pp, VLA physics time is dropped entirely |
| G0.5 | object-centric predicate vs official predicate on E2 and π0.5-dual calibration rollouts | ≥ 95% agreement per task | one revision allowed, using calibration data only. Second failure ⇒ task replaced by the next reserve |
| G0.6 | our one-arm expert reproduces the official one-arm expert on scenes where the official expert uses one arm (R1, R2, L1 one-arm scenes) | ≥ 95% success | fix the scripting (calibration only). Persistent failure ⇒ **BLOCKED** for planner-based feasibility |
| G0.7 | clamp validity | clamped arm max joint deviation from its hold target < 0.02 rad throughout | fix the wrapper |
| G0.8 | operating range: π0.5 dual SR on the calibration block (10 seeds × 2 replicates) | within [10%, 95%] for R1, R2, P1, P2, S1, L1 (H1 and C1 are exempt: descriptive/planner roles) | replace with the next reserve (`TASK_AUDIT.md` §3 order) and rerun Gate 0 for it |
| G0.9 | throughput | projected main run ≤ 30 h on 2× 4090 | reductions in this order only: (1) VML/VMR R 2 → 1; (2) N for P/S/H/C/L 30 → 24. R for V2/VCI and N for R1/R2 are never reduced |

## 5. Conditions

**Planner** (deterministic; see G0.2). Run on every analysis scene:

| id | definition |
|---|---|
| E2 | official dual expert (the demonstrator) |
| E2ser | E2 with every simultaneous `self.move(a, b)` split into `self.move(a); self.move(b)` (same motions, no overlap) |
| E1L / E1R | one-arm expert (ours, RoboTwin API), left / right arm only. The other arm stays at its initial pose. No time limit |
| E1best | derived per scene: the faster *successful* of E1L / E1R (valid because planners are deterministic and certified) |

**VLA (π0.5).** All rollouts run to **2× the official step limit**. Success is recorded both at the official limit and at 2× from the same rollout, so EXTENDED budgets come for free and "dual is faster" is never confused with "single timed out".

| id | definition | scenes | replicates (masters) |
|---|---|---|---|
| V2 | dual, unmodified | all tasks | R = 4 (m = 1–4) |
| VCI | **clamp the idle arm**: the arm E2 does not move in this scene (joint path < 1e-3 rad). Its 7 action dims are replaced by its initial joint/gripper state. Observations unmodified | R1, R2 (all scenes); L1 scenes where E2 uses one arm | R = 4 (m = 1–4, CRN with V2) |
| VCU | clamp the arm E2 *uses* | R1, R2 | R = 1 (m = 1). Descriptive: mask validity and arm-substitution rate |
| VML / VMR | action-masked one-arm (left only / right only) | all tasks | R = 2 (m = 1, 2). **Descriptive only; never used as physical or policy single-arm evidence** |

**BEST-SINGLE for masked VLA conditions** (descriptive) is chosen by cross-fitting:
- pick the arm on even seeds and score on odd seeds, and vice versa;
- either a fixed arm per task or the side rule, whichever wins on the training half.

A per-scene max over noisy conditions is never used.

## 6. Measurements (per rollout)

- **Y1 / Y2:** success at the official / 2× limit.
  - Primary predicate: object-centric (`TASK_AUDIT.md` §2), validated by G0.5.
  - The official predicate is also logged for V2 and VCI.
- **F:** policy actions to first success (censored at 2× the limit). For planners, **T** is physics steps to success.
- **Per arm, per action:** joint-path length, TCP path length, gripper command, active flag.
  - Active = joint displacement > 0.01 rad in that action, or gripper closed (BiCoord's definition, thresholded).
- **SMP** (simultaneous-movement percentage) = #actions with both arms active / #actions with ≥ 1 arm active.
  - For E2 it is computed on the expert trajectory resampled to the dataset's frame rate, so units match the VLA's actions.
- **Idle-arm motion:** TCP path length (m) of the arm E2 leaves idle.
- **Contacts:**
  - left-arm ↔ right-arm link contacts;
  - idle-arm ↔ object contacts (`get_scene_contact`);
  - stage completion (objects placed) for multi-object tasks.
- **Provenance:** TOPP-fallback counts, noise keys, initial-state hash, exception/traceback.

## 7. Estimands and primary analyses

All CIs are 95%, scene-cluster bootstrap with 10,000 resamples (replicates nested within scene). No estimate uses a per-scene max over noisy conditions.

**A. Physical potential (planner; Q1, Q4).** Per task:
- **necessity share** `NS` = #UNREACHABLE / N (Wilson CI). `1 − FEASIBLE share` is the upper bound, also reported.
- **efficiency potential** `BU_time(s) = (T_E1best − T_E2) / T_E1best` on FEASIBLE scenes (median, bootstrap CI).
- **overlap value** `OV(s) = (T_E2ser − T_E2) / T_E2ser`.
- **scene classes:** redundant / advantageous / necessary / undetermined (rules in `TASK_AUDIT.md` §4).

**C1. Idle-arm clamp (Q2), primary contrast 1.** Pool R1 ∪ R2 (120 scenes); per task also reported.

- `Δ_clamp = mean_s [ mean_m Y1(VCI,s,m) − mean_m Y1(V2,s,m) ]`.
- **Test:** paired sign-flip permutation on scene-level differences (10,000), two-sided.
- **Noise calibration:**
  - `Δ_null = mean_s [ mean_{m∈1,2} Y1(V2) − mean_{m∈3,4} Y1(V2) ]`, with its bootstrap distribution;
  - same-condition disagreement `D_same = P(Y1(V2,s,m) ≠ Y1(V2,s,m'))`, m ≠ m';
  - different-condition disagreement `D_cond = P(Y1(VCI,s,m) ≠ Y1(V2,s,m))`, same m.
- **Secondary:**
  - Δ at the 2× limit;
  - Δ in restricted mean actions-to-success (failures censored at 2×);
  - V2 idle-arm path length;
  - within V2, the association between idle-arm motion and failure (scene-stratified).
- Unpaired fallback if G0.3 voids CRN: two-sample permutation within scene.

**C2. Parallel realization (Q3), primary contrast 2.** P1 and P2, FEASIBLE scenes with ≥ 1 successful V2 replicate. S1 is reported alongside.

- `ρ(s) = mean_{successful m} SMP(V2,s,m) / SMP(E2,s)`.
- **Inflation** `I(s) = mean_{successful m} F(V2,s,m) / F(E2,s)`, with E2 in dataset frames.
- **Generic inflation baseline** `I_red` = median of `I(s)` over R1 ∪ R2 (single-arm demonstrations: no parallelism to lose).
- **Parallel-specific excess inflation** `E_par = median_s I(s) / I_red` (bootstrap the ratio jointly).
- **Replicate noise** `σ_rep` = median within-scene SD of `SMP(V2,s,·)/SMP(E2,s)`.

**E (scene structure).** Per task, the share of each scene class. Within-task variation counts when two classes each hold ≥ 20% of scenes.

**Descriptive (not gated):**
- VML / VMR / VCU success;
- cross-fitted best-single;
- the arm-substitution rate (VCU: does the VLA use the other arm when the expert's arm is clamped, on scenes where the other arm is certified FEASIBLE?);
- VLA V2 SR by scene class in H1 / C1 / L1;
- Case 4 per-phase SMP in L1 (exploratory).

## 8. Power (assumptions stated; not from pilot data)

C1 (binary outcome, scene-level paired difference over R = 4):
- Assume p ≈ 0.65 (public π0.5 SR for R1/R2 is 66–74%), CRN correlation c ≥ 0 and effect heterogeneity τ² ≈ 0.02.
- Then σ_d² ≈ τ² + 2p(1−p)(1−c)/R ≈ 0.02 + 0.114 ⇒ σ_d ≈ 0.37 (c = 0, worst case) or 0.31 (c = 0.3).
- With N = 120 scenes, SE ≈ 0.034–0.028. At Δ = 10 pp, z ≈ 2.96–3.53, so power ≈ 0.65–0.83 at two-sided α = 0.01 (critical z 2.576). At Δ = 15 pp, power is ≥ 0.97.
- N = 60 per redundant task was chosen for this reason. 10 pp is the prompt's "substantial" level.

C2 (continuous ratio, N = 30 per task): a ratio gap of 0.4 (ρ = 0.6) at a between-scene SD of 0.3 gives SE ≈ 0.055, so z ≈ 7; power ≈ 1 if the effect exists at that size.

## 9. Hard GO / NO-GO gate (prompt §13; evaluated by the frozen analysis script)

Multiplicity: two primary contrasts (C1, C2), each at α = 0.01 (Bonferroni over 2 at family α = 0.02).

| gate | criterion (all quantities from §7) |
|---|---|
| **A: nontrivial utility** | ≥ 2 distinct task families among {P1, P2, S1, H1, C1, L1} with either NS ≥ 10% (Wilson lower bound ≥ 3%), or median BU_time ≥ 0.20 with CI lower bound ≥ 0.10 on ≥ 10 FEASIBLE scenes |
| **B: not physical triviality** | the contrast that passes C is estimated on certified-FEASIBLE scenes. R1/R2 are FEASIBLE by construction; C2 is restricted to FEASIBLE scenes |
| **C: VLA-specific gap** | **C1 passes:** \|Δ_clamp\| ≥ 10 pp, permutation p < 0.01, and Δ has the same sign in R1 and R2. **or C2 passes:** median ρ ≤ 0.6 with CI upper < 0.8, **and** E_par ≥ 1.15 with CI lower > 1.0, in ≥ 1 of P1/P2, with ρ < 1 in the other |
| **D: robust to noise** | for C1: \|Δ_clamp\| exceeds the 97.5th percentile of the bootstrap distribution of \|Δ_null\|, **and** D_cond − D_same ≥ 0.05 (CI excludes 0). For C2: (1 − median ρ) ≥ 2 σ_rep |
| **E: not task-ID trivial** | ≥ 1 task among {P1, P2, H1, L1} shows within-task variation (two classes each ≥ 20%), **and** the passing C contrast holds in ≥ 2 tasks (C1: R1 and R2; C2: P1 and P2, direction-consistent) |
| **F: cross-model** | not part of the pilot verdict. If A–E pass, rerun **only the passing contrast(s)** on X-VLA-RoboTwin2 with the identical protocol. The direction must replicate at ≥ ½ the π0.5 magnitude with p < 0.05. **Only then** is a general-VLA claim allowed |

**Verdicts:**
- **PILOT PASS (π0.5)** = A ∧ B ∧ C ∧ D ∧ E. Then F decides **GO**.
- **NO-GO** = any of A–E fails.
  - A passes with C failing means a *data/design gap only*: a benchmark-audit result, not a VLA paper.
  - C passing with D failing means *noise*.
- **BLOCKED** = a Gate-0 failure with no preregistered remedy (G0.1, G0.6), exceptions in > 5% of rollouts of any condition, or reserves exhausted for a slot.

**Exceptions:** rerun once with an identical key. If still failing, record the rollout and drop that (scene, replicate) pair from paired contrasts.

## 10. Compute budget (estimate; Gate 0 G0.9 measures it)

| block | rollouts |
|---|---|
| planner: 240 analysis scenes × {E2, E2ser, E1L, E1R} (+ duplicates per G0.2) | 960 (1,920) |
| V2: 240 scenes × 4 | 960 |
| VCI: 120 × 4 + L1 one-arm scenes (≤ 30) × 4 | ≤ 600 |
| VCU: 120 × 1 | 120 |
| VML/VMR: 240 × 2 × 2 | 960 |
| **VLA total** | **≤ 2,640** |
| Gate 0 (8 tasks × 10 seeds: planners ×2, π0.5 ×2 + same-key repeat, VCI) | ≈ 1,000 |

At an assumed ~1 min per VLA rollout and 6–8 parallel simulator processes with 2 π0.5 servers per 4090, the VLA part is ≈ 6–8 h, plus ≈ 2–3 h of planner runs. This is within the 30 h cap. Throughput is unmeasured and is the main uncertainty (G0.9 has the reductions).

## 11. Explicitly not done

- No VLA training or fine-tuning.
- No architecture changes or new method; the method families are listed in `RESEARCH_STATUS.md` only.
- No task chosen or dropped by effect size; G0.8 uses dual-condition SR only.
- No threshold, N or R changed after freezing.
- No physical-necessity claim from any VLA condition.
- No COAST code modified.
- No change to the frozen `research/adaptive_inference/` study.
