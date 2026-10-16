# Novelty boundary: Bimanual Utilization Gap

Date: 2026-10-16. Inputs:
- [`LITERATURE.md`](LITERATURE.md);
- the static audit of RoboTwin 2.0 expert programs ([`data/robotwin_expert_audit_ea8b211.md`](data/robotwin_expert_audit_ea8b211.md));
- the adaptive-inference NO-GO (`research/adaptive_inference/RESEARCH_STATUS.md`).

## 1. Four quantities that must not be confused

For a matched initial scene s of a task, the four quantities below are kept apart. *Performance* means success, or time to success.

| quantity | definition | who can certify it | what it is **not** |
|---|---|---|---|
| **Physical capability** | ∃ a one-arm controller that succeeds from s (with this embodiment, gripper and success predicate) | a validated planner/expert that succeeds. Failure of a planner proves nothing unless the required poses are kinematically unreachable (IK has no solution) | a failed VLA, masked or not |
| **Policy capability** | this VLA succeeds from s with one arm | a VLA trained or validated for one-arm execution | an action-masked dual-arm VLA (out of distribution: it never saw one-arm demos for this task) |
| **Bimanual potential** | Perf(best validated dual strategy \| s) − Perf(best validated single-arm strategy \| s) | planners/experts run on the same seed | a property of a VLA |
| **Bimanual utilization** | how much of the potential a VLA *realizes* | VLA rollouts with noise replicates, against the references above | "the VLA fails at a task that needs two arms" |

## 2. Hard novelty stop check (prompt §3)

Criteria:
1. modern VLAs;
2. multiple bimanual task families;
3. dual vs matched left-only/right-only;
4. single-arm feasibility controlled;
5. success + efficiency;
6. separates physical necessity from policy inability;
7. quantifies the potential-vs-realized gap;
8. ≥ 2 VLA backbones.

| paper | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | met |
|---|---|---|---|---|---|---|---|---|---|
| VoxAct-B (2407.04152) | ✗ | ✓ | ✗ (rebuttal video, one arm) | ✗ | ✗ | ✗ | ✗ | ✗ | 1 |
| RoboTwin 1.0/2.0 | ✓ (2.0 leaderboard) | ✓ | ✗ | ✗ | ✗ (success only) | ✗ | ✗ | ✓ | 3 |
| BiCoord (2604.05831) | ✓ | ✓ | ✗ | ✗ | partial (TL) | ✗ | ✗ | ✓ | 3.5 |
| DuoBench (2606.11901) | ✓ | ✓ | ✗ | ✗ (taxonomy assertion) | ✗ | ✗ | ✗ | ✓ | 3 |
| TAO-DA (2609.33197) | ✓ | ✓ | ✗ (separate single-arm task suite) | ✗ | ✗ | ✗ | ✗ | ✓ (GR00T, π0.5 baselines) | 3 |
| ACG-Bench (2610.06184) | ✓ | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ | 3 |
| Co-VLA (2606.20285) | ✓ | ✓ | ✗ | ✗ | ✓ (time) | ✗ | ✗ | partial | 3.5 |
| RoboPARA (2506.06683) | ✗ (LLM planner) | ✓ | partial (parallel vs sequential plans) | n/a | ✓ (time) | ✗ | ✗ | ✗ | 2.5 |
| SkillVLA (2603.03836) | ✓ | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | partial | 2.5 |

Criteria 1, 2, 5 and 8 are generic benchmark properties. **Criteria 3, 4, 6 and 7 are the substance of the claim, and no paper meets any of them.** No collision, so the STOP condition is not triggered. This must be re-checked before any write-up, and specifically against BiCoord follow-ups, DuoBench v2 and RoboTwin 2.x.

## 3. The identification problem (the main design constraint)

The prompt's formula is

    U_bi(s) = Perf(dual | s) − max(Perf(left | s), Perf(right | s))

Evaluated on a VLA, it is **not identified** as a statement about bimanual utility, for two independent reasons.

1. **The VLA's single-arm condition is out of distribution.** A dual-arm VLA with one arm masked has never seen one-arm demonstrations for the task. Its failure measures robustness to amputation, not physical or policy single-arm capability. The prompt anticipates this, and we never use a masked VLA as evidence of necessity.

2. **The VLA's bimanual behaviour is inherited from its demonstrator.** Every public RoboTwin 2.0 checkpoint imitates the benchmark's scripted experts. By our static audit of those experts at RoboTwin `ea8b211`:
   - **26 of 50** "dual-arm" tasks are executed with **one arm**, chosen by the object's x-position;
   - **9** use one or two arms depending on the scene;
   - only **15** always use both arms;
   - **23** contain at least one simultaneous two-arm motion.

   BiCoord independently measures dataset SMP = 26%. A VLA cannot be blamed for not realizing a bimanual advantage its demonstrations never show.

The total "gap" between physical potential and what a VLA realizes therefore decomposes into:

    Potential(s) − Realized_VLA(s)
      = [Potential(s) − Realized_demo(s)]      ← DATA / DESIGN gap (property of the demonstrator)
      + [Realized_demo(s) − Realized_VLA(s)]   ← LEARNING gap (property of the VLA)

**Only the learning gap is a VLA claim.** The data gap is a benchmark finding: useful, but not "VLAs fail to use their second arm".

The learning gap is identifiable without any VLA single-arm counterfactual through two contrasts. Both are measured on matched seeds with noise replicates:

- **Idle-arm clamp (Cases 3 and 5: redundant or harmful arm).** The setting is scenes where the demonstrator uses only one arm. Clamp the arm the demonstrator leaves idle at its hold pose, and compare VLA success and time with the unclamped dual condition (common random numbers, R replicates).
  - Clamping an arm that the demonstrator never moves is a *minimal* intervention. The policy's intended output for that arm is "stay", so the contrast isolates the causal effect of the VLA's unnecessary second-arm motion.
  - Clamp helps ⇒ the second arm is harmful *for the VLA* (category D).
  - Clamp is neutral ⇒ redundant, and the arm is handled correctly.
- **Parallel-overlap realization (Case 2: unrealized efficiency).** The setting is scenes where the demonstrator moves both arms at once. Compare the VLA's simultaneous-movement share and frames-to-success with the demonstrator's on the same seed, normalized by the VLA's generic slow-down on single-arm tasks.
  - If the VLA reproduces the demonstrator's single-arm segments at near-demo speed but serializes the parallel ones, the efficiency utility available in its own training data goes unrealized.

**Case 1** (VLA dual gains more than its single-arm counterpart) and a *policy-capability* single-arm reference both need a VLA trained on validated single-arm demonstrations. That is training and out of scope for this session and for the pilot. It is the natural follow-up only if the pilot passes.

## 4. What we claim, and what we do not

**Claimed contribution, if the pilot passes:**
1. A **matched-scene, planner-certified** measurement of physical bimanual potential per scene, with necessity / advantage / redundancy classified per scene, not per task.
   - Physical potential is measured as success necessity, and as the completion-time advantage on scenes where one arm is certified feasible.
   - This also yields a benchmark finding: how many "bimanual" scenes actually need two arms.
2. A **decomposition of the potential–realized gap** into a data/design part and a learning part, with the learning part measured by the two contrasts above under explicit same-policy resampling controls.
3. Evidence on **≥ 2 independently designed VLA backbones**. π0.5 and X-VLA both have public multi-task RoboTwin 2.0 checkpoints.

**Not claimed (occupied):**
- arm routing or role prediction (TAO-DA, MA-VLA, Task-Aware Affordance, Hand Priors);
- single-arm skill composition (TwinVLA, AnyBimanual, SkillVLA, ACG-Bench);
- coordination metrics as such (BiCoord, RoboEval, DuoBench);
- interference as a model property (TAO-DA, AnyPos);
- planner-level parallel scheduling (RoboPARA);
- dynamic interaction (DynaMAC);
- "bimanual VLA X is faster than VLA Y" (Co-VLA).

We reuse BiCoord's SMP definition rather than inventing a coordination metric.

## 5. What would make the direction not novel or not important (fixed now, before any data)

1. **Collision:** a paper appears that runs matched dual-vs-validated-single-arm execution for VLAs with feasibility certification. Re-check at every gate.
2. **Pure data gap:**
   - clamping the idle arm changes VLA outcomes by less than resampling noise, **and**
   - the VLA's simultaneous-movement share is ≥ 0.8 of the demonstrator's on parallel scenes.

   Then all of the "utilization gap" is in the demonstrations, and the work collapses to a benchmark audit. That is a workshop/benchmark-track result, not a VLA paper.
3. **Physical triviality:** planner-certified single-arm feasibility is ≈ 0% or ≈ 100% in every family (no advantageous class), or the efficiency potential is < 20% everywhere.
4. **Task-ID triviality:** within every task, all scenes fall into the same class (necessary / advantageous / redundant). Utility would then be a task label.
5. **Noise:** the clamp contrast is not larger than same-condition replicate disagreement. This is the lesson of the adaptive-inference NO-GO, where apparent complementarity came from re-sampling.
6. **Single-backbone artifact:** an effect present on π0.5 is absent on X-VLA. The "general VLA" claim dies; a model-specific finding remains.

## 6. Importance (honest assessment)

- **Literature:** survives. The combination (matched scenes + certified feasibility + noise control + realized-vs-available) is absent from every paper above.
- **Scientific interest:** conditional.
  - The interesting outcomes are the learning-gap contrasts (harmful idle arm; serialized parallelism).
  - Prior signals are mixed:
    - SkillVLA and TAO-DA report idle-arm drift or empty grasps qualitatively, which favours a measurable clamp effect;
    - but BiCoord's 26% dataset SMP means the demonstrations contain little parallelism, so the realizable efficiency headroom inside the data may be small.
- **Top-conference strength:** only if the pilot finds a learning-gap effect that clears noise, sits on certified-feasible scenes, and replicates on X-VLA. Otherwise the defensible output is a benchmark audit ("how bimanual are bimanual benchmarks").
