# Research status: Bimanual Utilization Gap

## Update 2026-10-18: Gate 0 outcome. GATE0 FAIL — SCIENTIFICALLY INVALID (BLOCKED)

**Why it is blocked.** Slot P1 exhausted its preregistered reserves on G0.8 (π0.5 operating range [10%, 95%]):

| P1 task | result |
|---|---|
| `pick_dual_bottles` | removed (amendment 3) |
| `place_cans_plasticbox` | V2 SR 20/20 |
| `place_burger_fries` | V2 SR 20/20 (amendment 5) |

`PILOT_PREREG.md` §9 defines this as BLOCKED, so **the π0.5 pilot was not run**. No pilot decision (NO-GO / DATA-GAP / MODEL-GAP / MIXED-GAP) is issued, and X-VLA was not prepared.

**Other findings.**
- lift_pot failed G0.5 twice: 0.867, then 0.90 after the one allowed predicate revision. Every disagreement sits in the official predicate's two-TCP tolerance.
- The calibration block shows that one-arm feasibility on RoboTwin's parallel tasks is mostly reach-blocked, so the physical (dual vs certified single-arm) layer is weakly identifiable on this benchmark.
- π0.5 succeeds at or near ceiling on the parallel tasks.

**Full reports** are in the experiment repo `~/Stanley_ws/bimanual-utility/research/`:
- `GATE0_REPORT.md`
- `RESEARCH_STATUS.md`
- `STOCHASTICITY_AUDIT.md`
- `UTILIZATION_DECOMPOSITION.md`
- `FAILURE_ANALYSIS.md`
- `PILOT_RESULTS.md`
- `ANALYSIS_FREEZE.md`

They are at commit `934eb33` (tag `gate0-blocked-v1`), with provenance in `fa35bc9`.

**Recommended next step: one separately preregistered kill test.** It measures demonstration → VLA efficiency (E_par, ρ) on the frozen, unseen analysis seeds of the four two-arm parallel tasks. Its stop rule closes the direction if π0.5 realizes the demonstrated efficiency.

---


Date: 2026-10-16. Branch `bimanual-utility-2027` (COAST). Session scope: audit + design only.
- no VLA trained;
- no rollout run;
- nothing installed;
- no COAST code changed;
- the frozen `research/adaptive_inference/` study untouched.

## Answer

**Yes, but in a narrower form than proposed.**

A nontrivial, testable version of the gap survives the 2024–2026 literature:
- matched-scene dual vs **planner-certified** single-arm execution;
- with same-policy re-sampling controls;
- on ≥ 2 VLA backbones.

No paper does it. The best any paper meets is 3.5 of the 8 stop criteria, and **none** meets the four substantive ones (matched dual-vs-single, feasibility control, physical vs policy separation, potential-vs-realized gap).

The formula as posed, `Perf_VLA(dual) − max Perf_VLA(single)`, is **not identified**:
1. A masked dual-arm VLA is out of distribution.
2. Every public RoboTwin VLA imitates scripted experts that are themselves mostly single-arm.

The defensible claim is therefore the **learning gap** (demonstrator-realized minus VLA-realized utility). It is measured by two contrasts that need no VLA single-arm counterfactual:
- an **idle-arm clamp** on scenes the demonstrator solves one-armed (harmful / redundant arm);
- **parallel-overlap realization** relative to the demonstrator on matched seeds, net of the VLA's generic slow-down (unrealized efficiency).

The physical potential is measured separately, by planners.

## What was done

1. **Literature** ([`LITERATURE.md`](LITERATURE.md)):
   - 28 arXiv IDs verified against the arXiv API;
   - full text, code or project page inspected for VoxAct-B, RoboTwin 1.0/2.0, BiCoord, DuoBench, TAO-DA and ACG-Bench.
2. **Novelty boundary** ([`NOVELTY_BOUNDARY.md`](NOVELTY_BOUNDARY.md)):
   - 8-criterion stop check;
   - the identification argument;
   - what is occupied;
   - kill conditions.
3. **Infrastructure** ([`INFRASTRUCTURE_AUDIT.md`](INFRASTRUCTURE_AUDIT.md)):
   - 6 benchmark candidates;
   - licences checked on GitHub/HF APIs;
   - 7 public RoboTwin VLA checkpoints;
   - RoboTwin source read statically at `ea8b211`.
4. **Task audit** ([`TASK_AUDIT.md`](TASK_AUDIT.md)):
   - a static analysis of all 50 RoboTwin expert programs ([`scripts/audit_robotwin_experts.py`](scripts/audit_robotwin_experts.py) → [`data/robotwin_expert_audit_ea8b211.md`](data/robotwin_expert_audit_ea8b211.md) / `.csv`);
   - 18 candidate tasks across all 8 categories;
   - 8 pilot tasks plus ordered reserves.
5. **Pilot preregistration** ([`PILOT_PREREG.md`](PILOT_PREREG.md)):
   - Gate 0 (9 validity checks);
   - conditions, estimands, a power calculation, gates A–F with numeric thresholds, and the compute budget.
   - Status: *designed, not yet frozen*. Freezing requires the analysis script and its synthetic tests.

## Findings

1. **No novelty collision.**
   - Arm routing is occupied (TAO-DA, MA-VLA, Task-Aware Affordance, Hand Priors).
   - Single-arm skill composition is occupied (TwinVLA, AnyBimanual, SkillVLA, ACG-Bench).
   - Coordination metrics are occupied (BiCoord, RoboEval, DuoBench).
   - Interference as a model property is occupied (TAO-DA, AnyPos).
   - Planner-level parallel time savings are occupied (RoboPARA).
   - None of them compares the same scene under dual and certified single-arm execution.
2. **Prompt claims corrected:**
   - VoxAct-B's "Need For a Second Arm" exists only as **rebuttal videos on the project page**. It is in neither arXiv version nor the PMLR camera-ready, and it has no numbers.
   - **DialogVLA could not be found**.
   - BiManiBench v2 is now **ST-BiBench**, an MLLM benchmark.
   - RoboTwin 1.0's "Shoes Place must be completed using both arms" is **asserted, never tested**. The pilot tests it (P2).
3. **RoboTwin 2.0 is far less bimanual than its label** (static audit of the experts):
   - **26/50** tasks are executed with **one** arm, chosen by object position;
   - **9** use one or two arms depending on the scene;
   - **15** always use two;
   - **23** contain simultaneous two-arm motion;
   - **3 success predicates define success as two-armed** (lift_pot, grab_roller, scan_object);
   - 5 more reference the expert's arm.

   BiCoord independently measured a dataset SMP of 26%. This is useful in itself (a benchmark-audit result) and it shapes the design. It is also the **main risk**: VLAs can only realize parallelism their demonstrations contain.
4. **Infrastructure is available and clean enough:**
   - RoboTwin 2.0 is MIT, its assets are MIT, and its per-arm expert API makes one-arm experts writable with planner certification;
   - scene replay works by seed;
   - per-arm action masking is trivial (14-D absolute joint targets);
   - there are public multi-task checkpoints for independently designed VLAs: π0.5 (Apache), X-VLA (Apache), LingBot-VLA v2, Motus, SmolVLA, and a deterministic-head Qwen3-VL-OFT;
   - π0.5 already supports keyed noise in LeRobot; X-VLA needs a small patch.
5. **Measurement traps found in the code** (all handled in Gate 0):
   - RoboTwin executes each joint target by per-arm TOPP with a **fixed 50-step fallback**, so a still or clamped arm can inflate physics time. VLA time is therefore measured in policy actions.
   - Seed reproducibility was only fixed upstream on 2026-09-24 and must be verified.
   - Evaluation keeps only seeds that the dual expert solves, which defines the population.
6. **Importance is conditional:**
   - A top-conference paper needs a learning-gap effect that clears re-sampling noise on certified-feasible scenes and replicates on X-VLA.
   - If only the physical-potential audit holds, the honest output is a benchmark-audit paper, not a VLA paper.

## The single smallest preregistered pilot that can falsify it

**Setup:** π0.5 (`SidneyXie/pi05_robotwin`) on RoboTwin 2.0 `demo_clean`, 8 tasks:
- 2 redundant;
- 2 efficiency;
- 1 hold;
- 1 handover;
- 1 synchronous / constrained;
- 1 long-horizon.

**Scenes:** matched seeds that the dual expert solves:
- 60 scenes per redundant task;
- 30 for the others.

**Steps:**
1. **Planner layer** (cheap, CPU-heavy): dual expert, serialized dual, left-only and right-only experts per scene. This yields necessity share, efficiency potential and scene classes (gates A, B, E).
2. **VLA layer:**
   - dual ×4 noise replicates, all scenes;
   - idle-arm clamp ×4 (common random numbers) on the redundant tasks plus the long-horizon one-arm scenes;
   - descriptive masked/used-arm clamps, the first thing cut if compute is short.
3. **Falsifiers:**
   - C1: \|Δ_clamp\| ≥ 10 pp at p < 0.01, consistent across both redundant tasks, and larger than the resampling null.
   - C2: the VLA realizes ≤ 60% of the demonstrator's simultaneous-movement share, with ≥ 15% excess slow-down beyond its single-arm-task inflation.
   - **If neither passes, the VLA-specific direction stops.**
4. **Gate F:** only on a pass, rerun the passing contrast(s) on X-VLA.

**Cost:**
- ≈ 1,560 core VLA rollouts plus ≤ 1,080 descriptive ones;
- ≈ 960 planner runs;
- ≈ 1,000 Gate-0 runs;
- ≈ 10 h of compute on 2× 4090 under stated, unmeasured throughput assumptions.

Implementation order:
1. Separate workspace `~/Stanley_ws/bimanual-utility/`: RoboTwin `ea8b211` + assets + LeRobot fork.
2. Env wrapper: clamp/mask, 2× budget, dual-limit success logging, per-arm activity, keyed noise, JSONL.
3. One-arm experts and object-centric predicates for the 8 tasks; IK reachability probe.
4. `analyze_bimanual_pilot.py` + synthetic tests (null, resampling-only, clamp effect, parallel gap). Then **freeze**.
5. Gate 0 on the calibration block, then the main run, then the frozen analysis.

## If the pilot eventually passes (method families only; not designed, not implemented)

- Idle-arm suppression or utility-aware arm activation. This applies only if C1 shows a harmful second arm, and it must differ from TAO-DA's instruction routing by being driven by measured utility.
- Selective decoupling/coupling of the arm streams. Only if C1 is linked to cross-arm entanglement (AnyPos-style).
- Parallelism-preserving action generation or phase-conditioned coordination. Only if C2 shows serialization that the data does contain.
- Data-side fixes (parallelized demonstrators). Only if the gap is a data gap, in which case it is not a VLA method.

## Explicitly not done

- No VLA trained.
- No new architecture.
- No benchmark built.
- No experiment run.
- Nothing installed or vendored into COAST.
- No novelty claimed from the prompt.
- Physical necessity never inferred from a VLA.
- The adaptive-inference study untouched.

DIRECTION VIABLE — ready to implement preregistered pilot.
