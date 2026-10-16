# Literature audit: bimanual utility / arm necessity (as of 2026-10-16)

Scope: everything in the prompt's list, plus work found by search on these topics:
- dual-vs-single-arm comparison, arm necessity, arm contribution or ablation, bimanual utility;
- role grounding, dynamic arm allocation;
- coordination efficiency and cross-arm interference.

Every arXiv ID below was checked against the arXiv API on 2026-10-16 (title, first-submission date, comments).

Read depth is marked in the Paper column:
- **F**: full text, code or project page inspected;
- **A**: abstract or secondary summary only.

Rows are ordered from closest to us down to context only.

## Verification notes (claims in the prompt that did not check out as stated)

| prompt item | what was found |
|---|---|
| **VoxAct-B** "already includes a *Need For a Second Arm* experiment" | The experiment exists only as **rebuttal videos on the project page**. The section "Need For a Second Arm to Open Jar and Drawer" plays `static/videos/rebuttal/2024_08_05_open_jar_bimanual_manipulation_needed.mp4`, which shows one arm failing to open a real jar or drawer. arXiv v1, arXiv v2 and the PMLR camera-ready (v270, liu25i) carry only the one-line claim "the real-world jar and drawer cannot be opened without the use of a second arm". None of them has a single-arm condition, a number, a matched scene or a feasibility control. It is still the closest *conceptual* precedent: a qualitative necessity demonstration on 2 real tasks |
| **DialogVLA** | **Not found.** Three searches came up empty: the exact name, and "dialog/dialogue VLA dual-arm" in standard and extended modes. It is treated as unverified. The prompt's point it stands for (explicit arm-role prediction is not novel) is covered by TAO-DA, MA-VLA, Task-Aware Bimanual Affordance and Policy-Induced Hand Priors |
| **BiManiBench** | arXiv 2602.08392 **v2 is retitled "ST-BiBench: Benchmarking Multi-Stream Multimodal Coordination in Bimanual Embodied Tasks for MLLMs"**. It evaluates MLLMs, not VLAs |
| **DynaBench** | Exists, as the RLBench-based benchmark in DynaMAC (2607.22119). The name collides with unrelated Meta / protein-docking "Dynabench" |
| **RoboTwin "single-arm and dual-arm variants"** | Partly true, in RoboTwin 1.0 (2504.13059). Some tasks "have both dual-arm and single-arm versions, like Container Place and Empty Cup Place. For these dual-arm versions, the appropriate arm is selected based on the object's initial position", and "Shoes Place … must be completed using both arms". The variants are different task versions (and in the real world, different embodiments), not matched counterfactuals. The necessity of Shoes Place is **asserted, not tested** |

## Main table

| Paper | Year / Venue | Task / environment | Backbone(s) | Bimanual property studied | Single-arm baseline? | Matched dual-vs-single condition? | Controls physical single-arm feasibility? | Controls stochastic policy noise? | Task- or state-level | Main contribution | Overlap with us | Exact novelty boundary |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **VoxAct-B** [F] 2407.04152 | 2024 / CoRL (PMLR 270) | RLBench + second Panda: Open Jar, Open Drawer, Put Item in Drawer, Hand Over Item. Real: 2×UR5 jar, drawer | VoxPoser-style voxel policy (PerAct-derived), VLM crops | acting vs stabilizing role asymmetry | no (Bimanual PerActs = two per-arm policies, still two arms) | no. Project-page rebuttal videos show one arm failing on the real jar/drawer | assertion only | 5 training seeds, 25 test episodes. No inference-noise replicates | task | role-factored acting/stabilizing policy | necessity of the 2nd arm is *claimed* for 2 tasks | no measurement of single-arm feasibility or of dual-vs-single utility, no VLA, no efficiency |
| **RoboTwin 1.0** [F] 2504.13059 | 2025 / CVPR Highlight | 15 sim tasks (SAPIEN) + real COBOT Magic | DP, DP3 | single-arm *versions* vs dual-arm versions of tasks. Coordination difficulty | real: separate single-arm task table (different tasks) | no (different task versions or embodiments, not the same scene) | no ("Shoes Place must be completed using both arms": asserted) | no | task | generative digital twins + LLM expert code + benchmark | observes that "arm selection based on proximity" tasks are easy and coupled tasks hard | no matched counterfactual, no feasibility certification, no VLA |
| **RoboTwin 2.0** [F: code + paper] 2506.18088 | 2025 arXiv / ICML 2026 poster | 50 "dual-arm" tasks, 5 embodiments, domain randomization | leaderboard: ACT, DP, DP3, RDT, π0, π0.5, X-VLA, … | robust bimanual data generation and evaluation | no | no | **no**. By our static audit of the expert programs, 26/50 experts use one arm, 9 use 1–2 depending on the scene, 15 always use 2 (`data/robotwin_expert_audit_ea8b211.md`) | eval skips seeds where the expert fails. One policy run per seed | task | scalable expert-program data generator + 50-task benchmark | **our recommended platform**. Its per-arm expert API makes validated single-arm experts writable | the benchmark never asks whether its tasks need two arms. We would measure it per scene |
| **BiCoord** [F] 2604.05831 | 2026 arXiv | 18 long-horizon tasks built on RoboTwin 2.0 | DP, RDT, OpenVLA-OFT, π0 | spatial-temporal coordination. Metrics MRD, ARD, **SMT/SMP (simultaneous movement)**, STI, TL, SSR | no | no | no | 100 rollouts per task, single seed | task | coordination-heavy benchmark + coordination metrics | coordination metrics. Reports **RoboTwin 2.0 dataset SMP = 26.1% ("lack of temporal synergy")** | SMP etc. are computed **only on datasets, never on policy rollouts** (checked). No single-arm reference, no utility, no noise control |
| **DuoBench** [F] 2606.11901 | 2026 arXiv | 11 tasks, 4 categories (asymmetric support, bimanual manipulation, sequential handoff, parallel execution). FR3 Duo, MuJoCo, 4 tasks real | ACT, π0.5, X-VLA | coordination categories, stage-wise failure analysis | no | no | taxonomy *defines* one category as "cannot be solved unimanually". Not validated | 100 sim / 15 real rollouts. No noise replicates | task + stage | reproducible sim+real bimanual benchmark | reports that "current policies struggle with parallel execution of both arms" (Bin-Sort) | no single-arm condition, no completion time or idle-arm metric. Sim SR 7–13% (floor) |
| **TAO-DA** [F] 2609.33197 | 2026 arXiv (Sep 27) | AGIBOT G1, real, 9,080 teleop trajectories | Eagle-2.5-VL + DiT towers. Baselines GR00T N1.5, π0.5 | **arm routing** (left / right / dual), **cross-arm interference** | separate single-arm *task suite* (4–5 tasks) | no (different tasks) | no | 20 trials per condition, no statistics | task (routing per instruction/scene) | dual-arm expert towers + intent routing + progress predictor | interference: perturbing the inactive arm's proprioception. Qualitative "idle arm attempts an empty grasp" (Fig. 4) | no matched dual-vs-single, no feasibility, no idle-motion metric, no efficiency. Routing makes "which arm acts" **occupied** |
| **ACG-Bench / AE-VLA** [F] 2610.06184 | 2026 arXiv (Oct 5) | ManiSkill + RoboTwin-2.0 expert pipeline. 8 families, 23 task-condition pairs. Real SO101 | π0.5 (+ SkillLoRA, arm-wise attention). Baselines single π0.5, MA-VLA, Dual π0.5 | arm-wise **compositional generalization** of per-arm skills | Dual π0.5 = two independent single-arm policies (still two arms) | no | no | seeds 1000–1099. No replicate variance | task-condition | per-arm skill recomposition | per-arm skill segmentation incl. a *Wait* skill | composition, not utility. No idle-time metric, no single-arm execution |
| **SkillVLA** [A+] 2603.03836 | 2026 / RSS | dual-arm skill recombination (sim + real) | π0.5, π0-FAST baselines | reuse of single-arm skills in new left/right pairings | single-arm data collected with the other arm still | no | no | no | task | skill-decoupled dual-arm VLA (0% → 51% recomposition) | reports π0.5 **idle-arm jitter/drift** and fallback to one moving arm (qualitative) | no counterfactual utility, no feasibility control |
| **TwinVLA** [F: abstract, html] 2511.05275 | 2025 / ICLR 2026 | RoboTwin 2.0, Aloha sim, real | twin SingleVLA (OXE-pretrained) + joint attention + MoE | composing two single-arm VLAs | the single-arm VLA is the building block, not an execution baseline | no | no | no | task | data-efficient bimanual VLA from single-arm models | uses single-arm competence as a prior | no dual-vs-single execution comparison |
| **AnyBimanual** [A] 2412.06779 | 2024 / ICCV 2025 | RLBench2 (12 tasks), 9 real tasks | PerAct/RVT-style unimanual policies + skill manager | transferring unimanual policies to bimanual | unimanual policy as source | no | no | no | task | plug-and-play unimanual → bimanual transfer | single-arm knowledge reuse | as TwinVLA |
| **MA-VLA** [A+] 2608.25864 | 2026 / ECCV | RoboFactory (2–4 arms) + real | π0-style flow model + Arm Shuffle | per-arm atomic prompts, unseen collaboration patterns | per-arm independent VLAs as a baseline | no | no | no | task | multi-arm collaboration via atomic prompt allocation | explicit per-arm sub-task allocation (**occupied**) | no utility or feasibility |
| **Co-VLA** [A] 2606.20285 | 2026 arXiv | RoboTwin 2.0 concurrent-motion tasks + real | π0-based, Structured Action Expert | shared coordination latent + per-arm residual | no | no | no | not reported | task | coordination-aware action head. **Completion time −25%** vs monolithic VLA | efficiency of bimanual VLAs | efficiency compared *between VLAs*, never against a single-arm reference |
| **RoboPARA** [A+] 2506.06683 | 2025 / ICLR 2026 | dual-arm LLM planning (X-DAPT dataset) | LLM planners (DAG scheduling) | parallel allocation of sub-tasks to two arms | sequential fallback of baselines | planner-level parallel vs sequential | n/a (planner) | n/a | task | DAG-based parallel dual-arm planning. **30–50% execution-time reduction** | quantifies *planner-level* bimanual time potential | no learned policy, no VLA realization |
| **AnyPos** [A] 2507.12768 | 2025 arXiv | bimanual IDM / data | inverse-dynamics model | cross-arm attention entanglement (left-arm estimate attends to the right arm) | no | no | no | no | model-level | task-agnostic action data + arm-decoupled IDM | evidence that models entangle the arms | diagnostic of attention, not of execution utility |
| **Policy-Induced Hand Priors** [A] 2608.11769 | 2026 arXiv | humanoid dual-arm | VLA / IL policies | **hand-selection bias** under initial-pose perturbation | no | no | no | ? | scene | diagnosis + mitigation of initial-pose dependence | arm choice as a learned shortcut (**occupied**: "which arm") | not utility, not necessity |
| **Task-Aware Bimanual Affordance** [A] 2604.08726 | 2026 arXiv | zero-shot dual-arm | VLM + geometry | joint affordance localization + **arm allocation** | no | no | no | no | scene | VLM-guided arm allocation | arm allocation (**occupied**) | no learned-policy utility |
| **See Selectively, Act Adaptively** [A] 2606.13279 | 2026 arXiv | bimanual sim/real | VLA with dual-level structural decomposition | adapting visual conditioning and action structure to the bimanual interaction mode | no | no | no | no | task / phase | mode-adaptive bimanual VLA | phase-dependent coordination (our Case 4) | method paper. No counterfactual utility |
| **CoFreeVLA** [A] 2601.21712 | 2026 arXiv | dual-arm, real/sim | VLA + self-collision risk estimator | inter-arm **collision** | no | no | no | no | state (short horizon) | collision-aware gating | interference as collision | collision safety, not utility |
| **DynaMAC / DynaBench** [A] 2607.22119 | 2026 arXiv | RLBench-based dynamic bimanual tasks | MiDiGaP (few-shot), policy-agnostic | other arm as a dynamic task parameter | no | no | no | no | state | dynamic multi-agent cooperation | dynamic arm interaction (**occupied**) | not VLA, not utility |
| **ABot-M0** [A] 2602.11236 | 2026 arXiv | multi-embodiment incl. RoboTwin 2.0 | VLA (action manifold) | unified single-/dual-arm action space (inactive arm zero-padded). Model "infers whether one arm suffices" | no | no | no | no | task | VLA foundation model | claims the policy learns when a single arm suffices | claim not measured |
| **RoboEval** [A] 2507.00435 | 2025 arXiv | 8 bimanual tasks, 3–5 variations | IL policies | behavioural metrics incl. coordination, asymmetric arm usage, collisions | no | no | no | ? | task | structured bimanual evaluation | coordination / asymmetry metrics | no single-arm counterfactual |
| **PerAct2** [A] 2407.00278 | 2024 arXiv | RLBench2, 13 tasks | PerAct2 | taxonomy incl. **physical coupling** | no | no | no | no | task | bimanual RLBench + PerAct2 | taxonomy for categories | no utility measurement |
| **ST-BiBench** (ex BiManiBench) [A] 2602.08392 | 2026 arXiv | 3-tier bimanual benchmark | MLLMs (not VLAs) | spatial reasoning about reach and arm role, parallel/sequential planning, 16-DoF control | includes an "independent single-arm" tier | no | no | no | task | MLLM bimanual benchmark | arm reachability reasoning | MLLM agents, not VLAs |
| **Bimanual multi-agent ICL** [A] 2604.20348 | 2026 / CoRL | bimanual via multi-agent in-context learning | LLM agents | arm-as-agent decomposition | no | no | no | no | task | in-context bimanual control | — | not VLA utility |

Context-only mentions (not bimanual-utility papers):
- X-VLA (2510.10274) and Motus (2512.13030): backbones with public RoboTwin 2.0 checkpoints.
- The Krebs & Asfour (2022) bimanual taxonomy (not re-verified).
- The VLA survey 2607.06706.

## What is occupied (do not claim)

1. **Which arm should act / arm role / routing:**
   - TAO-DA (explicit left/right/dual routing);
   - MA-VLA (per-arm atomic prompts);
   - Task-Aware Bimanual Affordance (arm allocation);
   - Policy-Induced Hand Priors (hand-selection bias);
   - VoxAct-B (acting/stabilizing roles).
2. **Single-arm skills composed into bimanual policies:** TwinVLA, AnyBimanual, SkillVLA, ACG-Bench/AE-VLA.
3. **Coordination metrics and coordination-heavy benchmarks:** BiCoord (SMP/SMT/STI), RoboEval, DuoBench (categories, stages), PerAct2 (coupling taxonomy).
4. **Cross-arm interference as a model property:**
   - TAO-DA (inactive-arm state perturbation);
   - AnyPos (attention entanglement);
   - SkillVLA (idle-arm drift, qualitative);
   - CoFreeVLA (collisions).
5. **Planner-level parallel time savings:** RoboPARA (30–50%).
6. **Bimanual VLA efficiency vs other VLAs:** Co-VLA (−25% completion time).

## What no paper does (checked against all rows above)

- Runs the **same initial scene** under dual-arm and single-arm execution.
- Certifies **single-arm physical feasibility** independently of the learned policy, with a planner or expert.
- Measures **both success and efficiency** utility of the second arm.
- Separates physical necessity from policy inability.
- Controls for **same-policy re-sampling variance**.
- Quantifies how much of the available utility a VLA realizes, on ≥ 2 backbones.

The most criteria any single paper meets is 3.5 of 8 (BiCoord, Co-VLA), and those are only the generic criteria: modern VLAs, multiple task families, multiple backbones, partial efficiency. **No paper meets any of the four substantive criteria** (see `NOVELTY_BOUNDARY.md` §2).
