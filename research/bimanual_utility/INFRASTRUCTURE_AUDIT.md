# Infrastructure audit

Date: 2026-10-16. Nothing was installed. RoboTwin source was cloned (code only, 17 MB, no assets) into the session scratchpad for static reading; no part of it was executed. Nothing was added to COAST except these documents and one static-analysis script.

## 1. What already exists locally

| asset | where | relevance |
|---|---|---|
| LeRobot (fork, `update_dev` @ `1e5f2049`) | `~/Stanley_ws/lerobot_latest/lerobot` | has `envs/robotwin.py` (Gymnasium wrapper for RoboTwin 2.0, 14-D joint actions, 3 cameras, official instruction generator). Also a RoboTwin docker recipe pinned to RoboTwin `0aeea2d`, and policies `pi05`, `xvla`, `smolvla`, `lingbot_va`, `groot`, … **Best starting point for the policy side** |
| openpi ALOHA sim | `COAST/examples/aloha_sim`, configs `pi0_aloha_sim` | gym-aloha has 2 tasks (transfer cube, insertion) and one backbone. Too few task families |
| IsaacLab OpenArm handover | `~/Stanley_ws/IsaacLab` branch `pringles_handover` | your own VR-teleop handover task with GR00T N1.7 evaluation. One task, teleop data. Relevant later for real-robot transfer, not for the pilot |
| COAST fork harness (LIBERO) | `research/adaptive_inference/fork/` | single-arm MuJoCo. Its lessons apply (exact-state checks, keyed noise, identity branches); the code does not (RoboTwin is SAPIEN) |
| GPUs / RAM / disk | 2× RTX 4090 24 GB (both idle), 62 GB RAM, 3.2 TB free | enough for 2 policy servers per GPU and ~6–8 simulator processes |

## 2. Benchmark candidates

| | **RoboTwin 2.0** | BiCoord | DuoBench | RLBench2 / PerAct2 | gym-aloha | RoboFactory (MA-VLA) |
|---|---|---|---|---|---|---|
| repository | `RoboTwin-Platform/RoboTwin` (HEAD `ea8b211`, 2026-09-24) | `buaa-colalab/BiCoord` holds **only a project page**. DP/RDT checkpoints on HF (`Oshwiciqwq/BiCoord-checkpoints`) | `robotcontrolstack/duobench` (PyPI `duobench`), datasets on HF `RobotControlStack` | `markusgrotz/peract_bimanual` | `huggingface/gym-aloha` | `zhangzaibin/future-robots` |
| license | **MIT** (code); assets HF `TianxingChen/RoboTwin2.0` **MIT** | none stated; code unreleased | Apache-2.0 | Apache-2.0 (+ CoppeliaSim EDU licence) | Apache-2.0 | Apache-2.0 |
| tasks | 50 | 18 (long-horizon, coupled) | 11 in 4 categories | 13 | 2 | multi-arm (2–4) |
| simulator | SAPIEN 3.0.0b1 (PhysX), Vulkan render. Planners mplib 0.2.1 + CuRobo 0.7.8 | RoboTwin 2.0 | MuJoCo | CoppeliaSim/PyRep | MuJoCo | ManiSkill-based |
| embodiment | Aloha-AgileX (default; 6-DoF + gripper per arm). Also ARX-X5, Piper, Franka, UR5 | Aloha-AgileX | FR3 Duo + Robotiq 2F-85 | 2× Panda | ALOHA | various |
| **public VLA checkpoints** | **many, ≥ 6 independent designs** (§3) | none for VLAs found | none found (paper trained π0.5 / X-VLA, ~7% sim SR) | none for modern VLAs | π0 (openpi `pi0_aloha_sim`) | MA-VLA's own |
| scripted expert | **yes: per-task `play_once()` in a high-level per-arm API** (`grasp_actor`, `place_actor`, `move_by_displacement`, `back_to_origin`; `self.move(a, b)` runs two arms simultaneously) | RoboTwin-style | no (VR teleop data only) | waypoint demos | scripted for both tasks | partial |
| arm actions maskable independently | **yes**: 14-D absolute joint targets `[L 6 + grip, R 6 + grip]`, each arm interpolated separately (per-arm TOPP) | yes | yes (per-arm controllers) | yes | yes | yes |
| identical scene replay | **by seed** (`setup_demo(seed)`). Upstream PR #509 (2026-09-24) "make a given seed reproduce the same episode": **must be verified** | by seed | seeded initialization; recorder saves MuJoCo states | seeded variations | seeded | seeded |
| deterministic mid-episode restore | no full-state API used by RoboTwin (SAPIEN has no `mj_getState` equivalent). **Not needed**: the pilot design is episode-level from reset | same | yes (MuJoCo) | partial | yes | ? |
| fair completion time | **caveat** (§4). Use policy-action count for the VLA, physics steps only within the planner pipeline | same | 30 Hz control | keyframe actions (not time) | 50 Hz | ? |
| single-arm feasibility validation | **yes**: write one-arm expert variants in the same API. The planner gives `plan_success`. IK checks classify unreachable poses | via RoboTwin API (code unreleased) | would need new scripted controllers | possible but old stack | trivial tasks | ? |
| policy operating range | π0.5 (public card): 14–100% per task. Many tasks in 40–90% | DP/RDT/π0 struggle | 7–13% (**floor**) | n/a | high | ? |
| verdict | **recommended** | not usable until code is released | floor effects + no checkpoints. Revisit for *force* stabilization tasks later | stale VLA support | too small | multi-arm, no feasibility tooling |

## 3. VLA checkpoints on RoboTwin 2.0 (public, checked on the HF API)

| checkpoint | design | license | size | coverage | notes |
|---|---|---|---|---|---|
| **`SidneyXie/pi05_robotwin`** | π0.5 (PaliGemma + flow action expert), LeRobot | Apache-2.0 | 9.4 GB | one model, 50 tasks (`lerobot/robotwin_unified`, 27.5k episodes), step 50k | **primary**. Public per-task Easy SR for 33 tasks (e.g. lift_pot 57, pick_dual_bottles 69, place_dual_shoes 72, place_bread_skillet 60, place_phone_stand 66, move_pillbottle_pad 74, beat_block_hammer 93, handover_mic 23). Chunk 50, 10 flow steps, 224 px. **`predict_action_chunk(noise=...)` is supported**, so keyed noise and CRN are exact |
| `Myungkyu/pi05_robotwin_vanilla_b128_60k` | π0.5, leaderboard co-train protocol (50 × 50 clean demos) | **no license tag** | 112 GB (all steps) | 50 tasks | protocol-faithful alternative. Licence unclear, so internal use only |
| **`2toINF/X-VLA-RoboTwin2`** | X-VLA 0.9B (Florence-2 + soft-prompted flow transformer) | Apache-2.0 | 3.5 GB | RoboTwin 2.0 (50 demos per task) | **second backbone (gate F)**. Independently designed. The LeRobot `xvla` port **ignores the `noise` argument**, so it needs a per-call `torch.Generator` patch to key the noise |
| `lerobot/lingbot_vla_v2_robotwin` | Qwen3-VL-4B + sparse-MoE action expert | Apache-2.0 | 25.5 GB fp32 | RoboTwin 2.0 | third backbone (fits in bf16, ~13 GB) |
| `motus-robotics/Motus_robotwin2` | latent-action world model (MoT) | Apache-2.0 | 16 GB | 50+ tasks, 87% avg (claimed) | strongest claimed. Different (world-model) family |
| `StarVLA/Qwen3-VL-OFT-RoboTwin2-All` | Qwen3-VL-4B + **deterministic L1 regression head**, no state | no license tag | 9.8 GB | all tasks | **noise-free control**: no sampling noise, so only simulator nondeterminism remains |
| `lerobot/smolvla_robotwin` | SmolVLA 0.45B | Apache-2.0 | 0.9 GB | RoboTwin 2.0 | cheap smoke-testing |

GPU memory (estimates, to be measured in Gate 0):
- π0.5 bf16 inference: ~7–8 GB (same as COAST LIBERO π0.5, measured 7.3 GB);
- X-VLA: ~3–4 GB;
- SAPIEN rendering of 3 × 320×240 cameras plus CuRobo (planner runs only): ~1–2 GB per simulator process.

Two policy servers per 4090 fit, as in the previous pilot.

## 4. Measurement-validity findings from reading the code (must be checked in Gate 0)

1. **Completion time is not a free policy variable in RoboTwin.** `Base_Task.take_action` turns each 14-D joint target into per-arm TOPP (time-optimal) trajectories and steps physics `≈ max(n_left, n_right)` times. **If an arm's TOPP fails or returns 0 steps, that arm is assigned a fixed 50 steps.**
   - A clamped or stationary arm (zero-length path) may therefore hit the 50-step fallback on every action. That would inflate physics time per action in exactly the conditions we compare.
   - Consequence: **VLA time is measured in policy actions (frames)**, which match the demonstrator's recorded frames. Physics time is used only to compare planner against planner (expert runs move through `self.move`, not `take_action`).
   - Gate 0 G0.4 measures the fallback rate per condition.
2. **Success predicates sometimes encode two-armedness or arm identity** (from `data/robotwin_expert_audit_ea8b211.md`):
   - `lift_pot` requires both TCPs at the handles;
   - `grab_roller` and `scan_object` require both grippers closed;
   - `handover_mic`, `put_object_cabinet`, `place_object_scale`, `move_can_pot` and `open_laptop` reference the expert's chosen arm.

   Single-arm conditions therefore need **object-centric predicates**, defined per task in `TASK_AUDIT.md`. They must agree with the official predicate on dual rollouts (Gate 0 G0.5).
3. **Seed selection.** Official evaluation skips seeds where the dual expert fails (`expert_check`). Our scene set is defined the same way, and single-arm feasibility is measured *within* it. This is a dual-feasible population; the prereg says so.
4. **Seed reproducibility** was only fixed upstream on 2026-09-24 (#509). It is unverified on our machine (Vulkan driver, CPU PhysX threads).
5. **The step limit stops execution** (`take_action_cnt == step_lim`). The extended single-arm budget needs an env-config override, and so does recording success at both the official and the 2× budget from one rollout.
6. **The noise API differs per backbone** (π0.5: supported; X-VLA: needs a patch).

## 5. Can single-arm feasibility be validated independently? Yes, at three evidence levels

Evidence levels, in prompt priority order:
1. **Scripted expert, one arm.** A one-arm variant of the task's `play_once()` in RoboTwin's own API. Example for `pick_dual_bottles`: left grasps bottle 1 → place → left grasps bottle 2 → place.
   - Success = planner `plan_success` ∧ object-centric predicate ⇒ **FEASIBLE (certified)**.
   - For tasks where the official expert already uses one arm in a scene, this is the official program, so it is also a check on our scripting.
2. **IK reachability.** For a planner failure, test whether every candidate grasp/place pose of the required object (RoboTwin grasp-pose library) is outside the arm's IK solution set ⇒ **UNREACHABLE (kinematic necessity)**.
3. Otherwise **UNDETERMINED**: excluded from necessity claims and reported.

A failed (masked) VLA is never used as feasibility evidence.

## 6. Recommendation: ONE implementation stack

**RoboTwin 2.0 @ `ea8b211` + LeRobot (local fork) + π0.5 (`SidneyXie/pi05_robotwin`) as the primary backbone, X-VLA-RoboTwin2 for gate F, in a separate workspace outside COAST.**

- **Location:** a new directory `~/Stanley_ws/bimanual-utility/`, with its own uv/conda env (Python 3.10). RoboTwin, assets and checkpoints live there.
  - COAST only holds `research/bimanual_utility/` (docs, prereg, analysis scripts that read result JSONL).
  - This keeps COAST's four venvs untouched. RoboTwin needs `sapien==3.0.0b1`, `mplib==0.2.1`, CuRobo (nvcc build) and pytorch3d, which conflict with COAST pins.
- **Install cost (estimate):**
  - ~20 min plus the CuRobo/pytorch3d builds;
  - ~4 GB of embodiment/object assets (the 11 GB backgrounds are only needed for the randomized setting; the pilot uses `demo_clean`);
  - ~13 GB of checkpoints for π0.5 + X-VLA.
- **Harness to write (pilot implementation, not this session):**
  - an env wrapper with arm clamp/mask, extended budget, dual-budget success logging, per-arm path/activity logging, and keyed noise;
  - one-arm expert variants for the 8 pilot tasks;
  - object-centric predicates;
  - an IK reachability probe;
  - a JSONL schema mirroring the adaptive-inference pilot (`reference` / `condition` / `replicate` / `noise key` / hashes).
- **Rejected:** DuoBench (VLA floor and no checkpoints), BiCoord (code unreleased), RLBench2 (no modern VLA checkpoints), gym-aloha (2 tasks), forcing it into COAST (dependency conflicts; nothing to reuse except design lessons).
