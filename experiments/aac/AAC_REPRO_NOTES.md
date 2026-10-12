# AAC reproduction notes: what is specified where

Each item is labelled **PAPER-SPECIFIED**, **CODE-SPECIFIED** (only the official release determines it), **OUR NECESSARY ADAPTATION** (COAST differs, and this explains why), or **UNRESOLVED**. Upstream pins: `Adaptive-Action-Chunking/robocasa@fed3e6b`, `Adaptive-Action-Chunking/gr00t-multi-sample@11e926b`. See `AAC_UPSTREAM_MAPPING.md` for file and function references.

## Algorithm

| Item | Status | Value used | Source / note |
|---|---|---|---|
| Entropy of continuous parts | PAPER | ½ log[(2πe)^d det Σ], d=3 for translation and for rotation | Eq. 3 |
| Entropy of gripper | PAPER | Bernoulli, p = c/N | Eq. 2 |
| Prefix-mean entropy Ē_h | PAPER | mean of per-step totals over the first h steps | Eq. 4 |
| h* = max(argmax_h(Ē_{h+1} − Ē_h), ξ) | PAPER | as written; index convention below | Eq. 5 |
| Index convention | CODE | `best_k = argmax(np.diff(chunk_mean)) + 1` equals the h in Ē_{h+1} − Ē_h (1-based). This matches the paper | `select_chunk_size` |
| Clamp h ≥ 2 | CODE | `max(best_k, 2)`, and ξ is scanned from l=2, so h* ∈ [2, 16]. h* = 16 *is* possible, but only through ξ, because the elbow alone is at most 15 | paper allows h=1 |
| Space of continuous entropy | CODE | **normalized** model output dims 0:3 / 3:6 (`normalized_pose=True`) | unstated in paper |
| Gripper values used | CODE | unnormalized binary `action.gripper_close` (GR00T `binary` unnormalize `x>0.5`) | |
| Covariance estimator | CODE | `np.cov` (ddof=1) + 1e-6·I; Cholesky, `slogdet` fallback | unstated |
| Bernoulli clip | CODE | p clipped to [1e-9, 1−1e-9], so the floor is 2.2e-8 nats | |
| Component weights | CODE | 1, 1, 1 | the paper implies equal weights |
| ξ definition | PAPER + CODE | first l ∈ [2,16] with m(l) > α, else 16 | Supp. A Eq. 6; the fallback is code-only |
| α (move_th) | PAPER + CODE | 3.0 | Supp. A: "set to 3"; every RoboCasa command uses `--move_th 3.0` |
| m_t | PAPER | ‖Σ_{i<l} Δpos‖ of candidate 0, unnormalized | Eq. 8 |
| m_r | PAPER (ambiguous) + CODE | ‖axis-angle(∏ q(Δrot_i))‖. The paper's "‖∏Δq‖" is ambiguous; the code fixes it | Eq. 9; `compose_rotations` |
| m_g weight | **PAPER ≠ CODE** | **0.2** (code) vs 1 (paper) | D1 in mapping. Code followed |
| Sample used for ξ | CODE | candidate 0 (the `chunk_id` argument is ignored) | |
| Executed candidate | CODE | candidate 0 (`--chunk_id_selector 0`) | the paper does not say |
| N | PAPER + CODE | 20 | Tab. 4 text; `n_samples=20` |
| Noise / RNG | CODE | reseed python/numpy/torch/CUDA with **42 before every call**. The N noise tensors are therefore identical at every step of every episode, and fixed-h runs are deterministic given the observation (up to GPU nondeterminism) | `seed_per_step=True` |
| Fixed-h baselines | CODE | candidate 0 of the same N=20 server, first h actions | `fixed_{h}` selector |
| Denoising steps | CODE | 4 | `GrootWrapPolicy(denoising_steps=4)` |
| Candidates share conditioning | CODE | one backbone pass, `expand` to N; only the initial noise differs | verified on GPU (C in `AAC_BASELINE_IDENTITY.md`) |
| Degenerate profiles (N=1, identical candidates) | CODE behaviour, found here | All Ē_h differences are ≈0. `argmax` then picks the index of the largest float round-off, e.g. h_entropy=15 for N=1 in our tests. Reproduced as-is. **Consequence:** the paper's N=1 point (Tab. 4, LIBERO) is effectively "ξ or a round-off-chosen h", not "entropy-free" | `test_aac.py::test_n1_reduces_to_motion_rule` |
| Selectors `gaussian_only`, `variance`, `separate`, `binning` | CODE (dead) | not implemented upstream (`select_chunk_size` raises); not ported | |
| N=1 output shape | CODE bug | the global `np.squeeze` drops the N axis, so the upstream client would crash for N=1. COAST keeps the axis | |

## Environment and experiment

| Item | Status | Upstream | COAST | Note |
|---|---|---|---|---|
| Checkpoint | UNRESOLVED → ADAPTATION | authors' GR00T N1.5 diffusion-head fine-tune (100 MimicGen demos per task, 24 tasks, 90 epochs). **Not released** | `robocasa365_checkpoints/gr00t_n1-5/multitask_learning/checkpoint-120000` | the largest single difference; absolute rates are not comparable |
| RoboCasa version | ADAPTATION | `0eae0634` (pre-v1.0, 24 tasks) | v1.0-12 `9a3a786` | required by the COAST checkpoint |
| Task names | ADAPTATION | `PnPCounterToCab`, … | `PickPlaceCounterToCabinet`, … | map below |
| Split | ADAPTATION | (single, pre-v1.0) | `pretrain` (COAST default) | |
| Trials | PAPER | 100 per task | pilot: 20 per task | |
| Episode budget | CODE | 600 steps (1200 for OpenDoubleDoor) | 1.5 × `get_task_horizon` | same across arms |
| Success rule | CODE | `_check_success()` for 4 consecutive steps | first success step | same across arms |
| Scene seeds | CODE | new env per trial, `seed=trial_id`, `trial_id = 0..99` | `--seed 0 --seed_per_episode` → `env.reset(seed=k)` for episode k | same semantics; scene hashes logged to verify pairing |
| Images | CODE | raw 256² robosuite obs, no flip/resize in the client | gym wrapper (vertical flip); `--resize_size 256` → native 256² reaches the model | COAST's 224 round-trip avoided for AAC runs; same for all arms |
| Executed action dims | CODE | arm pos/rot + gripper; base/torso/mode zero | all 12 dims (`convert_action`) | AAC math does not read dims 7–11 |
| Selection location | ADAPTATION | client | server (`groot_aac.py`) | identical selector input; avoids shipping N×H×D over WebSocket |
| Transport | ADAPTATION | ZMQ + pickle | WebSocket + msgpack | |
| TF seeding | ADAPTATION | `tf.random.set_seed(42)` | omitted | TF not on the inference path |

### Task-name map (paper's 24 tasks → RoboCasa v1.0)

| Paper / upstream | v1.0 class | In COAST 7-task subset |
|---|---|---|
| PnPCounterToCab | PickPlaceCounterToCabinet | yes |
| PnPCabToCounter | PickPlaceCabinetToCounter | |
| PnPCounterToSink | PickPlaceCounterToSink | |
| PnPSinkToCounter | PickPlaceSinkToCounter | |
| PnPCounterToMicrowave | PickPlaceCounterToMicrowave | |
| PnPMicrowaveToCounter | PickPlaceMicrowaveToCounter | |
| PnPCounterToStove | PickPlaceCounterToStove | yes |
| PnPStoveToCounter | PickPlaceStoveToCounter | |
| CoffeeSetupMug / CoffeeServeMug | same names | CoffeeSetupMug |
| CoffeePressButton | StartCoffeeMachine (closest) | |
| OpenSingleDoor / CloseSingleDoor | OpenCabinet, OpenMicrowave / CloseCabinet, CloseMicrowave (v1 splits "single door") | |
| OpenDoubleDoor / CloseDoubleDoor | no direct class (v1 cabinets may have double doors under OpenCabinet/CloseCabinet) | |
| OpenDrawer / CloseDrawer | same names | OpenDrawer |
| TurnSinkSpout, TurnOnSinkFaucet, TurnOffSinkFaucet, TurnOnStove, TurnOffStove, TurnOnMicrowave, TurnOffMicrowave | same names | |
| (not in AAC) | CloseFridge, OpenStandMixerHead, TurnOnElectricKettle | yes |

The RoboCasa v1.0 versions are different task implementations: new scenes, objects and success checks. Per-task numbers are not comparable to the paper's Table 6/7, even where the names match.

### Operating-point sensitivity of the upstream protocol (found in this work, CRITICAL)

The fixed-h5 baseline (7-task subset × seeds 0–19, 140 episodes per cell; `results/aac/diag_*`, `results/aac/phaseA_coast_ref`):

| Server noise protocol | Client image path | Success |
|---|---|---:|
| plain COAST server, running RNG (1 sample) | 224 → 256 (COAST default) | **83/140 = 59.3%** |
| plain COAST server, running RNG | native 256 | 51/140 = 36.4% |
| AAC server N=20, candidate 0, **running RNG** (`--aac-seed None`) | native 256 | 62/140 = 44.3% |
| AAC server N=20, candidate 0, **reseed 42 every call** (upstream) | 224 → 256 | 15/140 = 10.7% |
| AAC server N=20, candidate 0, **reseed 42 every call** (upstream) | native 256 | 16/140 = 11.4% |

1. **Upstream's per-call reseed (`seed_per_step=True`) collapses this checkpoint**: 44% → 11% at 256 px, and 59% → 11% against COAST's default path. Batching N=20 and executing candidate 0 is harmless; it matches or beats the plain server at the same image path. Under reseeding the same noise tensor drives every call. Its candidate 0 predicts slower motion (median m(5) 0.75 vs 1.21 unseeded), and nothing stochastic helps the policy leave a stall. Failed episodes run to the step limit in both protocols.
2. **Native 256 px input is worse than COAST's 224 → 256 round trip** for this checkpoint (36% vs 59% on the plain server). I chose native 256 px to match upstream's raw-256 client and the checkpoint metadata; the data says COAST's validated path is better.
3. Whether either effect also holds for the paper's own checkpoint is unknown (U1). Their h=16 GR00T baseline (59.7%) suggests their checkpoint tolerated the reseed protocol.

Consequence: the faithful-protocol Phase A comparison (`phaseA_pilot`) is internally valid, since every arm shares the protocol, but it runs at a degraded operating point. Phase A-U (`results/aac/phaseA_unseeded/PREDECLARED.md`) repeats the key comparison at COAST's validated operating point: running RNG, 224 path, all AAC parameters at upstream values.

### RoboCasa v1.0 scene nondeterminism (found in this work)

`env.reset(seed=k)` does **not** fully determine the initial scene in RoboCasa v1.0 for tasks with optional fixtures (`enable_fixtures`), e.g. OpenStandMixerHead and TurnOnElectricKettle. For about a third of the seeds of those tasks, the robot's spawn pose takes one of **two** values, and which one is fixed per process (verified with 4 independent processes; mobile-base offset up to ~1.9 m). The cause was not pinned down:
- `PYTHONHASHSEED=0` does not fix it;
- seeding global numpy/python RNGs does not fix it;
- routing robosuite/robocasa's unseeded `np.random.default_rng()` calls (`fixtures/others.py:76`, `mjcf_utils.get_size`) through a seeded stream does not fix it.

The remaining suspect is `id()`-ordered iteration somewhere in placement. Seeds of the other tasks reproduce exactly, including across processes and after simulation steps (OpenDrawer, PickPlaceCounterToCabinet checked).

Handling: every episode logs `scene_hash` (MuJoCo state) and `model_hash` (body positions + geom sizes; added mid-pilot, so earlier episodes lack it). Paired comparisons use only pairs whose hashes match. Unpaired per-arm rates still sample the same scene distribution. Phase A: 10 of 140 (task, seed) keys differ between some arms, all in OpenStandMixerHead and TurnOnElectricKettle. Upstream AAC's RoboCasa (pre-v1.0, robosuite.make per trial) may or may not share this issue; this is unknown.

### Operational incidents (no effect on validity after handling)

- Host RAM: each RoboCasa client holds 4–5 GB, and 14 concurrent clients exhausted 62 GB. Two clients were OOM-killed and the servers stalled. The pilot was stopped and resumed with 8 clients. Episodes are logged only on completion, and calls from unfinished episodes are dropped (`run_id`, or episode index for pre-`run_id` records). Resuming from the first missing episode seed is exact up to the nondeterminism above.
- One client hit `mujoco.FatalError: Offscreen framebuffer is not complete` (EGL, GPU 0 under memory pressure) at its first reset. It was resumed. Counted under "Sim exceptions" in the reports.

## Unresolved

- U1. The exact checkpoint and data behind Table 1/6/7 (not released).
- U2. Whether the released code matches the code that produced the paper. Every repo is a single squashed commit made two months after arXiv v2.
- U3. Whether the paper's fixed-h baselines used the N=20 server (candidate 0) or a plain single-sample GR00T. The released client implies the former; the paper does not say.
