# AAC Phase B pre-registration (frozen 2026-10-08)

This was frozen **before** any Phase B episode ran, while the Phase A pilot was still running. No Phase A success rate was available, and none was used to set anything below. Every AAC parameter is the upstream default and was fixed before Phase A as well.

Change policy: nothing below changes after Phase B results are seen, except to fix a **proven implementation bug**. Any such fix must be dated and justified in a "Deviations" section appended to this file, and the affected arms must be rerun in full.

## Frozen identity

| Item | Value |
|---|---|
| COAST | branch `reproduce-aac`, base `2afa10ee256a3b3edfeb56500fea166a0837f119`. The uncommitted AAC code is pinned per run by `manifest.json:coast_worktree_sha256` and `code_snapshot.{diff,_untracked.tgz}` |
| Upstream AAC | `Adaptive-Action-Chunking/robocasa@fed3e6b5eb348160dd0570f326f726758fee9056`, `Adaptive-Action-Chunking/gr00t-multi-sample@11e926b0f34cf6acfcb92c0fe6127a1bdc7b856a` |
| GR00T | `third_party/Isaac-GR00T@4af2b622892f7dcb5aae5a3fb70bcb02dc217b96` (n1.5-release), torch 2.5.1+cu124, flash-attn 2.7.1.post4 |
| RoboCasa / robosuite | `third_party/robocasa@9a3a78680443734786c9784ab661413edb87067b` (v1.0-12), `third_party/robosuite@aaa8b9b214ce8e77e82926d677b4d61d55e577ab` |
| Checkpoint | `robocasa/robocasa365_checkpoints : gr00t_n1-5/multitask_learning/checkpoint-120000` (safetensors shard sizes 4999367032 / 2586705312 B) |

## Frozen protocol

| Item | Value |
|---|---|
| Tasks (22, v1.0 analogues of the paper's 24) | **Relocation (10):** PickPlaceCounterToCabinet, PickPlaceCabinetToCounter, PickPlaceCounterToSink, PickPlaceSinkToCounter, PickPlaceCounterToMicrowave, PickPlaceMicrowaveToCounter, PickPlaceCounterToStove, PickPlaceStoveToCounter, CoffeeSetupMug, CoffeeServeMug. **Container (4):** OpenCabinet, CloseCabinet, OpenDrawer, CloseDrawer. **Rotation (5):** TurnSinkSpout, TurnOnSinkFaucet, TurnOffSinkFaucet, TurnOnStove, TurnOffStove. **Button (3):** StartCoffeeMachine, TurnOnMicrowave, TurnOffMicrowave. The paper's Open/CloseDoubleDoor have no v1.0 class |
| Split | `pretrain` |
| Episodes | 50 per task per arm. Episode k uses `env.reset(seed=k)`, k = 0..49 (`--seed 0 --seed_per_episode`), so every arm sees the same 1100 scenes |
| Episode budget / success | COAST defaults: `max_steps = 1.5 × get_task_horizon`; success = first step with `_check_success()` |
| Observation | native 256² agentview_left/right + eye_in_hand (`--resize_size 256`), 16-D state, `ep_meta["lang"]` |
| Action execution | COAST 12-D `convert_action` |
| Server (all arms) | `serve.py --aac`: N = 20, reseed 42 before every call, move_th 3.0, `gaussian_bernoulli`, 4 denoising steps; candidate 0 executed |
| Arms | fixed2, fixed4, fixed8, fixed12, fixed16, aac. The arms differ only in how many actions of candidate 0 are executed per call |
| Optional COAST reference | fixed5, reported separately and not part of any test |

## Endpoints

- **Primary.** AAC vs fixed16, pooled over 1100 paired episodes. Effect = paired success difference. Test: exact two-sided McNemar, α = 0.05. 95% CI: task-stratified bootstrap of the paired difference.
- **Secondary** (descriptive, no multiplicity claims):
  - AAC vs each fixed h;
  - group means (Relocation/Container/Rotation/Button) and the 22-task mean of per-task rates, as in the paper's Table 1;
  - per-task rates with Wilson 95% CIs;
  - mean executed h, policy calls per episode, server latency per query, peak VRAM, and policy compute per successful episode;
  - h* distribution, the fraction of calls where the min-motion bound binds, and entropy by future index.
- **Fixed-h curve.** Success vs h ∈ {2, 4, 8, 12, 16}.

## Decision rules for `AAC_REPRO_STATUS.md`

- **Performance trend reproduced:** the AAC − fixed16 paired 95% CI lies entirely above 0. Also report whether AAC ≥ the best fixed h.
- **Consistent / inconclusive:** the CI contains 0, *and* its upper bound is ≥ +2.3 pp (the paper's 24-task gain), so the paper's effect is not excluded.
- **Not replicated:** the CI upper bound is < +2.3 pp, *and* the study had adequate power (below).
- **Power note.** With 1100 pairs and a discordance rate of about 20%, the 80%-power minimum detectable paired difference is about 3.8 pp (2.8·√(0.20/1100)) at α = 0.05. The paper's +2.3 pp is below that. A null result at this size is therefore expected even if AAC works as reported, and would be classified "inconclusive", not "failed".

## Exact commands

```bash
# Servers (groot_env venv), one per GPU, upstream defaults:
experiments/aac/serve_aac.sh 0 8000 20 > results/aac/logs/server_8000.log 2>&1 &
experiments/aac/serve_aac.sh 1 8001 20 > results/aac/logs/server_8001.log 2>&1 &

# Arms (robocasa_env venv, launched by the script):
RUN=phaseB_repro EPISODES=50 SEED=0 PORTS="8000 8001" WORKERS=8 \
ARMS="fixed2 fixed4 fixed16 aac fixed12 fixed8" \
TASKS="PickPlaceCounterToCabinet PickPlaceCabinetToCounter PickPlaceCounterToSink PickPlaceSinkToCounter PickPlaceCounterToMicrowave PickPlaceMicrowaveToCounter PickPlaceCounterToStove PickPlaceStoveToCounter CoffeeSetupMug CoffeeServeMug OpenCabinet CloseCabinet OpenDrawer CloseDrawer TurnSinkSpout TurnOnSinkFaucet TurnOffSinkFaucet TurnOnStove TurnOffStove StartCoffeeMachine TurnOnMicrowave TurnOffMicrowave" \
experiments/aac/run_arms.sh

# Analysis (root venv):
uv run python experiments/aac/analyze_aac.py results/aac/phaseB_repro
```

Estimated cost on 2× RTX 4090 at Phase A throughput: about 17 h wall for all six arms, of which fixed2 is about 45%. Paper scale (100 episodes per task) roughly doubles that.

## Amendment 1 (2026-10-08, before any Phase B episode)

Phase A diagnostics (`AAC_REPRO_NOTES.md`, "Operating-point sensitivity") showed two things about the fixed-h baseline. The frozen protocol (reseed 42 per call, native 256 px) holds it at ~11% (fixed-5) on the 7-task subset. COAST's validated path (running RNG, 224 → 256 px) reaches 59%. No AAC outcome was used to make this change; it concerns the baseline operating point shared by every arm. Phase B therefore runs **two replicates** with identical tasks, seeds, arms and endpoints:

- **B-upstream** (as frozen above): `serve_aac.sh … 20` (reseed 42), `RESIZE=256`.
- **B-coast** (COAST-native operating point): `serve_aac.sh … 20 --aac-seed None`, `RESIZE=224`.

Each replicate has its own primary test (AAC vs fixed16, McNemar), at α = 0.025 each (Bonferroni). Every AAC parameter stays at the upstream defaults.
