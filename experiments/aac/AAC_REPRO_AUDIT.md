# AAC reproduction — audit of the existing COAST GR00T + RoboCasa path

Audit date: 2026-10-07. Branch `reproduce-aac`, created from `main` @ `2afa10ee256a3b3edfeb56500fea166a0837f119`. Written before any AAC code was added.

## 1. Inference path (pre-AAC)

```
RoboCasa gym env (robocasa/wrappers/gym_wrapper.py, RoboCasaGymEnv; renders 3 cams at 256x256, flips images vertically)
  -> examples/robocasa_env/main.py::eval_task
       resize_with_pad(..., 224) on agentview_left/right + eye_in_hand
       state = ee_pos(3) + ee_quat(4) + base_pos(3) + base_quat(4) + gripper_qpos(2)  (16-D)
       prompt = ep_meta["lang"]
  -> openpi_client WebsocketClientPolicy.infer  (msgpack over WebSocket)
  -> groot_env/serve.py -> websocket_policy_server.WebsocketPolicyServer (single-threaded; one request at a time)
  -> groot_adapter.GR00TAdapterPolicy.infer
       build_robocasa_videos: cv2 INTER_LINEAR upsample 224 -> 256 (checkpoint metadata says 256x256)
       build_robocasa_state_dict: split 16-D state into 5 GR00T state keys
  -> gr00t.model.policy.Gr00tPolicy.get_action  (NVIDIA Isaac-GR00T @ n1.5-release, 4af2b62)
       eval transforms: VideoCrop(0.95, center in eval) -> VideoResize 224 -> (ColorJitter is train-only) -> state min-max/rot6d
       backbone (Eagle-2 VLM) once; flow-matching DiT, 4 Euler steps, ONE noise sample, no reseeding
  -> 16-step chunk, 5 action keys concatenated to (16, 12)
  -> client executes the first `replan_steps` = 5 actions (convert_action -> gym unmap_action), then re-queries
```

| Item | COAST value |
|---|---|
| Model | GR00T N1.5, `robocasa/robocasa365_checkpoints : gr00t_n1-5/multitask_learning/checkpoint-120000` (365-task multitask pretraining; RoboCasa reports 43.0% atomic-seen mean) |
| Action horizon H | 16 (`action_indices = range(16)`; checkpoint `config.json` action_horizon 16, action_dim 32 padded) |
| Fixed execution horizon | **`replan_steps = 5`** (main.py and eval_all.py default). The AAC paper baseline is h=16. COAST's fixed-5 is therefore **not** the paper baseline |
| Action dims (client) | 12 = ee_pos 3 + ee_rot 3 (axis-angle) + gripper_close 1 + base_motion 4 + control_mode 1 (mapping in `AAC_ACTION_MAPPING.md`) |
| Denoising steps | 4 (`serve.py --denoising-steps`, NVIDIA default; same as upstream AAC) |
| Cameras | agentview_left, agentview_right, eye_in_hand. Native 256² → client 224² → server 256² (lossy round-trip, documented in groot_env/README "Known limitations 1") |
| Task language | `env.get_ep_meta()["lang"]` |
| Episode horizon | `max_steps = 1.5 × get_task_horizon(env_name)` |
| Success | `info["success"]` = `env._check_success()` on any single step; the episode stops at once |
| Seeds | one env per task built with `seed=7`; episodes are successive `env.reset()` calls on one RNG stream; GR00T noise comes from an unseeded running torch RNG |
| Task set | curated 7-task `SUBSET` (CloseFridge, CoffeeSetupMug, OpenDrawer, OpenStandMixerHead, PickPlaceCounterToCabinet, PickPlaceCounterToStove, TurnOnElectricKettle), split `pretrain` |
| Published COAST GR00T numbers | 39% mean on the 7 tasks at 15 ep/task (groot_env/README) |
| Steering / collection | separate server modes (`--steer`, `--collect_activations`), mutually exclusive |

## 2. Environment state of this checkout (found during the audit)

At the start, none of these were present in this worktree: the GR00T checkpoint, `groot_env/.venv`, `examples/robocasa_env/.venv`, and the RoboCasa kitchen assets. They were created and downloaded during this work. The checkpoint was fetched without `optimizer.pt`, which is not needed for inference. The RoboCasa venv was built on Python 3.11, because uv's default choice of 3.14 has no torch wheel.

## 3. Changes AAC requires

1. **N-candidate sampling for one observation.** NVIDIA's `Gr00tPolicy` draws one noise sample. AAC needs N=20 chunks that share the backbone pass and differ only in their noise (port of `gr00t-multi-sample`).
2. **Per-call reseed (42)** in the AAC server, to match upstream's `seed_per_step=True`.
3. **The AAC selector** (entropy + elbow + minimum-motion bound), ported verbatim.
4. **A variable execution horizon on the client** (`--execution_mode aac`), keeping fixed h for the baselines.
5. **Machine-readable per-call and per-episode logs**, plus a per-episode seed so that runs can be paired by scene.

## 4. Differences from the official AAC infrastructure

| Aspect | Official AAC (RoboCasa) | COAST | Resolution here |
|---|---|---|---|
| Checkpoint | authors' fine-tune of the GR00T N1.5 diffusion head on 100 MimicGen demos per task, 24 old RoboCasa tasks, 90 epochs. **Not released** (no HF/GitHub artifact) | RoboCasa365 multitask checkpoint-120000 | Unavoidable. Main reason for PARTIAL |
| RoboCasa version | `0eae0634…` (pre-v1.0, 24 tasks, names like `PnPCounterToCab`) | v1.0-12 `9a3a786` (RoboCasa365 task names, `pretrain`/`target` splits) | Unavoidable with this checkpoint. Task-name map in `AAC_REPRO_NOTES.md` |
| Env construction | `robosuite.make(PandaOmron, cams 256², control_freq 20, horizon=600)`, new env per trial with `seed=trial_id` | gym wrapper with one env per task, successive resets | Added `--seed_per_episode` (episode k ← seed+k). AAC runs use seed 0 → seeds 0..n−1 like upstream |
| Images | raw 256² obs, no resize | 256² → 224² → 256² | AAC runs pass `--resize_size 256`, which is a no-op resize, so native 256² reaches the model as upstream. Same for every arm |
| Executed dims | arm pos/rot + gripper only (base/torso/mode zeroed) | all 12 dims | kept COAST (A3 in notes) |
| Episode budget | 600 steps (1200 OpenDoubleDoor) | 1.5 × task horizon | kept COAST, the same for every arm |
| Success criterion | `_check_success()` true for 4 consecutive steps (`completion_count=3` latch) | first success step | kept COAST, the same for every arm |
| Trials | 100 per task | 15 (COAST convention) | pilot 20/task; full-run commands provided |
| Transport | ZeroMQ + pickle; selection on the client | WebSocket + msgpack; selection on the server | same selector input; parity tested |
