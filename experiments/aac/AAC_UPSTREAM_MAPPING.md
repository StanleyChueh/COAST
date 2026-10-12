# AAC: paper ↔ official code ↔ COAST mapping

Paper: Liang et al., *Adaptive Action Chunking at Inference-time for Vision-Language-Action Models*, CVPR 2026, arXiv:2604.04161v2 (10 Apr 2026).

## Pinned upstream sources

| Repo | SHA | Commit date | Role |
|---|---|---|---|
| `Adaptive-Action-Chunking/robocasa` | `fed3e6b5eb348160dd0570f326f726758fee9056` | 2026-06-17 | RoboCasa client with the AAC selector (`robocasa/demos/action_optimization/`) |
| `Adaptive-Action-Chunking/gr00t-multi-sample` | `11e926b0f34cf6acfcb92c0fe6127a1bdc7b856a` | 2026-06-17 | GR00T N1.5 fork that samples N chunks in one pass |
| `Adaptive-Action-Chunking/libero`, `libero-pro` | not used here | — | LIBERO clients (π0.5 / GR00T); out of scope |

Local clones (gitignored) live in `external/aac/{aac-robocasa,gr00t-multi-sample}`. Each repo has one squashed "initial commit" plus README edits, all made two months after arXiv v2. So the history cannot show whether the code drifted from what produced the paper's numbers.

The upstream README tells users to check out RoboCasa `0eae0634a61ad2be33962c9de7000a2dd1ee573f`, which is the pre-v1.0 24-task RoboCasa.

## Component table

| AAC component | Paper | Official source (file : function) | COAST implementation |
|---|---|---|---|
| N candidate chunks, sampled in parallel | §4: "sample N candidate action chunks in parallel"; Tab. 4: N=20 default | `gr00t-multi-sample/gr00t/model/action_head/flow_matching_action_head.py : FlowmatchingActionHead.get_action(…, n_samples)`: one `process_backbone_output`, then `vl_embs.expand(N)`, `embodiment_id.expand(N)`, `state.expand(N)`, `torch.randn(N, H, 32)`, unchanged Euler loop. `gr00t/model/gr00t_n1.py : get_action(inputs, n_samples)`: one backbone pass. `scripts/policy_server_robocasa.py`: `n_samples=20` | `groot_env/groot_adapter.py : _get_action_multi_sample` (line-for-line port; NVIDIA's model files are untouched) via `GR00TAdapterPolicy.infer_multi_sample` |
| Noise / RNG protocol | not stated | `gr00t/model/policy.py : Gr00tPolicy.get_action`: with `seed_per_step=True` (set in `policy_server_robocasa.py`), it reseeds `random`/`numpy`/`torch`/`cuda` (and TF) with **42 before every call**. `scripts/inference_gr00t.py`: `set_seed(42)` plus `cudnn.deterministic=True`, `benchmark=False` at init | `groot_adapter.seed_everything(seed)` is called per request when `serve.py --aac-seed 42` (the default). `serve.py --aac` sets the cuDNN flags. TF seeding is omitted because TF is not on the inference path |
| `normalized_action` output | — | `Gr00tPolicy.get_action`: `unnormalized_action["normalized_action"] = normalized_action.cpu()` = `action_pred.float()` (N, 16, 32) | returned as the second output of `_get_action_multi_sample` |
| Output layout | — | `squeeze_dict_values` uses `np.squeeze` (all singleton axes), so the gripper is (N,16). For N=1 this also drops the N axis, which breaks the client | `aac.build_upstream_pred_action_dict` squeezes only the trailing singleton and keeps N (N=1 works) |
| Continuous entropy E_t, E_r (Eq. 3) | ½ log[(2πe)^d det Σ] | `action_entropy_v2.py : gaussian_entropy_from_samples`, `_safe_logdet_cov` (`np.cov` ddof=1, `+1e-6·I`, Cholesky with `slogdet` fallback; −inf on non-PD). Returns 0.0 if N ≤ 1 | `groot_env/aac.py`, verbatim |
| Which values feed E_t / E_r | translation / rotation offsets | `action_entropy_gausssian_bernoulli(normalized_pose=True)`: **normalized** `normalized_action[:, :, 0:3]` and `[:, :, 3:6]` | verbatim |
| Discrete entropy E_g (Eq. 2) | p = c/N | `bernoulli_entropy_from_samples`: p₁ = clip(mean(g), 1e-9, 1−1e-9), `scipy.stats.entropy([p0,p1], base=e)`. Input is **unnormalized** `action.gripper_close` (binary {0,1} after GR00T's `binary` unnormalize, `x > 0.5`) | verbatim |
| Per-step total | Σ_{j∈{t,r,g}} E_j | `H_total = 1·H_pos + 1·H_rot + 1·H_grip` | verbatim |
| Ē_h (Eq. 4) | (1/h) Σ_{i<h} total_i | `chunk_mean = [mean(total[:k]) for k in 1..T]` | verbatim |
| Elbow argmax_h (Ē_{h+1} − Ē_h) (Eq. 5) | h ∈ 1..H | `diffs = np.diff(chunk_mean)` (length 15); `best_k = argmax(diffs) + 1`, so diffs[j] = Ē_{j+2} − Ē_{j+1} and best_k = j+1 = h. Then **`max(best_k, 2)`** | verbatim (`select_chunk_size`). `aac_select` additionally logs `raw_argmax` and `h_entropy` |
| ξ lower bound (Eq. 6–10, Supp. A) | ξ = argmin_l (m(l) > α), α = 3; m = m_t + m_r + m_g; m_g = 1[switch] | `action_magnitude(end_to_end=True)`: **candidate 0**, unnormalized; l = 2..16; m_t = ‖Σ_{<l} Δpos‖; m_r = ‖axis-angle(∏ q(Δrot))‖ (`help_function.compose_rotations`); m_g = **0.2**·1[any toggle in gripper[:l]]. `find_first_idx`: first l with m > α, **else 16**. `--move_th 3.0` in every RoboCasa command | verbatim, including the rotation helpers |
| h* (Eq. 5) | max(argmax, ξ) | `max(find_first_idx, max(argmax+1, 2))`, so h* ∈ [2, 16] | verbatim; `aac_select` asserts `h* == max(h_min_motion, h_entropy)` |
| Executed chunk | not stated | `inference_client_action_entropy.py`: `--chunk_id_selector 0` in every command, so **candidate 0** (`select_chunk_id`, "mean"/"backward", is not used) | `groot_aac.EXECUTED_CANDIDATE = 0`; the server returns candidate 0's (16, 12) chunk |
| Execute first h* actions, then re-observe | §4 | client slices `[:, :chunk_size]`, then `[chunk_id]`, executes, and calls the server again | `examples/robocasa_env/main.py --execution_mode aac`: queues exactly `aac_horizon` actions; the queue is asserted empty at replan |
| Fixed-h baselines | Tab. 1/7: h ∈ {2,4,8,12,16}; h=16 is the GR00T default | same client with `--chunk_size_selector fixed_{h}` (the default is `fixed_16`) against the **same N=20, seed-42 server**, executing candidate 0 | `--execution_mode fixed --replan_steps h` against the same `serve.py --aac` server |
| Selection location | — | client side | **server side** (`groot_aac.AACGrootPolicyWrapper`), so the N×H×D tensor does not cross the WebSocket. The selector input dict is identical (see `test_groot_aac.py::test_horizon_matches_direct_selector`) |
| Denoising steps | K-step Euler | `GrootWrapPolicy(denoising_steps=4)` | `serve.py --denoising-steps 4` (unchanged COAST default) |
| Methods not used for the paper | — | `chunk_size_selector` ∈ {gaussian_only, variance, separate, binning} appear in the plotting map but are **not implemented** in `select_chunk_size` (it raises `ValueError`) | not ported |

## Paper ↔ code discrepancies (code wins for reproduction)

| # | Item | Paper | Official code | Used here |
|---|---|---|---|---|
| D1 | Gripper term in m(l) | m_g = 1 on switch (Eq. 10) | weight **0.2** | 0.2 |
| D2 | Smallest h | argmax over h=1..H | elbow clamped to **≥ 2**; magnitude scanned from l=2 | clamp |
| D3 | ξ when no l passes α | undefined | **16** (full chunk) | 16 |
| D4 | Sample used for m(l) | undefined | candidate **0** (the `chunk_id` argument is ignored) | 0 |
| D5 | Space of continuous entropy | "translation/rotation offsets" | **normalized** model output (dims 0:3, 3:6). Min-max normalization is a per-dim affine map, so this changes Ē_h only by a constant, *except* through the 1e-6 ridge | normalized |
| D6 | Covariance estimator | Σ | `np.cov` (ddof=1) + 1e-6·I | same |
| D7 | Executed candidate | "the first h* actions" | candidate 0 of the N | 0 |
| D8 | RNG | not stated | reseed 42 before **every** call, so the same N noise tensors are used at every step and the fixed-h baselines are deterministic given the observation | same (`--aac-seed 42`) |
| D9 | α | 3 (Supp. A) | `--move_th 3.0` for RoboCasa (LIBERO π0.5 scripts use 4; not relevant here) | 3.0 |
| D10 | Degenerate entropy profile | — | with N=1 or identical candidates every diff is ≈0, so `argmax` is decided by float round-off (e.g. h=15 for N=1 in our tests) | reproduced as-is; documented |
| D11 | Fixed baselines | "GR00T (h)" | candidate 0 of an N=20 batch, not a plain N=1 GR00T call | same, as primary; see notes for the N=1 reference |
