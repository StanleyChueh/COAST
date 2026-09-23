# Phase 1A — pi0.5 LIBERO Activation-Collection Notes

Paper: *Contrastive Conceptor Activation Steering (COAST)*, arXiv:2605.17144.
Code: `repro/arxiv-2605-17144` @ `99b395199bbebc231082c6d3c136dd1655c1da7f` (upstream `2afa10e` + Phase 0 records).
Scope: raw activation collection for **one** LIBERO-10 task, validated. No conceptors, steering, or sweeps.

## 1. Collection path in the current repository (static inspection)

| Question | Answer (source) |
|---|---|
| Server command | `CUDA_VISIBLE_DEVICES=0 uv run scripts/serve_policy.py --pytorch --collect-activations --output-dir activations/libero --port 8100 policy:checkpoint --policy.config pi05_libero --policy.dir checkpoints/openpi-libero-2000` (`docs/activation_collection.md`, `experiments/libero/README.md` step (a)). `--pytorch` is mandatory for pi0.5 (`scripts/serve_policy.py` raises otherwise). |
| Wrapper | `CollectingPolicy` (`src/openpi/serving/activation_collector.py`): rejects requests without `__collect__`/`__finalize_episode__`; calls `Policy.infer_with_intermediates` → `PI0Pytorch.sample_actions_with_intermediates` (`src/openpi/models_pytorch/pi0_pytorch.py:465`). Schema `v1` (`collection_mode="v1"`). |
| Hooks | On `paligemma_with_expert.gemma_expert.model.layers[i]` (action expert only, not the VLM backbone): (1) `register_forward_hook` capturing the decoder-layer output `output[0]` = post-layer residual stream; (2) `register_forward_hook` on `layers[i].mlp.down_proj` capturing its **input** = `gelu_tanh(gate_proj(x)) * up_proj(x)` (GeGLU product). Tensors are `.detach().cpu()` per denoising step; hooks are removed in `finally`. Eager mode, no torch.compile. |
| Layers collected | `collect_layers=(0, 5, 11, 17)` of the 18-layer gemma_300m expert (hard-coded default; `serve_policy.py` exposes no flag). Axis 1 of the residual tensor is the slot index `[0,1,2,3] → layers [0,5,11,17]`. |
| Denoising steps | `num_steps=10` Euler steps, t = 1.0 → 0.1, dt = −0.1. All 10 are recorded. |
| Action tokens | For pi0.5, `embed_suffix` emits **only** `action_horizon` action tokens (the state token exists only for pi0, `if not self.pi05`). `pi05_libero` has `action_horizon=10` → **10 tokens**. |
| Hidden dim | 1024 (gemma_300m width); MLP inner 4096. |
| Precision | Model runs in bf16; captures are `.float()` → stored as **fp32** (bf16 values upcast — verified: low 16 mantissa bits are zero). |
| Per-step files (`v1`) | `denoising.npz` {`all_x_t`, `all_v_t`}; `adarms_cond.npz` {`all_adarms_cond`}; `suffix_residual.npz` {`all_suffix_residual`}; `suffix_mlp_hidden.npz` {`all_suffix_mlp_hidden`}; `metadata.json`. `np.savez` (uncompressed). |
| Directory structure | `<output_dir>/<Path(policy.dir).name>/<task_name>/episode_{episode_id:03d}_env_{env_id:03d}/step_{step:04d}/` → here `activations/libero/openpi-libero-2000/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it/episode_NNN_env_000/step_NNNN/`. |
| Success/failure recording | Client (`openpi_client.collection_session.CollectionSession`, driven by `examples/libero_env/main.py`) records `done` from `env.step` after each post-settle step; `success=True` if `done` ever fires (the rollout loop breaks on the first `done`). `__finalize_episode__` makes the server write `episode_*/metadata.json` (`episode_success`, `steps_to_success`, `total_env_steps`, `total_inference_steps`, …) and `rewards.npz` (`per_step_reward`, `cumulative_reward`, `success_at_step`). The downstream builder labels an episode by `any(success_at_step)`. |
| Every env step? | **No.** One `step_NNNN` dir per inference call. With `replan_steps=5` that is every 5th post-settle env step (`step` = rollout step 0, 5, 10, …). The 10 settle steps (dummy actions) are never recorded. |
| `episode_id` | Loop index 0…N−1, **not** the LIBERO init-state index. The init state is `(seed + episode_id) % 50`; they coincide only for seed 0. |

## 2. Task selection

**Chosen:** libero_10 task 2, `KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it` ("Stove+Moka").

Evidence (no probe rollouts were needed):
- Repo: `examples/libero_env/figures/results_2000_3000_9000.json` (15 episodes/task, checkpoint 2000) gives task 2 = **0.47** (7/15), the closest to 0.5 of all ten tasks. Task 0 (Phase 0) = 0.67; tasks 4/8/9 are near floor (0.00/0.13/0.20).
- Paper Table 1 (pi0.5, LIBERO-10, Base): Stove+Moka = **0.53**. Table 4 also lists it first (KS3).
- Both sources put it near 50%, the best case for getting both classes from 15 episodes.

Observed in this run: **8 success / 7 failure (0.53)**, consistent with both sources.

## 3. Validation results (full scan of every file)

| Check | Result |
|---|---|
| Episodes | 15 (`episode_000`…`episode_014`), init states 0–14 (seed 0) |
| Success / failure | 8 / 7; `metadata.json` agrees with `rewards.npz` for all 15 |
| Success episodes | ids 2, 3, 7, 8, 9, 10, 12, 13; success at rollout step 206–297 |
| Failure episodes | ids 0, 1, 4, 5, 6, 11, 14; **all are timeouts** at the 520-step cap |
| Inference steps | 1,113 total (success eps 385, failure eps 728); `total_inference_steps` equals the step-dir count in every episode; step spacing is exactly 5 |
| NPZ files | 4 × 1,113 per-step + 15 `rewards.npz` = 4,467 |
| `denoising.npz` | `all_x_t`, `all_v_t`: (10, 10, 32) fp32 — (denoise step, action horizon, padded action dim 32) |
| `adarms_cond.npz` | `all_adarms_cond`: (10, 1024) fp32; depends only on the timestep (identical across all episodes/steps) |
| `suffix_residual.npz` | `all_suffix_residual`: **(10, 4, 10, 1024)** fp32 — (denoise step, layer slot [0,5,11,17], action token, hidden) |
| `suffix_mlp_hidden.npz` | `all_suffix_mlp_hidden`: (10, 4, 10, 4096) fp32 |
| NaN/Inf | **0** non-finite values across all arrays |
| Consistency | `x_t[k+1] == x_t[k] − 0.1·v_t[k]` exactly; `x_t[0]` ≈ N(0,1) (mean −0.018, std 1.027) |
| Official test | `ACTIVATIONS_DIR=<task dir> uv run pytest tests/test_activations.py` → **23/23 passed** |
| Disk | 9,198,361,060 bytes (8.6 GiB), ≈ 8.26 MB per inference step (79% of it MLP hidden) |

## 4. Scientific check: what the raw tensor represents vs the paper

**At collection time, the current repository stores:**
- **Token-level** residual states: all 10 action tokens separately, not mean-pooled. The tokens are distinct (max |tok_i − tok_0| ≈ 43 at L11).
- **Uncentered** raw activations. Nothing is subtracted at collection. At L11, ‖dataset mean‖ ≈ 133 vs mean row norm ≈ 203, so a large shared offset is present.
- **All 10 denoising steps**, and all 4 layers (0, 5, 11, 17).

Collecting token-level, uncentered, all-step data is compatible with the paper, because pooling and centering can be done downstream.
The discrepancies are in how the repo's **downstream builder** (`src/openpi/serving/conceptors.py`, read-only inspection; not run) consumes these tensors:

### D1. Token pooling
- PAPER SPECIFICATION: "mean-pooling across action tokens to obtain one vector h ∈ R^d per denoising or autoregressive step" (Sec. 3.2); "one per inference step, after mean-pooling over the token dimension" (App. A.9.1).
- CURRENT REPOSITORY IMPLEMENTATION: `flatten_global` / `flatten_per_step` reshape `(T, 10, L, 10, 1024)` → rows of 1024 with **every token** as a separate sample (no pooling). The docstring says "treat every token at every denoise step" as a sample.
- DIFFERENCE: 10× more rows; the covariance includes between-token (action-chunk position) variance that the paper's pooled vectors average out.
- POSSIBLE CONSEQUENCE: different R, therefore different C_success/C_failure/C_contrastive subspaces and different steering behavior.

### D2. Mean-centering before the correlation matrix
- PAPER SPECIFICATION: Eq. (1) and App. A.9.1 mean-center X (x̃ = x − x̄) before R = X̃ᵀX̃/N.
- CURRENT REPOSITORY IMPLEMENTATION: `correlation_matrix` computes the **uncentered** R = XᵀX/N (module docstring: "correlation, no centering").
- DIFFERENCE: the class mean direction (norm ≈ 133 here) enters R as a dominant rank-1 term.
- POSSIBLE CONCEPTOR CONSEQUENCE: C_s and C_f both strongly retain the shared mean direction; AND-NOT behavior and the spectrum differ from the paper's centered conceptors.

### D3. Boolean AND formula (verified on synthetic matrices only)
- PAPER SPECIFICATION: C_s ∧ ¬C_f = (C_s⁻¹ + (I − C_f)⁻¹ − I)⁻¹, via pseudoinverse (Eq. 3/7, A.9.1). The result is symmetric PSD with eigenvalues in [0, 1].
- CURRENT REPOSITORY IMPLEMENTATION: `boolean_and(A, B) = A @ inv(A + B − A@B) @ B`.
- DIFFERENCE: the canonical form equals `A @ inv(A + B − B@A) @ B`. The code's version matches the paper only when A and B commute. On random 16-D conceptor pairs (‖AB−BA‖ = 0.32), the code result differs from canonical by 13.7% (relative Frobenius) and is **not symmetric** (‖X − Xᵀ‖ = 0.29). For commuting pairs the difference is 0. This contradicts the function's "symmetric up to numerical error" docstring.
- POSSIBLE CONSEQUENCE: the contrastive conceptor used for steering is not the paper's operator whenever C_s and ¬C_f do not commute, which is the generic case for fitted conceptors.

### D4. Suffix composition
- PAPER SPECIFICATION: App. A.7.1 says the expert's suffix tokens "represent the current proprioceptive state and a noised action chunk".
- CURRENT REPOSITORY IMPLEMENTATION: for pi0.5, the suffix is only the 10 noisy-action tokens (a state token is added only for pi0). `pi05_libero` uses `discrete_state_input=False`.
- DIFFERENCE: no state token in S. S = 10, not 11.
- POSSIBLE CONSEQUENCE: none for collection correctness; it matters only when interpreting token indices.

### D5. MLP-hidden terminology
- PAPER SPECIFICATION: "post-GELU expanded activation" (A.7.1).
- CURRENT REPOSITORY IMPLEMENTATION: the input to `down_proj` is the GeGLU product `gelu_tanh(gate_proj(x)) * up_proj(x)`.
- DIFFERENCE: terminology only.
- POSSIBLE CONSEQUENCE: none for residual-stream conceptors; relevant if the MLP hidden is analysed.

### D6. Minimum class size
- PAPER SPECIFICATION: tasks with fewer than 3 successes or 3 failures are excluded from contrastive construction (A.8.3).
- CURRENT REPOSITORY IMPLEMENTATION: `experiments/libero/compute_conceptors.py` defaults to `min_episodes_per_class = 2`, and `compute_all_conceptors()` skips a task when either class has fewer episodes than that threshold. (There is also an inner per-layer guard that skips an empty class; the task-level threshold is the one that matters.)
- DIFFERENCE: paper threshold 3 vs repository default 2. A task with exactly 2 successes or 2 failures is included by the repository but excluded by the paper.
- POSSIBLE CONSEQUENCE: noisier conceptors on near-saturated tasks. Not triggered here (8/7).
- *Correction (Phase 1B review):* an earlier version of this entry wrongly said the repository skips only an empty class.

### D7. Class sample imbalance (not a stated paper protocol; recorded for Phase 1B)
All failures here are 520-step timeouts, so failures contribute 728 inference steps vs 385 for successes (1.9×). They also include long stretches after the policy has stalled. Neither the paper nor the code reweights by episode, so later failure steps weigh more in C_failure.

### Consistent with the paper
Layers {0, 5, 11, 17}; full decoder-layer output captured with forward hooks; 10 Euler denoising steps, all recorded; d = 1024; checkpoint step 2,000; 15 fitting rollouts; global strategy pools all denoising steps (code does too, per token).

## 5. Runtime / resource observations

- Collection: 15 episodes in 3 min 30 s wall (client). Server startup ≈ 45 s (PyTorch weights already converted).
- **GPU memory:** GPU 0 jumped to 22.8 GB of 24.5 GB at the first inference and stayed flat. Phase 0 plain inference peaked at 8.6 GB. Cause: `Policy.infer_with_intermediates` → `collate_transformed_singles` calls `jnp.stack`, which initializes the JAX CUDA backend inside the PyTorch server, and XLA preallocates GPU memory. The run was stable (no OOM, no errors) with ~1.7 GB of headroom. Not changed. `XLA_PYTHON_CLIENT_PREALLOCATE=false` (a standard JAX env var, which the repo already sets in `training/data_loader.py`) would avoid this if headroom becomes a problem.
- Storage: ≈ 8.3 MB per inference call. A full LIBERO-10 × 15-episode collection with similar lengths would be ≈ 80–90 GB.
