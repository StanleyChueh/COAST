# Phase 1C — Pilot Behavioral Comparison: Baseline vs Repository-Faithful vs Paper-Faithful COAST

**This is a development-split behavioral experiment (seed 15, init states 15–29). States 30–44 remain untouched.**
It is a preregistered single-configuration pilot, not a final held-out result, and not comparable to the paper's 30-episode held-out numbers.

Paper: *Contrastive Conceptor Activation Steering (COAST)*, arXiv:2605.17144.
Code commit: `93fbd75693e5cfbc68635528b3418ff0214a6c20` (branch `repro/arxiv-2605-17144`). No code was modified.

## 1. Setup

| Item | Value |
|---|---|
| Model / config / backend | pi0.5, `pi05_libero`, PyTorch (`--pytorch`) |
| Checkpoint | `checkpoints/openpi-libero-2000` (`brandonyang/openpi-libero-2000` @ `aaeeabc`) |
| Task | libero_10 task 2, `KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it` |
| Split | development: `--seed 15`, 15 episodes, init states 15–29 (episode k → state 15+k). Disjoint from the Phase 1A fitting states 0–14. |
| Steering (preregistered, not tuned) | layer 5, α = 0.5, β = 0.1, strategy `global` (paper Table 4/17 Stove+Moka oracle setting) |
| V0 repository-faithful | `conceptors/phase1b_repo_task02.npz` (sha256 `8b1c802a…ec80f3`) |
| V3 paper-faithful | `conceptors/phase1b_paper_task02.npz` (sha256 `4c7f771b…ee8f29`) |
| Key used | `KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it__L5__0.5__C_contrastive`: in both files (1024×1024 float32, finite). V0 quota 0.0185, sym_err 8.5e-2. V3 quota 0.0103, sym_err 5.1e-10. relFro(V0, V3) = 0.952. Matches Phase 1B. |
| GPU | physical GPU 0 (RTX 4090), `CUDA_VISIBLE_DEVICES=0`, used by both server and client |

### Commands (repo root for servers; `examples/libero_env` for clients)

Server 1 (repository NPZ). It served both **A (baseline)** and **B (repository-faithful)**:
```
CUDA_VISIBLE_DEVICES=0 uv run --no-sync scripts/serve_policy.py --pytorch --steer \
  --conceptor-npz conceptors/phase1b_repo_task02.npz --port 8101 \
  policy:checkpoint --policy.config pi05_libero --policy.dir checkpoints/openpi-libero-2000
```
Server 2 (paper NPZ, freshly launched after server 1 was stopped). It served **C (paper-faithful)**. The command is identical except `--conceptor-npz conceptors/phase1b_paper_task02.npz`.

Clients (all use `CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl LIBERO_CONFIG_PATH=/home/stanley/Stanley_ws/COAST-research/libero_config_coast uv run --no-sync python main.py --task_suite_name libero_10 --task_id 2 --num_episodes 15 --seed 15 --port 8101`, plus):
- A: `--output_dir output/phase1c_baseline_task02_seed15` (no steering payload)
- B: `--output_dir output/phase1c_repo_task02_seed15 --steer --steering_layer 5 --steering_alpha 0.5 --steering_beta 0.1 --steering_strategy global`
- C: `--output_dir output/phase1c_paper_task02_seed15 --steer --steering_layer 5 --steering_alpha 0.5 --steering_beta 0.1 --steering_strategy global`

Each condition was run exactly once. Nothing was rerun or tuned.

## 2. Runtime semantics verified (read-only, `src/openpi/serving/steering.py`)

- `SteeredPolicyWrapper.infer`: if the obs has no `__steering__`, it calls `Policy.infer` (plain `sample_actions`). Otherwise it validates the payload, builds or caches a `ConceptorSteeringHook`, and calls `Policy.infer_with_steering` → `PI0Pytorch.sample_actions_with_steering`, with the hook registered on `gemma_expert.model.layers[5]`.
- `global` looks up `{task}__L{layer}__{str(alpha)}__C_contrastive`, i.e. `…__L5__0.5__C_contrastive`.
- `ConceptorSteeringHook._build_M`: **M = (1−β)·I + β·C** (float32 on the GPU). `__call__`: M is cast to h's dtype (bf16), then **`h_steered = torch.matmul(h, M.T)`** is applied to the layer's output hidden state (all 10 action tokens) at every denoising step. There is no symmetrization of C.
- Both servers logged `Built steering hook ('KITCHEN_SCENE3_…', 5, 0.5, 0.1, 'global') [ConceptorSteeringHook] (cache size=1)`. **V0 and V3 pass through the identical hook, code path, and payload; only the loaded C matrix differs.**
- Baseline caveat: the baseline goes through `sample_actions`, while steered runs go through `sample_actions_with_steering`, an inlined copy of the same Euler loop with hooks. Both run eager (torch_compile off by default), use `num_steps=10`, forward `sample_kwargs`, and draw noise with `sample_noise` once per call. They are not the byte-identical function, but no numerical difference other than the hook was found by reading the code.

## 3. Policy stochasticity (RNG) findings

- PyTorch pi0.5 draws the initial flow noise with `torch.normal(...)` on the default CUDA generator (`PI0Pytorch.sample_noise`). `Policy.infer` / `infer_with_steering` pass no noise. The JAX `self._rng = jax.random.key(0)` applies only to the JAX backend.
- Nothing in the serving path calls `torch.manual_seed` or sets deterministic flags (grep of `src/openpi`, `scripts/serve_policy.py`).
- **Measured:** two fresh processes on GPU 0 reported different `torch.cuda.initial_seed()` values (5180366424706619 vs 2700767781462481), and their first `torch.normal` draws of shape (1, 10, 32) differed. **Policy sampling noise is not reproducible across server launches.**
- Therefore:
  - **Environment initial states are paired** across conditions: the same `--seed 15`, so episode k uses state 15+k, and `env.seed(15)` is the same.
  - **Policy sampling noise is NOT paired.** A and B shared one server process, so B continued A's RNG stream. C used a new process with a new random seed. In any case, the RNG position diverges as soon as episode lengths differ.
  - Additionally, bf16 GPU kernels are not guaranteed bit-deterministic.
  - **This is not a strictly paired causal comparison.** Episode-level pairing controls the initial state only.

## 4. Results

| Condition | Successes / 15 | Rate | Wilson 95% CI | Client wall time |
|---|---|---|---|---|
| A. Baseline (steering off) | **8** | 0.533 | 0.30–0.75 | 3m14s |
| B. Repository-faithful COAST (V0) | **10** | 0.667 | 0.42–0.85 | 2m58s |
| C. Paper-faithful COAST (V3) | **10** | 0.667 | 0.42–0.85 | 2m50s |

### Episode-level table (1 = success; rollout steps exclude the 10 settle steps; failures = 520-step timeout)

| init state | baseline | repo (V0) | paper (V3) | steps base / repo / paper |
|---|---|---|---|---|
| 15 | 1 | 1 | 1 | 256 / 200 / 195 |
| 16 | 0 | 1 | 1 | 520 / 234 / 230 |
| 17 | 1 | 0 | 1 | 218 / 520 / 218 |
| 18 | 0 | 1 | 1 | 520 / 200 / 248 |
| 19 | 0 | 0 | 1 | 520 / 520 / 203 |
| 20 | 0 | 1 | 1 | 520 / 216 / 237 |
| 21 | 1 | 0 | 1 | 211 / 520 / 236 |
| 22 | 1 | 1 | 1 | 216 / 221 / 213 |
| 23 | 0 | 1 | 1 | 520 / 227 / 209 |
| 24 | 1 | 1 | 0 | 220 / 230 / 520 |
| 25 | 1 | 0 | 0 | 239 / 520 / 520 |
| 26 | 1 | 1 | 0 | 256 / 443 / 520 |
| 27 | 1 | 0 | 0 | 248 / 520 / 520 |
| 28 | 0 | 1 | 0 | 520 / 203 / 520 |
| 29 | 0 | 1 | 1 | 520 / 216 / 224 |

Machine-readable: `research/reproduction/experiments/phase1c_episode_results.csv`.

### Transitions

| Transition | Count |
|---|---|
| baseline fail → repo success | 6 |
| baseline fail → paper success | 6 |
| baseline success → repo failure | 4 |
| baseline success → paper failure | 4 |
| repo success → paper failure | 3 |
| repo failure → paper success | 3 |
| agreement: baseline/repo, baseline/paper, repo/paper | 5/15, 5/15, 9/15 |
| states succeeding in all three / failing in all three | 2 / 0 |

Paired differences (descriptive): repo − baseline = +2 (6 gains, 4 losses); paper − baseline = +2 (6 gains, 4 losses); paper − repo = 0 (3 vs 3).
For reference only, exact McNemar two-sided p-values on the discordant pairs are 0.75 (base vs repo), 0.75 (base vs paper), and 1.00 (repo vs paper). They are underpowered and use unpaired policy noise, so they should not be read as evidence either way.

### Secondary behavioral diagnostics
- Time-to-success (rollout index of the success step, successes only): baseline median 228.5 (mean 232.0, range 210–255); repo median 217.5 (mean 238.0, range 199–442); paper median 220.0 (mean 220.3, range 194–247).
- Failure mode: every failure in every condition is a 520-step timeout. Failure types were not classified further (videos are available).
- Intervention norms: `ConceptorSteeringHook` records `intervention_norms` internally, but they are neither logged nor returned (`infer_with_steering` returns an empty diagnostics dict). They were **not available** without instrumentation, so none was added.

## 5. Resources, errors, warnings

- GPU 0 peak **8,183 MiB** and peak util 77% across all three runs (server idle 7,579 MiB). No JAX preallocation on this path, unlike Phase 1A collection.
- Server launch → listening ≈ 30–45 s each (PyTorch weights already converted). Total wall time for the three conditions ≈ 11 min (16:43:34–16:54:45).
- Errors: none. 0 Traceback/Error lines in either server log, and all clients exited 0. Warnings were only the known benign ones (robosuite private-macro notice, LIBERO `datasets` path missing).
- Housekeeping incident: an early `main.py --help` run **without** `LIBERO_CONFIG_PATH` made LIBERO's `__init__` create an empty `~/.libero/` (16:43:00) before its interactive prompt failed. It was removed immediately with `rmdir` (it was empty, and it did not exist before). All experiment commands set `LIBERO_CONFIG_PATH`. `~/.libero` does not exist at the end of this phase.

## 6. Interpretation and limitations

- **Primary metric:** both steered conditions reached 10/15 vs 8/15 for the baseline. V0 and V3 had identical success counts.
- **V0 vs V3:** despite matrices that differ by 95% (relative Frobenius) and have top-10 eigenspace alignment of only 0.41 (Phase 1B), their behavior at this setting is not distinguishable. The success counts are equal, and the 6 disagreements split 3/3.
- **The episode-level churn is large relative to the effects.** Baseline and each steered condition disagree on 10 of 15 states. Every state succeeded in at least one condition, and repo vs paper disagree on 6/15. With unpaired policy noise and N = 15, most of this variation is compatible with sampling noise. **There is no evidence here that either COAST variant improves on the baseline, or that V0 and V3 differ behaviorally.** There is also no evidence against it.
- A noise-floor estimate (for example, two independent baseline runs on the same states) and/or controlled policy noise would be needed to separate steering effects from sampling variance. That was out of scope and was not run.
- Not comparable to the paper: the paper's Table 4 reports 0.93 for this configuration on 30 held-out rollouts, using conceptors fitted on its own 15 rollouts. This pilot uses our Phase 1A conceptors, a development split, and 15 episodes.
- Single task, single configuration, single run per condition. No tuning was performed. The held-out states 30–44 were not used.
