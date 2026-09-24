# Phase 2A — Mechanism Ablation: COAST vs Pure Hidden-State Shrinkage

**Development split only (`--seed 15`, init states 15–29). States 30–44 remain untouched.**
This is a controlled mechanism ablation. It does not tune anything, does not sweep layers or β, and does not introduce a new COAST variant. Every statement below is conditional on this one task, layer, β, conceptor and noise seed.

Paper: *Contrastive Conceptor Activation Steering (COAST)*, arXiv:2605.17144.
Base commit: `743269d43788b44a5d1dc997917823b2c43358cd` (Phase 1E). Branch `exp/coast-mechanism-ablation`. The runs used the uncommitted working tree described in §2. `sha256(git diff HEAD -- src scripts examples packages)` at run time was `fb58d43e…aa1bfa`.

## 0. Summary

| | successes / 15 | Wilson 95% |
|---|---|---|
| A. Baseline (no steering) | **13** | 0.62–0.96 |
| B. COAST, `global`, M = 0.9·I + 0.1·C (V0) | **11** | 0.48–0.89 |
| C. Pure shrinkage, M = 0.9·I (no conceptor) | **10** | 0.42–0.85 |

- **Pure shrinkage reproduces COAST's episode-level outcomes on 14/15 states.**
  - Every state where COAST fails, shrinkage also fails (15, 18, 24, 28).
  - Both lose the same three baseline successes (states 15, 18, 24).
  - The only disagreement is state 21. There, baseline and shrinkage time out while COAST succeeds (at step 203).
- **At the hidden-state level the two interventions are nearly the same operation.**
  - COAST: norm ratio ‖h′‖/‖h‖ = **0.9006**, cos(Δh, h) = −0.9996.
  - Shrinkage: norm ratio **0.8984**, cos(Δh, h) = −0.9999.
  - The remaining magnitude difference comes from bf16 rounding of 0.9 (§3.3), not from the conceptor.
  - On the 15 first policy calls, where both conditions see an identical observation and identical noise, the layer-5 hidden-state norms stay within 0.07% of each other across all 10 denoising steps.
- **Answer to the Phase 2A question (for this configuration only).**
  - Removing the conceptor's direction and keeping only the contraction left 14 of 15 episode outcomes unchanged, including all three of COAST's losses relative to baseline.
  - The data show **no detectable behavioral contribution of C beyond hidden-state scaling**, apart from one state (21) whose cause cannot be attributed (§5).
  - This supports, but does not prove, the Phase 1E hypothesis that COAST at L5/β=0.1 is dominated by contraction.
- **All three success counts sit inside the baseline's own noise-seed spread** from Phase 1D (7–13 of 15 across master seeds). None of the pairwise differences is statistically distinguishable at N = 15 (exact McNemar p ≥ 0.25).

## 1. Design

- **Model:** pi0.5 (`pi05_libero`, PyTorch).
- **Checkpoint:** `checkpoints/openpi-libero-2000`.
- **Task:** libero_10 task 2, `KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it`.
- **Episodes:** `--seed 15 --num_episodes 15` (episode k → init state 15+k; `env.seed(15)`).
- **Rollout:** max 520 steps, replan every 5.
- **Paired policy noise:** master seed **100** for all three conditions.
- **Fixed steering:** layer 5, α 0.5, β 0.1. No tuning. For shrinkage, α is not used.
- **Conceptor for B:** repository-faithful V0, `conceptors/phase1b_repo_task02.npz` (sha256 `8b1c802a…ec80f3`, key `…__L5__0.5__C_contrastive`).
- **One server process** (V0 NPZ, `--steer --noise-control --steering-diagnostics`) served all three conditions on GPU 1 (RTX 4090).
  - A sends no `__steering__` payload.
  - B sends `strategy=global`.
  - C sends `strategy=shrinkage`.
- **Preregistered before any rollout** (`examples/libero_env/output/phase2a_servers/preregistration.txt`, written 00:10:38):
  - seed 100;
  - order A → B → C;
  - one run per condition;
  - the primary comparisons.
- Each condition ran exactly once.

## 2. Code changes

| File | Change |
|---|---|
| `packages/openpi-client/src/openpi_client/steering.py` | `"shrinkage"` added to `ALLOWED_STRATEGIES`, documented as the research ablation h′ = (1−β)h. |
| `src/openpi/serving/steering.py` | New `ShrinkageSteeringHook(ConceptorSteeringHook)`. It builds `M = self._build_M(zeros(d, d))` = (1−β)I + β·0 through the **parent's own** `_build_M`, and applies it through the **parent's own** `__call__` (same `M.to(h.dtype)` cast, same `h @ M.T`, same diagnostics). M is built on the first call from h's hidden dim, so **no conceptor is read**, and the hook logs `Shrinkage ablation: M = (1 - beta) I = 0.9 * I (d=1024); no conceptor used`. `SteeredPolicyWrapper._get_or_build_hook` gains one `shrinkage` branch, and the cache key ignores α for shrinkage. `intervention_summary` gains one scalar, `mean_norm_ratio` = mean ‖h′‖/‖h‖. |
| `examples/libero_env/main.py` | Comment only: lists `shrinkage` as a strategy value. |
| `tests/client/test_steering.py` | The expected strategy set includes `shrinkage`. |
| `tests/test_steering_diagnostics.py` | Schema includes `mean_norm_ratio`, plus 3 assertions on it. |
| `tests/test_steering_shrinkage.py` (new, 21 tests), `tests/models/test_steering_shrinkage_gpu.py` (new, 3 manual) | §4. |
| `research/reproduction/tools/analyze_phase2a.py` (new) | Analysis (§5–6). |

**Unchanged:**
- `ConceptorSteeringHook`, the `global` / `positive_only` / `random_matched` / `per_step` / `linear` code paths, conceptor construction, activation collection, and the paper/repository formulas.
- pi0-fast and GR00T reject `shrinkage` through their existing "unknown strategy" errors.
- `~/Stanley_ws/lerobot` was not touched.

## 3. Tests

### 3.1 CPU (`tests/test_steering_shrinkage.py`, 21 tests)

| Req. | Test | Result |
|---|---|---|
| 1 COAST unchanged | `global` still yields `ConceptorSteeringHook` with `M == 0.9·I + 0.1·C` from the NPZ key (`torch.equal`); forward `== h @ M.T`; COAST and shrinkage are distinct cached hooks | exact |
| 2 M = 0.9·I | shrinkage `M == 0.9·I` (float32, `torch.equal`); output `== 0.9·h` exactly in fp32; rebuilt for a new hidden dim; α not in the cache key; diagnostics give relative Δ 0.1, cosine −1, norm ratio 0.9; the ablation is logged | exact |
| 2 (bf16) | in bf16 the applied factor is **0.8984375** (= bf16(0.9)), pinned so it is not silently "corrected" | exact |
| 3 β = 0 → identity | `global`, `positive_only`, `random_matched`, `per_step`, `shrinkage`: every M `== I` and `h′ == h` (`linear` has no β; its α = 0 no-op is covered in `tests/test_steering.py`) | exact |
| 4 no conceptor read | counting proxy on the NPZ: two shrinkage calls make **0** array reads; one `global` call makes exactly 1 (`…__L5__0.5__C_contrastive`); shrinkage works for a layer with no conceptor in the NPZ, where `global` raises `KeyError`; the hook constructs with no matrix | pass |
| — | pi0-fast rejects `shrinkage`; payload validation unchanged | pass |

### 3.2 Existing tests (req. 5)

- `tests/test_steering.py`, `tests/client/test_steering.py`, `tests/test_steering_diagnostics.py` and `tests/test_serve_policy.py` all pass.
- The **full CI suite** (`CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu uv run pytest --strict-markers -m "not manual"`) gives **409 passed, 47 skipped, 0 failed**. Phase 1E had 388; the difference is exactly the 21 new tests.
- `ruff check` / `ruff format --check` are clean.

### 3.3 Manual GPU (`tests/models/test_steering_shrinkage_gpu.py`, 3 tests; the Phase 1E diagnostics GPU tests were re-run too, 7/7 pass)

Real checkpoint, real `sample_actions_with_steering`, one fixed observation with a fixed noise key:
- Shrinkage at β = 0 gives actions `np.array_equal` to baseline.
- Shrinkage at β = 0.1, layer 5: cosine −0.99989 to −0.99991, and norm ratio 0.89816–0.89858 around bf16(0.9) = 0.8984375.
  - The first version of this test asserted the ratio to within 1e-6 and **failed**. The output h′ = bf16(0.8984375·h) is itself rounded to bf16 (relative error up to 2⁻⁹ per element), so the per-token ratio scatters around 0.8984375.
  - The tolerance was set to 1e-3 from that bound. This was a test-expectation error, not a code change.
- Descriptive action distances on this observation:

| pair | max \|Δaction\| |
|---|---|
| COAST vs baseline | 1.0605 |
| shrinkage vs baseline | 1.0561 |
| COAST vs shrinkage | **0.0052** |

  On this single input, COAST's action is about 200× closer to shrinkage's than to the baseline's.

## 4. Run

Server launched 2026-09-24 00:10:48 +08:00:
```
CUDA_VISIBLE_DEVICES=1 uv run --no-sync scripts/serve_policy.py --pytorch --steer --noise-control --steering-diagnostics \
  --conceptor-npz conceptors/phase1b_repo_task02.npz --port 8101 \
  policy:checkpoint --policy.config pi05_libero --policy.dir checkpoints/openpi-libero-2000
```
Clients (`examples/libero_env`; driver `output/phase2a_servers/run_clients.sh`):
```
common: CUDA_VISIBLE_DEVICES=1 MUJOCO_GL=egl LIBERO_CONFIG_PATH=/home/stanley/Stanley_ws/COAST-research/libero_config_coast \
  uv run --no-sync python main.py --task_suite_name libero_10 --task_id 2 --num_episodes 15 --seed 15 --port 8101 \
  --policy_noise_seed 100 --output_dir output/phase2a_<cond>_task02_seed15
A baseline : <common>
B COAST    : <common> --steer --steering_layer 5 --steering_alpha 0.5 --steering_beta 0.1 --log_steering_diagnostics --steering_strategy global
C shrinkage: <common> --steer --steering_layer 5 --steering_alpha 0.5 --steering_beta 0.1 --log_steering_diagnostics --steering_strategy shrinkage
```

| run | wall clock | exit |
|---|---|---|
| A | 00:11:25–00:13:51 | 0 |
| B | 00:13:51–00:16:33 | 0 |
| C | 00:16:33–00:19:23 | 0 |

- The server log has 0 Traceback/ERROR lines. It shows `Built steering hook (…, 5, 0.5, 0.1, 'global') [ConceptorSteeringHook]`, then `Built steering hook (…, 5, 0.0, 0.1, 'shrinkage') [ShrinkageSteeringHook]` and `Shrinkage ablation: M = (1 - beta) I = 0.9 * I (d=1024); no conceptor used`.
- GPU 1 peaked at 8,349 MiB and 88% utilization.
- Each client showed only the 5 known benign warnings.

Analysis: `CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu uv run python research/reproduction/tools/analyze_phase2a.py`. All validation assertions passed.

## 5. Results

### 5.1 Validation

- **Noise pairing.**
  - All 2,741 logged fingerprints (A 845, B 922, C 974 requests) were recomputed from their keys and match.
  - Every episode's request schedule is 0, 5, … up to its length.
  - Shared coordinates carry identical noise: A–B 745/745, A–C 801/801, B–C 904/904.
- **Diagnostics.** B and C each have exactly 10 records per call (steps 0..9), layer 5, 10 tokens, dim 1024, β 0.1, all finite, and a schedule identical to the fingerprint log. B has 9,220 hook applications and C has 9,740.
- **Replay against earlier runs with the same configuration and seed.**

| reference | outcomes | rollout lengths |
|---|---|---|
| Phase 1D baseline (GPU 0) vs A | 15/15 | **15/15** |
| Phase 1D noise-floor seed 100 (GPU 0) vs A | 15/15 | **15/15** |
| Phase 1D repo (GPU 0) vs B | 15/15 | 13/15 (state 25: 221→222, state 29: 224→225) |
| Phase 1E repo+diag (GPU 1) vs B | 15/15 | 14/15 (state 29: 224→225) |

- The **unsteered** path replays exactly across processes *and* GPUs.
- The **steered** path (`sample_actions_with_steering`, eager attention with a forward hook) shows occasional one-step length jitter across processes, even on the same GPU. No steered outcome has changed so far.
- The cause is not isolated. It means small per-state trajectory differences in steered runs are not fully reproducible, which matters for §5.3.

### 5.2 Success and transitions (items 1–2)

| init | A base | B COAST | C shrink | steps A / B / C |
|---|---|---|---|---|
| 15 | 1 | 0 | 0 | 216 / 520 / 520 |
| 16 | 1 | 1 | 1 | 215 / 212 / 211 |
| 17 | 1 | 1 | 1 | 262 / 238 / 239 |
| 18 | 1 | 0 | 0 | 252 / 520 / 520 |
| 19 | 1 | 1 | 1 | 211 / 201 / 209 |
| 20 | 1 | 1 | 1 | 352 / 232 / 238 |
| 21 | 0 | **1** | 0 | 520 / 203 / 520 |
| 22 | 1 | 1 | 1 | 228 / 238 / 177 |
| 23 | 1 | 1 | 1 | 259 / 253 / 239 |
| 24 | 1 | 0 | 0 | 235 / 520 / 520 |
| 25 | 1 | 1 | 1 | 217 / 222 / 224 |
| 26 | 1 | 1 | 1 | 240 / 258 / 246 |
| 27 | 1 | 1 | 1 | 249 / 224 / 248 |
| 28 | 0 | 0 | 0 | 520 / 520 / 520 |
| 29 | 1 | 1 | 1 | 222 / 225 / 217 |
| **total** | **13** | **11** | **10** | |

Rollout steps exclude the 10 settle steps; 520 = timeout.

| pair | fail→success | success→fail | agreement | identical length | exact McNemar p (reference only) |
|---|---|---|---|---|---|
| baseline → COAST | 1 (21) | 3 (15, 18, 24) | 11/15 | 1/15 | 0.625 |
| baseline → shrinkage | 0 | 3 (15, 18, 24) | 12/15 | 2/15 | 0.25 |
| COAST ↔ shrinkage | shrink fails where COAST succeeds: 1 (21) | 0 | **14/15** | 4/15 | 1.0 |

All three succeed on 10 states and all three fail on 1 (state 28).

### 5.3 Intervention magnitude (item 3)

Per hook application at layer 5 (B n = 9,220; C n = 9,740):

| | mean ‖Δh‖ (sd; min–max) | mean ‖Δh‖/‖h‖ | mean cos(Δh, h) | mean ‖h′‖/‖h‖ |
|---|---|---|---|---|
| B COAST | 16.963 (2.668; 12.74–24.10) | 0.09942 ± 0.00027 | −0.99962 | **0.90062** ± 0.00029 |
| C shrinkage | 17.342 (2.690; 13.07–24.63) | 0.10163 ± 0.00014 | −0.99990 | **0.89838** ± 0.00014 |

By denoising step (mean ‖Δh‖; mean ‖h‖ in parentheses):

| step | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|
| B COAST | 22.30 (223.8) | 19.93 | 18.67 | 17.80 | 17.17 | 16.12 | 15.49 | 14.77 | 13.99 | 13.40 (135.4) |
| C shrink | 22.75 (223.8) | 20.33 | 19.06 | 18.17 | 17.54 | 16.48 | 15.84 | 15.13 | 14.35 | 13.78 (135.6) |
| B ‖h′‖/‖h‖ | 0.90039 | 0.90039 | 0.90042 | 0.90046 | 0.90051 | 0.90058 | 0.90066 | 0.90076 | 0.90092 | 0.90115 |
| C ‖h′‖/‖h‖ | 0.89837 | 0.89837 | 0.89838 | 0.89837 | 0.89837 | 0.89837 | 0.89836 | 0.89839 | 0.89841 | 0.89841 |

**The two conditions are not perfectly magnitude-matched.**
- β = 0.1 makes the nominal shrinkage M = 0.9·I, but the hook casts M to the activations' bf16. bf16(0.9) = 0.8984375, so C contracts by 10.16% instead of 10%.
- COAST's own M also goes through bf16. Its diagonal lands on {0.898, 0.902, 0.906, …}, and together with the 0.1·C·h term this gives a realized ratio of 0.9006.
- Net effect: over 9,040 shared coordinates, shrinkage's ‖Δh‖ is about 2.2% larger (B/C ratio 0.978), a norm-ratio gap of 0.0022.
- This was **not** corrected, because matching it would mean tuning β.

The shrinkage condition is therefore "pure contraction, 0.22 percentage points stronger than COAST's realized contraction".

### 5.4 Hidden-state norm change (item 4)

- **Across the run:** mean ‖h‖ at layer 5 is 170.56 (B) vs 170.64 (C), and per-episode means differ by at most 0.3% (CSV). The hook reduces ‖h‖ by 9.94% (B) and 10.16% (C).
- **Paired first calls (rollout step 0 of every episode, 15 calls × 10 denoising steps):**
  - The observation and noise are identical in B and C, so h at denoising step 0 is identical (15/15 exactly equal ‖h‖).
  - As the two interventions act across the 10 denoising steps, ‖h‖ at layer 5 drifts apart by only −0.007% (step 1) to −0.067% (step 9), with B slightly smaller.
  - Within a single policy call, then, the conceptor term changes the layer-5 hidden-state norm by less than 0.07% relative to pure shrinkage.

Per-episode values: `research/reproduction/experiments/phase2a_episode_results.csv`.

### 5.5 Does COAST differ from pure shrinkage behaviorally? (item 5)

- **In this experiment, almost not.**
  - Shrinkage matches COAST on 14/15 outcomes (4/15 identical rollout lengths, so the trajectories do differ in detail).
  - It reproduces all three COAST losses relative to baseline.
  - At the action level on a fixed input it lies within 0.005 of COAST, while both are about 1.06 from baseline.
- **The single disagreement (state 21, COAST success / shrinkage timeout)** cannot be attributed to the conceptor's direction. It could equally come from any of:
  1. the 0.22 pp magnitude mismatch;
  2. trajectory divergence amplifying a tiny per-step difference over about 40 closed-loop policy calls;
  3. the steered path's run-to-run jitter seen in §5.1.

  One state out of 15 gives an exact McNemar p of 1.0.
- **Relative to the baseline noise floor:** Phase 1D's baseline alone gave 13, 7 and 10 of 15 across master seeds 100/200/300. So 13 vs 11 vs 10 here, all on one seed, does not establish that either intervention changes the success rate.

## 6. Interpretation (bounded)

- For pi0.5 on LIBERO-10 task 2 at layer 5 with β = 0.1, the behavioral footprint of COAST (V0, global) is **reproduced by contracting the layer-5 residual stream by about 10%, with no conceptor**, on 14 of 15 paired episodes.
- This fits the Phase 1E measurement that COAST's change is almost exactly antiparallel to h (cos(Δh, h) = −0.9996). It is also consistent with Phases 1C/1D finding no behavioral difference between conceptors that differ by 95% in Frobenius norm: if both are dominated by the (1−β)·I term, their difference barely reaches the hidden state.
- It does **not** show that conceptor directions are useless in general. The conceptor term is small here because this C is small (λ_max 0.35, median 0.001) and β is small. Larger β, other layers or other tasks could behave differently and were deliberately not tested.

## 7. Limitations

- **Small N:** 15 episodes per condition, one master noise seed (100), one run per condition. The baseline's seed-to-seed spread (7–13/15) is larger than every difference observed here.
- **Single task** (libero_10 task 2), **single layer** (5), **single β** (0.1), single α (0.5), single conceptor (repository-faithful V0), development states only. Paper-faithful V3 was not run in this phase.
- **Magnitude mismatch:** the ablation contracts by 10.16% vs COAST's realized 9.94%, due to bf16 rounding of 0.9 (§5.3). It was not corrected (no tuning).
- **Steered-path reproducibility:** one-step rollout-length jitter across processes (§5.1) limits how far any single-state difference between steered conditions can be interpreted.
- **Mechanism measured only at the hooked layer:** effects on later layers, the velocity field or actions are not decomposed.
- No causal claim is made beyond: *in this configuration, replacing C by 0 did not detectably change episode outcomes except on one state.*

## 8. Housekeeping

- Outputs are under `examples/libero_env/output/phase2a_{baseline,coast,shrinkage}_task02_seed15/` (11–17 MB each: videos, client logs, fingerprint and diagnostics JSONL) and `…/phase2a_servers/` (server log, GPU trace, driver, preregistration). All gitignored; none committed.
- The server and GPU logger were stopped by PID after the runs.
- `~/.libero` was not created. `~/Stanley_ws/lerobot` was not touched. States 30–44 were not used.
