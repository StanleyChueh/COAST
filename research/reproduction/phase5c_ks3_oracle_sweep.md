# Phase 5C — KS3 Oracle Selection Reproduction

Question: **when we run the paper's own selection procedure with the released COAST implementation (15 fit rollouts → grid search → per-strategy best configuration → 30 test rollouts), do we reproduce the paper's KS3 results in Table 4 (global 28/30, per-step 26/30, positive-only 24/30)?**

**Answer: no, not in this run.**

| Strategy | Test result (ours) | Paper | Gap |
|---|---|---|---|
| Global | 21/30 | 28/30 | −7 |
| Positive-only | 17/30 | 24/30 | −7 |
| Per-step (repository version, α = 1.0) | 15/30 | 26/30 (α = 10) | −11 |

The unsteered baseline scored **17/30** on the same test states. None of the three selected configurations differs significantly from that baseline (Fisher p = 0.42, 1.0, 0.80).

The oracle-selected global configuration does better than Phase 5A's fixed paper configuration (21 vs 19/30). However, 21/30 is inside the range our own unsteered baselines have produced on these same states (17 to 21/30).

Each selected configuration's fit score drops on the test set, by 0.17 to 0.23. The best fit scores are what you would expect if every configuration performed the same and the winner was just lucky (§6.2).

This is a measured reproduction gap under one interpretation of a protocol the paper does not fully specify. It does **not** show that COAST fails (§8).

Paper: *Contrastive Conceptor Activation Steering (COAST)*, arXiv:2605.17144 v1.

No change was made to `src/`, `scripts/`, `examples/` or `experiments/`. The grid was exactly the paper's grid; nothing beyond it was searched. Test states were not used for selection.

## 0. Summary

| Condition | Selected config (from fit) | Fit | **Test (states 15–44)** | Wilson 95% | Paper Table 4 | Gap | Fisher vs paper | Reading | Δ vs our test baseline (Fisher p) |
|---|---|---|---|---|---|---|---|---|---|
| Unsteered baseline | — | — | **17/30** (0.567) | 0.39–0.73 | — | — | — | — | — |
| **Global** | L11, α 0.5, β 0.3 | 13/15 | **21/30** (0.700) | 0.52–0.83 | 28/30 (L5, α 0.5, β 0.1) | **−7** | 0.042 | inconsistent | +4 (0.42) |
| **Positive-only** | L0, α 2.0, β 0.3 | 12/15 | **17/30** (0.567) | 0.39–0.73 | 24/30 (L11, α 1, β 0.1) | **−7** | 0.095 | inconclusive | 0 (1.0) |
| **Per-step** (repo, α fixed 1.0) | L5, α 1.0, β 0.3 | 11/15 | **15/30** (0.500) | 0.33–0.67 | 26/30 (L5, **α 10**, β 0.3) | **−11** | 0.005 | inconsistent\* | −2 (0.80) |

\* The per-step comparison is not like for like. The paper's KS3 per-step oracle picked α = 10, but the released builder and hook only support per-step at α = 1.0 (§4.3).

The "Reading" column applies the rule written in the preregistration before any rollout:
- **consistent**: our rate falls inside the Wilson 95% interval of the paper's k/30;
- **inconsistent**: Fisher two-sided p < 0.05 against the paper's k/30;
- **inconclusive**: anything else.

**Interpretation case** (the phase brief's cases 1–3):
- **Case 3** (baseline-like) for positive-only and per-step.
- **Between case 2 and case 3** for global. It rose by +4 over the baseline run alongside it and by +2 over Phase 5A's fixed configuration, but neither difference can be told apart from run-to-run noise.

Overall, oracle search did **not** close the gap to the paper.

## 1. Protocol

| Item | Value |
|---|---|
| Repository | branch `exp/ks3-oracle-sweep`, HEAD `e84c39932a18cc183279466b38766a898d6f831a`, clean at the start of Phase 5C. **No** change to `src/`, `scripts/`, `examples/`, `experiments/` |
| Model / checkpoint | π0.5, `pi05_libero`, PyTorch; `checkpoints/openpi-libero-2000` (HF `brandonyang/openpi-libero-2000`) |
| Task | libero_10 task 2, `KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it` (paper KS3). Max 520 steps; replan every 5 steps |
| Conceptor | **Released implementation (V0)**: token-flattened, uncentered, repository Boolean AND. **Not** the paper-mathematics builder (V3). Uses `conceptors/libero_conceptors.npz`, built in Phase 5A from the fit states 0–14 (8 success / 7 failure). sha256 `1ac8fceb11c71d8bc6c190b0563a82dc8dc572b269833502e26ffd7d8e82709c` (re-verified after the run); 184 keys (4 layers × 5 α × {success, failure, contrastive}, per-step 0–9 per layer, 4 linear directions). Not committed |
| Fit / selection set | `--seed 0 --num_episodes 15` → init states **0–14**. These states were used both to build the conceptor and to select the configuration. Steered re-rollouts on these states give the selection score |
| Test set | `--seed 15 --num_episodes 30` → init states **15–44**. Run only after the selection file was frozen |
| Selection rule (preregistered) | Selected **separately for each strategy**, never across strategies: highest fit successes out of 15. Ties go to the first configuration in repository iteration order (layer ↑, α ↑, β ↑), which is the order `find_best_configs.py` expands the grid and `max()` keeps. The full tie set is reported. No reruns to break ties; no test data used |
| Sweep tool | `experiments/libero/find_best_configs.py`, **unmodified**, one strategy per call, changed only through CLI overrides; outputs written under `examples/libero_env/output/phase5c_*` |
| Policy noise | Released default, **unseeded**. No `--noise-control`, no diagnostics (the test analyzer checks that no noise-fingerprint or diagnostics file exists) |
| Success metric | LIBERO's native `done = _check_success()`. An episode ends at its first success; failure means reaching the 520-step timeout. Per-episode outcomes are recovered from `main.py` logs and video lengths and cross-checked: all 132 sweep configurations and all 120 test episodes agree |
| Preregistration | `examples/libero_env/output/phase5c_servers/preregistration.txt` (gitignored), written 2026-09-24 19:50:49 +08:00, before the first Phase 5C rollout |
| Selection frozen | 2026-09-24 23:55:18 +08:00. `phase5c_selected_configs.json` sha256 `3de5e4e26a63c2c69d7a4c76bdcf489a9913c9b4d98bf68c802e280cf4a98904`. The first test rollout started at 23:56:27; the file's hash was unchanged after the test |
| Hardware | 2 × RTX 4090, one server/sweep process per GPU |

### 1.1 Exact commands

Sweep driver: `examples/libero_env/output/phase5c_servers/sweep.sh` (gitignored). Each shard runs:

```bash
export LIBERO_CONFIG_PATH=/home/stanley/Stanley_ws/COAST-research/libero_config_coast
CUDA_VISIBLE_DEVICES=<gpu> uv run --no-sync python experiments/libero/find_best_configs.py \
  --tasks KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it \
  --num_episodes 15 --seed 0 --betas 0.1 0.3 0.5 \
  --port <8501|8502> --output_dir examples/libero_env/output/phase5c_sweep_<name> \
  --best_configs_path examples/libero_env/output/phase5c_sweep_<name>/best_configs.json \
  <shard args>
```

| Stage | Shard `<name>` | GPU | `<shard args>` |
|---|---|---|---|
| 1 | `global_L0_5` | 0 | `--strategies global --layers 0 5 --alphas 0.1 0.5 1.0 2.0 10.0` |
| 1 | `global_L11_17` | 1 | `--strategies global --layers 11 17 --alphas 0.1 0.5 1.0 2.0 10.0` |
| 2 | `positive_only_L0_5` | 0 | `--strategies positive_only --layers 0 5 --alphas 0.1 0.5 1.0 2.0 10.0` |
| 2 | `positive_only_L11_17` | 1 | `--strategies positive_only --layers 11 17 --alphas 0.1 0.5 1.0 2.0 10.0` |
| 3 | `per_step` | 0 | `--strategies per_step --layers 0 5 11 17 --alphas 1.0` |
| 3 | `linear` (optional) | 1 | `--strategies linear --layers 0 5 11 17 --alphas 0.1 0.5 1.0 2.0 10.0` |

Selection, freeze and test analysis. These run on the CPU from the repository root; the tool is `research/reproduction/tools/analyze_phase5c_oracle.py`:

```bash
CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu uv run --no-sync python research/reproduction/tools/analyze_phase5c_oracle.py select global
CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu uv run --no-sync python research/reproduction/tools/analyze_phase5c_oracle.py select positive_only
CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu uv run --no-sync python research/reproduction/tools/analyze_phase5c_oracle.py select per_step
CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu uv run --no-sync python research/reproduction/tools/analyze_phase5c_oracle.py freeze
CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu uv run --no-sync python research/reproduction/tools/analyze_phase5c_oracle.py test
```

The test driver is `examples/libero_env/output/phase5c_servers/test.sh` (gitignored). It starts two servers:

```bash
CUDA_VISIBLE_DEVICES=<0|1> uv run --no-sync scripts/serve_policy.py --pytorch --steer \
  --conceptor-npz conceptors/libero_conceptors.npz --port <8511|8512> \
  policy:checkpoint --policy.config pi05_libero --policy.dir checkpoints/openpi-libero-2000
```

Then it runs one client per condition, taking the steering flags from the frozen JSON:

```bash
cd examples/libero_env && CUDA_VISIBLE_DEVICES=<gpu> MUJOCO_GL=egl uv run --no-sync python main.py \
  --task_suite_name libero_10 --task_id 2 --num_episodes 30 --seed 15 --port <port> \
  --output_dir output/phase5c_test_<condition> [--steer --steering_layer L --steering_alpha A --steering_beta B --steering_strategy S]
```

| Condition | GPU | Steering flags |
|---|---|---|
| baseline | 0 | none |
| global | 0 | `--steer --steering_layer 11 --steering_alpha 0.5 --steering_beta 0.3 --steering_strategy global` |
| positive_only | 1 | `--steer --steering_layer 0 --steering_alpha 2.0 --steering_beta 0.3 --steering_strategy positive_only` |
| per_step | 1 | `--steer --steering_layer 5 --steering_alpha 1.0 --steering_beta 0.3 --steering_strategy per_step` |

The server logs show exactly the three expected hooks were built:
- `(…, 11, 0.5, 0.3, 'global')`
- `(…, 0, 2.0, 0.3, 'positive_only')`
- `(…, 5, 0.0, 0.3, 'per_step')`

For per-step, the hook cache sets α to 0 by design; the conceptors are built at α = 1.0. The logs contain 0 Traceback or ERROR lines.

### 1.2 Timeline and compute

| Stage | Wall time (2026-09-24/25, +08:00) | Rollouts |
|---|---|---|
| 1 Global | 19:51 → 21:27 (1 h 36 m) | 60 × 15 + 2 baselines × 15 = 930 |
| 2 Positive-only | 21:27 → 23:05 (1 h 38 m) | 930 |
| 3 Per-step | 23:05 → 23:45 (40 m) | 12 × 15 + 1 baseline × 15 = 195 |
| 3 Linear (optional) | 23:05 → interrupted after 23:50 | 12 of 20 configs + baseline = 195 finished |
| 4 Test | 23:56 → 00:09 (13 m) | 4 × 30 = 120 |
| **Total** | about 4 h 20 m wall on 2 GPUs | **2,370** |

## 2. Grid searched

| Strategy | Layers | α | β | Configurations | Status |
|---|---|---|---|---|---|
| Global (`C_contrastive`) | 0, 5, 11, 17 | 0.1, 0.5, 1, 2, 10 | 0.1, 0.3, 0.5 | **60** | complete |
| Positive-only (`C_success`) | 0, 5, 11, 17 | 0.1, 0.5, 1, 2, 10 | 0.1, 0.3, 0.5 | **60** | complete |
| Per-step (10 step conceptors) | 0, 5, 11, 17 | **1.0 only** (repository) | 0.1, 0.3, 0.5 | **12** (the paper's grid has 60) | complete |
| Linear (`h + α·v`) | 0, 5, 11, 17 | 0.1, 0.5, 1, 2, 10 | inert | 20 | **interrupted at 12/20; excluded** (§4.4) |

That is 132 selection-eligible configurations. Every one was run on the same 15 fit states. There were no crashes (no NaN) and none are missing.

## 3. Fit results and selection (Stages 1–3)

Full per-configuration table: `experiments/phase5c_fit_results.csv` (132 rows: strategy, order, layer, α, β, condition, fit_success, fit_n, fit_rate, mean_steps, video_agrees).

### 3.1 Best fit configuration per strategy and the tie sets

| Strategy | Selected | Fit | Tie set (all configs at the max, in iteration order) | Fit score of the paper's own KS3 config |
|---|---|---|---|---|
| Global | **L11, α 0.5, β 0.3** | **13/15** (0.867) | `global_L11_a0.5_b0.3` (unique) | L5 α 0.5 β 0.1: 9/15. 23 configs scored higher |
| Positive-only | **L0, α 2.0, β 0.3** | **12/15** (0.800) | `positive_only_L0_a2.0_b0.3`, `positive_only_L0_a10.0_b0.1`, `positive_only_L17_a1.0_b0.5` | L11 α 1 β 0.1: 8/15. 27 configs scored higher |
| Per-step (α 1.0) | **L5, α 1.0, β 0.3** | **11/15** (0.733) | `per_step_L5_a1.0_b0.3`, `per_step_L11_a1.0_b0.3` | L5 α 10 β 0.3: not expressible. At α 1.0 the same layer and β is our pick |

The runners-up were close:
- **Global:** 3 configs at 12/15 (L0 α0.5 β0.3, L0 α1 β0.3, L17 α0.1 β0.5) and 11 configs at 11/15.
- **Positive-only:** 2 configs at 11/15.

### 3.2 Distribution of fit successes (out of 15)

| k | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Global (60) | 3 | 2 | 2 | 4 | 7 | 8 | 11 | 8 | 11 | 3 | 1 |
| Positive-only (60) | 0 | 1 | 1 | 4 | 10 | 17 | 13 | 9 | 2 | 3 | 0 |
| Per-step (12) | 1 | 0 | 1 | 0 | 0 | 1 | 3 | 4 | 2 | 0 | 0 |

Mean fit success: global 8.55, positive-only 8.40, per-step 8.75.

Mean fit success by β:

| | β 0.1 | β 0.3 | β 0.5 |
|---|---|---|---|
| Global | 9.10 | 9.70 | **6.85** |
| Positive-only | 8.20 | 8.35 | 8.65 |
| Per-step | 9.00 | 10.50 | **6.75** |

Layer and α effects are small: layer means 7.9–9.0 and α means 7.9–8.9 for both conceptor strategies.

### 3.3 Fit-state baselines

Each sweep call also runs one unsteered 15-episode baseline on states 0–14. These are reported but were not used for selection.

| Shard | global L0/5 | global L11/17 | pos-only L0/5 | pos-only L11/17 | per-step | linear (partial) |
|---|---|---|---|---|---|---|
| Baseline successes /15 | 10 | 5 | 4 | 11 | 6 | 7 |

Pooled over the first five: **36/75 = 0.48**. Phase 1A measured 8/15 on these same states.

These baselines vary more than chance alone would explain: binomial dispersion χ² = 10.4 on 4 df, p = 0.035. The same unsteered policy on the same 15 states scored anywhere from 4/15 to 11/15.

## 4. Strategy handling

### 4.1 Global and 4.2 positive-only

These are the full paper grid (60 + 60), each selected on its own.

### 4.3 Per-step: the paper's α = 10 cannot be reproduced without a source change

In the released code, per-step conceptors are built only at `per_step_alpha = 1.0` (`src/openpi/serving/conceptors.py:380`). The steering hook ignores α for per-step and zeroes it in the cache key (`steering.py`). `find_best_configs.py` then collapses the α axis to the sentinel 1.0 (`find_best_configs.py:230, 259`).

The paper's KS3 per-step oracle is L5, **α 10**, β 0.3. Reproducing it would require changing the builder and the NPZ key schema. That is a source change, and this phase forbids source changes, so **it was not done**.

We ran the repository-supported version (α = 1.0, 12 configurations). Its winner has the same layer and β as the paper's (L5, β 0.3). The 26/30 comparison is therefore **across different α values** and should not be read as a like-for-like result.

### 4.4 Linear (optional; not a Table 4 column): excluded

The linear sweep ran on GPU 1 alongside per-step. It stopped after 12 of its 20 configurations; the last log line is at 23:50:18 and there is no exit line. No process was still running at 23:53.

The likely cause is that the launching terminal closed, since a new terminal session started at 23:51. The Stage 4 test run was therefore launched with `setsid`.

Linear is optional in the preregistration and is not part of the Table 4 comparison. We decided **before any test rollout** to exclude it rather than rerun it, so it has no selection and no test run.

Its partial fit scores are left in `examples/libero_env/output/phase5c_sweep_linear/` (gitignored), for reference only:
- baseline 7/15;
- L0: 9, 6, 7, 7, 0 (α 0.1, 0.5, 1, 2, 10);
- L5: 7, 8, 7, 7, 9;
- L11: 8, 8 (α 0.1, 0.5).

## 5. Test results (Stage 4)

Test states are 15–44, with one unpaired, unseeded run per condition. Per-episode data: `experiments/phase5c_test_episode_results.csv` (120 rows). Summary: `experiments/phase5c_test_results.csv`.

| Condition | Test | Fit rate → test rate | Gain states vs baseline (unpaired) | Loss states vs baseline (unpaired) |
|---|---|---|---|---|
| Baseline | 17/30 | — | — | — |
| Global L11 α0.5 β0.3 | **21/30** | 0.867 → 0.700 (−0.17) | 16, 23, 32, 34, 35, 36, 37, 39, 40, 42 | 15, 24, 26, 27, 28, 29 |
| Positive-only L0 α2 β0.3 | **17/30** | 0.800 → 0.567 (−0.23) | 16, 36, 37, 39, 41, 43 | 18, 21, 26, 28, 33, 38 |
| Per-step L5 α1 β0.3 | **15/30** | 0.733 → 0.500 (−0.23) | 16, 32, 34, 39, 40, 41, 43 | 17, 18, 21, 22, 24, 26, 29, 30, 44 |

- State 31 failed in all four conditions; 3 states succeeded in all four.
- The runs are not noise-paired, so individual flips cannot be attributed to steering. Phase 1D showed the noise seed alone moves this task's baseline between 7/15 and 13/15.

## 6. Comparison with the paper and difference analysis

### 6.1 Against Table 4

| | Paper | Ours (oracle-selected) | Ours (Phase 5A, paper's config fixed) |
|---|---|---|---|
| Global | 28/30 (L5 α0.5 β0.1) | 21/30 (L11 α0.5 β0.3) | 19/30 (L5 α0.5 β0.1) |
| Per-step | 26/30 (L5 α10 β0.3) | 15/30 (L5 α1 β0.3) | — |
| Positive-only | 24/30 (L11 α1 β0.1) | 17/30 (L0 α2 β0.3) | — |
| Unsteered on the test states | not reported (paper "Base" 0.53 appears to be the fit rate, Phase 4B F1) | 17/30 | 19/30 (5A), 21/30 (4A) |

The paper's strategy ranking is global > per-step > positive-only. Ours is global > positive-only > per-step, but none of our pairwise differences is significant.

### 6.2 Why oracle selection did not close the gap

1. **The best fit scores look like noise.** Suppose all configurations in a strategy had the same true success rate, equal to the pooled steered fit rate (0.57, 0.56, 0.58). Then the best of *m* independent 15-episode binomial draws would be expected at:

   | Strategy | Expected max (m draws) | Observed max | P(max ≥ observed) under no config effect |
   |---|---|---|---|
   | Global | 12.75 (m = 60) | 13 | 0.61 |
   | Positive-only | 12.63 (m = 60) | 12 | 0.95 |
   | Per-step | 11.75 (m = 12) | 11 | 0.91 |

   So the selected configurations' fit scores tell us nothing beyond chance. Their drop on the test set (−0.17 to −0.23) is what that predicts: the winner's curse of taking the argmax over 60 noisy 15-episode estimates.

2. **The only clear structure in the grid is harm, not benefit.**
   - Global fit scores vary more than chance (χ² = 93.8 on 59 df, p = 0.003). That excess comes mostly from β = 0.5 being worse (mean 6.85 vs 9.1 and 9.7).
   - Per-step shows the same β = 0.5 drop (χ² = 17.6 on 11 df, p = 0.09).
   - Positive-only scores vary *less* than chance (χ² = 42.3 on 59 df, p = 0.95), so there is no detectable configuration effect at all.
   - Averaged over configurations, steered fit rates (0.56–0.58) are slightly above the pooled fit baselines (0.48), but not significantly (Fisher p = 0.15–0.19). The comparison is also confounded by the baseline spread in §3.3.

3. **The test states do not distinguish steering from no steering.** Our three unsteered runs on states 15–44 scored 17, 19 and 21 out of 30 (pooled 57/90 = 0.63). Oracle-selected global's 21/30 matches the best of those (Fisher p = 0.66 against the pooled baselines).

4. **The paper's selected configurations are not special here.** On our fit states, the paper's global configuration scored 9/15, with 23 of 60 configurations higher. The paper's positive-only configuration scored 8/15, with 27 of 60 higher. The two selections do not agree on the winning configuration.

### 6.3 Possible causes of the gap (unresolved)

- **Unknown paper fit and test states and seeds.** Our 0–14 / 15–44 split is a repository-compatible interpretation, not the paper's documented split. Different 15-state windows of this task give baselines from 8/15 to 13/15 in earlier phases.
- **Selection noise in the paper itself.** If the paper also selected on 15 unseeded episodes, its test numbers inherit the same winner's-curse exposure. Its 28/30 would then be one draw from a wide distribution. How many runs lie behind each Table 4 cell is not stated.
- **Conceptor construction.** We tested only the released V0 builder. The paper text describes pooled, centered, canonical AND (V3). Which one produced Table 4 is unknown (Phase 1B/4B).
- **Code version.** `--seed` semantics changed on 2026-04-23 (PR #48). The per-step α axis in the paper is not in the released builder, which suggests the paper's numbers came from a different code state.
- **Per-step α.** Our per-step result uses α 1.0, not the paper's 10.
- **Environment and checkpoint.** LIBERO `d63b117`, robosuite 1.4.0, MuJoCo 3.2.3 here; the paper does not state versions. The checkpoint weights cannot be verified against the authors'.
- **Run-to-run non-determinism.** It is larger than binomial even on the fit states (§3.3).

## 7. What this phase does and does not show

**Shows:**
- Under the released implementation, our 15/30 split, unseeded noise and one test run per condition, the paper's oracle procedure applied separately per strategy gives 21/30 (global), 17/30 (positive-only) and 15/30 (per-step at α 1.0).
- The gaps to Table 4 are −7, −7 and −11.
- None of the selected configurations is distinguishable from the unsteered baseline on the test states.
- The earlier fixed-configuration test (Phase 5A) was not the reason the gap persisted: oracle selection changed the global test result by only +2.

**Does not show:**
- That COAST fails.
- That the paper's numbers are wrong.
- That the gap would persist under the paper's (unknown) states, seeds, builder or code version.

## 8. Limitations

- There is one unseeded, unpaired run per test condition, so the test-set SD is about 0.09 at N = 30. The fit selection uses 15 episodes per configuration, a single draw each.
- Test states 15–44 were used in earlier phases (4A, 5A), so they are not fresh to the researcher. No selection was made on them in this phase.
- The fit states are both the conceptor-construction data and the selection data. That is the paper's design, but it means fit scores are in-sample twice over.
- The linear sweep was interrupted and excluded (§4.4). Per-step is restricted to α 1.0 (§4.3).
- Fit-state baselines are over-dispersed (§3.3), so run-level effects may be present that the binomial model does not capture.

## 9. Files and housekeeping

- **Added:**
  - `research/reproduction/phase5c_ks3_oracle_sweep.md` (this report)
  - `research/reproduction/experiments/phase5c_ks3_oracle_sweep.yaml`
  - `research/reproduction/experiments/phase5c_fit_results.csv`
  - `research/reproduction/experiments/phase5c_selected_configs.json` (frozen)
  - `research/reproduction/experiments/phase5c_test_results.csv`
  - `research/reproduction/experiments/phase5c_test_episode_results.csv`
  - `research/reproduction/tools/analyze_phase5c_oracle.py`
- **Fix in the analysis tool:** `cmd_test` shared one dict between the CSV row and the printed summary, so the first `test` call failed while writing the CSV. It now copies the row. This is our research tool, not repository source, and the change affects no result.
- **Not committed (gitignored):**
  - `conceptors/libero_conceptors.npz`
  - `examples/libero_env/output/phase5c_*` (sweep and test videos and logs, per-shard `best_configs.json`, preregistration, drivers, server logs)
- Both servers were stopped after the test. Nothing beyond the paper grid was tuned. Nothing was committed or pushed.
