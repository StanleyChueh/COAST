# Phase 5A, Step 1 — KS3 Global Reproduction

Question: **can the released COAST implementation reproduce the paper's KS3 global headline (layer 5, α 0.5, β 0.1 = 28/30 = 0.93) under our repository-compatible interpretation of the paper's protocol?**

**Answer: not in this run. We measured 19/30 for global COAST, a reproduction gap of 9 episodes, and an equal 19/30 for the unsteered baseline.** This is a measured gap under one interpretation of an under-specified protocol, not a statement that COAST fails (§6).

Paper: *Contrastive Conceptor Activation Steering (COAST)*, arXiv:2605.17144 v1. Nothing was tuned or swept. No source, script, example or experiment file was changed, and `~/Stanley_ws/lerobot` was not touched.

## 0. Summary

| | successes / 30 | rate | Wilson 95% |
|---|---|---|---|
| **Paper, global COAST (Table 4)** | **28** | 0.933 | 0.79–0.98 |
| **Ours, global COAST (L5, α 0.5, β 0.1)** | **19** | 0.633 | 0.46–0.78 |
| Ours, unsteered baseline (same 30 states) | 19 | 0.633 | 0.46–0.78 |

- **Gap to the paper: −9 successes (−0.30).** Fisher exact, two-sided, ours vs 28/30: p = 0.010 (reference only; the paper's states are unknown).
- **Pre-declared reading rule** (written in the reviewed plan and in the preregistration before any rollout): ≥ 24/30 consistent, ≤ 20/30 inconsistent, 21–23 inconclusive. Our 19/30 falls in the *inconsistent-with-the-paper* band. Under the rule that means "reproduction gap measured", not "COAST fails".
- **Steered vs unsteered on these 30 states: no difference** (19 vs 19; 7 episodes flip up, 7 flip down; Fisher p = 1.0). The runs are not noise-paired, so the flips are descriptive only.
- **Steering was on.** The server log shows exactly one hook built: `('KITCHEN_SCENE3…', 5, 0.5, 0.1, 'global') [ConceptorSteeringHook]`. Noise control and diagnostics were off, and the server log has 0 Traceback/ERROR lines.

## 1. Exact protocol

| Item | Value |
|---|---|
| Repository commit | `537f425587395d3b523647811e344e06d960fa41`, branch `exp/ks3-global-reproduction`. Upstream base `2afa10ee256a3b3edfeb56500fea166a0837f119`. **No** change to `src/`, `scripts/`, `examples/`, `experiments/` |
| Model / config | π0.5, `pi05_libero`, PyTorch |
| Checkpoint | `checkpoints/openpi-libero-2000` ← HF `brandonyang/openpi-libero-2000`, revision `aaeeabc72f8a…` |
| Task | libero_10 task 2, `KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it` (paper KS3); max 520 steps, replan every 5 |
| Conceptor label | **RELEASED implementation (V0)**: token-flattened rows, uncentered, repository Boolean AND. **Not** the paper-mathematics builder (V3). The two were not mixed |
| Fit set | 15 episodes, `--seed 0`, init states **0–14**, 8 success / 7 failure. This is the Phase 1A collection, **reused, not recollected** (as decided in the reviewed plan). Used only to build the conceptor; **not** evaluated here |
| Test set | `--seed 15 --num_episodes 30`, init states **15–44** (`(seed + episode) % 50`) |
| Split interpretation | **"repository-compatible interpretation of the paper's 15 fit + 30 test split."** The paper publishes no seed or state IDs; 15 + 30 = 45 of the 50 states, and states 15–44 follow the fit window under the repository's seeding rule (`--seed 15`) |
| Conditions | exactly two, run once each in this order on one server: **A** unsteered baseline, **B** global COAST layer 5, α 0.5, β 0.1 |
| Policy noise | **default released path; unseeded.** No `--noise-control`, no `--policy_noise_seed`, no diagnostics |
| Success metric | LIBERO's native success predicate as used by `main.py`: `done = _check_success()` (`third_party/libero/…/bddl_base_domain.py:807`), episode ends at the first success, failure = 520-step timeout. **This LIBERO version exposes no `info["is_success"]` key**, so `done` is the equivalent native flag (paper A.8.3 describes the final-step flag; equivalent because LIBERO ends the episode at success) |
| Preregistration | `examples/libero_env/output/phase5a_ks3_servers/preregistration.txt` (gitignored), written 2026-09-24 18:42:35 (addendum after the NPZ build at 18:43:32), before the first rollout at 18:44:06 |
| Hardware | 1 × RTX 4090 (GPU 0) |

### Commands (as run)

```bash
# 1. Conceptor (CPU, ~36 s), unmodified released builder, defaults
CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu uv run --no-sync python experiments/libero/compute_conceptors.py \
  --activation_root activations/libero --output_path conceptors/libero_conceptors.npz \
  --task_filter KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it

# 2. Server (GPU 0, port 8301), default path
CUDA_VISIBLE_DEVICES=0 uv run --no-sync scripts/serve_policy.py --pytorch --steer \
  --conceptor-npz conceptors/libero_conceptors.npz --port 8301 \
  policy:checkpoint --policy.config pi05_libero --policy.dir checkpoints/openpi-libero-2000

# 3. Clients (cd examples/libero_env; both use)
CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl LIBERO_CONFIG_PATH=/home/stanley/Stanley_ws/COAST-research/libero_config_coast \
  uv run --no-sync python main.py --task_suite_name libero_10 --task_id 2 --num_episodes 30 --seed 15 --port 8301 \
  --output_dir output/phase5a_ks3_<condition>_test_seed15
#   A baseline: (no steering flags)
#   B COAST:    --steer --steering_layer 5 --steering_alpha 0.5 --steering_beta 0.1 --steering_strategy global
```

Runs: A 18:44:06–18:50:02, B 18:50:02–18:55:50 (2026-09-24, +08:00), both client exit 0. Driver: `examples/libero_env/output/phase5a_ks3_servers/run_clients.sh`. Analysis: `research/reproduction/tools/analyze_phase5a_ks3.py` (new, analysis only).

## 2. Conceptor (fit) verification

| Check | Result |
|---|---|
| File | `conceptors/libero_conceptors.npz`, 755,065,270 bytes, sha256 `1ac8fceb11c71d8bc6c190b0563a82dc8dc572b269833502e26ffd7d8e82709c` (gitignored, not committed) |
| Keys | **184**: exactly the expected set (4 layers × 5 α × {success, failure, contrastive}, per-step 0–9 for each layer, 4 linear directions); none missing, none extra |
| Task key | `KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it__L5__0.5__C_contrastive` present: shape 1024 × 1024, float32, finite |
| Global / layer 5 / α 0.5 | the key the `global` strategy loads for this run (`get_conceptor_matrix`) |
| Builder reproducibility | all **92** keys shared with the Phase 1B repository NPZ (layers 5, 11) are bit-identical (`np.array_equal`), so the builder and the fit data are unchanged |
| Fit data | `activations/libero/…/KITCHEN_SCENE3…`: 15 episodes, 8 success (episode ids 2, 3, 7, 8, 9, 10, 12, 13) / 7 failure (re-verified from `metadata.json`) |
| Property of V0 (not a defect) | quota tr(C)/d = 0.0185; ‖C − Cᵀ‖/‖C‖ = 0.085 (the released AND is not symmetric, D3 in Phase 1B) |

## 3. Results

Rollout steps exclude the 10 settle steps; 520 = timeout. Initial state = `(15 + episode) % 50`.

| ep | init state | A baseline | B global COAST |
|---|---|---|---|
| 0 | 15 | ✓ 260 | ✗ 520 |
| 1 | 16 | ✓ 232 | ✓ 223 |
| 2 | 17 | ✗ 520 | ✓ 221 |
| 3 | 18 | ✓ 224 | ✓ 222 |
| 4 | 19 | ✗ 520 | ✓ 204 |
| 5 | 20 | ✓ 220 | ✗ 520 |
| 6 | 21 | ✗ 520 | ✓ 227 |
| 7 | 22 | ✓ 199 | ✗ 520 |
| 8 | 23 | ✗ 520 | ✗ 520 |
| 9 | 24 | ✓ 232 | ✓ 207 |
| 10 | 25 | ✓ 204 | ✓ 243 |
| 11 | 26 | ✗ 520 | ✓ 274 |
| 12 | 27 | ✓ 326 | ✓ 214 |
| 13 | 28 | ✗ 520 | ✓ 217 |
| 14 | 29 | ✓ 266 | ✓ 197 |
| 15 | 30 | ✓ 213 | ✗ 520 |
| 16 | 31 | ✗ 520 | ✓ 227 |
| 17 | 32 | ✗ 520 | ✓ 177 |
| 18 | 33 | ✗ 520 | ✗ 520 |
| 19 | 34 | ✓ 245 | ✓ 243 |
| 20 | 35 | ✓ 249 | ✗ 520 |
| 21 | 36 | ✓ 209 | ✗ 520 |
| 22 | 37 | ✗ 520 | ✗ 520 |
| 23 | 38 | ✓ 244 | ✓ 205 |
| 24 | 39 | ✓ 228 | ✓ 197 |
| 25 | 40 | ✗ 520 | ✗ 520 |
| 26 | 41 | ✓ 290 | ✓ 258 |
| 27 | 42 | ✓ 266 | ✓ 245 |
| 28 | 43 | ✓ 260 | ✗ 520 |
| 29 | 44 | ✓ 254 | ✓ 241 |
| **total** | | **19 / 30** | **19 / 30** |

Machine-readable, one row per condition × episode: `research/reproduction/experiments/phase5a_ks3_episode_results.csv` (60 rows: condition, episode, init_state, success, rollout_steps).

- **Agreement A vs B:** 16 of 30 episodes have the same outcome (12 succeed in both, 4 fail in both: states 23, 33, 37, 40). Baseline → COAST: 7 gains (states 17, 19, 21, 26, 28, 31, 32) and 7 losses (states 15, 20, 22, 30, 35, 36, 43).
- **Because the noise is unseeded and unpaired**, these flips are not attributable to steering. Run-to-run policy stochasticity alone changes single-state outcomes (Phase 1D measured that the noise seed alone moves this task's baseline between 7/15 and 13/15).
- Among the 12 episodes that succeeded in both conditions, COAST finished sooner on average (250.9 → 224.6 steps). This is descriptive only.
- Baseline note: the paper's "Base" (0.53) appears to be the 15-episode fit-set rate (Phase 4B, finding F1), not a test-set baseline, so it is not the right comparator for our 19/30 baseline. Our fit-state rate is 8/15 = 0.53 (Phase 1A), which does match it.

## 4. Difference from the paper

| | Paper | Ours | Difference |
|---|---|---|---|
| Global COAST test success | 28/30 (0.933) | 19/30 (0.633) | **−9 episodes (−0.30)**; Fisher p = 0.010 |
| Wilson 95% | 0.79–0.98 | 0.46–0.78 | intervals do not overlap |
| Config | L5, α 0.5, β 0.1 | L5, α 0.5, β 0.1 | none |
| Model, checkpoint, task | π0.5, `openpi-libero-2000`, KS3 | same | none (weights unverifiable) |
| Fit set | 15 rollouts, seeds unknown | 15 rollouts, seed 0, states 0–14 | seed inferred |
| Test set | 30 rollouts, states unknown | 30 rollouts, states 15–44 | states are our interpretation |
| Conceptor construction | paper text: pooled, centered, canonical AND | released V0: flattened, uncentered, repo AND | **paper ≠ released code** (Phase 1B/4B); not decidable which built the paper's numbers |
| Selection | oracle on fit rollouts (config came out of a 240-configuration search) | none: the paper's stated config was run directly | by design (no sweep) |
| Policy noise | not described | unseeded default | matched by omission |

The gap is **not** attributable to a settings mismatch in what the paper states: model, checkpoint, task, layer, α, β and strategy are identical, and the hook fired as configured.

## 5. Context (descriptive only, not the comparison target)

Our earlier runs of the same configuration on states 15–44 (Phase 4A, different noise draws) gave baseline 21/30 and COAST 20/30; Phase 3A on states 30–44 gave 11/15. This run's 19/30 and 19/30 are in the same range, so the gap to 28/30 is stable across independent noise draws of the same protocol on the same states. That comparison is between our own runs; it does not make the paper's protocol any more or less matched.

## 6. Interpretation and limitations

**What this run measures:** with the released implementation (V0 builder), the paper's stated global KS3 configuration, our 15-episode fit set (states 0–14) and a 30-episode test on states 15–44 with unseeded noise, global COAST succeeded in 19/30 episodes. The paper reports 28/30. This is the reproduction gap under this interpretation.

**What it does not show:** it does not show that COAST fails. It does not exclude any of these causes of the gap:
1. **Unknown paper seeds and states.** The paper lists no fit or test state IDs. If its 30 test episodes were another window, or (if its runs predate PR #48) states 0–29 that overlap the fit states, the comparison here is not like for like. Difficulty varies by state: on the earlier phases the baseline alone ranged 8/15 to 13/15 across 15-state windows of this task.
2. **Unknown policy noise.** The paper does not say whether sampling noise was seeded or how many runs each cell reflects; our runs are single, unpaired, unseeded runs (test-set SD ≈ 0.09 at N = 30).
3. **Possible code-version differences.** Which repository or branch state produced the paper's numbers is unknown; `--seed` semantics changed on 2026-04-23 (PR #48), after the authors' released fit dataset.
4. **Conceptor construction.** The paper text (pooled, centered, canonical AND) and the released builder differ (Phase 1B); we tested the released one only. The paper-mathematics arm remains untested for behaviour at this configuration and split.
5. **Evaluation definition mismatch.** Whether the paper's unsteered "Base" is a fit-set rate and how its 30 test episodes were chosen are unresolved (Phase 4B).
6. **Other:** LIBERO/robosuite/MuJoCo versions not stated in the paper (ours: LIBERO `d63b117`, robosuite 1.4.0, MuJoCo 3.2.3); checkpoint weights cannot be hash-verified against the authors'; hardware (RTX 4090 here; the paper cites a B200 for latency).

**Other limitations of this run:** one run per condition; test states 15–44 were used in earlier phases and are not fresh to the researcher (no selection was made on them, since no search was run); A and B are not noise-paired; the fit-state steered behavior was not evaluated (by instruction).

**Reading:** the reproduction gap is real under this interpretation (Fisher p = 0.010, non-overlapping intervals) and stable against noise redraws of the same states, but its cause is unresolved. It should be closed by the author-facing questions in Phase 4B §9 (test states and seeds, code version, Base definition, builder) and not by tuning.

## 7. Files and housekeeping

- Added: `research/reproduction/phase5a_ks3_global_reproduction.md` (this report), `research/reproduction/experiments/phase5a_ks3_global_reproduction.yaml`, `research/reproduction/experiments/phase5a_ks3_episode_results.csv`, `research/reproduction/tools/analyze_phase5a_ks3.py`.
- Not committed (gitignored): `conceptors/libero_conceptors.npz` (755 MB), `examples/libero_env/output/phase5a_ks3_*` (videos, logs), preregistration, driver and server log.
- The server was stopped after the runs. No sweep and no improvement was started, per the stop condition. Nothing was committed or pushed.
