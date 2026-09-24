# Phase 3A — Held-Out Validation (init states 30–44)

**HELD-OUT split: `--seed 30`, init states 30–44.** This is the first behavioral use of these states. No Phase 3A result uses states 15–29; the development numbers appear only as the comparison asked for in item 5.

- Nothing was tuned, rerun, or changed in code: no steering, conceptor or checkpoint changes, and no new activations.
- Every statement holds only for this one task, layer, β, conceptor, random basis and noise seed.

Paper: *Contrastive Conceptor Activation Steering (COAST)*, arXiv:2605.17144.
- Commit: `a5c8763a32f891781a6c2d7ce2706faa130bbf41` (Phase 2B), branch `exp/heldout-validation`.
- The **working tree was clean** during the runs; no code changes were made.

## 0. Summary

| condition | M | held-out 30–44 | Wilson 95% | dev 15–29 (Phase 2B) |
|---|---|---|---|---|
| A. Baseline | I | **9** / 15 | 0.36–0.80 | 13 / 15 |
| B. Shrinkage | 0.9·I | **10** / 15 | 0.42–0.85 | 10 / 15 |
| C. Real COAST (`global`) | 0.9·I + 0.1·C_real | **11** / 15 | 0.48–0.89 | 11 / 15 |
| D. Random matched | 0.9·I + 0.1·C_random | **11** / 15 | 0.48–0.89 | 11 / 15 |

**Answers for this held-out task and configuration:**

1. **Does COAST improve performance on unseen states?** Not demonstrably.
   - COAST gains 2 baseline failures (states 39, 41) and loses none: 9 → 11, exact McNemar p = 0.5.
   - On the development states the sign was reversed (13 → 11: three losses, one gain).
   - Shrinkage and random matched show the same held-out pattern: gains only (1 and 2), no losses.
   - The difference is within the baseline's noise-seed spread (7–13/15 in Phase 1D) and reverses between splits. It is **not** evidence that COAST works, nor that it hurts.
2. **Does pure shrinkage explain COAST's behavior?** Yes, to the same degree as on the development states.
   - COAST and shrinkage agree on **14/15** held-out outcomes (development: 14/15).
   - At the hidden-state level the two interventions are the same ~10% contraction as before. The norm ratios, 0.8984 vs 0.9006, match development to 4 decimals.
3. **Does random matched behave like real COAST?** Largely, but not perfectly.
   - They agree on **13/15** held-out outcomes (development: 15/15).
   - The two disagreements go in opposite directions, and in both real COAST sides with shrinkage:
     - state 38: random succeeds; COAST and shrinkage time out;
     - state 39: random times out; COAST and shrinkage succeed.
   - Neither control is better: random matched and COAST have the same count (11 = 11), and random matched differs from shrinkage on 3 states in both directions.
- **Across all 30 states (development + held-out), real COAST is never the only condition that differs from the other steered conditions.**
  - Every disagreement among the steered conditions has exactly one condition deviating, and that condition is always a control: shrinkage at states 21 and 41, random matched at 38 and 39.
  - A direction-specific COAST effect would show up as the opposite pattern: COAST alone differing from both controls. That was not observed.
- **Mechanistic conclusions survive.** At layer 5 with β = 0.1, all three steered conditions act as a ~10% contraction of h along h, with the same magnitudes as on development. The behavioral pattern is consistent with that contraction plus trajectory-level chaos. It shows no detectable contribution of C_real's directions.

## 1. Pre-run verification

| check | result |
|---|---|
| `git status` | clean; branch `exp/heldout-validation` |
| `git rev-parse HEAD` | `a5c8763a32f891781a6c2d7ce2706faa130bbf41` |
| `nvidia-smi` | GPU 0: 447 MiB, 0%; GPU 1: 173 MiB, 11%. **GPU 1** used, the same GPU as Phases 2A/2B |
| `conceptors/phase1b_repo_task02.npz` | present, 377.5 MB, sha256 `8b1c802a…ec80f3` (same as Phases 2A/2B) |
| `random_matched` determinism | two separate processes: seed **1466119911** (= Phase 2B), bit-identical C_random (sha256 prefix `9f133892cb5a48b5`) |
| init-state mapping | `main.py`: episode k → `initial_states[(seed + k) % 50]`; `--seed 30` → states 30–44 |

- **Preregistration** was written at 11:29:12, before any Phase 3A rollout: `examples/libero_env/output/phase3a_servers/preregistration.txt`. It fixes:
  - the configuration;
  - the order A → B → C → D;
  - one run each, with no reruns based on results;
  - the primary questions;
  - the interpretation rule.
- **Split side effect.** `--seed` also sets `env.seed(30)` and `np.random.seed(30)`; development used 15. This is how the pipeline's held-out split is designed. It applies identically to all four held-out conditions, so it does not affect the within-split pairing.

## 2. Configuration and run

pi0.5 (`pi05_libero`, PyTorch), `checkpoints/openpi-libero-2000`, libero_10 task 2 (`KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it`), 15 episodes, max 520 steps, replan every 5, paired noise with master seed 100. Steering: layer 5, α 0.5, β 0.1.

Server, launched 2026-09-24 11:29:17 +08:00 on GPU 1:
```
CUDA_VISIBLE_DEVICES=1 uv run --no-sync scripts/serve_policy.py --pytorch --steer --noise-control --steering-diagnostics \
  --conceptor-npz conceptors/phase1b_repo_task02.npz --port 8103 \
  policy:checkpoint --policy.config pi05_libero --policy.dir checkpoints/openpi-libero-2000
```
Clients (`examples/libero_env`; driver `output/phase3a_servers/run_clients.sh`):
```
common: CUDA_VISIBLE_DEVICES=1 MUJOCO_GL=egl LIBERO_CONFIG_PATH=/home/stanley/Stanley_ws/COAST-research/libero_config_coast \
  uv run --no-sync python main.py --task_suite_name libero_10 --task_id 2 --num_episodes 15 --seed 30 --port 8103 \
  --policy_noise_seed 100 --output_dir output/phase3a_<cond>_task02_seed30
steered: <common> --steer --steering_layer 5 --steering_alpha 0.5 --steering_beta 0.1 --log_steering_diagnostics \
  --steering_strategy {shrinkage|global|random_matched}
```

| run | wall clock | exit |
|---|---|---|
| A baseline | 11:29:54–11:32:56 | 0 |
| B shrinkage | 11:32:56–11:35:47 | 0 |
| C COAST | 11:35:47–11:38:27 | 0 |
| D random matched | 11:38:27–11:41:10 | 0 |

- Each condition ran exactly once.
- The server log has 0 Traceback/ERROR lines.
- The hooks built were `ShrinkageSteeringHook`, `ConceptorSteeringHook` (`global`) and `ConceptorSteeringHook` (`random_matched`).
- The server logged `random_matched: seed=1466119911`.
- GPU 1 peaked at 8,347 MiB and 100% utilization.
- Each client printed only the 5 known benign warnings.
- The server and GPU logger were stopped afterwards.

Analysis: `CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu uv run python research/reproduction/tools/analyze_phase3a.py` (new, analysis only). All assertions passed. The same code run on the Phase 2B outputs reproduces the Phase 2B numbers exactly (13/10/11/11, 3,676 fingerprints, C–D 15/15).

## 3. Noise fingerprint validation (item 6)

- **All 3,845 logged fingerprints were recomputed** from their keys `(master_seed 100, task_id 2, init_state, rollout_step)` and match.
  - Requests per condition: A 1,051, B 967, C 905, D 922.
  - Every record's `init_state` equals 30 + episode. The set of states seen is exactly {30..44}.
- **Per-episode schedule.** Each episode's requests are exactly rollout steps 0, 5, … up to its length. The fingerprint count per episode is ⌈steps/5⌉, e.g. 104 for a 520-step timeout; per-episode counts are in the CSV.
- **Pairing.** At every shared `(init_state, rollout_step)` coordinate the noise is identical in all 6 condition pairs:

| pair | shared coordinates | identical |
|---|---|---|
| A–B | 947 | 947 |
| A–C | 902 | 902 |
| A–D | 920 | 920 |
| B–C | 887 | 887 |
| B–D | 842 | 842 |
| C–D | 844 | 844 |

  Example: init 30, rollout step 0 has fingerprint `e1f45eb191a4dc86…` in all four conditions.
- **Diagnostics.** B, C and D each have exactly 10 records per call (denoising steps 0..9), layer 5, 10 tokens, dim 1024, β 0.1, the correct strategy, the correct init_state, and all finite values. Their schedules equal the fingerprint schedules. Hook applications: B 9,670, C 9,050, D 9,220.
- **Identical inputs at the first call.** At rollout step 0 (identical observation and noise), all three steered conditions have identical layer-5 ‖h‖ at denoising step 0 in 15/15 episodes.
- **Held-out and development keys never collide.** They differ by init_state, so no noise vector is shared between the splits. The comparison in §6 is between splits, not paired.

The paired comparisons below are therefore valid within the held-out split.

## 4. Success and episode-level comparison (items 1–4)

| init | A base | B shrink | C COAST | D random | steps A / B / C / D |
|---|---|---|---|---|---|
| 30 | 0 | 0 | 0 | 0 | 520 / 520 / 520 / 520 |
| 31 | 1 | 1 | 1 | 1 | 220 / 224 / 230 / 223 |
| 32 | 1 | 1 | 1 | 1 | 232 / 210 / 194 / 201 |
| 33 | 0 | 0 | 0 | 0 | 520 / 520 / 520 / 520 |
| 34 | 1 | 1 | 1 | 1 | 216 / 216 / 216 / 225 |
| 35 | 1 | 1 | 1 | 1 | 251 / 210 / 222 / 217 |
| 36 | 1 | 1 | 1 | 1 | 228 / 226 / 232 / 229 |
| 37 | 1 | 1 | 1 | 1 | 211 / 193 / 181 / 209 |
| 38 | 0 | 0 | 0 | **1** | 520 / 520 / 520 / 264 |
| 39 | 0 | 1 | 1 | **0** | 520 / 154 / 177 / 520 |
| 40 | 1 | 1 | 1 | 1 | 255 / 230 / 235 / 230 |
| 41 | 0 | **0** | 1 | 1 | 520 / 520 / 250 / 247 |
| 42 | 0 | 0 | 0 | 0 | 520 / 520 / 520 / 520 |
| 43 | 1 | 1 | 1 | 1 | 264 / 224 / 259 / 226 |
| 44 | 1 | 1 | 1 | 1 | 239 / 334 / 229 / 238 |
| **total** | **9** | **10** | **11** | **11** | |

Rollout steps exclude the 10 settle steps; 520 means timeout. All four succeed on 9 states and all four fail on 3 (30, 33, 42). The three steered conditions agree on 12/15.

| pair | fail→success | success→fail | agreement | identical length | exact McNemar p (reference only) | disagreeing states |
|---|---|---|---|---|---|---|
| baseline → COAST | 2 | 0 | 13/15 | 5/15 | 0.5 | 39, 41 |
| baseline → shrinkage | 1 | 0 | 14/15 | 6/15 | 1.0 | 39 |
| baseline → random | 2 | 0 | 13/15 | 4/15 | 0.5 | 38, 41 |
| **COAST ↔ shrinkage** | COAST only: 1 | shrink only: 0 | **14/15** | 5/15 | 1.0 | 41 |
| **COAST ↔ random** | random only: 1 | COAST only: 1 | **13/15** | 3/15 | 1.0 | 38 (random ✓), 39 (COAST ✓) |
| shrinkage ↔ random | random only: 2 | shrink only: 1 | 12/15 | 4/15 | 1.0 | 38, 39, 41 |

- **Baseline → steered.** No steered condition loses a baseline success on the held-out states. All gains fall on states the baseline fails (38, 39, 41). Three baseline failures (30, 33, 42) are never rescued.
- **COAST vs shrinkage (item 3).** They agree on 14/15. The single disagreement is state 41: COAST (250 steps) and random matched (247) succeed, while shrinkage and baseline time out.
  - This is the same kind of pattern as development state 21, where both conceptor conditions succeeded and shrinkage did not.
  - On the development states the pattern was attributable to "either conceptor condition", not to C_real's direction. That holds here too, because random matched also succeeds at 41.
- **COAST vs random matched (item 4).** They agree on 13/15, with the two disagreements in opposite directions:
  - state 38: random succeeds at 264 steps while the other three time out;
  - state 39: random times out while shrinkage (154) and COAST (177) succeed.
  - In both, real COAST agrees with shrinkage. Neither disagreement favors a direction-carrying conceptor: random matched is "better" once and "worse" once.

## 5. Intervention magnitude and hidden-norm change (item 7)

Per hook application at layer 5:

| | n | mean ‖Δh‖ (sd; min–max) | mean ‖Δh‖/‖h‖ | mean cos(Δh, h) | mean ‖h′‖/‖h‖ | mean ‖h‖ |
|---|---|---|---|---|---|---|
| B shrinkage | 9,670 | 17.347 (2.691; 13.10–24.61) | 0.10163 | −0.99990 | **0.89838** ± 0.00014 | 170.68 |
| C real COAST | 9,050 | 16.974 (2.668; 12.76–24.13) | 0.09942 | −0.99961 | **0.90062** ± 0.00028 | 170.67 |
| D random matched | 9,220 | 16.691 (2.598; 12.59–23.66) | 0.09782 | −0.99850 | **0.90234** ± 0.00016 | 170.62 |

By denoising step (mean ‖Δh‖, with norm ratio):

| step | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|
| B ‖Δh‖ | 22.76 | 20.34 | 19.06 | 18.17 | 17.54 | 16.48 | 15.85 | 15.13 | 14.36 | 13.79 |
| C ‖Δh‖ | 22.32 | 19.94 | 18.68 | 17.80 | 17.17 | 16.13 | 15.49 | 14.78 | 14.01 | 13.42 |
| D ‖Δh‖ | 21.91 | 19.58 | 18.35 | 17.49 | 16.88 | 15.86 | 15.24 | 14.55 | 13.81 | 13.26 |
| C ratio | 0.9004 | 0.9004 | 0.9004 | 0.9005 | 0.9005 | 0.9006 | 0.9006 | 0.9008 | 0.9009 | 0.9011 |
| D ratio | 0.9023 | 0.9023 | 0.9023 | 0.9023 | 0.9023 | 0.9023 | 0.9023 | 0.9023 | 0.9023 | 0.9023 |
| B ratio | 0.8984 | 0.8984 | 0.8984 | 0.8984 | 0.8984 | 0.8984 | 0.8984 | 0.8984 | 0.8984 | 0.8984 |
| mean ‖h‖ | 224.0 | 200.1 | 187.5 | 178.8 | 172.5 | 162.1 | 155.9 | 148.8 | 141.3 | 135.7 |

- **Paired shared coordinates:**

| pair | applications | ‖Δh‖ ratio | norm-ratio difference |
|---|---|---|---|
| shrink / COAST | 8,870 | 1.0227 | −0.0022 |
| shrink / random | 8,420 | 1.0399 | −0.0040 |
| COAST / random | 8,440 | 1.0169 | −0.0017 |

- **Within-call effect** (rollout step 0, identical observation and noise): layer-5 ‖h‖ differs by at most 0.12% (COAST vs shrinkage, COAST vs random) to 0.18% (shrinkage vs random) across all 10 denoising steps.
- **Every magnitude statistic matches development to within 0.1%.**
  - Norm ratios: 0.89838 / 0.90062 / 0.90234 held-out vs 0.89838 / 0.90062 / 0.90233 on development.
  - Mean ‖Δh‖: 16.974 vs 16.963 (COAST).
  - Mean ‖h‖: 170.67 vs 170.56.
- The intervention is a fixed ~10% contraction regardless of the initial state. The contraction ordering is the same as on development: shrinkage 10.16% > COAST 9.94% > random 9.78%.

## 6. Comparison with Phase 2B development results (item 5)

| | dev 15–29 (Phase 2B) | held-out 30–44 (Phase 3A) |
|---|---|---|
| successes A / B / C / D | 13 / 10 / 11 / 11 | 9 / 10 / 11 / 11 |
| baseline → COAST | +1 −3 (p 0.625) | +2 −0 (p 0.5) |
| baseline → shrinkage | +0 −3 | +1 −0 |
| baseline → random | +1 −3 | +2 −0 |
| COAST ↔ shrinkage agreement | 14/15 (state 21) | 14/15 (state 41) |
| COAST ↔ random agreement | 15/15 | 13/15 (38, 39) |
| all three steered agree | 14/15 | 12/15 |
| odd-one-out among steered | shrinkage @21 | shrinkage @41, random @38, random @39 |
| realized ‖h′‖/‖h‖ B / C / D | 0.8984 / 0.9006 / 0.9023 | 0.8984 / 0.9006 / 0.9023 |
| cos(Δh, h) B / C / D | −0.9999 / −0.9996 / −0.9985 | −0.9999 / −0.9996 / −0.9985 |

- **The steered success counts are identical across splits (10 / 11 / 11). The baseline moves by 4 (13 → 9).**
  - Relative to baseline, the effect of steering therefore flips sign between splits: −2 on development, +2 on held-out.
  - This is what one expects if the steered policies and the baseline differ mainly by trajectory-level noise on a task whose per-state difficulty varies. Three held-out states fail under every condition; none did on development except state 28.
  - A reliable COAST effect in either direction is not supported.
- **Pooled over both splits, descriptively only (the final held-out result is §4):**
  - baseline 22/30, shrinkage 20/30, COAST 22/30, random matched 22/30;
  - baseline → COAST +3 −3;
  - COAST ↔ shrinkage 28/30;
  - COAST ↔ random 28/30.
- **One development observation weakens on the held-out states.** The perfect COAST ↔ random agreement (15/15) drops to 13/15.
  - The two disagreements go in opposite directions, and COAST agrees with shrinkage on both.
  - COAST ↔ random is now no closer than COAST ↔ shrinkage.
  - So the development result "random matched reproduces COAST exactly" should be restated as: *random matched, shrinkage and real COAST are behaviorally interchangeable up to occasional single-state divergences, none of which is specific to real COAST.*

## 7. Do the mechanistic conclusions survive? (item 8)

- **Hidden-state mechanism: survives.** On the held-out states the steered hidden states behave exactly as on development:
  - norm ratios equal to 4 decimals;
  - cos(Δh, h) between −0.9985 and −0.9999;
  - the same per-denoising-step profile;
  - a within-call ‖h‖ difference of at most 0.18% between steered conditions.

  COAST at layer 5, β = 0.1 remains an approximately 10% contraction of the residual stream along h.
- **"Shrinkage explains COAST" (Phase 2A): survives.** COAST ↔ shrinkage agreement is 14/15 on both splits. The single disagreement is in both cases a shrinkage-only failure where random matched also succeeds, so it is not attributable to C_real's directions.
- **"Conceptor direction does not matter" (Phase 2B): survives, in a weaker form.**
  - Random matched no longer matches COAST outcome for outcome (13/15).
  - However, there is no state among 30 where real COAST differs from both controls. The held-out disagreements are symmetric (random +1 / −1 relative to COAST).
  - So there is still **no detectable direction-specific effect**. The random/real divergences behave like the trajectory-level variability already seen in the steered path's cross-process length jitter (Phases 2A/2B).
- **"Steering changes success rate": not supported on either split.** The baseline-relative effect reverses sign between splits and stays within the Phase 1D baseline noise-seed spread (7–13/15). Nothing here supports "COAST works" or "COAST is useless".

**Interpretation (per the preregistered rule).** Under this held-out task and configuration, the observed mechanism generalizes beyond the development states:
- COAST acts as a state-independent ~10% hidden-state contraction;
- pure shrinkage and a spectrum-matched random conceptor reproduce its behavior up to single-state divergences not specific to C_real.

The behavioral *effect* of that mechanism on success rate is not stable across splits, and N = 15 per split cannot resolve it.

## 8. Limitations

- **Small N:** 15 held-out episodes per condition, one master noise seed, one run per condition, one random basis. All pairwise McNemar p ≥ 0.5.
- **Single configuration:** one task, layer 5, β 0.1, α 0.5, the V0 conceptor. Other tasks, layers and β values were not tested, by design.
- **Split seed side effect:** `--seed 30` also changes `env.seed` and `np.random.seed` relative to development. This affects the cross-split comparison (§6), not the within-split pairing.
- **No held-out replay:** each condition ran once. Steered-path cross-process jitter (up to 30 rollout steps in Phase 2B, no outcome change) means single-state disagreements between steered conditions cannot be individually interpreted.
- **Not magnitude-matched:** the contractions are 10.16% / 9.94% / 9.78% (B / C / D), as documented in Phases 2A/2B, and were not corrected.
- **Measured at the hooked layer only.**

## 9. Files and housekeeping

- **Added (analysis only; no source, test or config changes):**
  - `research/reproduction/tools/analyze_phase3a.py`
  - `research/reproduction/experiments/phase3a_episode_results.csv` — long format, one row per condition × episode: condition, strategy, episode, init_state, success, rollout_steps, noise_fingerprints, hook_applications, mean ‖Δh‖, relative Δ, norm ratio, cosine, mean ‖h‖
  - `research/reproduction/experiments/phase3a_heldout_validation.yaml`
  - this report
- **Run outputs** (gitignored, not committed):
  - `examples/libero_env/output/phase3a_{baseline,shrinkage,coast,random_matched}_task02_seed30/`, 13–17 MB each: videos, client logs, fingerprint and diagnostics JSONL;
  - `…/phase3a_servers/`: preregistration, driver, server log, GPU trace.
- `~/Stanley_ws/lerobot` was not touched. No activations were collected, and no checkpoint or NPZ was modified.
