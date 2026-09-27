# Phase 7B — Pre-registration (paired mechanism decomposition)

- Written 2026-09-26, **before any Phase 7B rollout**. No Phase 7B rollout of any kind, and so no Phase 7B outcome, existed when this file was written.
- Branch `exp/paired-mechanism-decomposition`, HEAD `4356a576125826bd65e4c6f2391110e947377d21` (Phase 7A).
- Before writing this file, the only things run were pre-rollout engineering checks on a **synthetic** observation (no LIBERO environment, no success outcome): noise pairing at the sampler, determinism, shrinkage math and baseline-path equivalence (§8).
- Paper: arXiv:2605.17144. Implementation baseline: the closest documented public pipeline, `upstream/miranda-v2` @ `29059a537ce123abdab7a4e164db0c72eeff8024`, unmodified. This is **not** claimed to be the unpublished final-paper implementation (author clarification, recorded in the report).

## 1. Research question

Does the behavioral effect of the COAST Global intervention come mainly from its generic contraction term, or does the learned conceptor term add measurable behavior beyond contraction?

With the candidate's row-vector hook, the implemented transformation is h′ = h·Mᵀ with M = (1−β)I + βC. In column notation that is h′ = (1−β)h + β·C·h. The decomposition is:
- COAST: h′_coast = (1−β)h + β·C·h
- Pure shrinkage: h′_shrink = (1−β)h
- **Learned-conceptor residual:** r_C = h′_coast − h′_shrink = β·C·h (row form: β·h·Cᵀ)

## 2. Hypotheses

- **H0 (contraction-dominated).** Under matched states and paired policy noise, ORIGINAL_COAST and PURE_SHRINKAGE produce similar behavior, so β·C·h adds little beyond contraction.
- **H1 (conceptor-specific).** Under matched states and paired policy noise, ORIGINAL_COAST produces reproducible behavioral differences from PURE_SHRINKAGE.

Neither is assumed.

## 3. Tasks and configurations (fixed; the Phase 7A pre-registered selection)

| Task | libero_10 task_id | Layer | α | β |
|---|---|---|---|---|
| LR2a | 0 | 11 | 0.5 | 0.3 |
| KS4 | 3 | 5 | 1.0 | 0.1 |
| LR1 | 7 | 11 | 0.5 | 0.3 |

- **Conceptors:** the exact Phase 7A per-task NPZs. They were built once, on 2026-09-25 17:37, by the candidate `for_subin/build_conceptors.py` (defaults) from the authors' fit activations (`brandonyang/pi05-libero-activations-v1-2000-15env` @ `3e3a8fe2`). They were never rebuilt and never fit on held-out states. SHA-256:
  - LR2a `eef08a12612bac7544120e9992d0ae675ad0f0478c1d75a8d553b6af9f47b1dc`
  - KS4 `7bf9e02f2768e1ea97402d1ad09bed33fcc0186ab5ec72c33817289a2b093c9e`
  - LR1 `6fda3c1d9aaf0d4ca6a58beefaacf7d810f555726e543ca76a970026ec3c274e`
- **Key:** `{task}__L{layer}__{alpha}__C_contrastive`.
- **Checkpoint:** `checkpoints/openpi-libero-2000` (HF `brandonyang/openpi-libero-2000` rev `aaeeabc72f8a`).

## 4. State window

- Held-out init states **15–44** only (30 per task).
- Implemented with `state_offset = 15` and `num_episodes = 30`: episode k runs init state 15 + k.
- States 0–14 are not used in the primary experiment.

## 5. Conditions (fixed order within each driver invocation)

Every steered condition uses the **candidate `ConceptorSteeringHook`**. It is a forward hook on action-expert layer `layer` that replaces the layer output with `h @ M.to(h.dtype).T`, on all suffix/action tokens and all 10 denoising steps.

1. **BASELINE:** no hook (the candidate `SteeredPolicyWrapper` routes to `Policy.infer`), so h′ = h.
2. **ORIGINAL_COAST:** C = the NPZ `C_contrastive`, M = (1−β)I + βC.
3. **PURE_SHRINKAGE:** C = a d×d float32 zero matrix, so M = (1−β)I exactly (fp32), giving h′ = (1−β)h. There is no conceptor, no random projection and no normalization. It is the same hook class, layer, tokens, denoising steps, dtype cast and matmul as COAST.
4. **RANDOM_CONTROL:** C = candidate `compute_random_conceptor(seed = 100·layer + int(10·β))`, which gives seed 1103 for L11/β0.3 and seed 501 for L5/β0.1. M = (1−β)I + βC_rand.

Fixed everywhere:
- the unmodified candidate client (`env.seed(7)`, `np.random.seed(7)`, max 520 steps, 10 settle steps, replan every 5);
- `TORCH_COMPILE_DISABLE=1`;
- `LIBERO_CONFIG_PATH` as in all earlier phases.

## 6. Deterministic paired policy noise (Phase 1D scheme, reused unchanged)

- **Key:** `(master_seed, task_id, init_state, rollout_step)`.
- **Noise:** `numpy.random.Generator(PCG64(SeedSequence([master_seed, task_id, init_state, rollout_step]))).standard_normal((10, 32), float32)`.
- **Code:** Phase 1D's own `derive_noise` / `noise_fingerprint`, loaded by path from `src/openpi/serving/noise_control.py`. The golden fingerprint for key (100, 2, 15, 0) is `87ae314a…`; it is checked at every driver start.
- **`rollout_step`:** the post-settle env step at which the client requests a chunk (0, 5, 10, …). The client wrapper counts actual `env.step` calls since `set_init_state`, and asserts that the count equals 5 × the request index.
- **Repeat:** encoded one-to-one by `master_seed`. Repeat 1 = **100**, repeat 2 = **200** (as in Phase 1D).
- **Denoising sample identity:** the flow sampler draws one initial noise tensor per request, and the 10 Euler steps are deterministic given it. No further key field is needed.
- **Why this is independent of trajectory length and RNG history:** the noise is a pure function of the key. The global torch RNG is never used for it, and no process-level seeding is relied on.
- **Injection:** `tools/phase7b_paired_driver.py` wraps the candidate stack. For each request it shadows the sampler entry points (`Policy._sample_actions` for BASELINE, `model.sample_actions_with_steering` for the steered conditions) with shims that pass `noise=torch.from_numpy(noise).to(device)[None]`. That is the same conversion `Policy.infer` uses.
- **Checks on every request:**
  - the shim fingerprints the tensor actually handed to the sampler;
  - there is exactly one sampler call;
  - the sampler path matches the condition;
  - the sampler SHA equals the derived SHA.
- **Client side** (`tools/phase7b_paired_client.py`): wraps the candidate client for the state window (as in Phase 7A) and the key. It requires a matching echo on every response and logs each request.
- **Required verification:** at every shared (init_state, rollout_step) coordinate within a (task, repeat), all four conditions carry identical noise SHA-256 values. Across repeats, no fingerprint is shared.

## 7. Repeats, execution and size

- 2 paired-noise repeats. Within a repeat, all four conditions are paired. Between repeats, the schedules are independent.
- **GPU 0:** repeat 1 (seed 100), ports 8810–8812, queue KS4 → LR1 → LR2a.
- **GPU 1:** repeat 2 (seed 200), ports 8910–8912, same queue order.
- One driver invocation per (task, repeat), one in-process server, conditions in the §5 order.
- **Size:** 3 tasks × 30 states × 4 conditions × 2 repeats = **720 primary episodes**.
- No early stopping and no tuning.
- **Infrastructure failure rule:** if an invocation crashes, the whole (task, repeat) invocation is rerun from scratch with the identical command, and the incident is recorded. Partial outcomes are never used to decide anything.
- **Smoke test (permitted after this file is frozen):** init state 0 (outside the primary window), `max_steps 15` (no success is possible), all four conditions, one task. It checks the pipeline end to end and is not a result.

## 8. Pre-rollout engineering checks already run (synthetic observation; no outcomes)

These were run on the real checkpoint on GPU 0 with `tools/phase7b_gpu_checks.py`:
- The sampler noise SHA is identical under all 4 conditions for the same key. Different keys give different noise, and the golden fingerprint matches.
- The same key twice gives max |Δaction| = 0.0 in every condition.
- `Policy.infer(noise)` (BASELINE path) vs the steering sampler with an identity hook (β = 0) gives max |Δaction| = **0.0** for all three task configurations. The two sampler paths are numerically identical.
- PURE_SHRINKAGE:
  - M == (1−β)I exactly;
  - fp32 output == (1−β)h exactly at "highest" matmul precision;
  - on the bf16 runtime path, the output is bit-identical to bf16(1−β)·h, with applied factors bf16(0.7) = 0.69921875 and bf16(0.9) = 0.8984375;
  - the relative deviation from exact (1−β)h is ≤ 0.27%.
- The candidate C_contrastive is **not symmetric** (‖C − Cᵀ‖_F/‖C‖_F ≈ 0.35–0.46, measured offline). All decompositions will therefore use the implemented row-vector form h·Cᵀ.

## 9. Primary comparisons and statistics

Comparisons, for each task:
1. **ORIGINAL_COAST vs PURE_SHRINKAGE (primary).**
2. ORIGINAL_COAST vs RANDOM_CONTROL.
3. ORIGINAL_COAST vs BASELINE.

For each task × comparison:
- **Paired 2×2 table per repeat** over the 30 states: both succeed / COAST only (b) / other only (c) / both fail.
- **Exact McNemar test per repeat (primary test):** two-sided binomial test on the discordant pairs, p = min(1, 2·P[X ≤ min(b, c)]) with X ~ Bin(b + c, ½). p = 1 when b + c = 0.
- **Pooled table (descriptive only).** The two repeats share init states, so they are not independent.
- **Paired risk difference:** RD = mean(success_COAST − success_other), per repeat and pooled.
- **State-cluster bootstrap:** resample the 30 init states with replacement within a task, keeping both repeats of a state together. 10,000 resamples with `numpy.random.default_rng(7070)` give a percentile 95% CI for the pooled RD.
- **Cross-task (descriptive):** the mean of the three task RDs, with a stratified state-cluster bootstrap (same seed). With n_tasks = 3, no population-level inference is made.

## 10. Operator decomposition (offline; after the behavioral records are frozen)

Computed on two sets of hidden states:
- (i) the authors' fit token vectors at the task's layer: every 10th inference step, all 10 denoising steps. This is the Phase 7A set.
- (ii) the BASELINE hidden states at the hooked layer on the first observation (rollout_step 0) of each held-out state 15–44, under each repeat's noise. These are captured by an exploratory probe after the primary records are frozen.

Metrics, in float64, using the implemented row form (δ_coast = h·Mᵀ − h, δ_shrink = −βh, r_C = β·h·Cᵀ):
1. ‖δ_coast‖/‖h‖
2. ‖δ_shrink‖/‖h‖
3. ‖r_C‖/‖h‖
4. ‖r_C‖/‖δ_coast‖
5. cos(δ_coast, δ_shrink)
6. cos(δ_coast, h)
7. cos(r_C, h)
8. hᵀCh/‖h‖²
9. tr(C)/d
10. ‖δ_coast − δ_shrink − r_C‖/(‖δ_coast‖ + ε)

Also reported:
- ‖C − Cᵀ‖_F/‖C‖_F;
- the same metrics for RANDOM_CONTROL;
- the realized bf16 variant (M cast to bf16, as the hook does).

## 11. Decision rules (primary comparison COAST vs PURE_SHRINKAGE)

**Definitions, per task:**
- **Clear paired difference:** the bootstrap 95% CI of the pooled RD excludes 0, **and** the per-repeat RDs have the same nonzero sign. The direction is the sign of the pooled RD.
- **Repeatable McNemar difference:** exact McNemar p < 0.05 in **both** repeats, in the same direction. This is reported. By itself it also counts as a clear difference.

**Cases:**
- **Case A (contraction dominates):** no task shows a clear COAST-vs-SHRINKAGE difference.
  - Supporting geometry, reported but not gating: cos(δ_coast, δ_shrink) ≥ 0.99 and ‖r_C‖/‖δ_coast‖ ≤ 0.2.
  - If the behavior fits Case A but the geometry does not, the case stays A and the discrepancy is flagged.
  - Phase 7A's diagnostics (cos(Δh, h) ≈ −0.998) already suggest the geometry will pass. Geometry is therefore supporting evidence, not the tested outcome.
- **Case B (learned conceptor contributes):** a clear **positive** (COAST > SHRINKAGE) difference in ≥ 2 of 3 tasks, and no task with a clear negative difference.
- **Case C (task dependent):** at least one task shows a clear difference, but Case B does not hold. In that case, examine whether quota, overlap, hᵀCh, ‖r_C‖, layer or β track the difference (n = 3, descriptive only).

COAST vs RANDOM and COAST vs BASELINE are secondary. They are reported with the same machinery and do not enter the case rule.

Wording follows the brief. In Case A: "largely consistent with its generic hidden-state contraction component", never "the conceptor has zero effect".

## 12. Exploratory (not gating; does not delay the primary experiment)

**Action-level analysis at identical observations.** For each held-out state and repeat:
- the first observation (rollout_step 0, after the 10 settle steps; identical across conditions because no policy action has yet been applied);
- the key (seed, task_id, state, 0);
- the actions under all four conditions.

Metrics:
- ‖a_coast − a_base‖, ‖a_shrink − a_base‖, ‖a_coast − a_shrink‖, ‖a_rand − a_base‖;
- cos(a_coast − a_base, a_shrink − a_base).

Later actions from diverged trajectories are **not** compared.

## 13. Stop condition

Phase 7B is complete when:
- all 720 primary episodes exist, from 2 repeats × 3 tasks × 4 conditions × states 15–44;
- noise pairing is verified;
- the paired analysis, clustered uncertainty and operator decomposition are done;
- the case is classified and a Phase 8 recommendation is written.

No Phase 8 method is implemented. No new steering equation is introduced.
