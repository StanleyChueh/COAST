# Phase 7B — Paired Mechanism Decomposition: COAST vs Pure Shrinkage (π0.5, LIBERO-10, held-out states)

**Question:** Does the behavioral effect of the COAST Global intervention come mainly from its generic contraction term (1−β)h? Or does the learned conceptor term β·C·h add measurable behavior beyond contraction?

| | |
|---|---|
| Branch | `exp/paired-mechanism-decomposition`, HEAD `4356a576…` (Phase 7A), clean at start |
| Pre-registration | `phase7b_preregistration.md`, sha256 `dd1a86bf…2df280`, frozen **2026-09-26 14:11:26 +08:00**, before any Phase 7B rollout |
| Implementation baseline | `upstream/miranda-v2` @ `29059a537ce1…`, worktree `../COAST-paper-candidate`, unmodified. This is the closest documented public pipeline. |
| Records | `experiments/phase7b_episode_results.csv` (720 rows), `phase7b_summary.csv`, `phase7b_paired_tables.csv`, `phase7b_analysis.json`, `phase7b_operator_decomposition.csv`, `phase7b_action_probe.csv`, `phase7b_exploratory.json`, `phase7b_paired_mechanism_decomposition.yaml` |

Evidence tags: **[M]** measured, **[P]** pre-registered rule, **[E]** exploratory, **[I]** inference.

## 1. Executive summary

- **Design.** We ran 3 tasks (LR2a, KS4, LR1), each at its paper Global configuration, on held-out states 15–44. Each task had 4 conditions: BASELINE, ORIGINAL_COAST, PURE_SHRINKAGE and RANDOM_CONTROL. Each condition ran in **2 paired-noise repeats** (master seeds 100 and 200), for a total of **720 episodes**. All six invocations exited with rc = 0, and nothing was rerun.
- **Noise pairing holds** [M]:
  - All 112,012 logged fingerprints were recomputed independently, and every one matched.
  - Within each (task, repeat), all four conditions carried **identical noise at every shared (state, rollout_step) coordinate**: 1,809–2,283 coordinates per condition pair, 100% identical.
  - No fingerprint is shared between the two repeats.
  - The noise tensor handed to the sampler matched the derived noise on every request.
- **Operator decomposition** [M]. COAST's displacement is almost exactly the shrinkage displacement:
  - cos(δ_coast, δ_shrink) = **0.9976–0.9995**;
  - ‖r_C‖/‖δ_coast‖ = **0.035–0.085**;
  - the identity δ_coast = δ_shrink + r_C holds to ≤ 7.6 × 10⁻¹⁵.

  The learned residual r_C = β·C·h is 0.3–2.4% of ‖h‖, and it is partly aligned with h (cos 0.44–0.65). Its main effect is to make COAST a slightly **weaker** contraction than pure shrinkage.
- **Action level at identical observation and noise** [E]. ‖a_coast − a_shrink‖ is 6–22% of ‖a_shrink − a_base‖. The two steering deltas point the same way (cos 0.98–0.999).
- **Behavior. Primary paired comparison, COAST − SHRINKAGE** [M]:

  | Task | Rep 1 b/c (McNemar p) | Rep 2 b/c (p) | Pooled RD | State-cluster 95% CI |
  |---|---|---|---|---|
  | LR2a | 4/7 (0.55) | 1/6 (0.13) | **−0.133** | **[−0.250, −0.017]** |
  | KS4 | 7/2 (0.18) | 3/4 (1.0) | +0.067 | [−0.067, +0.200] |
  | LR1 | 2/7 (0.18) | 2/4 (0.69) | −0.117 | [−0.233, 0.000] |

  - Cross-task mean RD = **−0.061** [−0.133, +0.006].
  - No McNemar test reached p < 0.05 in any repeat of any comparison.
- **Pre-registered classification: Case C (task dependent)** [P]. Only LR2a meets the "clear paired difference" rule, and **its sign is negative**: COAST did *worse* than pure shrinkage.
  - Case B (a positive contribution in ≥ 2 tasks) is clearly not met.
  - **No task shows a positive, conceptor-specific behavioral contribution.**
  - The single "clear" call is a bootstrap-CI result, not a McNemar result. It is one of nine task × comparison cells, so it is compatible with chance.
- **Secondary** [M]. COAST vs RANDOM: RD −0.083 / +0.083 / −0.067, none clear. COAST vs BASELINE: RD +0.033 / −0.017 / +0.083, none clear.
- **Reading** [I]. Under our checkpoint and held-out protocol, **nothing in the data credits the learned conceptor term with behavior beyond contraction**. The geometry is contraction-dominated. The one detectable paired difference goes against COAST. The formal label is Case C because the pre-registered rule counts clear differences of either sign.

## 2. Motivation from Phases 1E / 2A / 2B / 7A

- **1E** (KS3, L5 β0.1): the COAST hook is ≈ a uniform 0.90·h shrink. cos(Δh, h) = −0.9996, and hᵀCh/‖h‖² ≈ 0.006.
- **2A** (KS3, paired noise, one seed): pure shrinkage reproduced COAST's outcomes on 14/15 states.
- **2B** (KS3, paired noise, one seed): a spectrum-matched random conceptor reproduced COAST on 15/15.
- **7A** (LR2a, KS4, LR1; unpaired, 2 repeats):
  - On held-out states, Global and Random were identical in total (109/180 each).
  - The operators were contraction-dominated (cos(Δh, h) −0.998 to −0.9996).

Phases 2A/2B were one task with one seed. Phase 7A had no shrinkage arm and no paired noise. Phase 7B combines paired noise, a shrinkage arm, three tasks, held-out states and two independent noise repeats. That lets it test the contraction explanation directly.

## 3. Author clarification and scope

After Phase 7A, the first author clarified by email. This is author-provided context:
- The public GitHub branch we inspected is older.
- The initial evaluations used 15 rollouts.
- Later evaluations used 30 rollouts.
- The later work was done on a separate branch that was not published.
- Torch randomness was not fixed in the initial setup, so exact reruns are not expected to reproduce identical sample results. Environment and seed differences may also matter.

So Phase 7B is **not** a claim to reproduce the unpublished final-paper implementation. It does not target the paper's exact numbers. It is a controlled mechanistic analysis of the closest documented public pipeline (`miranda-v2` @ `29059a5`), under our checkpoint and protocol. We do not claim that this pipeline generated any particular paper table.

## 4. Mathematical decomposition

The candidate hook (`ConceptorSteeringHook`) applies, per token row vector h ∈ ℝ^d:

h′ = h · M.to(h.dtype)ᵀ, with M = (1−β)I + βC.

In column notation this is h′ = (1−β)h + β·C·h. Since the code multiplies by Mᵀ, it applies **C** (not Cᵀ) in column form.

**Symmetry check (Step 17)** [M]. The candidate C_contrastive = C_s(I − C_f) is a product of two symmetric matrices, and it is **not symmetric**. ‖C − Cᵀ‖_F/‖C‖_F:

| Task | Asymmetry |
|---|---|
| LR2a | 0.458 |
| KS4 | 0.347 |
| LR1 | 0.441 |

Eigenvalues are real, ranging from about −4 × 10⁻⁶ to 0.39 / 0.49 / 0.50. All numerical decompositions therefore use the implemented row form, h·Cᵀ.

Displacements (row form):
- δ_coast = h·Mᵀ − h = β(h·Cᵀ − h)
- δ_shrink = (1−β)h − h = −βh
- r_C = h′_coast − h′_shrink = β·h·Cᵀ (column form: β·C·h)
- Identity: δ_coast = δ_shrink + r_C

Note that hᵀCh = hᵀ·sym(C)·h, so the Rayleigh quotient is unaffected by the asymmetry.

## 5. Pre-registered hypotheses

- **H0 (contraction-dominated).** With matched states and paired noise, COAST and pure shrinkage behave similarly, and β·C·h contributes little.
- **H1 (conceptor-specific).** COAST shows reproducible behavioral differences from pure shrinkage.

**Decision rule** [P], applied per task to COAST vs SHRINKAGE. A "clear paired difference" means:
- the state-cluster bootstrap 95% CI of the pooled RD excludes 0, **and** the RD has the same nonzero sign in both repeats;
- **or** the McNemar p < 0.05 in both repeats, in the same direction.

The cases are:
- **A:** no task shows a clear difference.
- **B:** a clear positive difference in ≥ 2 tasks, and no clear negative difference.
- **C:** otherwise, when at least one task is clear.

## 6. Exact task / configuration selection

The task selection is Phase 7A's pre-registered one, made from paper values only. It was not changed.

| Task | task_id | Layer | α | β | Conceptor NPZ (Phase 7A, sha256) |
|---|---|---|---|---|---|
| LR2a | 0 | 11 | 0.5 | 0.3 | `eef08a12…b1dc` |
| KS4 | 3 | 5 | 1.0 | 0.1 | `7bf9e02f…c9e` |
| LR1 | 7 | 11 | 0.5 | 0.3 | `6fda3c1d…274e` |

About the conceptors:
- They are the exact Phase 7A NPZs: candidate `for_subin/build_conceptors.py` with defaults, run on the authors' fit activations.
- They were hard-linked from the Phase 7A scratch directory. They were not rebuilt and never fit on held-out states.
- RANDOM_CONTROL uses the candidate `compute_random_conceptor` with seed 1103 (L11/β0.3) or 501 (L5/β0.1).
- The checkpoint is `openpi-libero-2000` (rev `aaeeabc`).

Fixed settings:
- the unmodified candidate client: `env.seed(7)`, `np.random.seed(7)`, max 520 steps, 10 settle steps, replan every 5;
- `TORCH_COMPILE_DISABLE=1` (as in Phase 7A).

**PURE_SHRINKAGE implementation (Step 6).**
- It uses the **same candidate hook class**, built from a d×d float32 zero matrix. So M = (1−β)I + β·0 = (1−β)I exactly.
- It therefore acts at the identical point: the forward hook on `gemma_expert.model.layers[layer]`, all 10 suffix/action tokens, all 10 denoising steps.
- It has the same `M.to(h.dtype)` cast and the same `h @ M.T` matmul.
- Nothing else changes. There is no conceptor, no random projection and no normalization.

Verification on the served hook object (`verify_shrinkage_hook`, run at every driver start) [M]:
- M == (1−β)I exactly (fp32).
- At "highest" matmul precision, the fp32 output equals (1−β)h exactly (max |Δ| = 0).
- On the bf16 runtime path, the output is **bit-identical** to bf16(1−β)·h. The applied factors are 0.69921875 (β 0.3) and 0.8984375 (β 0.1), and the relative deviation from exact (1−β)h is ≤ 0.27%.
- COAST's M goes through the same bf16 cast, so both operators share this rounding.
- Under the model's default TF32 setting, an fp32 input would deviate by 3.6 × 10⁻⁴. This path is unused, because h is bf16 at this layer.

## 7. Deterministic paired-noise design

This reuses the Phase 1D scheme, unchanged.

**Key and noise.**
- The key is (master_seed, task_id, init_state, rollout_step).
- The noise is `PCG64(SeedSequence(key)).standard_normal((10, 32), float32)`.
- It is computed with Phase 1D's own `derive_noise` and `noise_fingerprint`, loaded by path from the main repo. The golden fingerprint `87ae314a…` is checked at every driver start.
- The repeat is encoded one-to-one by master_seed (repeat 1 = 100, repeat 2 = 200).
- The sampler draws one initial noise tensor per request, and the 10 Euler steps are deterministic given it. No per-denoising-step key is needed.
- The noise is a pure function of the key. It never touches the global torch RNG, so it does not depend on trajectory length or on prior RNG use.

**Injection** (`tools/phase7b_paired_driver.py`, outermost wrapper around the unmodified candidate `SteeredPolicyWrapper`). Per request:
- it pops the key and derives the noise;
- it shadows the sampler entry points, `Policy._sample_actions` (BASELINE path) and `model.sample_actions_with_steering` (steered path), with shims;
- each shim passes `noise = torch.from_numpy(noise).to(device)[None]`, the same conversion `Policy.infer` uses;
- it asserts exactly one sampler call, the expected sampler path, and that the SHA-256 of the tensor handed to the sampler equals the derived SHA.

**Client** (`tools/phase7b_paired_client.py`), wrapping the unmodified candidate client:
- It shifts the state window, as in Phase 7A, and checks each init state's SHA-1 against the original suite.
- It computes `rollout_step` by counting real `env.step` calls since `set_init_state`, minus 10. It asserts this equals 5 × the request index.
- It requires a matching echo on every response.

**Pre-rollout GPU checks** (synthetic observation, real checkpoint) [M]:
- the sampler-side noise is identical under all 4 conditions;
- different keys give different noise;
- the same key twice gives bit-identical actions;
- **`Policy.infer(noise)` vs the steering sampler with an identity (β = 0) hook: max |Δaction| = 0.0.** The BASELINE path and the steered path are numerically the same sampler.

**Execution.**
- GPU 0 ran repeat 1 on ports 8810–8812; GPU 1 ran repeat 2 on ports 8910–8912.
- Each GPU ran KS4 → LR1 → LR2a, with one in-process server per (task, repeat) and the conditions in the fixed order.
- Wall clock: 14:12:40–15:50:19.
- A smoke test on init state 0 with 15 steps ran after the pre-registration froze. It is not a result.

## 8. Verification that noise pairing actually holds

`tools/analyze_phase7b.py` asserts all of the following. All passed [M].

| Check | Result |
|---|---|
| Independent recomputation of every client- and server-logged fingerprint | **112,012 / 112,012** match (17,258 unique keys) |
| Sampler-side SHA == derived SHA, and sampler path matches the condition | every request |
| Per-episode schedule: rollout_step = 0, 5, …, < rollout_steps, and init_state = 15 + episode | every episode |
| Identical noise at shared (state, step) coordinates, all 6 condition pairs × 6 (task, repeat) | **36/36 cells, 100%** (1,809–2,283 coordinates each) |
| Fingerprints shared between repeat 1 and repeat 2 | **0** |
| Init-state content identical across the 4 conditions and both repeats | yes (SHA-1 map) |
| Exploratory probe (rollout_step 0) vs primary episodes' `noise_verification_hash` | 720/720 match |

Sample fingerprints (first 16 hex digits; identical in all 4 conditions):

| Coordinate | Repeat 1 (seed 100) | Repeat 2 (seed 200) |
|---|---|---|
| LR2a, state 15, step 0 | `4efa91b8661f920b` | `04d11ad2e5fd9694` |
| KS4, state 22, step 100 | `439ed454524deb87` | `6f4794012e03df65` |
| LR1, state 29, step 200 | `342e6319d60a264c` | `13248b6f53ae15b4` |

**Environment finding (KS4)** [M].
- In the KS4 scene, a fixed wooden fixture's pose is re-randomized at every `env.reset()`. It is not part of the MuJoCo state that `set_init_state` restores: re-rendering the same init state gives about 38k different pixels, with identical proprioception.
- We verified that the pose depends **only on the reset count**. The first images were identical after 0, 37, 200 or 520 random env steps between resets.
- Every condition and repeat runs states 15–44 from a fresh `seed(7)` env, in the same reset order. So each state's first observation was identical across all conditions in the primary experiment, and pairing is intact.
- This affects Phases 7A and 7B equally.

## 9. Behavioral success results

Held-out states 15–44; successes out of 30 per repeat. Wilson intervals are descriptive only, because the repeats share states.

| Task | Condition | Rep 1 | Rep 2 | Pooled /60 | Wilson 95% | States 2/1/0 of 2 |
|---|---|---|---|---|---|---|
| LR2a | BASELINE | 13 | 20 | 33 (0.55) | 0.43–0.67 | 11/11/8 |
| LR2a | ORIGINAL_COAST | 18 | 17 | 35 (0.58) | 0.46–0.70 | 11/13/6 |
| LR2a | PURE_SHRINKAGE | 21 | 22 | **43 (0.72)** | 0.59–0.82 | 15/13/2 |
| LR2a | RANDOM_CONTROL | 19 | 21 | 40 (0.67) | 0.54–0.77 | 14/12/4 |
| KS4 | BASELINE | 16 | 21 | 37 (0.62) | 0.49–0.73 | 12/13/5 |
| KS4 | ORIGINAL_COAST | 21 | 15 | 36 (0.60) | 0.47–0.71 | 13/10/7 |
| KS4 | PURE_SHRINKAGE | 16 | 16 | 32 (0.53) | 0.41–0.65 | 11/10/9 |
| KS4 | RANDOM_CONTROL | 15 | 16 | 31 (0.52) | 0.39–0.64 | 7/17/6 |
| LR1 | BASELINE | 17 | 18 | 35 (0.58) | 0.46–0.70 | 11/13/6 |
| LR1 | ORIGINAL_COAST | 22 | 18 | 40 (0.67) | 0.54–0.77 | 13/14/3 |
| LR1 | PURE_SHRINKAGE | 27 | 20 | **47 (0.78)** | 0.66–0.87 | 18/11/1 |
| LR1 | RANDOM_CONTROL | 22 | 22 | 44 (0.73) | 0.61–0.83 | 17/10/3 |

**Cross-checks** [M]:
- the log-derived rate equals the driver `summary.json` for all 24 cells;
- the video frame count equals the env-step count for all 720 episodes;
- every failure is a 520-step timeout.

## 10. COAST vs pure-shrinkage paired analysis (primary)

Paired 2×2 tables over the 30 held-out states. b = COAST-only successes, c = shrinkage-only successes; the McNemar test is exact and two-sided. The pooled row counts both repeats and is **descriptive** (60 pairs over 30 states).

| Task | Table | Both succeed | b | c | Both fail | RD | McNemar p |
|---|---|---|---|---|---|---|---|
| LR2a | Rep 1 | 14 | 4 | 7 | 5 | −0.100 | 0.549 |
| LR2a | Rep 2 | 16 | 1 | 6 | 7 | −0.167 | 0.125 |
| LR2a | Pooled | 30 | 5 | 13 | 12 | **−0.133** | (0.096) |
| KS4 | Rep 1 | 14 | 7 | 2 | 7 | +0.167 | 0.180 |
| KS4 | Rep 2 | 12 | 3 | 4 | 11 | −0.033 | 1.000 |
| KS4 | Pooled | 26 | 10 | 6 | 18 | +0.067 | (0.454) |
| LR1 | Rep 1 | 20 | 2 | 7 | 1 | −0.167 | 0.180 |
| LR1 | Rep 2 | 16 | 2 | 4 | 8 | −0.067 | 0.688 |
| LR1 | Pooled | 36 | 4 | 11 | 9 | −0.117 | (0.118) |

**State-clustered uncertainty (Step 13).** 10,000 state-cluster bootstrap resamples, keeping both repeats of a state together, seed 7070:

| Task | RD_COAST−SHRINK | 95% CI | Same sign in both repeats | Clear (pre-registered) |
|---|---|---|---|---|
| LR2a | −0.133 | **[−0.250, −0.017]** | yes (−, −) | **yes, negative** |
| KS4 | +0.067 | [−0.067, +0.200] | no (+, −) | no |
| LR1 | −0.117 | [−0.233, **0.000**] | yes (−, −) | no (the CI touches 0) |

**Reading** [I]:
- The primary quantity RD_COAST−SHRINK is **not consistently positive**. It is negative in 2 of 3 tasks (both L11/β0.3 tasks) and positive only in KS4, where it is not significant and the two repeats disagree in sign.
- Where the paired data point anywhere, they point toward pure shrinkage performing *at least as well as* COAST.
- Identical rollout lengths (COAST vs SHRINKAGE) were 5/9, 7/11 and 4/10 of 30 per repeat. The trajectories do diverge under the intervention difference; the success outcome simply does not favor COAST.

## 11. COAST vs random paired analysis (secondary)

| Task | Rep 1 b/c (p) | Rep 2 b/c (p) | Pooled RD | 95% CI | Clear |
|---|---|---|---|---|---|
| LR2a | 4/5 (1.0) | 4/8 (0.39) | −0.083 | [−0.233, +0.067] | no |
| KS4 | 8/2 (0.11) | 5/6 (1.0) | +0.083 | [−0.067, +0.217] | no |
| LR1 | 5/5 (1.0) | 4/8 (0.39) | −0.067 | [−0.233, +0.100] | no |

- Cross-task mean RD is −0.022 [−0.117, +0.067].
- Now under paired noise, this reproduces Phase 7A: **no advantage of the learned conceptor over the matched random conceptor.**
- Exploratory [E]: RANDOM vs SHRINKAGE RD = −0.050 / −0.017 / −0.050, none clear.

## 12. COAST vs baseline paired analysis (secondary)

| Task | Rep 1 b/c (p) | Rep 2 b/c (p) | Pooled RD | 95% CI | Clear |
|---|---|---|---|---|---|
| LR2a | 9/4 (0.27) | 5/8 (0.58) | +0.033 | [−0.150, +0.217] | no |
| KS4 | 10/5 (0.30) | 1/7 (0.070) | −0.017 | [−0.183, +0.150] | no |
| LR1 | 8/3 (0.23) | 6/6 (1.0) | +0.083 | [−0.067, +0.233] | no |

- Cross-task mean RD is +0.033 [−0.061, +0.128].
- There is no clear paired advantage over baseline in any task.
- Per-repeat signs flip in LR2a and KS4. One noise realization favors COAST and the other disfavors it. That is exactly the between-realization variability that unpaired single-seed runs cannot separate from a treatment effect.
- Exploratory [E], not pre-registered as a comparison: **PURE_SHRINKAGE vs BASELINE** RD = +0.167 [−0.017, +0.350] (LR2a), −0.083 [−0.267, +0.100] (KS4), **+0.200 [+0.033, +0.367]** (LR1; McNemar p = 0.013 in rep 1 and 0.79 in rep 2).

## 13. Operator decomposition

File: `experiments/phase7b_operator_decomposition.csv`. Computed in float64, in the implemented row form. Two hidden-state sets:
- **fit:** the authors' fit token vectors, every 10th inference step, all denoising steps; 13,000 / 13,200 / 11,200 vectors.
- **held-out:** BASELINE hidden states at the hooked layer on the first observation of states 15–44, both seeds, all 10 denoising steps; 6,000 vectors.

Values are shown as fit / held-out.

| Metric | LR2a (L11 β0.3) | KS4 (L5 β0.1) | LR1 (L11 β0.3) |
|---|---|---|---|
| 1. ‖δ_coast‖/‖h‖ | 0.2885 / 0.2884 | 0.0985 / 0.0985 | 0.2868 / 0.2861 |
| 2. ‖δ_shrink‖/‖h‖ | 0.300 | 0.100 | 0.300 |
| 3. ‖r_C‖/‖h‖ | 0.0186 / 0.0185 | 0.0034 / 0.0034 | 0.0229 / 0.0242 |
| 4. ‖r_C‖/‖δ_coast‖ | **0.065 / 0.064** | **0.035 / 0.035** | **0.080 / 0.085** |
| 5. cos(δ_coast, δ_shrink) | **0.99874 / 0.99878** | **0.99946 / 0.99947** | **0.99790 / 0.99765** |
| 6. cos(δ_coast, h) | −0.99874 / −0.99878 | −0.99946 / −0.99947 | −0.99790 / −0.99765 |
| 7. cos(r_C, h) | 0.640 / 0.647 | 0.439 / 0.443 | 0.604 / 0.604 |
| 8. hᵀCh/‖h‖² | 0.0395 / 0.0398 | 0.0151 / 0.0153 | 0.0459 / 0.0486 |
| 9. quota tr(C)/d | 0.0126 | 0.0082 | 0.0168 |
| 10. Relative decomposition error | 1.4e-15 / 1.5e-15 | 7.4e-15 / 6.0e-15 | 1.5e-15 / 1.3e-15 |
| ‖C − Cᵀ‖_F/‖C‖_F | 0.458 | 0.347 | 0.441 |
| Overlap(C_s, C_f) | 0.951 | 0.984 | 0.947 |

Notes:
- Metrics 5 and 6 are exact negatives of each other, because δ_shrink = −βh is parallel to −h. Both are listed as the brief asks.
- **Realized bf16 variant**, with M cast to bf16 as the hook does: ‖r_C‖/‖δ_coast‖ = 0.068 / 0.052 / 0.082; cos(δ_coast, δ_shrink) = 0.9987 / 0.9993 / 0.9979. The picture is the same.
- **RANDOM_CONTROL**, for comparison: ‖r‖/‖δ‖ ≈ 0.26, cos(δ, δ_shrink) ≈ 0.988, hᵀCh ≈ 0.175. The random conceptor is a larger departure from pure shrinkage than the learned one.
- The held-out and fit sets agree to within about 0.005 on every metric.

**Reading** [M/I]:
- ‖δ_coast‖ is **95–99% of ‖δ_shrink‖**, and δ_coast is **within 0.07 rad of δ_shrink in direction**.
- r_C has a large component along +h (cos 0.44–0.65): 44–65% of its norm projects onto +h. That part only **undoes some of the contraction**: the effective shrink along h is β(1 − hᵀCh/‖h‖²). The part of r_C orthogonal to h is about 1.4–1.9% of ‖h‖ for the β 0.3 tasks and 0.3% for KS4.
- The pre-registered supporting-geometry thresholds are cos ≥ 0.99 and ratio ≤ 0.2. They are met in every task, on both hidden-state sets.

**Action level** [E] (`phase7b_action_probe.csv`): identical observation and identical noise, i.e. the first request of every primary episode, 60 per task.

| Task | ‖a_coast − a_base‖ | ‖a_shrink − a_base‖ | ‖a_rand − a_base‖ | ‖a_coast − a_shrink‖ | cos(Δa_coast, Δa_shrink) | ‖a_coast − a_shrink‖/‖a_shrink − a_base‖ |
|---|---|---|---|---|---|---|
| LR2a | 0.484 | 0.519 | 0.409 | 0.049 | 0.997 | 0.095 |
| KS4 | 0.163 | 0.170 | 0.139 | 0.010 | 0.999 | 0.058 |
| LR1 | 0.479 | 0.540 | 0.409 | 0.116 | 0.980 | 0.217 |

- Mean ‖a_base‖ is about 3.2–3.5 (a 10 × 7 chunk).
- The action-level effect of COAST is the shrinkage effect, slightly attenuated, in almost exactly the same direction.
- LR1 has the largest conceptor-specific action residual (22%).

## 14. Cross-task interpretation

- **Is RD_COAST−SHRINK consistently positive? No.** The pattern is **mixed signs, leaning negative**:
  - −0.133 (LR2a), +0.067 (KS4), −0.117 (LR1);
  - mean −0.061, with CI [−0.133, +0.006].
- **Pre-registered case: C.** It is triggered by one task with a clear *negative* difference.
- **Examination required for Case C** (n = 3, descriptive only). The two tasks with negative RD (LR2a, LR1) share L11 and β = 0.3. Compared with KS4 (L5, β 0.1), they also have:
  - larger ‖r_C‖/‖h‖ (0.019–0.024 vs 0.003);
  - larger hᵀCh (0.040–0.049 vs 0.015);
  - larger quota (0.013–0.017 vs 0.008);
  - lower s/f overlap (0.95 vs 0.98).

  Every candidate predictor moves together with layer and β. With three tasks and only two distinct configurations, **these factors cannot be separated**.
  - The factor most directly tied to the mechanism is β. At β = 0.3, pure shrinkage itself was associated with the largest gains over baseline (exploratory: +0.167 and +0.200). There, COAST's learned residual mainly makes the contraction slightly *weaker*: 0.288 vs 0.300 of ‖h‖.
  - A reading consistent with these data [I]: at these configurations, the behavioral effect comes from contraction, and the conceptor term dilutes it a little. This is a hypothesis, not a demonstrated mechanism.
- **Statistical caution.** No per-repeat McNemar test is significant anywhere, and 9 task × comparison cells were tested. The single "clear" cell (LR2a) sits at about the nominal 5% level. We therefore do **not** claim that the conceptor term reliably *hurts*. The robust statement is that it shows **no paired advantage over contraction in any task**.

## 15. Which Phase 7A conclusions survive

- **No held-out advantage of the paper Global configuration over the matched random conceptor.** This survives, now under paired noise: RD −0.083 / +0.083 / −0.067, none clear.
- **No clear held-out advantage over baseline in any single task.** This survives: RD +0.033 / −0.017 / +0.083, none clear.
- **Contraction-dominated operators.** This survives and is strengthened by the explicit decomposition: ‖r_C‖/‖δ_coast‖ ≤ 0.085 and cos(δ_coast, δ_shrink) ≥ 0.9976, on both fit and held-out hidden states. It is also confirmed at the action level.
- **Case-A-style "general limitation" of the conceptor-specific component.** This survives. Across 4 studied tasks (KS3 from Phases 2A/2B, plus 3 here), no paired or unpaired analysis has found a positive, conceptor-specific behavioral contribution.

## 16. Which Phase 7A conclusions change

- **The small nonspecific pooled gain over baseline** (Phase 7A: Global +0.094, CMH p = 0.086) is **not reproduced for COAST** under paired noise: mean RD +0.033 [−0.061, +0.128]. Some of that Phase 7A difference was probably unpaired noise-realization variation. Phase 7A's own baselines ranged from 25 to 36/60, while Phase 7B's were 33–37/60.
- **What carries the nonspecific effect.** If anything carries it, the Phase 7B data point to **the contraction itself**, not the conceptor. Pure shrinkage was the numerically best condition in LR2a and LR1, and was +0.20 over baseline in LR1 (exploratory, CI [+0.033, +0.367]). This is exploratory and was not pre-registered.
- **New: the candidate conceptor is strongly asymmetric** (0.35–0.46 relative Frobenius). Phase 7A did not report this. Because the hook uses h·Mᵀ, the column-form operator is C itself.
- **New: KS4 fixture re-randomization** (§8). It does not affect pairing or the Phase 7A conclusions. It does mean that "same init state" in KS4 does not by itself pin down the full visual scene outside a fixed reset sequence.

## 17. Limitations

- **Scope.** 3 tasks, one checkpoint, one configuration per task (the paper's), two layer/β settings in total. Layer and β are confounded with task in the Case C examination.
- **Power.** 30 states × 2 repeats per task. Discordant pairs number only 6–15 per repeat (15–26 pooled) per task and comparison, so paired RDs smaller than about 0.15 cannot be resolved reliably. "No clear difference" is not evidence of no difference.
- **Two noise realizations.** Paired noise removes between-condition noise differences *within* a realization, but outcomes still vary between realizations. Per-repeat RDs flipped sign in several cells (e.g. KS4: +0.167 / −0.033).
- **Multiplicity.** 9 task × comparison cells with no correction. The LR2a "clear" call is at about the nominal level.
- **Closed-loop amplification.** After the first divergence, later requests reuse the same noise on different observations. Episode outcomes therefore include chaotic amplification of the small intervention difference, not just its direct effect. The action-level probe compares only first requests, at identical observations.
- **Candidate pipeline, not the paper's final implementation** (author clarification). The results may differ under the unpublished branch, other builders, or other checkpoints.
- **Operator metrics** use baseline-trajectory hidden states; states visited under steering may differ slightly. The held-out probe set covers only rollout_step 0.
- **One GPU per repeat** (GPU 0 = repeat 1, GPU 1 = repeat 2), so GPU and repeat are confounded. Phase 1D found closed-loop determinism across processes, and the repeat totals are similar (rep 1: 227, rep 2: 226 of 360).

## 18. Phase 8 decision

**Pre-registered outcome: Case C.** Per the brief, Phase 8 should therefore investigate **geometry-aware or task-adaptive steering**.

Constraints the Phase 7B evidence places on Phase 8 (recommendations only; nothing implemented):
1. **Pure shrinkage must be a mandatory control**, at matched effective contraction ‖δ‖/‖h‖. Any Phase 8 method must beat it, not only baseline and random, on disjoint held-out states, under deterministic paired noise, with ≥ 2 noise repeats and state-clustered inference.
2. **The Case C trigger was a negative difference.** So Phase 8 should not assume the current learned term is a useful signal to amplify. The geometric reason is that r_C is small (≤ 8.5% of δ_coast) and half aligned with h. Any geometry-aware method has to create a component that is substantially **orthogonal to h** before it can do anything contraction cannot.
3. **Separate the contraction effect first.** Exploratory data suggest contraction alone may matter at L11/β0.3. A pre-registered test of that, at several β values on held-out states, would tell Phase 8 whether it should remove the contraction (the Case A path) or build on it.
4. **Plan power in advance.** Resolving paired RDs of about 0.1 needs substantially more states or tasks than 30 × 2.

Phase 8 was not started. No new steering method, equation, β schedule or conceptor variant was implemented in Phase 7B.

## Appendix — reproduction

Scratch root: `/tmp/claude-1000/-home-stanley-Stanley-ws-COAST-research-COAST/79f17b80-c3cb-44a7-93eb-68f24af75be0/scratchpad/phase7b` (not in git). It contains:
- `runs/`, `logs/`, `probe/`, `odh/` (hard-linked Phase 7A NPZs);
- `run_queue.sh`;
- `PREREG_FROZEN.txt`, `RECORDS_FROZEN.txt`.

Commands:
```
# pre-rollout checks (candidate PYTHONPATH, TORCH_COMPILE_DISABLE=1)
python research/reproduction/tools/phase7b_gpu_checks.py <scratch>/logs/gpu_checks.json
# primary run (one per GPU)
GPU=0 SEED=100 REP=1 PORT0=8810 bash <scratch>/run_queue.sh
GPU=1 SEED=200 REP=2 PORT0=8910 bash <scratch>/run_queue.sh
# primary analysis (repo root, CPU)
.venv/bin/python research/reproduction/tools/analyze_phase7b.py <scratch>/runs <scratch>/logs
# exploratory
(libero venv, cwd candidate examples/libero_env) python research/reproduction/tools/phase7b_probe_observations.py <task_id> 15 30 <scratch>/probe/<KEY>_obs.npz
(root venv, candidate PYTHONPATH) python research/reproduction/tools/phase7b_action_probe.py <scratch>/probe <scratch>/probe
.venv/bin/python research/reproduction/tools/phase7b_operator_decomposition.py <scratch>/odh research/reproduction/experiments/phase7b_operator_decomposition.csv <scratch>/probe
.venv/bin/python research/reproduction/tools/analyze_phase7b_exploratory.py <scratch>/probe
```
