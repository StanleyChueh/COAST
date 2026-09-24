# Phase 6A — Reproduction-Gap Analysis (KS3, π0.5, LIBERO-10)

Question: **why does our reproduction not match the paper's KS3 result, and is the gap best explained by**
- **(A)** missing reproduction details,
- **(B)** differences in the released implementation, or
- **(C)** a possible limitation of the COAST method?

Analysis only. No rollouts, no code changes, no files touched under `src/`, `scripts/` or `experiments/`. Two new calculations use existing per-episode records (§4.2):
- a window scan over Phase 4A's all-50-state data;
- a binomial order-statistic check.

| | |
|---|---|
| Paper | *Contrastive Conceptor Activation Steering (COAST)*, arXiv:2605.17144 v1 |
| Released code | `github.com/COAST-VLA/COAST` `main` @ `2afa10ee256a3b3edfeb56500fea166a0837f119` (base of all our work) |
| Our tree | branch `exp/reproduction-gap-analysis` @ `978de727b4ae1708fb8dc05286f897569dcbff80` |
| Sources | Phases 0–5C records in `research/reproduction/`; `README.md`; `experiments/libero/`; `src/openpi/serving/` |

Evidence tags: **[M]** measured by us, **[P]** stated in the paper, **[R]** repository, **[I]** our inference.

## 1. Executive summary

**Measured [M]:**
- On KS3 the paper reports 28/30 for global COAST.
- Across three independent sessions, the paper's own configuration and our oracle-selected configuration scored **19, 20 and 21 of 30** on states 15–44. Pooled, that is 60/90 = 0.67.
- Our unsteered runs on the same states scored 17, 19 and 21 of 30 (pooled 57/90 = 0.63).
- Positive-only scored 17/30 (paper 24/30). Per-step at α = 1.0 scored 15/30 (paper 26/30 at α = 10).
- Our unsteered rate on the fit states (8/15 = 0.53) equals the paper's Base.
- **A reproduction gap is observed.**

**Interpretation, weighing A, B and C:**

- **Ruled out by measurement:**
  - The fixed configuration: oracle selection gave 21 vs 19 out of 30.
  - A missing oracle procedure: it was run with the released tool (Phase 5C).
- **Bounded by measurement, so unlikely to explain the gap alone:**
  - *Choice of test states.* In Phase 4A's all-50-state run of the paper's configuration, no contiguous 30-state window exceeds 22/30.
  - *Builder V0 vs V3.* With paired noise the two gave identical outcomes on 15/15 states.
  - *Checkpoint and environment.* Our unsteered fit-state rate matches the paper's Base.
  - *Single-run noise.* The chance of 28/30 or better at p ≈ 0.67 is 0.0007.
- **Not bounded by our data, so the most likely sources:**
  1. **How the Table 4 number was produced.** The paper does not say whether "oracle" means the best of the grid scored on the same 30 rollouts it reports, or a fresh test of one configuration selected on the fit rollouts. If it is the first, then given our measured true rate of about 0.67–0.70, the best of 60–240 configurations is expected at 25.7–27.4/30. 28/30 is then unremarkable (P ≈ 0.04–0.40), and the 28 > 26 > 24 ordering across strategies fits order statistics too [I].
  2. **Which code produced Table 4.** The paper's per-step α = 10 is not expressible in the released code, so at least part of the paper's pipeline is not the release [M/R].
- **(C) Method limitation: a hypothesis supported by our data, not a conclusion.**
  - With the released implementation, every steering condition we tested (COAST, pure shrinkage, a spectrum-matched random conceptor, 132 oracle configurations) was indistinguishable from no steering on this task.
  - At the paper's configuration, the operator acts as a uniform 0.9·h shrink. The paper's V3 builder does the same offline.
  - This is **one task, one checkpoint**, under our protocol. It cannot be separated from (A) until the paper's protocol is known.

**Recommendation: Option B, contact the authors** (questions in §5; not sent). The gap is not caused by anything we can still vary cheaply. The two leading explanations can only be resolved by the authors. Starting improvement work now (C) would build on a baseline claim whose protocol we cannot pin down.

## 2. Experiment history

Each row keeps the measured result separate from what we take it to mean.

| Phase | Question | Measured result | Meaning (interpretation) |
|---|---|---|---|
| 0 | Does the environment run? | Unsteered LIBERO smoke test runs; stack recorded (LIBERO `d63b117`, robosuite 1.4.0, MuJoCo 3.2.3) | Environment usable |
| 1A | Collect fit activations | KS3, states 0–14: **8 success / 7 failure**, schema `(10,4,10,1024)` | Same counts and schema as paper Table 21 / A.7; per-episode outcomes differ (unseeded noise) |
| 1B | Do repository (V0) and paper-text (V3) conceptors differ? | Offline: C differs by 53–123% relative Frobenius; quota tr(C)/d at L5 α 0.5 is V0 0.0185, V3 0.0103 | The builders differ numerically (token pooling, centering, Boolean AND) |
| 1C | Behavioural comparison, unpaired noise (states 15–29) | Baseline 8/15, V0 10/15, V3 10/15 | Too noisy to separate |
| 1D | Same, with paired noise | Baseline 13/15, V0 11/15, V3 11/15; **V0 = V3 on 15/15 states**; baseline alone varies 7–13/15 by noise seed | Builder difference has no behavioural effect at L5 α0.5 β0.1; noise dominates 15-episode comparisons |
| 1E | What does the intervention do to h? | cos(Δh, h) = −0.9996; ‖Δh‖/‖h‖ = 0.0994; hᵀCh/‖h‖² ≈ 0.006 | At the paper's configuration, COAST ≈ uniform 0.9·h shrink |
| 2A | COAST vs pure shrinkage 0.9·I | 11/15 vs 10/15; same outcome on 14/15 states | Shrinkage reproduces COAST's outcomes |
| 2B | COAST vs spectrum-matched random conceptor | 11/15 vs 11/15; same outcome on 15/15 states | The conceptor's direction does not matter behaviourally here |
| 3A | Held-out states 30–44 | Baseline 9, shrinkage 10, COAST 11, random 11 (of 15) | No demonstrable gain on unseen states (McNemar p = 0.5) |
| 4A | Sanity check on all 50 states | β 0.1: baseline 30/50, COAST 32/50, shrinkage 31/50; states 15–44: baseline 21/30, COAST 20/30 | COAST ≈ baseline ≈ shrinkage across the whole state space |
| 4B | Did we reproduce the paper setting? | Model, checkpoint, configuration and hook match. Test states, seeds, code version, Base definition and builder are unknown. Paper Base = fit-set rate for 10/10 tasks | Faithful on the model side; the evaluation protocol is unverifiable |
| **5A** | **Paper's fixed KS3 global configuration (L5 α0.5 β0.1), 30 test episodes (states 15–44)** | **COAST 19/30, baseline 19/30; paper 28/30** (Fisher p = 0.010) | **Reproduction gap observed** with the paper's stated configuration |
| 5B | Can the paper's oracle procedure be run with the released code? | Yes for global and positive-only (CLI flags only). Per-step only at α = 1.0 | Oracle reproduction feasible except per-step α |
| **5C** | **Paper-style oracle: 15 fit → grid → per-strategy argmax → 30 test** | Global L11 α0.5 β0.3: **21/30**; positive-only L0 α2 β0.3: **17/30**; per-step L5 α1 β0.3: **15/30**; baseline **17/30**. Fit maxima fall 0.17–0.23 on test | **Reproduction gap persists** under oracle selection; no selected configuration is distinguishable from baseline |
| 5C-final | Interpretation | — | Gap not explained by configuration or selection; recommend author questions |

## 3. Reproduction-gap table

Confidence means how sure we are that our reproduction matches the paper on that component.
- **High:** verified to match.
- **Medium:** a known difference exists, or the item is unverifiable but constrained by indirect evidence.
- **Low:** unknown.

"Possible impact" is judged against the 28/30 vs about 20/30 gap.

### 3.1 Checkpoint

| Component | Paper | Our reproduction | Confidence | Possible impact |
|---|---|---|---|---|
| Model | π0.5, `pi05_libero` [P] | π0.5 `pi05_libero`, PyTorch path [M] | High | none |
| Checkpoint | `openpi-libero-2000`, step 2,000 of 30,000 [P, A.8.3/A.11.2] | HF `brandonyang/openpi-libero-2000`, revision `aaeeabc72f8a…` (dated 2026-04-09, before the paper) [M] | High (name, step, repo) | none known |
| Training hyper-parameters | A.11.2 list [P] | Identical to the `pi05_libero` config (all values) [R] | High | none |
| Weight hash | **not published** [P] | Cannot be verified against the authors' weights | Medium | **Bounded:** our unsteered fit-state rate is 8/15 = 0.53, exactly the paper's KS3 Base; the authors' released fit dataset also has 8/7 [M]. A different checkpoint would be expected to shift the base policy. Low |
| JAX→PyTorch conversion / bf16 | bf16 inference [P]; conversion not described | Auto-converted on first serve; bf16 [M] | Medium | Low (base rate matches) |

### 3.2 Dataset and task

| Component | Paper | Our reproduction | Confidence | Possible impact |
|---|---|---|---|---|
| Suite / task | LIBERO-10; KS3 = "turn on the stove and put the moka pot on it" [P] | `libero_10` task 2, same name [M] | High | none |
| LIBERO version | **not stated** [P] | Fork `Robot-VLA/LIBERO` @ `d63b117` (repository submodule) [R] | Low | Low. The fork is the repository's own pin, and the base rate matches |
| Init-state source | "built-in per-task initial state distribution", "sequential episode seeds" [P] | 50 built-in init states; episode k → state `(seed + k) % 50` (post-PR #48) [R] | Medium | See the split row |
| Split | 15 fit + 30 test, "different seeds … initial conditions"; **no IDs** [P] | Fit 0–14 (`--seed 0`); test 15–44 (`--seed 15`) [M] | **Low** | **Bounded:** Phase 4A all-50 run of the paper's configuration: the best contiguous 30-state window is 22/30 (§4.2) |
| Seeds | **not given** [P] | `--seed` sets the init-state offset, env seed and `np.random` [R] | **Low** | Covered by the split and noise rows |
| Pre/post PR #48 seeding | unknown (released fit dataset dated 2026-04-09; PR #48 on 2026-04-23) [R] | post-#48 [M] | **Low** | Pre-#48, every run starts at state 0, so the test would overlap the fit states. Bounded: window 0–29 = 19/30 for COAST (§4.2) |
| Fit set | 15 rollouts, KS3 8 S / 7 F [P] | 15 rollouts, 8 S / 7 F [M] | High (counts) | none (episodes differ only by noise) |

### 3.3 Evaluation protocol

| Component | Paper | Our reproduction | Confidence | Possible impact |
|---|---|---|---|---|
| Fit episodes | 15 per task, used for the conceptors **and** for hyper-parameter selection [P] | 15 (states 0–14), same double use [M] | High | none |
| Test episodes | 30 per task per condition [P] | 30 (states 15–44), one run per condition [M] | Medium (states unknown) | See the split row |
| Oracle selection | "selected by maximizing success rate on the fitting rollouts" [P, A.8.5]; "oracle = best rollout success rate" [P, A.10.2–3] | Steered re-rollouts on the fit states; per-strategy argmax; frozen before test [M] | **Low** (the paper's wording is ambiguous about which rollouts the reported oracle number comes from) | **High, unbounded:** if the reported value is a best-of-grid score on the reporting rollouts, it carries winner's-curse inflation (§4.2) |
| Runs per cell | **not stated** [P] | 1 [M] | Low | Medium: best-of-k reporting would inflate |
| Base definition | Table 1 Base equals the fit-set rate for 10/10 tasks (inferred) [P/I] | Separate unsteered test runs: 17, 19, 21 of 30 [M] | Low | Affects the **claimed gain**, not the steered 28/30 |
| Success label | final-step `is_success` [P] | first `done` = `_check_success()` [R] | High | none (LIBERO ends at success) |
| Episode length | not stated | 520 steps (libero_10 default), replan every 5 steps [R] | Medium | Low |

### 3.4 Conceptor construction

| Component | Paper (text, "V3") | Repository ("V0", used in all runs) | Confidence | Possible impact |
|---|---|---|---|---|
| Token handling | mean-pool the 10 action tokens → 1 vector per (call, denoise step) [P, §3.2, A.9.1] | every (call, step, **token**) is a row; no pooling [R] | Medium (known difference) | **Bounded:** V0 = V3 on 15/15 states with paired noise (1D); both ≈ 0.9·h shrink offline (4B §5.4). Untested at other configurations |
| Centering | mean-centered R = X̃ᵀX̃/N [P] | uncentered R = XᵀX/N [R] | Medium | Same bound |
| Boolean AND | (A⁻¹ + B⁻¹ − I)⁻¹ via pseudoinverse (canonical, symmetric) [P] | A·inv(A + B − AB)·B (non-canonical, not symmetric) [R] | Medium | Same bound; the offline difference is 3–23% |
| Class threshold | ≥ 3 episodes per class [P] | ≥ 2 [R] | High (not triggered: 8/7) | none |
| Aperture formula | C = R(R + α⁻²I)⁻¹, float64 → float32 [P] | same [R] | High | none |
| Released NPZ | `brandonyang/libero-conceptors` | HTTP 401 for anonymous access [R/W] | Low | Would settle which builder was used |
| Quota diagnostics | L5 quota "≈ 1% of d"; overlap at L11/α 10 = 0.670 [P] | V3 L5 α0.5 quota 0.0103 (matches); V0 0.0185; neither reproduces the 0.670 overlap (V3 0.937) [M] | Medium | Indicates a builder or data difference somewhere, but one task vs the paper's 10-task mean |

### 3.5 Hyper-parameter search

| Component | Paper | Our reproduction | Confidence | Possible impact |
|---|---|---|---|---|
| Grid | layer {0,5,11,17} × α {0.1,0.5,1,2,10} × β {0.1,0.3,0.5} × 4 strategies = 240 [P, Table 14] | Global 60, positive-only 60, per-step 12, linear 12/20 (interrupted) [M] | High (global, positive-only) | none for these two |
| Per-step α | KS3 per-step oracle α = 10 [P] | fixed at 1.0 in the builder, hook and sweep tool [R] | **Low (known gap)** | Per-step column only; needs a source change |
| Selection scope | one oracle per strategy (Table 4 columns) [P] | same, via separate calls [M]. The released tool by default takes one argmax across all strategies, including `random_matched` [R] | High | none (handled) |
| Tie rule | not stated | first in iteration order; tie sets reported [M] | Medium | Low (global had a unique max) |
| Released tuned configs | README says they are committed | `experiments/libero/best_configs.json` is an empty placeholder [R] | Low | Would show which configurations the authors' own sweep produced |

### 3.6 Policy randomness

| Component | Paper | Our reproduction | Confidence | Possible impact |
|---|---|---|---|---|
| Flow-matching noise | **not described** [P] | Released default: unseeded `sample_noise` (Phases 5A/5C). Paired seeded noise only in our instrumented Phases 1D–4A [R/M] | Low | **Bounded for one run:** P(≥ 28/30 \| p = 0.67) = 0.0007. Large only combined with best-of-k or best-of-grid reporting |
| Run-to-run spread | not reported | Noise seed alone moves KS3 between 7/15 and 13/15 on the same states (1D). Fit baselines 4–11/15 (5C, over-dispersed, p = 0.035) [M] | High (ours) | Makes N = 15 selection unreliable. Explains our fit-to-test drop |

### 3.7 Software versions

| Component | Paper | Our reproduction | Confidence | Possible impact |
|---|---|---|---|---|
| LIBERO | not stated | `Robot-VLA/LIBERO` @ `d63b117` | Low | Low (base rate matches) |
| robosuite | not stated | 1.4.0 (PyPI, pinned by `examples/libero_env`) | Low | Low |
| MuJoCo | not stated | 3.2.3 | Low | Low |
| COAST / OpenPI code | not stated (commit unknown) | `COAST-VLA/COAST` @ `2afa10e` plus research instrumentation behind opt-in flags (7 files; default global path and builder unmodified, 4B §3) | **Low** | **High, unbounded:** the per-step α axis shows the paper used code not in the release |
| PyTorch / JAX | not stated | torch 2.7.1+cu126, JAX 0.5.3, transformers 4.53.2 (+ openpi patch) | Low | Low |
| Hardware | B200 (latency table) | RTX 4090 | Medium | Low (bf16 numerics, trajectory chaos) |

## 4. Unknowns ranked by expected impact

### 4.1 Ranking

Ranked by how much of the observed gap each unknown could explain *given the measurements we already have*. An unknown ranks high only if our data do not already bound it.

| Rank | Unknown | Could it explain 28/30 vs ≈ 20/30? | Evidence that bounds it | Category |
|---|---|---|---|---|
| **1** | **How the Table 4 oracle number was produced**: fresh 30-episode test of a configuration chosen on the fit rollouts, or best-of-grid on the reporting rollouts; runs per cell | **Yes, fully.** With our measured true rate of 0.67–0.70, the maximum of 60–240 configurations on the same 30 rollouts is expected at **25.7–27.4**, and P(≥ 28) = 0.04–0.40 (§4.2). Our own Phase 5C fit sweep shows this inflation directly: best-of-60 = 0.87 on fit, 0.70 on test | Not bounded. The paper's wording ("oracle = best rollout success rate", "92% of oracle") admits both readings | A (missing detail) |
| **2** | **Code version and pipeline behind Table 4**: commit; whether the paper's hook, builder or per-step code differs from the release | **Possibly.** If the paper's operator is not ≈ (1−β) shrink, steering could have a real effect there that the release does not have | Partly bounded: V0 = V3 behaviourally (1D) and V3 ≈ shrink offline (4B §5.4). **Not bounded:** code we have not seen. Per-step α = 10 proves some unreleased code exists | B (implementation) |
| **3** | **Policy noise and repeated runs**: seeding, number of runs, whether the best run is reported | Only together with rank 1 or best-of-k | Single run: P(≥ 28/30 \| p = 0.67) = 0.0007. Best of 12 independent runs: P ≈ 0.009 | A |
| **4** | **Test state IDs and seeds** (split) | **Unlikely alone** | Phase 4A all-50 run at the paper's configuration: every contiguous 30-state window gives 16–22/30; window 0–29 = 19, 15–44 = 20; **none ≥ 28**. A non-contiguous hand-picked set is not consistent with "sequential episode seeds" | A |
| **5** | **V0 vs V3 builder** | **Unlikely at the tested configuration** | Identical outcomes on 15/15 paired states (1D); both ≈ shrink offline. Untested at L11/β 0.3 etc. | B |
| **6** | **Exact checkpoint weights** | **Unlikely** | Unsteered fit rate 8/15 = paper Base 0.53; the authors' released fit data has the same 8/7 split | B |
| **7** | **Base definition** | **No** for the steered 28/30; **yes** for the claimed +0.40 gain | Paper Base = fit-set rate on 10/10 tasks (4B). Our test baselines are 0.57–0.70 | A |
| **8** | **LIBERO / robosuite / MuJoCo versions** | **Unlikely** | Same bound as rank 6 | A |
| — | **Per-step α = 10** | Per-step column only | Confirmed difference; needs a source change | B |

### 4.2 The two new calculations (no rollouts)

**Window scan** over `experiments/phase4a_episode_results.csv` (KS3, all 50 states, one paired-noise run). The table gives successes in each contiguous 30-state window, cyclic mod 50:

| Condition | All 50 | Window 0–29 | Window 15–44 | Best window | Worst window | Windows ≥ 28 |
|---|---|---|---|---|---|---|
| Baseline | 30/50 | 21 | 21 | 22 | 13 | 0 |
| COAST L5 α0.5 β0.1 (paper configuration) | 32/50 | 19 | 20 | 22 | 16 | 0 |
| COAST β 0.3 | 27/50 | 14 | 16 | 19 | 13 | 0 |
| Shrinkage 0.9·I | 31/50 | 17 | 21 | 22 | 16 | 0 |

Caveat: this is one noise realisation. Per-state outcomes are stochastic (1D: 11/15 states flip with the noise seed), so the scan bounds the *expected* window rate (≈ 0.64 ± 0.09), not every possible draw.

**Order-statistic check.** Assume the reported number is the maximum of m configurations, each scored once on the same 30 rollouts, with true rate p. Treating configurations as independent overstates m, because real configurations are correlated.

| True p | m = 1 | m = 12 | m = 60 | m = 240 |
|---|---|---|---|---|
| 0.63 | E[max] 18.9, P(≥ 28) 0.000 | 23.1, 0.002 | 24.7, 0.010 | 25.8, 0.038 |
| 0.67 | 20.1, 0.001 | 24.1, 0.009 | 25.7, 0.043 | 26.7, 0.162 |
| 0.70 | 21.0, 0.002 | 24.9, 0.025 | 26.4, 0.119 | 27.4, 0.398 |

Reading:
- A **single fresh test** at our measured rate essentially cannot produce 28/30.
- A **best-of-grid on the reporting rollouts** can.

This is a compatibility result, not evidence that the paper did this. It identifies the question most worth asking.

### 4.3 What (C), a method limitation, would mean, and why we cannot conclude it yet

Supporting a limitation *in our setup* [M]:
- COAST at the paper's configuration ≈ 0.9·h shrink (1E), ≈ pure shrinkage (2A) and ≈ a random conceptor with the same spectrum (2B).
- It is not distinguishable from baseline on 15–44, 30–44 or all 50 states (3A, 4A, 5A).
- None of 132 oracle-selected configurations beats baseline on test (5C).
- Positive-only fit scores vary *less* than binomial noise (χ² = 42.3 on 59 df), so there is no detectable configuration effect at all.

Why this is not yet a conclusion:
- It is one task and one checkpoint.
- The operator analysis applies to the released builder and, offline, to V3. It cannot apply to unreleased code (rank 2).
- If rank 1 holds, the paper's 28/30 would not be evidence of a steering effect in the first place. The question would then become "does COAST help on held-out tests?", which our data answer only for KS3.

## 5. Author questions (draft — **do not send**)

1. **Oracle number.** "For Table 4 (KS3, π0.5), is 28/30 the success of one configuration chosen on the 15 fitting rollouts and then evaluated once on 30 separate test rollouts? Or is it the best success rate across the configuration grid measured on the same 30 rollouts that are reported? How many evaluation runs does each cell represent?"
2. **Test states and seeds.** "Which LIBERO init-state indices and seeds were used for the 15 fitting and 30 test rollouts on KS3? Were the experiments run before or after the change to `--seed` semantics in PR #48 (2026-04-23)?"
3. **Code version.** "Which commit or branch of COAST-VLA/COAST produced Tables 1 and 4? The released code fixes per-step α at 1.0, but Table 4 lists per-step α = 10 for KS3. Is that code available?"
4. **Conceptor builder.** "Were the Table 4 conceptors built with the released `compute_conceptors.py` (token-flattened, uncentered, non-symmetric AND)? Or with the pipeline in App. A.9.1 (token-pooled, centered, pseudoinverse AND)? Could `brandonyang/libero-conceptors` be made public? It currently returns HTTP 401."
5. **Checkpoint.** "Were the results produced with `brandonyang/openpi-libero-2000` at revision `aaeeabc72f8a…`, through the PyTorch conversion in the repository? Can you share a weight hash?"
6. **Policy noise.** "Was the flow-matching sampling noise seeded during evaluation? If so, how were seeds assigned across configurations and episodes?"
7. **Base column.** "In Table 1, is the π0.5 LIBERO 'Base' value the 15 fitting rollouts, or a separate unsteered test run on the 30 test states?"
8. **Environment.** "Which LIBERO, robosuite and MuJoCo versions were used for the LIBERO rollouts?"

Suggested context to include: our measured numbers (§2, rows 5A and 5C), stated as "reproduction gap observed under a documented interpretation of the protocol". Questions 1–4 are the priority; 5–8 can be one combined question if brevity matters.

## 6. Recommended next step

| Option | Assessment |
|---|---|
| **A. Continue reproduction** | Low expected value now. What we can still vary ourselves (test window, builder, configuration, selection) is already measured or bounded (§4). Per-step α = 10 needs a source change and affects one column. Further replicates narrow our error bars but cannot move 0.67 to 0.93. The top two unknowns cannot be resolved by running experiments |
| **B. Contact the authors** | **Recommended.** Released code followed ✓. Oracle procedure followed ✓. Result differs significantly: 5C global p = 0.042; pooled over our three global runs p = 0.004 ✓. The leading explanations (ranks 1–2) are answerable only by the authors |
| **C. Start COAST improvement** | Premature. The target claim (28/30, +0.40) is not pinned to a known protocol. In our setup the method's effect is indistinguishable from zero and from shrinkage, so an "improvement" would have no agreed baseline to improve on or to compare with the paper |

**Why B:**
- The measurements already rule out or bound every explanation we can test ourselves: configuration, selection, state window, builder, checkpoint and single-run noise.
- What remains is how the paper's number was produced and which code produced it.

**Evidence:**
- Phases 5A and 5C: the gap under both the stated configuration and the oracle procedure.
- Phase 4A window scan: no contiguous 30-state window reaches more than 22/30.
- Phase 1D: V0 = V3 behaviourally.
- Fit-rate match with the paper's Base.
- The order-statistic check: the only scenario compatible in magnitude is best-of-grid reporting.

**Remaining uncertainty:**
- Everything here is one task (KS3) and one checkpoint.
- Our test numbers are single unpaired runs (SD ≈ 0.09).
- States 15–44 were used across several phases.
- Rank 1 is a compatibility argument, not a finding.
- Unreleased code (rank 2) could contain a real difference we cannot see.

**Offline work that fits alongside B** (optional, no rollouts, no source changes): Phase 4B §10(a). Rebuild V0 and V3 conceptors from the authors' public KS3 fit activations (`brandonyang/pi05-libero-activations-v1-2000-15env`) and compare them with ours. This would separate fit-data differences from builder differences.

**Not done in this phase:** no message sent, no experiment run, no improvement design started.

## 7. Files

- Added:
  - `research/reproduction/phase6a_reproduction_gap_analysis.md` (this report)
  - `research/reproduction/experiments/phase6a_reproduction_gap_analysis.yaml`
- No other file changed. Nothing committed.
