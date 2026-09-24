# Phase 2B — Random Matched Conceptor Ablation: Does the Conceptor Direction Matter?

**Development split only (`--seed 15`, init states 15–29). States 30–44 remain untouched.**
This is a controlled mechanism ablation. Nothing is tuned: no layer or β sweep, no new COAST variant. Every statement below holds only for this one task, layer, β, conceptor and noise seed.

Paper: *Contrastive Conceptor Activation Steering (COAST)*, arXiv:2605.17144.
- Base commit: `c460acd23c532c339ff8157324ecd4b111045170` (Phase 2A), branch `exp/random-conceptor-ablation`.
- The runs used the uncommitted working tree described in §2. At run time, `sha256(git diff HEAD -- src scripts examples packages)` = `9dcda0e5…f3b3e9`.

## 0. Summary

| condition | M | successes / 15 | Wilson 95% | ‖h′‖/‖h‖ (realized) |
|---|---|---|---|---|
| A. Baseline | I (no hook) | **13** | 0.62–0.96 | 1 |
| B. Pure shrinkage | 0.9·I | **10** | 0.42–0.85 | 0.8984 |
| C. Real COAST (`global`, V0) | 0.9·I + 0.1·C_real | **11** | 0.48–0.89 | 0.9006 |
| D. Random matched | 0.9·I + 0.1·C_random | **11** | 0.48–0.89 | 0.9023 |

- **Random matched reproduces real COAST on 15/15 episode outcomes.**
  - C_random has C_real's eigenvalue spectrum (to 2·10⁻⁹) and eigenvectors at chance overlap.
  - It keeps all three of COAST's losses relative to baseline (states 15, 18, 24).
  - It also keeps COAST's single gain (state 21, 203 → 200 steps). State 21 is the one state where COAST and shrinkage disagreed in Phase 2A and again here.
- **All three steered conditions agree on 14/15 states.** The one exception is state 21, where both conceptor conditions succeed and shrinkage times out.
- **Answer (this configuration only): Case 2, real COAST ≈ random ≈ shrinkage.**
  - Destroying the eigenvectors while keeping the spectrum left every episode outcome unchanged.
  - Removing the conceptor entirely (shrinkage) changed one outcome. Random matched shares that outcome with real COAST, so whatever separates COAST from shrinkage at state 21 does **not** depend on C_real's eigenvectors.
  - No behavioral contribution of the conceptor **direction** is detectable here.
- **Why they coincide, at the hidden-state level:**
  - All three interventions are an ~10% contraction of h along h itself: cos(Δh, h) = −0.9999 / −0.9996 / −0.9985 for B / C / D.
  - Their realized contractions differ by at most 0.39 percentage points (pp).
  - The real conceptor keeps *less* of h than a random rotation of it does. The Rayleigh quotient hᵀCh/‖h‖² is 0.0062 for C_real vs 0.0174 for C_random, where trace/d = 0.0185.
  - So on the model's own hidden states, C_real acts even closer to pure shrinkage than C_random does (§5.2).
- The success counts 13 / 10 / 11 / 11 all lie inside the baseline's own noise-seed spread from Phase 1D (7–13/15). No pairwise difference is statistically distinguishable at N = 15.

## 1. Design

- **Model:** pi0.5 (`pi05_libero`, PyTorch).
- **Checkpoint:** `checkpoints/openpi-libero-2000`.
- **Task:** libero_10 task 2, `KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it`.
- **Episodes:** `--seed 15 --num_episodes 15` (episode k → init state 15+k). Max 520 steps, replan every 5.
- **Paired policy noise:** master seed **100** for all four conditions.
- **Fixed steering:** layer 5, α 0.5, β 0.1. Nothing tuned. Shrinkage does not use α.
- **Conceptor:** repository-faithful V0, `conceptors/phase1b_repo_task02.npz` (sha256 `8b1c802a…ec80f3`, key `…__L5__0.5__C_contrastive`). It is identical to Phase 2A's.
- **One server process** served all four conditions on GPU 1 (RTX 4090, the same GPU as Phase 2A), with flags `--steer --noise-control --steering-diagnostics`.
  - A sends no `__steering__` payload.
  - B, C and D send `strategy = shrinkage / global / random_matched`.
- **Preregistered before any rollout** in `examples/libero_env/output/phase2b_servers/preregistration.txt` (written 10:07:42, before the first rollout at 10:08:31). It fixes:
  - seed 100;
  - the order A → B → C → D;
  - one run per condition;
  - the primary comparison (C vs D);
  - the interpretation rule.

  Each condition ran exactly once.

## 2. Implementation

### 2.1 The `random_matched` mode already existed and was reused unchanged

The repository already has a `random_matched` strategy, added upstream in `6c34a11` ("Make PR35 steering merge-ready with main"). It implements exactly the Phase 2B construction:

```
C_sym      = (C_real + C_realᵀ) / 2                           conceptors.random_matched_conceptor
Λ          = eigvalsh(C_sym)                                  (float64)
Q, R       = qr(default_rng(seed).standard_normal((d, d)))
C_random   = Q Λ Qᵀ   → float32
M_random   = (1-β) I + β C_random                             ConceptorSteeringHook._build_M (unchanged)
seed       = blake2b(repr((task, layer, α, "random_matched")))   SteeredPolicyWrapper, β-independent
```

- **The seed** for this configuration is `1466119911`. It is a stable hash, so it is identical across processes. It does not depend on β, so a β sweep would keep the same random basis.
- **QR sign fix.** The generator does not apply the conventional sign fix Q ← Q·diag(sign(diag R)).
  - The fix cannot change C_random: flipping the sign of column i of Q multiplies λᵢ·qᵢqᵢᵀ by (±1)², which is 1.
  - This was checked on the real conceptor and in a unit test: C_random built with the sign-fixed Q is **bit-identical** to production (max |Δ| = 0.0, with 537/1024 diag(R) entries negative).
  - The fix affects only whether Q, as a matrix, is exactly Haar-distributed. C_random's distribution is the same either way.
  - The upstream function was therefore **not modified**.

### 2.2 Code changes

| File | Change |
|---|---|
| `src/openpi/serving/steering.py` | +1 line: `logger.info("random_matched: seed=%d from seed_key=%r", …)` in the existing `random_matched` branch. The seed is now recorded in the server log. There is no behavior change. |
| `tests/test_steering_random_matched.py` (new, 11 tests) | §3. |
| `research/reproduction/tools/validate_phase2b_random_conceptor.py` (new) | Offline spectral and hidden-state validation (§4). |
| `research/reproduction/tools/analyze_phase2b.py` (new) | Run analysis (§5, §6). Reuses the Phase 1D/2A helpers. |

**Unchanged:**
- conceptor construction (`conceptors.py`, `experiments/libero/compute_conceptors.py`);
- activation collection;
- `ConceptorSteeringHook` and the `global` formula;
- `ShrinkageSteeringHook`;
- `random_matched_conceptor` itself;
- the client (`main.py` already listed `random_matched`);
- `ALLOWED_STRATEGIES`.

`~/Stanley_ws/lerobot` was not touched.

## 3. Tests

`tests/test_steering_random_matched.py` has 11 CPU tests. The fixture is a d = 64 `C_s AND NOT C_f` built from anisotropic data. Like the real NPZ conceptor, it is not exactly symmetric, and its top-4 eigengap is > 0.01.

| Req. | Test | Result |
|---|---|---|
| 1 eigenvalues preserved | the hook's C (recovered from M) and the generator output have the spectrum of sym(C_real) (atol 5e-6 / 1e-6); trace matches; output exactly symmetric | pass |
| 2 eigenvectors changed | top-4 subspace overlap ‖U₄ᵀV₄‖²_F/4: 1.0 against itself, < 0.25 against C_random (chance 1/16); ‖C_rand − C_real‖ > 0.5‖C_real‖ | pass |
| — QR sign fix | Q orthogonal to 1e-12; C with sign-fixed Q `array_equal` to production | pass |
| 3 same seed → identical | two independent wrappers give `torch.equal` M; generator deterministic; wrapper M equals the documented seed's matrix; seed logged | pass |
| 4 different seed → different eigenvectors | seeds 1 vs 2: same spectrum, top-4 overlap < 0.25, large Frobenius difference | pass |
| 5 `global` unchanged | `global` M `torch.equal` to 0.9·I + 0.1·C_real before and after building `random_matched` in the same wrapper; same cached object; fresh wrapper identical; NPZ array unchanged; forward = h @ Mᵀ | pass |
| 6 shrinkage still works | with `random_matched` and `global` hooks in the cache, shrinkage is still `ShrinkageSteeringHook`, M = 0.9·I, h′ = 0.9·h exactly | pass |

- The existing `random_matched` tests in `tests/test_conceptors.py` and `tests/test_steering.py` (spectrum, determinism, β-independent seed, cross-process stability) still pass.
- The full CI suite, `CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu uv run pytest --strict-markers -m "not manual"`, gives **420 passed, 47 skipped, 0 failed**. That is 409 from Phase 2A plus the 11 new tests.
- `ruff check .` is clean.
- One test was corrected during development. The first fixture check asserted a 2× eigengap; the smooth fixture spectrum has a 0.028 absolute gap. The check was changed to an absolute gap > 0.01. This was a test-expectation fix; no code changed.

## 4. Spectral validation (before any rollout)

`CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu uv run python research/reproduction/tools/validate_phase2b_random_conceptor.py`

C_random is built through the production path: the wrapper's seed derivation, then `get_conceptor_matrix`. The wrapper's M is asserted `torch.equal` to 0.9·I + 0.1·C_random.

**Setup**
- **seed** 1466119911;
- **dimension** 1024;
- **generator:** `numpy.random.default_rng(seed).standard_normal((1024,1024))` → `np.linalg.qr`.

### 4.1 Symmetry, eigenvalues, orthogonality

| | C_real | C_random |
|---|---|---|
| symmetry error max \|C − Cᵀ\| | 3.5·10⁻³ | **0** |
| relative ‖C − Cᵀ‖_F / ‖C‖_F | 0.085 (skew part 4.2% of ‖C‖) | 0 |
| λ_max / top-5 | 0.3522 / 0.352, 0.343, 0.334, 0.333, 0.322 | identical |
| median / min λ | 0.00132 / −0.00525 | identical |
| trace (trace/d) | 18.939 (0.01850) | 18.939 (0.01850) |
| # λ > 0.1 / > 0.01 / > 0.001 | 60 / 197 / 571 | identical |
| # λ < 0 | 3 (≥ −0.0053) | identical |
| ‖C‖_F | 1.8073 | 1.8057 |

- **‖eig(sym C_real) − eig(C_random)‖₂ = 2.1·10⁻⁹** (max 2.0·10⁻¹⁰, relative 1.1·10⁻⁹).
- **Orthogonality error of Q:** max |QᵀQ − I| = 1.3·10⁻¹⁵ (Frobenius 3.3·10⁻¹⁴).
- **C_real is not symmetric,** because `C_s AND NOT C_f` is a product of conceptors.
  - Its general eigenvalues are real (max |Im| = 0) and differ from those of its symmetric part by at most 0.0053.
  - C_random matches the **symmetric part's** spectrum, as the spec (step 1) requires. It therefore also drops C_real's skew part.
  - The skew part S contributes nothing along h (hᵀSh = 0). It is 4.2% of ‖C‖ and is scaled by β = 0.1.
- **Three slightly negative eigenvalues** (≥ −0.0053) are inherited from C_real's numerics. C_random is exactly as "conceptor-valid" as C_real's symmetric part: λ ∈ [−0.0053, 0.352].

### 4.2 Eigenvector similarity

Subspace overlap ‖U_kᵀV_k‖²_F / k is 1 for the same subspace and k/d at chance.

| k | real vs real | **real vs random** | chance k/d | random(seed) vs random(seed+1) |
|---|---|---|---|---|
| 1 | 1.000 | **0.00009** | 0.00098 | 0.0021 |
| 5 | 1.000 | **0.0032** | 0.0049 | 0.0029 |
| 10 | 1.000 | **0.0071** | 0.0098 | 0.0087 |
| 20 | 1.000 | **0.020** | 0.020 | 0.018 |
| 50 | 1.000 | **0.049** | 0.049 | 0.048 |
| 100 | 1.000 | **0.097** | 0.098 | 0.097 |

- |cos| between the leading eigenvectors (i = 1..5): 0.010, 0.006, 0.002, 0.014, 0.029.
- The largest |cos| between C_real's top eigenvector and any of C_random's top 100 is 0.082.
- Frobenius cosine of the trace-free parts: real vs random **0.0006**, real vs real 1.0.
- **Conclusion:** the eigenvectors are fully randomized; overlap is at chance for every k.

### 4.3 What M actually applies (fp32 vs bf16)

The hook casts M to the activations' dtype (bf16), as documented in Phase 2A.

| | fp32 mean diag(M) | bf16 diag(M) values (count) |
|---|---|---|
| real COAST | 0.90185 | 0.90234 (728), 0.89844 (185), 0.90625 (105), 0.91016 (5) |
| random matched | 0.90185 | **0.90234 (all 1024)** |
| shrinkage | 0.90000 | 0.89844 (all 1024) |

- diag(C_random) is concentrated: 0.0185 ± 0.0023, against 0.0185 ± 0.0183 for C_real. So every diagonal entry of M_random rounds to the same bf16 value, 0.90234.
- This is a real but small difference in realized contraction between the conditions. It follows from the construction and was not corrected (§7).

### 4.4 Predicted effect on real hidden states

Input: the Phase 1A layer-5 activations for this task. These are the conceptor's own training data: 15 episodes (8 success), every 4th rollout step, all denoising steps and tokens, n = 28,200 vectors.

| | hᵀCh/‖h‖² (p5–p95) | ‖h′‖/‖h‖ (bf16) | cos(Δh, h) (bf16) | ‖Δh‖/‖h‖ (bf16) |
|---|---|---|---|---|
| real COAST | **0.0062** (0.0049–0.0077) | 0.90063 | −0.99963 | 0.0994 |
| random matched | **0.0174** (0.0141–0.0211) | 0.90233 | −0.99852 | 0.0978 |
| shrinkage | — | 0.89838 | −0.99990 | 0.1016 |

- For a random rotation of C, the expected hᵀCh/‖h‖² is trace/d = 0.0185, and C_random gets 0.0174.
- **C_real gets about 3× less than that.** h lies mostly in C_real's *low*-eigenvalue subspace, which is consistent with the NOT C_f factor suppressing directions shared by success and failure.
- As a result, the real conceptor's term β·C·h is nearly orthogonal to, and very small relative to, the 0.9·h term.
- Distance between steered states (fp32, per vector, relative to ‖h‖):

| pair | distance |
|---|---|
| real vs shrinkage | 0.20% |
| random vs shrinkage | 0.54% |
| real vs random | 0.56% |

- **The runtime diagnostics confirm these offline predictions to four decimals.** Realized norm ratios were 0.90062 (C), 0.90233 (D) and 0.89838 (B) (§5.4).

## 5. Run and results

- **Server:** launched 2026-09-24 10:07:53 +08:00.
  - `CUDA_VISIBLE_DEVICES=1 uv run --no-sync scripts/serve_policy.py --pytorch --steer --noise-control --steering-diagnostics --conceptor-npz conceptors/phase1b_repo_task02.npz --port 8102 policy:checkpoint --policy.config pi05_libero --policy.dir checkpoints/openpi-libero-2000`
- **Clients:** driver `examples/libero_env/output/phase2b_servers/run_clients.sh`. The flags are the same as Phase 2A, except `--port 8102` and `--output_dir output/phase2b_<cond>_task02_seed15`.

| run | wall clock | exit | result |
|---|---|---|---|
| A baseline | 10:08:31–10:10:58 | 0 | 13/15 |
| B shrinkage | 10:10:58–10:13:49 | 0 | 10/15 |
| C COAST | 10:13:49–10:16:30 | 0 | 11/15 |
| D random matched | 10:16:30–10:19:13 | 0 | 11/15 |

- The server log has 0 Traceback/ERROR lines.
- The hooks built were `ShrinkageSteeringHook`, `ConceptorSteeringHook` (`global`) and `ConceptorSteeringHook` (`random_matched`).
- The server logged `random_matched: seed=1466119911 from seed_key=('KITCHEN_SCENE3_…', 5, 0.5, 'random_matched')`, the same seed as in §4.
- GPU 1 peaked at 8,343 MiB and 73% utilization.
- Each client printed only the 5 known benign warnings.

Analysis: `CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu uv run python research/reproduction/tools/analyze_phase2b.py`. All assertions passed.

### 5.1 Validation

- **Noise pairing.**
  - All 3,676 fingerprints were recomputed from their keys and match (A 845, B 980, C 922, D 929 requests).
  - Every episode's request schedule is 0, 5, … up to its length.
  - All 6 condition pairs carry identical noise at every shared coordinate: C–D 912/912, B–D 905/905, A–D 746/746, and so on.
- **Diagnostics.** B, C and D each have exactly 10 records per call (denoising steps 0..9), layer 5, 10 tokens, dim 1024, β 0.1, the correct strategy, and all finite values. The schedules equal the fingerprint schedules. Hook applications: B 9,800, C 9,220, D 9,290.
- **Replay against Phase 2A** (same configuration, seed and GPU; separate server process):

| condition | outcomes | rollout lengths |
|---|---|---|
| baseline | 15/15 | **15/15** |
| shrinkage | 15/15 | 14/15 (state 19: 209 → 239) |
| COAST | 15/15 | 14/15 (state 29: 225 → 224) |

  - The unsteered path again replays exactly.
  - The steered path again shows cross-process length jitter, which Phase 2A also saw. This time one case was 30 steps (shrinkage, state 19), not just one step. No outcome changed.
  - Single-state differences between steered conditions should be read with this in mind.

### 5.2 Success and episode-level transitions (items 1–2)

| init | A base | B shrink | C COAST | D random | steps A / B / C / D |
|---|---|---|---|---|---|
| 15 | 1 | 0 | 0 | 0 | 216 / 520 / 520 / 520 |
| 16 | 1 | 1 | 1 | 1 | 215 / 211 / 212 / 209 |
| 17 | 1 | 1 | 1 | 1 | 262 / 239 / 238 / 234 |
| 18 | 1 | 0 | 0 | 0 | 252 / 520 / 520 / 520 |
| 19 | 1 | 1 | 1 | 1 | 211 / 239 / 201 / 208 |
| 20 | 1 | 1 | 1 | 1 | 352 / 238 / 232 / 231 |
| 21 | 0 | 0 | **1** | **1** | 520 / 520 / 203 / 200 |
| 22 | 1 | 1 | 1 | 1 | 228 / 177 / 238 / 226 |
| 23 | 1 | 1 | 1 | 1 | 259 / 239 / 253 / 257 |
| 24 | 1 | 0 | 0 | 0 | 235 / 520 / 520 / 520 |
| 25 | 1 | 1 | 1 | 1 | 217 / 224 / 222 / 221 |
| 26 | 1 | 1 | 1 | 1 | 240 / 246 / 258 / 246 |
| 27 | 1 | 1 | 1 | 1 | 249 / 248 / 224 / 299 |
| 28 | 0 | 0 | 0 | 0 | 520 / 520 / 520 / 520 |
| 29 | 1 | 1 | 1 | 1 | 222 / 217 / 224 / 206 |
| **total** | **13** | **10** | **11** | **11** | |

Rollout steps exclude the 10 settle steps; 520 means timeout.

| pair | fail→success | success→fail | agreement | identical length | exact McNemar p (reference only) | disagreeing states |
|---|---|---|---|---|---|---|
| **COAST ↔ random matched** | 0 | 0 | **15/15** | 4/15 | 1.0 | none |
| COAST ↔ shrinkage | shrink→COAST: 1 | 0 | 14/15 | 4/15 | 1.0 | 21 |
| random ↔ shrinkage | shrink→random: 1 | 0 | 14/15 | 5/15 | 1.0 | 21 |
| baseline → shrinkage | 0 | 3 | 12/15 | 2/15 | 0.25 | 15, 18, 24 |
| baseline → COAST | 1 | 3 | 11/15 | 1/15 | 0.625 | 15, 18, 21, 24 |
| baseline → random | 1 | 3 | 11/15 | 1/15 | 0.625 | 15, 18, 21, 24 |

- All four conditions succeed on 10 states and all four fail on 1 (state 28).
- The three steered conditions agree on 14/15 states.
- COAST and random matched share the same success vector. Their trajectories still differ in detail: only 4/15 rollout lengths are identical, and in the most extreme case, state 27, the lengths are 224 vs 299.

### 5.3 Behavior: real COAST vs random matched (item 7)

- **Indistinguishable at the outcome level: 15/15 agreement.**
  - D reproduces every COAST effect relative to baseline: the three losses (15, 18, 24) and the one gain (21).
  - D also reproduces COAST's only difference from shrinkage, the success at state 21, with a near-equal length (200 vs 203).
- **Within-call hidden-state paths are close.** Rollout step 0 is the one point where C and D see the identical observation and noise. There, layer-5 ‖h‖ stays within **0.036%** between C and D across all 10 denoising steps (0 at step 0, rising to at most 3.9·10⁻⁴).

### 5.4 Behavior: real COAST vs shrinkage (item 8), intervention magnitude and norm change (items 3–4)

- **Behavior.** COAST vs shrinkage reproduces Phase 2A exactly: 14/15 agreement, disagreeing only at state 21.
  - In Phase 2A this single state could not be attributed.
  - Here the random matched conceptor, which shares nothing directional with C_real, gives the same state-21 success.
  - So the state-21 difference between COAST and shrinkage does not depend on C_real's direction.
  - It is consistent with either of two things, which this experiment cannot separate:
    1. the slightly weaker contraction of both conceptor conditions (9.94% / 9.78% vs 10.16%);
    2. any generic effect of adding β·C for a C with this spectrum.
- **Intervention magnitude, per hook application at layer 5:**

| | n | mean ‖Δh‖ (sd; min–max) | mean ‖Δh‖/‖h‖ | mean cos(Δh, h) | mean ‖h′‖/‖h‖ |
|---|---|---|---|---|---|
| B shrinkage | 9,800 | 17.341 (2.691; 13.07–24.63) | 0.10163 | −0.99990 | **0.89838** ± 0.00014 |
| C real COAST | 9,220 | 16.963 (2.668; 12.74–24.10) | 0.09942 | −0.99962 | **0.90062** ± 0.00029 |
| D random matched | 9,290 | 16.684 (2.596; 12.54–23.62) | 0.09783 | −0.99851 | **0.90233** ± 0.00015 |

- **By denoising step** (mean ‖Δh‖, with norm ratio):

| step | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|
| B ‖Δh‖ | 22.75 | 20.33 | 19.06 | 18.17 | 17.54 | 16.48 | 15.84 | 15.12 | 14.35 | 13.77 |
| C ‖Δh‖ | 22.30 | 19.93 | 18.67 | 17.80 | 17.17 | 16.12 | 15.49 | 14.77 | 13.99 | 13.39 |
| D ‖Δh‖ | 21.90 | 19.57 | 18.34 | 17.48 | 16.87 | 15.85 | 15.24 | 14.55 | 13.80 | 13.24 |
| C ratio | 0.90039 | 0.90039 | 0.90043 | 0.90046 | 0.90051 | 0.90058 | 0.90066 | 0.90076 | 0.90093 | 0.90115 |
| D ratio | 0.90232 | 0.90233 | 0.90233 | 0.90233 | 0.90234 | 0.90234 | 0.90234 | 0.90233 | 0.90234 | 0.90234 |
| mean ‖h‖ | 223.8 | 200.0 | 187.5 | 178.7 | 172.5 | 162.1 | 155.8 | 148.8 | 141.2 | 135.5 |

- **Paired shared coordinates** (the same episode, rollout step and denoising step in both conditions):

| pair | coordinates | ‖Δh‖ ratio | norm-ratio difference |
|---|---|---|---|
| COAST / random | 9,120 | 1.016 | −0.0017 |
| shrink / COAST | 9,040 | 1.023 | −0.0022 |
| shrink / random | 9,050 | 1.039 | −0.0040 |

- **Magnitude ordering:** the contraction strength is shrinkage (10.16%) > COAST (9.94%) > random (9.78%). The spread is 0.38 pp. All three are "shrink h by about 10% along itself".
- **Direction of Δh:** the part of Δh orthogonal to h is largest for random matched (cos −0.9985, versus −0.9996 for COAST and −0.9999 for shrinkage). It is still tiny: sin θ ≈ 0.055, i.e. about 0.5% of ‖h‖.
- **Hidden-state norm:** mean layer-5 ‖h‖ over the run is 170.63 (B), 170.56 (C) and 170.54 (D). Per-episode values are in `research/reproduction/experiments/phase2b_episode_results.csv`.

## 6. Interpretation (bounded)

- **Outcome: Case 2, real COAST ≈ random matched ≈ shrinkage** (for pi0.5, LIBERO-10 task 2, layer 5, α 0.5, β 0.1, V0 conceptor, noise seed 100, states 15–29).
- **Direction does not matter here.**
  - Keeping C_real's full eigenvalue spectrum while replacing its eigenvectors with a random basis left all 15 episode outcomes unchanged, including COAST's single non-shrinkage outcome.
  - The Case-1 prediction (real > random ≈ shrinkage) would require D to lose state 21 or otherwise side with shrinkage. It did not.
  - There is no evidence for Case 3 (random > real): the two conditions are identical in outcome.
- **Mechanistically this is expected, not surprising.**
  - At β = 0.1 the conceptor term is at most 0.1·λ_max = 0.035 of h along any direction.
  - On this model's actual layer-5 states it contributes hᵀCh/‖h‖² ≈ 0.006 (real) or 0.017 (random).
  - Either way, M·h ≈ 0.90·h. The directional content that distinguishes C_real from C_random changes the steered hidden state by about 0.5% of its norm.
- **This is the directional complement of Phase 2A.**
  - Phase 2A removed C and kept the contraction: 14/15 unchanged.
  - Phase 2B keeps C's spectrum and removes its direction: 15/15 unchanged.
  - Together, at this setting, COAST's behavioral footprint is accounted for by an approximately 10% residual-stream contraction plus, at most, a spectrum-level (not direction-level) term.
- **What this does not show.**
  - It does not show that conceptor directions are useless in general.
  - C_real is weak here (λ_max 0.35, median 0.0013), and the layer-5 hidden states even fall mostly in its low-eigenvalue subspace.
  - Larger β, other layers, other tasks, or a conceptor whose top subspace aligns with h could behave differently. None of these were tested, by design.

## 7. Limitations

- **Small N.**
  - There are 15 episodes per condition, one master noise seed and one run per condition.
  - The baseline's seed-to-seed spread (7–13/15) is larger than every difference here.
  - 15/15 agreement between C and D is strong within this sample, but it rests on one noise seed and one random basis (seed 1466119911).
  - Other random bases were not run (no seed sweep, by design).
- **One configuration only:** one task, layer 5, β 0.1, α 0.5, the V0 conceptor, the development split.
- **Magnitudes are not exactly matched.** Because of bf16 rounding of M's diagonal (§4.3) and the different Rayleigh quotients (§4.4), the realized contractions are 10.16% / 9.94% / 9.78% for B / C / D. They were not equalized, since that would require tuning β.
- **Symmetrization.** C_random matches the spectrum of sym(C_real). C_real's 4.2% skew part is not reproduced. It contributes nothing along h and was not ablated separately.
- **Steered-path jitter.** Rollout lengths differ across processes (up to 30 steps in this phase, with no outcome change; §5.1). Single-state length differences between steered conditions are not interpretable.
- **Measured at the hooked layer only.** Downstream layers, the velocity field and the actions are not decomposed.

## 8. Housekeeping

- **Run outputs** (gitignored, none committed):
  - `examples/libero_env/output/phase2b_{baseline,shrinkage,coast,random_matched}_task02_seed15/`, 11–17 MB each: videos, client logs, fingerprint and diagnostics JSONL.
  - `…/phase2b_servers/`: server log, GPU trace, driver, preregistration, pids.
- The server and GPU logger were stopped after the runs, and GPU 1 returned to idle.
- **Held-out states 30–44 were not used**, and `~/Stanley_ws/lerobot` was not touched.
