# Phase 1B — Repository-Faithful vs Paper-Faithful COAST Conceptors (offline)

Paper: *Contrastive Conceptor Activation Steering (COAST)*, arXiv:2605.17144.
Code commit: `171cc4e4c59e9ac20345e7df34b50d458df4acef` (branch `repro/arxiv-2605-17144`; upstream code = `2afa10e`).
Scope: offline numerical comparison only. No rollouts, recollection, steering, or sweeps. `src/openpi/serving/conceptors.py` was not modified.

Labels used throughout:
- **REPOSITORY-FAITHFUL** reproduces the current released implementation.
- **PAPER-FAITHFUL** implements the mathematical procedure stated in arXiv:2605.17144.

Neither label is a correctness judgement.

## 1. Source data

| Item | Value |
|---|---|
| Activations | `activations/libero/openpi-libero-2000/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it/` (Phase 1A, unmodified) |
| Task | libero_10 task 2, "turn on the stove and put the moka pot on it" |
| Episodes | 15, of which 8 success and 7 failure (re-verified from `rewards.npz`, cross-checked against `metadata.json`) |
| Inference calls | 385 success, 728 failure |
| Tensor | `all_suffix_residual` (10 denoise, 4 layer slots [0,5,11,17], 10 tokens, 1024), fp32 (bf16-origin) |
| Layers analysed | 5 (paper oracle for Stove+Moka, Table 17: L5, α=0.5, β=0.1) and 11 (paper's geometric layer, Table 16) |
| Alphas | 0.1, 0.5, 1.0, 2.0, 10.0 (global); per-step at α=1.0 (the repository's fixed per-step aperture), steps 0–9 |

## 2. Constructions

### Repository-faithful (`src/openpi/serving/conceptors.py`, `compute_task_conceptors`)
1. Samples: every (inference call, denoise step, **token**) is a row (`flatten_global`; `flatten_per_step` keeps one denoise step).
2. `R = XᵀX / N`: **uncentered** (`correlation_matrix`), float64.
3. `C = R (R + α⁻² I)⁻¹` via `np.linalg.solve`, then cast to **float32**.
4. `NOT(C_f) = I − C_f`.
5. `AND(A, B) = A · inv(A + B − A·B) · B` (`boolean_and`; `solve`, with a 1e-8 ridge on LinAlgError).
6. `C_contrastive = AND(C_s, NOT(C_f))`, cast to float32. Per-step uses a fixed α = 1.0.
7. Task-level class threshold: `min_episodes_per_class = 2`.

### Paper-faithful (Sec. 3.1–3.2, Eq. 1–4, App. A.9.1; implemented in `research/reproduction/tools/compare_conceptor_builders.py`)
1. **Token pooling:** `h = mean over the 10 action tokens`, giving one 1024-D vector per (inference call, denoise step). Global uses all 10 steps; per-step uses step t only.
2. **Class-specific centering:** `X̃ = X − mean(X)`, `R = X̃ᵀX̃ / N`, float64.
3. `C = R (R + α⁻² I)⁻¹`, computed as `U diag(λ/(λ+α⁻²)) Uᵀ` from `eigh(R)` (λ clipped at 0). This is mathematically identical to the solve form; the max |difference| against the repository's `conceptor()` on the same R was ≤ 1.1e-11.
4. `NOT(C_f) = I − C_f`.
5. `AND(A, B) = pinv(pinv(A) + pinv(B) − I)` (Eq. 3/7; A.9.1 pseudocode), using numpy 1.26.4 `pinv` defaults (`rcond=1e-15, hermitian=False`).
6. Stored as float32 (A.9.4); all computation in float64.

### Ablation chain (same data, task, layers, alphas)

| Variant | Samples | R | AND |
|---|---|---|---|
| **V0** (repository-faithful) | token-flattened | uncentered | repository |
| **V1** | token-pooled | uncentered | repository |
| **V2** | token-pooled | centered | repository |
| **V3** (paper-faithful) | token-pooled | centered | canonical pinv |

V0–V2 call the repository functions unchanged, including the float32 casts before AND. **V0 reproduces the official builder NPZ bit-for-bit (max |diff| = 0.0 over all 90 matrices)**, which validates the loader and sample ordering.

### Sample counts

| | flattened (V0) | pooled (V1–V3) |
|---|---|---|
| Global, per layer | success 38,500; failure 72,800 | success 3,850; failure 7,280 |
| Per-step, per layer and step | success 3,850; failure 7,280 | **success 385; failure 728** (both < d = 1024) |

With pooling, the per-step R has rank 384 (success) and 727 (failure). It is rank-deficient.

### Centering statistics (pooled vectors)

| Layer / set | ‖mean‖ | mean ‖row‖ | share of uncentered tr(R) from the mean |
|---|---|---|---|
| L5 global, success/failure | 76.0 / 75.7 | 95.5 / 95.2 | 0.62 / 0.62 |
| L11 global, success/failure | 127.7 / 126.1 | 143.6 / 141.9 | 0.78 / 0.78 |
| L11 per-step 9, success/failure | 157.0 / 156.3 | 164.9 / 163.6 | 0.91 / 0.91 |

For token-flattened rows the mean's share is lower (L5 0.20, L11 0.39), because between-token variance adds to the trace. ‖mean_success − mean_failure‖ (pooled) is 4.4 (L5) and 7.9 (L11), which is small compared with the shared mean.

## 3. Verification of both constructions

All 360 matrices (4 variants × 2 layers × 15 strategy keys × 3 kinds) are 1024 × 1024, float32 when stored, and finite.

| Property | V0 (repository) | V1 / V2 (repository AND) | V3 (paper), global | V3 (paper), per-step |
|---|---|---|---|---|
| C_s, C_f symmetry ‖C−Cᵀ‖/‖C‖ | ≤ 2.3e-10 | ≤ 2.3e-10 | 0 | 0 |
| C_s, C_f eigenvalues | within [0, 1] | within [0, 1] | within [0, 1] | within [0, 1] |
| **C_contrastive symmetry** | **0.08–0.27** | **0.04–0.36** | **≤ 2e-9** | 0.05–0.39 |
| C_contrastive eigenvalues | real part [~0, 0.59]; complex (max \|imag\| 1.1e-3); symmetric-part min −0.038 | symmetric-part min −0.046 | [9.7e-8, 0.502], PSD, ≤ 1 | symmetric-part min −0.335 (not PSD) |
| pinv tolerance sensitivity | n/a | n/a | rcond ≤ 1e-6: ≤ 9.4e-4 relFro; rcond 1e-4: ≤ 1.2% | **ill-posed**: 1e-12 → up to 3.2e7× relFro; 1e-4 → 0.09–97×; `hermitian=True` → 12–32% |

**Findings:**
- **Global V3 behaves as theory predicts:** symmetric to floating-point tolerance, PSD, eigenvalues in [0, 1] (max 0.50), and insensitive to the pinv tolerance.
- **Per-step V3, under a literal reading of the paper, is numerically ill-posed on this dataset.** Pooling leaves 385 success samples per step, so C_success has ≥ 639 exactly-zero eigenvalues. `pinv(C_s)` then depends entirely on where `rcond` cuts noise-level eigenvalues, and `pinv(C_s) + pinv(¬C_f) − I` is itself near-singular. The result is non-symmetric, non-PSD, and changes by orders of magnitude with `rcond`. The paper (A.9.4) reports "no numerical issues", which suggests its per-step fit did not have N < d per class. It may have used more samples or no pooling, but this cannot be determined from the paper. **The per-step matrices in `phase1b_paper_task02.npz` (default rcond) should not be used for steering until this is resolved.** No tolerance was silently chosen.
- **Every repository-AND contrastive matrix is non-symmetric** (4–36%). This matters because the steering hook applies `h @ M.T` without symmetrizing (`steering.py:403`). Using M instead of Mᵀ would change the intervention vector by 0.7–4.4% (§6).

## 4. Repository vs paper, and the V0→V3 chain

Relative Frobenius change `‖C_next − C_prev‖_F / ‖C_prev‖_F` (V0vsV3 = `‖C_V0 − C_V3‖/‖C_V3‖`). Ranges are over α (global) or over the 10 steps (per-step, α=1).

| Layer / strategy | Matrix | V0→V1 (pooling) | V1→V2 (centering) | V2→V3 (AND) | V0 vs V3 |
|---|---|---|---|---|---|
| L5 global | C_contrastive | **0.66–1.23** | 0.06–0.22 | 0.03–0.21 | 0.66–1.00 |
| L5 global | C_success / C_failure | 0.28–0.53 | 0.05–0.27 | 0.000 | 0.34–1.05 |
| L5 per-step | C_contrastive | **0.71–0.84** | 0.10–0.24 | 0.19–0.30* | 1.30–1.73* |
| L11 global | C_contrastive | **0.53–0.79** | 0.04–0.11 | 0.06–0.23 | 0.58–0.73 |
| L11 global | C_success / C_failure | 0.29–0.56 | 0.02–0.27 | 0.000 | 0.35–1.08 |
| L11 per-step | C_contrastive | **0.56–0.68** | 0.08–0.09 | 0.30–0.41* | 0.80–0.97* |

\* involves the ill-posed per-step V3 (§3), so read with caution.

V2→V3 is 0.000 for C_s and C_f because the conceptor formula is identical; only the AND differs.

**Top-k eigenspace alignment of C_contrastive** (mean cos² of principal angles; 1 = identical subspace, ≈ k/1024 = random):

| Config | V0↔V1 (k=10 / 32) | V1↔V2 | V2↔V3 | **V0↔V3** |
|---|---|---|---|---|
| L5 α=0.5 (Stove+Moka oracle) | 0.44 / 0.70 | 0.90 / 0.97 | 1.00 / 0.99 | **0.41 / 0.69** |
| L5 α=1.0 | 0.44 / 0.45 | 0.97 / 0.96 | 0.91 / 1.00 | 0.41 / 0.44 |
| L5 α=10 | 0.45 / 0.39 | 0.99 / 0.99 | 0.86 / 0.92 | 0.36 / 0.35 |
| L11 α=0.5 (geometric) | 0.62 / 0.64 | 0.99 / 0.99 | 0.97 / 0.97 | **0.60 / 0.63** |
| L11 α=1.0 (geometric) | 0.73 / 0.66 | 0.99 / 0.99 | 0.86 / 0.95 | 0.67 / 0.64 |
| L11 α=10 | **0.10 / 0.16** | 0.99 / 1.00 | 0.86 / 0.93 | **0.07 / 0.14** |

### C_contrastive quota and effective rank (erank = exp(entropy of the normalized singular values))

| Config | V0 quota / erank | V1 | V2 | V3 quota / erank |
|---|---|---|---|---|
| L5 α=0.5 | 0.0185 / 180 | 0.0104 / 114 | 0.0104 / 114 | 0.0103 / 114 |
| L5 α=1.0 | 0.0364 / 281 | 0.0184 / 156 | 0.0183 / 156 | 0.0183 / 156 |
| L5 α=10 | 0.1681 / 793 | 0.0976 / 523 | 0.0976 / 524 | 0.0952 / 531 |
| L11 α=0.5 | 0.0488 / 403 | 0.0328 / 273 | 0.0328 / 273 | 0.0324 / 276 |
| L11 α=1.0 | 0.0930 / 587 | 0.0581 / 394 | 0.0581 / 394 | 0.0571 / 397 |
| L11 α=10 | 0.1404 / 698 | 0.1612 / 706 | 0.1613 / 707 | 0.1554 / 725 |

### Success–failure overlap, Eq. (11): tr(C_s C_f)/√(tr C_s² · tr C_f²)

| α | L5 V0 | L5 V3 | L11 V0 | L11 V3 | Paper Table 19 (L11, 10-task mean) |
|---|---|---|---|---|---|
| 0.1 | 1.000 | 0.982 | 0.994 | 0.962 | 0.955 |
| 0.5 | 0.995 | 0.995 | 0.971 | 0.961 | 0.937 |
| 1.0 | 0.991 | 0.987 | 0.960 | 0.940 | 0.882 |
| 2.0 | 0.986 | 0.975 | 0.959 | 0.920 | 0.808 |
| 10 | 0.984 | 0.937 | 0.993 | 0.937 | 0.670 |

V1 and V2 overlaps equal V3's to 3 decimals, because the overlap depends only on C_s and C_f, which pooling determines.

**Comparison with paper diagnostics (single task vs the paper's 10-task means; indicative only):**
- Table 18 (quota at α=10): paper L11 0.092 > L5 0.059, ratio 1.56. V0 has L5 0.168 > L11 0.140 (reversed order, ratio 0.84). V3 has L11 0.155 > L5 0.095 (same order as the paper, ratio 1.63).
- Table 19 (overlap vs α, L11): the paper decreases monotonically from 0.955 to 0.670. V0 is non-monotone and returns to 0.993 at α=10. V3 decreases to 0.920 at α=2 and then rises slightly to 0.937. Neither reaches the paper's 0.670 at α=10.
- One task cannot establish which construction produced the paper's numbers, but the quota ordering and the small-α overlaps are closer to V3 (token-pooled) than to V0.

## 5. Which discrepancy matters most

1. **D1 token pooling (V0→V1) is the dominant source.** It moves C_contrastive by 0.53–1.23 (relative Frobenius). Top-10 eigenspace alignment falls to 0.16–0.45 (L5) and 0.10–0.90 (L11), over all α and per-step keys. For α ∈ {0.5, 1, 2}, quota drops by 33–49% and erank by 32–45%. The direction is not uniform: at α = 0.1 quota *rises* (L5 0.0064 → 0.0106), and at L11 α = 10 quota and erank rise slightly (0.140 → 0.161; 698 → 706). Between-token variance (the 10 action-chunk positions) is a large part of the flattened covariance and is absent from the paper's pooled covariance.
2. **D2 centering (V1→V2) matters for C_s and C_f (up to 0.27), but little for C_contrastive** (0.04–0.24; top-10 alignment ≥ 0.89). The mean direction is shared by both classes, so the AND-NOT largely cancels it.
3. **D3 Boolean AND (V2→V3)** changes C_contrastive by 0.03–0.23 (global) with alignment 0.86–1.00. Its qualitative effect is symmetry and PSD-ness: the repository AND yields non-symmetric matrices with slightly complex spectra, while the canonical AND is exactly symmetric PSD (global).

## 6. Offline gate diagnostic (not a behavioural result)

`M = (1−β)I + β·C_contrastive`, applied exactly as the runtime hook does (`h @ M.T`, per token). The sample is 1,600 stored per-token states from `episode_002` (success) and `episode_000` (failure): every 10th inference call × 10 denoise steps × 10 tokens. These are **in-sample** states, used for fitting.

| Config | β | V0 mean ‖hMᵀ−h‖/‖h‖ | V1 | V2 | V3 | ‖hMᵀ − hM‖/‖hMᵀ − h‖ (V0 / V1 / V2 / V3) |
|---|---|---|---|---|---|---|
| L5 α=0.5 (oracle) | 0.1 | 0.0994 | 0.0960 | 0.0945 | 0.0945 | 0.007 / 0.018 / 0.020 / 0.000 |
| L5 α=0.5 | 0.3 | 0.2982 | 0.2880 | 0.2834 | 0.2835 | same |
| L11 α=0.5 | 0.1 | 0.0993 | 0.0959 | 0.0929 | 0.0930 | 0.014 / 0.034 / 0.044 / 0.000 |
| L11 α=1.0 | 0.1 | 0.0997 | 0.0985 | 0.0964 | 0.0964 | 0.011 / 0.027 / 0.039 / 0.000 |

For every variant the relative change is ≈ β. C_contrastive keeps only a small part of h (quota ≤ 0.17, and the dominant shared-mean direction is suppressed), so on stored states the gate acts mostly as a uniform `(1−β)` shrink. It differs between variants only in the small component that is preserved. The last column shows that the repository matrices' asymmetry makes `M` vs `Mᵀ` a 0.7–4.4% difference in the intervention vector.

## 7. Numerical tolerance choices

- All R, C, and AND computation is float64; storage is float32 (as in the repository and in A.9.4).
- Paper `pinv`: numpy defaults (`rcond=1e-15`, `hermitian=False`), as literally specified. The sensitivity was checked at rcond ∈ {1e-15, 1e-12, 1e-10, 1e-8, 1e-6, 1e-4} and with `hermitian=True`.
- Eigenvalues of R were clipped at 0 before forming C. The raw minima are round-off level and are recorded in the results JSON.
- Symmetry and eigenvalue diagnostics use `eigvalsh` when ‖C−Cᵀ‖/‖C‖ < 1e-5, and otherwise general `eigvals` plus `eigvalsh` of the symmetric part.

## 8. Runtime, memory, tests

| Step | Wall time | Peak RSS |
|---|---|---|
| Official builder (`compute_conceptors.py`, layers 5 and 11, 5 α, 10 per-step) | 11.2 s | 4.29 GB |
| Comparison script (V0–V3 build, 360 diagnostics, 720 comparisons, rcond sweep, gate) | 7 min 33 s (build 186 s, diagnostics 264 s) | 3.5 GB |

- CPU only (`CUDA_VISIBLE_DEVICES=""`, `JAX_PLATFORMS=cpu`); no GPU was used.
- Tests: `tests/test_conceptors.py` and `tests/test_steering.py` give **118 passed**. The first attempt had one failure (`test_build_fast_steering_stack_global`) caused only by hiding the GPU without `JAX_PLATFORMS=cpu`; it passes with that set.
- Note: every `boolean_and` unit test uses commuting operands (identity, the same projection, zero), so the suite cannot detect D3.
- Reproducibility: the lint-clean script was re-run, and the outputs are identical to the first run (see YAML).

## 9. Conclusions (paper vs code)

- **Repository-faithful and paper-faithful COAST are materially different operators on this task.** C_contrastive differs by 58–100% (global, relative Frobenius). Top-10 eigenspaces overlap only 0.41 (L5 oracle α=0.5) and 0.60–0.67 (L11 α=0.5/1.0), and as little as 0.07 (L11 α=10) and 0.15 (L5 α=0.1).
- The **dominant cause is D1** (token flattening vs pooling). D2 (centering) mainly affects C_s and C_f and largely cancels in C_contrastive. D3 (AND formula) is a smaller magnitude change, but it determines symmetry and PSD-ness.
- **Global paper-faithful conceptors are well-posed.** The **per-step paper-faithful construction is ill-posed** with 15 episodes (N < d after pooling). This must be resolved (a documented choice) before any per-step paper-faithful steering run.
- Both the repository and paper matrices are stored for the later steering experiment. Neither replaces the other.
