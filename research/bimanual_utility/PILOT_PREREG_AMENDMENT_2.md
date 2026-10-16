# Pilot preregistration, amendment 2 (2026-10-17, before any rollout)

Amends gate **D for C1** in [`PILOT_PREREG.md`](PILOT_PREREG.md) §9. Written before any simulator, planner or VLA output exists; found by the synthetic tests of the analysis code.

## The defect (a design bug, demonstrated on synthetic data)

Prereg §9 requires, for C1: "D_cond − D_same ≥ 0.05 (CI excludes 0)", where
- `D_cond = P(Y(VCI, s, m) ≠ Y(V2, s, m))` uses the **same** noise key m (common random numbers);
- `D_same = P(Y(V2, s, m) ≠ Y(V2, s, m'))` uses **different** keys.

These are not comparable:

1. **Under common random numbers**, a genuine clamp effect δ produces `D_cond ≈ |δ|` (only outcomes the clamp changes disagree). `D_same = 2p(1−p)` is about 0.5 near p = 0.5. A real +20 pp effect therefore *fails* the criterion. Test: `test_disagreement_difference_is_not_a_directional_test` (`D_cond` 0.20 < `D_same` 0.49).
2. **Even with independent keys**, `D_cond − D_same = δ(1 − 2p)`, which is ≈ 0 near p = 0.5 regardless of δ.

A disagreement difference measures *complementarity*: whether conditions succeed on different draws. That was the right question for the adaptive-inference router. It is not a test of a directional mean effect, which is what C1 is.

## The correction

Gate D for C1 becomes both of the following:
- (i) `|Δ_clamp|` > 97.5th percentile of the bootstrap distribution of `|Δ_null|`. **Unchanged.** This is the placebo: the "effect" between two halves of the same condition's noise replicates.
- (ii) **Per-replicate consistency (new):** Δ computed separately within each noise master m = 1…4 (pooled R1 ∪ R2 scenes) has the same, non-zero sign as Δ_clamp in all four. A clamp effect that appears under one noise draw but not the others is re-sampling, not a condition effect. Test: `test_effect_in_one_master_only_fails_d`.

`D_same`, `D_cond` (matched key) and `D_cond` (independent keys) are still computed and reported, descriptively.

All other thresholds, gates (A–G), N, R and tasks are unchanged. C1's own pass rule (|Δ| ≥ 10 pp, sign-flip p < 0.01, same sign in R1 and R2) is unchanged.
