# Phase 7A — Cross-Task Protocol Validation (π0.5, LIBERO-10)

**Question:** does the KS3 pattern generalize to other LIBERO-10 tasks? On KS3, the historical fit-state evaluation looked high, while disjoint held-out states showed no COAST advantage.

| | |
|---|---|
| Branch | `exp/cross-task-protocol-validation`. It did not exist, so it was created at the base `ea7bfb0cf89611ee685e801d6690f031e5d2942a` (clean HEAD = base) |
| Paper | arXiv:2605.17144 v1 |
| Candidate code | `upstream/miranda-v2` @ `29059a537ce123abdab7a4e164db0c72eeff8024`, worktree `../COAST-paper-candidate`, unmodified |
| Records | `experiments/phase7a_episode_results.csv` (810 rows), `phase7a_summary.csv`, `phase7a_analysis.json`, `phase7a_operator_diagnostics.csv`, `phase7a_cross_task_validation.yaml` |

Evidence tags: **[P]** paper, **[M]** measured, **[I]** inference.

## 1. Executive summary

- **Tasks.** Three tasks were chosen from paper values alone, before any rollout (§2):
  - A = LR2a (largest paper gain, +0.40);
  - B = KS4 (median gain, +0.33);
  - C = LR1 (smallest gain, +0.13).
- **Design.** Each task ran BASELINE, PAPER_GLOBAL (the paper's exact configuration) and RANDOM_CONTROL. Everything used the historical candidate pipeline and the authors' own fit activations.
  - Arm H: states 0–14.
  - Arm T: states 15–44.
  - Two repeats per arm; **810 episodes**.
  - The same conceptor and the same steering code were used in both arms. Only the state window differs.
- **Held-out (Arm T).** The paper Global configuration was **not clearly better than baseline or random in any task**.

  | Task | Global vs baseline | Global vs random |
  |---|---|---|
  | LR2a | +0.07 (p = 0.58) | −0.05 |
  | KS4 | +0.08 (p = 0.46) | +0.10 |
  | LR1 | +0.13 (p = 0.17) | −0.05 |

  Pooled over the three tasks, Global and Random are **identical: 109/180 each**. Both sit above baseline at 92/180 (+0.094, CMH p = 0.086).
- **Historical (Arm H).** The paper's Global values were not reached in single-configuration runs.

  | Task | Ours (pooled) | Paper (as /15) | Fisher p |
  |---|---|---|---|
  | LR2a | 17/30 | 12/15 | 0.19 |
  | KS4 | 14/30 | 11/15 | 0.12 |
  | LR1 | 23/30 | 14/15 | 0.24 |

  One repeat (LR1, repeat 2) scored 15/15, and its twin scored 8/15.
- **Protocol Generalization Gap.** PGG = G_H − G_T was **+0.03 (LR2a), −0.15 (KS4), +0.13 (LR1)**; mean +0.006. The fit-state → held-out degradation is **not consistent**. It is within noise at these sample sizes.
- **Pre-registered classification: Case A** (G_H > G_T in 2/3 tasks, and no clear held-out advantage in 3/3). The robust, general part is not a protocol gap. It is that **the paper-selected conceptor does no better than a spectrum-free random conceptor at the same layer and β on held-out states**, in every task. Together with KS3 that makes 4 of 4 studied tasks.
- **Exploratory mechanism.** Every paper Global operator is contraction-dominated: cos(Δh, h) = −0.998 to −0.9996, ‖Δh‖/‖h‖ ≈ β(1 − hᵀCh/‖h‖²). This matches KS3 (Phase 1E). The random control is also a near-uniform shrink (cos −0.988).
- **Decision:** a **general limitation** (not KS3-specific) of the conceptor-specific component under our protocol. The protocol-sensitivity magnitude is task-dependent and noisy. A mechanism-first Phase 7B is justified (§14).

## 2. Pre-registered task selection

The full calculation is in `phase7a_preregistration.md` (sha256 `08293552…6639b5a`, frozen 16:44 +08:00, before any Phase 7A rollout).

- **Eligibility** used the paper's Table 21 fit counts (S ≥ 3 and F ≥ 3), with KS3 excluded.
  - Eligible: KS4, KS8, LR1, LR2a, LR2b, LR6, ST1.
  - Ineligible: KS6 (2 S) and LR5 (1 S).
- **Paper Global − Base** (Table 1): KS4 0.33, KS8 0.27, LR1 0.13, LR2a 0.40, LR2b 0.33, LR6 0.34, ST1 0.26.
- **Median** 0.33. KS4 and LR2b tie at distance 0, and paper table order picks **KS4**.
- **Robustness:** reading the values as k/15 gives the same A/B/C.

| Slot | Task | task_id | Fit S/F (paper = authors' data, verified) | Paper Base | Paper Global | Paper config (Table 4 = Table 17) |
|---|---|---|---|---|---|---|
| A | LR2a (alphabet soup + tomato sauce) | 0 | 6/9 | 0.40 | 0.80 | L11, α 0.5, β 0.3 |
| B | KS4 (black bowl in bottom drawer) | 3 | 6/9 | 0.40 | 0.73 | L5, α 1.0, β 0.1 |
| C | LR1 (alphabet soup + cream cheese) | 7 | 12/3 | 0.80 | 0.93 | L11, α 0.5, β 0.3 |

## 3. Exact experimental protocol

- **Conceptors.**
  - Built with the candidate `for_subin/build_conceptors.py` (defaults) from `brandonyang/pi05-libero-activations-v1-2000-15env` @ `3e3a8fe2`.
  - Built once per task, from a single-task input directory. The builder treats tasks independently, so the per-task arrays equal those of a joint build.
  - The builder's class counts equal paper Table 21 for all three tasks.
  - Never rebuilt, never fit on held-out data.
- **Steering.** This is the candidate driver's native condition set for one Global cell:
  - `baseline`;
  - `global_L{l}_a{α}_b{β}`, with h′ = h·((1−β)I + βC)ᵀ on action-expert layer l, all tokens, all 10 denoising steps;
  - `random_L{l}_b{β}`, the candidate's `compute_random_conceptor(seed = 100l + int(10β))`.
- **Arm H.** 15 episodes on states 0–14. This is the unmodified candidate client, which uses `initial_states[episode]`.
- **Arm T.** 30 episodes on states 15–44, through `tools/phase7a_state_window_client.py`. That helper wraps the unmodified client and changes one thing:
  - `get_task_suite(...).get_task_init_states(task_id)` returns `states[15:]`, so episode k runs state 15 + k.
  - Verified offline: `shifted[k] == original[15+k]` for task_ids 0, 3 and 7.
  - Verified at runtime: the `phase7a_state_map` log lines (with a state hash) were checked for every Arm-T condition.
- **Driver helper.** `tools/phase7a_candidate_driver.py` imports the unmodified candidate driver and replaces only `run_single_task_eval`. The replacement:
  - picks the client: the original `main.py` for Arm H, the state-window wrapper for Arm T;
  - keeps the client log so per-episode outcomes can be recovered.

  Model loading, NPZ lookup, the hook, the server, the random conceptor, the condition order and `summary.json` all come from the candidate module.
- **Identical in both arms.** NPZ, policy code, `env.seed(7)`, `np.random.seed(7)`, 520 max steps, replan every 5 steps, unseeded flow-matching noise, checkpoint `openpi-libero-2000` (HF rev `aaeeabc…`).
- **Repeats and GPUs.** Two independent driver invocations per (task, arm). Repeat 1 ran on GPU 0 (ports 8800–8805) and repeat 2 on GPU 1 (ports 8900–8905). Each invocation had its own output directory. Wall clock: 17:37–19:34.
- **Per-episode data.**
  - Success comes from the client log.
  - init_state comes from the state map (Arm T) or equals k (Arm H).
  - rollout_steps = video frames − 10.
- **Cross-checks, all passed:**
  - the log-derived rate equals the driver's `summary.json` for all 36 cells;
  - video length is consistent with success for all 810 episodes;
  - Arm-T maps are exactly 15–44.
- **Incidents, none affecting rollouts:**
  - HF download stalls were replaced by a stall-detecting curl fetch; completeness was verified against the per-episode step counts.
  - My wrapper's temp-file rename failed on the first launch (`np.savez` appends `.npz`). It was fixed before any rollout.
- **Code hygiene.** The two runtime helpers are kept byte-identical to what ran. Their remaining ruff findings are cosmetic: redundant `noqa` markers and no explicit `check=` on `subprocess.run`, which matches the candidate original. The analysis and diagnostics scripts pass ruff. Re-running them after the lint fixes reproduced the frozen outputs byte-for-byte.

## 4. Paper-reported configurations

These are the paper's Global configurations, verified in Table 4 and Table 17:
- LR2a: L11, α 0.5, β 0.3.
- KS4: L5, α 1.0, β 0.1.
- LR1: L11, α 0.5, β 0.3.

No other configuration was run. No tuning was done.

## 5. Historical-arm results (states 0–14; 15 per repeat)

| Task | Condition | Rep 1 | Rep 2 | Pooled | Wilson 95% | States 2/1/0 of 2 |
|---|---|---|---|---|---|---|
| LR2a | BASELINE | 7 | 7 | 14/30 (0.47) | 0.30–0.64 | 3/8/4 |
| LR2a | PAPER_GLOBAL | 10 | 7 | 17/30 (0.57) | 0.39–0.73 | 5/7/3 |
| LR2a | RANDOM_CONTROL | 9 | 7 | 16/30 (0.53) | 0.36–0.70 | 5/6/4 |
| KS4 | BASELINE | 8 | 8 | 16/30 (0.53) | 0.36–0.70 | 4/8/3 |
| KS4 | PAPER_GLOBAL | 8 | 6 | 14/30 (0.47) | 0.30–0.64 | 2/10/3 |
| KS4 | RANDOM_CONTROL | 10 | 7 | 17/30 (0.57) | 0.39–0.73 | 4/9/2 |
| LR1 | BASELINE | 8 | 7 | 15/30 (0.50) | 0.33–0.67 | 3/9/3 |
| LR1 | PAPER_GLOBAL | **8** | **15** | 23/30 (0.77) | 0.59–0.88 | 8/7/0 |
| LR1 | RANDOM_CONTROL | 9 | 10 | 19/30 (0.63) | 0.46–0.78 | 6/7/2 |

"States 2/1/0" is the number of init states that succeeded in both repeats, one repeat, or neither.

## 6. Held-out-arm results (states 15–44; 30 per repeat)

| Task | Condition | Rep 1 | Rep 2 | Pooled | Wilson 95% | States 2/1/0 of 2 |
|---|---|---|---|---|---|---|
| LR2a | BASELINE | 13 | 18 | 31/60 (0.52) | 0.39–0.64 | 8/15/7 |
| LR2a | PAPER_GLOBAL | 20 | 15 | 35/60 (0.58) | 0.46–0.70 | 11/13/6 |
| LR2a | RANDOM_CONTROL | 18 | 20 | 38/60 (0.63) | 0.51–0.74 | 12/14/4 |
| KS4 | BASELINE | 12 | 13 | 25/60 (0.42) | 0.30–0.54 | 5/15/10 |
| KS4 | PAPER_GLOBAL | 15 | 15 | 30/60 (0.50) | 0.38–0.62 | 7/16/7 |
| KS4 | RANDOM_CONTROL | 10 | 14 | 24/60 (0.40) | 0.29–0.53 | 5/14/11 |
| LR1 | BASELINE | 17 | 19 | 36/60 (0.60) | 0.47–0.71 | 12/12/6 |
| LR1 | PAPER_GLOBAL | 22 | 22 | 44/60 (0.73) | 0.61–0.83 | 19/6/5 |
| LR1 | RANDOM_CONTROL | 25 | 22 | 47/60 (0.78) | 0.66–0.87 | 19/9/2 |

**Between-repeat variation.**
- 17 of 18 repeat pairs differ by 0–5 successes (Fisher p ≥ 0.29).
- The exception is LR1 H PAPER_GLOBAL: 8/15 vs 15/15 (p = 0.006, one of 18 tests; not significant after Bonferroni, p = 0.11).
- Repeat 1 (GPU 0) and repeat 2 (GPU 1) totals are 229 vs 232 of 405, so there is no GPU or repeat effect.

**Independence.** Pooled counts are **not independent draws**: both repeats use the same init states. Pooled p-values are descriptive, and the state-level 2/1/0 columns show the paired structure.

## 7. Baseline and random comparisons (Fisher exact, two-sided; descriptive)

| Task | Arm | Global − Base (pooled) | Per repeat | p (pooled) | Per-repeat p | Global − Random | p |
|---|---|---|---|---|---|---|---|
| LR2a | H | +0.100 | +0.20 / 0.00 | 0.61 | 0.46 / 1.0 | +0.033 | 1.0 |
| LR2a | T | +0.067 | +0.23 / −0.10 | 0.58 | 0.12 / 0.60 | −0.050 | 0.71 |
| KS4 | H | −0.067 | 0.00 / −0.13 | 0.80 | 1.0 / 0.72 | −0.100 | 0.61 |
| KS4 | T | +0.083 | +0.10 / +0.07 | 0.46 | 0.60 / 0.80 | +0.100 | 0.36 |
| LR1 | H | +0.267 | 0.00 / +0.53 | 0.060 | 1.0 / 0.002 | +0.133 | 0.40 |
| LR1 | T | +0.133 | +0.17 / +0.10 | 0.17 | 0.28 / 0.58 | −0.050 | 0.67 |

By the pre-registered rule, a clear advantage needs pooled p < 0.05 **and** the same sign in both repeats. **No cell meets it.**

**Cross-task pooled** (Mantel–Haenszel over tasks):

| Arm | Global | Baseline | Random | Global vs Baseline | Global vs Random |
|---|---|---|---|---|---|
| H | 54/90 | 45/90 | 52/90 | +0.100 (p = 0.23) | +0.022 (p = 0.88) |
| T | **109/180** | 92/180 | **109/180** | +0.094 (p = 0.086) | **0.000** (p = 0.91) |

## 8. Protocol Generalization Gap

G = SR(Global) − SR(Baseline), pooled over repeats; PGG = G_H − G_T.

| Task | G_H | G_T | **PGG** | Same, vs random (G_H − G_T) |
|---|---|---|---|---|
| LR2a | +0.100 | +0.067 | **+0.033** | +0.083 |
| KS4 | −0.067 | +0.083 | **−0.150** | −0.200 |
| LR1 | +0.267 | +0.133 | **+0.133** | +0.183 |
| Mean | | | **+0.006** | +0.022 |

**Reading.**
- PGG > 0 in 2 of 3 tasks, but it is small for LR2a and reversed for KS4.
- The largest positive PGG (LR1) is driven by a single 15/15 repeat.
- In these data, fit-state → held-out degradation is **protocol-sensitive in some tasks, not systematic**. We do not call it overfitting.

**Comparison with the paper** (the paper's decimals as implied /15 counts, per Phase 6C):

| Task | Paper Base | Paper Global | Paper config | Historical ours (Global) | Held-out ours (Global) | Baseline H | Baseline T | Random H | Random T |
|---|---|---|---|---|---|---|---|---|---|
| LR2a | 0.40 (6/15 = fit data) | 0.80 (12/15) | L11 α0.5 β0.3 | 10, 7 → 17/30 | 20, 15 → 35/60 | 14/30 | 31/60 | 16/30 | 38/60 |
| KS4 | 0.40 (6/15 = fit data) | 0.73 (11/15) | L5 α1.0 β0.1 | 8, 6 → 14/30 | 15, 15 → 30/60 | 16/30 | 25/60 | 17/30 | 24/60 |
| LR1 | 0.80 (12/15 = fit data) | 0.93 (14/15) | L11 α0.5 β0.3 | 8, 15 → 23/30 | 22, 22 → 44/60 | 15/30 | 36/60 | 19/30 | 47/60 |
| KS3 (Phase 6C, H only) | 0.53 (8/15) | 0.93 (14/15) | L5 α0.5 β0.1 | 13, 12 → 25/30 | — (same-builder held-out not run) | 33/45 | — | 23/30 | — |

**Baseline plausibility.** Our Arm-H baselines agree with the authors' fit-data rates on the same states: LR2a 14/30 vs 6/15 (p = 0.76), KS4 16/30 vs 6/15 (p = 0.53), LR1 15/30 vs 12/15 (p = 0.063). The last is the weakest agreement; see §13.

## 9. Cross-task mechanistic diagnostics (exploratory; computed after the records were frozen)

Offline, on the authors' fit token vectors (every 10th inference step, all denoising steps). File: `experiments/phase7a_operator_diagnostics.csv`.

| Task | Operator | ‖Δh‖/‖h‖ | cos(Δh, h) | Quota tr(C)/d | hᵀCh/‖h‖² | Overlap(C_s, C_f) | Held-out G_T vs base / vs random |
|---|---|---|---|---|---|---|---|
| KS3 | Global L5 α0.5 β0.1 | 0.097 | −0.9996 | 0.0041 | 0.034 | 0.993 | (not run with this builder) |
| LR2a | Global L11 α0.5 β0.3 | 0.289 | −0.9987 | 0.0126 | 0.040 | 0.951 | +0.07 / −0.05 |
| KS4 | Global L5 α1.0 β0.1 | 0.099 | −0.9995 | 0.0082 | 0.015 | 0.984 | +0.08 / +0.10 |
| LR1 | Global L11 α0.5 β0.3 | 0.287 | −0.9979 | 0.0168 | 0.046 | 0.947 | +0.13 / −0.05 |
| — | Random L5 β0.1 | 0.083 | −0.989 | 0.175 | 0.175 | — | |
| — | Random L11 β0.3 | 0.250 | −0.988 | 0.180 | 0.176 | — | |

**Reading (exploratory; n = 3 tasks):**
- In every task, the paper's Global operator is ≈ a uniform (1 − β)·h shrink. Less than 5% of the token energy lies inside C (hᵀCh/‖h‖² ≤ 0.046).
- The random control has more energy inside its C (≈ 0.18). So it is a slightly *weaker* shrink, but it is still almost parallel to −h.
- There is no visible link between directional content and held-out gain. The task with the largest held-out gain vs baseline (LR1) also has the most negative gain vs random. The KS4 Global − Random gap (+0.10) sits with the most contraction-dominated of the three new operators.

## 10. What generalizes from KS3

- **No held-out advantage over the matched random conceptor.** On held-out states, the paper Global configuration is indistinguishable from a random conceptor at the same layer and β. This held in 3/3 new tasks, and pooled Global = Random (109/180 each). On KS3 (Phases 2B, 5A and 5C), COAST ≈ random ≈ shrinkage.
- **No clear held-out advantage over baseline** in any single task. There is a small nonspecific positive difference when pooled (+0.094, p = 0.086) that random steering shares.
- **Contraction-dominated operators** at the paper's configurations: cos(Δh, h) ≈ −1 in 4/4 tasks.
- **The paper's single-configuration values are not reached** on the fit states in typical runs (3/3 new tasks, pooled). As on KS3, one run can occasionally reach them (LR1 repeat 2: 15/15). This is consistent with, but does not prove, reporting that selects high draws.

## 11. What does not generalize

- **A consistent fit-state → held-out drop** (a positive PGG in every task) was not seen. PGG is +0.03, −0.15 and +0.13, and KS4 shows the opposite sign.
- **KS3's "near-paper value under historical semantics" (12–13/15 vs 14/15)** did not recur. LR2a and KS4 stay well below their paper Global values on states 0–14; LR1 reaches it only in one of two repeats.
- **Held-out states are not uniformly easier.** Held-out baselines were higher for LR2a (0.52 vs 0.47) and LR1 (0.60 vs 0.50), but lower for KS4 (0.42 vs 0.53). Differences between state windows are task-specific.

## 12. Implications for a future COAST improvement method (no method implemented here)

- At the paper's configurations, the operator acts mostly as a norm shrink. The conceptor's subspace contributes little energy (hᵀCh/‖h‖² of about 1.5–4.6%). This is a concrete, measured target for a later mechanism phase.
- Any future method must beat **both baseline and the matched random or shrinkage control on disjoint held-out states**, with repeats. Beating a fit-state oracle is not enough, given the between-repeat spread seen here (up to 7 of 15 episodes).
- Sample size: with 60 episodes per arm, only differences of about ≥ 0.2–0.25 are reliably detectable. A later phase should pre-register a power target, e.g. more states or tasks, or seeded paired noise as in Phase 1D.

## 13. Limitations

- **Scope.** Three new tasks plus KS3; one checkpoint; one configuration per task (the paper's); 2 repeats. Per-task power is low. Case A rests partly on "no significant difference", which is not evidence of no effect.
- **Non-independence.** Repeats share init states, so pooled counts overstate the effective sample size. The per-state columns show the paired structure.
- **Candidate pipeline, not main.** Global conceptors come from the candidate public builder: pooled, centered, `C_s(I−C_f)`, denoising step 0 only. The exact 2026-04-13 builder is not public (Phase 6C). Results may differ with the paper's text builder or with main's builder. Using the same builder in both arms keeps the protocol comparison clean.
- **Unseeded noise.** Arms and conditions are not noise-paired, which contributes the repeat spread. It matches the historical pipeline.
- **Repeat 1 ran entirely on GPU 0 and repeat 2 on GPU 1**, so repeat and GPU are confounded. The totals (229 vs 232) show no effect.
- **LR1 fit-data agreement is weakest** (authors' 12/15 vs our 15/30, p = 0.063). A small environment or checkpoint difference cannot be excluded. The checkpoint identity is unverified (Phase 6C).
- **KS3 has no same-builder held-out arm.** Its held-out evidence comes from main's builder (Phases 3A–5C). KS3 is therefore excluded from the PGG table.
- **Mechanistic diagnostics are exploratory.** They are offline, use fit-state activations only, and cover n = 3 tasks.

## 14. Is Phase 7B / method development justified?

**Decision: general limitation, not KS3-specific.**
- In all four tasks studied, the paper's Global configuration gives no held-out advantage over a matched random conceptor.
- Its operator is a near-uniform contraction.
- The historical-vs-held-out protocol gap is **task-dependent and noisy**, not a consistent effect.

**Phase 7B is justified as a mechanism-first phase, not yet as a claim of improvement.**
- **Target the contraction finding.** Is a directional, norm-preserving component what is missing?
- **Use the same held-out protocol.** Disjoint states 15–44, repeats, random and shrinkage controls, and a pre-registered power target.
- **Do not generalize further than the evidence.** Keep the claim to "under our protocol and checkpoint". The author's reply on the paper's protocol (pending) may change how the paper's numbers should be read.

No new steering method was introduced in Phase 7A.
