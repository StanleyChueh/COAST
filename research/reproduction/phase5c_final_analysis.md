# Phase 5C Final Analysis: KS3 Oracle Reproduction, Interpretation and Next Step

Analysis only: no new rollouts and no code changes. Based on:
- `phase5c_ks3_oracle_sweep.md` and `.yaml`;
- `phase5c_selected_configs.json` (frozen, sha256 `3de5e4e2…8904`);
- `phase5c_test_results.csv`;
- Phases 5A and 5B, with context from Phases 1D, 2A, 2B, 3A, 4A and 4B.

Branch `exp/ks3-oracle-sweep` @ `daf8cbfb7bce815b11e6bd1eb37fca4398e96338`.

Paper: *Contrastive Conceptor Activation Steering (COAST)*, arXiv:2605.17144 v1.

## 1. Executive summary

- **We did not reproduce the paper on this task.** This is a **measured reproduction gap**, not a partial reproduction. We followed the paper's selection procedure: 15 fit rollouts, the paper's grid, one best configuration per strategy, then 30 test rollouts. Global scored **21/30** against the paper's 28/30. Positive-only scored **17/30** against 24/30. Per-step, restricted to α = 1.0, scored **15/30** against 26/30. The unsteered test baseline scored **17/30**.
- **Oracle selection did not close the gap.** It changed the global result from 19/30 (Phase 5A's fixed L5/α 0.5/β 0.1) to 21/30. That +2 is not significant (Fisher p = 0.79), and 21/30 is inside the range of our own unsteered runs on these states (17, 19, 21 out of 30).
- **Neither a wrong fixed configuration nor missing oracle selection explains the gap.** The most likely remaining sources are **evaluation-protocol details the paper does not specify**: which test states and seeds were used, run count and noise, how the Base column is defined, and which code version was used. For per-step there is also **one confirmed implementation difference**: the α axis is missing from the released code.
- **Recommended next step: Option B, prepare the author questions.** The conditions the decision tree sets for B are met. The released code was followed. The oracle procedure was followed for global and positive-only. The global result differs significantly from the paper. The remaining unknowns cannot be resolved from public information. The evidence does not support starting improvement work (C) yet. More reproduction runs (A) would mostly repeat a protocol we already know is under-specified. Nothing is sent to the authors in this step.
- **Not claimed:** that COAST fails, or that the paper's numbers are wrong.

## 2. Phase 5C results

### 2.1 Oracle-selected configurations

Each strategy was selected separately from the fit states 0–14 (`--seed 0`, 15 episodes). The rule was highest fit success; ties went to the first configuration in repository iteration order (layer ↑, α ↑, β ↑). The selection was frozen before any test rollout.

| Strategy | Configs searched | Selected layer | α | β | Fit success | Fit rate | Tie set at the max |
|---|---|---|---|---|---|---|---|
| global | 60 | **11** | **0.5** | **0.3** | 13/15 | 0.867 | unique |
| positive_only | 60 | **0** | **2.0** | **0.3** | 12/15 | 0.800 | 3 configs: L0 α2 β0.3; L0 α10 β0.1; L17 α1 β0.5 |
| per_step | 12 (α fixed 1.0) | **5** | **1.0** | **0.3** | 11/15 | 0.733 | 2 configs: L5 β0.3; L11 β0.3 |

Linear is optional and not a Table 4 column. It was interrupted at 12 of 20 configurations and excluded before any test rollout.

### 2.2 Test results (states 15–44, 30 episodes, one unseeded run each)

| Method | Config | Test success | Test rate (Wilson 95%) | Paper value (Table 4) | Gap | Fisher vs paper | Preregistered reading |
|---|---|---|---|---|---|---|---|
| Unsteered baseline | — | 17/30 | 0.567 (0.39–0.73) | — | — | — | — |
| Global | L11 α0.5 β0.3 | **21/30** | 0.700 (0.52–0.83) | **28/30** (L5 α0.5 β0.1) | −7 | 0.042 | inconsistent |
| Per-step | L5 α1.0 β0.3 | **15/30** | 0.500 (0.33–0.67) | **26/30** (L5 α10 β0.3) | −11 | 0.005 | inconsistent, **not like for like** (α 1 vs 10) |
| Positive-only | L0 α2.0 β0.3 | **17/30** | 0.567 (0.39–0.73) | **24/30** (L11 α1 β0.1) | −7 | 0.095 | inconclusive |

**Against our own test baseline:** global +4 (p = 0.42), positive-only 0 (p = 1.0), per-step −2 (p = 0.80). **None of the selected configurations is distinguishable from no steering.**

## 3. Comparison with the paper

| | Paper | Ours | Verdict |
|---|---|---|---|
| Global | 28/30 (0.93) | 21/30 (0.70) | **measured gap**, significant (p = 0.042) |
| Per-step | 26/30 (0.87) | 15/30 (0.50) at α = 1.0 | **gap, but protocol not matched** (α 10 not expressible in released code) |
| Positive-only | 24/30 (0.80) | 17/30 (0.57) | **gap in the same direction**, not significant at N = 30 (p = 0.095) |
| Strategy ranking | global > per-step > positive-only | global > positive-only > per-step | ranking not reproduced (none of our pairwise differences is significant) |
| Steering gain over Base | Base 0.53 → 0.93 | test baseline 0.57 → 0.70 (not significant) | **gain not reproduced** |
| Selected configs | L5 α0.5 β0.1 / L5 α10 β0.3 / L11 α1 β0.1 | L11 α0.5 β0.3 / L5 α1 β0.3 / L0 α2 β0.3 | only the per-step layer and β match. On our fit states the paper's global pick scores 9/15 (23 of 60 configs score higher) and its positive-only pick scores 8/15 (27 higher) |

**One thing does match: the paper's "Base".** Its KS3 Base is 0.53. Our unsteered rate on the fit states is 8/15 = 0.53 in Phase 1A, and 36/75 = 0.48 pooled over the five Phase 5C fit baselines. Our unsteered policy on this task therefore behaves as the paper's Base did. This is weak evidence against a gross checkpoint or environment mismatch; see §5.

## 4. Comparison with Phase 5A (fixed configuration)

| | Phase 5A (fixed, paper's config) | Phase 5C (oracle-selected) |
|---|---|---|
| Global config | L5 α0.5 β0.1 | L11 α0.5 β0.3 |
| How chosen | taken from the paper | argmax of 60 on fit states 0–14 |
| Test (states 15–44) | 19/30 | **21/30** |
| Unsteered baseline in the same session | 19/30 | 17/30 |
| Steered − baseline | 0 | +4 (p = 0.42) |

**Did oracle selection improve over the fixed configuration?** Only by +2 successes, 19 → 21 (Fisher p = 0.79). That difference is the size of run-to-run noise.

The global COAST runs we have on states 15–44 are:
- 20/30 (Phase 4A, fixed configuration);
- 19/30 (Phase 5A, fixed configuration);
- 21/30 (Phase 5C, oracle-selected).

Pooled, that is **60/90 = 0.67**. The unsteered runs on the same states scored 21, 19 and 17, pooled **57/90 = 0.63** (steered vs unsteered p = 0.76).

If the true global rate on these states is about 0.67–0.70, a result of 28/30 or better would happen with probability about 0.001–0.002. The pooled steered result differs from 28/30 at p = 0.004.

**Conclusion:** the Phase 5A gap was not caused by using the paper's stated configuration instead of an oracle-selected one.

## 5. Why the gap exists: causes ruled out, unlikely, and open

| Candidate cause | Status | Evidence |
|---|---|---|
| **Wrong fixed configuration** | **ruled out as the main cause** | The oracle-selected global config reaches 21/30, only +2 over the fixed config. The paper's config is mid-ranked on our fit states |
| **Missing oracle selection** | **ruled out** for global and positive-only; **partly open** for per-step | The paper's procedure was run with the released tool (grid as in the paper, per-strategy selection, fit-only selection, frozen before test). Per-step lacks the paper's α axis |
| **Selection noise (winner's curse)** | **explains the fit-to-test drop, not the gap** | Selected fit rates of 0.73–0.87 fall to 0.50–0.70 on test. The best fit scores match what equal-performing configs would produce by luck (P = 0.61, 0.95, 0.91). This affects the paper's procedure as well, but it would make the paper's numbers *more* optimistic, not ours pessimistic |
| **Implementation: conceptor builder V0 vs V3** | **unlikely at the tested config** (limited evidence) | Phase 1D: with paired noise, V0 and V3 gave identical outcomes on 15/15 states (L5 α0.5 β0.1). Tested for one config and 15 states only |
| **Implementation: per-step α axis** | **confirmed difference** (per-step column only) | Released builder is fixed at α = 1.0; the paper's KS3 per-step oracle uses α = 10 |
| **Implementation: what the intervention does** | **context, not a mismatch** | Phases 1E, 2A, 2B, 3A: at L5 β0.1, COAST is about a uniform 0.9·h shrink, and pure shrinkage and a spectrum-matched random conceptor reproduce its outcomes (14/15 and 15/15 states). This is what the released code does. Whether the paper's code behaves the same is unknown |
| **Evaluation protocol: test states and seeds** | **open, most likely contributor** | The paper gives no state IDs or seeds. Baseline difficulty varies strongly between state windows (8/15 to 13/15 across earlier phases) |
| **Evaluation protocol: Base definition** | **open** | Paper's Base equals the fit-set rate (Phase 4B F1), so its "gain" compares fit-set Base with test-set steering. This cannot by itself produce 28/30: none of our runs on these test states, steered or not, exceeded 21/30 |
| **Policy noise and run count** | **open** | Unseeded noise moves the same states by ±3/15 (Phase 1D: 7–13/15). Fit baselines here are over-dispersed (4–11/15, p = 0.035). The paper does not state whether cells are single runs or how noise was handled |
| **Code version** | **open** | `--seed` semantics changed with PR #48 (2026-04-23). The per-step α axis in the paper is absent from the release, which suggests the paper numbers came from a different code state |
| **Checkpoint identity** | **open, weakly constrained** | Same HF repo and revision as released. Our unsteered fit rate matches the paper's Base (0.53), but weights cannot be verified against the authors' |
| **Environment versions / hardware** | **open, low prior** | LIBERO `d63b117`, robosuite 1.4.0, MuJoCo 3.2.3 here; the paper does not state versions |

**Strongest evidence** (in order):
1. Across three independent sessions, every global COAST test on states 15–44 fell between 19 and 21 out of 30. That includes the paper's configuration twice and the oracle selection once. Each run sat within ±4 of an unsteered run from the same session, while the paper reports 28/30.
2. The paper's procedure, run faithfully with the released tool, selected configurations that did no better than the baseline on test.
3. Our unsteered fit-state rate matches the paper's Base, so the gap is in the *steered* number (or in which states it was measured on), not in the base policy.

**What remains uncertain:**
- The paper's fit and test state IDs and seeds, and therefore its LIBERO split.
- Whether Table 4 cells are single runs and how policy noise was handled.
- The code version and builder behind the paper numbers (V0 vs V3, and the per-step α axis).
- Checkpoint identity at the weight level.
- Environment versions.
- On our side, the numbers are single unpaired runs on a test window that earlier phases already used (15–44). N = 30 cannot distinguish rates closer than about ±0.17.

## 6. Recommended next step

Checking each option against the evidence:

- **Option A, continue reproduction.** Worthwhile only if *important protocol gaps remain that we can close ourselves*.
  - The largest open items (test states, seeds, run count, Base definition, code version) cannot be settled by running more experiments. Each new run would be another guess at an unknown protocol.
  - The one closable gap, the per-step α axis, needs a source change (Phase 5B C1). It touches one Table 4 column only.
  - More replicates of our own runs would shrink our error bars. They cannot move a pooled 0.67 to 0.93.
  - **Not recommended as the next step.**
- **Option B, prepare the author questions.** The decision tree's conditions are all met:
  - released code followed ✓;
  - oracle procedure followed ✓ (global and positive-only; per-step documented as partial);
  - result differs significantly ✓ (global p = 0.042 alone; p = 0.004 pooled over our global runs).
  - The dominant uncertainties are items only the authors can answer.
  - **Recommended.**
- **Option C, COAST improvement.** Requires that the reproduction investigation be *sufficiently complete*. It is not. The claim we would be improving on is not pinned down: we do not know the evaluation protocol that produced 28/30. In our setup the gain is not distinguishable from no steering, and the tested configuration behaves like uniform shrinkage. An "improvement" measured under our protocol could not be compared with the paper's numbers. **Not yet.**

**Recommendation: Option B.** Update the Phase 4B §9 draft questions with the Phase 5C evidence and reorder them. Suggested priority:

1. **Test and fit states and seeds.** Which LIBERO init-state indices and seeds were used for the 15 fit rollouts and the 30 test rollouts on KS3? Were test states disjoint from fit states? Pre- or post-PR #48 seeding?
2. **Oracle procedure details.** How was fit success per configuration measured? Were there steered re-rollouts on the 15 fit states, how many episodes, and was noise seeded? What was the tie rule? Was each Table 4 cell a single 30-episode run?
3. **Code version and builder.** Which commit produced Table 4? Were conceptors built with the released `compute_conceptors.py` (V0) or the App. A.9.1 pipeline (V3)? How was per-step α = 10 produced when the release fixes it at 1.0?
4. **Base definition.** Is Base the 15 fit rollouts or a separate unsteered test run?
5. **Checkpoint and environment.** Are the exact weights `brandonyang/openpi-libero-2000` @ `aaeeabc72f8a…`? What are the LIBERO, robosuite and MuJoCo versions?

Include our numbers in the message (§2.2, §4), stated as a measured gap under a documented interpretation.

**Low-cost work that fits alongside B** (offline, no rollouts, optional): Phase 4B §10(a), rebuilding conceptors from the authors' public KS3 fit activations and comparing them with ours. This separates fit-data differences from builder differences.

**Do not start** improvement design, and do not contact the authors in this step. Sending is a separate decision.

## 7. Files

- Added: `research/reproduction/phase5c_final_analysis.md` (this file).
- No experiment was run, no code or other file was modified, and nothing was committed.
