# Phase 7A — Pre-registration (task selection and analysis plan)

- Written: 2026-09-25, **before any Phase 7A rollout**. No Phase 7A outcome existed when this file was written.
- Branch: `exp/cross-task-protocol-validation`, created at base `ea7bfb0cf89611ee685e801d6690f031e5d2942a`. The branch did not exist, so it was created from the clean HEAD, which equals the base.
- Paper: arXiv:2605.17144 v1 HTML (the copy fetched in Phase 6C). All values were copied from the paper's printed decimals, with no inference.

## 1. Source values (paper, π0.5, LIBERO-10, paper table order)

| Order | Task (Table 4 / Table 1 name) | libero_10 task_id | Table 21 fit S / F | Table 1 Base | Table 1 +Glob. (= Table 4 Global) | Table 4 Global config | Table 17 (cross-check) |
|---|---|---|---|---|---|---|---|
| 1 | KS3 Stove+Moka | 2 | 8 / 7 | 0.53 | 0.93 | L5 α0.5 β0.1 | 5 / 0.5 / 0.1 |
| 2 | KS4 Bowl+Drawer | 3 | 6 / 9 | 0.40 | 0.73 | L5 α1.0 β0.1 | 5 / 1.0 / 0.1 ✓ |
| 3 | KS6 Mug+Micro | 9 | 2 / 13 | 0.13 | 0.40 | L11 α0.1 β0.3 | 11 / 0.1 / 0.3 |
| 4 | KS8 Two Mokas | 8 | 3 / 12 | 0.20 | 0.47 | L5 α1.0 β0.1 | 5 / 1.0 / 0.1 |
| 5 | LR1 Soup+Cheese | 7 | 12 / 3 | 0.80 | 0.93 | L11 α0.5 β0.3 | 11 / 0.5 / 0.3 ✓ |
| 6 | LR2a Soup+Tomato | 0 | 6 / 9 | 0.40 | 0.80 | L11 α0.5 β0.3 | 11 / 0.5 / 0.3 ✓ |
| 7 | LR2b Cheese+Butter | 1 | 9 / 6 | 0.60 | 0.93 | L5 α0.5 β0.1 | 5 / 0.5 / 0.1 |
| 8 | LR5 Mugs+Plates | 4 | 1 / 14 | 0.07 | 0.60 | L11 α0.5 β0.5 | 11 / 0.5 / 0.5 |
| 9 | LR6 Mug+Choc | 6 | 8 / 7 | 0.53 | 0.87 | L5 α0.5 β0.1 | 5 / 0.5 / 0.1 |
| 10 | ST1 Book+Caddy | 5 | 10 / 5 | 0.67 | 0.93 | L5 α0.5 β0.1 | 5 / 0.5 / 0.1 |

The task_id mapping is taken from `LIBERO_TASK_IDS` in the candidate driver (`experiments/pi05_libero/src/conceptor_steering.py` @ `29059a5`).

## 2. Eligibility (historical builder `MIN_PER_CLASS = 3`, Table 21 counts)

- **Excluded by rule:** KS3 (already studied).
- **Ineligible:** KS6 (2 successes), LR5 (1 success).
- **Eligible (7):** KS4, KS8, LR1, LR2a, LR2b, LR6, ST1.

## 3. Paper Global − Base (printed decimals)

| Task | Base | Global | Global − Base |
|---|---|---|---|
| KS4 | 0.40 | 0.73 | 0.33 |
| KS8 | 0.20 | 0.47 | 0.27 |
| LR1 | 0.80 | 0.93 | **0.13** (min) |
| LR2a | 0.40 | 0.80 | **0.40** (max) |
| LR2b | 0.60 | 0.93 | 0.33 |
| LR6 | 0.53 | 0.87 | 0.34 |
| ST1 | 0.67 | 0.93 | 0.26 |

- Sorted: 0.13, 0.26, 0.27, 0.33, 0.33, 0.34, 0.40.
- **Median = 0.33.**
- Closest to the median: KS4 and LR2b, both at distance 0. The tie is broken by paper table order, which gives **KS4**.
- **Robustness:** reading the decimals as k/15 (Phase 6C) gives KS4 = LR2b = LR6 = 5/15 and median 5/15. The tie-break still gives KS4. A and C are unchanged.

## 4. Selected tasks (frozen)

| Slot | Task | libero_10 task_id | Full name | Fit S / F | Paper Base | Paper Global | Paper Global config |
|---|---|---|---|---|---|---|---|
| **A** (largest gain) | LR2a | 0 | `LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket` | 6 / 9 | 0.40 | 0.80 | **L11, α 0.5, β 0.3** |
| **B** (median gain) | KS4 | 3 | `KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it` | 6 / 9 | 0.40 | 0.73 | **L5, α 1.0, β 0.1** |
| **C** (smallest gain) | LR1 | 7 | `LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket` | 12 / 3 | 0.80 | 0.93 | **L11, α 0.5, β 0.3** |

Paper values as implied 15-episode counts (Phase 6C: all Table 4 cells are k/15). This is an interpretation, shown next to the printed decimal:

| Task | Paper Base | Paper Global |
|---|---|---|
| LR2a | 0.40 ≈ 6/15 (= Table 21 fit 6/9) | 0.80 ≈ 12/15 |
| KS4 | 0.40 ≈ 6/15 (= fit 6/9) | 0.73 ≈ 11/15 |
| LR1 | 0.80 ≈ 12/15 (= fit 12/3) | 0.93 ≈ 14/15 |

## 5. Conditions, arms and repeats (fixed before any rollout)

**Conditions** (the candidate driver's native set for one Global cell):
- `baseline`;
- `global_L{l}_a{α}_b{β}` (PAPER_GLOBAL);
- `random_L{l}_b{β}` (RANDOM_CONTROL): the candidate's `compute_random_conceptor(seed = l·100 + int(10β))` at the same layer and β.

**Arms:**
- **H:** 15 episodes, init states 0–14, via the unmodified candidate client (`initial_states[episode]`).
- **T:** 30 episodes, init states 15–44, via a state-selection-only client wrapper.
- Both arms use the same conceptor NPZ, driver, server, hook and policy code.

**Repeats:** 2 independent driver invocations per (task, arm), with unseeded policy noise and no early stopping.

**Totals:** 3 tasks × (3 × 15 × 2 + 3 × 30 × 2) = **810 episodes**.

**Conceptors:** built once by the candidate's `for_subin/build_conceptors.py` (defaults) from the authors' released fit activations (`brandonyang/pi05-libero-activations-v1-2000-15env` @ `3e3a8fe2`). Never rebuilt, never fit on held-out data.

**No tuning.** No configuration is chosen from our results.

## 6. Analysis plan (fixed)

Per task:
- **(A)** PAPER_GLOBAL vs BASELINE in Arm H.
- **(B)** the same in Arm T.
- **(C)** PAPER_GLOBAL vs RANDOM_CONTROL in both arms.
- Report per-repeat and pooled raw counts, Wilson 95% intervals and Fisher exact p (descriptive).
- Report between-repeat differences.
- Pooled episodes re-use the same init states, so they are **not independent**. We also report a state-level paired view: success count per state across repeats.

Metrics:
- G_H = SR_COAST,H − SR_BASE,H
- G_T = SR_COAST,T − SR_BASE,T
- **PGG = G_H − G_T**, computed from pooled rates.

Case rule, applied as written in the Phase 7A brief:
- **A:** in most tasks, G_H > G_T and COAST is not clearly better than baseline or random in Arm T.
- **B:** mixed.
- **C:** COAST consistently beats baseline and random in Arm T.
- "Clearly better" means a pooled Fisher p < 0.05 **and** the same sign in both repeats.

Mechanistic diagnostics are exploratory. They are computed only after the success-rate records are frozen.
