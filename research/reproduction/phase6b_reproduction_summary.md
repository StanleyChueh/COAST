# COAST KS3 Reproduction — Summary

A short summary of our attempt to reproduce the KS3 result in Table 4 of *Contrastive Conceptor Activation Steering (COAST)* (arXiv:2605.17144 v1). It accompanies the questions in `phase6b_author_questions.md`.

**In brief:** we ran the released implementation under the protocol documented below. The result on this task falls short of the reported value, so there is a reproduction gap. We think the most likely cause is a difference in evaluation details the paper does not specify. We are asking for those details. We are not suggesting that the method or the paper is incorrect.

## 1. Setup

| Item | Our setting |
|---|---|
| Model | π0.5, config `pi05_libero`, PyTorch inference path |
| Checkpoint | `openpi-libero-2000` (step 2,000), from HF `brandonyang/openpi-libero-2000`, revision `aaeeabc72f8a50a8fa2d04544332c8ec1cd0142e` |
| Task | LIBERO-10 task 2, `KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it` (KS3); 520 max steps, replan every 5 steps |
| Code | `COAST-VLA/COAST` `main` @ `2afa10ee256a3b3edfeb56500fea166a0837f119`. The default steering path and conceptor builder are unmodified |
| Conceptors | Released builder (`experiments/libero/compute_conceptors.py`, defaults) |
| Fit set | 15 rollouts, `--seed 0` → LIBERO init states 0–14 (8 success / 7 failure, the same counts as Table 21) |
| Test set | 30 rollouts, `--seed 15` → init states 15–44 (our interpretation; the paper does not list state IDs) |
| Policy noise | Released default (unseeded); one run per condition |
| Success | LIBERO native success check; an episode ends at the first success, and failure is a timeout |
| Software | LIBERO `d63b117` (repository submodule), robosuite 1.4.0, MuJoCo 3.2.3; 2 × RTX 4090 |

## 2. Experiments completed

- **Phase 5A — fixed configuration.** We ran the paper's stated KS3 oracle configuration directly: global, layer 5, α 0.5, β 0.1. Both it and the unsteered baseline were run on the 30 test states.
- **Phase 5C — oracle selection.** We followed the paper's procedure:
  1. Run the released `experiments/libero/find_best_configs.py` on the 15 fit states over the Table 14 grid: layers {0, 5, 11, 17} × α {0.1, 0.5, 1, 2, 10} × β {0.1, 0.3, 0.5}.
  2. Select the best configuration **separately for each strategy**, by highest fit success, with ties broken by repository order. This covered 60 global, 60 positive-only and 12 per-step configurations.
  3. Freeze the selection to a file before any test rollout.
  4. Evaluate each selected configuration once on the 30 test states.

## 3. Results (KS3, 30 test episodes per condition)

| Condition | Configuration | Paper (Table 4) | Ours |
|---|---|---|---|
| **Global COAST, fixed (Phase 5A)** | L5, α 0.5, β 0.1 | **28/30** | **19/30** |
| **Global COAST, oracle-selected (Phase 5C)** | L11, α 0.5, β 0.3 (fit 13/15) | **28/30** | **21/30** |
| Positive-only, oracle-selected (5C) | L0, α 2.0, β 0.3 (fit 12/15) | 24/30 | 17/30 |
| Per-step, oracle-selected (5C) | L5, α 1.0, β 0.3 (fit 11/15) | 26/30 (α 10) | 15/30 (α fixed at 1.0 in the release) |
| Unsteered baseline, same test states | — | Base 0.53 (Table 1) | 19/30 (5A), 17/30 (5C) |

Additional observations:
- Our unsteered success on the 15 fit states is 8/15 = 0.53. This matches the paper's KS3 Base.
- Across three independent sessions on the same 30 test states, global COAST scored 19, 20 and 21 of 30. The unsteered policy scored 17, 19 and 21 of 30.
- The global result differs from 28/30 at Fisher exact p = 0.010 (Phase 5A) and p = 0.042 (Phase 5C). This is for reference only, because the paper's test states are unknown.

## 4. Limitations of our reproduction

- **Unknown seeds and states.** The paper does not list the init-state IDs or seeds for the 15 fit or 30 test rollouts. Our split (0–14 / 15–44) is an interpretation.
- **Unknown code version.** We used the public `main` branch. The paper's per-step α = 10 cannot be expressed in the released code (per-step α is fixed at 1.0), so some results may come from a different code state.
- **Unknown checkpoint hash.** We used the public HF revision above. No weight hash is published to confirm it is identical to the one behind Table 4.
- **Policy randomness.** Flow-matching noise is unseeded by default. On this task the noise seed alone moves the unsteered result between 7/15 and 13/15 on the same states. Each of our test numbers is a single run, with a standard deviation of about 0.09 at N = 30.
- **Scope.** One task (KS3) and one checkpoint. We have not tested the other LIBERO-10 tasks.

Full records: `phase5a_ks3_global_reproduction.md`, `phase5c_ks3_oracle_sweep.md`, `phase6a_reproduction_gap_analysis.md`.
