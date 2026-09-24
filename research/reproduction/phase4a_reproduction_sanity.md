# Phase 4A — COAST Reproduction Sanity Study

Paper: *Contrastive Conceptor Activation Steering (COAST)*, arXiv:2605.17144.
- Base commit `377838583d35b4c1a9c1bb013ab293efacfa7990`, branch `exp/coast-reproduction-sanity`, clean working tree during all runs.
- Validation only. No change to steering, conceptor construction, activation collection or training; no adaptive β, no token-wise steering, no tuning; `~/Stanley_ws/lerobot` untouched. The only code added is the analysis script.
- Every number below holds only for this checkpoint, layer 5, α 0.5, the listed β values, one master noise seed and one run per condition.

## 0. Summary

**1. Which states are actually new (the requested 30 fresh states do not exist).** LIBERO-10 task 2 has 50 initial states, and 45 of them were already used:

| task-2 states | previous use | status in Phase 4A |
|---|---|---|
| 0–14 | Phase 1A activation collection = **conceptor fitting states** | never evaluated under steering, but **in-sample for the conceptor** |
| 15–29 | development (Phases 1C–2B, 1E) | **reused** |
| 30–44 | held-out in Phase 3A | **reused** (no longer held-out) |
| 45–49 | never used | **strictly untouched (only 5 states)** |

The largest set never behaviorally evaluated is **P = states 45–49 + 0–14 (20 states)**. It is the primary set, reported as `P_fresh` (5) and `P_fit` (15, in-sample). States 15–44 were run again as a secondary set **S**, labelled *reused, exploratory, not new evidence*. Only the β = 0.2 / 0.3 and the added shrinkage-control conditions are new information on them.

**2. Task-2 results (exact counts).**

| set (n) | baseline | COAST β0.1 | COAST β0.2 | COAST β0.3 | shrink β0.1 | shrink β0.2 | shrink β0.3 |
|---|---|---|---|---|---|---|---|
| **P** primary (20) | 9 | 12 | 12 | 11 | 10 | 13 | 14 |
| ↳ P_fresh 45–49 (5) | 1 | 3 | 3 | 3 | 3 | 4 | 4 |
| ↳ P_fit 0–14 (15, in-sample) | 8 | 9 | 9 | 8 | 7 | 9 | 10 |
| S reused 15–44 (30) | 21 | 20 | 21 | 16 | 21 | 21 | 19 |
| ↳ 15–29 (15) | 13 | 10 | 9 | 6 | 10 | 9 | 11 |
| ↳ 30–44 (15) | 8 | 10 | 12 | 10 | 11 | 12 | 8 |
| pooled all 50 (descriptive) | 30 | 32 | 33 | 27 | 31 | 34 | 33 |

Wilson 95% intervals are wide: baseline P 0.26–0.66, COAST β0.1 P 0.39–0.78; pooled-50 baseline 0.46–0.72, COAST β0.1 0.50–0.76. The shrinkage β0.2 / β0.3 rows are **added controls** beyond the requested A–E.

**3. Answers in one line each** (details in §5):
1. **Does increasing β make COAST more effective?** No. Pooled 50: 32 → 33 → 27 for β 0.1 → 0.2 → 0.3; β = 0.3 is the worst COAST setting and below baseline (27 vs 30).
2. **Does COAST outperform shrinkage?** No. It never beats the same-β shrinkage control (pooled 50: β0.1 +3/−2, β0.2 +3/−4, β0.3 +4/−10). At β = 0.3 it is *worse* than shrinkage, an unattributed signal (§5.2).
3. **Does COAST help only when the baseline is weak?** The net change tracks the baseline's strength, and so does pure shrinkage's. Steered success varies less across subsets than baseline success does (§5.3).
4. **Do the Phase 2 conclusions remain valid?** 2A (COAST ≈ shrinkage) is confirmed at β0.1 and β0.2. 2B (direction does not matter) stands at β0.1 but was not re-tested here. It is complicated at β = 0.3 (§5.4).
5. **Is there evidence that conceptor directions contribute?** No beneficial contribution at any β tested. The only conceptor-vs-shrinkage difference found is negative, at β = 0.3, and it is unattributed (§5.5).

**4. Hypothesis screen (this configuration only):**
- *task saturation* — not supported (§6).
- *weak β* — not supported (§6).
- *task selection* — not resolved by one pilot (§6).
- *reproduction mismatch* — **cannot be excluded**. The baseline reproduces the paper's, which points at the steering side.

**Bottom line for the stop rule:** this evidence does not say "COAST fails" and does not say "COAST works". What it does show is a consistent pattern: the released V0 conceptor at L5/α0.5 behaves as a uniform ~β contraction of the residual stream, and matched shrinkage does as well or better everywhere tested (§7).

## 1. Pre-run verification

- **Preregistration** written before any rollout: `examples/libero_env/output/phase4a_servers/preregistration.txt` (gitignored). It fixes the state sets, conditions, order, seed, Part-2 task rule with fallback, and the interpretation rule. An addendum (before any pilot rollout) records the task-3 collection result.
- `git status` clean, HEAD `3778385…`.
- **Conceptor builder check** (needed for Part 2): `experiments/libero/compute_conceptors.py --layers 5 --alphas 0.5 --per_step_indices --task_filter <task 2>` reproduces the Phase-1B NPZ arrays **bit-for-bit** (`np.array_equal`, 4/4 arrays, including `C_contrastive`). The same command was then used, unchanged, for task 3.
- Task-2 conceptor `conceptors/phase1b_repo_task02.npz` sha256 `8b1c802a…ec80f3` (unchanged since Phase 1B).
- **State mapping** (`main.py`): episode k → `initial_states[(seed + k) % 50]`. `--seed 45 --num_episodes 20` gives states 45–49, 0–14. `--seed 15 --num_episodes 30` gives 15–44.

## 2. Configuration

pi0.5 `pi05_libero` (PyTorch), `checkpoints/openpi-libero-2000`, max 520 steps, replan every 5, paired noise master seed 100. Steering: layer 5, α 0.5; β ∈ {0.1, 0.2, 0.3} for COAST (`global` strategy, V0 conceptor); shrinkage h′ = (1 − β)h. One server process per part, each condition run exactly once in the preregistered order (A baseline, B–D COAST β 0.1/0.2/0.3, E shrinkage β0.1, then the added F/G shrinkage β0.2/0.3), set P first, then S.

Commands (from the driver scripts, gitignored under `examples/libero_env/output/phase4a_servers/`):
```
# Part 1 server (GPU 1)
CUDA_VISIBLE_DEVICES=1 uv run --no-sync scripts/serve_policy.py --pytorch --steer --noise-control --steering-diagnostics \
  --conceptor-npz conceptors/phase1b_repo_task02.npz --port 8104 \
  policy:checkpoint --policy.config pi05_libero --policy.dir checkpoints/openpi-libero-2000
# Part 1 client (cd examples/libero_env)
CUDA_VISIBLE_DEVICES=1 MUJOCO_GL=egl LIBERO_CONFIG_PATH=/home/stanley/Stanley_ws/COAST-research/libero_config_coast \
  uv run --no-sync python main.py --task_suite_name libero_10 --task_id 2 --num_episodes {20|30} --seed {45|15} --port 8104 \
  --policy_noise_seed 100 --output_dir output/phase4a_<cond>_task02_{P|S} \
  [--steer --steering_layer 5 --steering_alpha 0.5 --steering_strategy {global|shrinkage} --steering_beta {0.1|0.2|0.3} --log_steering_diagnostics]
```
Runs: P baseline 14:44:02 → S shrink β0.3 15:53:42 (+08:00), all 14 clients exit 0, server log 0 Traceback/ERROR.

## 3. Noise-fingerprint validation (pairing)

All fingerprints were **recomputed from their keys** `(master_seed 100, task_id, init_state, rollout_step)` and matched. Every episode's request schedule is exactly rollout steps 0, 5, … up to its length, and each record's `init_state` equals `(seed + episode) % 50`.

| run group | fingerprints recomputed | condition pairs | identical at every shared `(init_state, rollout_step)` | shared coordinates per pair |
|---|---|---|---|---|
| Part 1, set P (7 conditions) | 9,777 | 21 | **21/21** | 1,026–1,367 |
| Part 1, set S (7 conditions) | 13,846 | 21 | **21/21** | 1,591–1,867 |
| Part 2 pilot (3 conditions) | 2,285 | 3 | **3/3** | 587–688 |

- Sample: set S state 15, step 0 has fingerprint `87ae314ae03d4e54…`, the Phase-1D golden value, in all 7 conditions.
- Diagnostics for every steered run: 10 records per call (denoising steps 0–9), layer 5, correct strategy and β, finite values, schedule equal to the fingerprint schedule.
- **Noise fingerprints matched in every set, so within-set comparisons share initial state, environment seed and initial flow noise.** The trajectories themselves diverge, so this is a *paired-noise* comparison, not identical rollouts.
- **The pairing does not cover sets against each other.** P uses `env.seed(45)`, S uses `env.seed(15)`, and the pilot uses a different task.

**Reproducibility check (S vs earlier phases).** Same configuration as Phases 2B / 3A:

| S states | condition | same outcome | same rollout length |
|---|---|---|---|
| 15–29 vs Phase 2B (same `--seed 15`) | baseline | 15/15 | 15/15 |
| | COAST β0.1 | 14/15 (10 vs 11 successes) | 14/15 |
| | shrinkage β0.1 | 15/15 | 14/15 |
| 30–44 vs Phase 3A (`env.seed` 15 here, 30 there) | baseline | 12/15 (8 vs 9) | 7/15 |
| | COAST β0.1 | 14/15 (10 vs 11) | 5/15 |
| | shrinkage β0.1 | 14/15 (11 vs 10) | 5/15 |

With the same client seed the run reproduces Phase 2B almost exactly (COAST has the one-state cross-process flip noted in Phases 2A/2B). Changing only `--seed` (which also sets `env.seed` and `np.random.seed`) on the same states changes rollout lengths in most episodes and outcomes in 1–3 of 15. So the environment seed is itself a source of trajectory variability; results from P (seed 45) and S (seed 15) are not interchangeable.

## 4. Episode-level results

### 4.1 Primary set P (per state, ✓/✗ with rollout steps; 520 = timeout)

| state | baseline | COAST β0.1 | COAST β0.2 | COAST β0.3 | shrink β0.1 | shrink β0.2 | shrink β0.3 |
|---|---|---|---|---|---|---|---|
| 45 | ✗ | ✗ | ✓ 205 | ✓ 202 | ✗ | ✓ 208 | ✓ 199 |
| 46 | ✗ | ✓ 237 | ✓ 224 | ✓ 229 | ✓ 230 | ✓ 236 | ✓ 232 |
| 47 | ✗ | ✓ 215 | ✗ | ✗ | ✓ 213 | ✗ | ✗ |
| 48 | ✓ 284 | ✓ 196 | ✓ 187 | ✓ 228 | ✓ 189 | ✓ 185 | ✓ 243 |
| 49 | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ 242 | ✓ 219 |
| 0 | ✗ | ✓ 199 | ✓ 247 | ✗ | ✓ 243 | ✓ 246 | ✗ |
| 1 | ✓ 258 | ✓ 250 | ✗ | ✗ | ✗ | ✗ | ✗ |
| 2 | ✓ 202 | ✓ 206 | ✓ 212 | ✓ 193 | ✓ 211 | ✓ 201 | ✓ 203 |
| 3 | ✓ 296 | ✓ 250 | ✓ 244 | ✓ 224 | ✓ 249 | ✓ 237 | ✓ 228 |
| 4 | ✓ 219 | ✓ 218 | ✓ 164 | ✓ 162 | ✓ 204 | ✓ 163 | ✓ 161 |
| 5 | ✓ 230 | ✗ | ✗ | ✓ 217 | ✗ | ✓ 228 | ✓ 227 |
| 6 | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| 7 | ✗ | ✗ | ✓ 230 | ✓ 215 | ✗ | ✓ 229 | ✓ 227 |
| 8 | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| 9 | ✗ | ✓ 212 | ✓ 212 | ✓ 212 | ✓ 211 | ✓ 212 | ✗ |
| 10 | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ 352 |
| 11 | ✗ | ✓ 264 | ✓ 194 | ✗ | ✗ | ✗ | ✓ 176 |
| 12 | ✓ 204 | ✗ | ✗ | ✓ 204 | ✗ | ✗ | ✓ 314 |
| 13 | ✓ 219 | ✓ 226 | ✓ 217 | ✗ | ✓ 229 | ✓ 229 | ✓ 376 |
| 14 | ✓ 216 | ✓ 213 | ✓ 209 | ✓ 208 | ✓ 214 | ✓ 208 | ✓ 213 |
| **total** | **9** | **12** | **12** | **11** | **10** | **13** | **14** |

The per-episode table for S (30 states) and all per-episode intervention metrics are in `experiments/phase4a_episode_results.csv` (380 rows: condition, task, init_state, success, rollout_steps, fingerprints, hook applications, mean ‖Δh‖, mean ‖h′‖/‖h‖, cosine, mean ‖h‖).

### 4.2 Transitions relative to baseline (fail → success / success → fail; exact McNemar p is reference only)

| set | COAST β0.1 | COAST β0.2 | COAST β0.3 | shrink β0.1 | shrink β0.2 | shrink β0.3 |
|---|---|---|---|---|---|---|
| P (20) | +5 −2 (0.45) | +6 −3 (0.51) | +4 −2 (0.69) | +4 −3 (1.0) | +6 −2 (0.29) | +6 −1 (0.13) |
| P_fresh (5) | +2 −0 | +2 −0 | +2 −0 | +2 −0 | +3 −0 | +3 −0 |
| P_fit (15) | +3 −2 | +4 −3 | +2 −2 | +2 −3 | +3 −2 | +3 −1 |
| S reused (30) | +4 −5 (1.0) | +5 −5 (1.0) | +3 −8 (0.23) | +3 −3 (1.0) | +5 −5 (1.0) | +5 −7 (0.77) |
| pooled 50 | +9 −7 (0.80) | +11 −8 (0.65) | +7 −10 (0.63) | +7 −6 (1.0) | +11 −7 (0.48) | +11 −8 (0.65) |

No steered condition is separated from baseline at any set size. The smallest reference p in this table is 0.125 (shrinkage β0.3 on P), and the ~15 comparisons per set are uncorrected.

## 5. Answers to the analysis questions

### 5.1 Does increasing β make COAST more effective?

**No.**
- COAST successes by β (0.1 / 0.2 / 0.3): P 12 / 12 / 11; S 20 / 21 / 16; pooled 32 / 33 / 27.
- Pooled β transitions: 0.1 → 0.2 is +6 −5; 0.2 → 0.3 is +3 −9. Going from β0.2 to β0.3 costs 6 successes pooled (33 → 27), mostly on states 15–29 where the baseline was strong (COAST β0.3: 6/15 vs baseline 13/15, +0 −7, post-hoc reference p = 0.016).
- Pure shrinkage does not show a monotone benefit either: P 10 / 13 / 14 but S 21 / 21 / 19, pooled 31 / 34 / 33. Its P-set increase is not repeated on S.
- The realized contraction scales exactly as intended: ‖h′‖/‖h‖ = 0.9006 / 0.8010 / 0.7023 for COAST and 0.8984 / 0.8009 / 0.6993 for shrinkage; cos(Δh, h) ≈ −0.9996 / −0.9997 / −0.9997 (COAST) and −0.9999 (shrinkage). Stronger β is a stronger uniform contraction, not a stronger useful signal.

### 5.2 Does COAST outperform shrinkage?

**No, not at any β tested.**
- COAST β0.1 vs shrinkage β0.1 (the requested comparison): P +2 −0 (agree 18/20); S +1 −2 (agree 27/30); pooled +3 −2 (agree 45/50).
- At matched β0.2: pooled +3 −4 (agree 43/50). At matched β0.3: pooled +4 −10 (agree 36/50, reference p 0.18); on states 15–29 it is +0 −5 (p 0.0625).
- Against the *requested* single shrinkage β0.1 the picture is the same: COAST β0.2 +7 −5, COAST β0.3 +6 −10 (pooled).
- **The β = 0.3 gap is the only place where COAST and shrinkage diverge noticeably, and it points the wrong way for COAST.** The contractions are near-equal in size (0.7023 vs 0.6993), so it is not a magnitude effect. The residual difference is a small deviation of Δh from the −h direction (cos −0.99973 vs −0.99999). Two things prevent reading it as a direction effect: no random-matched control was run at β = 0.3, and the comparison is post hoc, uncorrected and largely on reused states.

### 5.3 Does COAST only help when baseline is weak?

Net change of COAST β0.1 vs baseline by subset (gains − losses), against the baseline's success rate on that subset:

| subset | baseline | COAST β0.1 net | shrinkage β0.1 net | steered success (COAST β0.1 / shrinkage β0.1) |
|---|---|---|---|---|
| S 15–29 (reused) | 13/15 | −3 | −3 | 10/15, 10/15 |
| S 30–44 (reused) | 8/15 | +2 | +3 | 10/15, 11/15 |
| P_fit 0–14 | 8/15 | +1 | −1 | 9/15, 7/15 |
| P_fresh 45–49 | 1/5 | +2 | +2 | 3/5, 3/5 |

- Every gain occurs on a baseline failure and every loss on a baseline success (by construction). The subset-level sign follows baseline strength: negative where baseline is high, positive where it is low.
- Steered success is flatter than baseline success across the three 15-state subsets: COAST β0.1 60–67%, shrinkage β0.1 47–73%, against baseline 87% / 53% / 53%.
- That is what one expects if steering acts mainly as a perturbation that re-draws each trajectory (outcomes regress toward a middle rate), rather than as a consistent capability gain. This is a reading consistent with the data, not a demonstrated mechanism.
- Shrinkage, which carries no conceptor, shows the same pattern. So the pattern is **not specific to COAST**.
- Among episodes both baseline and steered succeed, the steered rollouts are shorter: on set S, COAST β0.1 249.8 → 231.8 steps, and the same for shrinkage (246.7 → 233.4). So contraction may speed up motion; this is not investigated.

### 5.4 Do the previous Phase 2 conclusions remain valid?

- **2A (COAST ≈ pure shrinkage): confirmed** at β0.1 (agreement 45/50 pooled; P set 18/20, with the 2 disagreements COAST-only gains on states 1 and 11) and at β0.2 (43/50). Every disagreement is a single-state flip in either direction.
- **2B (conceptor direction does not matter): stands at β0.1 but was not re-tested.** No random-matched condition was run in Phase 4A. In Part 1 the hidden-state mechanism at β0.1 is identical to Phases 2A/2B/3A: ‖h′‖/‖h‖ = 0.90062 (COAST), 0.89838 (shrinkage), cos −0.9996 / −0.9999.
- **Complication at β = 0.3:** COAST and shrinkage disagree more (§5.2). This does not overturn 2B, but the 2B conclusion must not be extended to β = 0.3 without a random-matched control there.
- **"Steering changes success rate":** still not supported. All steered pooled counts (27–34 of 50) are within the baseline's own noise-seed spread (7–13 of 15, Phase 1D) and the pooled baseline is 30/50.

### 5.5 Is there evidence that conceptor directions contribute?

**No evidence of a beneficial contribution.**
- COAST does not beat the matched-β shrinkage control at any β on P, S or pooled.
- COAST vs shrinkage disagreements are symmetric at β0.1/0.2 and negative for COAST at β0.3 (§5.2).
- The gains COAST shows over baseline on P (+3 at β0.1) are shared by shrinkage at higher β (+4, +5 at β0.2/0.3).
- A direction-specific effect would appear as COAST differing from *both* shrinkage and a random matched conceptor in the same direction. This phase cannot show that, since it lacks the random control. The evidence that exists (Phases 2B/3A, β0.1) found none.

## 6. Why is there no improvement? (screen of the four suspects)

| suspect | evidence | verdict |
|---|---|---|
| **task saturation** | Pooled baseline is 30/50 = 0.60 (CI 0.46–0.72); 20 of 50 states fail at baseline and many are rescued by any steering. Baseline on the fit/fresh sets is 8/15 and 1/5 | **Not the cause.** There is room to improve, and the steered runs do not exploit it consistently |
| **weak β** | β 0.1 → 0.3 tripled the intervention (‖Δh‖ 17 → 52) with no monotone benefit; β = 0.3 is the worst COAST setting | **Not supported** within β ≤ 0.3 at L5/α0.5. Larger β was not tried and not requested |
| **task selection** | One task (2) with 50 states, plus a 10-state pilot on task 3 (below). The pilot is the largest positive result (+4 −0) but has no shrinkage control and is 10 episodes | **Unresolved.** Two tasks, one layer/α, cannot represent the paper's per-task oracle configs |
| **reproduction mismatch** | Baseline reproduces the paper's: paper Table 1 = 0.53 for this task (Phase 1A notes) vs 30/50 = 0.60 here (CI includes 0.53). The paper reports 0.93 for L5/α0.5/β0.1 on 30 held-out rollouts; we get 9–11/15 for COAST β0.1 on comparable 15-state sets (Phase 3A, and 4A P_fit and 30–44) | **Cannot be excluded.** The baseline matching shifts suspicion to the steering side, but the paper-faithful V3 conceptor was only tested at β0.1 on 15 development states (Phase 1D), and the paper's exact evaluation states, seeds and per-task configuration are unknown here |

### Part 2 pilot (harder task)

- **Selection.** Task 3, `KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it`. The rule was set in the preregistration from the repo's own recorded baseline (`examples/libero_env/figures/results_2000_3000_9000.json`, checkpoint 2000, 15 eps/task): task 3 and task 7 tie at 0.33 (5/15), and task 9 (0.20) is too close to the ≥ 3 success minimum a contrastive conceptor needs. Tie-break: lower task_id. No performance numbers were invented.
- **The selection did not turn out clearly harder.** A fresh seed-0 collection on task 3 gave 8 successes / 7 failures (0.53, unpaired noise) against the recorded 0.33. That is roughly as hard as task 2 (8/7). Fallback (task 7) was declared for < 3 in a class only, so the task was kept and the discrepancy is recorded rather than re-selected.
- **Conceptor.** Built with the unchanged official builder from that collection (layer 5, α 0.5): `conceptors/phase4a_repo_task03.npz`, sha256 `374e1b19…ef6df`. The layer/α are **transferred from task 2, not tuned for task 3**.
- **Pilot.** GPU 0, `--seed 15 --num_episodes 10` (states 15–24, disjoint from the fitting states), paired noise, master seed 100. Each condition run once.

| condition | successes /10 | vs baseline | states gained | states lost | ‖h′‖/‖h‖ | cos(Δh, h) |
|---|---|---|---|---|---|---|
| baseline | 5 | — | — | — | — | — |
| COAST β0.1 | **9** | +4 −0 (ref p 0.125) | 15, 18, 19, 21 | — | 0.9005 | −0.9996 |
| COAST β0.2 | 6 | +3 −2 (ref p 1.0) | 18, 19, 21 | 16, 24 | 0.8011 | −0.9997 |

Per state (state, success, steps): baseline `15✗ 16✓428 17✗ 18✗ 19✗ 20✓465 21✗ 22✓470 23✓272 24✓358`; β0.1 `15✓296 16✓442 17✗ 18✓252 19✓251 20✓323 21✓188 22✓374 23✓226 24✓240`; β0.2 `15✗ 16✗ 17✗ 18✓215 19✓249 20✓409 21✓284 22✓228 23✓219 24✗`.

- The β0.1 improvement is the largest single positive result of the study, but it is non-monotone in β (β0.2 is 6/10) and it has **no shrinkage control**: given Part 1, a contraction-only control could give the same. It cannot be attributed to conceptor directions.
- The hook mechanism on task 3 matches task 2 (same contraction and alignment), so the mechanistic finding is not specific to task 2.

## 7. Is the reproduction sufficient to justify designing a new COAST method?

Stated as evidence, not as a verdict on COAST:

- **Supported:** with the released V0 conceptor at L5/α0.5 (task 2, 50 states; plus a sanity check on task 3), the steering hook is a fixed ~β contraction of h along h. Behavior matches matched pure shrinkage to within single-state flips at β0.1/0.2. More steering strength does not reveal a useful conceptor effect, and at β = 0.3 the conceptor version does worse than shrinkage. This holds on fresh, in-sample-fit and reused states.
- **Not supported / not settled:**
  - that this is an implementation-independent limitation of COAST — the paper-faithful V3 builder, other layers, and the paper's own states and per-task configs were not covered here;
  - that the pilot's task-3 gain is or is not a conceptor effect — no control was run;
  - anything about β > 0.3, other α, or other layers.
- **What the evidence recommends before designing a new method** (suggestions, not done here, per the stop rule): (i) run the paper-faithful V3 conceptor and a random-matched control at β 0.2–0.3 on the same states, to settle whether the β = 0.3 gap and the paper mismatch are (A) reproduction or (B) mechanism; (ii) if the collapse to shrinkage persists, the design problem is that the conceptor term carries almost no energy on the hidden states (Rayleigh quotient hᵀCh/‖h‖² ≈ 0.006 vs trace/d 0.0185 in Phase 2B), so a new method should target that directly rather than raise β.

## 8. Limitations

- **Small N and one noise seed.** 20 primary states (5 fresh + 15 in-sample), 30 reused, 10 pilot; one master seed (100), one run per condition. Phase 1D showed the baseline alone moves 7–13 successes of 15 with the noise seed. All reference p-values ≥ 0.125 except one post-hoc 0.016 and one 0.0625, with ~15 uncorrected comparisons per set.
- **The 30 fresh states do not exist.** Only 5 strictly untouched states; the 15 fitting states are in-sample for the conceptor (favourable to COAST, and it still shows no separation).
- **Set S is reused** and pooled-50 numbers mix fit, reused and fresh states. They are descriptive only.
- **`--seed` couples state choice with `env.seed`/`np.random.seed`**; P and S use different env seeds, so they are not paired with each other (§3).
- **Steered-path cross-process jitter** (up to a few steps, occasionally one outcome; seen again in the S replicate) means individual single-state disagreements are not interpretable.
- **Added controls** (shrinkage β0.2/0.3) go beyond the requested A–E; they were run after A–E and change no requested result.
- **No random-matched control and no paper-faithful V3** in this phase.
- **Pilot:** 10 episodes, one task, transferred L5/α0.5, no shrinkage control; the "harder" task turned out to be similar in difficulty (0.53 at collection).
- **Contraction magnitudes are not matched exactly** between COAST and shrinkage (e.g. 0.9006 vs 0.8984 at β0.1); they are close but were not corrected.
- **Measured at the hooked layer only**, GPU 1 for Part 1 and GPU 0 for the pilot (both RTX 4090).

## 9. Files and housekeeping

- **Added (nothing else changed; no source, test or config edits):**
  - `research/reproduction/phase4a_reproduction_sanity.md` (this report)
  - `research/reproduction/experiments/phase4a_reproduction_sanity.yaml` (generated from the analysis output)
  - `research/reproduction/experiments/phase4a_episode_results.csv` (380 rows)
  - `research/reproduction/tools/analyze_phase4a.py` (analysis only; a wrong regex capture group, which made every episode read as a failure, was found and fixed on the first run, before any reported number was produced)
- **Run outputs, not committed (gitignored):** `examples/libero_env/output/phase4a_*` (videos, client logs, fingerprint/diagnostic JSONL), `examples/libero_env/output/phase4a_servers/` (preregistration, drivers, server logs); `activations/libero/…/KITCHEN_SCENE4…` (task-3 collection, ~8 GB); `conceptors/phase4a_repo_task03.npz`.
- Servers were stopped after the runs.
