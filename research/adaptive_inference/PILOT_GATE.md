# Stage-5 pilot gate: same-state intervention matrix

Date: 2026-10-16. **Verdict: NO-GO.** The preregistered gate fails on A, B and D (C and E pass). The verdict is robust to the one infrastructure caveat found (§4).

| item | value |
|---|---|
| preregistration | [`PILOT_PREREG.md`](PILOT_PREREG.md) and [`INTERVENTION_SPEC.md`](INTERVENTION_SPEC.md), frozen 14:07:56 +08:00. Analysis code [`analyze_pilot.py`](analyze_pilot.py) and its tests frozen 14:24:41, before any outcome was inspected. All hashes in `results/adaptive_inference/pilot/PREREG_FROZEN.txt`, re-verified after the run |
| run | `results/adaptive_inference/pilot/` (gitignored). 40 units = tasks {0, 2, 3, 7} × inits 15–24; 306 decision states; 6,120 branches + 306 identity branches. Run 14:10 → 21:05, 47.6 h of client wall time on 4 π0.5 servers (2× RTX 4090), 8 workers |
| archive | `results/adaptive_inference/pilot_archive_20261016.tgz`, sha256 `5151df7e…ea63`; per-file checksums in `pilot/RESULTS_SHA256.txt` (395 files) |
| analysis | `uv run python research/adaptive_inference/analyze_pilot.py` → `pilot/analysis/{summary.json, intervention_matrix.csv}`. Post-hoc sensitivity (labelled as such): `sensitivity_pilot.py` → `pilot/analysis/sensitivity.json` |
| code | HEAD `4239b15`; research tree uncommitted. Fork-client and snapshot hashes are recorded in all 8 worker manifests and equal the frozen ones |

## 1. Always-on success (306 states × 4 replicates; state-level means)

| arm | all | LR2a (0) | KS3 (2) | KS4 (3) | LR1 (7) | server ms / branch | Δ vs I0 pairs (arm-only / I0-only) |
|---|---:|---:|---:|---:|---:|---:|---|
| I0 base | 41.3% | 37.5 | 37.0 | 31.1 | 61.0 | 9,416 | — |
| I0P resampling placebo | 43.1% | 42.9 | 41.9 | 34.6 | 53.4 | 9,319 | 171 / 150 |
| IH h = 2 | 39.2% | 45.2 | 35.9 | 21.8 | 54.1 | 23,986 | 137 / 163 |
| **IL COAST global** | **48.0%** | 53.9 | 39.1 | 32.1 | 66.8 | 9,076 | 172 / 91 |
| IC CAG ω = 1.5 | 45.0% | 33.3 | 40.5 | 43.3 | 64.7 | 16,825 | 193 / 148 |

States per task: 84 / 71 / 78 / 73. Pairs are (state, replicate) under common random numbers. McNemar p-values (IL 7e-7, IC 0.017, IH 0.15, I0P 0.26) are descriptive only: (s, r) pairs are dependent within units.

## 2. Gate

| gate | rule (frozen) | observed | pass |
|---|---|---|---|
| **A** | LOO state oracle − best always-on ≥ 5 pp, with lower CI > 0 (or the Pareto route) | O_state = 46.1% vs **B = 48.0% (always IL)** → **−1.9 pp**, unit bootstrap 95% [−4.9, +1.1]. Pareto route: best O_state(λ) = 48.0% at 11.05 s/branch vs IL at 9.08 s | **✗** |
| **B** | ≥ 2 interventions with unique-rescue ≥ 5% **and** more unique rescues than the resampling placebo | unique-rescue among {IH, IL, IC}: 11.4 / 14.4 / 22.5% (all ≥ 5%). With the placebo as competitor: IH 6.3, IL 9.5, IC 11.2 vs **I0P 11.8%**. No intervention rescues uniquely more often than re-sampling the base policy | **✗** |
| **C** | max rescue share ≤ 90% | IH 27%, IL 34%, IC 38% | ✓ |
| **D** | LOO state oracle − LOO task oracle ≥ 2.5 pp, with lower CI > 0 | O_task = 50.4% > O_state → **−4.3 pp** [−8.2, −0.3] | **✗** |
| **E** | identity ≥ 99% (failures investigated), restore 100%, exceptions < 1% | identity **303/306 (99.0%)**; restore state + obs hash identity **6,426/6,426**; exceptions **0**. The 3 failures were investigated (§4) | ✓ |

**GO requires A ∧ B ∧ C ∧ D ∧ E → NO-GO.**

## 3. The preregistered descriptive analyses

1. **Always-on:** §1. IL is the best single operator (+6.6 pp over I0), IC is second (+3.7), the placebo gives +1.8 (noise), and IH is −2.1 at 2.5× compute.
2. **State-level oracle.**
   - Noise-honest LOO: 46.1%.
   - The *naive* single-draw oracle over {I0, IL, IC, IH} is **66.3%** (+18 pp over B). The naive oracle over just {I0, I0P}, i.e. the base policy with two noise draws, is already **55.3%** (+14 pp over I0).
   - Almost all of the apparent "complementarity" is re-sampling noise. This is the same phenomenon as Stage 1 on GR00T.
3. **Unique rescues:** see gate B. On the 347 salvageable (state, replicate) pairs with the placebo included, I0P has the **most** unique rescues (41), ahead of IC 39, IL 33 and IH 22.
4. **Pairwise complementarity** P(row ✓, column ✗), in %:

   | row \ column | I0 | I0P | IH | IL | IC |
   |---|---:|---:|---:|---:|---:|
   | I0 | — | 12.3 | 13.3 | 7.4 | 12.1 |
   | I0P | 14.0 | — | 15.8 | 12.4 | 14.0 |
   | IH | 11.2 | 11.9 | — | 9.1 | 12.3 |
   | IL | 14.1 | 17.3 | 17.8 | — | 15.7 |
   | IC | 15.8 | 15.9 | 18.1 | 12.7 | — |

   Discordance between I0 and its own re-sample (I0P) is 12–14% in each direction, about the same as between any two different operators. IL vs I0 has the lowest reverse discordance (7.4%), consistent with IL being a small, mostly beneficial perturbation (≈ shrinkage, per Phase 7B).
5. **Δ_i(s) distribution** (states with Δ = −1 … +1 in steps of 0.25):
   - Δ = 0 for 52–58% of states for every arm, the placebo included.
   - Help / hurt fraction of states: IL 30.1 / 12.4%, IC 27.8 / 19.3%, IH 21.2 / 26.5%, I0P (placebo) 25.8 / 22.2%.
6. **Harm** (base succeeds, arm fails, per pair): IL 18.0%, IC 29.2%, IH 32.2%, I0P 29.6%.
7. **Risk vs winner** (bins of base success p̂_I0 from the 4 base replicates; "tie" = no unique best arm):

   | p̂_I0 | states | winner counts | mean Δ IH / IL / IC | states with any Δ > 0 |
   |---|---:|---|---|---:|
   | 0 | 104 | tie 82, IC 10, IL 9, IH 3 | +.06 / +.09 / +.11 | 32% |
   | .25 | 53 | IL 15, IC 15, tie 14, IH 8, I0 1 | +.04 / +.19 / +.14 | 79% |
   | .5 | 52 | tie 29, IC 11, IL 7, IH 4, I0 1 | −.07 / +.08 / +.09 | 79% |
   | .75 | 39 | tie 22, IL 10, IC 4, I0 2, IH 1 | −.09 / +.04 / −.10 | 59% |
   | 1 | 58 | tie 53, I0 5 | −.13 / −.09 / −.14 | 0% |

   - Bin × winner χ² = 64.0, p = 4e-9. Most of this is the trivial structure that fully successful states cannot be improved.
   - **Within the same risk bin, different operators "win" (IL vs IC vs IH), but these winners are not reproducible:** a router that knows each state's winner from 3 replicates does *worse* on the 4th than always using IL (gate A). Equal-risk states do not exhibit *detectable* operator-specific utility at this sample size.
   - Failure-gated (RL²-style) LOO router: 45.0%. It captures 78% of the (negative) state-oracle gain over I0, and both lose to always-IL.
8. **Δ vs pre-intervention features** (Spearman ρ across states):
   - Every |ρ| ≤ 0.17.
   - Base-sample spread and CAG contrast: ρ ≈ 0 for every Δ_i.
   - Decision step and goal progress: weakly negative for IL and IC benefit (ρ −0.14 … −0.17, p < 0.02). Later or more advanced states benefit less.
   - Uncertainty proxies predict neither failure (|ρ| ≤ 0.14) nor operator benefit.
9. **Cost-aware frontier:** O_state(λ) is 46.1% at λ = 0 and 48.0% at λ ≥ 0.01, where it collapses onto "always IL" (the cheapest arm with the best mean). There is no point above always-IL at any cost.

**Task-level structure exists:** O_task = 50.4% (+2.4 pp over always-IL). The per-task winners are IL for LR2a and LR1 and IC for KS4. KS3 is near-flat; its best arm is IC on means, while I0P is the best always-on arm there. Whatever operator heterogeneity is detectable in this pilot is **between tasks**, which is exactly kill condition 3 of `NOVELTY_BOUNDARY.md`.

## 4. Infrastructure investigation (gate E: "any failure investigated")

All 3 identity failures are in KS3 (task 2, "turn on the stove…"): groups 2_21_0, 2_21_50 and 2_23_50. For all three:
- restore state and observation hashes matched;
- outcome class was unchanged in two of them (end step 199 / 195 vs 201);
- one flipped (success at 303 vs timeout).

Reproduction with [`fork/debug_identity.py`](fork/debug_identity.py) (outputs in `results/adaptive_inference/validation/debug_*.json`):
- **Cause:** **non-deterministic rendering**. In 2 of 4 identical repeats of the same unit, the wrist-camera image at query step 75 differs in **one pixel by one intensity level**, while the MuJoCo state is bitwise identical. This happens just after the burner flame becomes visible. The pixel takes one of two values in the *reference* rollout as well, so it is a property of EGL rendering of this scene, not of restore. The 1-LSB difference changes the π0.5 action slightly, and the closed loop amplifies it.
- **Latent, harmless here:** LIBERO toggles the flame via `MjModel.site_rgba` (alpha `1 − alpha`), which the v1 snapshot does not save. After a restore the flame alpha can be stale for the *first* simulated step only. LIBERO's own post-process corrects it at the end of that step, and the image rendered in that step is never a policy input for h ∈ {2, 5}. Tested: alpha 1.0 and 2.0 render identically, and alpha ≤ 0 is invisible. A v2 snapshot that also restores `site_rgba` / `geom_rgba` and the `vis_site_names` flags is saved as `results/adaptive_inference/validation/sim_snapshot.v2_proposal.py`. It **must** be adopted before any h = 1 operator is used. The frozen v1 file is unchanged.
- **Tasks 0, 3 and 7 have no visualization sites or tracked object states** (checked), and their identity branches are 235/235 exact.

**Sensitivity (post hoc): tasks 0, 3 and 7 only, all exactly reproducible, 235 states.**
- A = −2.7 pp [−6.3, +0.8].
- D = −6.4 pp [−10.0, −1.5].
- Placebo unique-rescue 11.0% vs IH 7.8, IL 10.6, IC 11.3%. IC exceeds the placebo by 0.3 pp, so at most one arm passes and B still fails.

**Same verdict.** Per task, the state oracle is below the best always-on arm in every task: −6.0 / −0.7 / −8.0 / −5.1 pp.

Other operational notes:
- At launch the server on port 8600 died silently during model load. It was restarted before it had served any request; the two workers on that port waited, and no data was affected. Log kept as `logs/server_8600.died.log`.
- `run_pilot.sh` says units are "interleaved across tasks so every worker sees every task". With 8 workers and 4 tasks, each worker actually received a single task. This only affects load balance, not data.

## 5. What the pilot establishes

1. **At the same π0.5 state, operator "complementarity" is indistinguishable from re-sampling noise.**
   - Re-rolling the base policy produces as many unique rescues as any operator.
   - The single-draw oracle (+18 pp) is mostly re-sampling (+14 pp from I0 + I0P alone).
   - Under honest leave-one-replicate-out selection, choosing per state loses to always using the best operator.
2. **Operator differences that exist are global or task-level.** IL is the best operator overall (+6.6 pp over I0 on mid-episode states, CRN-paired; exploratory and not preregistered as a hypothesis). Its effect has the same sign across states, and per-task choice adds +2.4 pp.
3. **Pre-intervention uncertainty features carry almost no information** about which operator helps (|ρ| ≤ 0.17).
4. **Infrastructure:** exact same-state restore for LIBERO is achievable (and LIBERO's own restore is not). Residual non-determinism is a rare 1-px render flip, which the identity branch detects.

## 6. Post-hoc: is there real state-level heterogeneity that R = 4 cannot resolve?

**Not preregistered, labelled exploratory.** Script: [`heterogeneity_pilot.py`](heterogeneity_pilot.py) → `pilot/analysis/heterogeneity.json`.

Method of moments for each paired contrast d = Y_a − Y_b: between-state variance τ² = Var_s(mean_r d) − mean_s(Var_r(d)/R), overall and after removing task means, with a unit bootstrap.

| contrast | τ (all) | 95% CI of τ² | τ within task | 95% CI of τ² (within) |
|---|---:|---|---:|---|
| I0P − I0 (placebo, true τ = 0) | 0.077 | [−.008, +.021] | 0.056 | [−.011, +.015] |
| IL − I0 | 0.103 | [−.003, +.024] | 0.081 | [−.007, +.019] |
| IC − I0 | 0.126 | [−.004, +.036] | 0.111 | [−.008, +.029] |
| IH − I0 | 0.111 | [−.005, +.028] | 0.088 | [−.009, +.023] |
| **IL − IC** | 0.181 | **[+.010, +.057]** | 0.136 | [−.001, +.034] |
| **IL − IH** | 0.162 | **[+.008, +.047]** | **0.159** | **[+.006, +.044]** |
| **IC − IH** | 0.234 | **[+.028, +.083]** | **0.198** | **[+.017, +.061]** |

- Operator-vs-base contrasts are indistinguishable from the placebo.
- **Operator-vs-operator contrasts show real between-state variation**, including within task, mostly involving the temporal arm IH. τ ≈ 0.14–0.20 in success-probability units.
- A Gaussian model of the true contrasts (task means plus the pooled within-task covariance; rough, ignores the [0, 1] bounds) implies that a router with *perfect* per-state knowledge would gain about **+7.9 pp over always-IL** and **+4.7 pp over the per-task best arm**.

That is roughly the size the gate asked for. It is invisible at R = 4 because the per-state sampling SE (≈ 0.25–0.35) exceeds τ. No measured pre-intervention feature tracks it: every Spearman |ρ| ≤ 0.17 (§3.8).

The preregistered verdict stands. But the NO-GO means **"not detectable or exploitable at pilot scale, and not predicted by the available features"**, not "absent".

## 7. What it does not establish

- That *no* state-level heterogeneity exists. With R = 4 replicates per arm, per-state effects are estimated with SE ≈ 0.25–0.35. True per-state Δ variance below roughly that level is undetectable, and the LOO oracle is conservative by construction. What is established is that it is not large enough to beat always-on selection at pilot scale, which is the preregistered question.
- Anything about other backbones, other operator portfolios (e.g. SCALE, retrieval, representation-level KV patching), or LIBERO-CF, where CAG's premise applies.
- That IL's always-on gain is conceptor-specific. Phase 7B found COAST ≈ shrinkage, and no shrinkage control was run here.
