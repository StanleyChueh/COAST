# Stage-5 pilot preregistration: same-state intervention matrix

Frozen 2026-10-16, before any pilot rollout. The files hashed in `results/adaptive_inference/pilot/PREREG_FROZEN.txt` are:
- this file;
- `INTERVENTION_SPEC.md`;
- `intervention_server.py`;
- `fork/libero_fork.py`;
- `fork/sim_snapshot.py`;
- `conceptors/aiv_pilot_conceptors.npz`.

These hashes are re-verified after the run. Only smoke runs (3 units, ≤ 2 decision states, 1 replicate, outputs in `results/adaptive_inference/smoke{1,2,3}`) were executed beforehand. They are excluded from all analysis.

## Question

At the same π0.5 state, do different inference-time operators rescue different failures often enough that choosing among them matters, beyond what re-sampling the base policy already gives?

## Design

| item | frozen value |
|---|---|
| backbone | π0.5 `openpi-libero-2000`, PyTorch eager (INTERVENTION_SPEC.md) |
| tasks | LIBERO-10 task ids **0 (LR2a), 2 (KS3), 3 (KS4), 7 (LR1)**. These are the only LIBERO-10 tasks with COAST fit activations and paper IL configurations. Paper base rates 0.40 / 0.53 / 0.40 / 0.80 give different failure profiles |
| initial states | **15–24** (10 per task → 40 units). States 0–14 were used to fit the conceptors. **States 25–49 stay untouched** for any later router test |
| reference rollout | I0, h = 5, noise master **1000** |
| decision states | post-settle steps **t ∈ {0, 50, 100, …, 400}** reached by the reference rollout (action queue empty there by construction), with ≥ 100 steps of budget left. Selection does not look at any branch outcome; states after a reference success are simply not reached |
| arms | **I0, I0P, IH, IL, IC** (INTERVENTION_SPEC.md); each runs from the decision state to success or step 520 |
| replicates | **R = 4** noise masters **2001–2004**; within a replicate every arm uses the same noise key (master, task, init, rollout_step) (common random numbers); I0P uses master + 1000 |
| identity branch | per decision state: I0 with master 1000, which must reproduce the reference continuation exactly |
| primary outcome | terminal task success (LIBERO `done`) within the episode budget |
| secondary | goal predicates satisfied at the end; steps used; policy calls; server inference ms |
| expected size | about 250–320 decision states × 21 branches |
| execution | 4 servers (2 per RTX 4090), 8 client workers, units interleaved across tasks |

## Estimands and frozen definitions

Notation: Y_{i,r}(s) ∈ {0,1} for arm i, replicate r and state s; p̂_i(s) = mean_r Y_{i,r}(s). The intervention set is 𝓘 = {IH, IL, IC}. The arm set used for routing is 𝓐 = {I0} ∪ 𝓘. I0P is reported separately as placebo / resampling.

1. **Always-on success:** S_i = mean_s p̂_i(s), for every arm.
2. **State-level oracle (noise-honest, primary).**
   - Leave-one-replicate-out: for each state and each held-out replicate r, choose i*(s, −r) = argmax_{i∈𝓐} mean_{r'≠r} Y_{i,r'}(s). Ties go to the cheaper arm in the order I0 < IL < IC < IH.
   - Score Y_{i*, r}(s).
   - O_state = mean over (s, r). This estimates what a router that knows each state's response could achieve. It is conservative, because selection uses only 3 draws.
   - Also reported but **not** used for the gate: the naive single-draw oracle mean_{s,r} max_{i∈𝓐} Y_{i,r}(s).
3. **Task-level oracle:** the same leave-one-replicate-out procedure, but the arm is chosen per *task* from mean_{s∈task, r'≠r} Y_{i,r'}(s). This gives O_task.
4. **Best always-on:** B = max_{i∈𝓐} S_i.
5. **Unique rescue (per replicate).**
   - A pair (s, r) is *salvageable* if Y_{I0,r}(s) = 0 and ∃ i ∈ 𝓘 with Y_{i,r}(s) = 1.
   - A *unique rescue* by i is a salvageable (s, r) where i is the only member of 𝓘 that succeeds.
   - U_i = #unique rescues of i / #salvageable.
   - The placebo version adds I0P to the competitor set: 𝓘⁺ = 𝓘 ∪ {I0P}, with the same definitions; it gives U_i⁺ and U_I0P⁺.
6. **Rescue share:** R_i = #(s, r) with Y_{I0,r} = 0, Y_{i,r} = 1 for i ∈ 𝓘, divided by the sum over 𝓘.
7. **Harm:** #(s, r) with Y_{I0,r} = 1 and Y_{i,r} = 0, divided by #(s, r) with Y_{I0,r} = 1.
8. **Δ distribution:** Δ_i(s) = p̂_i(s) − p̂_I0(s), with values in {−1, −0.75, …, 1}.
9. **Pairwise complementarity:** P(Y_i = 1, Y_j = 0) over (s, r), for all ordered pairs in 𝓐 ∪ {I0P}.
10. **Risk ≠ utility.**
    - Risk bins use p̂_I0(s) ∈ {0, .25, .5, .75, 1}.
    - Within each bin: the distribution of the leave-one-out winner (states with a unique LOO-majority winner), the variance of Δ_i, and a Fisher / χ² test of independence between bin and winner.
    - **Failure-gated baseline (RL²-style):** a leave-one-replicate-out risk estimate p̃_I0(s, −r). If p̃ ≤ 0.5, use the single globally best i ∈ 𝓘 (chosen on the other replicates), else I0. Score on replicate r; this gives O_risk.
    - **Captured fraction:** (O_risk − S_I0) / (O_state − S_I0).
11. **Feature conditioning (descriptive):** Δ_i vs the base sample spread, CAG contrast, IL displacement, decision step and goal progress at the state.
12. **Cost-aware frontier:** cost_i = mean server inference ms per branch of arm i. Report O_state(λ) with the cost-penalized leave-one-out choice argmax[mean Y − λ·cost/cost_I0], for λ ∈ {0, 0.01, 0.02, 0.05, 0.1}.

Uncertainty: bootstrap over **units** (task × init), stratified by task, 4000 resamples. Paired binary tests: exact McNemar on (s, r) pairs, descriptive only, because of within-unit dependence.

## Go / no-go gate (thresholds from the research brief; operationalisation frozen here)

| gate | rule |
|---|---|
| **A** | O_state − B ≥ **5 pp** (point estimate) with the bootstrap 95% lower bound > 0. Alternative Pareto route: O_state(λ) ≥ B + 2.5 pp at a mean cost ≤ the cost of the arm achieving B |
| **B** | at least **two** i ∈ 𝓘 with U_i ≥ **5%**, **and** U_i⁺ > U_I0P⁺ (each rescues more uniquely than the resampling placebo) |
| **C** | max_i R_i ≤ **90%** |
| **D** | O_state − O_task ≥ **2.5 pp** with the bootstrap 95% lower bound > 0 (within-task, state-level heterogeneity) |
| **E** | identity branch reproduces the reference continuation in ≥ 99% of decision states (any failure investigated); restore state and observation hash identity in 100% of branches; exceptions < 1% of branches, excluded and reported |

**GO** requires A ∧ B ∧ C ∧ D ∧ E. Otherwise the result is **NO-GO**, or **BLOCKED** if E fails. A NO-GO is reported plainly, and no router is trained.

Also reported, and part of the novelty kill check (`NOVELTY_BOUNDARY.md`), though not part of GO:
- the captured fraction of the failure-gated baseline (≥ 90% means risk ≡ utility);
- O_state vs best-of-placebo resampling at matched calls.

## What is not done

- No operator parameter is tuned, and no state from 25–49 is touched.
- No router is fitted in the pilot.
- No threshold changes after data.
