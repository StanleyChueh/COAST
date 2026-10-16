# Novelty boundary: intervention-response routing

Date: 2026-10-16. Inputs: [`LITERATURE_2025_2026.md`](LITERATURE_2025_2026.md); the sibling repo `~/Stanley_ws/VLA-steering-2027` @ `de650cb` (read only; `CR0_NOVELTY_KILL_CHECK.md`, `CR0_FEASIBILITY_DECISION.md`, the S2b/S2c0/S2d decisions); the COAST reproduction branches `exp/*` (Phases 1–8A); and [`HORIZON_ORACLE_REPORT.md`](HORIZON_ORACLE_REPORT.md).

## The claim we would need to own

The object is the treatment-effect vector at a decision state s_t of one frozen VLA:

    Δ_i(s_t) = E[Y | do(I_i), s_t] − E[Y | do(I_0), s_t]

Here the I_i are **heterogeneous automatic inference operators**. The branch label is the operator, never an action candidate.

The paper needs three things together:
- **(E1)** Δ is measured from identical restored states under controlled noise.
- **(E2)** Equal-risk states have different argmax_i Δ_i, and failure probability or scalar uncertainty does not predict the winner.
- **(E3)** A router that predicts the Δ vector before execution beats failure-gated and uncertainty-gated selection, task-ID routing, and best-of-K resampling at matched compute, on held-out scenes and/or tasks.

## Comparison with the closest work

| work | what it does | why it is not our claim | what it forces us to include |
|---|---|---|---|
| **RL²-VLA** (2607.26991) | SAFE failure score on action-expert latents with a conformal threshold. Each timestep it turns **one** operator (RL-latent flow composition + verifier) on or off. Shows steering helps more in likely-failure states | Binary *whether*, from P(fail). It never compares operators and never measures a per-operator effect from identical states | RL²-style failure gate (base vs strongest single operator) as a baseline. We must show P(fail) does not determine *which* operator helps |
| **UPS** (2602.22474) | Conformal sets over K samples; routes execute / clarify / ask a human, then learns residually | Routes on uncertainty set structure. Options include humans and training, not automatic operators | Uncertainty-source router as a baseline (where feasible) |
| **Act, Think or Abstain** (2603.05147) | GMM complexity / novelty on VL embeddings → act / reason / halt | Routes on *novelty*, not measured response. Abstain does not repair | Complexity / OOD-score router as a baseline |
| **SCALE** (2602.04208) | One self-uncertainty scalar modulates perception and action | One scalar drives one coupled operator | Scalar-uncertainty gate as a baseline; SCALE itself is a possible operator (not in the first portfolio) |
| **AAC / AutoHorizon / KWTS / GeoAAC / PICES / ChunkTrust / MoH** | A signal (entropy, attention, cross-attention, geometry, perturbation, spectral) → execution horizon (and the chunk, for PICES) | Horizon-specific selectors | A horizon operator may appear only as an existing method used as a treatment (e.g. AAC or a fixed short h). Stage 1 shows horizon has no detectable state-level effect beyond rollout noise on GR00T/RoboCasa, so it is a low-priority arm |
| **Reuse Before You Retrieve** (2608.17484) | Diagnoses recoverable headroom (pass@N − pass@1) vs retrieval complementarity per policy/suite. Episode-level selector over N retried rollouts | Episode-level and retry-based. Two families. The choice is drawn from diagnostics, not predicted per state before execution | **Best-of-K native resampling at matched compute is mandatory**, both as a negative control (base replicate branches) and as a baseline |
| **DoVLA-CIL** (in progress) | Restores identical states, evaluates **action** candidates, learns an action-utility field | Its treatments are actions. Ours are inference algorithms (each induces a closed-loop policy) | Branch identity = operator id. Never rank action chunks as "interventions" |
| **IMPACT-VLA** (2609.15005) | Closed-loop counterfactual re-execution for **attribution** of phase × modality blocks | Attribution of inputs, not selection of repair operators | Cite as methodological precedent for closed-loop re-execution; distinguish behavioural vs functional recovery as they do |
| **COAST** (2605.17144) | Per-task success-subspace latent steering, always on | Single operator, no routing. Our own reproduction found no held-out conceptor-specific benefit over random or pure-shrinkage controls on 4 LIBERO-10 tasks (Phases 5–7B) | IL must be accompanied by a matched shrinkage/random control. IL cannot be assumed beneficial |
| **Mostly Harmless** (2606.12299) | Learned *improvement head*: when will language steering improve performance (conformalized) | Benefit prediction for **one** operator family | Single-operator benefit gate as a baseline. Our increment is the vector over heterogeneous operators + E2 |
| **ELASTIC** (2606.31132) | RL meta-policy chooses denoising stride / parallel samples per denoising iteration | Options are homogeneous compute amounts of one sampler, trained by RL, no same-state Δ labels | Cost-aware frontier vs compute-only scaling |
| **BEST-Route** (ICML 2025, LLMs) | Per-query routing over (model, #samples) | Establishes per-input inference-strategy routing as a known idea in LLMs | The robotics contribution has to be the closed-loop causal labelling and E2, not "routing" per se |

## What would make the idea no longer novel enough

Fixed now, before any pilot data:

1. **Collision.** A paper appears that performs online state-level routing among ≥ 2 heterogeneous automatic VLA inference operators, using operator-specific utility estimated before execution. As of 2026-10-16 none was found. Re-check before any write-up and at every phase gate.
2. **Risk ≡ utility.** In matched-state data the oracle-best operator is (nearly) a function of failure probability or scalar uncertainty. For example, all high-risk states prefer the same operator, or an RL²-style gate captures ≥ 90% of the oracle gain. In that case the paper collapses into RL²-VLA / VLA-ATTC territory.
3. **Task ≡ utility.** A per-task operator choice (task-ID router) captures essentially all of the routing gain, i.e. no within-task state-level heterogeneity. It then becomes "per-task hyperparameter selection".
4. **Resampling ≡ utility.** Best-of-K native resampling or retry at matched compute matches the operator oracle. Then RBYR / LBYL already explain the phenomenon.
5. **One operator dominates.** One operator accounts for > 90% of beneficial branches. Then it is an always-on method plus a gate, i.e. RL² again.
6. **Noise ≡ complementarity.** Base-vs-base replicate branches from the same state show as much "complementarity" as different operators. Stage 1 already found exactly this for horizons on GR00T/RoboCasa.

## Status of the claim after Stage 0/1

- **Literature:** survives. No paper found that does E1+E2+E3. Components exist separately (RL² gating, Mostly-Harmless benefit head, ELASTIC state-dependent compute, RBYR headroom diagnosis), so the contribution has to be the combination, plus the empirical "risk ≠ utility" finding.
- **Prior empirical evidence in our own repos is unfavourable:**
  - Horizon: Stage 1 found no state-level horizon effect beyond noise.
  - COAST: our reproduction found no conceptor-specific held-out effect.
  - CAG: rescue concentrated in 2 of 13 tasks, so task identity dominates (S2b, CR0).
  - Base resampling alone: about 11 pp best-of-2 headroom on GR00T.

  Kill conditions 3, 4 and 6 are therefore the live risks, and the pilot must be designed to detect them (see `PILOT_PREREG.md`).

## Status after the Stage-5 pilot (2026-10-16)

The preregistered gate is **NO-GO** (`PILOT_GATE.md`). Against the kill conditions above:
- **condition 6 (noise ≡ complementarity) holds** for operator-vs-base effects: the resampling placebo rescues uniquely as often as any operator;
- **condition 3 (task ≡ utility) holds** for exploitable heterogeneity: the task oracle beats the state oracle.

The formal novelty survives the literature check, but the empirical claim is unsupported. The one open loophole is a post-hoc within-task operator-vs-operator variance (τ ≈ 0.14–0.20) that no measured feature predicts. See `RESEARCH_STATUS.md` for the single confirmatory test with a fixed kill rule.
