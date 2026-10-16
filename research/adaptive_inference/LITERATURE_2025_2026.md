# Literature collision matrix: inference-time interventions for frozen VLAs (2025–2026)

Compiled 2026-10-16.

**How it was verified.**
- Every arXiv identifier below was fetched from the arXiv API on 2026-10-16. Title, first author, submission date and comment field were checked against the brief. The raw API dump is `scratchpad/lit/arxiv*.xml` and is not committed.
- Rows marked **[full text]** were also read in HTML for the fields that matter for collision: backbone, decision granularity, what is selected, and how labels are obtained.
- All other rows rest on the abstract and the arXiv metadata. Cells not stated in the abstract are marked "n/s" (not stated) rather than guessed.

Corrections to the research brief:
- PICES (2610.19656) is titled *"Accelerating VLA Inference with Perturbation Informed Execution"*; PICES stands for "Perturbation-Informed Chunk and Execution Horizon Selection". v1 was submitted 2026-10-14 and v2 also exists.
- Mixture of Horizons is arXiv **2511.19433** (ICML 2026, PMLR 306).
- OneTwoVLA is arXiv **2505.11917** (ICLR 2026).
- "Mechanistic Interpretability for Steering VLAs" is arXiv **2509.00328** (CoRL 2025, PMLR 305; Häon et al.).
- Reuse Before You Retrieve is an **ECCV 2026 workshop** paper.
- GeoAAC is *submitted* to ICRA 2027.
- COAST is *submitted* to NeurIPS 2026.
- VLA-ATTC, RL²-VLA, IMPACT-VLA, CofactVLA, Act-Think-Abstain and Knowing When to Stop list no venue in their arXiv metadata.

New close neighbours not in the brief, found by search:
- **ELASTIC** (2606.31132);
- **ChunkTrust** (2609.39754);
- **Mostly Harmless VLA Steering** (2606.12299), which has a learned *improvement head*;
- **LBYL/SVA** (2607.03751);
- **BEST-Route** (ICML 2025, LLM routing; 2506.22716), an out-of-domain analogue.

DoVLA-CIL (Counterfactual Intervention Lattice) is in progress and not public. Its row is taken from the brief.

## Collision matrix

| Paper | Year / venue | Backbone | Signal | Decision granularity | Intervention | Training required? | Multiple candidate execution? | Main novelty | Overlap with our hypothesis | What we MUST do differently |
|---|---|---|---|---|---|---|---|---|---|---|
| **SCALE** 2602.04208 | 2026, ICML (spotlight) | autoregressive VLAs (OpenVLA in our sibling reproduction) | token-level self-uncertainty | per decoding step | one coupled operator: visual-attention temperature + action sampling temperature | no | no (single pass) | one uncertainty signal jointly modulates perception and action | uncertainty → intervention strength | not one scalar signal driving one operator; choose *which* operator from predicted response |
| **AAC** 2604.04161 | 2026, CVPR | GR00T N1.5, π0.5 | action entropy over N samples + min-motion bound | per policy call | execution horizon | no | N samples, executes one | entropy → chunk size | temporal intervention family | horizon is one arm at most; never an entropy → horizon map. Our Stage 1: horizon has no state-level effect beyond noise on GR00T/RoboCasa |
| **AutoHorizon / VLA Knows Its Limits** 2602.21445 | 2026, ECCV | flow-based VLAs | action self-attention weights | per chunk | execution horizon | no | no | attention as the model's predictive limit | same | same |
| **Mixture of Horizons** 2511.19433 | 2025/2026, ICML | π0, π0.5, π_reg | cross-horizon consensus (trained gate) | per chunk | horizon mixture + adaptive execution | yes (fine-tune) | no | train multiple horizons jointly | temporal family | we do not train the VLA |
| **Knowing When to Stop (KWTS)** 2609.00908 | 2026, arXiv (v3) | π0.5, X-VLA | action→observation cross-attention entropy plateau | per chunk | horizon truncation | no | no | cross-attention signal for horizon | temporal family | same as AAC |
| **GeoAAC** 2609.20776 | 2026, ICRA'27 submission | GR00T N1.5, π0.5 | geometry of flow-denoising trajectories | per chunk | horizon | no | no | denoising geometry → horizon | temporal family; same backbone (GR00T N1.5, RoboCasa365) | same |
| **PICES** 2610.19656 [full text] | 2026-10-14, arXiv | π0.5, GR00T N1.5/N1.7 | sensitivity of each chunk to local noise perturbation | per policy call | joint chunk selection + horizon | no | yes (nominal + perturbed batch; executes 1) | perturbation-informed chunk + horizon | perturbation probing of one operator family | perturbation is a feature at most; treatments are operators, not chunks |
| **ChunkTrust** 2609.39754 | 2026, arXiv | π0.5, Qwen3GR00T | spectral stability + inter-chunk continuity, Beta posterior; optional learned prior | per replan | horizon | optional light adapter | no | horizon as a latent variable | temporal family with a learned context prior | same |
| **VLA-ATTC** 2605.01194 | 2026, arXiv | π0.5 (LIBERO-LONG) | uncertainty "cognitive clutch" | per step | extra candidate generation + Relative Action Critic | yes (critic) | yes | uncertainty gates test-time compute | scalar gate → one heavier operator | gate is not a choice among heterogeneous operators; ours predicts Δ per operator |
| **RL²-VLA** 2607.26991 [full text] | 2026, arXiv | OpenVLA, π0, π0.5 (SIMPLER, PolaRiS) | SAFE LSTM failure score on action-expert features, time-varying conformal threshold | per timestep | **one** operator: RL-latent flow composition + verifier (else base resampling) | yes (offline RL policy, failure detector) | yes (K candidates + verifier) | failure-gated compositional steering; "scaling laws differ in success vs failure states" | **closest gating paper**: per-state, pre-execution, internal latents | must show failure risk ≠ intervention utility. RL² decides *whether* to steer from P(fail); we decide *which* operator from per-operator Δ. RL²-style gate is a mandatory baseline |
| **Act, Think or Abstain** 2603.05147 | 2026, arXiv | SmolVLA, π0 | GMM / estimators on fused VL embeddings (state complexity / OOD) | per state | route among act / reason / abstain | yes (density estimators) | no | complexity-aware routing of execution modes | routing among heterogeneous responses | routes on novelty/complexity, not measured response; abstain is not an automatic repair. Ours is labelled by same-state outcome differences |
| **OneTwoVLA** 2505.11917 | 2025/2026, ICLR | π0 (fine-tuned) | learned decision tokens ([BOR]/[BOA]) | per step | reason vs act | yes (co-training) | no | unified adaptive reasoning/acting | adaptive mode switch | frozen VLA, no retraining; operators are inference algorithms |
| **UPS** (When to Act, Ask, or Learn) 2602.22474 | 2026, RSS | diffusion policy + VLM verifier | conformal sets over K samples (semantic + action uncertainty) | per state | execute / clarify / human intervention → residual learning | yes (residual) | yes (K samples) | calibrated choice of uncertainty-resolution strategy | per-state routing among heterogeneous responses | UPS's options include humans and learning; routing is from uncertainty sets. Ours: automatic operators, utility from same-state counterfactual outcomes |
| **Mech. Interp. for Steering VLAs** 2509.00328 | 2025, CoRL | π0, OpenVLA | FFN directions projected to token space | per episode (fixed steering) | activation steering (speed, direction) | no | no | interpretable semantic steering | latent intervention family | a candidate operator, not a router |
| **COAST** 2605.17144 | 2026, NeurIPS submission | π0.5, π0-FAST, DP, GR00T N1.5 | success/failure conceptor subspace (per task) | always-on per task | latent residual steering | no (fit from rollouts) | no | contrastive conceptor steering | latent family (IL) | our reproduction (exp/* branches) found no held-out conceptor-specific gain over random or pure shrinkage on 4 LIBERO-10 tasks; IL must be treated as an unvalidated operator, with shrinkage as its control |
| **World Pilot** 2606.12403 | 2026, arXiv | VLA + World-Action Model | WAM scene-evolution latent + trajectory prior | always on | perception latent steering + action prior | yes (WAM, integration) | no | two complementary steering pathways | multi-pathway steering | always-on combination, no routing; not frozen-VLA-only |
| **Reuse Before You Retrieve** 2608.17484 [full text] | 2026, ECCV workshop | π0, SmolVLA, π0.5 (LIBERO); OpenVLA (SimplerEnv) | recoverable headroom H_N = pass@N − pass@1; demonstration-fit error | **episode** (retry / parallel execution) | two families: resampling + episode selector; external retrieval | no (frozen encoder scorer) | yes (N full rollouts) | diagnoses which augmentation opportunity exists | **closest diagnostic**: "which intervention does a deployed policy need" | theirs is per policy/suite, chosen from rollout diagnostics, and needs retries. Ours is online per *state*, before execution, among ≥ 2 heterogeneous inference operators, with labels from same-state branches. **Resampling (best-of-K) must be a baseline arm**: our Stage 1 shows best-of-2 alone recovers about 11 pp on GR00T |
| **IMPACT-VLA** 2609.15005 | 2026, arXiv | OpenVLA-OFT | closed-loop counterfactual re-execution of phase × modality blocks | per phase | input replacement (attribution, not repair) | no | re-executions | phase-dependent modality attribution | same-state counterfactual re-execution as methodology | ours re-executes *inference operators* to estimate repair utility, not input attribution |
| **CofactVLA** 2608.04396 | 2026, arXiv | flow VLAs (n/s in abstract) | language-masked counterfactual branch | per step | orthogonal projection guidance + covariance reduction | yes | no | causal deconfounding | semantic/counterfactual operator family (IC) | candidate operator, not a router |
| **CAG / LIBERO-CF** 2602.17659 | 2026, arXiv | π0.5 (+ VA module) | conditional − language-masked action contrast | per call (always on) | counterfactual action guidance (ω) | no (TF variant) | 2 passes | CFG-like language guidance | IC operator; our sibling S2b: +11.9 pp on LIBERO-CF, rescue concentrated in 2 tasks | as IC operator only |
| **Mostly Harmless VLA Steering** 2606.12299 | 2026, arXiv | arbitrary frozen VLAs | learned **improvement head** predicting when language steering helps (conformalized) | per state / episode | language feedback policy (one operator) | yes (LFP + head) | search during training | benefit prediction for one steering operator | **predicts intervention benefit, not failure**, but for a single operator family | ours predicts a *vector* of operator-specific Δ and must show that operators disagree at equal risk; improvement-head-style single-operator gating is a baseline |
| **ELASTIC** 2606.31132 [full text] | 2026, arXiv | Diffusion Policy; π0.5 (LIBERO-10, DROID) | meta-policy on observation embeddings (SAC) | per denoising iteration | compute allocation only: denoising stride + parallel sample pruning | yes (RL meta-policy, sim rollouts) | yes (parallel samples + verifier) | state-dependent test-time compute schedules | **state-dependent choice among inference options**, learned from rollouts | options are homogeneous compute amounts of one sampler, trained by RL. Ours: heterogeneous operators (horizon / latent / semantic …), labelled by same-state counterfactual Δ, plus an explicit test that risk ≠ utility |
| **LBYL / SVA** 2607.03751 | 2026, arXiv | frozen VLAs (9B/27B) | MCTS-distilled Q evaluator | per step | best-of-N candidate selection | yes (Q model) | yes | evaluation bottleneck; pass@k 33→92% | headroom via sampling | action candidates are not our treatments |
| **BOKBO** 2605.30660 | 2026, arXiv | OpenVLA-OFT, π0-FAST | learned violation predictor + conformal risk control | per decision | abstain if all K bad | yes | yes | calibrated abstention | uncertainty-based gating | abstention is not a repair operator |
| **ReSteer** 2603.17300 | 2026, arXiv | multitask policies (LIBERO) | steerability / CMI proxy | per state | data generation + fine-tuning | yes | no | quantifying steerability | per-state rollout-measured response to a task switch | training-based repair; ours is inference-only |
| **ECT** 2609.39971 | 2026, arXiv | π0.5, GR00T N1.7 | readouts; prefix-KV interventions (diagnostic) | episode | training (ECT loss/data) | yes | no | instruction–action binding failure | KV injection is a possible representation-level operator | — |
| **FailPatch** 2609.34175 | 2026, arXiv | VLA (RoboTwin) | linear separability of reliable vs failure hidden states | per state | gated residual expert bank (learned) | yes (0.52% params) | no | failure-driven residual patching | failure-gated learned operator selection among experts | experts are trained patches; ours chooses among frozen inference algorithms |
| **AutoIntervene** 2608.07065 | 2026, arXiv | action-chunking IL policies | visual–action support memory, calibrated thresholds | per chunk | human takeover | retrains on interventions | no | calibrated takeover | — | automatic operators only |
| **DoVLA-CIL** (in progress; brief only) | — | — | same-state restoration | per state | **action candidates** evaluated from identical simulator states → action-utility field | yes (utility field) | yes | counterfactual action lattice | same-state restore methodology | our treatments are inference operators, never action candidates; branch label = operator id |
| BEST-Route 2506.22716 (LLM analogue) | 2025, ICML | LLMs | query difficulty → (model, #samples) | per query | routing among models × best-of-n | yes (router) | yes | cost-aware routing over test-time compute | per-input routing of inference strategy with a cost term | establishes per-input inference-strategy routing as known in LLMs. A robotics paper must contribute the closed-loop, same-state causal labelling and the risk ≠ utility evidence, not the routing idea itself |

## Summary of the collision landscape

1. **Temporal (horizon) selection is saturated.** Six or more methods exist: AAC, AutoHorizon, MoH, KWTS, GeoAAC, ChunkTrust, PICES. Any horizon component of ours must be an existing operator used as a treatment, never a new selector.
2. **Gating a single operator on failure, uncertainty or complexity is taken.**
   - RL²-VLA gates on failure.
   - VLA-ATTC and SCALE gate on uncertainty.
   - Act/Think/Abstain gates on complexity.
   - BOKBO gates on risk.
   - FailPatch gates on failure state.
3. **Predicting the *benefit* of one intervention is taken in a narrow form.**
   - Mostly Harmless has an improvement head for language steering.
   - ELASTIC uses a learned value for compute allocation.
   - RBYR uses episode-level headroom diagnostics.
4. **Not found:** online, state-level routing among **≥ 2 heterogeneous automatic inference operators on one frozen VLA**, where the router is trained on **operator-specific treatment effects measured from identical restored states**, together with direct evidence that **equal-risk states prefer different operators**. Searches covered arXiv listings to 2610.20318, the venues above, and keyword searches for "router / meta-policy / treatment effect / uplift / per-state intervention selection" over VLA steering.

Search queries used, 2026-10-16 (WebSearch, standard and extended):
- "VLA inference-time intervention router select among steering methods per state treatment effect";
- "robot policy adaptive selection among multiple test-time strategies per state (resampling, steering, replanning horizon) learned selector 2026";
- "vision-language-action meta-policy OR router selects intervention among guidance, activation steering, resampling, horizon";
- "robot manipulation policy treatment effect OR uplift per-state intervention selection counterfactual simulator branching 2026";
- "arXiv 2026 VLA which intervention test-time adaptation selection counterfactual rollouts same state";
- plus title lookups for MoH, OneTwoVLA, the Mechanistic Interpretability paper and LLM routing.

The sibling repo's `reports/CR0_NOVELTY_KILL_CHECK.md` (VLA-steering-2027 @ `de650cb`) adds LBYL, ReSteer, ECT, BOKBO, Mostly Harmless, SDN, Golden Ticket, Proxy Policy Steering, Feasible-Future Decoding and CRAFT. None of them collides with item 4.
