# Intervention operators (frozen for the Stage-5 pilot)

Frozen 2026-10-16, before any pilot rollout. Backbone: **π0.5 (`pi05_libero` config), checkpoint `checkpoints/openpi-libero-2000`**.
- HF `brandonyang/openpi-libero-2000`, revision `aaeeabc72f8a50a8fa2d04544332c8ec1cd0142e`.
- PyTorch `model.safetensors` sha256 `e649fa62bf8c28f3fcedbfdef419f335d98b53ed40780417352f98635a10a761`.
- PyTorch, eager (no `torch.compile`), cuDNN deterministic, TF32 off.
- LIBERO-10, 10 settle steps, 520 post-settle steps, 224-px inputs.

Server: [`intervention_server.py`](intervention_server.py) (`IMPL_ID = aiv-intervention-server-v1`). Client: [`fork/libero_fork.py`](fork/libero_fork.py).

| id | operator | exact definition | parameters and their source | inference cost / call (RTX 4090, measured) |
|---|---|---|---|---|
| **I0** | base | `PI0Pytorch.sample_actions_with_steering(noise=n, steering_hooks=None)`, 10 Euler steps; replan every **5** steps (the openpi LIBERO default; the COAST reproductions used the same) | none | ≈ 120 ms |
| **I0P** | resampling placebo | I0 with independent noise (noise master + 1000) | none. It is a negative control for complementarity, and also the "re-roll the base policy" arm, i.e. the Reuse-Before-You-Retrieve style resampling family | ≈ 120 ms |
| **IH** | temporal | I0 inference, execution horizon **h = 2** (replan every 2 steps) | Stage 1 rated temporal routing LOW. A simple preregistered short-horizon operator is therefore used, not an adaptive selector, so no new horizon method is invented. h = 2 is the "more closed-loop" end of the fixed-h grid used in our AAC study. AAC itself is not used because the π0.5 multi-sample server of the AAC authors is not public (Phase 8A) | ≈ 120 ms, at 2.5× as many calls |
| **IL** | latent (COAST) | COAST `global` conceptor gate `h' = h·((1−β)I + βC)ᵀ` on action-expert layer *L* output, all tokens, all 10 denoising steps (`ConceptorSteeringHook`) | the COAST paper's Table 4 "Global" configuration per task (verified in Phases 4B / 7A): **LR2a (task 0) L11 α0.5 β0.3; KS3 (task 2) L5 α0.5 β0.1; KS4 (task 3) L5 α1.0 β0.1; LR1 (task 7) L11 α0.5 β0.3**. Conceptors come from the released builder (`compute_all_conceptors`, `C_s(I−C_f)` contrastive) run on the authors' fit activations (`brandonyang/pi05-libero-activations-v1-2000-15env`, init states 0–14; success/failure 8/7, 6/9, 6/9, 12/3, the same as the paper and Phase 7A) → `conceptors/aiv_pilot_conceptors.npz` | ≈ 123 ms |
| **IC** | semantic / counterfactual (CAG-TF) | second pass with `tokenized_prompt` and `tokenized_prompt_mask` zeroed *after* the input transforms, same noise; `a = a_u + ω (a_c − a_u)` in model (normalized) action space, then the output transform. This is a line-for-line port of LIBERO-CF `serve_policy_TF.py::CounterfactualActionGuidedPolicy` (JAX) to PyTorch | **ω = 1.5**, frozen in the sibling project (VLA-steering-2027 S2 preregistration) | ≈ 240 ms (2 passes) |

**Known validity caveats on the operators (recorded before the pilot):**
- **IL.** Our COAST reproduction (branches `exp/*`, Phases 5–7B, 4 LIBERO-10 tasks, held-out states) found no conceptor-specific behavioural effect: the operator is about 95–99% a uniform `(1−β)` shrink of h. IL is therefore an *unvalidated-for-benefit* operator; its implementation is validated. Phase 7A/7B used the paper candidate builder; this pilot uses the released `main` builder (frozen choice, recorded).
- **IC.** CAG was validated (+11.9 pp on LIBERO-CF, S2b) on the official `pi05_libero` checkpoint and on counterfactual-instruction tasks. Here it runs on the COAST 2,000-step checkpoint and on standard LIBERO-10 instructions, where its premise (vision overriding language) may not apply. The port has not been checked numerically against the JAX implementation: the two frameworks are not bit-compatible, and the sibling's JAX server uses a different checkpoint.
- **IH.** On GR00T/RoboCasa, Stage 1 found no state-level horizon effect beyond noise. It is included because temporal commitment is one of the three operator families in the research question.

## Identity tests (passed before the pilot)

[`validate_operators.py`](validate_operators.py) → `results/adaptive_inference/validation/operators.json`. Real LIBERO observations from tasks 0, 2, 3 and 7 (init 15), GPU 1.

| check | result |
|---|---|
| noise derivation reproduces the Phase-1D pinned fingerprint `87ae314a…` for key (100, 2, 15, 0) | ✓ |
| I0 (server path) vs stock `Policy.infer(obs, noise=…)` | **bitwise identical** (4/4 tasks) |
| I0 repeat | bitwise identical |
| IL with β = 0 (M = I) vs I0 | **bitwise identical**: disabling the latent operator is the base policy |
| IC with ω = 1 vs I0 | max \|Δ\| ≤ 6.0e-8 (float rounding of a_u + (a_c − a_u)); the CAG conditional pass is bitwise I0 |
| read-only feature hook | actions bitwise unchanged |
| feature bundle repeat | bitwise identical |
| IL / IC change the chunk | ‖a_IL − a_I0‖ 0.15–0.65, ‖a_IC − a_I0‖ 0.63–2.65 (L2 over 10×7) |
| peak GPU memory | 7.3 GB per server |

## Pre-intervention features (router inputs; computed only at decision states, from the reference state)

All of the following are computed at the decision state from the base noise key, before any branch:
- the first chunk proposed by I0, IL and IC under the same noise;
- K = 8 extra I0 samples;
- the CAG contrast norm ‖a_c − a_u‖;
- the IL displacement norm and intervention norm;
- the token-mean action-expert layer-11 hidden state at denoising steps 0 and 9 (1024-d each);
- the decision step (task phase);
- goal predicates satisfied at the state (progress).

**Never** used as inputs: branch outcomes, post-intervention activations, future observations.
