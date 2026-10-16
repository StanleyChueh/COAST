# Stage 2: common-backbone feasibility audit

Date: 2026-10-16. This is a code audit of COAST at `4239b15` (`adaptive-inference-2027` = `reproduce-aac`, both on top of `main` `2afa10e`).

Also inspected:
- the COAST reproduction branches `exp/*` (Phases 1A–8A, latest `exp/paired-mechanism-decomposition` `f8e93af`);
- the sibling repo `~/Stanley_ws/VLA-steering-2027` @ `de650cb`, read only.

## Environment record

| item | value |
|---|---|
| Python envs | root `.venv` 3.11.16 (openpi, PyTorch π0.5 serving); `examples/libero_env/.venv` 3.8.20 (LIBERO, robosuite 1.4.0, MuJoCo 3.2.3, NumPy 1.22.4); `examples/robocasa_env/.venv` 3.11.16 (robocasa 1.0.0, robosuite 1.5.2, MuJoCo 3.3.1); `groot_env/.venv` 3.10.12 (GR00T N1.5, torch 2.5.1) |
| submodules | Isaac-GR00T `4af2b62` (n1.5-release); libero `d63b117`; robocasa `9a3a786` (v1.0-12); robosuite `aaa8b9b` (v1.5.2-5) |
| π0.5 checkpoint | `checkpoints/openpi-libero-2000` (HF `brandonyang/openpi-libero-2000` @ `aaeeabc7…`); PyTorch `model.safetensors` sha256 `e649fa62…a10a761` |
| GR00T checkpoint | `checkpoints/groot_n15/gr00t_n1-5/multitask_learning/checkpoint-120000`; shard sha256 `08f18919…` / `deb9c9cf…` |
| activations | `activations/pi05-libero-activations-v1-2000-15env`: the COAST authors' fit rollouts for 4 LIBERO-10 tasks, states 0–14. 9.2 GB, with `rewards.npz` present for 4/60 episodes; labels come from `metadata.json` |
| hardware | 2× RTX 4090 (24 GB), 62 GB RAM |

## Capability matrix

| capability | **A. π0.5 + LIBERO (PyTorch)** | **B. GR00T N1.5 + RoboCasa** | C. OpenVLA / other |
|---|---|---|---|
| base / no intervention | ✓ `Policy.infer` | ✓ `groot_env/serve.py` | ✓ in the sibling repo (SCALE env), different stack |
| variable execution horizon | ✓ client `replan_steps` (1–10) | ✓ `--execution_mode fixed/aac` (1–16) | ✓ (OpenVLA single-step; OFT chunks) |
| multi-sample generation | ✓ explicit noise per sample; batched `infer_batched` exists | ✓ `infer_multi_sample` (shared conditioning, validated) | ✗ (token sampling only) |
| COAST-style latent steering | ✓ `ConceptorSteeringHook`. Per-request payload, paper configs and fit activations exist for 4 tasks (LR2a, KS3, KS4, LR1). Our reproduction found the operator ≈ shrinkage, with no held-out conceptor-specific effect | ⚠ hooks exist (`groot_steering.py`), but `experiments/robocasa/best_configs.json` is an empty placeholder and no GR00T conceptors or fit activations exist locally. Fully unvalidated | ✗ |
| semantic / counterfactual steering (CAG-like) | ✓ CAG-TF is a 2-pass language-mask guidance in model space. Ported here from the LIBERO-CF JAX code (≈ 40 lines); prior S2b evidence is on the official π0.5 checkpoint | ✗ nothing exists; GR00T's Eagle backbone would need a masked-language pass and a guidance rule not defined by any paper | ✗ |
| exact state snapshot / restore | ✓ **built and validated here** (`fork/sim_snapshot.py`: mjSTATE_INTEGRATION + robosuite python state + stored obs). LIBERO's own `set_init_state` is **not** exact (diverges on step 1) | ⚠ same robosuite family, likely feasible. But RoboCasa v1.0 `reset(seed)` is not reproducible across processes for 5 sink tasks and 2 fixture tasks (Stage 1, AAC notes) | n/a |
| RNG capture / restore | ✓ explicit server-side noise from an integer key (Phase-1D construction, golden fingerprint reproduced). Env has no stochasticity after reset | ⚠ GR00T draws `torch.randn` internally. Explicit-noise injection would need new code (upstream per-call reseed = 42 collapses this checkpoint, AAC notes) | — |
| internal pre-intervention features | ✓ action-expert hidden states via read-only hooks; K-sample spread; CAG contrast; IL displacement | ✓ DiT hidden states (collector), AAC entropies (logged for all 6,600 Phase-B episodes) | partial |
| efficient forked evaluation | ✓ LIBERO sim ≈ 10–20 ms/step; π0.5 eager ≈ 120 ms/call (240 ms CAG); 2 servers per GPU (7.3 GB each) | ⚠ RoboCasa ≈ 4–5 GB RAM per client (max ≈ 8 concurrent), slower sim | — |
| rollout noise (same state, other noise) | high: KS3 baseline 7–13/15 across seeds (Phase 1D) | high: 25% outcome flips between replicate runs (Stage 1) | low for π0.5 on LIBERO-CF (CR0, official ckpt) |
| prior evidence for each operator | IL: none beyond shrinkage. IC: +11.9 pp on LIBERO-CF (task-concentrated). IH: openpi default h = 5 | IH: AAC −2.0 pp vs fixed16, no state-level structure | SCALE: no detected gain at n = 30 |

## Decision

**Primary backbone for Stage 3–5: π0.5 (`openpi-libero-2000`, PyTorch) on LIBERO-10.** It is the only stack where every operator family (temporal, latent, semantic) exists in faithful form on **one** frozen model. It also has an exactly restorable simulator and controllable policy noise.

**GR00T + RoboCasa is not chosen for this pilot:**
- it has no CAG-like operator;
- it has no fitted COAST conceptors;
- noise injection would need new code;
- its reset nondeterminism complicates pairing;
- its only validated operator (horizon) showed no state-level structure in Stage 1.

**OpenVLA (C)** has no flow sampler, so no multi-sample, latent or CAG operators of the same kind, and it would add a third framework. Not justified.

Costs of choosing A, and how they are handled:
1. **CAG is ported from JAX to PyTorch.** It is validated by construction checks (conditional pass ≡ base bitwise; ω = 1 ≡ base to 6e-8), not by numerical parity with the JAX server, whose checkpoint differs.
2. **The checkpoint is COAST's 2,000-step checkpoint**, not the official `pi05_libero` on which CAG was validated. This choice is forced by IL: conceptors and paper configurations exist only for it.
3. **Only 4 LIBERO-10 tasks have IL configurations**, which limits task diversity in the pilot.
