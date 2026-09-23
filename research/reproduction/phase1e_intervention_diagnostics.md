# Phase 1E — Intervention Diagnostics: What Hidden-State Change Does COAST Make?

**Development split only (`--seed 15`, init states 15–29). States 30–44 remain untouched.**
This is an instrumentation and measurement phase. It does not redesign COAST, change the conceptor math, or tune anything. Everything below is descriptive. Nothing here establishes that the intervention causes any behavioral outcome.

Paper: *Contrastive Conceptor Activation Steering (COAST)*, arXiv:2605.17144.
Base commit: `f28021643c8aa9bc93bb4fce0a81d642456e0d52` (Phase 1D). Branch `exp/intervention-diagnostics`. The run used the uncommitted working tree described in §2. `sha256(git diff HEAD -- src scripts examples packages)` at run time was `d558bf4b…f675a2`.

## 0. Summary

- **At layer 5 the COAST intervention is almost exactly a uniform shrink of the residual stream: h′ ≈ 0.90·h.** Over all 9,220 hook applications:
  - The mean per-token cos(h′ − h, h) is **−0.99962** (range −0.99980 to −0.99850).
  - The relative change ‖h′ − h‖/‖h‖ is **0.0994 ± 0.0003** (range 0.0980–0.1000).
  - This follows from the conceptor itself. The V0 `C_contrastive` at L5/α=0.5 has eigenvalues in [−0.005, 0.352], with median 0.0013 and trace/d 0.018. So M = 0.9·I + 0.1·C has eigenvalues of only about 0.8995–0.935 in any direction.
  - For the hidden states actually visited, the implied Rayleigh quotient hᵀCh/‖h‖² is about **0.006**: those states lie almost entirely outside the conceptor's high-aperture subspace.
- **"Where" the intervention is largest is decided by ‖h‖, not by COAST.**
  - Because the relative change is essentially constant, the absolute ‖Δh‖ simply follows ‖h‖. It falls steadily across the 10 denoising steps, from **22.30** at step 0 to **13.39** at step 9 (×1.665). ‖h‖ falls by ×1.653 over the same steps.
  - Across rollout time and across episodes, the absolute change is almost constant. Per-episode means range from 16.87 to 17.03 (about 1%).
- **Successful and failed episodes cannot be told apart on intervention magnitude.**
  - Over whole episodes: success 16.975 vs failure 16.948, a 0.16% gap. Over the common window of rollout steps 0–200: success 16.972 vs failure 17.008, and the sign reverses.
  - Two whole-episode associations have small nominal p-values: relative delta (Mann–Whitney exact p = 0.003) and cosine (p = 0.018). Both involve differences of about 10⁻⁵, and both **disappear in the common window** (p = 0.95 and 0.85). That pattern fits the confound that failed episodes run to the 520-step timeout and pass through task phases successful episodes never reach. It does not suggest a property of the intervention.
- **The diagnostics did not perturb the policy measurably.**
  - The run reproduced Phase 1D's repository-faithful condition on 15/15 outcomes, with all 922 noise fingerprints identical.
  - 14/15 rollout lengths matched. State 25 succeeded at 222 steps instead of 221 (§6.3).
  - On a fixed input with fixed noise, actions with diagnostics on and off were bit-identical on the real checkpoint.

## 1. Question and scope

Phases 1B–1D showed that the paper's formulation and the released implementation build very different conceptor matrices (Phase 1B), yet this difference produced no observable behavioral difference on this task and configuration (Phases 1C/1D). Phase 1E instruments the steering hook to answer three descriptive questions:

1. How large is the hidden-state change?
2. At which denoising steps does it occur, and how does it vary across them?
3. How does its size differ between successful and failed trajectories?

Fixed configuration (same as Phase 1D condition B):
- **Model:** pi0.5 (`pi05_libero`, PyTorch).
- **Checkpoint:** `checkpoints/openpi-libero-2000`.
- **Task:** libero_10 task 2, `KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it`.
- **Episodes:** `--seed 15 --num_episodes 15` (episode k → init state 15+k; `env.seed(15)`).
- **Rollout:** max 520 steps, replan every 5.
- **Steering:** layer 5, α 0.5, β 0.1, `global`. **Not tuned.**
- **Conceptor:** repository-faithful **V0**, `conceptors/phase1b_repo_task02.npz` (sha256 `8b1c802a…ec80f3`).
  - The brief did not say which conceptor to use. V0 was chosen because it is the conceptor the released pipeline actually runs, and because Phase 1D already ran this exact condition, which turns the run into a replay check (§6.3).
  - Paper V3 was **not** instrumented (one-run stop condition).
- **Paired noise:** on, master seed **100**.
- **GPU:** GPU 1 (RTX 4090). Phase 1D used GPU 0, the same model. GPU 1 had the lowest memory use at launch.

## 2. Code changes

| File | Change |
|---|---|
| `src/openpi/serving/steering.py` | New `intervention_summary(h, h_steered)` returns scalar summaries only. `ConceptorSteeringHook` and `LinearSteeringHook` gain keyword args `layer=None, record_diagnostics=False`. When recording is on, each call appends `{layer, denoising_step, beta or alpha, **summary}` to `hook.diagnostics`, and `reset_logs()` clears it. `SteeredPolicyWrapper(..., record_diagnostics=False)` labels each hook with its payload layer. On **steered** calls it attaches `result["steering_diagnostics"]` (that call's records only; the hook is reset per call). Metadata gains `steering_diagnostics_enabled`. pi0-fast combined with diagnostics raises an error, since it has no hooks. |
| `packages/openpi-client/src/openpi_client/steering.py` | Wire constant `STEERING_DIAGNOSTICS_KEY = "steering_diagnostics"`. |
| `scripts/serve_policy.py` | Opt-in `--steering-diagnostics` flag. It requires `--steer` and rejects pi0-fast. |
| `examples/libero_env/main.py` | Opt-in `--log_steering_diagnostics`. Requires `--steer` and server metadata `steering_diagnostics_enabled`, and fails if any steered response lacks records. Each record is tagged with `episode, init_state, rollout_step, alpha, strategy` and appended to `<task_output_dir>/steering_diagnostics.jsonl` (gitignored output). |
| `tests/test_steering_diagnostics.py` (new, 35 tests), `tests/models/test_steering_diagnostics_gpu.py` (new, 4 manual tests), `tests/test_serve_policy.py` (+3 tests; stub accepts `record_diagnostics`) | See §4. |
| `research/reproduction/tools/analyze_phase1e.py` (new) | Validation and analysis (§5–7). |

**What is unchanged.**
- The steering equation and `h_steered` are computed exactly as before, `h_steered = h @ M.T` with `M = (1−β)I + βC` cast to h's dtype. Recording only reads `h` and `h_steered` through out-of-place ops (`detach().to(float32)`).
- The legacy `intervention_norms` log is kept unchanged.
- Model weights, preprocessing, denoising, action decoding and conceptor construction are untouched.
- With the flag off, the hooks do no extra work, and responses and metadata values are unchanged apart from the added `steering_diagnostics_enabled: False` metadata field.

**Denoising-step tracking (verified from source, not assumed).**
- `PI0Pytorch.sample_actions_with_steering` registers the forward hook on `gemma_expert.model.layers[layer]` only **after** the prefix/KV-cache pass (`pi0_pytorch.py:871-877`).
- Before each Euler step of the `while time >= -dt/2` loop, it calls `hook.set_denoise_step(step_counter)` (`pi0_pytorch.py:887-894`).
- The hook therefore fires exactly once per denoising step, with `current_denoise_step` = 0…9. No new context mechanism was needed.
- The layer label comes from the steering payload (the cache key's layer), so nothing is hardcoded to layer 5.

## 3. Diagnostic schema (one JSONL line per hook application)

| Field | Source | Meaning |
|---|---|---|
| `episode`, `init_state`, `rollout_step` | client | rollout coordinate (`rollout_step` = post-settle step at which a chunk was requested: 0, 5, 10, …) |
| `layer` | hook (from payload) | expert layer the hook is attached to |
| `denoising_step` | hook (set by sampler) | 0…9 |
| `beta` (conceptor) / `alpha` (linear) | hook | hook parameter; the client also adds `alpha`, `strategy` from its args |
| `token_count` | hook | number of token rows in h (leading dims flattened) — 10 action tokens for pi0.5 |
| `hidden_dim` | hook | 1024 |
| `mean_delta_norm`, `max_delta_norm` | hook | mean / max over tokens of ‖h_steered − h‖₂ |
| `mean_relative_delta`, `max_relative_delta` | hook | mean / max over tokens of ‖h_steered − h‖ / ‖h‖ |
| `mean_hidden_norm` | hook | mean over tokens of ‖h‖ |
| `mean_cosine_delta_hidden` | hook | mean over tokens of cos(h_steered − h, h); −1 means a pure shrink along h |

- Norms are computed in float32 from the tensors the model actually sees. At this hook, h is **bfloat16, shape (1, 10, 1024)**, a tuple output (verified on GPU), so Δh is the change realized after bf16 rounding.
- A zero-norm token yields 0 rather than NaN.
- No tensors, hidden states or matrices are stored. Raw log: 9,220 lines, 3.9 MB, gitignored and not committed.

`mean_cosine_delta_hidden` goes beyond the brief's required fields. It was added after the first GPU test showed the relative change sitting almost exactly at β. The existing fields could not tell a shrink along h from a same-size change in another direction.

## 4. Tests

CPU (`tests/test_steering_diagnostics.py`, 35 tests). These cover the hooks directly, plus a real `Policy` inside `NoiseControlledPolicyWrapper(SteeredPolicyWrapper(Policy))` with a fake model. The fake model's `sample_actions_with_steering` follows the PI0Pytorch hook contract: register on `layers[i]`, call `set_denoise_step(t)` before each of 10 expert passes, then remove the hook.

| Req. | Tests | Tolerance / result |
|---|---|---|
| 1 disabled → none | default hook records nothing (global/per_step/linear); wrapper without flag → no key, metadata False; diagnostics server + unsteered request → no key | pass |
| 2 enabled → produced | one record per application with steps 0..9, the payload's layer, `token_count` = horizon; through the full stack, 10 records per call and per-call only on a cached hook; linear strategy records alpha | pass |
| 3 β = 0 → ~0 | fp32 and bf16 hook: every magnitude field **== 0.0 exactly**; full stack β = 0: zero magnitudes and actions equal to the unsteered call (`assert_array_equal`) | exact |
| 4 finite | bf16 input including a zero token: all float fields finite; zero-h summary gives 0, not NaN | pass |
| 5 no hidden states | record values are JSON scalars only; hook's tensor attributes identical before/after 10 calls; response payload `json.dumps`-able; magic keys never reach input transforms | pass |
| 6 behavior unchanged | hook outputs `torch.equal` with diagnostics on vs off (global/per_step/linear × plain/tuple output), legacy `intervention_norms` identical; full stack actions identical on vs off, and the only added response key is `steering_diagnostics` | exact |
| — | `intervention_summary` vs independent NumPy (rel 1e-6; cosine rel 1e-5); pure shrink 0.9·h → cosine −1, relative 0.1; pi0-fast + diagnostics rejected | pass |

`tests/test_serve_policy.py` adds 3 tests: flag without `--steer` rejected, pi0-fast rejected, and pi0.5 passes `record_diagnostics=True` to the wrapper.

Manual GPU tests (`tests/models/test_steering_diagnostics_gpu.py`, 4 tests) run on the real checkpoint, the V0 NPZ, and the **real** `sample_actions_with_steering` (GPU 1, 31.5 s):
- 10 records, steps 0..9, layer 5, `token_count` 10, `hidden_dim` 1024, all finite.
- β = 0 gives every magnitude **exactly 0.0**.
- Actions with diagnostics on vs off, same noise: **max |diff| = 0.0** (`np.array_equal`).
- An unsteered request carries no diagnostics.

The Phase 1D GPU tests (7) were re-run on the modified tree and all passed.

Regression. The full CI suite, run with `CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu uv run pytest --strict-markers -m "not manual"`, gave **388 passed, 47 skipped, 0 failed**. Phase 1D had 350 passed; the difference is exactly the 38 new tests. `ruff check` and `ruff format --check` are clean on all changed files.

## 5. Experiment

Server (repo root; started 2026-09-23 23:42:33 +08:00):
```
CUDA_VISIBLE_DEVICES=1 uv run --no-sync scripts/serve_policy.py --pytorch --steer --noise-control --steering-diagnostics \
  --conceptor-npz conceptors/phase1b_repo_task02.npz --port 8101 \
  policy:checkpoint --policy.config pi05_libero --policy.dir checkpoints/openpi-libero-2000
```
Client (`examples/libero_env`; ran 23:43:17–23:46:00, exit 0):
```
CUDA_VISIBLE_DEVICES=1 MUJOCO_GL=egl LIBERO_CONFIG_PATH=/home/stanley/Stanley_ws/COAST-research/libero_config_coast \
  uv run --no-sync python main.py --task_suite_name libero_10 --task_id 2 --num_episodes 15 --seed 15 --port 8101 \
  --output_dir output/phase1e_repo_diag_task02_seed15 --policy_noise_seed 100 \
  --steer --steering_layer 5 --steering_alpha 0.5 --steering_beta 0.1 --steering_strategy global --log_steering_diagnostics
```
- Executed exactly once.
- Server log: 0 Traceback/ERROR lines, one line `Built steering hook ('KITCHEN_SCENE3_…', 5, 0.5, 0.1, 'global') [ConceptorSteeringHook] (cache size=1)`.
- GPU 1 peaked at 8,360 MiB and 74% utilization.
- The client showed only the known benign warnings (robosuite private-macro notice, LIBERO `datasets` path missing).

Analysis (CPU): `CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu uv run python research/reproduction/tools/analyze_phase1e.py`. It asserts:
- every inference call has exactly 10 records with steps 0..9;
- layer is always 5, `token_count` 10, `hidden_dim` 1024, β/α/strategy as configured;
- all values are finite;
- each episode's request schedule is 0, 5, … up to its rollout length;
- the diagnostic request schedule equals the noise-fingerprint schedule.

All assertions passed.

## 6. Results

### 6.1 Counts (report items 1–3)

| | |
|---|---|
| Episodes | **15** (11 successes) |
| Inference calls | **922** |
| Steering-hook applications | **9,220** (= 922 × 10 denoising steps × 1 layer) |

### 6.2 Magnitude distribution over all 9,220 applications (item 4)

| field | mean | std | min | p5 | median | p95 | max |
|---|---|---|---|---|---|---|---|
| mean ‖Δh‖ | 16.963 | 2.668 | 12.738 | 13.379 | 16.656 | 22.279 | 24.101 |
| max ‖Δh‖ (worst token) | 18.429 | 3.419 | 12.967 | 13.892 | 17.993 | 24.869 | 29.600 |
| mean ‖Δh‖/‖h‖ | 0.09942 | 0.00027 | 0.09801 | 0.09886 | 0.09949 | 0.09973 | 0.10001 |
| max ‖Δh‖/‖h‖ | 0.10005 | 0.00022 | 0.09893 | 0.09970 | 0.10005 | 0.10041 | 0.10085 |
| mean ‖h‖ | 170.56 | 26.51 | 128.91 | 135.15 | 167.41 | 223.58 | 241.63 |
| mean cos(Δh, h) | −0.99962 | 0.00015 | −0.99980 | −0.99977 | −0.99965 | −0.99934 | −0.99850 |

- The spread in absolute ‖Δh‖ (CV 16%) comes almost entirely from the denoising step (§6.4). The relative change has a CV of 0.3%.
- A relative change slightly above β (max 0.10085) is not possible for a PSD conceptor with eigenvalues in [0, 1) in exact arithmetic. Two things allow it here:
  - This C is slightly indefinite and asymmetric (min eigenvalue −0.0053, max |C − Cᵀ| 0.0035).
  - M is rounded to bf16 when applied. Its diagonal lands on multiples of 1/256 (0.8984, 0.9023, …, 0.9180). Offline, this rounding perturbs M − I by **1.4% (Frobenius)**, almost all on the diagonal (max off-diagonal error 7.5e-6).

  So the realized per-dimension shrink is not exactly β.

### 6.3 Replay check against Phase 1D condition B (same config, same noise)

| | Phase 1E vs Phase 1D repo |
|---|---|
| success outcomes | **15/15 identical** (11/15 both) |
| noise fingerprints | **922/922 identical** (same 922 request coordinates) |
| rollout lengths | **14/15 identical**. State 25: 222 (1E) vs 221 (1D), success in both |

- The one-step difference at state 25 left the request schedule unchanged (45 chunks in both runs).
- Its cause was **not isolated**. Phase 1E ran on a different physical GPU (GPU 1 vs GPU 0, same model), and GPU/bf16 kernel nondeterminism is possible.
- The diagnostics themselves produced bit-identical actions on a fixed input on GPU 1. That rules out a *deterministic* effect of the instrumentation on a single inference, but does not rule out every possible cause.
- Separating these would take a second run, which the stop condition does not allow.

### 6.4 By denoising step (item 5)

| step | mean ‖Δh‖ (± std, min–max over 922 calls) | mean ‖h‖ | mean ‖Δh‖/‖h‖ | mean cos(Δh, h) |
|---|---|---|---|---|
| 0 | **22.298** ± 0.484 (20.77–24.10) | 223.80 | 0.09963 | −0.99976 |
| 1 | 19.930 ± 0.431 (18.56–21.47) | 200.02 | 0.09964 | −0.99975 |
| 2 | 18.672 ± 0.373 (17.53–20.01) | 187.45 | 0.09961 | −0.99970 |
| 3 | 17.796 ± 0.321 (16.81–18.98) | 178.72 | 0.09957 | −0.99968 |
| 4 | 17.168 ± 0.279 (16.34–18.20) | 172.49 | 0.09953 | −0.99965 |
| 5 | 16.121 ± 0.232 (15.42–17.02) | 162.08 | 0.09946 | −0.99963 |
| 6 | 15.486 ± 0.210 (14.90–16.37) | 155.81 | 0.09939 | −0.99960 |
| 7 | 14.769 ± 0.205 (14.22–15.58) | 148.74 | 0.09929 | −0.99956 |
| 8 | 13.990 ± 0.217 (13.37–14.76) | 141.13 | 0.09913 | −0.99948 |
| 9 | **13.394** ± 0.259 (12.74–14.33) | 135.39 | 0.09893 | −0.99934 |

- The intervention is applied at **every** denoising step, as `global` implies.
- Its absolute size decreases monotonically from step 0 (x_t = pure noise, t = 1) to step 9 (t = 0.1). ‖h‖ at layer 5 decreases in parallel.
- The relative change and the cosine drift only slightly: from 0.0996 to 0.0989, and from −0.99976 to −0.99934.
- That drift means the component of h inside the conceptor's subspace grows a little in the late denoising steps. The implied hᵀCh/‖h‖² ≈ 1 − rel·|cos|/β rises from about 0.004 to about 0.011. This is an approximation built from token means.

Machine-readable: `research/reproduction/experiments/phase1e_denoising_step_intervention.csv`.

### 6.5 Active layer (item 6)

Only layer 5 was instrumented, since it is the only active steering layer.

| field | mean |
|---|---|
| mean ‖Δh‖ | 16.963 |
| worst-token ‖Δh‖ | 18.429 |
| mean ‖Δh‖/‖h‖ | 0.09942 |
| mean ‖h‖ | 170.56 |
| mean cos(Δh, h) | −0.99962 |

Implied effective scale, if the change is a pure shrink: h′ ≈ **0.9006·h**.

### 6.6 Success vs failure (items 7–8)

- Per-episode means are averaged over all of that episode's hook applications. Groups: 11 successes and 4 failures (init states 15, 18, 24, 28; all timeouts at 520 steps).
- "Common window" means rollout steps 0–200, which every episode queried; the shortest successful episode lasted 201 steps.
- p-values come from 8 uncorrected tests at n = 15 and are listed **for reference only**.

| metric | window | success mean (sd) | failure mean (sd) | fail − succ | point-biserial r | Mann–Whitney U, exact p |
|---|---|---|---|---|---|---|
| mean ‖Δh‖ | whole episode | 16.975 (0.050) | 16.948 (0.027) | −0.026 | +0.27 | 32, p = 0.23 |
| mean ‖Δh‖ | steps 0–200 | 16.972 (0.050) | 17.008 (0.031) | +0.036 | −0.35 | 12, p = 0.23 |
| ‖Δh‖/‖h‖ | whole episode | 0.099411 (1.0e-5) | 0.099428 (0.8e-5) | +1.7e-5 | −0.64 | 1, p = 0.003 |
| ‖Δh‖/‖h‖ | steps 0–200 | 0.099408 (0.9e-5) | 0.099410 (0.6e-5) | +1e-6 | −0.07 | 23, p = 0.95 |
| cos(Δh, h) | whole episode | −0.999610 | −0.999621 | −1.1e-5 | +0.58 | 40, p = 0.018 |
| cos(Δh, h) | steps 0–200 | −0.999608 | −0.999607 | +1e-6 | −0.05 | 20, p = 0.85 |
| mean ‖h‖ | whole episode | 170.70 (0.49) | 170.41 (0.28) | −0.29 | +0.30 | 32, p = 0.23 |

Application-weighted mean ‖Δh‖: success 16.974 (5,060 applications) vs failure 16.948 (4,160).

Mean ‖Δh‖ in 50-step rollout bins:

| bin | 0–49 | 50–99 | 100–149 | 150–199 | 200–249 | 250–299 | 300–349 | 350–399 | 400–449 | 450–499 | 500–519 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| success | 16.999 | 16.936 | 17.063 | 16.883 | 17.008 | 16.847 | — | — | — | — | — |
| failure | 17.002 | 16.957 | 17.134 | 16.959 | 16.993 | 16.977 | 16.941 | 16.797 | 16.913 | 16.868 | 16.806 |

Reading:
- The absolute intervention magnitude does not separate successful from failed episodes. The group difference is about 0.2%, its sign reverses between the two windows, and every episode lies within 16.87–17.03.
- The two whole-episode associations (relative delta, cosine) involve differences of about 10⁻⁵ and vanish when all episodes are compared over the same rollout steps.
- The most economical description is that the failures' extra late-rollout steps (250–519) visit hidden states with a very slightly different ‖h‖ and alignment with C.
- None of this is evidence that intervention size affects success, or the reverse. Per-episode CSV: `research/reproduction/experiments/phase1e_episode_intervention.csv`.

### 6.7 Largest and smallest intervention (items 9–10)

| | coordinate | mean ‖Δh‖ | notes |
|---|---|---|---|
| **largest application** | episode 6 (init 21), rollout step 125, **denoising step 0** | 24.101 | ‖h‖ 241.6, rel 0.0997 |
| **smallest application** | episode 3 (init 18), rollout step 380, **denoising step 9** | 12.738 | ‖h‖ 129.0, rel 0.0987 |
| largest single-token ‖Δh‖ | episode 9 (init 24), rollout step 280, denoising step 0 | 29.600 (token max) | |
| largest / smallest denoising step (mean over calls) | step **0** / step **9** | 22.298 / 13.394 | |
| largest / smallest rollout step (common window, mean over 15 episodes × 10 steps) | step **115** / step **160** | 17.198 / 16.745 | range only 2.7% |

Every application has its largest intervention at denoising step 0 and its smallest at denoising step 9, because ‖h‖ is largest and smallest there. Over rollout time the size is essentially flat.

## 7. Interpretation (descriptive; no causal claims)

- **What COAST does to the hidden state in this configuration.**
  - At layer 5 of the action expert, `global` V0 steering with β = 0.1 multiplies each action token's residual-stream vector by about 0.90, at every denoising step and every policy call. The only other term is 0.1·C·h, whose projection onto h is about 0.06% of ‖h‖ (hᵀCh/‖h‖² ≈ 0.006).
  - This is fixed mostly by the conceptor's spectrum. Since λ_max(C) = 0.35 and the median eigenvalue is 0.001, the term 0.1·C·h can never exceed 3.5% of ‖h‖ for any hidden state.
  - The measured states sit near the bottom of that range (hᵀCh/‖h‖² ≈ 0.006).
- **Why this might matter for Phases 1B–1D (hypothesis only).**
  - If the paper-faithful V3 conceptor likewise has hᵀCh/‖h‖² ≪ 1 on visited states, V0 and V3 would both act as roughly (1 − β)·h. Their 95% relative Frobenius difference (Phase 1B) would then have little behavioral leverage.
  - That is consistent with Phase 1D's identical V0/V3 success vectors and the small V0–V3 action gap (0.042 vs about 1.0 against baseline) seen in the Phase 1D GPU test.
  - **V3 was not measured here.** The hypothesis needs its own measurement, either the same diagnostics with the V3 NPZ or offline Rayleigh quotients of both conceptors on Phase 1A activations.
- **A uniform shrink is not a no-op.**
  - In the pre-norm Gemma expert, scaling the residual stream at layer 5 changes the balance between layers 0–5 and what later layers add. Later RMSNorms do not undo it.
  - The actions differ from baseline by about 1.0 under the same noise (Phase 1D GPU test), so the intervention clearly acts. This phase measures *what* it changes, not whether that change helps.

## 8. Limitations

- **One run:** one task, one steering configuration (L5/α0.5/β0.1/global), one conceptor (V0), one master noise seed, development states only. The paper-faithful V3 conceptor was not instrumented.
- **Only the direct change at the hooked layer is measured.** How Δh propagates to later layers, the velocity field or the actions is not measured. Neither is the hidden state an unsteered run would have had: in a steered rollout, h at step t already reflects steering at earlier steps and earlier calls.
- **Summaries are token means and maxima.** The Rayleigh-quotient and effective-scale figures are approximations built from means of per-token quantities (their variances are tiny, so the approximation is tight).
- **Success-vs-failure comparisons** have n = 15 (4 failures), many metrics, and uncorrected p-values. Whole-episode comparisons are confounded by episode length.
- **The bf16 rounding of M** is quantified offline; the realized per-dimension β is not logged.
- **The one-step rollout-length difference at state 25** relative to Phase 1D is not explained (§6.3).

## 9. Resources and housekeeping

- **Timeline:** server launched 23:42:33; client 23:43:17–23:46:00 (2m43s). The GPU tests took 61 s (Phase 1D + Phase 1E tests) plus 31.5 s (Phase 1E tests after adding the cosine field). One extra 30 s GPU check confirmed the hook dtype.
- **Errors:** none in the run.
  - During shutdown, a `pkill -f` pattern also matched the invoking shell. The server was still stopped, and the GPU logger was then stopped by PID. This did not affect any recorded data.
- **Outputs:**
  - Under `examples/libero_env/output/phase1e_repo_diag_task02_seed15/` (15 MB: videos, client log, `steering_diagnostics.jsonl` 3.9 MB, `noise_fingerprints.jsonl`).
  - Under `examples/libero_env/output/phase1e_servers/` (server log, GPU trace).
  - All gitignored; no videos, arrays, checkpoints, NPZs or caches are committed.
- `~/.libero` was not created (`LIBERO_CONFIG_PATH` was set). `~/Stanley_ws/lerobot` was not touched.
