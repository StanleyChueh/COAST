# Phase 1D — Paired-Noise Control: Baseline vs Repository-Faithful vs Paper-Faithful COAST

**Development split only (`--seed 15`, init states 15–29). States 30–44 remain untouched.**
This is a reproducibility/control experiment. It is not a held-out result, and it is not comparable to the paper's 30-episode held-out numbers. No steering parameter was tuned.

Paper: *Contrastive Conceptor Activation Steering (COAST)*, arXiv:2605.17144.
Base commit: `f1ad68b285614d6848bea07a90f18781a63d4d75` (Phase 1C). Branch `exp/paired-noise-control`. The runs used the uncommitted working tree described in §2. `sha256(git diff HEAD -- src scripts examples packages)` at run time was `950b2248…63dc00`. After the runs, `ruff format` rewrapped three lines in `examples/libero_env/main.py` (whitespace/line breaks only, no semantic change), so the current diff hash differs from the run-time value.

## 0. Summary

| | Phase 1C (unpaired noise) | Phase 1D (paired noise, master seed 100) |
|---|---|---|
| Baseline | 8/15 | **13/15** |
| Repository-faithful V0 | 10/15 | **11/15** |
| Paper-faithful V3 | 10/15 | **11/15** |
| Agreement baseline/V0, baseline/V3, V0/V3 | 5, 5, 9 of 15 | **11, 11, 15** of 15 |
| Baseline noise floor (3 master seeds) | — | **13, 7, 10** of 15 |

- **Noise pairing is verified end to end.** All 5,705 logged fingerprints were recomputed independently and every one matches. At every coordinate the same-seed runs share, the noise is identical (745 to 915 shared coordinates per pair), and runs with different seeds share no identical noise.
- **The closed loop is deterministic given the noise.** The Phase 1D baseline and the seed-100 noise-floor run were served by different server processes. They reproduced each other exactly: 15/15 identical outcomes and 15/15 identical rollout lengths.
- **Pairing removes most between-condition churn.** V0 and V3 now agree on all 15 states. Baseline and steered runs agree on 11/15, up from 5/15.
- **The baseline noise floor is wide.** The master noise seed alone moves the baseline between 7/15 and 13/15. 11 of 15 states flip outcome depending on the seed. The Phase 1C gap (+2 for steering) and the Phase 1D gap (−2 for steering) are both inside that range.
- **Conclusion:** there is still no evidence that COAST at L5/α=0.5/β=0.1 changes the success rate on this task, in either direction. There is also no evidence that V0 and V3 differ behaviorally; with paired noise their success vectors are identical. Because Phase 1D uses a single noise realization, it cannot estimate a steering effect.

## 1. Question and design

Phase 1C compared the three conditions without paired policy noise: the PyTorch flow noise came from an unseeded CUDA generator, and its stream position drifted with episode length. Phase 1D asks whether the conditions behave differently when every policy query at the same rollout coordinate starts denoising from the **same explicit initial flow-noise tensor**.

Fixed across all runs:
- **Model:** pi0.5, `pi05_libero`, PyTorch.
- **Checkpoint:** `checkpoints/openpi-libero-2000`.
- **Task:** libero_10 task 2, `KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it`.
- **Episodes:** `--seed 15 --num_episodes 15` (episode k → init state 15+k; `env.seed(15)`).
- **Rollout:** max 520 steps, replan every 5.
- **Steering (not tuned):** layer 5, α 0.5, β 0.1, `global`.
- **Conceptors:** V0 = `conceptors/phase1b_repo_task02.npz` (sha256 `8b1c802a…ec80f3`), V3 = `conceptors/phase1b_paper_task02.npz` (sha256 `4c7f771b…ee8f29`).
- **GPU:** GPU 0 (RTX 4090) for servers and clients.

**Preregistered before any rollout:**
- The three-way run uses master noise seed **100**.
- The noise floor uses baseline runs at master seeds 100, 200 and 300.
- Baseline A (seed 100) runs in a different server process from the seed-100 noise-floor run, so the pair doubles as a cross-process determinism check.

## 2. Code changes

| File | Change |
|---|---|
| `src/openpi/policies/policy.py` | `Policy.infer_with_steering(..., noise=None)`. If noise is given, it is converted exactly as in `Policy.infer` (`torch.from_numpy(noise).to(device)`, plus a batch dim) and passed as `noise=` to `sample_actions_with_steering`. When `noise` is None, the call is unchanged. |
| `src/openpi/serving/steering.py` | `SteeredPolicyWrapper.infer(obs, *, noise=None)` forwards `noise` to `Policy.infer` or `Policy.infer_with_steering`, **only when it is not None**, so older inner policies without a `noise` kwarg keep working. The pi0-fast path raises `NotImplementedError` if noise is given. |
| `src/openpi/serving/noise_control.py` (new) | Payload validation, deterministic noise derivation, SHA-256 fingerprint, `noise_shape_from_policy`, and `NoiseControlledPolicyWrapper`. |
| `packages/openpi-client/src/openpi_client/noise_control.py` (new) | Stdlib-only wire protocol (Python 3.8-safe): `NOISE_CONTROL_KEY = "__noise_control__"`, `NOISE_CONTROL_ECHO_KEY = "noise_control"`, `NOISE_KEY_FIELDS`, `build_noise_control_payload`. |
| `scripts/serve_policy.py` | Opt-in `--noise-control` flag. Rejects pi0-fast and `--collect_activations`. Reads the noise shape from the **loaded model** and checks it against the train config. Installs `NoiseControlledPolicyWrapper` as the **outermost** wrapper. |
| `examples/libero_env/main.py` | Opt-in `--policy_noise_seed N`. Requires server metadata `noise_control_enabled`. Attaches `{master_seed, task_id, init_state, rollout_step}` to each request, checks the server's echo on every response, and appends `{key, episode, sha256}` to `<task_output_dir>/noise_fingerprints.jsonl` (gitignored output). |
| `tests/test_noise_control.py`, `tests/models/test_noise_control_gpu.py` (new) | See §4. |
| `research/reproduction/tools/analyze_phase1d.py` (new) | Analysis and fingerprint verification (§6–7). |

**Rationale.**
- The explicit noise tensor replaces only the sampler's `sample_noise()` draw. The global torch RNG is not reseeded, and model weights, preprocessing, the denoising equations, action decoding and the steering math are unchanged.
- The noise is derived **on the server** from a small integer key, so the shape comes from the actual model rather than the client, and nothing large goes over the wire.
- Backward compatibility:
  - Without `--noise-control` the wrapper is not installed.
  - With it installed, a request without the key passes straight through (`inner.infer(obs)`, no `noise` kwarg).
  - The client refuses to run `--policy_noise_seed` against a server whose metadata lacks noise control. It also fails on any response without a matching echo. Without these checks, an ignored key would silently fall back to random noise.

## 3. Deterministic noise construction

```
key    = (master_seed, task_id, init_state, rollout_step)         # non-negative ints, in this order
ss     = numpy.random.SeedSequence(list(key))
noise  = numpy.random.Generator(numpy.random.PCG64(ss)).standard_normal((H, D), dtype=float32)
sha256 = SHA-256 of noise as little-endian float32 bytes, C order
```

- **(H, D) = (10, 32).** Read from the loaded `PI0Pytorch.config` (`action_horizon=10`, `action_dim=32`) and checked against `pi05_libero`'s model config at server start. Server log: `Noise control enabled: explicit flow noise of shape (10, 32)`. The sampler receives `(1, 10, 32)` float32 on `cuda:0`.
- **Distribution.** i.i.d. N(0, 1), the same as `PI0Pytorch.sample_noise` (`torch.normal(0, 1)`, float32). Only the generator changes: NumPy PCG64 instead of the CUDA Philox generator.
- **Coordinate.** `rollout_step` is the post-settle step index at which the client requests a new chunk (0, 5, 10, …). Conditions that diverge physically still get identical noise whenever they query at the same `(init_state, rollout_step)`. The trajectories themselves are not forced to match.
- **Master seed:** 100 for A/B/C. Noise floor: 100, 200, 300. `task_id = 2`.
- **Golden fingerprint.** Key (100, 2, 15, 0) gives `87ae314ae03d4e549d3bfa442c28cb94f594bff03d5c196422ae8b165ab113a0`. This is pinned in `test_derive_noise_golden_fingerprint` and matches the first request logged in every seed-100 run. The environment used NumPy 1.26.4; NumPy does not guarantee `Generator` streams across versions, which is why the value is pinned.

## 4. Unit tests (all passed before any LIBERO rollout)

CPU (`tests/test_noise_control.py`, 36 tests). These drive a **real `Policy`** inside the serve_policy stack `NoiseControlledPolicyWrapper(SteeredPolicyWrapper(Policy))`, with a fake torch model whose samplers record the `noise` they receive and a recording input transform:

| Req. | Test | Result / tolerance |
|---|---|---|
| A | `test_same_key_is_reproducible_through_stack` (the global torch RNG is reseeded in between) | exact (`np.array_equal`, `torch.equal`) |
| B | `test_paired_noise_reaches_both_samplers_identically`: the same tensor reaches `sample_actions` and `sample_actions_with_steering`, with shape (1,10,32), float32, equal to the derived noise; echo SHA equals the fingerprint of that array | exact |
| B | `test_policy_infer_with_steering_explicit_noise_matches_infer` | exact |
| D | `test_different_master_seed_changes_sampler_noise`; `test_each_key_field_changes_noise[×4]` | noise differs |
| E | `test_no_noise_key_preserves_previous_behavior[±steer]`: sampler gets `noise=None`, no echo | pass |
| E | `test_without_wrapper_policy_api_unchanged[±steer]`; `test_steered_wrapper_forwards_noise_only_when_given` (old inner API without a `noise` kwarg) | pass |
| F | `test_magic_keys_do_not_reach_input_transforms[±steer]`: the transform sees only `{observation/state, prompt}`, and the caller's obs is not mutated | pass |
| — | payload validation (7 invalid cases, numpy ints accepted), determinism/dtype/shape, golden fingerprint, N(0,1) moments over 64,000 samples (\|mean\| < 0.02, \|std−1\| < 0.02), shape resolution for PyTorch-style/JAX-style/missing model, batched obs rejected, metadata, pi0-fast + noise rejected | pass |

Manual GPU tests (`tests/models/test_noise_control_gpu.py`, 7 tests) on the real checkpoint and both real NPZs, on one fixed synthetic observation (`default_rng(0)` images/state, real task prompt):

| Req. | Test | Measured max \|diff\| | Asserted tolerance |
|---|---|---|---|
| A | same key ×4 through the stack (global torch/CUDA RNG reseeded) + direct `Policy.infer(obs, noise=…)` | **0.0** (all 4) | exact equality |
| B | spy on `sample_actions_with_steering` (β=0.1, V3): received noise is (1,10,32) float32 on cuda, equal to the derived noise | exact | `torch.equal` |
| C | β=0 steering vs unsteered, same noise, through the full stack: V0 **0.0**, V3 **0.0** | 0.0 | exact equality |
| C | raw sampler: `sample_actions` vs `sample_actions_with_steering(steering_hooks=None)`, same noise | **0.0** | `torch.equal` |
| D | change `master_seed` +100 / `rollout_step` +5 | 0.855 / 0.860 | > 1e-3 |
| sanity | β=0.1, same noise: V0 vs baseline 1.061, V3 vs baseline 1.023, V0 vs V3 0.042 | — | > 0 |

With β=0, M = (1−0)·I + 0·C = I exactly, and the bf16 `h @ I.T` reproduces `h` bit for bit. So the steered and unsteered code paths are **numerically identical** on this GPU (tolerance 0), not merely close.

Regression check. Full CI suite (`CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu uv run pytest --strict-markers -m "not manual"`): **350 passed, 47 skipped, 0 failed**. The same suite on the pre-change tree gave 314 passed and the same 47 skipped. `ruff check` and `ruff format --check` are clean on all changed files in ruff scope (`examples/libero_env` is excluded from ruff by `pyproject.toml`).

Commands:
```
CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu uv run pytest tests/test_noise_control.py -v
CUDA_VISIBLE_DEVICES=0 uv run pytest tests/models/test_noise_control_gpu.py -m manual -v -s
```

## 5. Rollout commands

Servers (repo root). The only difference between them is the NPZ:
```
CUDA_VISIBLE_DEVICES=0 uv run --no-sync scripts/serve_policy.py --pytorch --steer --noise-control \
  --conceptor-npz <NPZ> --port 8101 \
  policy:checkpoint --policy.config pi05_libero --policy.dir checkpoints/openpi-libero-2000
```
| Server (process) | NPZ | Runs, in order |
|---|---|---|
| S1 | repo | noise floor seed 100, 200, 300 (baseline, no steering payload) |
| S2 (fresh) | repo | A baseline, then B V0 (both master seed 100) |
| S3 (fresh) | paper | C V3 (master seed 100) |

Clients (`examples/libero_env`): `CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl LIBERO_CONFIG_PATH=/home/stanley/Stanley_ws/COAST-research/libero_config_coast uv run --no-sync python main.py --task_suite_name libero_10 --task_id 2 --num_episodes 15 --seed 15 --port 8101 --output_dir output/<name> --policy_noise_seed <S>`. B and C add `--steer --steering_layer 5 --steering_alpha 0.5 --steering_beta 0.1 --steering_strategy global`. Each run was executed exactly once. The driver script is recorded in the YAML.

## 6. Fingerprint validation

`analyze_phase1d.py` asserts, and all assertions passed:
1. **Independent recomputation.** For all 5,705 logged requests, re-deriving the noise from the logged key gives the server-echoed SHA-256.
2. **Request schedule.** Every episode's logged `rollout_step`s are exactly `0, 5, …` up to that episode's length, and `init_state = 15 + episode`.
3. **Cross-condition pairing.** For each pair of runs, compare fingerprints at the `(init_state, rollout_step)` coordinates both runs queried:

| Pair | Same seed | Shared coordinates | Identical |
|---|---|---|---|
| baseline vs repo | yes | 745 | **745** |
| baseline vs paper | yes | 750 | **750** |
| repo vs paper | yes | 915 | **915** |
| baseline vs noise-floor seed 100 | yes | 845 | **845** |
| any seed-100 run vs seed 200/300, and 200 vs 300 | no | 694–934 | **0** |

Sample (first 16 hex digits):

| coordinate | baseline | repo | paper | seed 200 | seed 300 |
|---|---|---|---|---|---|
| state 15, step 0 | 87ae314ae03d4e54 | 87ae314ae03d4e54 | 87ae314ae03d4e54 | d3cfa08b760d8887 | 0388e46c6f552150 |
| state 22, step 100 | 0d11894d0d36ae25 | 0d11894d0d36ae25 | 0d11894d0d36ae25 | 24f36b820f62fe3b | 377ac47d9840ee70 |
| state 29, step 200 | 2748c9262cf9e541 | 2748c9262cf9e541 | 2748c9262cf9e541 | 52a71313381ad5fe | 23bec3f1d3c27d95 |

Requests per run: baseline 845, repo 922, paper 936; noise floor 845 / 1148 / 1009.

The fingerprint covers the array the server wrapper passes to `inner.infer(noise=…)`. The path from that array to the sampler is covered by unit tests B (CPU and GPU).

## 7. Results

### 7.1 Noise floor (baseline, steering off, three master noise seeds)

| init state | seed 100 | seed 200 | seed 300 | steps 100 / 200 / 300 |
|---|---|---|---|---|
| 15 | 1 | 0 | 1 | 216 / 520 / 201 |
| 16 | 1 | 0 | 0 | 215 / 520 / 520 |
| 17 | 1 | 1 | 1 | 262 / 209 / 241 |
| 18 | 1 | 0 | 1 | 252 / 520 / 322 |
| 19 | 1 | 1 | 1 | 211 / 208 / 210 |
| 20 | 1 | 1 | 1 | 352 / 250 / 253 |
| 21 | 0 | 1 | 1 | 520 / 215 / 260 |
| 22 | 1 | 1 | 1 | 228 / 219 / 209 |
| 23 | 1 | 0 | 0 | 259 / 520 / 520 |
| 24 | 1 | 0 | 1 | 235 / 520 / 235 |
| 25 | 1 | 0 | 0 | 217 / 520 / 520 |
| 26 | 1 | 0 | 0 | 240 / 520 / 520 |
| 27 | 1 | 0 | 1 | 249 / 520 / 294 |
| 28 | 0 | 1 | 1 | 520 / 242 / 205 |
| 29 | 1 | 1 | 0 | 222 / 226 / 520 |
| **total** | **13** (Wilson 0.62–0.96) | **7** (0.25–0.70) | **10** (0.42–0.85) | |

- Pairwise agreement: 100 vs 200 5/15, 100 vs 300 8/15, 200 vs 300 10/15 (mean 7.7/15).
- States that succeed under all three seeds: 4. States that fail under all three: 0. States with mixed outcomes: **11/15**.
- Together with Phase 1C's baseline (8/15, unseeded noise), four baseline runs on the same 15 states gave 7, 8, 10 and 13 successes. The policy noise alone accounts for a spread of 6 successes.

### 7.2 Controlled three-way comparison (master seed 100)

| Condition | Successes / 15 | Rate | Wilson 95% | Median steps-to-success | Client wall |
|---|---|---|---|---|---|
| A. Baseline | **13** | 0.867 | 0.62–0.96 | 234 | 2m26s |
| B. Repository-faithful V0 | **11** | 0.733 | 0.48–0.89 | 223 | 2m39s |
| C. Paper-faithful V3 | **11** | 0.733 | 0.48–0.89 | 237 | 2m43s |

| init state | baseline | repo (V0) | paper (V3) | steps base / repo / paper | Phase 1C base / repo / paper |
|---|---|---|---|---|---|
| 15 | 1 | 0 | 0 | 216 / 520 / 520 | 1 / 1 / 1 |
| 16 | 1 | 1 | 1 | 215 / 212 / 206 | 0 / 1 / 1 |
| 17 | 1 | 1 | 1 | 262 / 238 / 239 | 1 / 0 / 1 |
| 18 | 1 | 0 | 0 | 252 / 520 / 520 | 0 / 1 / 1 |
| 19 | 1 | 1 | 1 | 211 / 201 / 213 | 0 / 0 / 1 |
| 20 | 1 | 1 | 1 | 352 / 232 / 238 | 0 / 1 / 1 |
| 21 | 0 | 1 | 1 | 520 / 203 / 201 | 1 / 0 / 1 |
| 22 | 1 | 1 | 1 | 228 / 238 / 226 | 1 / 1 / 1 |
| 23 | 1 | 1 | 1 | 259 / 253 / 241 | 0 / 1 / 1 |
| 24 | 1 | 0 | 0 | 235 / 520 / 520 | 1 / 1 / 0 |
| 25 | 1 | 1 | 1 | 217 / 221 / 221 | 1 / 0 / 0 |
| 26 | 1 | 1 | 1 | 240 / 258 / 246 | 1 / 1 / 0 |
| 27 | 1 | 1 | 1 | 249 / 224 / 291 | 1 / 0 / 0 |
| 28 | 0 | 0 | 0 | 520 / 520 / 520 | 0 / 1 / 0 |
| 29 | 1 | 1 | 1 | 222 / 224 / 243 | 0 / 1 / 1 |

Rollout steps exclude the 10 settle steps, and 520 is a timeout failure. This is the same convention as Phase 1C. It is computed from video frame counts, and the script checks it against the committed Phase 1C CSV. Machine-readable: `research/reproduction/experiments/phase1d_episode_results.csv` (includes the noise-floor columns).

### 7.3 Transitions and agreement

| | Phase 1D (paired) | Phase 1C (unpaired) |
|---|---|---|
| baseline fail → repo success | 1 (state 21) | 6 |
| baseline success → repo fail | 3 (states 15, 18, 24) | 4 |
| baseline fail → paper success | 1 (state 21) | 6 |
| baseline success → paper fail | 3 (states 15, 18, 24) | 4 |
| repo ↔ paper disagreements | **0** | 6 (3/3) |
| agreement baseline/repo, baseline/paper, repo/paper | **11, 11, 15** of 15 | 5, 5, 9 |
| identical rollout length baseline/repo, baseline/paper, repo/paper | 1, 1, 5 of 15 | 1, 2, 2 |
| all three succeed / all three fail | 10 / 1 (state 28) | 2 / 0 |
| exact McNemar p (reference only): base vs repo, base vs paper, repo vs paper | 0.625, 0.625, 1.0 | 0.754, 0.754, 1.0 |

Paired differences (descriptive): repo − baseline = −2, paper − baseline = −2, paper − repo = 0.

**Cross-process determinism.** Baseline A (server S2) vs noise floor seed 100 (server S1): 15/15 identical outcomes **and 15/15 identical rollout lengths**. Given the same noise schedule, the whole closed loop (simulator plus policy) replays exactly across server processes on this GPU. Any residual bf16 kernel nondeterminism was not large enough to change a single rollout length.

### 7.4 Comparison with Phase 1C

- **Success counts:** baseline 8 → 13, V0 10 → 11, V3 10 → 11. The sign of the steered−baseline difference flipped, from +2 to −2. Both values lie inside the baseline's own 7–13 spread across noise seeds.
- **Same condition across phases:** Phase 1C vs Phase 1D agreement is baseline 8/15, V0 6/15, V3 10/15. Changing only the noise realization reshuffles individual episodes about as much as the choice of condition did in Phase 1C.

### 7.5 Did controlled noise reduce episode-level churn?

**Yes, for comparisons between conditions.**
- V0 vs V3 went from 6 disagreements to 0.
- Baseline vs steered went from 10 disagreements to 4.
- As a reference, unpaired baseline-vs-baseline runs with different noise seeds disagree on 5–10 of 15 states (mean 7.3).

With pairing, the remaining baseline-vs-steered disagreements (4/15) can be attributed to the β=0.1 intervention for **this** noise realization. That includes its chaotic amplification: once a trajectory diverges, later queries use the same noise on different observations.

The underlying stochasticity is **not** reduced. Pairing makes the conditions share one draw of it: 11/15 states still flip with the master seed. A single paired seed is therefore a sample of size one over noise realizations, and it cannot estimate the steering effect on success rate.

## 8. Interpretation and limitations

- **No steering effect is established.** In Phase 1D, steering lost 2 episodes relative to baseline (1 gain, 3 losses). In Phase 1C it gained 2. Neither difference is distinguishable from noise-seed variation at N = 15 with one master seed.
- **V0 vs V3.** Their success vectors are identical under paired noise (15/15), but only 5/15 rollout lengths coincide, so the trajectories do differ. The 95% relative Frobenius difference between the matrices (Phase 1B) does not translate into a detectable behavioral difference at β = 0.1. On the synthetic GPU-test observation their actions differed by at most 0.042, compared with about 1.0 between steered and baseline actions. This is one off-distribution observation and is only a hint.
- **Seed 100 is a favorable baseline realization** (13/15, the highest of the three seeds). Master seed 100 was fixed before any rollouts, but the paired comparison is still conditional on it.
- Measuring the steering effect would take paired comparisons across several master seeds (a mixed or paired analysis over seeds × states). That is a design decision for the next phase and was not run here.
- The noise scheme uses a different generator than `sample_noise`, although the distribution is the same (float32 i.i.d. N(0,1)). Absolute Phase 1D rates are therefore another draw, not a replay of Phase 1C.
- Scope limits: single task, single steering configuration, development split. States 30–44 were not used. No tuning, and no per-step steering.

## 9. Resources, errors, housekeeping

- **Timeline:**
  - S1 launched 17:48:15; the three noise-floor clients ran 17:48:46–17:57:20.
  - S2 launched 17:57:25; A and B ran 17:57:55–18:03:00.
  - S3 launched 18:03:05; C ran 18:03:35–18:06:18.
  - Total ≈ 18 min.
- **GPU 0:** peak 8,183 MiB, peak utilization 72% (same as Phase 1C). GPU unit tests ≈ 34 s.
- **Errors:** none. 0 Traceback/error lines in the three server logs, and all six clients exited 0. Each client showed only the 5 known benign warnings (robosuite private-macro notice, LIBERO `datasets` path missing). Both steered servers logged `Built steering hook ('KITCHEN_SCENE3_…', 5, 0.5, 0.1, 'global') [ConceptorSteeringHook] (cache size=1)`.
- `~/.libero` was not created; all runs set `LIBERO_CONFIG_PATH`. `~/Stanley_ws/lerobot` was not touched.
- Outputs are under `examples/libero_env/output/phase1d_*` (videos, client logs, `noise_fingerprints.jsonl`; ≈ 73 MB) and are gitignored. No videos, arrays, checkpoints or caches were committed.
