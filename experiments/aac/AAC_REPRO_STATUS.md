# AAC reproduction in COAST — research status

Paper: Liang et al., *Adaptive Action Chunking at Inference-time for VLA Models*, CVPR 2026 (arXiv:2604.04161).
Branch `reproduce-aac` (base `2afa10e`); upstream `Adaptive-Action-Chunking/robocasa@fed3e6b`, `gr00t-multi-sample@11e926b`.
Model: GR00T N1.5 `robocasa365_checkpoints/gr00t_n1-5/multitask_learning/checkpoint-120000`; RoboCasa v1.0-12 (`9a3a786`).
Date: 2026-10-08. Companion docs: `AAC_REPRO_AUDIT.md`, `AAC_UPSTREAM_MAPPING.md`, `AAC_ACTION_MAPPING.md`, `AAC_BASELINE_IDENTITY.md`, `AAC_MECHANISM_VALIDATION.md`, `AAC_REPRO_NOTES.md`, `AAC_REPRO_PREREG.md`.

## Classification: **PARTIAL**

- **Mechanism: reproduced faithfully.** The selector is a verbatim port of the upstream code. It matches upstream bit-for-bit on 144 synthetic cases and a live re-run. Multi-sample GR00T follows the official fork line for line, and is bit-identical to NVIDIA's single-sample path at N=1.
- **Performance: the paper's setting cannot be reproduced.** The authors' fine-tuned checkpoint and pre-v1.0 RoboCasa task set are not available. In our setting (RoboCasa365 multitask checkpoint, RoboCasa v1.0 tasks):
  - under the **upstream protocol** (reseed 42 every call), AAC is **significantly worse** than fixed h=16: −11.5 pp [−16.0, −7.1], p < 0.001, n = 338 pairs;
  - at **COAST's validated operating point** (running RNG), AAC is **indistinguishable** from every fixed h: vs h=16, −2.9 pp [−8.6, +2.7], p = 0.35, n = 339. The paper's +2.3 pp gain is not reproduced, but it is not excluded either.
- This is not "FAILED" in the rubric's sense: conditions differ materially from the paper's (checkpoint, environment version, task implementations), and the large-scale Phase B has not been run.

---

### A. Did we faithfully reproduce the AAC mechanism? **YES**

- Selector (`groot_env/aac.py`): ported verbatim from `action_entropy_v2.py`. That includes normalized-space Gaussian entropy with a 1e-6 ridge, a Bernoulli gripper term on unnormalized {0,1}, the `argmax(diff)+1` elbow clamped to ≥ 2, and the ξ lower bound from candidate 0's motion with α=3, gripper weight 0.2 and fallback 16. Parity with upstream: 144 fixture cases plus 50 live cases, all matching at rtol 1e-12 (`groot_env/tests/test_aac.py`).
- Multi-sample GR00T (`groot_adapter._get_action_multi_sample`): one backbone pass, then `expand` to N and N independent noise draws, as in the official fork.
  - At N=1 it is bit-identical to NVIDIA `Gr00tPolicy.get_action`.
  - With the noise forced identical, all candidates are identical, which shows shared conditioning.
  - The same seed reproduces the candidates bit-exactly, and no two candidates are identical (`AAC_BASELINE_IDENTITY.md`).
- Plain non-AAC serving is bit-identical to `main`.
- Executed candidate = 0 and RNG = reseed 42 per call, both as in upstream.
- In rollouts, on matched scenes all arms receive bit-identical first responses (922/934), so the arms differ only in execution horizon.

### B. Does action entropy vary across the predicted horizon? **YES within each call; the systematic rise is protocol-dependent**

Over 23,348 AAC calls (Phase A, upstream protocol), mean per-step entropy in nats:

| future step i | 1 | 2 | 4 | 8 | 12 | 16 |
|---|---:|---:|---:|---:|---:|---:|
| translation E_t | −3.58 | −3.41 | −2.98 | −2.90 | −2.68 | −2.55 |
| rotation E_r | −8.35 | −8.37 | −7.77 | −7.56 | −7.17 | −7.07 |
| gripper E_g | 0.03 | 0.04 | 0.05 | 0.05 | 0.05 | 0.06 |
| total | −11.90 | −11.74 | −10.70 | −10.41 | −9.80 | −9.56 |

The last step's total exceeds the first step's in 81% of calls (mean rise +2.34 nats). Candidate spread rises from 0.06 to 0.08 (normalized std). Because the rise is steepest early, the elbow argmax_h(Ē_{h+1} − Ē_h) is 2 or 3 in 87% of calls.

**This depends on the protocol.** Under the COAST-native protocol (Phase A-U, running RNG, 22,406 calls), the mean profile is flat: −7.99 → −8.14 nats, with the last step above the first in only 43.5% of calls. Per-call profiles still vary, and the elbow is 2–3 in 79% of calls. Entropy *varies* across the horizon within a call, but there is no systematic "later actions are more uncertain" structure in that regime.

### C. Does AAC choose different h dynamically? **YES — but mostly through the motion bound ξ**

Under the COAST-native protocol (Phase A-U): mean h* 6.42; elbow decides 22.5%; ξ binds 77.5%; Spearman −0.72. Upstream protocol (Phase A) below.


- h* distribution over 23,348 calls (h = 2…16): 198, 3129, 2967, 2756, 2260, 1529, 1143, 917, 879, 781, 718, 484, 478, 370, **4739**. Mean 8.46; 15 distinct values; 20% at h* = 16, all of them from the ξ = 16 fallback.
- Mean h* by task ranges from 6.0 (PickPlaceCounterToStove) to 13.3 (CoffeeSetupMug).
- **The entropy elbow sets h* in only 9.5% of calls; ξ does in 90.5%.** h* tracks candidate 0's predicted motion: Spearman(h*, m(16)) = −0.91. Slow predicted motion gives long chunks; fast motion gives short ones. That is the opposite of the paper's Fig. 3 narrative, and it holds under the released parameters.

### D. Does AAC outperform fixed h=16 in our initial experiment? **NO — it is significantly worse (faithful upstream protocol)**

Phase A (`results/aac/phaseA_pilot/`): 7-task COAST subset, `pretrain` split, seeds 0–49 (50 episodes per task), upstream protocol (N=20, reseed 42 per call, candidate 0, α=3, 4 denoising steps), native 256 px.

| Arm | Success | Wilson 95% CI |
|---|---:|---:|
| fixed h=16 | 125/350 = **35.7%** | [30.9, 40.9] |
| AAC | 86/350 = **24.6%** | [20.4, 29.3] |

Paired, over 338 scene-matched episodes: AAC − fixed16 = **−11.5 pp**, task-stratified bootstrap 95% CI **[−16.0, −7.1]**. Discordant pairs: 14 AAC-only vs 53 fixed16-only, exact McNemar **p < 0.001**. Twelve pairs were excluded because of RoboCasa scene nondeterminism (see G).

AAC used 3.2× more policy compute per success (46.6 s vs 14.4 s) and twice as many policy calls per episode (66.7 vs 32.6).

**At COAST's validated operating point** (Phase A-U: running RNG, 224 px path): AAC 191/350 (54.6%) vs fixed16 197/350 (56.3%). Paired −2.9 pp [−8.6, +2.7], p = 0.35. Not different, and not better.

### E. Comparison with fixed h ∈ {2,4,8,12,16}

Common seeds 0–19, all arms (`results/aac/phaseA_pilot_20ep/`, 140 episodes per arm, upstream protocol):

| Method | Success | Wilson 95% CI | Mean h | Policy calls / ep | Server latency / query | Policy compute / success |
|---|---:|---:|---:|---:|---:|---:|
| Fixed-2 | 13/140 (9.3%) | [5.5, 15.2] | 2 | 307.8 | 151 ms | 500.9 s |
| Fixed-4 | 13/140 (9.3%) | [5.5, 15.2] | 4 | 153.9 | 147 ms | 243.0 s |
| Fixed-8 | 34/140 (24.3%) | [17.9, 32.0] | 8 | 70.0 | 148 ms | 42.7 s |
| Fixed-12 | 39/140 (27.9%) | [21.1, 35.8] | 12 | 45.2 | 170 ms | 27.6 s |
| Fixed-16 | 48/140 (34.3%) | [26.9, 42.5] | 16 | 32.7 | 173 ms | 16.6 s |
| **AAC** | 37/140 (26.4%) | [19.8, 34.3] | 8.66 (adaptive) | 63.5 | 171 ms | 41.0 s |
| *COAST ref.* Fixed-5 | 16/140 (11.4%) | [7.2, 17.8] | 5 | 120.9 | 141 ms | 149.4 s |

Paired AAC differences: vs h=2 +17.0 pp [+11.1, +23.7]; vs h=4 +17.0 [+11.1, +23.0]; vs h=8 +2.2 [−2.2, +6.7]; vs h=12 −1.5 [−7.5, +4.5]; vs h=16 −7.5 [−14.2, −0.7] (McNemar p = 0.064 at 20 seeds; p < 0.001 at 50 seeds, D). With 50 seeds: vs fixed-8 +1.5 pp [−2.1, +5.0], 339 pairs.

In this setting success rises monotonically with h, and h=16 is best. That matches the paper's ordering only at the extremes (paper: h=2 worst at 47.0%, h=8 best at 61.2%, h=16 59.7%). AAC performs like a fixed horizon close to its mean executed h (8–12), as expected when the horizon itself is what drives success. Per-task results: `results/aac/phaseA_pilot*/report.md`.

### F. Computational cost

Standalone (RTX 4090, one real observation, 20 repetitions; `AAC_BASELINE_IDENTITY.md`):

| N | 1 | 5 | 10 | 20 | 30 | 40 | plain `infer` |
|---|---:|---:|---:|---:|---:|---:|---:|
| sampling ms | 45.5 | 58.7 | 82.9 | 125.2 | 170.9 | 213.1 | 45.6 |
| peak VRAM MB | 5513 | 5513 | 5513 | 5575 | 5745 | 5915 | 5513 |

The selector adds 4–6 ms on CPU. N=20 is 2.7× the single-sample latency on a 4090; the paper reports +23 ms on an A800. In rollouts with 8 clients per GPU, server-side latency per query was 141–173 ms. AAC needs about 2× the policy calls of fixed-16 per episode.

N ablation (7 tasks × seeds 0–19, upstream protocol; `results/aac/phaseA_nablation/`):

| N | 1 | 5 | 10 | 20 |
|---|---:|---:|---:|---:|
| success | 39/140 (27.9%) | 37/140 (26.4%) | 38/140 (27.1%) | 37/140 (26.4%) |
| mean executed h | 11.28 | 9.40 | 8.88 | 8.66 |
| server ms/query (in rollouts) | 90 | 91 | 112 | 171 |

Success is flat in N. Mean h falls as N grows because, at small N, the elbow lands late more often (at N=1, h_entropy = 15 in about half of calls). With a well-estimated, front-loaded entropy rise, the elbow sits at 2–3. At N=1 the elbow is float round-off (h_entropy ∈ {2, 15}), so N=1 AAC amounts to "ξ, or a round-off-determined h".

### G. Differences from the official AAC setup

Exact list (details in `AAC_REPRO_NOTES.md`):
1. **Checkpoint**: the authors' fine-tune is unreleased; we use the RoboCasa365 multitask N1.5 checkpoint.
2. **RoboCasa** v1.0 instead of pre-v1.0 `0eae063`. Tasks are v1.0 analogues, with different scenes, objects and success checks.
3. **Episode budget and success rule**: COAST's (1.5 × task horizon; first success step) instead of 600 steps with a 4-step success latch.
4. **Executed action dims**: COAST executes all 12 dims; upstream zeroes base/torso/mode. AAC math is unaffected.
5. **Images**: native 256² (Phase A) or COAST's 224 → 256 path (Phase A-U); upstream uses raw robosuite 256².
6. **Selection location**: server instead of client, with an identical selector input. Transport: WebSocket instead of ZMQ.
7. **Trials**: 50 per task × 7 tasks (Phase A) instead of 100 × 24.
8. **Found here**: RoboCasa v1.0 does not fully seed scenes for tasks with optional fixtures (OpenStandMixerHead, TurnOnElectricKettle). Mismatched pairs are excluded via scene hashes.
9. **Found here**: the upstream reseed-42-every-call protocol collapses this checkpoint's baseline (fixed-5: 44% → 11% at 256 px; 59% on COAST's default path → 11%).

### H. Is this AAC implementation safe to use as a baseline for later work? **CONDITIONAL — yes for the mechanism; with stated conditions for performance**

- **Yes**, as a faithful, tested implementation of the *released* AAC algorithm: parity with upstream, provenance pinned, diagnostics logged per call, and fixed/AAC arms that differ only in horizon.
- **Conditions:**
  1. Report the noise protocol explicitly. Upstream's per-call reseed is a property of the release, not of AAC. With this checkpoint it puts every arm at a degraded operating point: fixed-5 drops from 59% to 11%, and AAC appears significantly worse than h=16. At COAST's operating point (`--aac-seed None`, 224 px) AAC is indistinguishable from fixed h. Use that operating point for COAST-native comparisons, report the upstream protocol as a secondary replicate, and state the deviation.
  2. Report that, with the released α=3, chunk sizing on this checkpoint is driven by the minimum-motion bound ξ in 78–90% of calls (protocol-dependent), not by entropy. Any claim about "entropy-driven chunking" needs the elbow component ablated (counterfactual h_entropy is logged per call).
  3. Compare against the strongest fixed h, which is h=16 here, never against COAST's historical h=5.
  4. Do not quote paper numbers next to ours; checkpoint and task implementations differ.
  5. Pair by scene hash and exclude the nondeterministic RoboCasa scenes.

## Phase A-U (sensitivity: COAST validated operating point)

*Pre-declared in `results/aac/phaseA_unseeded/PREDECLARED.md` before any episode ran. Results are filled in below.*

Setup: identical to Phase A except the server draws from a running torch RNG (`--aac-seed None`) and the client uses COAST's default 224 → 256 image path. All AAC parameters are the upstream defaults (N=20, α=3, candidate 0, 4 denoising steps). 7 tasks × seeds 0–49. Results: `results/aac/phaseA_unseeded/`.

| Method | Success | Wilson 95% CI | Mean h | Policy calls / ep | Server latency / query | Policy compute / success |
|---|---:|---:|---:|---:|---:|---:|
| Fixed-5 (COAST default h) | 191/350 (54.6%) | [49.3, 59.7] | 5 | 80.9 | 143 ms | 21.1 s |
| Fixed-8 | 194/350 (55.4%) | [50.2, 60.5] | 8 | 50.2 | 156 ms | 14.2 s |
| Fixed-16 | 197/350 (56.3%) | [51.0, 61.4] | 16 | 25.2 | 150 ms | 6.7 s |
| **AAC** | 191/350 (54.6%) | [49.3, 59.7] | 6.42 (adaptive) | 64.0 | 155 ms | 18.2 s |

Paired, scene-matched comparisons (exact McNemar; task-stratified bootstrap 95% CI):

| Comparison | Pairs | AAC-only | other-only | Difference | 95% CI | p |
|---|---:|---:|---:|---:|---:|---:|
| AAC vs fixed16 (**primary**) | 339 | 41 | 51 | **−2.9 pp** | [−8.6, +2.7] | 0.35 |
| AAC vs fixed8 | 341 | 45 | 49 | −1.2 pp | [−6.7, +4.4] | 0.76 |
| AAC vs fixed5 | 341 | 45 | 46 | −0.3 pp | [−5.9, +5.3] | 1.00 |

- **At COAST's validated operating point, AAC is statistically indistinguishable from every fixed h, and it does not outperform fixed-16.** The CI includes the paper's +2.3 pp, so a small gain is not excluded, but the point estimate is negative. Per the pre-registered rule this is *inconclusive*.
- **The fixed-h curve is nearly flat here** (54.6 → 56.3%). Under the reseeded upstream protocol it was steep (9 → 34%). The noise protocol, not the chunk size, is what made short chunks harmful in Phase A. The paper's premise that "fixed chunk size strongly affects performance" is therefore *protocol-dependent* for this checkpoint.
- Mechanism under this protocol (22,406 AAC calls):
  - The entropy profile is on average **flat** across the horizon (−7.99 → −8.14 nats; the last step exceeds the first in only 43.5% of calls).
  - The elbow still sits at 2–3 in 79% of calls.
  - The elbow sets h* in 22.5% of calls; ξ binds in 77.5%.
  - Spearman(h*, predicted motion) = −0.72.
  - 15 distinct h*, 7% at 16, 0 non-finite values.
- Cost: AAC uses 2.5× the policy calls of fixed-16 and 2.7× the policy compute per success, at +5 ms server latency per query in rollouts. Its N=20 sampling runs in the same batched pass.

## Recommended next experiment

**Phase B-coast first** (`AAC_REPRO_PREREG.md`, Amendment 1): 22 v1.0-analogue tasks × 50 seeds at the COAST-native operating point, arms fixed{2,4,8,12,16} + AAC. That is about 1,100 pairs per comparison, with a minimum detectable difference of about 3.8 pp. At that operating point episodes are shorter and fixed2 is cheaper, so expect roughly 12–15 h on 2× RTX 4090. Then run B-upstream if a strict-protocol replicate is needed.

Run this **mechanism ablation** at the same time; it needs no new code, only `--aac-move-th`. Set α = 0 so that ξ = 2 and h* = h_entropy (entropy only); α = 3 stays as the released AAC. This isolates whether the entropy component contributes anything beyond the motion bound. Declare it as an ablation, not as tuning.

## Commands to reproduce / extend

```bash
# 0. Environments (once): see groot_env/README.md and examples/robocasa_env/README.md.
#    flash-attn: uv pip install <flash_attn-2.7.1.post4+cu12torch2.5cxx11abiFALSE-cp310 wheel URL>
# 1. Unit tests
cd groot_env && uv run pytest -q -m "not manual" tests && cd ..
cd examples/robocasa_env && uv run pytest -q tests -m "not manual" && cd ../..
# 2. GPU validation (identity, diversity, latency vs N)
cd groot_env && CUDA_VISIBLE_DEVICES=0 uv run python ../experiments/aac/validate_multisample.py \
    --obs_npz ../results/aac/validation/real_obs_OpenDrawer_seed0.npz && cd ..
# 3. Servers (upstream protocol); add `--aac-seed None` for the COAST-native protocol
experiments/aac/serve_aac.sh 0 8000 20 > results/aac/logs/server_8000.log 2>&1 &
experiments/aac/serve_aac.sh 1 8001 20 > results/aac/logs/server_8001.log 2>&1 &
# 4. Phase A (as run)
RUN=phaseA_pilot TASKS="CloseFridge CoffeeSetupMug OpenDrawer OpenStandMixerHead PickPlaceCounterToCabinet PickPlaceCounterToStove TurnOnElectricKettle" \
  EPISODES=50 ARMS="fixed16 aac fixed8" PORTS="8000 8001" WORKERS=4 experiments/aac/run_arms.sh
uv run python experiments/aac/analyze_aac.py results/aac/phaseA_pilot
# 5. Phase B (frozen design; ~17 h on 2x4090): see AAC_REPRO_PREREG.md "Exact commands" and its Amendment.
```
