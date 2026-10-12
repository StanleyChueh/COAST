# AAC baseline identity and multi-sample validation

Script: `experiments/aac/validate_multisample.py`. Raw output: `results/aac/validation/multisample_validation.json` (gitignored).

Setup: RTX 4090, torch 2.5.1+cu124, flash-attn 2.7.1.post4, checkpoint `gr00t_n1-5/multitask_learning/checkpoint-120000`, 4 denoising steps. Observation: a real RoboCasa observation (OpenDrawer, `pretrain`, `env.reset(seed=0)`, native 256² cameras, prompt "Open the left drawer.") saved to `results/aac/validation/real_obs_OpenDrawer_seed0.npz`.

| Check | What is compared | Result |
|---|---|---|
| A. Baseline identity | `GR00TAdapterPolicy.infer` from **`main`'s** `groot_adapter.py` (loaded via `git show main:…`) vs this branch's `infer`, both after `torch.manual_seed(1234)`, same loaded model | **bit-identical** (max abs diff 0.0, shape (16, 12) float32) |
| A'. Repeatability | this branch's `infer` twice under the same seed | bit-identical |
| B. N=1 multi-sample vs NVIDIA | `Gr00tPolicy.get_action` vs `_get_action_multi_sample(N=1, seed=None)`, same torch seed | **bit-identical** (max abs diff 0.0). Shapes (1, 16, d); `normalized_action` (1, 16, 32) float32 |
| C. Shared conditioning | `torch.randn` patched so all 8 candidates start from the same noise | all candidates identical (max dev 0.0). Only the initial noise distinguishes candidates |
| D. Diversity | N=20, seed 42 | no identical pairs; pairwise L2 over normalized dims 0–6 min 0.41, mean 1.35. Per-step std rises from 0.019 (step 1) to ~0.09 (step 5) |
| D'. Seed determinism | N=20, seed 42 twice | bit-identical |
| D''. Different seed | seed 42 vs 43 | max abs diff 5.40 |
| E. Candidate 0 at N=20 vs N=1 (both reseeded 42) | — | not identical; max abs diff 0.0039 in normalized space (bf16 batch-size numerics). Fixed-h arms therefore use candidate 0 *of the N=20 batch*, as upstream does |

**Conclusion.** Adding AAC does not change ordinary (non-`--aac`) GR00T inference: it is bit-identical to `main`. The multi-sample path with N=1 and no reseed reproduces NVIDIA's single-sample math exactly.

## Latency and memory vs N (same observation, 20 reps after 3 warm-up calls)

`sample_ms` = full `infer_multi_sample` (transforms, backbone, N-batch DiT, unnormalize). `select_ms` = AAC selector on CPU.

| N | sample ms (mean ± sd) | select ms | peak VRAM (MB) |
|---:|---:|---:|---:|
| plain `infer` | 45.6 | — | 5513 |
| 1 | 45.5 ± 0.3 | 4.4 | 5513 |
| 5 | 58.7 ± 0.6 | 5.6 | 5513 |
| 10 | 82.9 ± 1.2 | 5.6 | 5513 |
| 20 | 125.2 ± 0.6 | 5.5 | 5575 |
| 30 | 170.9 ± 0.3 | 5.6 | 5745 |
| 40 | 213.1 ± 0.7 | 5.6 | 5915 |

N=20 costs about 2.7× the single-sample latency on a 4090, which is +80 ms per query. The paper's Tab. 4 reports +23 ms on an A800. The difference is the batched DiT cost scaling with N; the backbone runs once. VRAM barely changes.

## AAC on this observation

h* = 5 (h_entropy = 2, ξ = 5). H_total rises from −18.2 at step 1 to about −7 by step 7, then stays flat. Ē_h therefore increases fastest at the first step, so the elbow sits at its minimum of 2, and the minimum-motion bound (m(l) crosses 3 at l=5) sets h*.

## Cross-arm determinism in rollouts (Phase A)

On matched initial scenes, the first policy call of every arm sees the same observation. Under the upstream protocol (reseed 42 per call) the server's response must then be identical across arms. Over 934 matched (arm, task, seed) first calls versus fixed16, AAC diagnostics (h*, H_total, m(l)) are **bit-identical in 922**. The remaining 12 (CloseFridge) differ by ≤ 0.19 nats in H_total, with the same h*, which is consistent with cross-GPU numerics (the arms ran on two servers). So the arms differ only in the execution horizon, as designed.
