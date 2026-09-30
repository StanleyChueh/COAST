# Phase 8A — AAC Public Reproducibility Audit

**Paper:** Liang et al., "Adaptive Action Chunking at Inference-time for Vision-Language-Action Models", CVPR 2026, pp. 20802–20811 (arXiv:2604.04161v2, 10 Apr 2026, 12 pages including supplementary material).
**Target:** π0.5 + AAC on LIBERO (paper Table 2). LIBERO-Pro (Table 3, π0.5 rows) is covered as a secondary target.
**Audit date:** 2026-09-30.
**Mode:** read-only. No AAC repository was modified, no packages were installed, no checkpoints were downloaded, and nothing was run on a GPU. The only files fetched were the paper PDF, small source/metadata files from GitHub, GCS object metadata (listing only, no checkpoint bytes), and three LIBERO-Pro init-state files of about 4 KB and 53 KB. All of these went to the session scratchpad.

---

## 1. Executive Summary

**Classification: PARTIAL. Recommendation: NO-GO for an exact reproduction.** A clearly labelled, assumption-declared re-implementation is technically feasible (§13).

What is public and exact:
- **AAC algorithm code for π0.5** (`action_optimization/action_entropy_pi05.py::select_chunk_size`) and **the π0.5 LIBERO simulation client** (`pi05_benchmark/inference_client_action_entropy_pi05.py`). The whole AAC computation (entropy, elbow, action-magnitude lower bound) runs **client-side**, as the GR00T README says. This was verified in code.
- **π0.5 checkpoint**: the paper gives it explicitly as `gs://openpi-assets/checkpoints/pi05_libero/` (footnote 1). It is public. All 16 objects carry a 2025-09-08 upload timestamp.
- **LIBERO benchmark**: the authors' own LIBERO copy (present in the first commit, deleted later) is byte-identical to upstream `8f1084e` in 495 of 497 files. The 2 differing files are env-wrapper changes (§8).
- **LIBERO-Pro perturbation suites** (`libero_object_temp_x0.{2,3,4}`) exist upstream at the pinned commit `13613f0`. Each has 50 unique init states per task.

What is missing or conflicting:
1. **The π0.5 multi-sample policy server is not public (CRITICAL).** The client needs a modified OpenPI server that returns `actions` with shape `(N, H, 7)` and a non-standard key `normalized_action`. No such code exists in any of the 3 repos, in the GitHub org (4 repos, none an OpenPI fork), in the author's personal repos, or on the project page. Its *interface* can be inferred cleanly. Its *stochastic protocol* cannot: GR00T re-seeds with 42 on every call, OpenPI splits a PRNG key, and nothing documents which one was used for π0.5.
2. **The α (move_th) value conflicts (MAJOR).** Supplementary §A says "α … we set to 3 in our experiments". Every public π0.5 **LIBERO** AAC command, commented or active, uses `--move_th 4`, and the output directories are named `action_entropy_mv4_seed_4x`. The π0.5 LIBERO-Pro scripts use 3. Nothing shows which value produced Table 2.
3. **The π0.5 baseline chunk size is not specified (MAJOR).** Table 2's "π0.5" row gives no h. The public scripts run `fixed_5`, `fixed_7` and `fixed_10`. OpenPI's own LIBERO example uses `replan_steps=5`.
4. **Seeds are not stated in the paper (MAJOR).** Table 2 values such as 98.5 and 98.7 cannot come from a single 500-episode run, whose resolution is 0.2 %. That implies several seeds were averaged, which fits the 42/43/44 in the scripts. This is inferred, not stated.
5. **The π0.5 client does not run on stock LIBERO `8f1084e`**, which is the version the README tells users to install. It imports `DemoRenderEnv`, which upstream does not export. The authors' private copy patched this. The fix is trivial and recoverable from git history (MAJOR for "runs out of the box", TRIVIAL to reconstruct).
6. **Paper vs code mismatches** in the action-magnitude term: the code weights the gripper at 0.2 where the paper uses 1, and the code clamps h ≥ 2, which the paper does not (§10).
7. **No pre-release history.** All three repos were created on 2026-06-13, two months after arXiv v2. Each has one squashed "initial commit" plus README edits. Semantic drift between the paper runs and this snapshot cannot be audited.

The AAC selector semantics otherwise match paper Eq. 3–6. The remaining gaps are experiment configuration and a private server, not the core algorithm. Even so, reproducing *the exact Table 2 rows* would require choosing among undocumented alternatives: RNG protocol, α ∈ {3, 4}, baseline h ∈ {5, 7, 10}. That meets the definition of PARTIAL.

---

## 2. Paper ↔ Repository Provenance

```
CVPR 2026 paper / arXiv:2604.04161v2
  │  PDF link annotation: https://lance-lot.github.io/adaptive-chunking.github.io/
  │  (the extracted text line-breaks the URL; the annotation itself is correct)
  ▼
Project page  https://lance-lot.github.io/adaptive-chunking.github.io/
  │  links: arXiv 2604.04161, https://github.com/orgs/Adaptive-Action-Chunking/repositories
  │  (source repo: github.com/lance-lot/adaptive-chunking.github.io, a fork of the nerfies template)
  ▼
GitHub org  Adaptive-Action-Chunking   (4 repos, all created 2026-06-13, none are forks)
  ├── libero            "libero benchmark for AAC"            ← π0.5 LIBERO client
  ├── libero-pro        "LIBERO-PRO benchmark for AAC"        ← π0.5 LIBERO-Pro client
  ├── gr00t-multi-sample "Support multiple samples at inference, built from official GR00T N1.5"
  └── robocasa          (not audited; GR00T/RoboCasa only)
  ▼
π0.5 evaluation files:
  aac-libero/pi05_benchmark/inference_client_action_entropy_pi05.py
  aac-libero/action_optimization/action_entropy_pi05.py   (select_chunk_size)
  aac-libero/action_optimization/help_function.py          (compose_rotations)
  aac-libero/pi05_benchmark/{action_entropy_benchmark,fixed_5,fixed_7,fixed_10}_benchmark.sh
  aac-libero-pro/pi05_benhmark/...  (sic: "benhmark")
  π0.5 SERVER → NOT PUBLIC (only an OpenPI websocket client is used)
  π0.5 CHECKPOINT → gs://openpi-assets/checkpoints/pi05_libero/ (paper footnote 1)
```

Identity evidence:
- All three READMEs cite the same CVPR 2026 BibTeX, link to the arXiv paper and the project page, and call themselves "the official codebase of AAC".
- Commit author on every commit: `lance-lot`. The project page lives under the same GitHub user, and the first author is Yuanchang Liang.
- Hard-coded paths such as `/home/sangfor/code/lyc/LIBERO/...` match author affiliation 3 (Sangfor Technologies) and the initials "lyc".
- README defects:
  - `aac-libero/README.md` says "official codebase of AAC for **LIBERO-PRO** evaluation", a copy-paste error; the repo is for LIBERO.
  - It points to `action_entropy_v2.py::select_chunk_size` as "the" AAC implementation. The π0.5 client actually imports **`action_entropy_pi05.py`**, which differs (§5).
  - The LIBERO-Pro README tells users to copy a `pi05_benchmark` folder, but the folder is named `pi05_benhmark`.
  - The GR00T README says `cd Isaac-GR00T` and `python Isaac-GR00T/scripts/policy_server_robocasa.py`, but the repo clones as `gr00t-multi-sample`.

---

## 3. Repository SHAs and History

| Repo | Remote | Branch | HEAD | HEAD date | Tags | Commits |
|---|---|---|---|---|---|---|
| aac-libero | github.com/Adaptive-Action-Chunking/libero | main (tracks origin/main) | `ff8c8e421f7805b5845a72d07fbd8f1171df74bf` | 2026-06-17 10:49 +0800 | none | 6 |
| aac-libero-pro | github.com/Adaptive-Action-Chunking/libero-pro | main | `1c83fa78a172707ab04e6a11d8b148d027f5fb23` | 2026-06-17 10:46 +0800 | none | 3 |
| aac-gr00t-multi-sample | github.com/Adaptive-Action-Chunking/gr00t-multi-sample | main | `11e926b0f34cf6acfcb92c0fe6127a1bdc7b856a` | 2026-06-17 10:34 +0800 | none | 4 |

All working trees are clean. The HEADs match the SHAs given in the task.

Full history (chronological):

| Date (+0800) | Repo | Commit | Content | Semantic impact |
|---|---|---|---|---|
| 2026-06-13 19:26 | gr00t | `b435c32` initial commit | Full GR00T N1.5 fork with multi-sample changes | Baseline snapshot |
| 2026-06-13 19:59 | libero | `cc2c4e2` initial commit | AAC folders **plus a full `libero/` package copy** (497 files) | Baseline snapshot; reveals the authors' LIBERO patches |
| 2026-06-13 20:12 | libero | `f695199` remove libero folder | `.gitignore` 1-line change | None |
| 2026-06-13 20:24 | libero | `483afb9` remove libero folder | Deletes `libero/` | **Removes the authors' `DemoRenderEnv` export.** The π0.5 client then no longer imports on stock LIBERO |
| 2026-06-13 20:30 | libero-pro | `e27b34d` initial | AAC folders only; no `libero/` package ever committed | The authors' modified LIBERO-Pro package was never published |
| 2026-06-16 21:22 | gr00t | `94db423` update readme | README; `policy_server_robocasa.py` 1-line change | Robocasa server only |
| 2026-06-16 21:23 | gr00t | `03f897b` | README | None |
| 2026-06-16 22:46 | libero | `476bda1` | README rewrite | None |
| 2026-06-17 10:22 | libero-pro | `9e12421` | README rewrite | None |
| 2026-06-17 10:27 | libero | `1257f5e` | README; renames `gr00t_benchmark/action_entropy_seed43*.sh` → `action_entropy*.sh` | GR00T scripts only |
| 2026-06-17 10:34–10:49 | all | `11e926b`, `1c83fa7`, `ff8c8e4` | README | None |

**Conclusion.** Nothing after the initial commits touches entropy logic, chunk selection, evaluation loops, π0.5 protocol, sample count, or server APIs. The catch is that the initial commits are *post-publication squashed dumps* made 2 months after arXiv v2. The repos hold no evidence of what the code looked like when Table 2 was produced. The committed shell scripts are a working-directory state: lines are commented out, and only `libero_goal` seeds 43 and 44 are active in `action_entropy_benchmark.sh`. They are not a curated protocol.

---

## 4. CVPR Experimental Protocol

Labels: **PS** = PAPER-STATED, **INF** = INFERRED (with source), **NS** = NOT SPECIFIED in the paper. Code values are noted separately.

### 4.1 π0.5 on standard LIBERO (Table 2)

| Item | Value | Label | Source / note |
|---|---|---|---|
| Base VLA | π0.5 | PS | §5.1.5, Tab. 2 |
| Checkpoint | `gs://openpi-assets/checkpoints/pi05_libero/` | PS | Footnote 1 |
| Checkpoint revision | none given. GCS objects: 16 files, 12.44 GB, all uploaded 2025-09-08T17:59Z (e.g. `params/_METADATA` generation 1757354341847223) | NS | Metadata observed by this audit |
| Fine-tuning for AAC | none | INF | "AAC … without additional training" (§1, §2.1); checkpoint is OpenPI's LIBERO fine-tune |
| OpenPI version / backend | — | NS | Client uses `openpi_client` only |
| Benchmark | LIBERO: Spatial, Long (=libero_10), Object, Goal | PS | Tab. 2 |
| Tasks | 10 per suite, 40 total | PS | §5.1.1 |
| Rollouts/task | 50 | PS (general) | §5.1.2 "50 per task in LIBERO". Stated in the GR00T context and INF to apply to π0.5; scripts use 50 |
| Seeds | — | NS | Scripts: 42, 43, 44. Table 2 decimals (98.5, 92.5, 98.7, 98.1, 99.1) are impossible at single-run 0.2 % resolution, so ≥2 seeds were averaged (INF) |
| Initial states | — | NS | Code: LIBERO default `get_task_init_states(task_id)[episode_idx]`, episodes 0–49 |
| Env seed | — | NS | Code: `env.seed(args.seed)` and `np.random.seed(args.seed)` |
| Control frequency | — | NS | Code: LIBERO default `control_freq=20` |
| Observation frequency | once per executed chunk | INF | Alg. 1 plus client loop |
| Predicted horizon H | — | NS (for π0.5) | OpenPI `pi05_libero`: `action_horizon=10`. Scripts' max `fixed_10` agrees |
| Executed horizon (baseline) | — | **NS** | Scripts: `fixed_5`, `fixed_7`, `fixed_10`. Which one is Table 2's "π0.5" row is unknown |
| AAC candidate horizons | h = 1…H | PS | Alg. 1. Code: h ∈ [2, H] (see §10) |
| # sampled chunks N | 20 | PS | §5.1.5: "we use 20 samples … in all benchmarks" |
| Sampling method | N chunks "in parallel" from the flow-matching head | PS (qualitative) | §4, §5.1.5. RNG/noise protocol NS |
| Flow-matching steps | — | NS | OpenPI default `num_steps=10`; GR00T server uses 4 |
| Continuous entropy | Gaussian differential entropy, Eq. 3, on translation (3-D) and rotation (3-D) separately | PS | §4 |
| Discrete entropy | Bernoulli, Eq. 2, p = c/N | PS | §4 |
| Entropy aggregation | Ē_h = (1/h) Σ_{i<h} (E_t+E_r+E_g), Eq. 4, unit weights | PS | §4 |
| Chunk-size rule | h* = max(argmax_h(Ē_{h+1} − Ē_h), ξ), Eq. 5 | PS | §4 |
| ξ | ξ = argmin_l (m(l) > α), Eq. 6 | PS | Supp. §A |
| Action magnitude m(l) | m_t + m_r + m_g, Eq. 7–10, unit weights | PS | Supp. §A |
| α (move_th) | **3** | PS | Supp. §A ("we set to 3 in our experiments"). **Conflicts with the π0.5 LIBERO scripts (4)** |
| Fallback if no l satisfies m(l) > α | — | NS | Code: ξ = H (execute the full chunk) |
| Space where entropy is computed | — | NS | Code: normalized model output (`normalized_action[..., :6]`) |
| Gripper binarization | G ∈ {0,1} | PS (concept) | Threshold NS. Code: `action[...,6] ≥ 0` → 1 |
| Which of the N chunks is executed | — | NS | "the first h* actions will be executed". Code: sample index 0 |
| Smoothing / temporal ensembling | none mentioned | INF (none) | Code: none |
| Max episode length | — | NS | Code: 220/280/300/520 sim steps, **including** 10 wait steps |
| Metric | success rate (%) per suite plus mean | PS | Tab. 2 |
| Hardware (π0.5) | — | NS | A800 timing in Tab. 4 is GR00T-on-LIBERO |

**Table 2 (paper):**

| Method | Spatial | Long | Object | Goal | Avg |
|---|---|---|---|---|---|
| π0.5 | 98.5 | 92.5 | 98.7 | 98.1 | 97.0 |
| π0.5 + AAC | 99.1 | 95.2 | 99.2 | 98.0 | 97.9 |

### 4.2 π0.5 on LIBERO-Pro (Table 3)

| Item | Value | Label |
|---|---|---|
| Subset | "position perturbation subset of LIBERO-Pro" | PS |
| Levels | ×0.2, ×0.3, ×0.4 | PS |
| Suite | libero_object | INF (only `libero_object_temp_x*` appears in any script) |
| Axis | x | INF (`_x0.2`; upstream also has `_y*`) |
| Rollouts, seeds | — | NS. Scripts: 50 trials; seeds 42/43/44. Decimals (29.9, 9.5, 35.3) imply multiple seeds |
| π0.5 baseline h | — | NS |
| α | — | NS. π0.5 LIBERO-Pro scripts: 3 |
| Public π0.5 scripts cover | x0.2, x0.4 only. **No x0.3 π0.5 script** | — |

Table 3, π0.5 rows: π0.5 = 53.2 / 29.9 / 9.5 (avg 30.9); π0.5+AAC = 57.4 / 35.3 / 11.8 (avg 34.8).

### 4.3 GR00T N1.5 context (not the target)

- Fine-tuned by the authors from public GR00T N1.5: diffusion head only, all 4 LIBERO suites jointly, 90 epochs, 8×A800 (PS).
- Server config hard-codes the local checkpoint `output/gr00t15_libero_lerobot_h16`, which is not public.
- H = 16 (PS). Fixed baselines h ∈ {2, 4, 8, 12, 16} (PS). N = 20 (PS). Denoising steps 4 (code).
- Table 4's sample-count sweep (1–40 samples; 94.1 at 1 sample = GR00T default) is GR00T on LIBERO.

---

## 5. AAC Algorithm → Code Mapping

**Where AAC lives:** entirely in the simulation client (`action_optimization/` plus the client loop). The GR00T server only generates N samples; `grep -i "entropy|chunk_size|move_th"` over the server finds nothing AAC-related. This confirms the README's claim.

Call chain for π0.5 (standard LIBERO):

```
inference_client_action_entropy_pi05.py::main
  └─ policy.infer(obs_dict)                        [openpi_client websocket → PRIVATE server]
       ← {"actions": (N,H,7), "normalized_action": (N,H,≥7)}
  ├─ if "fixed" in selector: slice [:, :k]
  └─ else select_chunk_size(pred_action_dict, method="gaussian_bernoulli", move_th)
        ├─ convert_gripper_to_binary(action.gripper_close, threshold=0)
        ├─ action_entropy_gausssian_bernoulli(...)
        │     ├─ gaussian_entropy_from_samples(normalized_action[:, t, 0:3])  → _safe_logdet_cov
        │     ├─ gaussian_entropy_from_samples(normalized_action[:, t, 3:6])
        │     └─ bernoulli_entropy_from_samples(binary_gripper[:, t])
        ├─ elbow: argmax(diff(chunk_mean)) + 1, clamped ≥ 2
        ├─ action_magnitude(pred_action_dict, chunk_id=0) → help_function.compose_rotations
        └─ find_first_idx(chunk_m, move_th)
  ├─ chunk_id = 0 (chunk_id_selector "0")  [select_chunk_id is used only for "backward"]
  └─ env.step(...) for the first h* actions of sample 0
```

| FILE : FUNCTION | CALLED BY | INPUT → OUTPUT | LOGIC | PAPER | MATCH |
|---|---|---|---|---|---|
| `pi05_benchmark/inference_client_action_entropy_pi05.py : main` (L184–305) | CLI | obs → env actions | Builds OpenPI-style obs (images 256² flipped 180°, 8-D state, prompt); calls server; asserts `actions.ndim==3`; fixed or AAC slicing; executes sample 0 | Alg. 1, §4 "first h* actions" | MATCH (sample choice NS in paper) |
| `action_optimization/action_entropy_pi05.py : select_chunk_size` (L247) | client L227 | dict of (N,H,·) → (h*, breakdown) | Deep-copies, binarizes gripper, gets entropy breakdown, elbow `max(argmax(diff)+1, 2)`, `h* = max(ξ, elbow)` | Eq. 5, 6 | MATCH except extra clamp ≥ 2 |
| `… : convert_gripper_to_binary` (L403) | select_chunk_size | continuous gripper → {0,1} | `pred ≥ 0` | "G ∈ {0,1}" | Threshold NS in paper |
| `… : action_entropy_gausssian_bernoulli` (L46) | select_chunk_size | dict → per-step H_pos, H_rot, H_grip, total, chunk_mean | total_t = H_pos + H_rot + H_grip (weights 1); chunk_mean[k] = mean(total[:k]) | Eq. 3, 2, 4 | MATCH. Entropy computed on **normalized** actions (NS in paper) |
| `… : gaussian_entropy_from_samples`, `_safe_logdet_cov` (L10–32) | above | (N, d) → scalar | ½(d·log 2πe + logdet(np.cov + 1e-6·I)); Cholesky with slogdet fallback | Eq. 3 | MATCH plus unstated ε-regularization and unbiased covariance |
| `… : bernoulli_entropy_from_samples` (L34) | above | (N,) binary → scalar | p1 = clip(mean, 1e-9, 1−1e-9); natural-log entropy. Its `assert len(np.unique(g) <= 2)` is vacuous (always true) | Eq. 2, p = c/N | MATCH |
| `… : action_magnitude` (L200) | select_chunk_size | sample 0 unnormalized actions → m(l), l = 2…H | ‖Σ Δpos‖ + ‖axis-angle(∏ q(Δrot))‖ + **0.2**·1[gripper toggles within l] | Eq. 7–10 | **MISMATCH**: gripper weight 0.2 vs paper 1. Eq. 9's "‖∏Δq‖" is read as the axis-angle norm, a sensible reading of ambiguous notation |
| `help_function.py : compose_rotations` (L83) | action_magnitude | (l, 3) axis-angles → axis-angle | Sequential quaternion product | Eq. 9 | MATCH |
| `… : find_first_idx` (L233) | select_chunk_size | m(l) list → ξ | First l (starting at 2) with m(l) > α; else returns H | Eq. 6 | MATCH; the fallback (ξ = H) is code-only |
| `action_sampler.py : select_chunk_id` (L40) | client only if `--chunk_id_selector backward` | — | Backward-coherence sample choice (BID-like) | Not in paper | Unused in all scripts. In aac-libero the client passes the kwarg `pred_chunk_szie` but the function expects `pred_chunk_size`, so the call would raise TypeError. The aac-libero-pro version uses the typo'd name and works |
| Client fixed branch (L222–225) | main | → first k actions of sample 0 | `fixed_{k}` | Fixed-h baseline | MATCH |
| `action_entropy_v2.py : select_chunk_size` | GR00T LIBERO client only | — | Same as the pi05 file except: no gripper binarization; horizon hard-coded to 16; `find_first_idx` fallback = 16 | — | Referenced by the README but **not** used by the π0.5 path |

Consequence of the fallback: if the arm barely moves within the chunk, so that m(l) ≤ α for all l, then ξ = H and AAC executes the **full** chunk regardless of entropy. The value of α therefore controls how often entropy matters at all. That is why the α = 3 vs 4 conflict matters.

A useful property: continuous entropy is invariant to any fixed linear rescaling of the action space, because logdet(AΣAᵀ) = logdet Σ + const, and the constant cancels in `diff(chunk_mean)`. The choice between "normalized" and raw actions therefore affects h* only through the ε = 1e-6 regularizer, i.e. when per-dimension sample variance is near 1e-6.

---

## 6. π0.5 Reproduction Path

Paths: `aac-libero/pi05_benchmark/` (8 files) and `aac-libero-pro/pi05_benhmark/` (10 files). The two clients differ only in suite handling, init-state loading and the BDDL path (diff in §8.2).

| # | Question | Answer (evidence) |
|---|---|---|
| 1 | Exact π0.5 implementation expected? | OpenPI π0.5, `pi05_libero` config/checkpoint (paper footnote). Client input keys `observation/image`, `observation/wrist_image`, `observation/state`, `prompt` match OpenPI `LiberoInputs` exactly |
| 2 | Does the client use OpenPI? | Yes: `from openpi_client import websocket_client_policy` (client L18–19, L134) |
| 3 | Which OpenPI repo/version? | Not stated. Only the pip package `openpi_client` is implied |
| 4 | Exact commit/revision? | **No.** Not for OpenPI, `openpi_client`, or the checkpoint |
| 5 | Is the public code only a client? | **Yes.** README: "This repo only includes the client part" |
| 6 | Expected endpoint/API | OpenPI websocket (`WebsocketClientPolicy.infer`, msgpack-numpy). Default host `10.72.1.16` (private IP). `--port 8000` maps through a hard-coded `PORT_MAP` to **2668**, so the server must listen on 2668, or `--host` must point to a forwarder |
| 7 | Request schema | `{"observation/image": uint8 (256,256,3) agentview flipped [::-1,::-1], "observation/wrist_image": uint8 (256,256,3) flipped, "observation/state": float (8,) = eef_pos(3) ‖ axis-angle(eef_quat)(3) ‖ gripper_qpos(2), "prompt": task.language}`. No client-side resize (the OpenPI example resizes to 224 with padding; the OpenPI model resizes server-side in `preprocess_observation`) |
| 8 | Response schema | `{"actions": float (N, H, 7) unnormalized LIBERO actions (gripper continuous in [−1,1]), "normalized_action": float (N, H, D≥7)}`. `actions.ndim == 3` is asserted (L211). **Stock OpenPI returns `(H, 7)` and has no `normalized_action` key**, so an unmodified server fails the assert or raises KeyError |
| 9 | Must one call return multiple chunks? | **Yes** (dimension 0 = N; entropy is over axis 0) |
| 10 | How many samples? | Not in the client. Paper: 20 "in all benchmarks". GR00T server: `n_samples=20` |
| 11 | Is the exact multi-sample π0.5 server public? | **No** |
| 12 | Search of all three repos | No π0.5/OpenPI server code. `grep openpi|pi05|websocket|normalized_action` hits only the π0.5 client, `test_client.py`, scripts, the entropy module, and one comment in `scripts/convert_hdf5_to_lerobot.py`. `normalized_action` is produced only by GR00T `policy.py` |
| 13 | External links / comments / history | READMEs link only to GR00T-multi-sample, RoboCasa, LIBERO, LIBERO-Pro, the project page and arXiv. The org has 4 repos, no OpenPI fork; the author's personal repos have none either. `test_client.py` connects to `10.72.1.16:2668` and reads `["normalized_action"]`, which proves the authors ran a modified private server. History has no deleted server files |
| 14 | Base checkpoint specified exactly? | Path yes (`gs://openpi-assets/checkpoints/pi05_libero/`); revision no |
| 15 | Public? | Yes. GCS listing is readable: `params/*` plus `assets/physical-intelligence/libero/norm_stats.json`; 12.44 GB total |
| 16 | Exact model revision? | No. Object timestamps (2025-09-08) are the only pin available |
| 17 | Normalization stats | Not in AAC code. Public inside the checkpoint (`norm_stats.json`, 1914 B); OpenPI uses quantile normalization for π0.5. Which normalization the private server used for `normalized_action` is NS. Largely irrelevant to entropy (§5 invariance), relevant only through ε |
| 18 | Action/state dims | State 8 (client). Actions 7 (client slices 0:3, 3:6, 6). `normalized_action` uses `[..., :7]` only. OpenPI model action_dim = 32 (padded) |
| 19 | Prediction horizon | Not in client or paper. OpenPI `pi05_libero`: H = 10. The client derives T from the response |
| 20 | Executed horizon | Baseline: `fixed_{5,7,10}` in scripts; which is Table 2's is unknown. AAC: h* ∈ [2, H] |
| 21 | Cameras / preprocessing | `agentview_image` → `observation/image`, `robot0_eye_in_hand_image` → `observation/wrist_image`, both rotated 180°, rendered at 256×256 (`LIBERO_ENV_RESOLUTION=256`), no client resize. Server-side resize depends on the OpenPI version |
| 22 | Language instruction format | Raw `task.language` string (same as the OpenPI example) |
| 23 | Seeds | Client seeds `np.random` and `env.seed`; scripts use 42/43/44. **Server-side sampling seed not specified** |
| 24 | Initial LIBERO states | LIBERO default `.pruned_init` files, `initial_states[episode_idx]` for episodes 0–49. Exact if LIBERO `8f1084e` is used |
| 25 | Can the exact experiment run without implementing missing research logic? | **No.** The multi-sample server must be written (small, interface-determined), and three protocol choices must be made that no public source settles: RNG protocol, α, baseline h |

Additional protocol notes, all public and exact but differing from OpenPI's reference LIBERO eval:
- **Episode budget.** AAC stops at `t ≥ max_steps` with t counting the 10 wait steps. OpenPI allows `max_steps + num_steps_wait`. AAC therefore gets 10 fewer policy steps than OpenPI's published π0.5 LIBERO numbers.
- **Environment class.** AAC uses `DemoRenderEnv`; OpenPI uses `OffScreenRenderEnv`.
- **Seed.** AAC uses 42/43/44; OpenPI uses seed 7.
- **CLI quirk.** `--save_videos`, `--debug` etc. are declared as `argparse type=bool`, so `--save_videos False` evaluates to True. Videos can only be disabled via `--debug 1`. As a side effect, `args.json` is written only when videos are saved.

---

## 7. GR00T Multi-Sample Reference Path

Diff against upstream `NVIDIA/Isaac-GR00T@n1.5-release`:

| File | Change |
|---|---|
| `gr00t/model/action_head/flow_matching_action_head.py::get_action` | New arg `n_samples`. `vl_embs.expand(n_samples,…)`, `embodiment_id.expand(n_samples)`, `state.expand(n_samples,…)`. Noise `torch.randn(batch=n_samples, H, action_dim)`, so each sample gets **independent noise** over **one shared backbone forward pass**. Euler integration unchanged |
| `gr00t/model/gr00t_n1.py::get_action` | Passes `n_samples` through |
| `gr00t/model/policy.py::Gr00tPolicy` | New ctor args `seed_per_step`, `n_samples`. If `seed_per_step`, it **re-seeds python/numpy/torch/cuda/tf with 42 before every call**, so the same N noise tensors are drawn at every step. Output adds `normalized_action` = pre-unnormalization model output (N, 16, 32). No `squeeze` when batched |
| `scripts/inference_gr00t.py::GrootWrapPolicy` | `denoising_steps=4`, `embodiment_tag="new_embodiment"`, `set_seed(42)` once at init |
| `scripts/policy_server_libero.py` | `n_samples=20`, `seed_per_step=True`, `data_config_name="libero"`, `model_path="output/gr00t15_libero_lerobot_h16"` (**private** checkpoint), port 8080 |
| `scripts/base_server.py` | ZeroMQ REP plus pickle; endpoints `ping`, `kill`, and the `get_action` registration |

Returned shapes: `action.end_effector_position` (20,16,3), `action.end_effector_rotation` (20,16,3), `action.gripper_close` (20,16) with the gripper "binary" normalization mode, and `normalized_action` (20,16,32). Sample order is the batch index; sample 0 is the executed chunk under `chunk_id_selector 0`.

| Responsibility | GR00T | π0.5 (as implied by client) |
|---|---|---|
| N-sample generation | Server (public) | Server (**private**) |
| Entropy | Client | Client (public) |
| Chunk-size selection | Client | Client (public) |
| Chunk (sample) selection | Client (index 0) | Client (index 0) |
| Transport | ZMQ + pickle, `get_action` | OpenPI websocket + msgpack, `infer` |
| RNG | Re-seed 42 every call | **Unknown** |
| Denoising steps | 4 | Unknown (OpenPI default 10) |

**"If the π0.5 server is missing, can its required behavior be inferred unambiguously from the GR00T implementation plus the π0.5 client interface?"**

*Functional behavior: yes.* Run one OpenPI `pi05_libero` forward on the observation, batched or tiled to N = 20, with independent Gaussian noise per sample. Return unnormalized 7-D actions `(20, 10, 7)` via OpenPI's output transforms, and the pre-unnormalization model output `(20, 10, 32)` as `normalized_action`, on port 2668. This mirrors the GR00T change one-for-one. OpenPI's `Pi0.sample_actions` already draws noise of shape `(batch, H, 32)`; only `Policy.infer`'s batch-1 packing (`[np.newaxis]`, `x[0, ...]`) and the output dict need changes. Historical note: OpenPI changed `LiberoOutputs` from `[:, :7]` to `[..., :7]` on 2026-06-16 (commit `15a9616`, "support batching"). Any server the authors used before that date must therefore have carried its own patch.

*Stochastic protocol: no.* Whether the π0.5 server re-seeded on every call (GR00T-style, which makes noise identical across steps and makes the "fixed" baseline effectively deterministic given the observation) or used OpenPI's PRNG-key splitting cannot be determined. The same applies to denoising step count and JAX vs PyTorch backend.

**Reconstruction class: SMALL BUT UNAMBIGUOUS for the interface and sampling semantics. The per-call RNG protocol is an undocumented choice** that affects exact numbers, not the method's definition. Taken alone, this would be borderline RECONSTRUCTIBLE. The experiment-level conflicts in §10 and §11 are what push the overall classification to PARTIAL.

---

## 8. LIBERO / LIBERO-Pro Version Map

### 8.1 Standard LIBERO

- Pinned: `Lifelong-Robot-Learning/LIBERO@8f1084e3132a39270c3a13ebe37270a43ece2a01` (2025-03-15, "Add support for dataset download from huggingface"). Upstream HEAD is also `8f1084e`.
- Repo type: `aac-libero` is an **overlay** repo. The README says to install official LIBERO and copy `action_optimization/`, `gr00t_benchmark/` and `pi05_benchmark/` into the LIBERO root. The scripts `cd <LIBERO>/pi05_benchmark` and import `action_optimization.*`, so the LIBERO root must be on `sys.path` (a legacy editable install or PYTHONPATH). The top-level `benchmark_scripts/`, `scripts/`, `templates/` and `notebooks/` are upstream LIBERO leftovers plus some authors' scripts.
- The **authors' actual LIBERO copy** (commit `cc2c4e2`, 497 files) was compared blob-SHA by blob-SHA with upstream `8f1084e`: **495 identical, 2 different, 0 extra**. All BDDL files, init states and assets are identical. The differences:
  - `libero/libero/envs/__init__.py`: adds `DemoRenderEnv` to the exports. **Required by the π0.5 and GR00T clients** (`from libero.libero.envs import OffScreenRenderEnv, DemoRenderEnv`). Stock `8f1084e` fails with ImportError.
  - `libero/libero/envs/env_wrapper.py`: `DemoRenderEnv` sets `has_renderer=True` (upstream `False`), which needs an on-screen display. `SegmentationRenderEnv` id-mapping changes are not used by the clients. Neither affects dynamics or observations.
- Task definitions, init states and task order are therefore **PUBLIC + EXACT** once the `DemoRenderEnv` export is restored. The restoration can be recovered verbatim from `git show cc2c4e2:libero/libero/envs/__init__.py`.
- Simulator: robosuite/mujoco versions follow LIBERO's `requirements.txt`. The AAC repo's `requirements.txt` is the upstream copy. No lock file exists.

### 8.2 LIBERO-Pro

- Pinned: `Zxy-MLlab/LIBERO-PRO@13613f0cc90e3869b157d34f7a88e7d6041cd423` (2026-01-09). This commit only modifies `images/wechat.png`. The later upstream HEAD `eafdb80` (2026-03-23) also only touches `wechat.png`, so the pin is code-equivalent to HEAD.
- `libero_object_temp_{x,y}0.{1..5}` BDDL and init files exist upstream (added 2025-10-22 to 2025-11-13). Each init file for x0.2 and x0.4 was checked: 50 states × 110 dims, 50 unique. Task order comes from `libero_suite_task_map["libero_object_temp"]`.
- `aac-libero-pro` is an **overlay** repo too. It never contained the `libero/` package. Its `perturbation.py`, `evaluation_config.yaml`, `libero_ood/*.yaml` and `notebooks/generate_init_states.py` are upstream LIBERO-Pro files.
- **The client needs unpublished LIBERO-Pro modifications:**
  1. `task_suite.get_task_init_states(task_id, args.task_suite_name)`. Upstream signature is `get_task_init_states(self, i)`, so the call raises TypeError. Intended behavior (unambiguous from the upstream file layout): load `init_files/<task_suite_name>/<task>.pruned_init`.
  2. `DemoRenderEnv` export (same as LIBERO).
  3. Hard-coded `BDDL_FILE_PATH = "/home/sangfor/code/lyc/LIBERO-PRO/libero/libero/bddl_files"`.
  4. `benchmark_key` is undefined unless "temp" is in the suite name, so only `*_temp_*` suites run.
- Perturbation definition: position perturbation along x at levels 0.2/0.3/0.4 (the naming follows the AAC README). The generator of the `_x0.2` BDDLs is not in either repo; the files themselves are upstream.

---

## 9. Public Asset Matrix

| COMPONENT | REQUIRED FOR | PUBLIC? | EXACT VERSION? | LOCATION | STATUS | NOTES |
|---|---|---|---|---|---|---|
| AAC π0.5 client (LIBERO) | Table 2 | Yes | Yes (snapshot) | `aac-libero@ff8c8e4:pi05_benchmark/inference_client_action_entropy_pi05.py` | PUBLIC + EXACT | Snapshot is post-paper; no earlier history |
| AAC π0.5 client (LIBERO-Pro) | Table 3 | Yes | Yes | `aac-libero-pro@1c83fa7:pi05_benhmark/…` | PUBLIC + EXACT | Needs private LIBERO-Pro API (§8.2) |
| π0.5 policy implementation | All π0.5 | Yes | No | github.com/Physical-Intelligence/openpi | PUBLIC + VERSION UNSPECIFIED | Latest checked `215abfb` (2026-08-24) |
| π0.5 multi-sample server | All π0.5 AAC and baselines | **No** | — | author-private (`10.72.1.16:2668`) | **MISSING** | Interface inferable (§7) |
| π0.5 checkpoint | All π0.5 | Yes | Path only | `gs://openpi-assets/checkpoints/pi05_libero/` | PUBLIC + VERSION UNSPECIFIED | Objects dated 2025-09-08; 12.44 GB |
| Model revision | All π0.5 | — | No | — | UNCLEAR | GCS generation numbers are the only pin |
| Normalization statistics | Server | Yes | Tied to checkpoint | `…/pi05_libero/assets/physical-intelligence/libero/norm_stats.json` | PUBLIC + VERSION UNSPECIFIED | How `normalized_action` was formed is NS |
| LIBERO | Table 2 | Yes | Yes | LIBERO@`8f1084e` | PUBLIC + EXACT | Plus the `DemoRenderEnv` export patch (from `cc2c4e2`) |
| LIBERO-Pro | Table 3 | Yes | Yes | LIBERO-PRO@`13613f0` | PUBLIC + EXACT | Plus unpublished `get_task_init_states(i, suite)` |
| Task definitions (BDDL) | both | Yes | Yes | upstream | PUBLIC + EXACT | Blob-identical to the authors' copy |
| Assets | both | Yes | Yes | upstream | PUBLIC + EXACT | — |
| Simulator (robosuite/mujoco) | both | Yes | Via LIBERO requirements | upstream `requirements.txt` | PUBLIC + VERSION UNSPECIFIED | No lock file |
| Evaluation configs (scripts) | both | Yes | Conflicting | `pi05_benchmark/*.sh` | UNCLEAR | α = 4 (LIBERO) vs paper 3; lines commented out |
| Action entropy code | AAC | Yes | Yes | `action_optimization/action_entropy_pi05.py` | PUBLIC + EXACT | Minor paper mismatches (§10) |
| Chunk selector | AAC | Yes | Yes | same, `select_chunk_size` | PUBLIC + EXACT | — |
| Fixed-chunk baseline | Baseline row | Yes (logic) | h unknown | client `fixed_{k}` | UNCLEAR | Which k is Table 2's is NS |
| Number of action samples | AAC | Paper yes | 20 | paper §5.1.5; GR00T server | PUBLIC + EXACT (paper) | Not in the π0.5 client (server-side) |
| Random seed logic (client) | Protocol | Yes | Yes | client | PUBLIC + EXACT | 42/43/44 in scripts |
| Random seed logic (server) | Protocol | No | — | — | MISSING | GR00T precedent: re-seed 42 per call |
| Evaluation / init-state logic | Protocol | Yes | Yes | client plus LIBERO | PUBLIC + EXACT | LIBERO-Pro loader missing |
| Datasets | Not required | — | — | — | N/A | Checkpoint already fine-tuned; no training |

---

## 10. Paper vs Code Consistency

| ITEM | PAPER | PUBLIC CODE (π0.5 path) | MATCH? |
|---|---|---|---|
| Action entropy equation | Ē_h = (1/h) Σ_i Σ_{j∈{t,r,g}} E_j^i (Eq. 4) | `chunk_mean[k] = mean(H_pos+H_rot+H_grip)[:k]`, weights 1 | MATCH |
| Continuous estimator | ½ log[(2πe)^d det Σ] (Eq. 3) | Same, with `np.cov` (ddof=1) + **1e-6·I**, on **normalized** actions | MATCH with unstated details (ε, normalization) |
| Discrete estimator | p = c/N, Shannon entropy (Eq. 2) | Same; binarization `≥ 0`; clip 1e-9 | MATCH (threshold NS in paper) |
| Sample count | 20, all benchmarks | Not in client; GR00T server 20 | MATCH (π0.5 server private) |
| Candidate horizons | h = 1…H (Alg. 1) | Elbow clamped to ≥ 2; magnitude from l = 2 | **MISMATCH (minor)**: h* = 1 is impossible in code |
| Threshold α | 3 (Supp. A) | π0.5 LIBERO scripts **4** (all 12 lines, `mv4` dirs); π0.5 LIBERO-Pro 3; GR00T 3 | **MISMATCH** for the target experiment |
| select_chunk_size logic | h* = max(argmax_h(Ē_{h+1}−Ē_h), ξ) | `max(find_first_idx, max(argmax(diff)+1, 2))` | MATCH (plus clamp) |
| ξ / move_th | argmin_l(m(l) > α); m = m_t + m_r + m_g | First l ≥ 2 with m > α, else H; m = ‖Σpos‖ + ‖∏rot‖ + **0.2**·toggle, from **sample 0** unnormalized actions | **MISMATCH**: gripper weight 0.2 vs 1. Fallback and sample index NS in paper |
| Sampling procedure | N chunks sampled in parallel | GR00T: shared backbone, independent noise, re-seed 42 per call; π0.5: unknown | UNCLEAR (π0.5) |
| Fixed baseline | "π0.5" (h not given) | fixed_5 / fixed_7 / fixed_10 | UNCLEAR |
| π0.5 action horizon | NS | T from server; OpenPI H = 10; fixed_10 max | CONSISTENT (inferred) |
| GR00T action horizon | 16 | `action_indices = range(16)`; `fixed_16` | MATCH |
| Rollout count | 50/task (LIBERO) | `--num_trials_per_task 50` | MATCH |
| Seed protocol | NS | 42/43/44; Table 2 decimals imply multi-seed | UNCLEAR |
| Task definitions | LIBERO 4 suites × 10 | LIBERO `8f1084e` (blob-exact) | MATCH |
| LIBERO-Pro levels | ×0.2, ×0.3, ×0.4 | π0.5 scripts: x0.2, x0.4 only | **GAP**: no π0.5 x0.3 script |
| Executed chunk among N | NS | Sample 0 | NS (not a mismatch) |
| README "AAC implementation" | — | Points to `action_entropy_v2.py`; π0.5 uses `action_entropy_pi05.py` | Documentation mismatch |

---

## 11. Missing Components and Ambiguities

**Missing code**
- M1. π0.5 multi-sample OpenPI server. Interface-determined; RNG protocol undetermined.
- M2. LIBERO `DemoRenderEnv` export. Trivial; verbatim recoverable from `cc2c4e2`.
- M3. LIBERO-Pro `get_task_init_states(i, suite)`, BDDL path, `benchmark_key`. Small; unambiguous from the upstream file layout.

**Undocumented or conflicting protocol parameters** (each changes Table 2 numbers)
- A1. Server RNG: re-seed 42 per call (GR00T precedent) vs OpenPI key-splitting (default key 0). The paper is silent.
- A2. α for π0.5 LIBERO: 3 (paper) vs 4 (every public π0.5 LIBERO AAC command).
- A3. π0.5 baseline executed horizon: 5, 7 or 10.
- A4. OpenPI version/backend and flow steps (default 10); server-side image resize path.
- A5. Seed count and averaging (inferred 3 seeds: 42, 43, 44).
- A6. LIBERO-Pro: suite (object), axis (x) and baseline h are inferred; the x0.3 π0.5 command is absent.

**Not ambiguous** (public and exact): task set, init states, episode budget, observation construction, state encoding, prompt, image flipping, entropy and elbow math, magnitude math as coded, N = 20, 50 trials.

---

## 12. Reproduction Risk / Red Flags

| # | Severity | Red flag | Evidence |
|---|---|---|---|
| 1 | **CRITICAL** | Client public, **π0.5 server missing**; the server exists only for GR00T | No OpenPI fork in the org or user repos; `test_client.py` reads a non-standard `normalized_action` from `10.72.1.16:2668` |
| 2 | **MAJOR** | Paper/code conflict on α (3 vs 4) for the exact target experiment | Supp. §A vs `action_entropy_benchmark.sh` (`--move_th 4`, `mv4` dirs) |
| 3 | **MAJOR** | π0.5 baseline chunk size unspecified | Tab. 2 caption; scripts carry fixed_5/7/10 |
| 4 | **MAJOR** | Server-side RNG protocol hidden; GR00T precedent is non-standard (re-seed per step) | `gr00t/model/policy.py` `seed_per_step` |
| 5 | **MAJOR** | Post-publication squashed code dump, so drift cannot be audited | All repos created 2026-06-13; 1 code commit each; scripts contain edited/commented state |
| 6 | **MAJOR** | The client does not import on the pinned stock LIBERO | `DemoRenderEnv` not exported at `8f1084e`; the authors' patched copy was deleted in `483afb9` |
| 7 | **MAJOR** | Seeds and averaging not stated in the paper | Tab. 2 decimals imply multi-seed; scripts 42–44 |
| 8 | MINOR | Gripper magnitude weight 0.2 (code) vs 1 (paper) | `action_magnitude: wgrip = 0.2` |
| 9 | MINOR | h ≥ 2 clamp not in the paper | `max(best_k, 2)` |
| 10 | MINOR | ε = 1e-6 covariance regularization and normalized-space entropy not in the paper | `_safe_logdet_cov` |
| 11 | MINOR | Checkpoint and OpenPI revisions unpinned | Footnote gives the path only |
| 12 | MINOR | LIBERO-Pro client depends on unpublished LIBERO-Pro API changes and hard-coded paths | §8.2 |
| 13 | MINOR | README points to `action_entropy_v2.py`, which is not the π0.5 path | README vs client import |
| 14 | MINOR | Episode budget is 10 steps shorter than OpenPI's reference LIBERO eval, so π0.5 baseline numbers are not directly comparable to OpenPI's published numbers | client L302 vs OpenPI `main.py` |
| 15 | MINOR | `type=bool` CLI flags; `select_chunk_id` kwarg typo (aac-libero); vacuous gripper assert | client L61–69; `action_sampler.py` |
| 16 | MINOR | The GR00T LIBERO-Pro client imports `action_entropy_pi05` (gripper threshold 0) while GR00T grippers are binary {0,1}. After thresholding every value is 1, so gripper entropy is always 0 | `aac-libero-pro/gr00t_benchmark/inference_client_gr00t.py:23`. Not the target; noted as further evidence that the scripts are not a curated protocol |
| 17 | MINOR | Exact stochastic replay is impossible anyway (GPU/JAX nondeterminism, unknown RNG) | — |

---

## 13. Proposed Minimal Phase 8B Experiment

**By the rubric this is not triggered.** The classification is PARTIAL, so an exact reproduction cannot be designed from public information. The design below is a **conditional, assumption-declared re-implementation**. It applies only if you explicitly accept A1–A4 as declared protocol variants and Phase 8B is labelled "faithful re-implementation", not "exact reproduction". Nothing below was executed.

**Required transparent reconstructions** (not scientific, but code the authors did not publish):
- R1. OpenPI `pi05_libero` multi-sample server on port 2668:
  - Tile or batch the observation to N = 20.
  - Keep the batch dimension in the outputs.
  - Return `actions` with shape (20, 10, 7) via the standard output transforms (`[..., :7]`).
  - Return the pre-`Unnormalize` model output (20, 10, 32) as `normalized_action`.
  - Log the per-call RNG mode.
- R2. Add the `DemoRenderEnv` export to LIBERO `8f1084e`'s `libero/libero/envs/__init__.py`, copied verbatim from `cc2c4e2`. Keep upstream `has_renderer=False` for headless runs, or match the authors with `True` plus a virtual display.
- R3. Copy `action_optimization/` and `pi05_benchmark/` into the LIBERO root; set `PYTHONPATH=<LIBERO root>`.

**Declared variants** (enumerate, do not choose silently):
- A1: RNG = `reseed42_per_call` (GR00T precedent) **and** `openpi_default_split`.
- A2: α ∈ {3 (paper), 4 (script)}.
- A3: baseline h ∈ {5, 10}; add 7 if budget allows.
- A4: pin an OpenPI commit (e.g. `215abfb`), JAX backend, `num_steps=10`.

**Stage 8B-0: smoke test** (answers Q1–Q4)
```sh
# server: modified OpenPI (R1) — reference command for the *unmodified* public server:
#   uv run scripts/serve_policy.py --port 2668 policy:checkpoint \
#       --policy.config=pi05_libero --policy.dir=gs://openpi-assets/checkpoints/pi05_libero
cd <LIBERO@8f1084e>/pi05_benchmark
PYTHONPATH=<LIBERO@8f1084e> python inference_client_action_entropy_pi05.py \
  --host 127.0.0.1 --port 8000 --num_trials_per_task 2 --task_suite_name libero_spatial \
  --out_path ~/aac8b/smoke_aac_mv3 --chunk_size_selector gaussian_bernoulli \
  --chunk_id_selector 0 --move_th 3 --seed 42
# same with --chunk_size_selector fixed_10
```
Pass criteria:
- (1) Episodes run and some succeed.
- (2) The server returns `actions.shape == (20,10,7)`, and the per-step sample variance is > 0.
- (3) `chunk_size_log.csv` shows non-constant chunk sizes in [2, 10].
- (4) The fixed run logs a constant chunk size of 10.

**Stage 8B-1: direction check** (Q5), LIBERO-Long, which has the largest reported gain (92.5 → 95.2):
```sh
for SEL in "gaussian_bernoulli --move_th 3" "gaussian_bernoulli --move_th 4" "fixed_5 --move_th 3" "fixed_10 --move_th 3"; do
  PYTHONPATH=<LIBERO> python inference_client_action_entropy_pi05.py --host 127.0.0.1 --port 8000 \
    --num_trials_per_task 50 --task_suite_name libero_10 --out_path ~/aac8b/l10 \
    --chunk_size_selector ${SEL%% *} --chunk_id_selector 0 ${SEL#* } --seed 42
done
```
That is 4 × 500 = 2,000 episodes per RNG mode.

Caveat: at 500 episodes per arm, a 2.7-point difference near 93 % has a standard error of about 1.6 points, so a single seed cannot establish direction statistically. Q5 can only be answered weakly at this scale; the full 3-seed suite is the minimum for a meaningful comparison.

---

## 14. Estimated Full Reproduction Cost

| Quantity | Value | Basis |
|---|---|---|
| Tasks | 40 (Table 2); 10 per level × 3 levels (Table 3) | PS / INF |
| Rollouts per task | 50 | PS |
| Seeds | 3 | INF (scripts) |
| Episodes per method, Table 2 | 40 × 50 × 3 = **6,000** | Derived |
| Methods, Table 2 | 2 as reported; **4–5** given the ambiguities (AAC α ∈ {3,4}; fixed 5/7/10) | A2, A3 |
| Episodes, Table 2 (ambiguity-covering) | 24,000–30,000 (×2 if both RNG modes are run) | Derived |
| Episodes, Table 3 | 10 × 50 × 3 × 3 levels × ≥2 methods = ≥ 9,000 | INF |
| Max policy steps per episode | 210 / 270 / 290 / 510 (Spatial / Object / Goal / Long) | Code (max_steps − 10) |
| Policy calls, fixed_10, per seed (upper bound) | (210+270+290+510)/10 × 50 × 10 ≈ **64,000** | Derived; success ends episodes earlier |
| Policy calls, fixed_5 / AAC | ≤ 128,000 / between 64,000 and 320,000 per seed | Derived (h* ∈ [2,10]) |
| Samples per call | 20 | PS |
| Checkpoint download | 12.44 GB | GCS listing |
| GPU memory | UNKNOWN. ESTIMATE: > 8 GB for π0.5 batch-1 inference (OpenPI guidance); a batch of 20 adds activations and KV cache, likely ≤ 24 GB | ESTIMATE |
| Inference latency (batch 20) | UNKNOWN. Paper: GR00T 106 ms at N=20 on A800 (Tab. 4); π0.5 not reported | — |
| Runtime | UNKNOWN. ESTIMATE, assuming ~0.15 s per 20-sample call and ~20 ms per sim step: fixed_10 ≤ ~6 h per seed per method (64k × 0.15 s + 640k × 0.02 s) on one GPU/CPU pair. Full ambiguity-covering Table 2 ≈ 4–5 methods × 3 seeds × ~6–15 h ≈ **70–225 GPU-hours** | ESTIMATE; reasoning shown |

---

## 15. Final Reproducibility Classification

**π0.5 + AAC on the CVPR 2026 LIBERO experiment: PARTIAL.**

Justification:
- **Not COMPLETE.** The multi-sample π0.5 server is private and required, and the client does not import on the pinned stock LIBERO.
- **Not RECONSTRUCTIBLE.** The server's interface is reconstructible, but the *experiment* depends on configuration that no public source pins down, and one detail (α) directly contradicts the paper. The unpinned items are: the π0.5 baseline horizon, α (paper 3 vs every public π0.5 LIBERO command 4), the server RNG protocol, and the seed/averaging scheme. Matching Table 2 would require choosing among these alternatives, i.e. assumptions.
- **Not NOT PUBLICLY REPRODUCIBLE.** The checkpoint, benchmark, tasks, init states, client and AAC selector are public and blob- or line-exact. The AAC math matches the paper apart from minor, documented deviations.

---

## 16. Recommendation

**NO-GO** for Phase 8B as an *exact* reproduction of Table 2: critical information is missing (a private server, an α conflict, and an unspecified baseline horizon).

Ways to turn this into CONDITIONAL GO:
1. **Ask the authors** (GitHub issue on `Adaptive-Action-Chunking/libero`) for: the π0.5 server code or diff; the α used for Table 2; the π0.5 baseline h; the seeds and averaging; and the OpenPI commit. This is the cleanest route.
2. **Or accept a relabelled Phase 8B.** Run it as an assumption-declared re-implementation with enumerated variants A1–A4 (§13), and report every variant, never just the best-matching one. This avoids the COAST failure mode of silently fitting protocol choices to the paper's numbers.

COAST-like reproduction risk: **HIGH** for exact numbers. There is no evidence of post-release semantic drift, but the code is a post-publication snapshot, a required component is private, and one paper-stated hyperparameter conflicts with the public scripts.
