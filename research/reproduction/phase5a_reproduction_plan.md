# Phase 5A, Step 0 — Faithful Reproduction Plan (KS3)

**Status: PLAN ONLY. No rollout, server, sweep, activation collection or code change has been run for this phase.** Everything below is read from the paper, the repository, the Phase 4B audit and our records. Nothing is launched until this plan is reviewed.

| | |
|---|---|
| Paper | *Contrastive Conceptor Activation Steering (COAST)*, arXiv:2605.17144 v1 |
| Question | Can the **released COAST implementation** reproduce the paper's reported LIBERO result for KS3 under the paper's documented evaluation protocol? |
| Branch / HEAD | `exp/faithful-coast-reproduction` @ `64bf3a0ef04abca1a13e43850c42f61dedc1496d` (clean at start) |
| Only command executed for this plan | `find_best_configs.py --help` (prints options; loads no model, runs no rollout) |
| Scientific rule | The result will be compared with the paper **only under the protocol defined here**. Phases 1C–4A numbers are not the comparison target. |

## 0. Decisions needed from the reviewer (read this first)

Each item has a recommended default, and the plan below assumes the defaults. Every one of them is a place where the paper is silent or the repository cannot follow the paper.

| # | Decision | Recommended default | Alternative and consequence |
|---|---|---|---|
| D1 | Which mathematics is tested (must not be mixed) | **Released implementation** (`V0`: token-flattened, uncentered, repo Boolean AND), built by the unmodified official builder | Paper mathematics (`V3`) needs a builder that is not in the repository and a per-step construction that is ill-posed with 15 episodes (N < d). Separate arm, only after D1-default is done |
| D2 | Per-step strategy | **Run per-step at the only α the repository can express (α = 1.0, 12 configurations)** and mark the paper's per-step column *not reproducible as specified* (paper's KS3 per-step config uses α = 10) | Implement a per-step α sweep: a code change to the builder and the hook-cache key (`src/`), **+48 configurations, +2.3 GPU-h**. Needs explicit approval because it touches conceptor construction |
| D3 | Which states the oracle sweep runs on | **The 15 fit states 0–14 (`--seed 0`)**, the literal reading of the paper (A.8.5) | Repository docs sweep on a separate window (`--seed 15`). Same cost. Changes what "selected on fit rollouts" means |
| D4 | Test states | **States 15–44 (`--seed 15 --num_episodes 30`)** | See §2.3 for why no fully fresh 30-state set exists; alternative 20–49 |
| D5 | Policy sampling noise | **Unseeded (default released path)**, as the paper describes no noise control | Our paired-noise instrumentation (Phase 1D) would lower pair variance but is not part of the released protocol, so it is a deviation |
| D6 | Optional extra arms | Include **paper-stated-config test** (§4.4, +90 rollouts, ≈ 20 min). Defer authors'-activation arm and replicates | See §5 for costs |
| D7 | GPUs | **Two GPUs** (both free now), one shard each | One GPU roughly doubles wall time |

## 1. Which exact paper result are we reproducing?

**Paper Table 4 and Table 1, KITCHEN_SCENE3 (KS3) = libero_10 task 2, `KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it`, π0.5 (`pi05_libero`), checkpoint `openpi-libero-2000`.**

| Strategy | Paper's selected config (oracle, Table 4) | Paper's test SR (30 held-out rollouts) |
|---|---|---|
| **Global** (primary target) | layer **5**, α **0.5**, β **0.1** (also Table 17) | **0.93** = 28/30 |
| Per-step | layer 5, α **10.0**, β 0.3 | 0.87 (26/30) |
| Positive-only | layer 11, α 1.0, β 0.1 | 0.80 (24/30) |
| Unsteered "Base" (Table 1) | n/a | 0.53, which equals the fit-set rate 8/15 (Table 21); see Phase 4B finding F1 |

The reproduction target is the **procedure**: fit on 15 rollouts → search the paper's grid on the fit rollouts → evaluate the selected configurations on 30 held-out rollouts. Success is not defined as "recover L5/α0.5/β0.1"; that would be tuning toward the answer. We report what the search selects and separately what the paper's stated configs score (D6).

**Pre-declared reading rule** (written before any result exists). Paper's 28/30 has Wilson 95% [0.79, 0.98].
- *Consistent with the paper*: selected-global test ≥ 24/30.
- *Inconsistent*: ≤ 20/30 (Fisher exact vs 28/30, p ≈ 0.02 or smaller).
- *Inconclusive*: 21–23/30.
- Any conclusion is qualified by the unresolved protocol items in §6. No "COAST fails" statement is made unless every item in §3 marked *matched* is verified and the unresolved items are listed.

## 2. What was the paper evaluation protocol?

### 2.1 Stated in the paper

| Item | Paper statement | Source |
|---|---|---|
| Fit episodes | **15 rollouts per task**, unsteered, used for conceptor construction **and** hyper-parameter selection | §4.1, A.8, A.8.3 |
| Fit data collection | `--num_episodes 15`; checkpoint `/openpi-libero-2000`; KS3 fit outcome 8 success / 7 failure | A.8.3, Table 21 |
| Test episodes | **30 per task per steering condition**, "fresh seeds disjoint from the fitting set", used exclusively for the final numbers | A.8, A.8.3, A.8.5 |
| Split | "different environment seeds, object placements, and initial conditions … No episode identifiers or seeds are shared" | A.8.5 |
| Task selection | all 10 LIBERO-10 tasks; tasks with < 3 successes or failures excluded from contrastive fitting | A.8.3 |
| Oracle selection | "selected by maximizing success rate on the fitting rollouts", per task, full grid; the selected configuration is then run on the test rollouts and only test rates are reported | A.8.5, A.10 |
| Grid | layer {0,5,11,17} × α {0.1,0.5,1,2,10} × β {0.1,0.3,0.5} × strategies {global, per-step, positive-only, linear} = 240 | Table 14 |
| Strategies reported per task | Global, Per-Step, Pos.-Only, each with its own oracle config | Table 4 |
| Success metric | LIBERO `info["is_success"]`; success if the final step's flag is True | A.8.3 |
| Policy noise / seeds | not described | none |

### 2.2 Not stated in the paper (unresolved, no guessing)

- Which LIBERO initial states and seeds define "fit" and "test".
- Whether the oracle sweep's steered rollouts ran on the fit states (literal reading) or another window.
- How ties among configurations were broken (with 15 episodes, ties at the maximum will be common).
- Which code version produced the numbers (before or after PR #48, 2026-04-23, when `--seed` began selecting initial states).
- Whether Table 1 "Base" is the fit set or a separate test baseline (Phase 4B: numerically equal to the fit set for 10/10 tasks).

### 2.3 Split we will use, with evidence and uncertainty

| Set | States | Evidence | Confidence |
|---|---|---|---|
| **Fit** (conceptors) | **0–14** (`--seed 0`, 15 episodes) | Repository convention (`experiments/libero/README.md`: collect `--seed 0`), our Phase 1A collection, and the released dataset's episode ids 0–14. The released dataset's metadata carries no seed or init-state field | **medium**: seed 0 is convention, not proof |
| **Sweep** (oracle) | **0–14** (same fit states, steered) | Paper A.8.5, literal reading | **low–medium**: repository docs use a separate window (D3) |
| **Test** | **15–44** (`--seed 15 --num_episodes 30`) | 15 + 30 = 45 of the 50 states; the natural window after the fit set under the repository's seeding rule | **low**: the paper never lists them |

Consequences to state up front:
1. **Test states 15–44 are not fresh to us.** They were used in Phases 1C–4A (including the paper's KS3 config, 20/30 in Phase 4A). Selection is done only on fit-state data, so there is no leakage into the *selection*, but the researcher-level novelty is limited. Only states 45–49 have never been used, which is too few for a 30-episode test. I am marking this limitation rather than hiding it.
2. If the paper's runs predate PR #48, its 30 test episodes were states 0–29 (overlapping the fit states). We cannot know; it is recorded as an open item and as an author question (Phase 4B §9).
3. Steering on the sweep states 0–14 means the conceptor is **in-sample**; this is intended by the paper's text and is expected to inflate fit scores relative to test scores.

## 3. What does the repository actually support?

Base commit `2afa10e` (= upstream `main`), plus our research-only additions (unused in this phase's default path).

### 3.1 Strategies for pi0.5 (`src/openpi/serving/steering.py`)

| Paper strategy | Repository | Verdict for this reproduction |
|---|---|---|
| Global (`C_success ∧ ¬C_failure`, all steps) | `global`: NPZ key `{task}__L{layer}__{α}__C_contrastive`; `M = (1−β)I + βC`, `h @ M.T`, every denoise step | **Supported, α ∈ {0.1,0.5,1,2,10}, any layer in {0,5,11,17}** |
| Positive-only (`C_success`) | `positive_only`: key `…__C_success` | **Supported**, same grid |
| Per-step (one conceptor per denoise step) | `per_step`: keys `L{layer}__per_step_{t}__C_contrastive`, t = 0…9; **α baked in at 1.0** (`conceptors.py` line 380 `per_step_alpha = 1.0`); the wrapper zeroes α in the cache key | **Supported only at α = 1.0.** The paper's KS3 per-step config (α = 10) **cannot be expressed** (Decision D2). The paper's α sweep for per-step (5 values) is not reproducible without a code change |
| Linear (ActAdd) | `linear` | Not requested in this phase (paper Table 4 omits it) |
| (not paper) `random_matched`, `shrinkage` | exist | **Excluded by instruction** (no ablations) |

### 3.2 CLI and scripts

| Tool | What it does | Notes for this plan |
|---|---|---|
| `experiments/libero/compute_conceptors.py` | Builds the NPZ from collected activations. Flags `--layers`, `--alphas`, `--per_step_indices`, `--task_filter`, `--min_episodes_per_class` (default 2, paper 3; KS3 has 8/7 so irrelevant) | Defaults already equal the paper's grid (layers 0/5/11/17, α 0.1/0.5/1/2/10, per-step 0–9). **No change needed** |
| `experiments/libero/find_best_configs.py` | Loads the policy in-process, serves it, calls `examples/libero_env/main.py` once per (task, condition), parses `success_rate=`, writes the argmax | Flags `--tasks --layers --alphas --betas --strategies --num_episodes --seed --port --output_dir --best_configs_path` (verified with `--help`). Defaults **differ from the paper** (layer 11 only, α {0.1,0.5,1.0}, β {0.1,0.3}, includes `random_matched`/`linear`), so every axis must be passed on the command line |
| `scripts/serve_policy.py` | `--pytorch --steer --conceptor-npz PATH` (for the test stage) | Our extra flags (`--noise-control`, `--steering-diagnostics`) stay off |
| `examples/libero_env/main.py` | Runs N episodes at `--seed`; episode k uses init state `(seed+k) % 50`; steering via `--steer --steering_layer/alpha/beta/strategy` | Supports 30 episodes directly; used for the test stage |
| `examples/libero_env/eval_all.py` | Runs all tasks with `--steering_config best_configs.json` | Not needed (single task) |
| `experiments/libero/best_configs.json` | **Empty placeholder** (`"tasks": {}`), although the README says tuned params are committed | The paper's per-task oracle table exists only in the paper (Table 17) |

### 3.3 Gaps and quirks that affect the reproduction

| # | Gap or quirk | Effect | Handling |
|---|---|---|---|
| G1 | Per-step α fixed at 1.0 | Paper's per-step column not reproducible | D2 |
| G2 | The script picks **one** argmax over all strategies (`max(steered.items())`), but the paper reports one oracle **per strategy** | A single run would only give one column | Shard by strategy and read each shard's `partial_results.jsonl`; per-strategy argmax computed by an analysis script, not by editing the released code |
| G3 | Ties: paper silent; repository `max()` returns the first in iteration order (strategy, then layer ascending, α ascending, β ascending) | With 15 episodes, many configs will tie at the maximum | **Pre-declared:** use the repository's rule (first in that order). The full tie set is recorded. No post-hoc tie-breaking |
| G4 | Sweep results are **rates rounded to 2 decimals** only; client logs are discarded | Per-episode outcomes are not stored | Recoverable from each run's mp4 frame count (timeout = 520 steps, as in our earlier analysers). No script edit |
| G5 | Failed subprocess returns `nan` silently | A crashed config would be skipped | Analysis flags any `nan`; missing configs are rerun (rerun for a crash, never for a result) |
| G6 | `CONCEPTOR_NPZ` is hard-coded to `conceptors/libero_conceptors.npz` (gitignored path); the authors' `brandonyang/libero-conceptors` returns HTTP 401 | The NPZ must be built locally and placed at that path | Build from our fit set with the unmodified builder (§4.1). Not a code change |
| G7 | Subprocess uses `examples/libero_env/.venv/bin/python` | Environment must already be synced (it is) | none |
| G8 | No 30-episode-test convenience flow | none | `main.py --num_episodes 30 --seed 15` |
| G9 | Paper-mathematics (V3) builder is not in the repository | Only our Phase 1B research tool has it | Only in an optional separate arm (D1) |
| G10 | Policy sampling noise unseeded in the released path | Runs are not repeatable | Faithful to the paper; recorded as a limitation (D5) |

## 4. What modifications are required?

### 4.1 Required for reproduction (no change to `src/`, `scripts/`, `examples/` or `experiments/`)

1. **Build the NPZ** with the released builder, unchanged, from the Phase 1A fit activations (states 0–14, 15 episodes, 8/7):
   ```
   CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu uv run --no-sync python experiments/libero/compute_conceptors.py \
     --activation_root activations/libero --output_path conceptors/libero_conceptors.npz \
     --task_filter KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it
   ```
   (defaults = 4 layers × 5 α × 3 matrices + linear + per-step 0–9; ≈ 0.75 GB.) **Label: RELEASED implementation (V0).**
2. **Pre-flight checks (CPU, no rollouts):** (a) L5 and L11 arrays equal the Phase 1B NPZ bit-for-bit (same builder, same data); (b) all 184 expected keys present; (c) `uv run pytest tests/test_steering.py tests/test_conceptors.py` passes.
3. **Sweep on the fit states**, two shards (no edits to the sweep script):
   ```
   export LIBERO_CONFIG_PATH=/home/stanley/Stanley_ws/COAST-research/libero_config_coast
   # Shard A, GPU 0: global (60 configs + baseline)
   CUDA_VISIBLE_DEVICES=0 uv run --no-sync python experiments/libero/find_best_configs.py \
     --tasks KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it \
     --layers 0 5 11 17 --alphas 0.1 0.5 1.0 2.0 10.0 --betas 0.1 0.3 0.5 --strategies global \
     --num_episodes 15 --seed 0 --port 8201 \
     --output_dir examples/libero_env/output/phase5a_sweep_A \
     --best_configs_path examples/libero_env/output/phase5a_sweep_A/best_configs.json
   # Shard B, GPU 1: positive_only (60) + per_step (12, α=1.0 only) + baseline
   CUDA_VISIBLE_DEVICES=1 ... --strategies positive_only per_step --port 8202 --output_dir …/phase5a_sweep_B
   ```
   Each shard also runs its own unsteered baseline on the fit states, which gives two independent fit-baseline estimates for free.
4. **Analysis script** (new, under `research/reproduction/tools/`, analysis only): per-strategy argmax with the pre-declared tie rule, full tie sets, NaN check, per-episode outcomes recovered from mp4 lengths.
5. **Test stage** (single server, `serve_policy.py --pytorch --steer --conceptor-npz conceptors/libero_conceptors.npz`), `main.py --seed 15 --num_episodes 30`, conditions: unsteered baseline, selected-global, selected-per-step, selected-positive-only (+ D6 paper-stated configs).
6. **Preregistration file** written before the first rollout (state sets, grid, tie rule, reading rule from §1, decisions D1–D7), as in earlier phases.

### 4.2 Not required (and deliberately not done)

- No change to steering mathematics, conceptor construction, hook code or wire protocol.
- No shrinkage, random-matched or any other ablation; no adaptive β; no token-wise steering; no training.
- No paired-noise instrumentation (D5); no diagnostics flags.
- No new activation collection (Phase 1A fit set is reused; see below).
- No use of `eval_all.py` or `best_configs.json`.
- No edit of `find_best_configs.py` (per-strategy argmax and per-episode logs are handled by analysis).

### 4.3 Conditionally required (only if approved)

| Item | Trigger | Change | Cost |
|---|---|---|---|
| Per-step α sweep | D2 alternative | builder must emit per-step keys per α; wrapper cache key must stop zeroing α; new key scheme | +48 configs (+2.3 GPU-h) and a `src/` diff that touches conceptor construction |
| Paper-mathematics arm (V3) | D1 alternative | NPZ writer for V3 matrices in repo key format (research tool); per-step V3 is ill-posed at N < d | second full sweep ≈ +6.8 GPU-h |
| Authors' released fit activations (`brandonyang/pi05-libero-activations-v1-2000-15env`, KS3 subset) | to separate "fit-data difference" | download ≈ 8–9 GB, rebuild NPZ, second sweep | ≈ +6.8 GPU-h |

### 4.4 Fit data: reuse, not recollect

The Phase 1A collection (`--seed 0`, 15 episodes, 8 S / 7 F, `all_suffix_residual (10,4,10,1024)`, 23/23 activation tests) already has the paper's fit-set structure and the same class counts as Table 21. Recollecting would only draw another unseeded-noise sample. **Reuse.** (The per-episode outcomes differ from the authors' released fit set, 7/15 agree, which is expected noise, not a defect.)

## 5. Compute estimate

Per-episode wall time from our own records: Phase 1D 9.7–10.9 s (single client), Phase 4A 11.4–12.8 s (including concurrent load and low-success conditions). **Planning value 11.5 s (range 10–13 s).** Per 15-episode configuration ≈ 2.6–3.2 min including client start-up.

### 5.1 Rollouts

| Stage | Configurations | Episodes each | Rollouts |
|---|---|---|---|
| Sweep, global (4 layers × 5 α × 3 β) | 60 + 1 baseline | 15 | 915 |
| Sweep, positive-only | 60 | 15 | 900 |
| Sweep, per-step at α = 1.0 (4 layers × 3 β) | 12 + 1 baseline | 15 | 195 |
| **Sweep subtotal** | **134 runs** | | **2,010** |
| Test: baseline + selected global + per-step + positive-only | 4 | 30 | 120 |
| Test, D6 option: paper-stated global / per-step / positive-only | 3 | 30 | 90 |
| **Total (default)** | | | **2,130** (2,220 with D6) |
| If D2 alternative (per-step α sweep): 60 per-step configs | +48 | 15 | +720 → 2,850 |

### 5.2 GPU time and wall clock

| Scenario | GPU-hours | Wall clock |
|---|---|---|
| Default (2,130 rollouts) | **≈ 6.8 h** (5.9–7.7) | **≈ 3.5–4.5 h on 2 GPUs** (shard A ≈ 2.9 h, shard B ≈ 3.5 h, + test ≈ 0.4 h, + 25% concurrency contingency), ≈ 7–8 h on one GPU |
| Default + D6 | ≈ 7.2 h | + 0.2 h |
| D2 alternative (2,850 rollouts) | ≈ 9.1 h | ≈ 4.6–5.5 h on 2 GPUs |
| Optional second arm (V3, or authors' activations) | + ≈ 6.8 h each | + ≈ 3.5–4.5 h each |

Low-success configurations (e.g. large β at layers 0/17) run to the 520-step timeout and cost more (~26 s vs ~11 s per episode), which is what the 10–13 s range and the contingency cover.

### 5.3 Storage

| Item | Size | Committed? |
|---|---|---|
| NPZ, 4 layers (184 keys × ≈ 4.1 MB) | ≈ 0.75 GB | no (gitignored) |
| Sweep run outputs (videos and logs, 13–17 MB per 15-episode run × 134) | ≈ 2.0–2.5 GB | no |
| Test outputs (4–7 conditions × ≈ 30 MB) | ≈ 0.1–0.2 GB | no |
| Fit activations | 0 new (8.6 GB already present) | no |
| If D2 alternative | + ≈ 2 GB NPZ (per-step keys × 5 α) | no |
| If authors' activations arm | + ≈ 8–9 GB | no |

Free disk is 3.3 TB, so storage is not a constraint. GPU memory per shard ≈ 8.4 GB (Phase 3A peak) on a 24 GB card; each shard gets its own GPU.

## 6. Protocol differences identified (paper vs what we will run)

| # | Paper | This reproduction | Status |
|---|---|---|---|
| P1 | 15 fit rollouts, seed unstated | states 0–14, `--seed 0` (Phase 1A) | matched in size; seed inferred |
| P2 | Sweep "on fitting rollouts" | steered runs on fit states 0–14 | literal reading; alternative in D3 |
| P3 | 30 test rollouts, states unstated | states 15–44 | **uncertain**; not fresh to us (§2.3) |
| P4 | Grid: 4 layers × 5 α × 3 β × {global, per-step, pos-only} | identical for global and positive-only; per-step only at α = 1.0 | **partial** (G1) |
| P5 | Oracle per strategy; tie rule unstated | per-strategy argmax; repository first-in-order tie rule, pre-declared | matched in intent; tie rule is ours |
| P6 | Conceptor math | released V0 (not paper V3) | **labelled**; paper math is a separate arm (D1) |
| P7 | Unsteered "Base" = fit-set rate (inferred) | fit baseline (15 eps, ×2 shards) and test baseline (30 eps) both reported | reported both; paper's definition unverified |
| P8 | Policy noise not described | unseeded | matched by omission |
| P9 | Success = final-step flag | first `done` (episode ends at first success) | equivalent in LIBERO |
| P10 | Hardware B200 (latency) | RTX 4090 | unmatched, low expected effect |
| P11 | Code version (pre/post PR #48) unknown | post-#48 (`(seed+k) % 50`) | **unresolved** |
| P12 | Hyper-parameter selection best-of-240 (with linear) | best-of-132 (three strategies, per-step reduced) | documented |

## 7. Risks and how they are handled

| Risk | Handling |
|---|---|
| Best-of-132 on 15 in-sample episodes → large ties and a noisy selection | Pre-declared tie rule; full tie sets reported; fit scores reported separately from test scores |
| Unseeded noise makes the sweep unrepeatable | Accepted (D5); the test stage is a single run per condition, like the paper. Replicates are optional (each test replicate ≈ 18–24 min) |
| Test states already seen in earlier phases | Selection uses only fit-state data; limitation stated |
| Sweep subprocess failure returns `nan` | Analysis checks; crashed configs may be rerun, results never |
| Someone reads a failed reproduction as "COAST fails" | Reading rule in §1; §6 items listed with every result |
| Per-step column cannot be matched | Reported as not reproducible as specified, not as a failure |

## 8. Stop point

This plan ends here. **No sweep, server or rollout is launched until the plan is reviewed and D1–D7 are answered** (or the defaults are approved). After approval the order is: preregistration file → NPZ build and pre-flight checks → two-shard sweep → per-strategy selection (frozen, written to a file before any test rollout) → test stage → `phase5a_faithful_reproduction.md` and `experiments/phase5a_faithful_reproduction.yaml`.

Files produced in this step: this plan only. Nothing is committed.
