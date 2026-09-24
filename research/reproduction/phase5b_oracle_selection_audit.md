# Phase 5B, Step 0: Oracle Selection Audit (KS3)

Question: **can the released implementation reproduce the paper's *hyper-parameter selection procedure* (15 fit rollouts → oracle selection over the grid → 30 test rollouts), and what would it cost to run?**

**Status: AUDIT AND PLAN ONLY.** No sweep, server, rollout, activation collection or code change was run. The only actions were reading source files, earlier phase records and `nvidia-smi`. This report does **not** conclude that COAST fails; Phase 5A Step 1 tested one fixed configuration and did not test the selection procedure.

| | |
|---|---|
| Paper | *Contrastive Conceptor Activation Steering (COAST)*, arXiv:2605.17144 v1 |
| Branch / HEAD | `exp/oracle-selection-reproduction` @ `db85bedd6f3e65e3ca4a883091ce0b1ad87b161e`, clean at start |
| Upstream base | `2afa10ee256a3b3edfeb56500fea166a0837f119` (`experiments/libero/*` unchanged from it) |
| Files audited | `experiments/libero/find_best_configs.py`, `experiments/libero/compute_conceptors.py`, `experiments/libero/README.md`, `experiments/libero/run_end_to_end.sh`, `experiments/libero/best_configs.json`, `examples/libero_env/main.py`, `scripts/serve_policy.py`, `src/openpi/serving/steering.py`, `src/openpi/serving/conceptors.py` (builder defaults only) |
| Hardware | 2 × RTX 4090 (24 GB), both free at audit time (447 MiB / 254 MiB used) |
| Prior records used | Phase 4B protocol audit (paper grid, selection text), Phase 5A Step 0 plan (decisions D1–D7), Phase 5A Step 1 run (timing and NPZ) |

## 0. Short answers

| # | Question | Answer |
|---|---|---|
| 1 | Does `find_best_configs.py` reproduce the paper's oracle selection? | **Partly. It uses the same method, but its defaults are not the paper's.** It runs steered rollouts per configuration, parses the success rate, and takes the argmax. As shipped, however, it uses a different grid, mixes strategies (including the non-paper `random_matched` control), picks **one** winner across strategies, uses a default state window (`--seed 7`) that overlaps both our fit and test states, and writes into a tracked file. With CLI overrides and one strategy per invocation it reproduces the procedure for **global** and **positive-only**, and for **per-step only at α = 1.0**. |
| 2 | Separate selection per strategy? | **No.** One `max()` over all steered conditions of a task (`find_best_configs.py:314–318`). The paper reports one oracle per strategy (Table 4). Workaround without code change: one invocation per strategy, or recompute per-strategy argmax from `partial_results.jsonl`. |
| 3 | Does it accidentally compare strategies together? | **Yes, with the default `--strategies`.** The argmax runs over `global`, `per_step`, `positive_only`, `random_matched` and `linear` together, so a random control or linear steering could be written to `best_configs.json` as "the" COAST config. |
| 4 | Does it use fit states only? | **No. It has no notion of fit or test.** It evaluates on whatever window `--seed` selects (episode k → init state `(seed + k) % 50`). Default `--seed 7 --num_episodes 10` = states **7–16**: 8 fit states (7–14) and 2 of our test states (15–16). The README's `--seed 15` = states **15–29**, a separate window that **overlaps our Phase 5A test set (15–44)**. Fit-state selection needs `--seed 0 --num_episodes 15` explicitly. |
| 5 | Does it save all configuration results? | **Rates, yes; episodes, no.** `partial_results.jsonl` (one line per condition, streamed), `per_task_results.json`, `args.json`. It stores success **rates only** (parsed from a `%.2f` log line); subprocess stdout, including per-episode lines, is discarded unless the run fails. Per-episode outcomes can be recovered from the saved `episode_NNN.mp4` frame counts (the method our Phase 5A analyzer already uses). Tie sets are not recorded. `best_configs.json` keeps only the winner. |
| 6 | Can it express the paper's layer × α × β × strategy grid? | **For global and positive-only, fully (60 + 60). For per-step, only 12 of 60 (α fixed at 1.0 in the builder and the hook cache).** Linear: 20 distinct configurations, because β has no effect on `h' = h + α·v` (the paper's nominal 60 collapse to 20). `random_matched` must be excluded. Nominal paper grid 240 → **152 distinct configurations expressible** without code changes. |

## 1. Current script capability

### 1.1 What `find_best_configs.py` does (read line by line)

1. Loads `pi05_libero` once and wraps it in `SteeredPolicyWrapper(conceptor_npz_path=conceptors/libero_conceptors.npz)`, with the NPZ path hard-coded at `:38`. It serves the wrapped policy in a background thread on `--port` (`:205–217`). No noise control and no diagnostics, so this is the released, unseeded path.
2. Expands a **strategy-gated** grid (`:228–235`): `per_step` gets `alpha_axis = (nan,)` and the wire value α = 1.0; `linear` gets `beta_axis = (nan,)` and the wire value β = 0.0. Other strategies get the full `layers × alphas × betas`.
3. Per task: runs one **unsteered baseline** and then every condition, each as a subprocess `examples/libero_env/.venv/bin/python main.py --task_id … --num_episodes N --seed S --steer --steering_layer … --steering_task <task>` (`:126–189`). Timeout 7,200 s. A non-zero exit or a missing `success_rate=` line returns `nan`.
4. Parses the **last** `success_rate=X.YY` from the log. `main.py:572` prints `%.2f`. At N = 15 all 16 values k/15 are distinct at 2 decimals, so ranking is lossless.
5. After each condition it appends to `partial_results.jsonl`. After each task it rewrites `per_task_results.json`.
6. Selection (`:311–332`): per task, `max(steered.items(), key=value)` over **all steered conditions of all strategies** (baseline excluded, NaN dropped). Python's `max` keeps the **first** maximal item in insertion order, which is strategy (CLI order) → layer ↑ → α ↑ → β ↑. The winner is written with `baseline_sr` and `steered_sr` to `--best_configs_path`.

### 1.2 Defaults vs what the paper needs

| Arg | Default | Paper need | Overridable by CLI? |
|---|---|---|---|
| `--layers` | `(11,)` | `0 5 11 17` | yes |
| `--alphas` | `(0.1, 0.5, 1.0)` | `0.1 0.5 1.0 2.0 10.0` | yes (keys use `str(float)`, so `10.0` matches the NPZ key `…__10.0__…`) |
| `--betas` | `(0.1, 0.3)` | `0.1 0.3 0.5` | yes |
| `--strategies` | global, per_step, positive_only, **random_matched**, linear | one paper strategy per invocation | yes |
| `--num_episodes` | 10 | 15 (fit set size) | yes |
| `--seed` | **7** (states 7–16) | 0 (fit states 0–14) | yes |
| `--tasks` | all 10 | KS3 only | yes |
| `--output_dir` | `experiments/libero/steering_results` (gitignored) | outside `experiments/` | yes |
| `--best_configs_path` | **`experiments/libero/best_configs.json` (tracked file)** | outside `experiments/` | yes; **must be overridden**, or the run modifies a tracked `experiments/` file |
| `CONCEPTOR_NPZ` | `conceptors/libero_conceptors.npz` | the V0 NPZ from Phase 5A Step 1 | **not a flag**; the file at that path is used |

### 1.3 Supporting components

| Component | Relevant behavior | Verdict |
|---|---|---|
| `compute_conceptors.py` / `conceptors.py` | Defaults are layers (0,5,11,17), α (0.1,0.5,1,2,10), per-step 0–9, which is the paper grid. Per-step conceptors are built at `per_step_alpha = 1.0` only (`conceptors.py:380`). Also emits `L{layer}__linear_direction` | global / positive-only / linear: complete. **Per-step: α axis missing** |
| Existing NPZ | `conceptors/libero_conceptors.npz`, KS3 only, 184 keys, sha256 `1ac8fceb…709c`, built in Phase 5A Step 1 with the unmodified builder from fit states 0–14 (8 S / 7 F). Label **V0 (released implementation)** | Reusable as is; no rebuild needed |
| `steering.py` | `global` → `C_contrastive`, `positive_only` → `C_success`, `per_step` → 10 matrices at α 1.0 (α zeroed in cache key), `linear` → `h + α·v` (β zeroed), `random_matched`, plus our research-only `shrinkage` | Paper strategies supported, except the per-step α axis |
| `main.py` | `--seed` sets the init-state offset `(seed + k) % 50`, the env physics seed and `np.random`. Episode ends at the first `done`; timeout 520 steps (libero_10). Saves `episode_NNN.mp4`. Logs per-episode success to stdout | Correct for fit-state or test-state runs |
| `serve_policy.py` | `--pytorch --steer --conceptor-npz`. Our research flags (`--noise-control`, `--steering-diagnostics`) are off by default | Used only for the test stage; the sweep script runs its own server |
| `run_end_to_end.sh` / README | Documented protocol: collect seed 0, **sweep seed 15**, eval seed 30, 15 episodes each, repository default grid | **Not the paper protocol** (separate sweep window, 15-episode test, small grid, strategies mixed) |

### 1.4 Other quirks

| # | Quirk | Consequence | Handling (no code change) |
|---|---|---|---|
| Q1 | A baseline is run on every invocation and cannot be switched off | +15 rollouts per invocation | Accept. Each gives an independent fit-state baseline estimate |
| Q2 | NaN (crash) is silently dropped from argmax | A crashed config is never a candidate | Analysis checks for NaN or missing entries; rerun crashes only, never results |
| Q3 | Tie rule is implicit (first in insertion order) | With 15 episodes, many ties at the max are likely; the rule favors low layer, α and β | Pre-declare the rule; record the full tie set (§5) |
| Q4 | Subprocess logs discarded | No per-episode record in the results files | Recover from mp4 frame count (timeout = 530 frames incl. 10 settle steps), as in `analyze_phase5a_ks3.py` |
| Q5 | `LIBERO_CONFIG_PATH` inherited from the parent env | Must be exported before launch | Export, as in Phase 5A |
| Q6 | Serial: one client at a time per sweep process | Wall time scales with configs | Shard across 2 GPUs (separate processes, ports and output dirs) |

## 2. Mismatch with the paper

| Component | Paper | Repository (`find_best_configs.py` as shipped) | Required change |
|---|---|---|---|
| Layers | {0, 5, 11, 17} | default (11,) | CLI: `--layers 0 5 11 17` |
| α | {0.1, 0.5, 1, 2, 10} | default (0.1, 0.5, 1.0) | CLI: `--alphas 0.1 0.5 1.0 2.0 10.0` |
| β | {0.1, 0.3, 0.5} | default (0.1, 0.3) | CLI: `--betas 0.1 0.3 0.5` |
| Strategy: global | full grid, 60 | supported, 60 | CLI: `--strategies global` (own invocation) |
| Strategy: positive-only | full grid, 60 | supported, 60 | CLI: `--strategies positive_only` (own invocation) |
| Strategy: per-step | full grid, 60 (KS3 oracle uses **α = 10**) | **α fixed at 1.0**: 12 configs (4 layers × 3 β) | **Code change** needed for the α axis (§3, C1). Without it, the per-step column is only partly reproducible |
| Strategy: linear | in grid (nominal 60) | supported; β has no effect, so 20 distinct configs | CLI: `--strategies linear`. Equivalent to the paper's grid, provided the paper's linear is also `h + α·v` (not verified in the text; [I]) |
| Non-paper strategy | none | `random_matched` in the default list | CLI: exclude it |
| Grid size | 240 nominal per task | 152 distinct expressible (60 + 60 + 12 + 20) | per-step α change → 200 distinct (60 + 60 + 60 + 20) |
| Selection target | "maximize success rate on the **fitting rollouts**" (A.8.5), per task | success rate on the window chosen by `--seed` | CLI: `--seed 0 --num_episodes 15` (fit states 0–14). This means *steered re-rollouts on the fit initial states*: the 15 original unsteered rollouts cannot score a configuration |
| Selection scope | **one oracle per strategy** (Table 4 columns) | **one argmax across all strategies** | One invocation per strategy (no code change), or per-strategy argmax in an analysis tool |
| Tie rule | not stated | implicit first-in-order | Pre-declared rule plus tie-set reporting (analysis tool) |
| Results saved | not stated | all rates saved; no per-episode data; winner only in JSON | Recover episodes from mp4 (analysis tool) |
| Output location | n/a | writes tracked `experiments/libero/best_configs.json` | CLI: `--best_configs_path` and `--output_dir` under `examples/libero_env/output/phase5b_*` (gitignored) |
| Fit episodes | 15 per task | `--num_episodes` default 10 | CLI: `--num_episodes 15` |
| Conceptor fit data | 15 unsteered rollouts | whatever built the NPZ | Reuse the Phase 5A NPZ (V0, states 0–14, 8/7) |
| Conceptor math | paper text: pooled, centered, canonical AND (V3) | released builder V0 | None for this reproduction (label as V0). V3 arm is a separate decision (Phase 5A D1) |
| Policy noise | not described | unseeded | none (matched by omission) |
| Evaluation | selected config, **30 test rollouts** per strategy | not part of the script (README: `eval_all.py`, 15 episodes, seed 30) | `serve_policy.py` + `main.py --num_episodes 30 --seed 15` per selected config (as in Phase 5A Step 1) |
| Test states | not stated ("fresh seeds, disjoint from fit") | README: 30–44 | 15–44 (Phase 5A interpretation, keeps continuity with the 19/30 run) |
| Unsteered "Base" | appears to equal the fit-set rate (Phase 4B, inferred) | baseline re-run on the sweep window | Report fit-state baselines (from the sweep) and a 30-episode test baseline separately |
| Tasks | all 10 LIBERO-10 | all 10 by default | CLI: KS3 only (`--tasks KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it`) |

### Important consequence of the seed choice

The README's sweep window (`--seed 15`, states 15–29) is a legitimate *repository* protocol but conflicts with our Phase 5A test set (15–44). Selecting on 15–29 and then testing on 15–44 would put half the test states into the selection. **Use `--seed 0` (fit states), the literal paper reading.** The cost is that steered selection runs are in-sample for the conceptor, which the paper's text implies anyway.

## 3. Required code changes

| ID | Change | Needed for | Files | Status |
|---|---|---|---|---|
| none | Stages 1–3 below (global, positive-only, per-step at α = 1.0, optional linear, 30-episode test) run with **CLI flags only** | the recommended plan | — | **no change to `src/`, `scripts/`, `examples/`, `experiments/`** |
| T1 | New **analysis tool** (research only): per-strategy argmax with the pre-declared tie rule, full tie sets, NaN and missing check, per-episode outcomes from mp4 frame counts, Wilson intervals | Stages 1–3 | `research/reproduction/tools/analyze_phase5b_oracle.py` (new) | to write in the next step, before any sweep result is read |
| C1 | **Per-step α axis**: builder emits per-step keys per α (e.g. `L{L}__per_step_{t}__{α}__C_*`); `get_per_step_conceptor_matrices` takes α; `_cache_key` stops zeroing α for `per_step`; `find_best_configs.py` stops gating the per-step α axis; rebuild NPZ (+≈ 2 GB) | Paper's per-step column (KS3 oracle per-step = L5, α 10, β 0.3) | `src/openpi/serving/conceptors.py`, `src/openpi/serving/steering.py`, `experiments/libero/find_best_configs.py` | **conditional, needs explicit approval**: it changes conceptor construction and adds 48 configs |
| C2 | Per-strategy selection and per-episode logging inside `find_best_configs.py` | convenience only | `experiments/libero/find_best_configs.py` | **not recommended**: T1 does the same without touching released code |

Note on C1: the per-step conceptors come from about 385–728 samples per step after token flattening, for d = 1024 (Phase 4B). α changes their spectrum strongly, so the α axis is not cosmetic.

## 4. Estimated compute cost

### 4.1 Timing basis (measured on this machine)

- Phase 5A Step 1, 1 × RTX 4090, 30 episodes per condition: baseline 356 s, COAST 348 s, both 19/30 → **≈ 11.7 s per episode** including client start-up.
- Episode time scales with env steps: a success takes ≈ 200–330 steps and a failure 520 (+10 settle). Planning values: **11.5 s** typical, **≈ 8.7 s** if all 15 succeed, **≈ 17.8 s** if all 15 time out. Plus ≈ 15 s client start-up per configuration.
- Per 15-episode configuration: **≈ 3.1 min typical (2.4–4.7 min)**. Large-β and layer-0/17 configurations that break the policy will sit near the upper bound.
- Two concurrent processes (one per GPU) ran at 11.4–12.8 s per episode in Phase 4A, so add a **15% contingency** when sharding.

### 4.2 Configurations per strategy (KS3, one task)

| Strategy | Paper nominal | Distinct, no code change | Distinct with C1 |
|---|---|---|---|
| global | 60 (4 × 5 × 3) | **60** | 60 |
| positive-only | 60 | **60** | 60 |
| per-step | 60 | **12** (4 layers × 3 β; α = 1.0) | 60 |
| linear | 60 nominal (β inert) | **20** (4 layers × 5 α) | 20 |
| **Total** | **240** | **152** | **200** |

### 4.3 Rollouts and GPU hours (15 fit episodes per configuration, 30 test episodes per condition)

| Item | Runs (incl. 1 baseline per invocation) | Rollouts | GPU-h typical (range) |
|---|---|---|---|
| Stage 1: global | 60 + 1 (or +2 if split across 2 GPUs) | 915–930 | **3.2** (2.5–4.8) |
| Stage 2a: positive-only | 60 + 1 | 915 | **3.2** (2.5–4.8) |
| Stage 2b: per-step (α = 1.0) | 12 + 1 | 195 | **0.7** (0.5–1.0) |
| Stage 2c: linear (optional) | 20 + 1 | 315 | **1.1** (0.8–1.6) |
| Stage 3: test (baseline + 3 selected) | 4 × 30 eps | 120 | **0.4** (0.3–0.6) |
| Stage 3 option: + selected linear, + paper-stated positive-only (L11, α 1, β 0.1) | 2 × 30 eps | 60 | 0.2 |
| **Required total (Stages 1, 2a, 2b, 3)** | 139 | **2,145** | **≈ 7.5 GPU-h (5.8–10.9)** |
| With linear and both test options | 162 | **2,520** | ≈ 8.8 GPU-h (6.8–12.8) |
| If C1 approved: per-step 60 instead of 12 | +48 | +720 | +2.5 GPU-h |
| For comparison: nominal 240 run literally (duplicates included) | 240 | 3,600 | ≈ 12.5 GPU-h |
| For comparison: all 10 LIBERO-10 tasks (not proposed) | ≈ 10× | ≈ 21,000 | ≈ 75 GPU-h |

### 4.4 Wall clock on 2 × RTX 4090

| Stage | GPU 0 | GPU 1 | Wall (+15%) |
|---|---|---|---|
| 1 (global) | layers 0, 5 (30 + baseline) | layers 11, 17 (30 + baseline) | **≈ 1.9 h** |
| 2 | positive-only (60 + baseline) | per-step (12 + baseline), then linear (20 + baseline, optional) | **≈ 3.7 h** |
| 3 | one server, conditions sequential | — | **≈ 0.5 h** |
| **Total** | | | **≈ 6 h wall** (≈ 11–12 h on one GPU) |

Storage: ≈ 15 MB of video per 15-episode run → ≈ 2.5 GB for all stages, gitignored. The NPZ already exists (755 MB). GPU memory is ≈ 8.4 GB per process, so one process per 24 GB card is fine. **The sweep is affordable; staging is recommended for decision gates, not because of cost.**

## 5. Recommended sweep plan

### Stage 0: preparation (no rollouts)

1. **Preregistration** file (gitignored, timestamped before the first rollout) that fixes: states (sweep 0–14 via `--seed 0 --num_episodes 15`; test 15–44 via `--seed 15 --num_episodes 30`), grid per strategy, NPZ sha256 `1ac8fceb…709c` (V0), noise unseeded, the tie rule, the reading rule, and the stage gates below.
2. **Tie rule (pre-declared):** within a strategy, highest fit success wins. Among ties, take the repository's first-in-order rule (layer ↑, α ↑, β ↑), i.e. what `find_best_configs.py` itself would pick for a single-strategy invocation. The **full tie set** is reported. No re-running of tied configurations to break ties (that would be extra selection on the fit states), and no use of test data to break ties.
3. **Pre-flight (CPU):** NPZ sha256 and the 184 keys; every (layer, α) key for global/positive-only and per-step 0–9 for all 4 layers present; `uv run pytest tests/test_steering.py tests/test_conceptors.py` passes; `find_best_configs.py --help` reflects the flags.
4. Write `research/reproduction/tools/analyze_phase5b_oracle.py` (T1) and test it on the Phase 5A outputs.

### Stage 1: KS3, global only (the Phase 5A comparison target)

```bash
export LIBERO_CONFIG_PATH=/home/stanley/Stanley_ws/COAST-research/libero_config_coast
TASK=KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it
# GPU 0: layers 0 5  |  GPU 1: layers 11 17 (same command, --layers 11 17, --port 8402, suffix _L11_17)
CUDA_VISIBLE_DEVICES=0 uv run --no-sync python experiments/libero/find_best_configs.py \
  --tasks $TASK --strategies global --layers 0 5 \
  --alphas 0.1 0.5 1.0 2.0 10.0 --betas 0.1 0.3 0.5 \
  --num_episodes 15 --seed 0 --port 8401 \
  --output_dir examples/libero_env/output/phase5b_sweep_global_L0_5 \
  --best_configs_path examples/libero_env/output/phase5b_sweep_global_L0_5/best_configs.json
```

Output: 60 global fit-state success rates, 2 fit-state baselines, the selected global configuration and its tie set. **Gate 1** (a record, not a stop criterion): does the selection include the paper's L5/α 0.5/β 0.1 in its tie set? How large is the tie set? Did any run crash? Proceed to Stage 2 whatever the result; the selection is frozen in a file before Stage 3.

### Stage 2: other strategies, each in its own invocation (never mixed)

- 2a `--strategies positive_only` (60), GPU 0.
- 2b `--strategies per_step` (12, α = 1.0; `--alphas` is ignored for it), GPU 1.
- 2c `--strategies linear` (20), GPU 1, optional (a paper baseline method, not a COAST strategy column in Table 4).
- Per-strategy selection by the T1 tool with the same tie rule. **All selections are written to a frozen file (with sha256) before any test rollout.**
- Per-step with the paper's α axis only if C1 is approved; it would then run as a separate Stage 2b′ after rebuilding the NPZ. The α = 1.0 sweep is still reported as the released-code result.

### Stage 3: test the selected configurations (30 episodes, states 15–44)

One `serve_policy.py --pytorch --steer --conceptor-npz conceptors/libero_conceptors.npz` server; `main.py --task_id 2 --num_episodes 30 --seed 15`, run sequentially:
1. unsteered baseline
2. selected global
3. selected positive-only
4. selected per-step (α = 1.0)
5. optional: selected linear; paper-stated positive-only (L11, α 1.0, β 0.1). The paper-stated global already has a test run (Phase 5A: 19/30).

If a selected configuration equals one already tested (e.g. global L5/α 0.5/β 0.1), it is **still re-run** in Stage 3, so every Stage 3 number comes from the same session and protocol. The Phase 5A number is kept as context only.

**Reading rule (to be copied into the preregistration, not changed after results):** for each strategy against the paper's Table 4 test rate (global 28/30, per-step 26/30, positive-only 24/30), use the Phase 5A bands scaled to that target: consistent if within the paper's Wilson 95% interval, inconsistent if Fisher exact p < 0.05, otherwise inconclusive. Also report the **fit-to-test drop** of each oracle. The selected fit score is expected to be optimistic: with 60 noisy 15-episode estimates (SD ≈ 0.13 at p ≈ 0.6), the maximum alone can sit ≈ 0.25–0.3 above the true rate. That is what the separate test set is for. No "COAST fails" statement; any gap is reported together with the open protocol items in §6.

### Not recommended now

- All 10 tasks (≈ 75 GPU-h). KS3 first, as instructed.
- Seeded or paired noise (a deviation from the released path; Phase 5A D5).
- V3 (paper-math) or authors'-activation arms (each ≈ +7 GPU-h); decide after Stage 3.
- Shrinkage or random controls (out of scope for this phase).

## 6. Is a faithful oracle reproduction possible?

**Yes for the procedure; not verifiably identical to the authors' run.**

| Aspect | Faithful? |
|---|---|
| Selection method (grid → steered rollouts → argmax success) | **yes**, with CLI overrides |
| Grid for global and positive-only (4 × 5 × 3) | **yes, exact** |
| Grid for linear | **equivalent** (β inert; 20 distinct = paper's nominal 60), assuming the same `h + α·v` form [I] |
| Grid for per-step | **partial without C1** (α = 1.0 only; the paper's KS3 per-step oracle α = 10 is not expressible). **Yes with C1** |
| Per-strategy oracle | **yes**, via separate invocations and the T1 analysis (no code change) |
| Selection on fit rollouts | **yes under the literal reading** (steered re-rollouts on fit states 0–14). The paper does not say whether it re-ran the fit states or which noise was used |
| 30-episode test of the selected config | **yes** (states 15–44, our interpretation) |
| Things that cannot be matched from public information | paper's fit and test state IDs and seeds; code version (pre/post PR #48); builder V0 vs V3 behind the paper numbers; tie rule; policy noise; single vs repeated runs; checkpoint weight identity (all from Phase 4B §8–9) |

So the reproduction can be *faithful to the stated procedure* for global and positive-only on KS3, *partially faithful* for per-step unless C1 is approved, and it stays *not provably identical* to the authors' runs. A result in either direction should be reported with that qualification.

## 7. Decisions needed before Stage 1

| # | Decision | Recommended default |
|---|---|---|
| E1 | Sweep states | fit states 0–14 (`--seed 0`), literal paper reading. Not the README's `--seed 15`, which overlaps our test set |
| E2 | Test states | 15–44 (`--seed 15 --num_episodes 30`), continuity with Phase 5A |
| E3 | Per-step α (C1) | **not now**: run α = 1.0 (released code), decide on C1 after Stage 2 |
| E4 | Linear (Stage 2c) | include (+1.1 GPU-h); it is in the paper's grid |
| E5 | Tie rule | first-in-order (layer ↑, α ↑, β ↑) + report tie sets |
| E6 | GPUs | both RTX 4090s, one sweep process each |

## 8. Files

- Added: `research/reproduction/phase5b_oracle_selection_audit.md` (this report), `research/reproduction/experiments/phase5b_oracle_selection_audit.yaml`.
- No source, script, example, experiment or tracked config file was modified. Nothing was run on a GPU. Nothing committed or pushed.
