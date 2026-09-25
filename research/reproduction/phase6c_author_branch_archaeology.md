# Phase 6C — Author-Branch Archaeology (π0.5, LIBERO-10, KS3)

Question: **can any public COAST branch, historical commit, script or committed artifact reconstruct the pipeline that produced the π0.5 LIBERO numbers in Tables 1 and 4?** Target cell: KS3 (`KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it`). Table 4 reports Global L5 α0.5 β0.1 = 0.93, Per-step L5 α10 β0.3 = 0.87 and Positive-only L11 α1 β0.1 = 0.80.

| | |
|---|---|
| Paper | *Contrastive Conceptor Activation Steering (COAST)*, arXiv:2605.17144 v1 (HTML text, fetched 2026-09-25) |
| Our branch | `exp/author-branch-archaeology` @ `bbb31c89cd2182e5d36ac2fd06fe117698c42193` (clean at start) |
| Upstream | `COAST-VLA/COAST`: 22 branches and 50 PR heads. PR heads were fetched read-only into the local namespace `refs/archaeology/pr/*`; no upstream ref was modified |
| Candidate executed | `upstream/miranda-v2` @ `29059a537ce123abdab7a4e164db0c72eeff8024`, in the isolated worktree `../COAST-paper-candidate` (detached, no commits) |

Evidence tags: **[P]** paper, **[R]** public git history, **[M]** measured by us, **[I]** our inference.

## 1. Executive summary

1. **Closest public pipeline.** The Table 4 / Table 15 LIBERO numbers were almost certainly produced by the **miranda-v2 `experiments/pi05_libero/src/` sweep pipeline**. It was introduced in `2fdc5ad` (2026-04-13, Muqing Miao). It was then summarised by `experiments/shared/oracle_gap_table.py` (`ec62cad`, 2026-04-21). The evidence is textual and arithmetic, not only a matching branch name:
   - **Caption.** The paper's Table 15 caption is a near-verbatim rendering of `oracle_gap_table.py`'s LaTeX f-string, with the benchmarks in the same order [P/R].
   - **Grid counts.** The paper's "up to 152 configurations per task" is exactly the size of the src sweep: 135 steered + 9 random + 8 positive-only [P/R]. The paper's "Selected Grid = 18" for π0.5 LIBERO is exactly what `oracle_gap_table.py` counts from the src condition names [P/R].
   - **The per-step "α = 10" label.** The paper's KS3 per-step value is the loop label of the src condition `per_step_{0|9}_L5_a10.0_b0.3`. In that code α is **not used** to build or pick the per-step conceptor [R].
   - **Positive-only grid.** All 10 positive-only cells of Table 4 lie in the src positive-only grid, L{5,11} × α{0.5,1} × β{0.1,0.3}. The Table 14 grid the paper describes would allow 60 cells [P/R].
   - **Denominators.** All 30 per-task cells of Table 4 are multiples of 1/15. If each cell were a 30-episode rate, the chance that all 30 numerators are even is **4.0 × 10⁻⁹** [P/M]. This matches the src pipeline's hard-coded `NUM_EPISODES=15`. As a control, the same test finds k/16 denominators in Table 8 (π0-FAST MetaWorld, a 16-environment pipeline).
2. **No public code matches the paper exactly.** The pipeline that most plausibly generated the numbers does **not** implement the paper's stated protocol:
   - It has no disjoint 30-episode test set. The pre-PR#48 client evaluates `initial_states[episode]`, so 15 episodes are states 0–14, which are the same states as the authors' fit rollouts.
   - Its "oracle" is the maximum success over about 152 conditions on those 15 rollouts.
   - Its per-step strategy is a fixed step-0 or step-9 conceptor at α = 1, with α as a label only.
   - Its public builder uses `C_s·(I−C_f)` and only denoising step 0.
   - Conversely, the paper's stated protocol exists in no public code.
3. **Targeted execution (KS3, candidate's native semantics, states 0–14, 15 episodes).** We ran the candidate code unmodified at `29059a5`, on the authors' own public fit activations. The paper's 28/30 = 0.93 corresponds to **14/15** in this pipeline.
   - **Global L5 α0.5 β0.1** scored **13/15 and 12/15** in two runs. That is within one or two episodes of the paper (p = 0.65), so **the number is approximately reproduced under the candidate's semantics.**
   - **The same-session unsteered baselines** on the same states scored **11, 10 and 12 of 15** (33/45 = 0.73). The candidate's own random-conceptor control scored **11 and 12 of 15**. Global COAST is not distinguishable from either (p = 0.40 and 0.75).
   - **The Table 4 per-step cell, as the code defines it,** scored **9/15 and 9/15** (paper 13/15). **Positive-only** scored **7/15** (paper 12/15).
   - **These single-configuration values are compatible with the paper's values being maxima over the sweep** on the same 15 rollouts (§7.3).
   - **Classification: Case A (qualified) with Case D elements.** We found a likely paper-code path, and it reproduces the KS3 global number approximately. However, that path reports 15-episode, fit-state, best-of-sweep scores, not the 30-episode held-out result the paper describes.
4. **What is missing.** The following are not public, so no faithful reproduction of the exact 28/30 is possible:
   - the 2026-04-13 NPZ builder and the `libero_conceptors.npz` it wrote;
   - the per-condition `steering_results/*/summary.json`;
   - the 45-episode rerun outputs;
   - proof that the steering checkpoint `libero_b200_bs512/2000` is `brandonyang/openpi-libero-2000`.
5. **Contacting the authors is now justified, and the questions are sharper.** The public history strongly suggests that Table 4 reports best-of-sweep scores on the 15 fitting-state rollouts. It does not prove it. Only the authors can confirm it (§10).

Wording: *the released main implementation does not reproduce the reported KS3 result under our documented protocol*, and *the closest publicly available author-development pipeline, run under its native evaluation semantics, gives the result in §7*. We do not claim the paper is wrong.

## 2. Full upstream branch inventory

Full table: `experiments/phase6c_branch_matrix.csv` (all 22 branches plus the PR heads). `git fetch upstream --prune` reported no changes. The HEADs are listed below.

| Branch | HEAD | Last date | LIBERO rel. | COAST rel. | Deep? |
|---|---|---|---|---|---|
| main | 2afa10e | 2026-06-01 | high | high | yes |
| **miranda-v2** | **d5ae9c8** | 2026-05-02 | **high** | **high** | **yes** |
| miranda-groot | df5cca1 | 2026-04-16 | medium (2fdc5ad copy) | high | yes |
| miranda-groot-droid | 622b06f | 2026-04-16 | medium (29059a5 copy) | high | yes |
| steering-refactor = copilot/steering-refactor-research | 8447aca | 2026-04-24 | high | high | yes |
| conceptor-experiments | 9d91d4e | 2026-04-02 | none | high (MetaWorld) | yes |
| miranda | bfb757b | 2026-04-10 | none | high (MetaWorld) | yes |
| worktree-libero-integration | f1b4e54 | 2026-04-09 | high (client + training run) | low | yes |
| worktree-denoising-step-ablation | 194609e | 2026-04-06 | none | low | yes |
| rl-integration | 08a1ae2 | 2026-05-02 | low | none | no |
| preference-bc | b234856 | 2026-04-23 | low | none | no |
| dp-integration | 25b9071 | 2026-04-23 | low | none | no |
| worktree-dp-integration | 710ccb5 | 2026-04-21 | low | none | no |
| subin/dp_robocasa | c51bf17 | 2026-04-26 | none | medium (DP steering) | no |
| subin/temp-droid-activation | 16f231b | 2026-04-10 | low | low | no |
| temp-droid-activation | ab6e517 | 2026-04-09 | low | low | no |
| worktree-migrate | a6f3aa5 | 2026-04-18 | low | none | no |
| worktree-robolab | 50fab34 | 2026-04-12 | none | none | no |
| metaworld-reorganization | a7ff5ce | 2026-04-08 | none | none | no |
| revert-29-worktree-claude-features | e53c009 | 2026-04-13 | none | none | no |
| setup-metaworld | 0a9df72 | 2026-02-19 | none | none | no |
| PR heads #1–#54 (50 refs) | various | 2026-02-19..06-02 | mixed | mixed | searched |

The "Deep?" branches were inspected commit by commit. All branches and PR heads were covered by the pickaxe (`-S`) searches and the artifact searches.

## 3. Historical timeline (2026)

| Date | Commit | Event | Relevance |
|---|---|---|---|
| 04-02 | 9d91d4e | Miranda: first π0.5 conceptor steering (MetaWorld). Token-flattened, uncentered, `A·inv(A+B−AB)·B` AND | Builder lineage |
| 04-07..08 | 5acad0d, 2760dce | LIBERO client scaffold: `set_init_state(initial_states[episode])` | Pre-#48 state semantics |
| 04-09 | f1b4e54 | Miranda `train_libero.sh`: `--exp-name=libero_b200_bs512`, adds `fsdp_devices=4` | Checkpoint path used by the steering sweep |
| 04-09 | 6f0aa99 | Brandon releases `openpi-libero-{2000,3000,9000}`. The README config includes the same `fsdp_devices=4`. Baseline eval: 15 episodes/task, KS3 = 0.47 | Checkpoint identity (circumstantial) |
| 04-09 | HF | `brandonyang/pi05-libero-activations-v1-2000-15env`: KS3 `episode_000..014`, env 0, 8 S / 7 F, `checkpoint_dir: checkpoints/openpi-libero-2000/` | Fit set = states 0–14 [R/M] |
| 04-10 | bfb757b | Miranda: 26-task MetaWorld steering results committed | MetaWorld only |
| **04-13** | **2fdc5ad** | **miranda-v2: `experiments/pi05_libero/src/` sweep. Global/per_step_0/per_step_9 over L{5,11,17} × α5 × β{0.1,0.3,0.5}, random controls, positive-only L{5,11} × α{0.5,1} × β{0.1,0.3}. 15 episodes. Checkpoint `libero_b200_bs512/2000`** | **Table 4 generator (§6)** |
| 04-14 | 117524d | `shared/select_parameters.py` (quota layer, overlap band [0.85,0.95], β{0.1,0.3}) | Table 16 geometric procedure |
| 04-14 | 4df91bd, c19ac52 | Brandon: `serving/conceptors.py` and `--steer` (steering-refactor) | Became main in June |
| 04-15 | 29059a5 | `for_subin/` handoff: `build_conceptors.py` (pooled, centered, `C_s(I−C_f)`, ds 0, per-step at α = 1 for all 10 steps) and a time-varying per-step driver | First public builder writing the NPZ schema |
| 04-19 | 60ca18d | steering-refactor "switch per_step strategy to all-10-steps (miranda-aligned)" | Refactor tracks Miranda |
| **04-21** | **ec62cad** | **`shared/oracle_gap_table.py` (Table 15 source), `make_45ep_latex.py` and `submit_45ep_all.sh` (45-episode "oracle" reruns)** | **Reporting scripts** |
| 04-23 | 1797c0c (#48) | LIBERO `--seed` now offsets init states: `(seed+k) % 50` | First held-out mechanism; miranda-v2 never merged it |
| 04-27 | a84a6f0 | "Fix pi05_robocasa per_step steering to be true time-varying" (RoboCasa only) | Confirms the earlier per-step was not time-varying |
| 05-01..02 | 417c7de, d5ae9c8 | SAE baseline pipeline; "bump α grid" | Table 1 SAE column |
| May | — | Paper submitted (arXiv 2605.17144) | |
| 06-01 | 6c34a11 (#50) | Steering merged to main | Released code postdates the paper |

## 4. Paper / main / author-branch comparison

| Component | Paper | main @ 2afa10e | miranda (MW) | miranda-v2 src (2fdc5ad) + for_subin builder (29059a5) | for_subin driver |
|---|---|---|---|---|---|
| Checkpoint | openpi-libero-2000 [P] | HF openpi-libero-2000 — MATCH | n/a | `libero_b200_bs512/2000` — UNKNOWN (likely the same run: same-day `fsdp_devices=4` tweak) | same as src — UNKNOWN |
| Client state semantics | "different seeds…initial conditions" [P] | `(seed+k)%50` — UNKNOWN vs paper | n/a | `initial_states[k]`, k = 0..N−1; seed does not select — DIFFERENT | DIFFERENT |
| Fit split | 15 rollouts [P] | 15 (states 0–14 at seed 0) — MATCH | n/a | authors' HF data: 15 episodes, states 0–14 — MATCH | MATCH |
| Test split | 30 disjoint rollouts [P] | 30 via `--seed` offset — MATCH (as mechanism) | n/a | none. Eval = 15 episodes on states 0–14 (= fit states) — DIFFERENT | DIFFERENT |
| Token pooling | mean over 10 action tokens [P] | flattened — DIFFERENT | flattened — DIFFERENT | mean-pooled — MATCH (public builder; Apr-13 builder UNKNOWN) | MATCH |
| Centering | centered [P] | uncentered — DIFFERENT | uncentered — DIFFERENT | centered — MATCH | MATCH |
| Contrastive formula | pseudoinverse Boolean AND [P] | `A·inv(A+B−AB)·B` — DIFFERENT | same as main — DIFFERENT | `C_s·(I−C_f)` — DIFFERENT | DIFFERENT |
| Global aggregation | all (call, denoise-step) vectors [P] | all steps × all tokens — MATCH (steps) | all steps | **denoising step 0 only** — DIFFERENT | DIFFERENT |
| Per-step α | KS3 α = 10 [P] | fixed 1.0 — DIFFERENT | n/a | fixed 1.0; α is a name label only — label MATCHES, math DIFFERENT | no α in name — DIFFERENT |
| Per-step mechanism | separate C per step [P] | time-varying — MATCH | time-varying | fixed ds-0 or ds-9 C for all steps — DIFFERENT | time-varying — MATCH |
| Steering gate | M = (1−β)I + βC on h [P] | `h@M.T` on the expert-layer output, all tokens, all steps — MATCH | MATCH | MATCH | MATCH |
| Search grid | L{0,5,11,17} × α5 × β3 × 4 strategies = 240 (Table 14); Table 4 says L{5,11} [P] | configurable — MATCH (Table 14) | — | global/per-step L{5,11,17} × α5 × β3; positive-only L{5,11} × α{0.5,1} × β{0.1,0.3}; = 152 — MATCH to Table 4 cells and the "152" text | β{0.1,0.3} — DIFFERENT (Table 4 has β = 0.5 cells) |
| Evaluation count | 30 per condition [P] | configurable | — | **15** — DIFFERENT from the text, MATCH to the Table 4 denominators | 15 |
| Reporting | "best on fit → evaluated on test" [P] | `find_best_configs.py` argmax | — | `oracle = max SR over summary.json` on the same 15 rollouts; Table 15 generated from these | — |

## 5. Result-artifact search

- **Method.** We listed every blob path reachable from all 22 upstream branches and all 50 PR heads (`git rev-list --objects`). We then ran pickaxe searches for `KITCHEN_SCENE3`, `moka`, `0.933`, `28/30`, `26/30`, `24/30`, `global_L5_a0.5`, `a10.0_b0.3`, `45ep`, `oracle`, `best_configs`, `num-episodes 30`, `libero_conceptors`, `per_step_alpha` and `steering_results_45ep`.
- **No LIBERO steering result artifact was ever committed.** There is no summary.json, CSV, JSON, tex or plot for LIBERO steering on any branch or PR head.
- **Committed LIBERO numbers are unsteered checkpoint evals only.**
  - `examples/libero_env/figures/results_2000_3000_9000.json`: 15 episodes/task, pre-#48 states 0–14. At step 2000, KS3 = 0.47 (7/15) and the libero_10 mean = 0.447.
  - The paper's Base of 0.53 (8/15) instead equals the authors' fit dataset (8 S / 7 F).
- **`experiments/libero/best_configs.json` on main is an empty placeholder.**
- **The committed steering results are MetaWorld (`miranda`).** They are `experiments/steering_results/*-v3/`, `mech_interp_analysis/*.tex` and `methodology.tex`.
- **Hits for `0.933` are unrelated** (MetaWorld, denoising ablation, LIBERO baseline JSON). There are no hits for `28/30` or `24/30`. The `26/30` hits are in the RoboCasa inference docs (`b6196a5`) and a 2025 LeRobot upgrade, unrelated to steering.
- **Expected on-disk (gitignored) artifacts that would settle Table 4:**
  - `/vast/projects/ungar/stellar/miaom/openpi-new/experiments/pi05_libero/steering_results/<task[:60]>/summary.json`. Schema: `{"task": str, "conditions": [{"condition": "global_L5_a0.5_b0.1", "success_rate": float}, …]}`, sorted by success rate descending. Conditions: `baseline`, `global_L{l}_a{a}_b{b}`, `per_step_{0|9}_L{l}_a{a}_b{b}`, `random_L{l}_b{b}`, `pos_only_L{l}_a{a}_b{b}`. Per-condition client videos sit alongside.
  - `…/pi05_libero/selected_params.json` (`best_layer`, `selected_alphas`, `selected_betas`).
  - `…/shared/analysis_output/oracle_gap_table.{tex,json}` (per-task `oracle_sr`, `oracle_cond`, `geo_sr`, `geo_cond`).
  - `…/pi05_libero/steering_results_45ep/<task>/summary.json`, with its `scripts/task_i.sh`, and `table_45ep.tex`.
  - `$OPENPI_DATA_HOME/libero_conceptors.npz` (the HF repo `brandonyang/libero-conceptors` still returns 401).
- **Paper denominators.** Share of per-task cells that are k/15: Table 4 30/30 (0 odd/30); Table 6 (π0.5 RoboCasa) 21/21; Table 9 (GR00T RoboCasa) all but one (a 0.75 cell). Table 8 (π0-FAST MetaWorld) is 108/115 values k/16, matching that pipeline's 16 environments. So the test does discriminate between pipelines.

## 6. Candidate pipeline ranking

Details: `experiments/phase6c_candidate_pipelines.json`.

**C1 — miranda-v2 src sweep + oracle_gap_table.** Commits `2fdc5ad` (04-13), `ec62cad` (04-21); executed at `29059a5`. **Confidence: high** that the Table 4/15 LIBERO numbers come from this pipeline family.

- For:
  - the Table 15 caption template;
  - the 152 and 18 grid counts;
  - the per-step α = 10 label mechanics;
  - the positive-only cells inside the 8-condition grid;
  - k/15 denominators (p = 4 × 10⁻⁹ against N = 30);
  - Table 17 = Global column (best-by-prefix);
  - author cluster paths and April dates.
- Against or missing:
  - The Apr-13 builder and its NPZ are not public.
  - Checkpoint identity is unverified.
  - No results are committed.
  - It does not implement the paper's stated disjoint 30-episode test.

**C2 — miranda-v2 45-episode "oracle" rerun (`ec62cad`).** **Confidence: low** as the Table 4 source. States 0–44 would be 15 fit + 30 new states, which fits the paper's 15/30 description. But the Table 4 cells are not k/45 (p = 3.9 × 10⁻¹⁴). The scripts and outputs are off-repo. The paper's Table 1 z-test is not in this script. Whether any paper number (e.g. Table 1 significance) came from it is unknown.

**C3 — for_subin driver (`29059a5`).** **Confidence: low.** Its per-step names carry no α, and its default β set excludes 0.5, so it cannot produce Table 4's labels.

**C4 — released main / steering-refactor.** **Confidence: low.** It postdates submission, fixes per-step α at 1.0, uses a flattened and uncentered builder, and scored 19–21/30 in our tests.

"No public branch exactly matches the paper" holds. C1 matches the paper's **numbers and labels** but not its **stated protocol**.

## 7. Targeted reproduction (C1, native semantics)

### 7.1 Why execution was justified (Step 8)

C1 is materially closer than main **to the pipeline that produced the numbers**. It differs in the evaluation semantics, the builder, the per-step definition and the reporting. A cheap targeted test could separate "the paper's code yields about 14/15 under its own semantics" (Case A) from "it also gives about 19–21/30-equivalent" (Case B).

### 7.2 Setup (only machine-specific adaptations)

| Item | Value |
|---|---|
| Worktree | `/home/stanley/Stanley_ws/COAST-research/COAST-paper-candidate`, detached at `29059a537ce123abdab7a4e164db0c72eeff8024`. `git status`: clean (only ignored `examples/libero_env/.venv` symlink and `__pycache__`). No commits |
| Code executed unmodified | `experiments/pi05_libero/for_subin/build_conceptors.py` (defaults); `experiments/pi05_libero/src/conceptor_steering.py`; `experiments/pi05_libero/src/positive_only_steering.py`; the worktree's `src/openpi` (steering sampler, `infer_with_steering`), `packages/openpi-client` and `examples/libero_env/main.py` |
| Fit data | Authors' `brandonyang/pi05-libero-activations-v1-2000-15env` @ `3e3a8fe2`, KS3 only: 15 episodes, 1,141 inference steps, 8 S / 7 F. Builder: n(s,f) = (413, 728) pooled samples at ds 0 |
| Adaptations | Checkpoint path → `checkpoints/openpi-libero-2000` (HF rev `aaeeabc…`; the conversion hash matched, so no re-conversion); `OPENPI_DATA_HOME` → scratch (tokenizer copied); `LIBERO_CONFIG_PATH` (as in all phases); GPU 0; ports 8770–8773; output dirs; SLURM → local bash; `uv run` → our root/libero venvs with `PYTHONPATH` = worktree `src` + `openpi-client` (torch 2.7.1 in both; the candidate pins the cu128 build, we use cu126; same LIBERO `d63b117`; identical `transformers_replace`) |
| Invocation choice | The driver's own CLI flags restrict the sweep to the Table 4 KS3 cells (`--layers/--alphas/--betas/--strategies`). The driver natively adds baseline and one random control per (layer, β) |
| Not changed | Conceptor equations, pooling, centering, state selection (`initial_states[episode]`, `env.seed(7)`), α/β math, hook site, `h@M.T`, success criterion, 15 episodes, unseeded policy noise |
| Infrastructure failures | Attempt 1 (tokenizer symlink rejected by `download.py`) and attempt 2 (missing `LIBERO_CONFIG_PATH` → LIBERO interactive prompt) failed before any rollout. Both logs were kept. The empty `~/.libero` that attempt 2 created was removed |
| Per-episode outcome | Recovered from video length (530 frames = timeout). Matches the driver's `success_rate` for all 11 conditions |

**Evaluated states:** init states **0–14** for every condition (`initial_states[episode]`, episode 0..14). These are the same states as the authors' fit rollouts.

**15 vs 30.** The candidate code evaluates 15 episodes. We did not change it to 30. The historical evidence (§5, the parity test) says the Table 4 LIBERO values are 15-episode rates, so no 30-episode invocation is implied.

### 7.3 Results (KS3; `experiments/phase6c_candidate_test_results.csv`)

| Run | Condition (candidate naming) | Table 4 cell | Paper (native equiv.) | Ours |
|---|---|---|---|---|
| 1 | baseline | Base 0.53 (= fit data 8/15) | 8/15 | **11/15** |
| 1 | `global_L5_a0.5_b0.1` | Global 0.93 | 14/15 | **13/15** |
| 1 | `random_L5_b0.1` | — (control) | — | 11/15 |
| 2a | baseline | | | 10/15 |
| 2a | `global_L5_a0.5_b0.1` | Global 0.93 | 14/15 | **12/15** |
| 2a | `random_L5_b0.1` | — | — | 12/15 |
| 2b | baseline | | | 12/15 |
| 2b | `per_step_0_L5_a10.0_b0.3` | Per-step 0.87 | 13/15 | **9/15** |
| 2b | `per_step_9_L5_a10.0_b0.3` | Per-step 0.87 | 13/15 | **9/15** |
| 2b | `random_L5_b0.3` | — | — | 8/15 |
| 2c | `pos_only_L11_a1.0_b0.1` | Pos-only 0.80 | 12/15 | **7/15** |

Pooled:
- baseline **33/45 (0.73)**;
- global **25/30 (0.83)**;
- random β0.1 **23/30 (0.77)**;
- per-step **18/30 (0.60)**.

Fisher exact tests:
- global vs baseline p = 0.40;
- global vs random p = 0.75;
- global vs paper-native 14/15 p = 0.65;
- per-step vs 13/15 p = 0.09;
- positive-only vs 12/15 p = 0.13.

**Offline operator check.** This uses the candidate NPZ on 11,500 authors' token vectors. At L5 α0.5 β0.1: ‖Δh‖/‖h‖ = 0.097 and cos(Δh, h) = −0.9996. So the candidate operator is also ≈ a uniform 0.9·h shrink, like main (Phase 1E). Other values:
- quota of `C_s(I−C_f)` = 0.0041;
- `per_step_0` is byte-identical to the global α = 1.0 conceptor;
- KS3 overlap at L11 α10 = 0.739. The paper's 10-task mean is 0.670; our main/V3 builders gave 0.937.

**Order-statistic compatibility.** Assume the reported value = the maximum over the strategy's conditions on the same 15 rollouts, with conditions treated as independent (this overstates the effective number of conditions):

| Strategy | Conditions | Assumed rate | P(max ≥ reported) |
|---|---|---|---|
| global | 45 | 0.73 | P(≥ 14) = 0.93 |
| per-step | 90 names | 0.60 | P(≥ 13) = 0.92 |
| positive-only | 8 | 0.60 | P(≥ 12) = 0.53 |
| positive-only | 8 | 0.73 | P(≥ 12) = 0.98 |

### 7.4 Interpretation

**Case A (qualified), with Case D elements.**

What the run shows:
- A public author pipeline, run as written, reproduces the KS3 Global value approximately: 13 and 12 of 15 vs 14/15.
- **What differs from main is mainly *what is measured*, not the steering math.**
  - 15 rollouts on the fit states 0–14.
  - Reported as best-of-sweep.
  - Versus main's seed-offset held-out states.
  - The builder differs as well (pooled, centered, product, ds 0), but the operator is still ≈ 0.9·h at this configuration.
- In the same sessions, unsteered and random-conceptor runs reach the same range. So the fit-state number is **not evidence of a steering gain**.
- This is consistent with our held-out results (Phases 3A–5C).

Case D elements: the branch contains the result-*generation* scripts (sweep driver, `oracle_gap_table.py`, `make_45ep_latex.py`), but not their outputs. §5 lists the expected files.

## 8. What is now known

1. **Closest public branch/commit:** `upstream/miranda-v2`.
   - The sweep pipeline was introduced at `2fdc5ad` (2026-04-13).
   - The reporting script (`oracle_gap_table.py`) was added at `ec62cad` (2026-04-21).
   - `29059a5` = `2fdc5ad` + additive files, plus the first public NPZ builder. This is the commit we executed.
2. **Exact match to the paper: none.**
   - The code whose grid, labels, counts and denominators match Tables 4/15 does not implement the paper's text: no disjoint 30-episode test, a non-time-varying per-step with α as a label, a `C_s(I−C_f)` builder at ds 0 only.
   - No public code implements the paper's text protocol either.
3. **Several Table 4 anomalies have mechanical explanations in that code:**
   - the per-step "α = 10";
   - the positive-only cells confined to α{0.5,1}, β{0.1,0.3};
   - the "layer sweep {5,11}";
   - the "up to 152 configurations";
   - "Selected Grid 18";
   - k/15 denominators.
4. **Table 17 = Table 4 Global column**, consistent with best-by-prefix from one `summary.json`. The Table 1 Base (0.53) = the fit dataset (8/15), not a baseline run from the sweep.
5. **Under the candidate's semantics the KS3 global number is approximately reproducible** (12–13/15 vs 14/15). So is the baseline (10–12/15 on the same states). Main's 19–21/30 on held-out states and the candidate's 12–13/15 on fit states measure different things.
6. **Per-step α = 10 cannot be generated by any public code as a real α-10 per-step conceptor.** In the candidate it is a name for the α = 1 ds-0 (or ds-9) conceptor. In main it is fixed at 1.0. In for_subin it is absent. This is strong evidence that the Table 4 per-step column is not the paper's described per-step method (Step 12).
7. The 45-episode "oracle" reruns existed as scripts (states 0–44 under this client). Table 4's values are not 45-episode rates.

## 9. What remains unknowable from public code

| Missing item | Why it matters | Where it would be |
|---|---|---|
| Per-condition `steering_results/<task>/summary.json` (all 10 tasks) | Would confirm directly that Table 4 = per-prefix max of the 15-episode sweep, and which of `per_step_0`/`per_step_9` won | `/vast/.../openpi-new/experiments/pi05_libero/steering_results/` |
| Builder used for the 2026-04-13 `libero_conceptors.npz` (and the NPZ) | The global conceptor may differ from the for_subin builder (pooling, centering, AND, ds) | `$OPENPI_DATA_HOME/libero_conceptors.npz`; HF `brandonyang/libero-conceptors` (401) |
| Checkpoint identity `libero_b200_bs512/2000` vs HF `openpi-libero-2000` | Fit data and steering may come from different weights | Miranda's run directory; weight hash |
| 45-episode rerun outputs and scripts | Whether any held-out-state (15–44) numbers exist, and where they were used (e.g. Table 1 significance) | `steering_results_45ep/` |
| Policy-noise seeding and runs per cell | Best-of-k inflation | Launch logs |
| Whether "30 held-out test rollouts" (captions, A.8) describes any run behind Tables 1/4 | Central question for the claim | Authors |
| Other nine LIBERO tasks | Our execution covers KS3 only | — |

We did not run the full 152-condition candidate sweep (≈ 2,300 episodes). That would test the best-of-sweep reading directly on our hardware, but it cannot recover the authors' own draws.

## 10. Is contacting the authors now justified?

**Yes.** We have now exhausted public code, public history and the public fit data.

Why contacting the authors is now justified:
- The remaining questions are answerable only by the authors.
- They are much more specific than in Phase 6B.

Revised priority questions (draft only; **not sent**):
1. "Were the π0.5 LIBERO values in Tables 4, 17 and 15 produced by `experiments/pi05_libero/src/conceptor_steering.py` (miranda-v2, 2026-04-13) with `NUM_EPISODES=15`, and taken as the best condition per strategy from `summary.json`? All 30 Table 4 cells are multiples of 1/15, and the Table 15 caption matches `oracle_gap_table.py`."
2. "With the pre-PR#48 LIBERO client, those 15 episodes use init states 0–14, which are the same states as the 15 fitting rollouts. Was a separate 30-rollout held-out evaluation (as described in A.8.5) run for Tables 1/4, e.g. the `steering_results_45ep` reruns? If so, which states and which numbers?"
3. "Is the KS3 per-step 'α = 10' the loop label of `per_step_{0|9}_L5_a10.0_b0.3`? That condition applies the α = 1.0 conceptor of a single denoising step at every step."
4. "Which builder wrote the `libero_conceptors.npz` used by that sweep? Is `checkpoints/pi05_libero/libero_b200_bs512/2000` the checkpoint released as `brandonyang/openpi-libero-2000`?"
5. "Could the `steering_results/*/summary.json` files (or `oracle_gap_table.json`) be shared?"

Recommended framing:
- "The released main implementation does not reproduce the reported KS3 result under our documented held-out protocol."
- "The public miranda-v2 pipeline, run under its native semantics (15 rollouts on the fitting states), gives 12–13/15 for the reported configuration, close to the reported 14/15. Unsteered and random-conceptor runs on the same states give 10–12/15."
- We are asking which protocol the tables report.

No emails were sent. No method changes were made.
