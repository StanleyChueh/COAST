# Phase 4B — Paper / Code / Reproduction Protocol Audit

Question: **did our experiments actually reproduce the paper setting?**

Documentation and verification only. No LIBERO rollout, no activation collection, no code, steering or parameter change. The only actions were reading files and public web pages, and a read-only comparison of the authors' released per-episode metadata.

| | |
|---|---|
| Paper | *Contrastive Conceptor Activation Steering (COAST)*, arXiv:2605.17144, **v1** (submitted 2026-05-16); full HTML text (main paper, Appendices A/B, checklist) read on 2026-09-24 |
| Released code | `github.com/COAST-VLA/COAST`; upstream `main` HEAD = `2afa10ee256a3b3edfeb56500fea166a0837f119` (`git ls-remote upstream`, pushed 2026-06-02), which is the base of all our work |
| Our tree | branch `exp/paper-protocol-audit`, HEAD `6e1afd2084d1bb55d178103c538d02f0c81cbbfa`, clean at start |
| Research records | `research/reproduction/` Phases 0–4A |

Evidence tags: **[P]** paper, **[R]** repository, **[O]** our records, **[W]** public web artifact (HF metadata), **[I]** inference by the auditor (stated as such).

## 1. Executive summary

**Verdict: a faithful reproduction of the *steering configuration and model* for one task, a partial reproduction of the *data-collection protocol*, and a different experiment from the paper's *evaluation protocol*. The paper's headline number for this task (0.93) was not reproduced, and the audit cannot yet say why.**

What matches, with evidence:
1. **Model and checkpoint.** `pi05_libero` at 2,000 gradient steps from `brandonyang/openpi-libero-2000` (revision `aaeeabc7…` is the repo's current revision). Every training hyper-parameter in the paper's A.11.2 equals the repository config. The weights themselves cannot be hash-verified against the authors' [I].
2. **Steering configuration.** For KITCHEN_SCENE3 (Stove+Moka = libero_10 task 2) the paper's oracle configuration is *global, layer 5, α 0.5, β 0.1* (Tables 4 and 17). We ran exactly that, with the same gate `M = (1−β)I + βC`, applied to the same module (`layers[5]` output, all 10 denoising steps). Layer index 5 is 0-based in both (A.7.1).
3. **Activation format.** Our `all_suffix_residual` (10,4,10,1024) is the paper's `(D,L,S,1024)` schema with L = {0,5,11,17}, D = 10, S = 10 (A.7, A.8).
4. **Fit set.** 15 rollouts, seed 0, and our class counts (8 success / 7 failure) equal the paper's Table 21 for this task. Per-episode outcomes do not match (7/15 episodes agree, chance level, because policy noise is unseeded), so the fit data are equivalent in *size and split*, not identical.

What differs or cannot be verified, ranked by how much it could matter:
1. **Table 1's unsteered "Base" column appears to be the 15 *fitting* rollouts, not a paired test baseline** [I]. For all 10 LIBERO tasks the π0.5 "Base" equals the Table 21 fit success rate (10/10 exact matches, mean 0.43 = 65/150). If confirmed, reported gains compare steered *30-episode test* success with unsteered *15-episode fit* success on different states. On our data the fit states are harder for the baseline (0.53 on states 0–14) than states 15–44 (0.70), which alone would shrink KS3's Δ from +0.40 to about +0.23.
2. **The paper's test states, seeds and code version are unspecified**, and the LIBERO `--seed` semantics changed on 2026-04-23 (PR #48) *after* the authors' released dataset (2026-04-09). Before #48 every run used init states 0…N−1. If any paper run predates it, the test set would have overlapped the fit states.
3. **The paper selects hyper-parameters per task on the fit rollouts from a 240-configuration grid**; the repository's documented protocol (three disjoint 15-episode windows, small default grid, empty `best_configs.json`) is different from the paper's (15 fit + 30 test).
4. **Conceptor construction differs between paper and released code** (token pooling, centering, Boolean AND; D1–D3 of Phase 1A/1B). Which builder produced the paper's numbers is unknown. Our V0 = released code; V3 = paper text. Phase 1B/1D found both behave alike offline and on 15 states.
5. **We tested only global steering.** The paper's LIBERO headline uses per-step (0.43 → 0.80); its per-step configs use α up to 10, which the repository's per-step strategy cannot express (α fixed at 1.0).

Numeric gap for this task: the paper reports 28/30 (0.93) steered. Our COAST β0.1 gives 20/30 on states 15–44, 12/20 on states 45–49 + 0–14, and 32/50 pooled. Fisher exact p = 0.021 / 0.009 / 0.003 against 28/30 (reference only, since state sets differ). So the discrepancy is unlikely to be sampling noise alone *if the test sets are comparable*.

**Recommended next action: contact the authors** (questions in §9), while doing only *offline* work with the authors' public artifacts. Do **not** begin an improvement method yet (§10).

## 2. Paper configuration (what arXiv:2605.17144 v1 states)

| Item | Statement | Source |
|---|---|---|
| Model | π0.5 (also π0-FAST, GR00T N1.5, Diffusion Policy for other benchmarks); LIBERO: π0.5 and π0-FAST | §4.1 |
| Base weights / fine-tune | `gs://openpi-assets/checkpoints/pi05_base/params`; dataset `physical-intelligence/libero`; config `pi05_libero`; AdamW, clip 1.0; cosine schedule warmup 10,000, peak LR 5e-5, decay 1,000,000 to 5e-5; batch 256 over 4 FSDP devices; EMA 0.999; 30,000 total steps; action horizon 10 | A.11.2 |
| Checkpoint for fit and eval | `/openpi-libero-2000`, step 2,000 of 30,000, "the earliest checkpoint exhibiting nonzero success across most LIBERO-10 tasks"; released 2,000 / 3,000 / 9,000 | A.8.3, A.11.2 |
| Suite | LIBERO-10, 10 long-horizon tasks (no LIBERO/robosuite version given) | §4.1 |
| Split | 15 fitting rollouts per task (conceptors *and* hyper-parameter selection), 30 held-out test rollouts per task per condition, "different environment seeds, object placements, and initial conditions", no shared episode ids or seeds | §4.1, A.8, A.8.5 |
| LIBERO initial conditions / seeds | "built-in per-task initial state distribution. The environment is reset with sequential episode seeds." No seed values | A.8.3 |
| Success | LIBERO `info["is_success"]`, "labeled successful if the final step's success flag is True" | A.8.3 |
| Fit class counts | KS3 8 S / 7 F; KS4 6/9; KS6 2/13; KS8 3/12; LR1 12/3; LR2a 6/9; LR2b 9/6; LR5 1/14; LR6 8/7; ST1 10/5 (total 65/85, mean SR 0.43); `MIN_PER_CLASS = 3` | Table 21, App. B |
| Hook site | forward hook on `paligemma_with_expert.gemma_expert.model.layers[i]`, i ∈ {0,5,11,17}; layer output; 10 Euler steps, all recorded; token dimension S = 10 for LIBERO | A.7.1 |
| Fitting samples | activations **mean-pooled over the 10 tokens** → one 1024-vector per inference call per denoising step; N stacked vectors | §3.2, A.7.1, A.9.1 |
| Conceptor | **mean-centered**; R = X̃ᵀX̃/N; C = R(R + α⁻²I)⁻¹; float64, stored float32 | Eq. 1, A.9.1, A.9.4 |
| Contrastive | C_s ∧ ¬C_f, ¬C = I − C, AND = (A⁻¹ + B⁻¹ − I)⁻¹ via Moore–Penrose pseudoinverse | Eq. 3, 4, 7, A.9.1 |
| Gate | h′ = hMᵀ, M = (1−β)I + βC_steer, applied at the hooked layer output at every denoising step (global) | Eq. 5, A.9.2, Alg. 1 |
| Strategies | global (one C over all steps), per-step (one C per step), positive-only (C = C_s); linear/random controls in App. A.3 | §3.2, A.10.1 |
| Sweep grid (LIBERO) | layer {0,5,11,17} × α {0.1,0.5,1,2,10} × β {0.1,0.3,0.5} × 4 strategies = 240 configurations per task | Table 14 |
| Selection | "Hyperparameter configurations … are selected by maximizing success rate on the fitting rollouts"; oracle = best rollout success rate; also a geometric shortcut (layer 11, α {0.5,1.0}, β {0.1,0.3}) recovering ≈ 92% of oracle | A.8.5, A.10.2–A.10.3, Table 15/16 |
| KS3 oracle | Global **L5, α 0.5, β 0.1** = **0.93** (28/30); per-step L5, α 10, β 0.3 = 0.87; positive-only L11, α 1.0, β 0.1 = 0.80 | Tables 4, 17 |
| LIBERO π0.5 means | Base 0.43; +Glob 0.76; +Per 0.80; +Pos 0.63; Δ(per-step) = +0.37 | Table 1, 4 |
| KS3 baseline | Base 0.53 | Table 1 |
| Controls | Random-eigenvector conceptor with matched spectrum: only GR00T-RoboCasa (mean 0.58 vs base 0.59) and π0.5-MetaWorld (0.55 vs base 0.58); **none on LIBERO**; best-of-sweep, 15 rollouts per cell. No pure-shrinkage (M = (1−β)I) control anywhere | Table 10 |
| Diagnostics | C_steer effective rank ≈ 1% of d; layer quota at α = 10 (LIBERO mean): L0 0.016, L5 0.059, L11 0.092, L17 0.082; overlap at L11 vs α: 0.955, 0.937, 0.882, 0.808, 0.670 | §4.3, Tables 18, 19 |
| Hardware / numerics | inference bf16; latency measured on one B200 | A.7.1, Table 24 |

## 3. Released code configuration (`COAST-VLA/COAST` @ `2afa10e`)

| Item | What the code / docs do | Source |
|---|---|---|
| Checkpoint | README maps `pi05_libero` to `brandonyang/openpi-libero-{2000,3000,9000}`; all scripts default to `checkpoints/openpi-libero-2000` | `examples/libero_env/README.md`, `run_end_to_end.sh` |
| Training config | `pi05_libero` (values equal paper A.11.2; see §5.1) | `src/openpi/training/config.py` |
| Hook | `ConceptorSteeringHook`: `M = (1−β)I + βC`, `h @ M.T`, all tokens, all denoise steps; single matrix for `global`; 10 matrices for `per_step` | `src/openpi/serving/steering.py` |
| Collected layers | `(0, 5, 11, 17)` via hooks on `expert_layers[i]` (0-based) | `pi0_pytorch.py`, `conceptors.py` |
| Builder samples | `flatten_global`: every (call, denoise step, **token**) is a row: **no pooling** | `conceptors.py` (D1) |
| Builder centering | `correlation_matrix`: **uncentered** R = XᵀX/N | `conceptors.py` (D2) |
| Builder AND | `A @ inv(A + B − A@B) @ B` (equals the paper's only for commuting A, B); result not exactly symmetric | `conceptors.py` (D3) |
| Class threshold | `min_episodes_per_class = 2` (paper: 3) | `conceptors.py`, `compute_conceptors.py` (D6) |
| Defaults | collect layers (0,5,11,17); α (0.1,0.5,1,2,10); per-step indices 0–9; per-step α fixed at **1.0** | `conceptors.py`, `find_best_configs.py` |
| Client seeds | since PR #48 (2026-04-23): episode k uses init state `(seed + k) % 50`; `--seed` also sets env/numpy RNG. Before #48: `initial_states[episode]` regardless of seed | `examples/libero_env/main.py`, `git show 1797c0c` |
| Success | `done` from `env.step`; episode ends at first success | `main.py` |
| Max steps | `libero_10`: 520 | `main.py` |
| Documented protocol | collect `--seed 0`, sweep `--seed 15`, final eval `--seed 30`, **15 episodes each** (disjoint windows) | `experiments/libero/README.md`, `run_end_to_end.sh` |
| Sweep defaults | layers `(11,)`, α `(0.1,0.5,1.0)`, β `(0.1,0.3)`, 5 strategies, 10 episodes; picks argmax steered SR per task | `find_best_configs.py` |
| Tuned configs | `experiments/libero/best_configs.json` is a **placeholder** (`"tasks": {}`) although the README says tuned params are "committed" | `best_configs.json`, `examples/libero_env/README.md` |
| Released artifacts | activation dataset `brandonyang/pi05-libero-activations-v1-2000-15env` (public, 64,086 files, last modified 2026-04-09); conceptor NPZ `brandonyang/libero-conceptors` returns **HTTP 401** for anonymous access, so its existence and contents are unverifiable [W] | README, HF API |
| Other upstream branches | `miranda`, `miranda-groot`, `conceptor-experiments`, `dp-integration`, `metaworld-reorganization`, `copilot/steering-refactor-research` (research code; not inspected) | `git ls-remote` |

Our local tree adds only research instrumentation over `2afa10e` (noise control, `shrinkage` strategy, diagnostics, a log line); the `global` math and the builder are unmodified (`git diff 2afa10e HEAD` limited to 7 files under src/scripts/examples/packages; Phases 2A/2B).

## 4. Our reproduction configuration (from Phases 0–4A records)

| Item | Value | Record |
|---|---|---|
| Model / config | π0.5, `pi05_libero`, PyTorch (converted on first serve) | Phase 0 |
| Checkpoint | `checkpoints/openpi-libero-2000` ← `brandonyang/openpi-libero-2000` rev `aaeeabc72f8a…` | environment_snapshot |
| Stack | Python 3.8 client venv, MuJoCo 3.2.3, robosuite 1.4.0, LIBERO `d63b117`, numpy 1.22.4; 2× RTX 4090, driver 580.178 | environment_snapshot |
| Task | libero_10 task 2 (KS3) primary; task 3 (KS4) pilot | Phases 1A–4A |
| Fit data | `--seed 0`, 15 episodes, states 0–14; 8 success / 7 failure; activations `(10,4,10,1024)`, fp32 | Phase 1A |
| Conceptor | V0 (released builder) and V3 (paper text: pooled, centered, canonical pinv-AND); V0 used for all steering runs | Phases 1B, 1C+ |
| Steering | global, layer 5, α 0.5, β 0.1 (all phases); β 0.2 and 0.3 added in 4A; shrinkage and random-matched controls | Phases 2A–4A |
| Evaluation states | 15–29 (dev), 30–44 (held-out, 3A), 45–49 + 0–14 (4A P), 15–44 reused (4A S); 15–30 episodes per condition; max 520; replan 5 | Phases 1C–4A |
| Noise | paired flow noise, master seed 100 (our own instrumentation; the paper does not describe policy-noise control) | Phase 1D |
| Baseline (task 2) | fit states 0–14: 8/15; states 15–44: 21/30; all 50: 30/50 | Phases 1A, 4A |
| Steered (task 2, β 0.1) | dev 11/15, held-out 11/15 (3A), P 12/20, S 20/30, all 50: 32/50 | Phases 2B–4A |

## 5. Audit questions

### 5.1 Model and checkpoint

| Item | Paper | Repository | Ours | Match |
|---|---|---|---|---|
| Base VLA | π0.5 | π0.5 (`pi05_libero`, PyTorch hooks) | π0.5 PyTorch | yes |
| Pretrained base | `pi05_base` | via `pi05_libero` weight loader | (inherited in the checkpoint) | yes [I] |
| Fine-tuning data | `physical-intelligence/libero` | same in `pi05_libero` | not retrained | n/a |
| Optimizer / schedule | AdamW; warmup 10,000; peak 5e-5; decay to 5e-5 over 1e6 steps; batch 256; EMA 0.999; 30,000 steps; horizon 10 | config: warmup 10,000, peak 5e-5, decay_steps 1,000,000, decay_lr 5e-5, batch 256, EMA 0.999, 30,000 steps, horizon 10 | logged in our server output | **yes (all equal)** |
| Checkpoint step | 2,000 | 2,000 (default) | 2,000 | yes |
| Checkpoint source | name only (`/openpi-libero-2000`) | HF `brandonyang/openpi-libero-2000` | same, rev `aaeeabc7…` = current (2026-04-09, before the paper) | **name and repo match; identity of the weights not hash-verifiable** |
| Precision | bf16, activations to fp32 | same | same | yes |

Verdict: **matches on everything that is checkable.** Unknown: whether the paper's numbers came from this exact revision, and whether JAX→PyTorch conversion details are identical (the paper gives none).

### 5.2 Dataset and task suite

| Item | Paper | Repository | Ours | Difference |
|---|---|---|---|---|
| Suite / tasks | LIBERO-10, 10 tasks | `libero_10`, same task ids/names | task 2 (+ task 3 pilot) | we cover 1 (+1) of 10 tasks |
| Task identity | KS3 = "turn on the stove and put the moka pot on it" | task 2, same name | task 2 | none |
| LIBERO version | not stated | `third_party/libero` submodule; robosuite 1.4.0 pinned | LIBERO `d63b117`, robosuite 1.4.0 | paper silent |
| Init states | "built-in per-task initial state distribution", "sequential episode seeds" | 50 per task; index rule changed by PR #48 | 50 per task; index `(seed+k) % 50` | see 5.5 |
| Split | 15 fit / 30 test per task | docs: 15 collect / 15 sweep / 15 eval | fit 0–14; eval dev 15–29, held-out 30–44, plus 45–49 | **paper: 30 test; repo docs and our phases: 15 per window** (4A's set S has 30 states) |
| Episode counts per condition | 30 (test) | 15 | 15 (Phases 1–3), 20 and 30 (4A) | N differs |
| Fit outcomes (KS3) | 8 S / 7 F | (dataset released: 8/7 confirmed from metadata [W]) | 8 S / 7 F | counts equal; per-episode pattern equal in only 7/15 (noise) |
| Fit outcomes (KS4) | 6 S / 9 F | released dataset 6/9 [W] | 8 / 7 (Phase 4A collection) | differs; single-run noise |

### 5.3 Steering configuration

| Item | Paper | Repository defaults / configs | Ours | Match |
|---|---|---|---|---|
| Layer | oracle KS3: 5 | default CLI 11; sweep default `(11,)`; `best_configs.json` empty | 5 | matches paper oracle |
| α | oracle KS3: 0.5 | CLI default 0.1; sweep `(0.1,0.5,1.0)` | 0.5 | matches paper oracle |
| β | oracle KS3: 0.1; grid {0.1,0.3,0.5} | CLI default 0.3; sweep `(0.1,0.3)` | 0.1 primary; 0.2 (off-grid) and 0.3 | 0.1 matches; **0.2 is not a paper grid value** |
| Strategy | global (0.93), per-step (0.87), pos-only (0.80) | `global`, `per_step`, `positive_only`, `random_matched`, `linear` (+ our `shrinkage`) | global only | **only 1 of the paper's 3 strategies; the headline (per-step 0.80 mean) untested** |
| Per-step α | up to 10 (KS3 per-step α = 10) | per-step α fixed at 1.0 | not run | repository cannot express the paper's per-step configs |
| Gate | h′ = hMᵀ, all denoise steps | same | same | yes |
| Selection | oracle on fit rollouts, 240 configs | argmax over a small sweep on a separate window | none: config transferred from the paper | ours is a fixed-config test |

The chosen settings match the paper's Table 17 row for KS3 exactly. They were not selected by us, which is the correct choice for a reproduction of the paper's stated oracle, but it also means our runs cannot confirm the *selection procedure*.

### 5.4 Activation collection

| Aspect | Paper | Repository | Ours | Same? | Could it affect performance? |
|---|---|---|---|---|---|
| Hidden state | layer output (residual) of the action expert, layers 0/5/11/17 | same hooks | same | yes | no |
| Denoising steps | 10, all recorded | same | same | yes | no |
| Tensor | `(D,L,S,1024)` fp32 | `all_suffix_residual` | `(10,4,10,1024)` | **exact schema match** | no |
| Tokens | mean-pool the 10 action tokens → 1 vector per step | **no pooling**, every token a row (D1) | V0 = repo; V3 = paper | paper ≠ repo | yes: C differs by 53–123% relative Frobenius; quota −33–49% (Phase 1B) |
| Centering | mean-centered | **uncentered** (D2) | V0 uncentered; V3 centered | paper ≠ repo | mainly C_s, C_f; small on C_contrastive (Phase 1B) |
| Boolean AND | canonical pinv form | non-canonical, non-symmetric (D3) | V0 repo; V3 canonical | paper ≠ repo | small (3–23%) |
| Class threshold | ≥ 3 per class | ≥ 2 (D6) | 8/7 here | differs, not triggered | none for task 2 |
| Class imbalance | not addressed | not addressed (D7) | timeouts weigh more in C_f | same as paper as far as known | possible |
| Precision | float64 build, float32 store | same | same | yes | no |

Phase 1B found the paper's own quota ordering (L11 > L5) is reproduced by V3 (0.155 vs 0.095) but *not* by V0 (0.140 vs 0.168). V3's L5 α 0.5 quota is 0.0103 (≈ 1% of d), matching the paper's "roughly one percent" statement. Neither V0 nor V3 reproduces the paper's α = 10 overlap of 0.670 (V3 0.937). One task versus the paper's 10-task mean, so this is indicative only.

**A structural point that is not a repository-vs-paper difference.** The paper's Eq. 5 applies a conceptor built from *mean-centered* activations to *uncentered* h. Our Phase 1B shows the shared mean carries 62% of the uncentered trace of the token-pooled activations at L5 (20% for token-flattened rows), and both V0 and V3 have quota ≤ 0.17 (α 0.5: 0.0185 and 0.0103). An offline in-sample check found the paper-faithful V3 gate changes h by 9.45% at β = 0.1 (V0 9.94%), i.e. it also acts mostly as a uniform `(1−β)` shrink. So the collapse to shrinkage observed in Phases 2A–4A is **not attributable to the repository's builder deviating from the paper**; it appears to be a property of the operator as written [I]. The paper never tests a pure-shrinkage control.

### 5.5 Evaluation protocol

| Item | Paper | Repository | Ours | Verdict |
|---|---|---|---|---|
| Test episodes per task | 30 | 15 (docs) | 15 / 20 / 30 | different N |
| Seeds | "different seeds", values not given | `--seed` = offset into init states (post-#48) | 0 fit; 15, 30, 45 eval; master noise seed 100 | paper undefined |
| Init states of test set | not stated | disjoint windows by construction | 15–44, 45–49, 0–14 | **cannot compare to the paper** |
| Success metric | final-step success flag | first `done` | first `done` | equivalent (LIBERO ends at success) |
| Task selection | all 10 tasks | all 10 | 1 (+1 pilot) | partial |
| Oracle settings | per-task, selected on the fit rollouts | placeholder JSON | taken from paper Table 17 | ours transfers the paper's oracle |
| Unsteered baseline | Table 1 "Base": **equals the fit-set rate for 10/10 tasks** | measured separately per window | measured on the *same* states, paired noise | ours is paired; the paper's may not be [I] |
| Policy noise | not described | unseeded CUDA generator | seeded and paired (Phase 1D) | ours is stricter |
| Random / shrinkage control | random on MetaWorld and RoboCasa-GR00T only; no shrinkage | `random_matched` strategy exists | random (dev, held-out), shrinkage (all) | ours adds controls the paper lacks on LIBERO |

**The Base-column observation (item 6 of the summary).** Paper Table 1 (π0.5 LIBERO-10 "Base") versus Table 21 (fit rollouts, N = 15): KS3 0.53 = 8/15; KS4 0.40 = 6/15; KS6 0.13 = 2/15; KS8 0.20 = 3/15; LR1 0.80 = 12/15; LR2a 0.40 = 6/15; LR2b 0.60 = 9/15; LR5 0.07 = 1/15; LR6 0.53 = 8/15; ST1 0.67 = 10/15. **All ten agree; the mean 0.43 = 65/150.** π0-FAST agrees on the five tasks whose Base values could be read (KS3 0.80, KS4 0.67, KS6 0.47, KS8 0.33, LR1 0.67 = 12, 10, 7, 5, 10 of 15). A separate 30-episode unsteered test run would be expected to differ from the fit-set rate on several tasks by sampling noise (SD ≈ 0.09), so exact agreement on every task suggests the Base column *is* the fit set. **This is an inference, not a statement in the paper.** If true, the paper's Δ compares steered test success with unsteered fit success. Our own paired data show why that matters: baseline is 8/15 on fit states 0–14 but 21/30 on states 15–44 (Fisher p = 0.33; small sample, one task), so the fit baseline may understate the test baseline.

**Classification of our evaluation:**
- *Faithful* for: model, checkpoint, layer/α/β/global strategy for KS3, hook mechanics, activation schema, fit-set construction (size, split, seed 0).
- *Partial* for: task coverage (1 of 10), N per condition, strategy coverage (global only), no per-task selection.
- *Different experiment* for: the test-set definition and unsteered-baseline definition (paired, noise-controlled, states specified), which make our comparison stricter than what the paper appears to report and impossible to line up state-for-state.

## 6. Implementation difference table

| Component | Paper | GitHub | Our setup | Impact |
|---|---|---|---|---|
| Model | π0.5 | π0.5 `pi05_libero` (PyTorch hooks) | same | none |
| Checkpoint | `openpi-libero-2000` (step 2,000) | `brandonyang/openpi-libero-2000` | same, rev `aaeeabc7…` | none known; weight identity unverifiable |
| Fine-tune config | A.11.2 list | `pi05_libero` (equal) | not retrained | none |
| Dataset / suite | LIBERO-10, 10 tasks | `libero_10` | task 2 (+ task 3 pilot) | **coverage: 1 of 10 tasks** |
| LIBERO/robosuite version | not given | robosuite 1.4.0 pinned | robosuite 1.4.0, LIBERO `d63b117` | unknown; possible rendering/physics differences |
| Layer | KS3 oracle 5 (0-based, A.7.1) | default 11 | 5 | none (matches oracle) |
| α | 0.5 | default 0.1 | 0.5 | none |
| β | 0.1 (grid 0.1/0.3/0.5) | default 0.3; sweep 0.1/0.3 | 0.1 primary; 0.2 (off-grid), 0.3 | 0.2 is off-grid; otherwise none |
| Steering strategy | global 0.93 / per-step 0.87 / pos-only 0.80 (KS3) | all + random/linear; per-step α fixed 1.0 | global (+ shrinkage, random controls) | **high**: headline per-step untested; repo cannot run paper per-step configs |
| Hook semantics | layer output, every denoise step, all tokens | same | same | none |
| Fit samples | 15 rollouts; mean-pooled tokens | 15 rollouts; every token a row | 15 rollouts (states 0–14); V0 and V3 built | **medium**: V0 ≠ V3 by 58–100% Frobenius; behavior alike so far |
| Token pooling | mean over 10 tokens | none (D1) | V0 none, V3 pooled | medium (dominant construction difference) |
| Centering | mean-centered | uncentered (D2) | V0 uncentered, V3 centered | low–medium |
| Boolean AND | canonical pinv | non-canonical (D3) | V0 repo, V3 canonical | low–medium |
| Min class size | 3 | 2 | 8/7 | none for task 2 |
| Fit-set outcomes (KS3) | 8/7 | (released: 8/7) | 8/7 | none in counts; episodes differ (noise) |
| Test-set size | 30 | 15 (docs) | 15–30 | medium: CI width |
| Test states / seeds | not stated | window scheme (post-#48) | 0–14, 15–44, 45–49 | **high unknown**: cannot align to the paper |
| Pre-/post-PR #48 seeding | unknown | changed 2026-04-23 | post-#48 | **high unknown**: could imply fit/test init-state overlap in paper |
| Hyperparameter selection | best-of-240 on fit rollouts | small sweep on separate window; empty `best_configs.json` | none (paper's values) | medium: selection/winner's-curse not reproduced |
| Unsteered baseline | Table 1 Base = fit-set rate (inferred) | measured per window | paired, same states and noise | **high**: Δ definitions differ |
| Policy noise | not described | unseeded | paired (own instrumentation) | ours stricter; reduces variance for pairs |
| Random / shrinkage control (LIBERO) | none | `random_matched` exists | run in Phases 2B, 3A, 4A | ours extends the paper |
| Hardware | B200 latency table | not specified | RTX 4090 | low–medium (bf16 numerics, trajectory chaos) |
| Success label | final-step flag | first `done` | first `done` | none |

## 7. Impact assessment

Assessed for the question "why is KS3's 0.93 not reproduced, and is the COAST≈shrinkage finding an artifact of our setup?"

1. **Could the mismatch be the model/steering settings? Unlikely.** They match the paper's stated oracle for this task.
2. **Could it be the conceptor builder (repo vs paper)? Unlikely to be the whole story.** V3 (paper text) was tested at β 0.1 on the 15 development states in Phase 1D, where its success vector was identical to V0's on 15/15 states (both 11/15 against a baseline of 13/15), and offline both act as ≈ uniform `(1−β)` contraction (Phase 1B §6). Not tested: V3 at other β, on more states, or per-step.
3. **Could it be the evaluation protocol? Plausible and currently unresolvable.**
   - *Baseline definition:* a fit-set baseline (0.53) versus our test baselines (0.70 on 15–44; 0.60 pooled) changes Δ but does not by itself change the *steered* absolute rate, and the steered 28/30 versus our 20/30 (Fisher p = 0.021) is the larger gap.
   - *Test-set difficulty:* KS3 outcomes vary strongly by state (Phase 1D: 11 of 15 states flip with the noise seed alone). A different 30-state set could plausibly shift a success rate by several episodes, but a shift from ~0.67–0.73 to 0.93 is toward the edge of what state difficulty alone explains (our per-subset COAST rates at β 0.1–0.2 range 9/15 to 12/15).
   - *Selection:* per-task oracle configs are best-of-240 on the fit rollouts; a fixed config re-tested elsewhere regresses toward the mean. This affects the paper's *fit* number, not its test number, so it does not explain the test gap.
   - *Possible fit/test overlap under pre-#48 code:* if the paper's tests reused states 0–14, conceptors would be in-sample on half the test states. Our `P_fit` (in-sample states 0–14) gives 9/15 versus baseline 8/15, so overlap alone does not create a 0.93.
4. **Could it be the strategy? Not for KS3's global 0.93**, which is the entry we mirrored. It may matter for the paper's LIBERO *mean* (per-step 0.80), which we did not test.
5. **Is the COAST≈shrinkage finding an artifact of a repository deviation?** Evidence says probably not (point 2 and §5.4). But no paper artifact tests a pure-shrinkage control, so nothing in the paper contradicts or supports it. The paper's random control (Table 10) was not run on LIBERO, and on MetaWorld random ≈ baseline, whereas our LIBERO random ≈ COAST ≈ shrinkage; the benchmarks differ so this is not a contradiction.
6. **Refinement to Phase 4A §6.** It said "the baseline reproduces the paper's" (0.60 pooled vs 0.53). Given the Base-column finding, the accurate statement is: *our fit-state unsteered rate (8/15 = 0.53) reproduces the paper's Table 1/21 value; the paper's Base is probably a fit-set rate and is not comparable to our test-state baselines.* Phase 4A is otherwise unchanged; this audit does not rewrite it.

## 8. Reproduction confidence

| Confidence | Items |
|---|---|
| **HIGH** (paper and reproduction match, evidence in hand) | base model family and config `pi05_libero` (all hyper-parameters equal); checkpoint name, step 2,000, HF repo and revision consistency; task identity (KS3 = task 2); oracle config L5/α 0.5/β 0.1/global; layer indexing (0-based; A.7.1 lists `layers[i]`); hook site, gate formula, all-steps global application; activation schema `(D,L,S,1024)` with L = {0,5,11,17}; 10 denoising steps; fit N = 15 and split size; class counts 8/7; conceptor formula C = R(R + α⁻²I)⁻¹ and float64/float32 precision; task-2 `max_steps` treatment is not paper-specified but is the code default |
| **MEDIUM** (minor unknown differences) | weight identity of the checkpoint (no hash published); LIBERO/robosuite/MuJoCo versions (paper silent); token pooling, centering and Boolean AND (paper ≠ repo; both variants built and behave alike on 15 states); success labeling (final step vs first `done`, equivalent in LIBERO); GPU/bf16 numerics (B200 vs RTX 4090); β = 0.2 off-grid (only affects our extra conditions); class threshold 2 vs 3 (not triggered) |
| **LOW** (major missing information) | the paper's 30 test states and seeds; whether Table 1 "Base" is the fit-set rate (inferred, not stated); code version behind the paper's numbers (pre- or post-PR #48; which conceptor builder; `libero_conceptors.npz` inaccessible, HTTP 401); the hyper-parameter selection procedure on fit rollouts (which rollouts, how many, which noise); per-step strategy implementation (α up to 10, N < d after pooling); any pure-shrinkage or random control on LIBERO; whether the KS3 0.93 is a single-run number and how many runs were made |

Overall: **the reproduced setting is "same model, same task, same stated oracle configuration"; the missing pieces are all on the evaluation side.** Nothing found so far shows that our steering implementation is incorrect relative to the paper.

## 9. Remaining unknowns and draft questions for the authors

*Draft only. Nothing has been sent.* Suggested order of importance.

1. **Test set.** For the reported 30 held-out LIBERO-10 rollouts per task, which LIBERO init-state indices and which seeds were used, and did they exclude the 15 fitting states? Were the fit rollouts init states 0–14 (`--seed 0`)?
2. **Code version.** Were the LIBERO experiments run before or after PR #48 (2026-04-23, `--seed` began selecting init states)? Which commit/branch produced Tables 1 and 4?
3. **Baseline definition.** In Table 1 the LIBERO π0.5 "Base" equals the fit-set success rates of Table 21 for all 10 tasks. Is Base the 15 fit rollouts, or a separate 30-episode unsteered test run on the same states as the steered runs?
4. **Hyper-parameter selection.** How was "success on the fitting rollouts" obtained for each of the 240 configurations: steered rollouts on the 15 fit states, how many episodes per configuration, and with what noise control? Was the selected KS3 config (L5, α 0.5, β 0.1) re-evaluated on the 30 test states exactly once?
5. **Conceptor builder.** Were the reported conceptors built with the released token-flattened, uncentered builder (`compute_conceptors.py`) or with the mean-pooled, centered, pinv-AND pipeline of App. A.9.1? Is `brandonyang/libero-conceptors` (`libero_conceptors.npz`) public? (Anonymous access returns HTTP 401.)
6. **Per-step strategy.** The repository fixes per-step α at 1.0, but Table 4 lists per-step α = 10.0 (KS3) and up to 2.0 elsewhere. Which code produced these? How is the per-step covariance handled when N (≈ 385–728 inference steps) < d = 1024?
7. **Checkpoint.** Is `brandonyang/openpi-libero-2000` revision `aaeeabc72f8a…` (2026-04-09) the exact weights behind the reported results? Any EMA/conversion differences for the PyTorch path?
8. **Controls.** Were the random-eigenvector control (Table 10) or a uniform-shrinkage control `M = (1−β)I` run on LIBERO? At the selected β = 0.1 the gate changes h by ≈ 10% along h itself (our measurement); does the reported gain survive a shrinkage control?
9. **Run count and noise.** Are Table 1/4 numbers single runs per cell? Was policy sampling noise (`sample_noise`) seeded?
10. **Environment.** LIBERO, robosuite and MuJoCo versions, and GPU used for evaluation (B200 for latency; same for rollouts?). Any preprocessing beyond the 180° image rotation and 224×224 resize-with-pad?

## 10. Recommended next action

**Contact the authors** (questions 1–5 first), and in parallel do only *offline* work with the authors' public artifacts. Rationale:
- The unresolved items are all evaluation-side and cannot be settled from public sources; running more experiments before they are known risks repeating a setup that cannot be aligned with the paper.
- **Do not begin an improvement method yet.** The mechanistic finding (COAST ≈ shrinkage at L5) is real in our data and probably not due to repository deviations, but the paper's evaluation protocol, and therefore the claim we would be improving on, is not pinned down.
- **Offline options that need no LIBERO rollout** (suggested, not done): (a) rebuild V0/V3 conceptors from the authors' public KS3 fit activations and compare with ours (same style as Phase 1B), which separates "fit-data difference" from "builder difference"; (b) compute paper diagnostics (quota, overlap) on those activations against Tables 18/19.
- "Continue reproduction" is the fallback if the authors do not reply: the cheapest informative run would be V3 and random-matched controls at β 0.2–0.3 on states 15–44 (30 states), plus per-step steering, but only after the choices above are documented and preregistered.

## 11. Limits of this audit

- The paper text was read from the arXiv v1 HTML (tables flattened by conversion); numbers were cross-checked programmatically where used (Table 1 vs Table 21, Table 17, Table 4). Figures were not viewed.
- The Base-column identity is a numerical observation, not a stated protocol.
- The pre-#48 seeding effect on the paper is a possibility, not a finding.
- The Fisher tests compare different state sets and are reference values only.
- Other upstream branches and the authors' released activation dataset were only listed or sampled (30 small `metadata.json` files), not analyzed.
- No claim here is about whether COAST "works"; only about which parts of the paper's setting our runs did and did not reproduce.

## 12. Files

- Added: `research/reproduction/phase4b_paper_protocol_audit.md` (this report) and `research/reproduction/experiments/phase4b_paper_protocol_audit.yaml`.
- No source, test, config or record was modified; nothing was committed. Scratch copies of the paper text are outside the repository (session scratchpad).
