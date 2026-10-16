# Research status: intervention-response routing for frozen VLAs

Date: 2026-10-16. Branch `adaptive-inference-2027`. **Commit `4239b15eeb861e644c4d10da0ea8b928888e866c`**. All work below is in the uncommitted `research/adaptive_inference/` tree; no tracked file was modified.

Sibling repo `~/Stanley_ws/VLA-steering-2027` was read only, @ `de650cb4cde090bcaf0709302dd1365dc821d97a`.

## Question and answer

> At the same VLA state, do different inference-time interventions rescue different failures often enough that choosing among them matters?

**Not at a level that is detectable or exploitable at pilot scale.** On π0.5 + LIBERO-10 (306 restored decision states × 5 arms × 4 noise replicates):
- a noise-honest state-level oracle loses to always applying the single best operator;
- no operator rescues failures uniquely more often than simply re-sampling the base policy;
- whatever operator heterogeneity is exploitable is between tasks, not within them.

The same was found retrospectively for execution horizon on GR00T N1.5 + RoboCasa (Stage 1).

## Stage results

| stage | deliverable | result |
|---|---|---|
| 0 | [`LITERATURE_2025_2026.md`](LITERATURE_2025_2026.md), [`NOVELTY_BOUNDARY.md`](NOVELTY_BOUNDARY.md) | All 23 listed arXiv IDs verified against the API (PICES title corrected; MoH 2511.19433, OneTwoVLA 2505.11917 and MechInterp 2509.00328 added). New neighbours found: ELASTIC 2606.31132, ChunkTrust 2609.39754, Mostly Harmless 2606.12299 (improvement head). **No collision** with "online state-level routing among ≥ 2 heterogeneous automatic VLA operators with operator-specific counterfactual utility". The stop condition did not trigger |
| 1 | [`HORIZON_ORACLE_REPORT.md`](HORIZON_ORACLE_REPORT.md) (plan frozen in `HORIZON_ORACLE_PLAN.md`) | Phase-B numbers verified exactly from raw logs. Matched set 905 scenes (the 5 sink tasks fail scene-hash pairing). Per-scene oracle 71.4% vs fixed16 51.0% → **+20.3 pp [17.8, 22.7], raw gate HIGH**. But two replicates of the *same* h disagree on 22–26% of scenes, the same as different h. Δ oracle gain (different h − same h) = −1.5 pp [−5.5, +3.0]. A null with no scene × h interaction predicts a 73.5% 5-draw oracle (observed 71.4%). Cross-fitted per-task h choice: 47.0% < fixed16 51.0%. **Temporal routing: LOW priority (not distinguishable from rollout noise)** |
| 2 | [`COMMON_BACKBONE_AUDIT.md`](COMMON_BACKBONE_AUDIT.md) | **π0.5 (`openpi-libero-2000`, PyTorch) + LIBERO-10.** It is the only stack with temporal, latent (COAST) and semantic (CAG) operators on one frozen model, exactly restorable simulation and keyed noise. GR00T/RoboCasa has no CAG-like operator or conceptors and its resets are non-deterministic |
| 3 | [`INTERVENTION_SPEC.md`](INTERVENTION_SPEC.md) | I0 base (h = 5); I0P resampling placebo; IH h = 2; IL COAST global (paper Table-4 configs, released builder on the authors' fit activations); IC CAG-TF ω = 1.5 (PyTorch port). I0 path is **bitwise** equal to stock `Policy.infer`; IL β = 0 is bitwise equal to I0; IC ω = 1 is within 6e-8 of I0 |
| 4 | [`COUNTERFACTUAL_FORK_VALIDATION.md`](COUNTERFACTUAL_FORK_VALIDATION.md) | Exact restore = `mjSTATE_INTEGRATION` + robosuite env/robot/controller/**gripper**/observable state + stored observation: 16/16 snapshots bit-exact for 300 steps. **LIBERO's own `set_init_state` restore diverges on step 1** (it drops warm-start, gripper state and the step counter) |
| 5 | [`PILOT_PREREG.md`](PILOT_PREREG.md) → [`PILOT_GATE.md`](PILOT_GATE.md) | **NO-GO** (A ✗, B ✗, C ✓, D ✗, E ✓) |

## Key numbers (pilot)

| | value |
|---|---|
| always-on success | I0 41.3%, I0P 43.1%, IH 39.2%, **IL 48.0%**, IC 45.0% |
| **oracle headroom** (LOO state oracle − best always-on) | **−1.9 pp** [−4.9, +1.1] (gate A needs ≥ +5) |
| LOO task oracle vs state oracle | 50.4% vs 46.1% → **−4.3 pp** [−8.2, −0.3] (gate D needs ≥ +2.5) |
| naive single-draw oracle | 66.3% (+18 pp). Base + its own re-sample alone: 55.3% (+14 pp) → mostly noise |
| **unique rescue rates** (share of salvageable pairs) | among interventions: IH 11.4%, IL 14.4%, IC 22.5%. **With the placebo competing: IH 6.3, IL 9.5, IC 11.2 vs placebo 11.8%** |
| rescue share | IH 27%, IL 34%, IC 38% (no dominance) |
| harm (base ✓ → arm ✗) | IL 18%, IC 29%, IH 32%, placebo 30% |
| risk vs winner | winners differ within risk bins (χ² p = 4e-9, largely the trivial p̂ = 1 bin), but the winners are not reproducible across replicates. RL²-style failure gate 45.0% < always-IL |
| features → Δ | all \|Spearman ρ\| ≤ 0.17. Uncertainty proxies (sample spread, CAG contrast) ≈ 0 |
| post-hoc heterogeneity | operator-vs-operator within-task τ ≈ 0.14–0.20 (IC−IH and IL−IH CIs exclude 0). The rough implied perfect-knowledge gain is +7.9 pp over always-IL, +4.7 pp over task-best: real, but below per-state noise at R = 4 and not feature-predictable |
| sensitivity (tasks 0/3/7, exactly reproducible) | A −2.7 pp [−6.3, +0.8]; D −6.4 pp [−10.0, −1.5]; same verdict |

## Infrastructure validity

- Identity branch: **303/306** decision states reproduce the reference continuation bitwise (tasks 0/3/7: 235/235).
- Restore state and observation hashes: **6,426/6,426**. Exceptions: **0**.
- The 3 failures (KS3 only) are rare **EGL render non-determinism**: one wrist-camera pixel, one intensity level, after the stove flame appears; reproduced in the reference itself.
- A latent `MjModel.site_rgba` staleness (one step, never a policy input for h ≥ 2) has a hardened v2 snapshot ready; it is required before any h = 1 operator.
- Provenance:
  - Preregistered files and analysis code are frozen with sha256 and verified after the run.
  - One incident: mid-run, a `ruff --fix` rewrote three frozen files on disk (running processes unaffected). Their exact bytes were rebuilt by replaying the edit history until all hashes matched. All 8 worker manifests record the frozen hashes. A local `ruff.toml` now keeps lint off those files.
  - Raw results are archived at `results/adaptive_inference/pilot_archive_20261016.tgz` (sha256 `5151df7e…ea63`), with per-file checksums.

## Nearest literature collisions, and whether the novelty claim survives

- **Nearest:**
  - RL²-VLA (failure-gated single operator);
  - Mostly Harmless (benefit head for one language operator);
  - ELASTIC (state-dependent choice among compute amounts);
  - Reuse Before You Retrieve (episode-level resampling-vs-retrieval headroom);
  - Act/Think/Abstain and UPS (routing on complexity / uncertainty).
- **The formal idea survives the literature check** (nobody routes heterogeneous operators on same-state utility).
- **The empirical claim does not survive in this pilot:**
  - kill condition 6 (noise ≡ complementarity) holds for operator-vs-base contrasts;
  - kill condition 3 (task ≡ utility) holds for what is exploitable.
- The one open loophole is the post-hoc within-task operator-vs-operator variance. It would need confirmation **and** a pre-intervention predictor, and no current feature has one.
- Our own prior evidence points the same way:
  - COAST's held-out effect ≈ shrinkage;
  - CAG's rescues were task-concentrated (CR0: task prior AUROC 0.86);
  - horizon effects on GR00T are noise.

## Files

New (untracked; nothing committed; no tracked file modified):

| path | contents |
|---|---|
| `research/adaptive_inference/*.md` | 11 reports: LITERATURE, NOVELTY_BOUNDARY, HORIZON_ORACLE_PLAN/REPORT, COMMON_BACKBONE_AUDIT, INTERVENTION_SPEC, COUNTERFACTUAL_FORK_VALIDATION, PILOT_PREREG, PILOT_GATE, RESEARCH_STATUS |
| `analyze_horizon_oracle.py` | Stage-1 zero-GPU analysis |
| `intervention_server.py` | π0.5 I0/IL/IC operator server with keyed noise |
| `validate_operators.py` | operator identity checks |
| `build_pilot_conceptors.py` | IL conceptors via the released builder |
| `fork/sim_snapshot.py`, `fork/libero_fork.py` | fork harness |
| `fork/probe_restore.py`, `fork/debug_identity.py`, `fork/dump_obs.py` | validation tools |
| `run_pilot.sh` | pilot launcher |
| `analyze_pilot.py` (frozen), `sensitivity_pilot.py`, `heterogeneity_pilot.py` | pilot analyses |
| `tests/test_analyze_pilot.py`, `ruff.toml` | tests, lint config |

Outside `research/`, all gitignored:
- `conceptors/aiv_pilot_conceptors.npz`;
- `results/adaptive_inference/`;
- `research/adaptive_inference/results/horizon_oracle/`.

Memory notes are in `~/.claude/.../memory/`.

## Tests and checks run

- `uv run pytest research/adaptive_inference/tests -q` → **5 passed**. Synthetic checks:
  - under the null the LOO gain is ≈ 0 while the naive oracle is inflated;
  - state- vs task-specific heterogeneity is recovered;
  - tie order;
  - risk gate.
- `fork/probe_restore.py` on tasks 0, 2, 3, 7 → 16/16 exact (v1), LIBERO-native 0/16.
- `validate_operators.py` (GPU 1) → all identity checks pass (`results/adaptive_inference/validation/operators.json`).
- End to end: identity branches 303/306; smoke runs 5/5.
- `uv run ruff check research/adaptive_inference/` → clean. `ruff format --check` → clean (frozen files excluded by `ruff.toml`).

## Exact commands

```bash
uv run python research/adaptive_inference/analyze_horizon_oracle.py                       # Stage 1 (CPU, ~6 s)
CUDA_VISIBLE_DEVICES="" uv run python research/adaptive_inference/build_pilot_conceptors.py
CUDA_VISIBLE_DEVICES=1 uv run python research/adaptive_inference/validate_operators.py     # needs obs_samples.npz (fork/dump_obs.py)
cd examples/libero_env && MUJOCO_GL=egl LIBERO_CONFIG_PATH=$PWD/../../../libero_config_coast \
  .venv/bin/python ../../research/adaptive_inference/fork/probe_restore.py --task_id 2 --n_steps 300
bash research/adaptive_inference/run_pilot.sh                                               # ~7 h on 2x4090
uv run python research/adaptive_inference/analyze_pilot.py                                  # gate (frozen)
uv run python research/adaptive_inference/sensitivity_pilot.py                              # post hoc
uv run python research/adaptive_inference/heterogeneity_pilot.py                            # post hoc
```

## Single recommended next experiment

**Stop the router direction. If anything is run, make it one confirmatory falsification test of the post-hoc loophole, with the kill rule fixed now.**

- **Design:**
  - Re-use the frozen harness unchanged (v2 snapshot only if an h = 1 arm is added).
  - Fresh decision states: same 4 tasks, reference noise master 1001 on inits 15–24 (states 25–49 stay reserved).
  - About 80 states, arms {I0, IL, IC, IH}, **R = 12** replicates each (about 3,840 branches ≈ one pilot's compute).
- **Primary statistic:** the split-half honest state oracle (choose on 6 replicates, score on the other 6) minus the per-task-best arm.
- **Kill rule:** < 2.5 pp, **or** a cross-fitted predictor from the pre-intervention features (latent + chunk statistics, leave-one-unit-out) recovers < 50% of it → the direction is closed for good.
- **Only if both pass** would a router study (Stage 6) be justified, and only then on a second backbone.

NO-GO — insufficient intervention complementarity/headroom.
