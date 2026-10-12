# Adaptive Action Chunking (AAC) reproduction

This directory reproduces Liang et al., *Adaptive Action Chunking at Inference-time for VLA Models* (CVPR 2026, arXiv:2604.04161) on COAST's GR00T N1.5 + RoboCasa stack. The verdict is in **[`AAC_REPRO_STATUS.md`](AAC_REPRO_STATUS.md)**.

## Documents

| File | Contents |
|---|---|
| `AAC_REPRO_STATUS.md` | Final status, evidence for questions A–H, result tables, commands |
| `AAC_REPRO_AUDIT.md` | Audit of the pre-AAC COAST GR00T/RoboCasa path |
| `AAC_UPSTREAM_MAPPING.md` | Paper ↔ official code ↔ COAST mapping; pinned SHAs; discrepancies |
| `AAC_ACTION_MAPPING.md` | GR00T 12-D / 32-D action dims → AAC entropy groups |
| `AAC_BASELINE_IDENTITY.md` | Non-AAC inference unchanged (bit-identical); multi-sample checks; latency vs N |
| `AAC_MECHANISM_VALIDATION.md` | Mechanism criteria; what AAC actually does on this checkpoint |
| `AAC_REPRO_NOTES.md` | PAPER / CODE / ADAPTATION / UNRESOLVED; protocol sensitivity; RoboCasa nondeterminism |
| `AAC_REPRO_PREREG.md` | Frozen Phase B design, plus Amendment 1 (made before any Phase B data) |

## Code

| Path | Role |
|---|---|
| `groot_env/aac.py` | Verbatim port of the upstream selector (`select_chunk_size`), plus the `aac_select` diagnostics wrapper |
| `groot_env/groot_adapter.py` | `_get_action_multi_sample` / `infer_multi_sample`: N candidates from one backbone pass |
| `groot_env/groot_aac.py`, `groot_env/serve.py --aac` | Server mode: N samples → selector → candidate 0 + `aac_horizon` + diagnostics |
| `examples/robocasa_env/main.py` | `--execution_mode {fixed,aac}`, `--seed_per_episode`, `--start_episode`, `--log_dir` (JSONL), `--no-save_video` |
| `serve_aac.sh` | Start an AAC server on one GPU (upstream defaults; pass `--aac-seed None` for a running RNG) |
| `run_arms.sh` | Run fixed-h / AAC arms with exact per-task resume, manifests and code snapshots |
| `analyze_aac.py` | Success with Wilson CIs, paired McNemar/bootstrap, AAC decision statistics, plots |
| `validate_multisample.py` | GPU identity / diversity / latency checks |
| `make_upstream_parity_fixtures.py` | Runs the **official** selector to generate `groot_env/tests/fixtures/aac_upstream_parity.npz` |
| `write_manifest.py`, `plot_latency.py` | Provenance manifest; latency/VRAM vs N plots |

Upstream clones for parity live in `external/aac/` (gitignored):

```bash
git clone https://github.com/Adaptive-Action-Chunking/robocasa external/aac/aac-robocasa              # fed3e6b
git clone https://github.com/Adaptive-Action-Chunking/gr00t-multi-sample external/aac/gr00t-multi-sample  # 11e926b
```

Run outputs go to `results/aac/<run>/` (gitignored). Each run directory holds:
- `manifest.json`, which pins the git SHA, a worktree hash, upstream SHAs, the checkpoint, server metadata and package versions;
- `code_snapshot.*`;
- `arms/<arm>/logs/<task>/{episodes,inference}.jsonl`;
- `summary.json`, `report.md` and `plots/`.
