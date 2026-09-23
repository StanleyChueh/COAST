# Environment Snapshot — COAST Reproduction, Phase 0

**Paper:** Contrastive Conceptor Activation Steering (COAST):
Unlocking Vision-Language-Action Models through Hidden States

**arXiv:** 2605.17144 (https://arxiv.org/abs/2605.17144)
Authors: Miranda Muqing Miao, Subin Kim, Brandon Yang, Lyle Ungar

**Phase:** 0 — environment setup + minimal unsteered LIBERO baseline smoke test.
The 3-episode smoke test is an engineering check only; it is **not** evidence that the paper has been reproduced.

## Repository

| Item | Value |
|---|---|
| Date | 2026-09-23 (timezone +08:00) |
| Local repo path | `/home/stanley/Stanley_ws/COAST-research/COAST` |
| origin | https://github.com/StanleyChueh/COAST.git |
| upstream | https://github.com/COAST-VLA/COAST.git |
| Branch | `repro/arxiv-2605-17144` |
| Git commit | `2afa10ee256a3b3edfeb56500fea166a0837f119` ("Format README rollout videos (#54)") |
| Git status at start | clean (`nothing to commit, working tree clean`) |
| Git status after Phase 0 | only the new untracked `research/` directory (checkpoints/, .venv/, output/ are gitignored) |

### Submodules (initialized with `git submodule update --init --recursive`; all at pinned commits, none changed)

| Path | Commit | Describe |
|---|---|---|
| `third_party/libero` | `d63b117ffd512be05f333a6609767b2ad8c30bae` | heads/master ("added init file for uv to work", 2026-01-24) — URL https://github.com/Robot-VLA/LIBERO.git |
| `third_party/robosuite` | `aaa8b9b214ce8e77e82926d677b4d61d55e577ab` | v1.5.2-5-gaaa8b9b2 |
| `third_party/robocasa` | `9a3a78680443734786c9784ab661413edb87067b` | v1.0-12-g9a3a786 |
| `third_party/Isaac-GR00T` | `4af2b622892f7dcb5aae5a3fb70bcb02dc217b96` | n1.5-release |
| `third_party/aloha` | `d1dc83afd89ded4379851257fe5d85632d31d5ec` | d1dc83a |

Note: the LIBERO venv installs robosuite **1.4.0 from PyPI** (pinned in `examples/libero_env/pyproject.toml`), not the `third_party/robosuite` submodule (v1.5.2+), which is used by RoboCasa.

## Host

| Item | Value |
|---|---|
| OS | Ubuntu 22.04.5 LTS (jammy) |
| Kernel | Linux 6.8.0-138-generic #138~22.04.1-Ubuntu SMP PREEMPT_DYNAMIC x86_64 |
| CPU | 13th Gen Intel Core i7-13700 (24 logical CPUs) |
| RAM | 62 GiB (+2 GiB swap) |
| GPUs | 2 × NVIDIA GeForce RTX 4090 |
| GPU VRAM | 24564 MiB each |
| NVIDIA driver | 580.178.04 |
| CUDA reported by nvidia-smi | 13.0 |
| Physical GPU selected | **GPU 0** (Bus 00000000:01:00.0). Both GPUs were essentially idle (<0.5 GiB used); GPU 1 drives the desktop (Xorg/gnome-shell), so GPU 0 was chosen. |
| CUDA_VISIBLE_DEVICES | `0` (server and client) |
| Disk (repo filesystem) | 3.7 T, ~3.2 T free at start |
| git | 2.34.1 (git-lfs **not installed**; not needed — all installs used `GIT_LFS_SKIP_SMUDGE=1` per README) |

## Tooling

| Item | Value |
|---|---|
| uv | 0.12.18 — **was not installed**; installed via the official Astral standalone installer into `~/.local/bin` (user-level, already on PATH). No system/conda installs. The repo has no uv version pin; CI uses `astral-sh/setup-uv@v7`; both lockfiles are `revision = 3`. |
| Python interpreters | uv-managed CPython 3.11.16 (root) and 3.8.20 (libero_env). Conda base (Python 3.14) was not used or modified. |

## Root COAST environment (`.venv`, repo root)

Installed exactly per README: `GIT_LFS_SKIP_SMUDGE=1 uv sync` then `GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .` (both exit 0).

| Item | Value |
|---|---|
| Python | 3.11.16 |
| PyTorch | 2.7.1+cu126 |
| torch.version.cuda | 12.6 |
| cuDNN | 90501 |
| torch.cuda.is_available() | True |
| Visible GPUs (CUDA_VISIBLE_DEVICES=0) | 1 × NVIDIA GeForce RTX 4090 |
| JAX | 0.5.3 (jax[cuda12]); devices: `[CudaDevice(id=0)]` |
| flax | 0.10.2 |
| transformers | 4.53.2 (+ openpi `transformers_replace` patch, auto-applied into `.venv` on first PyTorch model import) |
| orbax-checkpoint | 0.11.13 |
| numpy | 1.26.4 |
| openpi-client | 0.1.0 (editable workspace package) |
| huggingface_hub | 0.32.3 |

## LIBERO environment (`examples/libero_env/.venv`)

Installed per `examples/libero_env/README.md`: `cd examples/libero_env && uv sync` (exit 0).

| Item | Value |
|---|---|
| Python | 3.8.20 |
| MuJoCo | 3.2.3 |
| robosuite | 1.4.0 |
| LIBERO | editable from `third_party/libero` @ `d63b117` — imports OK |
| numpy | 1.22.4 |
| torch (client venv, unused for inference) | 1.11.0+cu113 |
| imageio | 2.35.1 |
| openpi-client | 0.1.0 |
| Rendering backend | `MUJOCO_GL=egl` works (offscreen 256×256 agentview + wrist images rendered on GPU 0) |
| LIBERO config | `LIBERO_CONFIG_PATH=/home/stanley/Stanley_ws/COAST-research/libero_config_coast` (see deviation below) |
| bddl_files / init_states / assets | all resolve to `third_party/libero/libero/libero/{bddl_files,init_files,assets}` — exist |
| libero_10 init states | 50 per task (task 0 array shape (50, 123)) |
| `datasets` path | does not exist (not needed for evaluation; LIBERO prints a harmless warning) |

### Deviation: LIBERO config location

`examples/libero_env/README.md` says to run `uv run python setup_libero_config.py`, which writes `~/.libero/config.yaml`.
That file is global to the user account, and `~/Stanley_ws/lerobot` also imports `libero.libero`, which reads the same default path.
To keep the two projects isolated, `~/.libero` was **not** created. Instead, the config was generated with the official script's own
`setup_libero_config.build_config_text()` (identical content) and written to
`/home/stanley/Stanley_ws/COAST-research/libero_config_coast/config.yaml`, outside the git repo. LIBERO's supported
`LIBERO_CONFIG_PATH` environment variable points to it (see `third_party/libero/libero/libero/__init__.py`).
Every LIBERO client command must therefore export `LIBERO_CONFIG_PATH=/home/stanley/Stanley_ws/COAST-research/libero_config_coast`.

## Checkpoint

| Item | Value |
|---|---|
| Model / config | pi0.5, `pi05_libero` |
| Checkpoint repository | `brandonyang/openpi-libero-2000` (Hugging Face model repo) |
| Checkpoint revision | `aaeeabc72f8a50a8fa2d04544332c8ec1cd0142e` (lastModified 2026-04-09) |
| Local path | `checkpoints/openpi-libero-2000` (gitignored) |
| Files downloaded | `params/*`, `assets/*`, `_CHECKPOINT_METADATA` only (17 files) |
| Download size | 12,440,614,554 bytes params + 1,922 bytes norm stats (~12.4 GB); verified against HF API sizes |
| Not downloaded | `train_state/*` (~32.3 GB optimizer state; not used for inference — `policy_config.create_trained_policy` reads only `params/` and `assets/`) |
| Derived files | `model.safetensors` (7,233,650,408 bytes, bf16), `config.json`, `.pytorch_conversion_hash` — written by `ensure_pytorch_checkpoint()` on first `--pytorch` serve |
| Why this checkpoint | `experiments/libero/README.md`, `run_end_to_end.sh`, `find_best_configs.py` and steering tests all default to `checkpoints/openpi-libero-2000`; paper App. A.8.3/A.11.2 names step 2,000 (`/openpi-libero-2000`) as the checkpoint used |
| Download command | `HF_HOME=$PWD/checkpoints/.hf_home uvx --from huggingface_hub hf download brandonyang/openpi-libero-2000 --revision aaeeabc72f8a50a8fa2d04544332c8ec1cd0142e --include "params/*" --include "assets/*" --include "_CHECKPOINT_METADATA" --local-dir checkpoints/openpi-libero-2000` |
| Other caches | PaliGemma tokenizer (4.07 MiB) auto-downloaded by openpi to `~/.cache/openpi/big_vision/paligemma_tokenizer.model` (openpi default) |

## Problems encountered during setup

1. **uv not installed.** Installed uv 0.12.18 at user level (see Tooling).
2. **Slow network.** The submodule clone took 30.6 min; root `uv sync` took 3 h 15 min (PyPI CDN at 0.1–2.6 MB/s for the torch/nvidia wheels). No action taken; downloads completed.
3. **LIBERO `uv sync` first attempt failed** with `Failed to acquire lock on the distribution cache ... Timeout (300s)` on the editable `openpi-client` sdist. This was self-inflicted: it ran concurrently with the root `uv sync`, which was building the same editable package. Fix: re-ran the identical `uv sync` after the root sync finished → exit 0.
4. **First `hf download` invocation fetched only `assets/` + metadata**: this `hf` CLI version takes one pattern per `--include`. Fix: repeated `--include` flags.
5. **First `serve_policy.py --pytorch` launch failed** during JAX→PyTorch conversion:
   `ValueError: Unrecognized configuration class <class 'transformers.models.gemma.configuration_gemma.GemmaConfig'> for this kind of AutoModel: AutoModel.`
   Root cause: on the first run `_ensure_transformers_patched()` (`src/openpi/models_pytorch/pi0_pytorch.py`) copies
   `transformers_replace/` into `.venv` and then deletes `transformers.models.{gemma,paligemma,siglip}*` from `sys.modules`. The re-imported
   `GemmaConfig` becomes a new class object that no longer matches the class held by transformers' AutoModel registry.
   Evidence: after the failed run all 5 patch files are byte-identical in `.venv`, and in a fresh process `AutoModel.from_config(GemmaConfig(...))` succeeds.
   Fix: re-ran the identical server command. **No code changed.** This is a first-run-only repository bug that anyone doing a fresh install will hit once.
