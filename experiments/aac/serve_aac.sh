#!/usr/bin/env bash
# Start a GR00T N1.5 AAC server (groot_env venv) on one GPU.
#
#   experiments/aac/serve_aac.sh <gpu> <port> [num_samples] [extra serve.py flags...]
#
# Defaults reproduce the official AAC server (N=20, reseed 42 per call,
# move_th 3.0, 4 denoising steps). Logs go to stdout; redirect as needed.
set -euo pipefail
GPU=${1:?gpu}
PORT=${2:?port}
N=${3:-20}
shift $(( $# >= 3 ? 3 : $# ))
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO/groot_env"
export CUDA_VISIBLE_DEVICES="$GPU"
exec uv run --frozen python serve.py \
    --model-path ../checkpoints/groot_n15/gr00t_n1-5/multitask_learning/checkpoint-120000 \
    --device cuda:0 --port "$PORT" --denoising-steps 4 \
    --aac --aac-num-samples "$N" "$@"
