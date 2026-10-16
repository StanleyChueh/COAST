#!/usr/bin/env bash
# Stage-5 pilot launcher: intervention servers + resumable fork-client workers.
#
#   bash research/adaptive_inference/run_pilot.sh            # from the repo root
#
# Env overrides: OUT, SERVERS ("gpu:port ..."), WORKERS, TASKS, INITS, MASTERS, ARMS.
# Never edit this file while it runs (bash reads scripts incrementally).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
OUT="${OUT:-results/adaptive_inference/pilot}"
SERVERS="${SERVERS:-1:8600 1:8601 0:8602 0:8603}"
WORKERS="${WORKERS:-8}"
TASKS="${TASKS:-0 2 3 7}"
INITS="${INITS:-15 16 17 18 19 20 21 22 23 24}"
MASTERS="${MASTERS:-2001,2002,2003,2004}"
ARMS="${ARMS:-I0,I0P,IH,IL,IC}"
LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-$ROOT/../libero_config_coast}"
export LIBERO_CONFIG_PATH MUJOCO_GL=egl
mkdir -p "$OUT/logs"

# 1. servers (skip ports already listening)
read -r -a SRV <<< "$SERVERS"
for s in "${SRV[@]}"; do
  gpu="${s%%:*}"; port="${s##*:}"
  if (exec 3<>"/dev/tcp/127.0.0.1/$port") 2>/dev/null; then echo "port $port already up"; continue; fi
  CUDA_VISIBLE_DEVICES="$gpu" nohup uv run --frozen python research/adaptive_inference/intervention_server.py \
    --port "$port" > "$OUT/logs/server_$port.log" 2>&1 &
  echo "$!" > "$OUT/logs/server_$port.pid"
  echo "started server gpu=$gpu port=$port pid=$!"
done
for s in "${SRV[@]}"; do
  port="${s##*:}"
  for _ in $(seq 1 120); do
    (exec 3<>"/dev/tcp/127.0.0.1/$port") 2>/dev/null && break
    sleep 5
  done
done

# 2. units, interleaved across tasks so every worker sees every task
UNITS=()
for i in $INITS; do for t in $TASKS; do UNITS+=("$t:$i"); done; done
declare -a WUNITS
for idx in "${!UNITS[@]}"; do
  w=$(( idx % WORKERS ))
  WUNITS[$w]="${WUNITS[$w]:-}${WUNITS[$w]:+,}${UNITS[$idx]}"
done

# 3. workers
PIDS=()
for w in $(seq 0 $((WORKERS - 1))); do
  s="${SRV[$(( w % ${#SRV[@]} ))]}"
  gpu="${s%%:*}"; port="${s##*:}"
  (
    cd examples/libero_env
    MUJOCO_EGL_DEVICE_ID="$gpu" .venv/bin/python ../../research/adaptive_inference/fork/libero_fork.py \
      --out "../../$OUT" --port "$port" --units "${WUNITS[$w]}" \
      --branch_masters "$MASTERS" --arms "$ARMS" >> "../../$OUT/logs/worker_$w.log" 2>&1
  ) &
  PIDS+=("$!")
  echo "worker $w port=$port units=${WUNITS[$w]} pid=$!"
done
rc=0
for p in "${PIDS[@]}"; do wait "$p" || rc=1; done
echo "all workers finished rc=$rc $(date -Iseconds)" | tee -a "$OUT/logs/run.log"
