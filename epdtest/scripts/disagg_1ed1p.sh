#!/bin/bash
set -euo pipefail

# 1ed1p topology:
# - one prefill worker (P)
# - one combined encoder+decode worker (ED)

declare -a PIDS=()

MODEL="${MODEL:-Qwen/Qwen2.5-VL-3B-Instruct}"
LOG_PATH="${LOG_PATH:-./epdtest/logs}"
mkdir -p "$LOG_PATH"
BENCHMARK="${BENCHMARK:-randommm}"

PREFILL_PORT="${PREFILL_PORT:-19535}"
ENCODE_DECODE_PORT="${ENCODE_DECODE_PORT:-19536}"
PROXY_PORT="${PROXY_PORT:-10001}"

GPU_ED="${GPU_ED:-${GPU_D:-${GPU_E:-1}}}"
GPU_P="${GPU_P:-${GPU_PD:-0}}"
ALLOW_SHARED_GPU="${ALLOW_SHARED_GPU:-0}"

EC_SHARED_STORAGE_PATH="${EC_SHARED_STORAGE_PATH:-/tmp/ec_cache}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-12000}"

NUM_PROMPTS="${NUM_PROMPTS:-300}"
BENCH_REQUEST_RATE="${BENCH_REQUEST_RATE:-32}"
BENCH_MAX_CONCURRENCY="${BENCH_MAX_CONCURRENCY:-32}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-${PD_MAX_MODEL_LEN:-65536}}"
P_MAX_NUM_BATCHED_TOKENS="${P_MAX_NUM_BATCHED_TOKENS:-${PD_MAX_NUM_BATCHED_TOKENS:-32768}}"
P_MAX_NUM_SEQS="${P_MAX_NUM_SEQS:-${PD_MAX_NUM_SEQS:-32}}"
ED_MAX_NUM_BATCHED_TOKENS="${ED_MAX_NUM_BATCHED_TOKENS:-${PD_MAX_NUM_BATCHED_TOKENS:-32768}}"
ED_MAX_NUM_SEQS="${ED_MAX_NUM_SEQS:-${PD_MAX_NUM_SEQS:-32}}"
P_GPU_MEMORY_UTILIZATION="${P_GPU_MEMORY_UTILIZATION:-${PREFILL_GPU_MEMORY_UTILIZATION:-0.85}}"
ED_GPU_MEMORY_UTILIZATION="${ED_GPU_MEMORY_UTILIZATION:-${DECODE_GPU_MEMORY_UTILIZATION:-0.85}}"
NIXL_BASE_PORT="${NIXL_BASE_PORT:-$((5200 + ($$ % 1000)))}"
PREFILL_NIXL_SIDE_CHANNEL_PORT="${PREFILL_NIXL_SIDE_CHANNEL_PORT:-$NIXL_BASE_PORT}"
ED_NIXL_SIDE_CHANNEL_PORT="${ED_NIXL_SIDE_CHANNEL_PORT:-$((NIXL_BASE_PORT + 1000))}"

VISUAL_TOKEN_PRUNING_METHOD="${VISUAL_TOKEN_PRUNING_METHOD:-}"
VISUAL_TOKEN_PRUNING_RATE="${VISUAL_TOKEN_PRUNING_RATE:-}"
VISION_ZIP_DOMINANT_RATIO="${VISION_ZIP_DOMINANT_RATIO:-}"
VISION_ZIP_ATTENTION_LAYER="${VISION_ZIP_ATTENTION_LAYER:-}"
IMAGES_PER_REQ="${IMAGES_PER_REQ:-1}"
HF_DATASET_PATH="${HF_DATASET_PATH:-lmarena-ai/VisionArena-Chat}"
METRICS_SAMPLING_INTERVAL_SECONDS="${METRICS_SAMPLING_INTERVAL_SECONDS:-1}"

usage() {
    cat <<'EOF'
Usage:
  bash epdtest/scripts/disagg_1ed1p.sh
EOF
}

if [[ $# -gt 0 ]]; then
    if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
        usage
        exit 0
    fi
    echo "This script does not accept positional arguments." >&2
    usage >&2
    exit 1
fi

case "${BENCHMARK,,}" in
    simple|randommm)
        ;;
    *)
        echo "Unsupported BENCHMARK: $BENCHMARK (expected simple or randommm)" >&2
        exit 1
        ;;
esac

if [[ "$GPU_ED" == "$GPU_P" && "$ALLOW_SHARED_GPU" != "1" ]]; then
    echo "1ed1p expects distinct GPUs: GPU_ED=$GPU_ED and GPU_P=$GPU_P." >&2
    echo "Set ALLOW_SHARED_GPU=1 only if you intentionally want to overlap them." >&2
    exit 1
fi

export UCX_TLS=all
export UCX_NET_DEVICES=all

ulimit -n "${ULIMIT_NOFILE:-65535}" >/dev/null 2>&1 || true

GIT_ROOT=$(git rev-parse --show-toplevel)

START_TIME=$(date +"%Y%m%d_%H%M%S")
RUN_DIR="${RUN_DIR:-$LOG_PATH/$START_TIME}"
mkdir -p "$RUN_DIR"
P_LOG="$RUN_DIR/prefill.log"
ED_LOG="$RUN_DIR/encoder_decode.log"
PROXY_LOG="$RUN_DIR/proxy.log"

wait_for_server() {
    local port=$1
    timeout "$TIMEOUT_SECONDS" bash -c "
        until curl -s localhost:$port/v1/chat/completions > /dev/null; do
            sleep 1
        done" && return 0 || return 1
}

port_in_use() {
    local port=$1
    ss -ltn 2>/dev/null | awk '{print $4}' | grep -Eq "[:.]${port}$"
}

log_gpu_sm_utilization() {
    local ts=$1
    local output
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        return 0
    fi
    output=$(nvidia-smi \
        --query-gpu=index,utilization.gpu,utilization.memory,memory.used,power.draw \
        --format=csv,noheader,nounits \
        -i "$GPU_ED,$GPU_P" 2>/dev/null || true)
    if [[ -z "$output" ]]; then
        return 0
    fi
    while IFS=',' read -r gpu_index sm_util mem_util mem_used power_draw; do
        gpu_index=$(echo "$gpu_index" | xargs)
        sm_util=$(echo "$sm_util" | xargs)
        mem_util=$(echo "$mem_util" | xargs)
        mem_used=$(echo "$mem_used" | xargs)
        power_draw=$(echo "$power_draw" | xargs)

        local -a roles=()
        if [[ "$gpu_index" == "$GPU_ED" ]]; then
            roles+=("encoder_decode")
        fi
        if [[ "$gpu_index" == "$GPU_P" ]]; then
            roles+=("prefill")
        fi
        local role
        if [[ "${#roles[@]}" -eq 0 ]]; then
            role="unknown"
        else
            role=$(IFS='+'; echo "${roles[*]}")
        fi

        echo "$ts,$role,$gpu_index,$sm_util,$mem_util,$mem_used,$power_draw" >> "$SM_LOG"
    done <<< "$output"
}

cleanup() {
    local rc=$?
    set +e
    trap - EXIT INT TERM USR1
    for pid in "${PIDS[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
    sleep 2
    for pid in "${PIDS[@]}"; do
        kill -9 "$pid" 2>/dev/null || true
    done
    kill -- -$$ 2>/dev/null
    exit "$rc"
}

trap cleanup EXIT
trap cleanup INT
trap cleanup USR1
trap cleanup TERM

rm -rf "$EC_SHARED_STORAGE_PATH"
mkdir -p "$EC_SHARED_STORAGE_PATH"

CANONICAL_VISUAL_TOKEN_PRUNING_METHOD="${VISUAL_TOKEN_PRUNING_METHOD,,}"
VLLM_VISUAL_TOKEN_PRUNING_METHOD=""
case "$CANONICAL_VISUAL_TOKEN_PRUNING_METHOD" in
    ""|none)
        ;;
    visionzip)
        VLLM_VISUAL_TOKEN_PRUNING_METHOD="vision_zip"
        ;;
    cdpruner)
        VLLM_VISUAL_TOKEN_PRUNING_METHOD="cdpruner"
        ;;
    *)
        echo "Unsupported VISUAL_TOKEN_PRUNING_METHOD: ${VISUAL_TOKEN_PRUNING_METHOD} (expected visionzip, cdpruner, or none)" >&2
        exit 1
        ;;
esac

declare -a VISION_ZIP_ARGS=()
if [[ -n "$VLLM_VISUAL_TOKEN_PRUNING_METHOD" ]]; then
    VISION_ZIP_ARGS+=(--visual-token-pruning-method "$VLLM_VISUAL_TOKEN_PRUNING_METHOD")
fi
if [[ -n "${VISUAL_TOKEN_PRUNING_RATE:-}" ]]; then
    VISION_ZIP_ARGS+=(--vt-prune-rate "$VISUAL_TOKEN_PRUNING_RATE")
fi
if [[ "$CANONICAL_VISUAL_TOKEN_PRUNING_METHOD" == "visionzip" && -n "${VISION_ZIP_DOMINANT_RATIO:-}" ]]; then
    VISION_ZIP_ARGS+=(--vision-zip-dominant-ratio "$VISION_ZIP_DOMINANT_RATIO")
fi
if [[ "$CANONICAL_VISUAL_TOKEN_PRUNING_METHOD" == "visionzip" && -n "${VISION_ZIP_ATTENTION_LAYER:-}" ]]; then
    VISION_ZIP_ARGS+=(--vision-zip-attention-layer "$VISION_ZIP_ATTENTION_LAYER")
fi

while port_in_use "$PREFILL_NIXL_SIDE_CHANNEL_PORT" || port_in_use "$ED_NIXL_SIDE_CHANNEL_PORT"; do
    PREFILL_NIXL_SIDE_CHANNEL_PORT=$((PREFILL_NIXL_SIDE_CHANNEL_PORT + 1))
    ED_NIXL_SIDE_CHANNEL_PORT=$((ED_NIXL_SIDE_CHANNEL_PORT + 1))
done

CUDA_VISIBLE_DEVICES="$GPU_P" UCX_NET_DEVICES=all VLLM_NIXL_SIDE_CHANNEL_PORT="$PREFILL_NIXL_SIDE_CHANNEL_PORT" \
vllm serve "$MODEL" \
    --gpu-memory-utilization "$P_GPU_MEMORY_UTILIZATION" \
    --port "$PREFILL_PORT" \
    --enforce-eager \
    --enable-request-id-headers \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-batched-tokens "$P_MAX_NUM_BATCHED_TOKENS" \
    --max-num-seqs "$P_MAX_NUM_SEQS" \
    --allowed-local-media-path "${GIT_ROOT}"/tests/v1/ec_connector/integration \
    --ec-transfer-config '{
        "ec_connector": "ECExampleConnector",
        "ec_role": "ec_consumer",
        "ec_connector_extra_config": {
            "shared_storage_path": "'"$EC_SHARED_STORAGE_PATH"'"
        }
    }' \
    --kv-transfer-config '{
        "kv_connector": "NixlConnector",
        "kv_role": "kv_producer"
    }' \
    "${VISION_ZIP_ARGS[@]}" \
    >"${P_LOG}" 2>&1 &
PIDS+=($!)

CUDA_VISIBLE_DEVICES="$GPU_ED" UCX_NET_DEVICES=all VLLM_NIXL_SIDE_CHANNEL_PORT="$ED_NIXL_SIDE_CHANNEL_PORT" \
vllm serve "$MODEL" \
    --gpu-memory-utilization "$ED_GPU_MEMORY_UTILIZATION" \
    --port "$ENCODE_DECODE_PORT" \
    --enforce-eager \
    --enable-request-id-headers \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-batched-tokens "$ED_MAX_NUM_BATCHED_TOKENS" \
    --max-num-seqs "$ED_MAX_NUM_SEQS" \
    --allowed-local-media-path "${GIT_ROOT}"/tests/v1/ec_connector/integration \
    --ec-transfer-config '{
        "ec_connector": "ECExampleConnector",
        "ec_role": "ec_producer",
        "ec_connector_extra_config": {
            "shared_storage_path": "'"$EC_SHARED_STORAGE_PATH"'"
        }
    }' \
    --kv-transfer-config '{
        "kv_connector": "NixlConnector",
        "kv_role": "kv_consumer"
    }' \
    "${VISION_ZIP_ARGS[@]}" \
    >"${ED_LOG}" 2>&1 &
PIDS+=($!)

wait_for_server "$PREFILL_PORT"
wait_for_server "$ENCODE_DECODE_PORT"

python "${GIT_ROOT}/examples/online_serving/disaggregated_encoder/disagg_epd_proxy.py" \
    --host "0.0.0.0" \
    --port "$PROXY_PORT" \
    --encode-servers-urls "http://localhost:$ENCODE_DECODE_PORT" \
    --prefill-servers-urls "http://localhost:$PREFILL_PORT" \
    --decode-servers-urls "http://localhost:$ENCODE_DECODE_PORT" \
    >"${PROXY_LOG}" 2>&1 &
PIDS+=($!)

wait_for_server "$PROXY_PORT"

echo "1ed1p benchmark"
echo "  benchmark      : $BENCHMARK"
echo "  model          : $MODEL"
echo "  prefill_port   : $PREFILL_PORT"
echo "  ed_port        : $ENCODE_DECODE_PORT"
echo "  proxy_port     : $PROXY_PORT"
echo "  gpu_prefill    : $GPU_P"
echo "  gpu_ed         : $GPU_ED"
echo "  run_dir        : $RUN_DIR"

SM_LOG="$RUN_DIR/sm.log"
echo "timestamp,role,gpu_index,sm_utilization_pct,memory_utilization_pct,memory_used_mib,power_draw_watts" > "$SM_LOG"

(
  while true; do
    ts=$(date +%s)
    log_gpu_sm_utilization "$ts"
    sleep "$METRICS_SAMPLING_INTERVAL_SECONDS"
  done
) &
PIDS+=($!)

declare -a BENCH_ARGS=(
    --model "$MODEL"
    --backend openai-chat
    --endpoint /v1/chat/completions
    --seed 0
    --num-prompts "$NUM_PROMPTS"
    --request-rate "$BENCH_REQUEST_RATE"
    --max-concurrency "$BENCH_MAX_CONCURRENCY"
    --port "$PROXY_PORT"
)

if [[ "$BENCHMARK" == "randommm" ]]; then
    BENCH_ARGS+=(
        --dataset-name random-mm
        --random-mm-base-items-per-request "${IMAGES_PER_REQ}"
        --random-mm-num-mm-items-range-ratio 0
        --random-mm-limit-mm-per-prompt "{\"image\": ${IMAGES_PER_REQ}, \"video\": 0}"
    )
else
    BENCH_ARGS+=(
        --dataset-name hf
        --dataset-path "$HF_DATASET_PATH"
    )
fi

vllm bench serve "${BENCH_ARGS[@]}"

if [[ "$BENCHMARK" == "simple" ]]; then
    curl http://127.0.0.1:"${PROXY_PORT}"/v1/chat/completions \
        -H "Content-Type: application/json" \
        -d '{
        "model": "'"${MODEL}"'",
        "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "file://'"${GIT_ROOT}"'/tests/v1/ec_connector/integration/hato.jpg"}},
            {"type": "text", "text": "What is in this image?"}
        ]}
        ]
        }'
fi

cleanup
