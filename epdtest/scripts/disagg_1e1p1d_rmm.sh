#!/bin/bash
set -euo pipefail

declare -a PIDS=()

MODEL="${MODEL:-Qwen/Qwen2.5-VL-3B-Instruct}"
LOG_PATH="${LOG_PATH:-./epdtest/logs}"
mkdir -p "$LOG_PATH"

ENCODE_PORT="${ENCODE_PORT:-19534}"
PREFILL_PORT="${PREFILL_PORT:-19535}"
DECODE_PORT="${DECODE_PORT:-19536}"
PROXY_PORT="${PROXY_PORT:-10001}"

GPU_E="${GPU_E:-2}"
GPU_P="${GPU_P:-2}"
GPU_D="${GPU_D:-3}"

EC_SHARED_STORAGE_PATH="${EC_SHARED_STORAGE_PATH:-/tmp/ec_cache}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-12000}"

NUM_PROMPTS="${NUM_PROMPTS:-500}"
BENCH_REQUEST_RATE="${BENCH_REQUEST_RATE:-32}"
BENCH_MAX_CONCURRENCY="${BENCH_MAX_CONCURRENCY:-32}"
PD_MAX_MODEL_LEN="${PD_MAX_MODEL_LEN:-65536}"
PD_MAX_NUM_BATCHED_TOKENS="${PD_MAX_NUM_BATCHED_TOKENS:-32768}"
PD_MAX_NUM_SEQS="${PD_MAX_NUM_SEQS:-32}"
NIXL_BASE_PORT="${NIXL_BASE_PORT:-$((5200 + ($$ % 1000)))}"
PREFILL_NIXL_SIDE_CHANNEL_PORT="${PREFILL_NIXL_SIDE_CHANNEL_PORT:-$NIXL_BASE_PORT}"
DECODE_NIXL_SIDE_CHANNEL_PORT="${DECODE_NIXL_SIDE_CHANNEL_PORT:-$((NIXL_BASE_PORT + 1000))}"
PREFILL_GPU_MEMORY_UTILIZATION="${PREFILL_GPU_MEMORY_UTILIZATION:-0.85}"
DECODE_GPU_MEMORY_UTILIZATION="${DECODE_GPU_MEMORY_UTILIZATION:-0.85}"
VISUAL_TOKEN_PRUNING_METHOD="${VISUAL_TOKEN_PRUNING_METHOD:-vision_zip}"
VISION_ZIP_RATE="${VISION_ZIP_RATE:-}"
VISION_ZIP_DOMINANT_RATIO="${VISION_ZIP_DOMINANT_RATIO:-}"
VISION_ZIP_ATTENTION_LAYER="${VISION_ZIP_ATTENTION_LAYER:-}"
IMAGES_PER_REQ="${IMAGES_PER_REQ:-1}"

export UCX_TLS=all
export UCX_NET_DEVICES=all

ulimit -n "${ULIMIT_NOFILE:-65535}" >/dev/null 2>&1 || true

GIT_ROOT=$(git rev-parse --show-toplevel)

START_TIME=$(date +"%Y%m%d_%H%M%S")
ENC_LOG=$LOG_PATH/encoder_${START_TIME}.log
P_LOG=$LOG_PATH/p_${START_TIME}.log
D_LOG=$LOG_PATH/d_${START_TIME}.log
PROXY_LOG=$LOG_PATH/proxy_${START_TIME}.log

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

declare -a VISION_ZIP_ARGS=()
if [[ -n "${VISUAL_TOKEN_PRUNING_METHOD:-}" ]]; then
    VISION_ZIP_ARGS+=(--visual-token-pruning-method "$VISUAL_TOKEN_PRUNING_METHOD")
fi
if [[ -n "${VISION_ZIP_RATE:-}" ]]; then
    VISION_ZIP_ARGS+=(--vt-prune-rate "$VISION_ZIP_RATE")
fi
if [[ "${VISUAL_TOKEN_PRUNING_METHOD:-}" == "vision_zip" && -n "${VISION_ZIP_DOMINANT_RATIO:-}" ]]; then
    VISION_ZIP_ARGS+=(--vision-zip-dominant-ratio "$VISION_ZIP_DOMINANT_RATIO")
fi
if [[ "${VISUAL_TOKEN_PRUNING_METHOD:-}" == "vision_zip" && -n "${VISION_ZIP_ATTENTION_LAYER:-}" ]]; then
    VISION_ZIP_ARGS+=(--vision-zip-attention-layer "$VISION_ZIP_ATTENTION_LAYER")
fi

while port_in_use "$PREFILL_NIXL_SIDE_CHANNEL_PORT" || port_in_use "$DECODE_NIXL_SIDE_CHANNEL_PORT"; do
    PREFILL_NIXL_SIDE_CHANNEL_PORT=$((PREFILL_NIXL_SIDE_CHANNEL_PORT + 1))
    DECODE_NIXL_SIDE_CHANNEL_PORT=$((DECODE_NIXL_SIDE_CHANNEL_PORT + 1))
done

CUDA_VISIBLE_DEVICES="$GPU_E" vllm serve "$MODEL" \
    --gpu-memory-utilization 0.05 \
    --port "$ENCODE_PORT" \
    --enforce-eager \
    --enable-request-id-headers \
    --no-enable-prefix-caching \
    --max-num-batched-tokens 114688 \
    --max-num-seqs 128 \
    --allowed-local-media-path "${GIT_ROOT}"/tests/v1/ec_connector/integration \
    --ec-transfer-config '{
        "ec_connector": "ECExampleConnector",
        "ec_role": "ec_producer",
        "ec_connector_extra_config": {
            "shared_storage_path": "'"$EC_SHARED_STORAGE_PATH"'"
        }
    }' \
    "${VISION_ZIP_ARGS[@]}" \
    >"${ENC_LOG}" 2>&1 &
PIDS+=($!)

CUDA_VISIBLE_DEVICES="$GPU_P" UCX_NET_DEVICES=all VLLM_NIXL_SIDE_CHANNEL_PORT="$PREFILL_NIXL_SIDE_CHANNEL_PORT" \
vllm serve "$MODEL" \
    --gpu-memory-utilization "$PREFILL_GPU_MEMORY_UTILIZATION" \
    --port "$PREFILL_PORT" \
    --enforce-eager \
    --enable-request-id-headers \
    --max-model-len "$PD_MAX_MODEL_LEN" \
    --max-num-batched-tokens "$PD_MAX_NUM_BATCHED_TOKENS" \
    --max-num-seqs "$PD_MAX_NUM_SEQS" \
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

CUDA_VISIBLE_DEVICES="$GPU_D" UCX_NET_DEVICES=all VLLM_NIXL_SIDE_CHANNEL_PORT="$DECODE_NIXL_SIDE_CHANNEL_PORT" \
vllm serve "$MODEL" \
    --gpu-memory-utilization "$DECODE_GPU_MEMORY_UTILIZATION" \
    --port "$DECODE_PORT" \
    --enforce-eager \
    --enable-request-id-headers \
    --max-model-len "$PD_MAX_MODEL_LEN" \
    --max-num-batched-tokens "$PD_MAX_NUM_BATCHED_TOKENS" \
    --max-num-seqs "$PD_MAX_NUM_SEQS" \
    --allowed-local-media-path "${GIT_ROOT}"/tests/v1/ec_connector/integration \
    --kv-transfer-config '{
        "kv_connector": "NixlConnector",
        "kv_role": "kv_consumer"
    }' \
    "${VISION_ZIP_ARGS[@]}" \
    >"${D_LOG}" 2>&1 &
PIDS+=($!)

wait_for_server "$ENCODE_PORT"
wait_for_server "$PREFILL_PORT"
wait_for_server "$DECODE_PORT"

python "${GIT_ROOT}/examples/online_serving/disaggregated_encoder/disagg_epd_proxy.py" \
    --host "0.0.0.0" \
    --port "$PROXY_PORT" \
    --encode-servers-urls "http://localhost:$ENCODE_PORT" \
    --prefill-servers-urls "http://localhost:$PREFILL_PORT" \
    --decode-servers-urls "http://localhost:$DECODE_PORT" \
    >"${PROXY_LOG}" 2>&1 &
PIDS+=($!)

wait_for_server "$PROXY_PORT"

KV_LOG=$LOG_PATH/kv_${START_TIME}.log
(
  while true; do
    ts=$(date +%s)
    enc=$(curl -s "http://127.0.0.1:${ENCODE_PORT}/metrics" | awk '/vllm:kv_cache_usage_perc/ {print $NF; exit}')
    prefill=$(curl -s "http://127.0.0.1:${PREFILL_PORT}/metrics" | awk '/vllm:kv_cache_usage_perc/ {print $NF; exit}')
    decode=$(curl -s "http://127.0.0.1:${DECODE_PORT}/metrics" | awk '/vllm:kv_cache_usage_perc/ {print $NF; exit}')
    echo "$ts,$enc,$prefill,$decode" >> "$KV_LOG"
    sleep 1
  done
) &
PIDS+=($!)

vllm bench serve \
    --model "$MODEL" \
    --backend openai-chat \
    --endpoint /v1/chat/completions \
    --dataset-name random-mm \
    --seed 0 \
    --num-prompts "$NUM_PROMPTS" \
    --request-rate "$BENCH_REQUEST_RATE" \
    --max-concurrency "$BENCH_MAX_CONCURRENCY" \
    --random-mm-base-items-per-request "${IMAGES_PER_REQ:-1}" \
    --random-mm-num-mm-items-range-ratio 0 \
    --random-mm-limit-mm-per-prompt "{\"image\": ${IMAGES_PER_REQ:-1}, \"video\": 0}" \
    --port "$PROXY_PORT"

cleanup
