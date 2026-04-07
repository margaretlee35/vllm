#!/bin/bash
set -euo pipefail

declare -a PIDS=()

MODEL="${MODEL:-Qwen/Qwen2.5-VL-3B-Instruct}"
LOG_PATH="${LOG_PATH:-./epdtest/logs}"
mkdir -p "$LOG_PATH"

ENCODE_PORT="${ENCODE_PORT:-19534}"
PREFILL_DECODE_PORT="${PREFILL_DECODE_PORT:-19535}"
PROXY_PORT="${PROXY_PORT:-10001}"

GPU_E="${GPU_E:-0}"
GPU_PD="${GPU_PD:-1}"

EC_SHARED_STORAGE_PATH="${EC_SHARED_STORAGE_PATH:-/tmp/ec_cache}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-12000}"

NUM_PROMPTS="${NUM_PROMPTS:-500}"
PD_GPU_MEMORY_UTILIZATION="${PD_GPU_MEMORY_UTILIZATION:-0.85}"
PD_MAX_MODEL_LEN="${PD_MAX_MODEL_LEN:-65536}"
PD_MAX_NUM_BATCHED_TOKENS="${PD_MAX_NUM_BATCHED_TOKENS:-32768}"
PD_MAX_NUM_SEQS="${PD_MAX_NUM_SEQS:-32}"
BENCH_REQUEST_RATE="${BENCH_REQUEST_RATE:-32}"
BENCH_MAX_CONCURRENCY="${BENCH_MAX_CONCURRENCY:-32}"
VISUAL_TOKEN_PRUNING_METHOD="${VISUAL_TOKEN_PRUNING_METHOD:-}"
VISUAL_TOKEN_PRUNING_RATE="${VISUAL_TOKEN_PRUNING_RATE:-}"
VISION_ZIP_DOMINANT_RATIO="${VISION_ZIP_DOMINANT_RATIO:-}"
VISION_ZIP_ATTENTION_LAYER="${VISION_ZIP_ATTENTION_LAYER:-}"
IMAGES_PER_REQ="${IMAGES_PER_REQ:-1}"
METRICS_SAMPLING_INTERVAL_SECONDS="${METRICS_SAMPLING_INTERVAL_SECONDS:-1}"
GPU_PROFILER="${GPU_PROFILER:-none}"
NSYS_ENABLE_GPU_METRICS="${NSYS_ENABLE_GPU_METRICS:-1}"
NSYS_GPU_METRICS_DEVICES="${NSYS_GPU_METRICS_DEVICES:-$GPU_E,$GPU_PD}"
NSYS_GPU_METRICS_FREQUENCY="${NSYS_GPU_METRICS_FREQUENCY:-1000}"

ulimit -n "${ULIMIT_NOFILE:-65535}" >/dev/null 2>&1 || true

GIT_ROOT=$(git rev-parse --show-toplevel)

START_TIME=$(date +"%Y%m%d_%H%M%S")
RUN_DIR="${RUN_DIR:-$LOG_PATH/$START_TIME}"
mkdir -p "$RUN_DIR"
ENC_LOG="$RUN_DIR/encoder.log"
PD_LOG="$RUN_DIR/prefill_decode.log"
PROXY_LOG="$RUN_DIR/proxy.log"
PROFILE_LOG_DIR="${PROFILE_LOG_DIR:-$RUN_DIR/profiler}"

wait_for_server() {
    local port=$1
    timeout "$TIMEOUT_SECONDS" bash -c "
        until curl -s localhost:$port/v1/chat/completions > /dev/null; do
            sleep 1
        done" && return 0 || return 1
}

scrape_kv_cache_usage() {
    local port=$1
    local response
    response=$(curl -fsS "http://127.0.0.1:${port}/metrics" 2>/dev/null || true)
    if [[ -z "$response" ]]; then
        echo "NA"
        return 0
    fi
    awk '$1 == "vllm:kv_cache_usage_perc" {print $NF; found=1; exit} END {if (!found) print "NA"}' <<< "$response"
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
        -i "$GPU_E,$GPU_PD" 2>/dev/null || true)
    if [[ -z "$output" ]]; then
        return 0
    fi
    while IFS=',' read -r gpu_index sm_util mem_util mem_used power_draw; do
        gpu_index=$(echo "$gpu_index" | xargs)
        sm_util=$(echo "$sm_util" | xargs)
        mem_util=$(echo "$mem_util" | xargs)
        mem_used=$(echo "$mem_used" | xargs)
        power_draw=$(echo "$power_draw" | xargs)
        local role="unknown"
        if [[ "$gpu_index" == "$GPU_E" ]]; then
            role="encoder"
        elif [[ "$gpu_index" == "$GPU_PD" ]]; then
            role="prefill_decode"
        fi
        echo "$ts,$role,$gpu_index,$sm_util,$mem_util,$mem_used,$power_draw" >> "$SM_LOG"
    done <<< "$output"
}

validate_profiler() {
    case "$GPU_PROFILER" in
        none)
            return 0
            ;;
        nsys)
            command -v nsys >/dev/null 2>&1 || {
                echo "GPU_PROFILER=nsys requested, but nsys is not installed." >&2
                exit 1
            }
            mkdir -p "$PROFILE_LOG_DIR"
            ;;
        ncu)
            command -v ncu >/dev/null 2>&1 || {
                echo "GPU_PROFILER=ncu requested, but ncu is not installed." >&2
                exit 1
            }
            mkdir -p "$PROFILE_LOG_DIR"
            ;;
        *)
            echo "Unsupported GPU_PROFILER='$GPU_PROFILER'. Use one of: none, nsys, ncu." >&2
            exit 1
            ;;
    esac
}

start_worker() {
    local worker_name=$1
    local log_file=$2
    local cuda_visible_devices=$3
    shift 3
    local -a nsys_args=()

    case "$GPU_PROFILER" in
        none)
            CUDA_VISIBLE_DEVICES="$cuda_visible_devices" "$@" >"${log_file}" 2>&1 &
            ;;
        nsys)
            if [[ "$NSYS_ENABLE_GPU_METRICS" == "1" ]]; then
                nsys_args+=(
                    --gpu-metrics-devices="$NSYS_GPU_METRICS_DEVICES"
                    --gpu-metrics-frequency="$NSYS_GPU_METRICS_FREQUENCY"
                )
            fi
            nsys profile \
                --force-overwrite true \
                --sample=none \
                --trace=cuda,nvtx,osrt \
                "${nsys_args[@]}" \
                --output "${PROFILE_LOG_DIR}/${worker_name}" \
                env CUDA_VISIBLE_DEVICES="$cuda_visible_devices" "$@" >"${log_file}" 2>&1 &
            ;;
        ncu)
            env CUDA_VISIBLE_DEVICES="$cuda_visible_devices" ncu \
                --target-processes all \
                --set default \
                --force-overwrite \
                --export "${PROFILE_LOG_DIR}/${worker_name}" \
                "$@" >"${log_file}" 2>&1 &
            ;;
    esac

    PIDS+=($!)
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

validate_profiler

rm -rf "$EC_SHARED_STORAGE_PATH"
mkdir -p "$EC_SHARED_STORAGE_PATH"

declare -a VISION_ZIP_ARGS=()
if [[ -n "${VISUAL_TOKEN_PRUNING_METHOD:-}" ]]; then
    VISION_ZIP_ARGS+=(--visual-token-pruning-method "$VISUAL_TOKEN_PRUNING_METHOD")
fi
if [[ -n "${VISUAL_TOKEN_PRUNING_RATE:-}" ]]; then
    VISION_ZIP_ARGS+=(--vt-prune-rate "$VISUAL_TOKEN_PRUNING_RATE")
fi
if [[ "${VISUAL_TOKEN_PRUNING_METHOD:-}" == "vision_zip" && -n "${VISION_ZIP_DOMINANT_RATIO:-}" ]]; then
    VISION_ZIP_ARGS+=(--vision-zip-dominant-ratio "$VISION_ZIP_DOMINANT_RATIO")
fi
if [[ "${VISUAL_TOKEN_PRUNING_METHOD:-}" == "vision_zip" && -n "${VISION_ZIP_ATTENTION_LAYER:-}" ]]; then
    VISION_ZIP_ARGS+=(--vision-zip-attention-layer "$VISION_ZIP_ATTENTION_LAYER")
fi

start_worker encoder "$ENC_LOG" "$GPU_E" \
    vllm serve "$MODEL" \
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
    "${VISION_ZIP_ARGS[@]}"

start_worker prefill_decode "$PD_LOG" "$GPU_PD" \
    vllm serve "$MODEL" \
    --gpu-memory-utilization "$PD_GPU_MEMORY_UTILIZATION" \
    --port "$PREFILL_DECODE_PORT" \
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
    "${VISION_ZIP_ARGS[@]}"

wait_for_server "$ENCODE_PORT"
wait_for_server "$PREFILL_DECODE_PORT"

python "${GIT_ROOT}/examples/online_serving/disaggregated_encoder/disagg_epd_proxy.py" \
    --host "0.0.0.0" \
    --port "$PROXY_PORT" \
    --encode-servers-urls "http://localhost:$ENCODE_PORT" \
    --prefill-servers-urls "disable" \
    --decode-servers-urls "http://localhost:$PREFILL_DECODE_PORT" \
    >"${PROXY_LOG}" 2>&1 &
PIDS+=($!)

wait_for_server "$PROXY_PORT"

KV_LOG="$RUN_DIR/kv.log"
SM_LOG="$RUN_DIR/sm.log"

echo "timestamp,encoder_kv_cache_usage,prefill_decode_kv_cache_usage" > "$KV_LOG"
echo "timestamp,role,gpu_index,sm_utilization_pct,memory_utilization_pct,memory_used_mib,power_draw_watts" > "$SM_LOG"

(
  while true; do
    ts=$(date +%s)
    enc=$(scrape_kv_cache_usage "$ENCODE_PORT")
    pd=$(scrape_kv_cache_usage "$PREFILL_DECODE_PORT")
    echo "$ts,$enc,$pd" >> "$KV_LOG"
    log_gpu_sm_utilization "$ts"
    sleep "$METRICS_SAMPLING_INTERVAL_SECONDS"
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
