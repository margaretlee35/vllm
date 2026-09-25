#!/bin/bash
set -euo pipefail

# Shared 1e1p1d runner for BENCHMARK=simple, randommm and video. The first two
# are driven by `vllm bench serve`; video is driven by aiperf (see run.sh).

declare -a PIDS=()

MODEL="${MODEL:-Qwen/Qwen2.5-VL-3B-Instruct}"
LOG_PATH="${LOG_PATH:-./vlmtest/logs}"
mkdir -p "$LOG_PATH"
BENCHMARK="${BENCHMARK:-randommm}"

ENCODE_PORT="${ENCODE_PORT:-19534}"
PREFILL_PORT="${PREFILL_PORT:-19535}"
DECODE_PORT="${DECODE_PORT:-19536}"
PROXY_PORT="${PROXY_PORT:-10001}"

GPU_E="${GPU_E:-0}"
GPU_P="${GPU_P:-1}"
GPU_D="${GPU_D:-2}"

EC_SHARED_STORAGE_PATH="${EC_SHARED_STORAGE_PATH:-/tmp/ec_cache}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-12000}"

NUM_PROMPTS="${NUM_PROMPTS:-300}"
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
ENCODER_GPU_MEMORY_UTILIZATION="${ENCODER_GPU_MEMORY_UTILIZATION:-0.05}"
VISUAL_TOKEN_PRUNING_METHOD="${VISUAL_TOKEN_PRUNING_METHOD:-}"
VISUAL_TOKEN_PRUNING_RATE="${VISUAL_TOKEN_PRUNING_RATE:-}"
VISION_ZIP_DOMINANT_RATIO="${VISION_ZIP_DOMINANT_RATIO:-}"
VISION_ZIP_ATTENTION_LAYER="${VISION_ZIP_ATTENTION_LAYER:-}"
IMAGES_PER_REQ="${IMAGES_PER_REQ:-1}"
HF_DATASET_PATH="${HF_DATASET_PATH:-lmarena-ai/VisionArena-Chat}"
METRICS_SAMPLING_INTERVAL_SECONDS="${METRICS_SAMPLING_INTERVAL_SECONDS:-1}"
# video mode only (run.sh sets both)
VIDEO_JSONL="${VIDEO_JSONL:-}"
AIPERF_BIN="${AIPERF_BIN:-aiperf}"
BENCH_OUTPUT_LEN="${BENCH_OUTPUT_LEN:-128}"
# VLLM_VIDEO_LOADER_BACKEND=nvdec only: a spare GPU for every server's NVDEC
# decode. Unset = each server decodes on its own model GPU, where the
# NV12->RGB kernels and D2H copy time-slice with the model's kernels.
NVDEC_GPU="${NVDEC_GPU:-}"
# 1 = every server writes stage events (vllm/v1/stage_trace.py) to
# $RUN_DIR/trace, broken down per request after the benchmark.
STAGE_TRACE="${STAGE_TRACE:-0}"
TRACE_SKIP_FIRST="${TRACE_SKIP_FIRST:-0}"
# 1 = only the encoder renders each request (media decode + HF processor); the
# proxy forwards its rendered prompt to P and D. 0 = all three render it.
FORWARD_RENDERED="${FORWARD_RENDERED:-1}"

case "${BENCHMARK,,}" in
    simple|randommm)
        ;;
    video)
        if [[ ! -f "$VIDEO_JSONL" ]]; then
            echo "BENCHMARK=video needs VIDEO_JSONL (got '${VIDEO_JSONL}')" >&2
            exit 1
        fi
        ;;
    *)
        echo "Unsupported BENCHMARK: $BENCHMARK (expected simple, randommm or video)" >&2
        exit 1
        ;;
esac

# UCX settings for the NIXL prefill->decode KV transfer.
#
# `UCX_TLS=all` / `UCX_NET_DEVICES=all` (what epdtest uses) do not work
# everywhere. NIXL asks UCX for an active-messages transport with peer error
# handling; shared-memory and CUDA transports do not provide one ("no am bcopy" /
# "no peer failure handler"), so a TCP transport on a real interface has to be in
# the list. Loopback alone is not enough either -- UCX advertises a non-loopback
# address and then fails to route to it.
#
# Any UCX_TLS / UCX_NET_DEVICES already in the environment is deliberately
# ignored: cluster profiles commonly export an mlx5 device list aimed at a system
# UCX, and the UCX bundled with the NIXL wheel reports those devices as
# unavailable. Use NIXL_UCX_TLS / NIXL_UCX_NET_DEVICES to pick a fabric on
# purpose.
UCX_TLS="${NIXL_UCX_TLS:-tcp,cuda_copy,cuda_ipc}"
UCX_NET_DEVICES="${NIXL_UCX_NET_DEVICES:-}"
if [[ -z "$UCX_NET_DEVICES" ]]; then
    # First non-loopback IPv4 interface.
    UCX_NET_DEVICES=$(ip -o -4 addr show 2>/dev/null \
        | awk '$2 != "lo" {print $2; exit}')
    UCX_NET_DEVICES="${UCX_NET_DEVICES:-lo}"
fi
export UCX_TLS UCX_NET_DEVICES

ulimit -n "${ULIMIT_NOFILE:-65535}" >/dev/null 2>&1 || true

GIT_ROOT=$(git rev-parse --show-toplevel)

START_TIME=$(date +"%Y%m%d_%H%M%S")
RUN_DIR="${RUN_DIR:-$LOG_PATH/$START_TIME}"
mkdir -p "$RUN_DIR"
ENC_LOG="$RUN_DIR/encoder.log"
P_LOG="$RUN_DIR/prefill.log"
D_LOG="$RUN_DIR/decode.log"
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

start_worker() {
    local worker_name=$1
    local log_file=$2
    local cuda_visible_devices=$3
    shift 3

    # The model takes the first visible GPU; the decoder is pointed at the
    # second via VLLM_NVDEC_DEVICE.
    if [[ -n "$NVDEC_GPU" && "$NVDEC_GPU" != "$cuda_visible_devices" ]]; then
        cuda_visible_devices="$cuda_visible_devices,$NVDEC_GPU"
        set -- env VLLM_NVDEC_DEVICE=1 "$@"
    fi

    env CUDA_VISIBLE_DEVICES="$cuda_visible_devices" \
        VLLM_STAGE_TRACE_ROLE="$worker_name" \
        "$@" >"${log_file}" 2>&1 &
    PIDS+=($!)
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
        -i "$GPU_E,$GPU_P,$GPU_D" 2>/dev/null || true)
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
        if [[ "$gpu_index" == "$GPU_E" ]]; then
            roles+=("encoder")
        fi
        if [[ "$gpu_index" == "$GPU_P" ]]; then
            roles+=("prefill")
        fi
        if [[ "$gpu_index" == "$GPU_D" ]]; then
            roles+=("decode")
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

if [[ "$STAGE_TRACE" == "1" ]]; then
    export VLLM_STAGE_TRACE_DIR="$RUN_DIR/trace"
    rm -rf "$VLLM_STAGE_TRACE_DIR"
fi

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

declare -a SERVER_RENDER_ARGS=()
declare -a PROXY_RENDER_ARGS=()
if [[ "$FORWARD_RENDERED" == "1" ]]; then
    SERVER_RENDER_ARGS+=(--enable-prerendered-prompts)
    PROXY_RENDER_ARGS+=(--forward-rendered-prompt)
fi

while port_in_use "$PREFILL_NIXL_SIDE_CHANNEL_PORT" || port_in_use "$DECODE_NIXL_SIDE_CHANNEL_PORT"; do
    PREFILL_NIXL_SIDE_CHANNEL_PORT=$((PREFILL_NIXL_SIDE_CHANNEL_PORT + 1))
    DECODE_NIXL_SIDE_CHANNEL_PORT=$((DECODE_NIXL_SIDE_CHANNEL_PORT + 1))
done

start_worker encoder "$ENC_LOG" "$GPU_E" \
    vllm serve "$MODEL" \
    --gpu-memory-utilization "$ENCODER_GPU_MEMORY_UTILIZATION" \
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
    "${SERVER_RENDER_ARGS[@]}"

start_worker prefill "$P_LOG" "$GPU_P" \
    env VLLM_NIXL_SIDE_CHANNEL_PORT="$PREFILL_NIXL_SIDE_CHANNEL_PORT" \
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
    "${SERVER_RENDER_ARGS[@]}"

start_worker decode "$D_LOG" "$GPU_D" \
    env VLLM_NIXL_SIDE_CHANNEL_PORT="$DECODE_NIXL_SIDE_CHANNEL_PORT" \
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
    "${SERVER_RENDER_ARGS[@]}"

wait_for_server "$ENCODE_PORT"
wait_for_server "$PREFILL_PORT"
wait_for_server "$DECODE_PORT"

python "${GIT_ROOT}/examples/online_serving/disaggregated_encoder/disagg_epd_proxy.py" \
    --host "0.0.0.0" \
    --port "$PROXY_PORT" \
    --encode-servers-urls "http://localhost:$ENCODE_PORT" \
    --prefill-servers-urls "http://localhost:$PREFILL_PORT" \
    --decode-servers-urls "http://localhost:$DECODE_PORT" \
    "${PROXY_RENDER_ARGS[@]}" \
    >"${PROXY_LOG}" 2>&1 &
PIDS+=($!)

wait_for_server "$PROXY_PORT"

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

if [[ "$BENCHMARK" == "video" ]]; then
    # Streaming so TTFT / ITL are real; ignore_eos pins the output length so
    # runs differ only in their video input. No min_tokens: the proxy rewrites
    # max_tokens=1 for the prefill hop but passes min_tokens through, and
    # prefill then rejects the request with a 400. Results land in $RUN_DIR/aiperf/
    # (profile_export_aiperf.json / .csv, per-request *_raw.jsonl).
    declare -a AIPERF_RATE=()
    # inf / empty = closed loop at --concurrency only
    if [[ -n "$BENCH_REQUEST_RATE" && "$BENCH_REQUEST_RATE" != "inf" ]]; then
        AIPERF_RATE=(--request-rate "$BENCH_REQUEST_RATE")
    fi
    "$AIPERF_BIN" profile \
        -m "$MODEL" \
        -u "http://localhost:$PROXY_PORT" \
        --endpoint-type chat \
        --streaming \
        --input-file "$VIDEO_JSONL" \
        --custom-dataset-type single_turn \
        --request-count "$NUM_PROMPTS" \
        --concurrency "$BENCH_MAX_CONCURRENCY" \
        "${AIPERF_RATE[@]}" \
        --random-seed 0 \
        --extra-inputs "max_tokens:$BENCH_OUTPUT_LEN" \
        --extra-inputs ignore_eos:true \
        --artifact-dir "$RUN_DIR/aiperf" \
        --ui none \
        --no-server-metrics
else
    vllm bench serve "${BENCH_ARGS[@]}"
fi

if [[ "$STAGE_TRACE" == "1" ]]; then
    python "$(dirname -- "${BASH_SOURCE[0]}")/stage_breakdown.py" \
        "$VLLM_STAGE_TRACE_DIR" \
        --skip-first "$TRACE_SKIP_FIRST" \
        --csv "$RUN_DIR/stage_breakdown.csv" \
        --json "$RUN_DIR/stage_breakdown.json" \
        | tee "$RUN_DIR/stage_breakdown.txt" || true
fi

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
