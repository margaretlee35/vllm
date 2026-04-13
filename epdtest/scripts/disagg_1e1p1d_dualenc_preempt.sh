#!/bin/bash
set -euo pipefail

# Experimental scheduler:
# - Disaggregate prefill and decode (P and D are separate workers)
# - Run encoder workers on both P GPU and D GPU
# - Preempt encoder workers when P/D load is non-zero

# -----------------------------------------------------------------------------
# Configuration defaults
# -----------------------------------------------------------------------------

declare -a PIDS=()

MODEL="${MODEL:-Qwen/Qwen2.5-VL-3B-Instruct}"
LOG_PATH="${LOG_PATH:-./epdtest/logs}"
mkdir -p "$LOG_PATH"
BENCHMARK="${BENCHMARK:-randommm}"

ENCODE_ON_P_PORT="${ENCODE_ON_P_PORT:-19534}"
PREFILL_PORT="${PREFILL_PORT:-19535}"
DECODE_PORT="${DECODE_PORT:-19536}"
ENCODE_ON_D_PORT="${ENCODE_ON_D_PORT:-19537}"
PROXY_PORT="${PROXY_PORT:-10001}"

GPU_P="${GPU_P:-0}"
GPU_D="${GPU_D:-1}"
ALLOW_SHARED_GPU="${ALLOW_SHARED_GPU:-0}"

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
PREFILL_GPU_MEMORY_UTILIZATION="${PREFILL_GPU_MEMORY_UTILIZATION:-0.75}"
DECODE_GPU_MEMORY_UTILIZATION="${DECODE_GPU_MEMORY_UTILIZATION:-0.75}"
ENCODER_ON_P_GPU_MEMORY_UTILIZATION="${ENCODER_ON_P_GPU_MEMORY_UTILIZATION:-0.10}"
ENCODER_ON_D_GPU_MEMORY_UTILIZATION="${ENCODER_ON_D_GPU_MEMORY_UTILIZATION:-0.10}"

# Coarse preemption controls
PREEMPT_ENCODING="${PREEMPT_ENCODING:-1}"
PREEMPT_CHECK_INTERVAL_SECONDS="${PREEMPT_CHECK_INTERVAL_SECONDS:-1}"
PREEMPT_LOAD_THRESHOLD="${PREEMPT_LOAD_THRESHOLD:-1}"
RESUME_LOAD_THRESHOLD="${RESUME_LOAD_THRESHOLD:-0}"

VISUAL_TOKEN_PRUNING_METHOD="${VISUAL_TOKEN_PRUNING_METHOD:-}"
IMAGES_PER_REQ="${IMAGES_PER_REQ:-1}"
HF_DATASET_PATH="${HF_DATASET_PATH:-lmarena-ai/VisionArena-Chat}"
METRICS_SAMPLING_INTERVAL_SECONDS="${METRICS_SAMPLING_INTERVAL_SECONDS:-1}"

die() {
    echo "$*" >&2
    exit 1
}

validate_benchmark() {
    case "${BENCHMARK,,}" in
        simple|randommm)
            ;;
        *)
            die "Unsupported BENCHMARK: $BENCHMARK (expected simple or randommm)"
            ;;
    esac
}

validate_gpu_layout() {
    if [[ "$GPU_P" == "$GPU_D" && "$ALLOW_SHARED_GPU" != "1" ]]; then
        die "This scheduler expects distinct GPUs for P and D: GPU_P=$GPU_P GPU_D=$GPU_D. Set ALLOW_SHARED_GPU=1 only if you intentionally want overlap."
    fi
}

export UCX_TLS=all
export UCX_NET_DEVICES=all

ulimit -n "${ULIMIT_NOFILE:-65535}" >/dev/null 2>&1 || true

GIT_ROOT=$(git rev-parse --show-toplevel)
PRUNE_CONFIG_FILE="${PRUNE_CONFIG_FILE:-$GIT_ROOT/epdtest/visual_token_pruning_configs.json}"
PRUNE_CONFIG_PARSER="${PRUNE_CONFIG_PARSER:-$GIT_ROOT/epdtest/parse_config.py}"

ensure_pruning_config_files() {
    if [[ ! -f "$PRUNE_CONFIG_FILE" ]]; then
        die "Missing pruning config file: $PRUNE_CONFIG_FILE"
    fi
    if [[ ! -f "$PRUNE_CONFIG_PARSER" ]]; then
        die "Missing pruning config parser: $PRUNE_CONFIG_PARSER"
    fi
}

apply_visual_token_pruning_config() {
    local raw_method="${1:-}"
    local python_bin
    local parsed
    local cfg_method=""
    local cfg_rate=""
    local cfg_dom_ratio=""
    local cfg_attn_layer=""
    raw_method="${raw_method,,}"

    if command -v python3 >/dev/null 2>&1; then
        python_bin="python3"
    elif command -v python >/dev/null 2>&1; then
        python_bin="python"
    else
        echo "python3 (or python) is required to parse ${PRUNE_CONFIG_FILE}" >&2
        return 1
    fi

    parsed=$("$python_bin" "$PRUNE_CONFIG_PARSER" "$PRUNE_CONFIG_FILE" "$raw_method") || return 1

    while IFS=$'\t' read -r key value; do
        case "$key" in
            VISUAL_TOKEN_PRUNING_METHOD)
                cfg_method="$value"
                ;;
            VISUAL_TOKEN_PRUNING_RATE)
                cfg_rate="$value"
                ;;
            VISION_ZIP_DOMINANT_RATIO)
                cfg_dom_ratio="$value"
                ;;
            VISION_ZIP_ATTENTION_LAYER)
                cfg_attn_layer="$value"
                ;;
        esac
    done <<< "$parsed"

    if [[ -z "$cfg_method" || "$cfg_method" == "none" ]]; then
        VISUAL_TOKEN_PRUNING_METHOD="none"
        VISUAL_TOKEN_PRUNING_RATE=""
        VISION_ZIP_DOMINANT_RATIO=""
        VISION_ZIP_ATTENTION_LAYER=""
        return 0
    fi

    VISUAL_TOKEN_PRUNING_METHOD="$cfg_method"
    VISUAL_TOKEN_PRUNING_RATE="${VISUAL_TOKEN_PRUNING_RATE:-$cfg_rate}"
    if [[ "$cfg_method" == "visionzip" ]]; then
        VISION_ZIP_DOMINANT_RATIO="${VISION_ZIP_DOMINANT_RATIO:-$cfg_dom_ratio}"
        VISION_ZIP_ATTENTION_LAYER="${VISION_ZIP_ATTENTION_LAYER:-$cfg_attn_layer}"
    else
        VISION_ZIP_DOMINANT_RATIO=""
        VISION_ZIP_ATTENTION_LAYER=""
    fi
}

build_visual_token_pruning_args() {
    declare -g -a VISUAL_TOKEN_PRUNING_ARGS=()

    case "${VISUAL_TOKEN_PRUNING_METHOD,,}" in
        ""|none|visionzip|cdpruner)
            if [[ "${VISUAL_TOKEN_PRUNING_METHOD,,}" != "none" && -n "${VISUAL_TOKEN_PRUNING_METHOD:-}" ]]; then
                VISUAL_TOKEN_PRUNING_ARGS+=(--visual-token-pruning-method "${VISUAL_TOKEN_PRUNING_METHOD,,}")
            fi
            ;;
        *)
            die "Unsupported VISUAL_TOKEN_PRUNING_METHOD: ${VISUAL_TOKEN_PRUNING_METHOD} (expected visionzip, cdpruner, or none)"
            ;;
    esac

    if [[ -n "${VISUAL_TOKEN_PRUNING_RATE:-}" ]]; then
        VISUAL_TOKEN_PRUNING_ARGS+=(--vt-prune-rate "$VISUAL_TOKEN_PRUNING_RATE")
    fi
    if [[ "${VISUAL_TOKEN_PRUNING_METHOD,,}" == "visionzip" ]]; then
        if [[ -n "${VISION_ZIP_DOMINANT_RATIO:-}" ]]; then
            VISUAL_TOKEN_PRUNING_ARGS+=(--vision-zip-dominant-ratio "$VISION_ZIP_DOMINANT_RATIO")
        fi
        if [[ -n "${VISION_ZIP_ATTENTION_LAYER:-}" ]]; then
            VISUAL_TOKEN_PRUNING_ARGS+=(--vision-zip-attention-layer "$VISION_ZIP_ATTENTION_LAYER")
        fi
    fi
}

START_TIME=$(date +"%Y%m%d_%H%M%S")
RUN_DIR="${RUN_DIR:-$LOG_PATH/$START_TIME}"
mkdir -p "$RUN_DIR"
ENC_P_LOG="$RUN_DIR/encoder_on_p.log"
P_LOG="$RUN_DIR/prefill.log"
D_LOG="$RUN_DIR/decode.log"
ENC_D_LOG="$RUN_DIR/encoder_on_d.log"
PROXY_LOG="$RUN_DIR/proxy.log"
PREEMPT_LOG="$RUN_DIR/preemption.log"

ENC_P_PID=""
ENC_D_PID=""
ENCODERS_PAUSED=0

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
        -i "$GPU_P,$GPU_D" 2>/dev/null || true)
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
        if [[ "$gpu_index" == "$GPU_P" ]]; then
            roles+=("encode_on_p")
            roles+=("prefill")
        fi
        if [[ "$gpu_index" == "$GPU_D" ]]; then
            roles+=("encode_on_d")
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

get_pd_request_load() {
    local port=$1
    local m
    m=$(curl -fsS "http://127.0.0.1:${port}/metrics" 2>/dev/null || true)
    if [[ -z "$m" ]]; then
        echo 0
        return 0
    fi
    awk '
        /^vllm:num_requests_running\{/ {sum += $2}
        /^vllm:num_requests_waiting\{/ {sum += $2}
        END {print int(sum + 0)}
    ' <<< "$m"
}

pause_encoder_workers() {
    if [[ "$ENCODERS_PAUSED" -eq 1 ]]; then
        return 0
    fi
    [[ -n "$ENC_P_PID" ]] && kill -STOP "$ENC_P_PID" 2>/dev/null || true
    [[ -n "$ENC_D_PID" ]] && kill -STOP "$ENC_D_PID" 2>/dev/null || true
    ENCODERS_PAUSED=1
    echo "$(date +%F\ %T) preempt: pause encoders" >> "$PREEMPT_LOG"
}

resume_encoder_workers() {
    if [[ "$ENCODERS_PAUSED" -eq 0 ]]; then
        return 0
    fi
    [[ -n "$ENC_P_PID" ]] && kill -CONT "$ENC_P_PID" 2>/dev/null || true
    [[ -n "$ENC_D_PID" ]] && kill -CONT "$ENC_D_PID" 2>/dev/null || true
    ENCODERS_PAUSED=0
    echo "$(date +%F\ %T) preempt: resume encoders" >> "$PREEMPT_LOG"
}

preemption_controller_loop() {
    echo "$(date +%F\ %T) preemption controller started (threshold=$PREEMPT_LOAD_THRESHOLD, resume=$RESUME_LOAD_THRESHOLD)" >> "$PREEMPT_LOG"
    while true; do
        local p_load d_load total_load
        p_load=$(get_pd_request_load "$PREFILL_PORT")
        d_load=$(get_pd_request_load "$DECODE_PORT")
        total_load=$((p_load + d_load))

        if [[ "$total_load" -ge "$PREEMPT_LOAD_THRESHOLD" ]]; then
            pause_encoder_workers
        elif [[ "$total_load" -le "$RESUME_LOAD_THRESHOLD" ]]; then
            resume_encoder_workers
        fi

        sleep "$PREEMPT_CHECK_INTERVAL_SECONDS"
    done
}

cleanup() {
    local rc=$?
    set +e
    trap - EXIT INT TERM USR1
    # Ensure encoders are resumable before kill.
    resume_encoder_workers
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

validate_benchmark
validate_gpu_layout
ensure_pruning_config_files

trap cleanup EXIT
trap cleanup INT
trap cleanup USR1
trap cleanup TERM

rm -rf "$EC_SHARED_STORAGE_PATH"
mkdir -p "$EC_SHARED_STORAGE_PATH"

if ! apply_visual_token_pruning_config "${VISUAL_TOKEN_PRUNING_METHOD:-}"; then
    die "Failed to load VISUAL_TOKEN_PRUNING_METHOD config: ${VISUAL_TOKEN_PRUNING_METHOD:-}"
fi

build_visual_token_pruning_args

while port_in_use "$PREFILL_NIXL_SIDE_CHANNEL_PORT" || port_in_use "$DECODE_NIXL_SIDE_CHANNEL_PORT"; do
    PREFILL_NIXL_SIDE_CHANNEL_PORT=$((PREFILL_NIXL_SIDE_CHANNEL_PORT + 1))
    DECODE_NIXL_SIDE_CHANNEL_PORT=$((DECODE_NIXL_SIDE_CHANNEL_PORT + 1))
done

# Encoder worker on prefill GPU
CUDA_VISIBLE_DEVICES="$GPU_P" vllm serve "$MODEL" \
    --gpu-memory-utilization "$ENCODER_ON_P_GPU_MEMORY_UTILIZATION" \
    --port "$ENCODE_ON_P_PORT" \
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
    "${VISUAL_TOKEN_PRUNING_ARGS[@]}" \
    >"${ENC_P_LOG}" 2>&1 &
ENC_P_PID=$!
PIDS+=($ENC_P_PID)

# Prefill worker
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
    "${VISUAL_TOKEN_PRUNING_ARGS[@]}" \
    >"${P_LOG}" 2>&1 &
PIDS+=($!)

# Decode worker
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
    "${VISUAL_TOKEN_PRUNING_ARGS[@]}" \
    >"${D_LOG}" 2>&1 &
PIDS+=($!)

# Encoder worker on decode GPU
CUDA_VISIBLE_DEVICES="$GPU_D" vllm serve "$MODEL" \
    --gpu-memory-utilization "$ENCODER_ON_D_GPU_MEMORY_UTILIZATION" \
    --port "$ENCODE_ON_D_PORT" \
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
    "${VISUAL_TOKEN_PRUNING_ARGS[@]}" \
    >"${ENC_D_LOG}" 2>&1 &
ENC_D_PID=$!
PIDS+=($ENC_D_PID)

wait_for_server "$ENCODE_ON_P_PORT"
wait_for_server "$PREFILL_PORT"
wait_for_server "$DECODE_PORT"
wait_for_server "$ENCODE_ON_D_PORT"

python "${GIT_ROOT}/examples/online_serving/disaggregated_encoder/disagg_epd_proxy.py" \
    --host "0.0.0.0" \
    --port "$PROXY_PORT" \
    --encode-servers-urls "http://localhost:$ENCODE_ON_P_PORT,http://localhost:$ENCODE_ON_D_PORT" \
    --prefill-servers-urls "http://localhost:$PREFILL_PORT" \
    --decode-servers-urls "http://localhost:$DECODE_PORT" \
    >"${PROXY_LOG}" 2>&1 &
PIDS+=($!)

wait_for_server "$PROXY_PORT"

if [[ "$PREEMPT_ENCODING" == "1" ]]; then
    preemption_controller_loop &
    PIDS+=($!)
fi

echo "1e1p1d dual-encoder preempt benchmark"
echo "  benchmark          : $BENCHMARK"
echo "  model              : $MODEL"
echo "  encode_on_p_port   : $ENCODE_ON_P_PORT (GPU $GPU_P)"
echo "  prefill_port       : $PREFILL_PORT (GPU $GPU_P)"
echo "  decode_port        : $DECODE_PORT (GPU $GPU_D)"
echo "  encode_on_d_port   : $ENCODE_ON_D_PORT (GPU $GPU_D)"
echo "  proxy_port         : $PROXY_PORT"
echo "  preempt_encoding   : $PREEMPT_ENCODING"
echo "  preempt_threshold  : $PREEMPT_LOAD_THRESHOLD"
echo "  resume_threshold   : $RESUME_LOAD_THRESHOLD"
echo "  run_dir            : $RUN_DIR"

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

cleanup
