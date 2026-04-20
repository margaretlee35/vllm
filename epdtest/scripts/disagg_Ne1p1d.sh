#!/bin/bash
set -euo pipefail

# Shared Ne1p1d runner for both BENCHMARK=simple and BENCHMARK=randommm.
# This topology runs N encoders (GPU_E list), 1 prefill (GPU_P), and 1 decode (GPU_D).

# -----------------------------------------------------------------------------
# Configuration defaults
# -----------------------------------------------------------------------------

declare -a PIDS=()
declare -a ENCODE_GPUS=()
declare -a ENCODE_PORT_LIST=()
declare -a ENCODE_LOGS=()
declare -a CRITICAL_PIDS=()
declare -a CRITICAL_NAMES=()

MODEL="${MODEL:-Qwen/Qwen2.5-VL-3B-Instruct}"
LOG_PATH="${LOG_PATH:-./epdtest/logs}"
mkdir -p "$LOG_PATH"
BENCHMARK="${BENCHMARK:-randommm}"

ENCODE_PORT="${ENCODE_PORT:-19534}"
ENCODE_PORTS="${ENCODE_PORTS:-}"
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
BENCH_MIN_TOKENS="${BENCH_MIN_TOKENS:-1}"
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
IMAGES_PER_REQ="${IMAGES_PER_REQ:-1}"
HF_DATASET_PATH="${HF_DATASET_PATH:-lmarena-ai/VisionArena-Chat}"
METRICS_SAMPLING_INTERVAL_SECONDS="${METRICS_SAMPLING_INTERVAL_SECONDS:-1}"
FAIL_FAST_POLL_SECONDS="${FAIL_FAST_POLL_SECONDS:-1}"
ENFORCE_EAGER="${ENFORCE_EAGER:-0}"
declare -a VLLM_EAGER_ARGS=()
case "${ENFORCE_EAGER,,}" in
    1|true|yes|on)
        VLLM_EAGER_ARGS+=(--enforce-eager)
        ;;
esac

ENCODE_URLS_CSV=""
MONITOR_GPU_CSV=""

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

trim_spaces() {
    local value="$1"
    # shellcheck disable=SC2001
    echo "$(echo "$value" | sed 's/^ *//; s/ *$//')"
}

contains_value() {
    local needle="$1"
    shift
    local item
    for item in "$@"; do
        if [[ "$item" == "$needle" ]]; then
            return 0
        fi
    done
    return 1
}

build_encoder_layout() {
    ENCODE_GPUS=()
    ENCODE_PORT_LIST=()

    local -a raw_gpus=()
    local item
    IFS=',' read -ra raw_gpus <<< "$GPU_E"

    for item in "${raw_gpus[@]}"; do
        item="$(trim_spaces "$item")"
        if [[ -n "$item" ]]; then
            ENCODE_GPUS+=("$item")
        fi
    done

    if [[ ${#ENCODE_GPUS[@]} -eq 0 ]]; then
        die "GPU_E must contain at least one GPU id (example: GPU_E=0 or GPU_E=0,2)"
    fi

    local idx
    if [[ -n "$ENCODE_PORTS" ]]; then
        local -a raw_ports=()
        local port
        IFS=',' read -ra raw_ports <<< "$ENCODE_PORTS"
        for port in "${raw_ports[@]}"; do
            port="$(trim_spaces "$port")"
            if [[ -n "$port" ]]; then
                ENCODE_PORT_LIST+=("$port")
            fi
        done

        if [[ ${#ENCODE_PORT_LIST[@]} -ne ${#ENCODE_GPUS[@]} ]]; then
            die "ENCODE_PORTS count (${#ENCODE_PORT_LIST[@]}) must match GPU_E count (${#ENCODE_GPUS[@]})"
        fi
    else
        local candidate_port="$ENCODE_PORT"
        local -a reserved_ports=("$PREFILL_PORT" "$DECODE_PORT" "$PROXY_PORT")

        for idx in "${!ENCODE_GPUS[@]}"; do
            while contains_value "$candidate_port" "${reserved_ports[@]}" || \
                  contains_value "$candidate_port" "${ENCODE_PORT_LIST[@]}"; do
                candidate_port=$((candidate_port + 1))
            done
            ENCODE_PORT_LIST+=("$candidate_port")
            candidate_port=$((candidate_port + 1))
        done
    fi

    local port
    for port in "${ENCODE_PORT_LIST[@]}"; do
        if contains_value "$port" "$PREFILL_PORT" "$DECODE_PORT" "$PROXY_PORT"; then
            die "Encoder port collision detected: ${port}. Choose non-overlapping ENCODE_PORT/ENCODE_PORTS."
        fi
    done

    ENCODE_LOGS=()
    for idx in "${!ENCODE_GPUS[@]}"; do
        ENCODE_LOGS+=("$RUN_DIR/encoder_${idx}.log")
    done

    local -a encode_urls=()
    for port in "${ENCODE_PORT_LIST[@]}"; do
        encode_urls+=("http://localhost:$port")
    done
    ENCODE_URLS_CSV=$(IFS=','; echo "${encode_urls[*]}")

    local -a monitor_gpus=("${ENCODE_GPUS[@]}" "$GPU_P" "$GPU_D")
    local -a unique_gpus=()
    local gpu
    for gpu in "${monitor_gpus[@]}"; do
        if ! contains_value "$gpu" "${unique_gpus[@]}"; then
            unique_gpus+=("$gpu")
        fi
    done
    MONITOR_GPU_CSV=$(IFS=','; echo "${unique_gpus[*]}")
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
    declare -g -a HF_OVERRIDES_ARGS=()

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

    if [[ "${VISUAL_TOKEN_PRUNING_METHOD,,}" == "visionzip" || "${VISUAL_TOKEN_PRUNING_METHOD,,}" == "cdpruner" ]]; then
        HF_OVERRIDES_ARGS+=(--hf-overrides '{"architectures":["Qwen2_5_VLPruneForConditionalGeneration"]}')
    fi
}

START_TIME=$(date +"%Y%m%d_%H%M%S")
RUN_DIR="${RUN_DIR:-$LOG_PATH/$START_TIME}"
mkdir -p "$RUN_DIR"
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

ensure_required_ports_free() {
    local -a required_ports=("${ENCODE_PORT_LIST[@]}" "$PREFILL_PORT" "$DECODE_PORT" "$PROXY_PORT")
    local port
    for port in "${required_ports[@]}"; do
        if port_in_use "$port"; then
            die "Port ${port} is already in use. Stop stale processes or change ports."
        fi
    done
}

register_critical_pid() {
    local name="$1"
    local pid="$2"
    CRITICAL_NAMES+=("$name")
    CRITICAL_PIDS+=("$pid")
}

monitor_critical_processes() {
    while true; do
        local idx
        for idx in "${!CRITICAL_PIDS[@]}"; do
            local pid="${CRITICAL_PIDS[$idx]}"
            local name="${CRITICAL_NAMES[$idx]}"
            if ! kill -0 "$pid" 2>/dev/null; then
                echo "Critical process exited unexpectedly: ${name} (pid=${pid})" >&2
                kill -USR1 "$$" 2>/dev/null || true
                exit 0
            fi
        done
        sleep "$FAIL_FAST_POLL_SECONDS"
    done
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
        -i "$MONITOR_GPU_CSV" 2>/dev/null || true)
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
        if contains_value "$gpu_index" "${ENCODE_GPUS[@]}"; then
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

validate_benchmark
ensure_pruning_config_files
build_encoder_layout

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
ensure_required_ports_free

while port_in_use "$PREFILL_NIXL_SIDE_CHANNEL_PORT" || port_in_use "$DECODE_NIXL_SIDE_CHANNEL_PORT"; do
    PREFILL_NIXL_SIDE_CHANNEL_PORT=$((PREFILL_NIXL_SIDE_CHANNEL_PORT + 1))
    DECODE_NIXL_SIDE_CHANNEL_PORT=$((DECODE_NIXL_SIDE_CHANNEL_PORT + 1))
done

for idx in "${!ENCODE_GPUS[@]}"; do
    encode_gpu="${ENCODE_GPUS[$idx]}"
    encode_port="${ENCODE_PORT_LIST[$idx]}"
    encode_log="${ENCODE_LOGS[$idx]}"

    CUDA_VISIBLE_DEVICES="$encode_gpu" vllm serve "$MODEL" \
        "${HF_OVERRIDES_ARGS[@]}" \
        --gpu-memory-utilization "$ENCODER_GPU_MEMORY_UTILIZATION" \
        --port "$encode_port" \
        "${VLLM_EAGER_ARGS[@]}" \
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
        >"${encode_log}" 2>&1 &
    PIDS+=($!)
    register_critical_pid "encoder_${idx}" "$!"
done

CUDA_VISIBLE_DEVICES="$GPU_P" UCX_NET_DEVICES=all VLLM_NIXL_SIDE_CHANNEL_PORT="$PREFILL_NIXL_SIDE_CHANNEL_PORT" \
vllm serve "$MODEL" \
    "${HF_OVERRIDES_ARGS[@]}" \
    --gpu-memory-utilization "$PREFILL_GPU_MEMORY_UTILIZATION" \
    --port "$PREFILL_PORT" \
    "${VLLM_EAGER_ARGS[@]}" \
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
register_critical_pid "prefill" "$!"

CUDA_VISIBLE_DEVICES="$GPU_D" UCX_NET_DEVICES=all VLLM_NIXL_SIDE_CHANNEL_PORT="$DECODE_NIXL_SIDE_CHANNEL_PORT" \
vllm serve "$MODEL" \
    "${HF_OVERRIDES_ARGS[@]}" \
    --gpu-memory-utilization "$DECODE_GPU_MEMORY_UTILIZATION" \
    --port "$DECODE_PORT" \
    "${VLLM_EAGER_ARGS[@]}" \
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
register_critical_pid "decode" "$!"

for encode_port in "${ENCODE_PORT_LIST[@]}"; do
    wait_for_server "$encode_port"
done
wait_for_server "$PREFILL_PORT"
wait_for_server "$DECODE_PORT"

python "${GIT_ROOT}/examples/online_serving/disaggregated_encoder/disagg_epd_proxy.py" \
    --host "0.0.0.0" \
    --port "$PROXY_PORT" \
    --encode-servers-urls "$ENCODE_URLS_CSV" \
    --prefill-servers-urls "http://localhost:$PREFILL_PORT" \
    --decode-servers-urls "http://localhost:$DECODE_PORT" \
    >"${PROXY_LOG}" 2>&1 &
PIDS+=($!)
register_critical_pid "proxy" "$!"

wait_for_server "$PROXY_PORT"

monitor_critical_processes &
PIDS+=($!)

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

if [[ "${BENCH_MIN_TOKENS:-0}" =~ ^[0-9]+$ ]] && (( BENCH_MIN_TOKENS > 0 )); then
    BENCH_ARGS+=(--extra-body "{\"min_tokens\": ${BENCH_MIN_TOKENS}}")
fi

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
