#!/bin/bash
set -euo pipefail

# Shared 1e1pNd runner for both BENCHMARK=simple and BENCHMARK=randommm.
# This topology runs:
# - 1 encoder instance (E)
# - 1 prefill instance (P)
# - N decode instances (D1..DN)
#
# Multi-decode is controlled via comma-separated GPU_D, e.g. GPU_D="2,3".

# -----------------------------------------------------------------------------
# Configuration defaults
# -----------------------------------------------------------------------------

declare -a PIDS=()

declare -a DECODE_GPUS=()
declare -a DECODE_PORT_LIST=()
declare -a DECODE_LOGS=()
declare -a DECODE_NIXL_PORT_LIST=()

MODEL="${MODEL:-Qwen/Qwen2.5-VL-3B-Instruct}"
LOG_PATH="${LOG_PATH:-./epdtest/logs}"
mkdir -p "$LOG_PATH"
BENCHMARK="${BENCHMARK:-randommm}"

ENCODE_PORT="${ENCODE_PORT:-19534}"
PREFILL_PORT="${PREFILL_PORT:-19535}"
DECODE_PORT="${DECODE_PORT:-19536}"
DECODE_PORTS="${DECODE_PORTS:-}"
PROXY_PORT="${PROXY_PORT:-10001}"

GPU_E="${GPU_E:-0}"
GPU_P="${GPU_P:-1}"
GPU_D="${GPU_D:-0,1}"

EC_SHARED_STORAGE_PATH="${EC_SHARED_STORAGE_PATH:-/tmp/ec_cache}"
KV_SHARED_STORAGE_PATH="${KV_SHARED_STORAGE_PATH:-/tmp/kv_cache}"
KV_CONNECTOR="${KV_CONNECTOR:-auto}"
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
IMAGES_PER_REQ="${IMAGES_PER_REQ:-1}"
HF_DATASET_PATH="${HF_DATASET_PATH:-lmarena-ai/VisionArena-Chat}"
METRICS_SAMPLING_INTERVAL_SECONDS="${METRICS_SAMPLING_INTERVAL_SECONDS:-1}"
ENFORCE_EAGER="${ENFORCE_EAGER:-0}"
declare -a VLLM_EAGER_ARGS=()
case "${ENFORCE_EAGER,,}" in
    1|true|yes|on)
        VLLM_EAGER_ARGS+=(--enforce-eager)
        ;;
esac

KV_CONNECTOR_NAME=""
PREFILL_KV_TRANSFER_CONFIG=""
DECODE_KV_TRANSFER_CONFIG=""
DECODE_URLS_CSV=""
MONITOR_GPU_CSV=""

START_TIME=$(date +"%Y%m%d_%H%M%S")
RUN_DIR="${RUN_DIR:-$LOG_PATH/$START_TIME}"
mkdir -p "$RUN_DIR"
ENC_LOG="$RUN_DIR/encoder.log"
P_LOG="$RUN_DIR/prefill.log"
PROXY_LOG="$RUN_DIR/proxy.log"


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


parse_decode_gpus() {
    DECODE_GPUS=()

    local raw_gpus=()
    local item
    IFS=',' read -ra raw_gpus <<< "$GPU_D"

    for item in "${raw_gpus[@]}"; do
        item="$(trim_spaces "$item")"
        if [[ -n "$item" ]]; then
            DECODE_GPUS+=("$item")
        fi
    done

    if [[ ${#DECODE_GPUS[@]} -eq 0 ]]; then
        die "GPU_D must contain at least one GPU id (example: GPU_D=2 or GPU_D=2,3)"
    fi
}


parse_decode_ports() {
    DECODE_PORT_LIST=()

    local idx
    if [[ -n "$DECODE_PORTS" ]]; then
        local raw_ports=()
        local port
        IFS=',' read -ra raw_ports <<< "$DECODE_PORTS"
        for port in "${raw_ports[@]}"; do
            port="$(trim_spaces "$port")"
            if [[ -n "$port" ]]; then
                DECODE_PORT_LIST+=("$port")
            fi
        done

        if [[ ${#DECODE_PORT_LIST[@]} -ne ${#DECODE_GPUS[@]} ]]; then
            die "DECODE_PORTS count (${#DECODE_PORT_LIST[@]}) must match GPU_D count (${#DECODE_GPUS[@]})"
        fi
    else
        for idx in "${!DECODE_GPUS[@]}"; do
            DECODE_PORT_LIST+=("$((DECODE_PORT + idx))")
        done
    fi
}


build_decode_layout() {
    DECODE_LOGS=()
    DECODE_NIXL_PORT_LIST=()

    local idx
    for idx in "${!DECODE_GPUS[@]}"; do
        DECODE_LOGS+=("$RUN_DIR/decode_${idx}.log")
        DECODE_NIXL_PORT_LIST+=("$((DECODE_NIXL_SIDE_CHANNEL_PORT + idx))")
    done

    local -a decode_urls=()
    local port
    for port in "${DECODE_PORT_LIST[@]}"; do
        decode_urls+=("http://localhost:$port")
    done
    DECODE_URLS_CSV=$(IFS=','; echo "${decode_urls[*]}")

    local -a monitor_gpus=("$GPU_E" "$GPU_P" "${DECODE_GPUS[@]}")
    local -a unique_gpus=()
    local gpu
    for gpu in "${monitor_gpus[@]}"; do
        if ! contains_value "$gpu" "${unique_gpus[@]}"; then
            unique_gpus+=("$gpu")
        fi
    done
    MONITOR_GPU_CSV=$(IFS=','; echo "${unique_gpus[*]}")
}


detect_nixl_available() {
    local python_bin
    if command -v python3 >/dev/null 2>&1; then
        python_bin="python3"
    elif command -v python >/dev/null 2>&1; then
        python_bin="python"
    else
        return 1
    fi

    "$python_bin" - <<'PY'
import importlib.util
import sys

try:
    available = importlib.util.find_spec("nixl._api") is not None
except ModuleNotFoundError:
    available = False

sys.exit(0 if available else 1)
PY
}


resolve_kv_connector() {
    case "${KV_CONNECTOR,,}" in
        auto)
            if [[ ${#DECODE_GPUS[@]} -gt 1 ]]; then
                KV_CONNECTOR_NAME="ExampleConnector"
                echo "KV_CONNECTOR=auto resolved to ExampleConnector for multi-decode topology."
            elif detect_nixl_available; then
                KV_CONNECTOR_NAME="NixlConnector"
                echo "KV_CONNECTOR=auto resolved to NixlConnector."
            else
                KV_CONNECTOR_NAME="ExampleConnector"
                echo "KV_CONNECTOR=auto resolved to ExampleConnector because NIXL is unavailable."
            fi
            ;;
        nixl|nixlconnector)
            KV_CONNECTOR_NAME="NixlConnector"
            ;;
        example|exampleconnector)
            KV_CONNECTOR_NAME="ExampleConnector"
            ;;
        *)
            die "Unsupported KV_CONNECTOR: $KV_CONNECTOR (expected auto, nixl, or example)"
            ;;
    esac

    if [[ "$KV_CONNECTOR_NAME" == "NixlConnector" && ${#DECODE_GPUS[@]} -gt 1 ]]; then
        die "NixlConnector is currently configured for 1P1D only in this script. For 1e1pNd, use KV_CONNECTOR=example (or KV_CONNECTOR=auto)."
    fi

    if [[ "$KV_CONNECTOR_NAME" == "NixlConnector" ]]; then
        PREFILL_KV_TRANSFER_CONFIG='{"kv_connector":"NixlConnector","kv_role":"kv_producer"}'
        DECODE_KV_TRANSFER_CONFIG='{"kv_connector":"NixlConnector","kv_role":"kv_consumer"}'
    else
        PREFILL_KV_TRANSFER_CONFIG='{"kv_connector":"ExampleConnector","kv_role":"kv_producer","kv_connector_extra_config":{"shared_storage_path":"'"$KV_SHARED_STORAGE_PATH"'"}}'
        DECODE_KV_TRANSFER_CONFIG='{"kv_connector":"ExampleConnector","kv_role":"kv_consumer","kv_connector_extra_config":{"shared_storage_path":"'"$KV_SHARED_STORAGE_PATH"'"}}'
    fi
}


prepare_storage_paths() {
    rm -rf "$EC_SHARED_STORAGE_PATH"
    mkdir -p "$EC_SHARED_STORAGE_PATH"

    if [[ "$KV_CONNECTOR_NAME" == "ExampleConnector" ]]; then
        rm -rf "$KV_SHARED_STORAGE_PATH"
        mkdir -p "$KV_SHARED_STORAGE_PATH"
    fi
}


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


reserve_nixl_side_channel_ports() {
    if [[ "$KV_CONNECTOR_NAME" != "NixlConnector" ]]; then
        return
    fi

    local stride=$(( ${#DECODE_GPUS[@]} + 1 ))
    while true; do
        local conflict=0
        local idx

        if port_in_use "$PREFILL_NIXL_SIDE_CHANNEL_PORT"; then
            conflict=1
        fi

        DECODE_NIXL_PORT_LIST=()
        for idx in "${!DECODE_GPUS[@]}"; do
            local port_candidate=$((DECODE_NIXL_SIDE_CHANNEL_PORT + idx))
            DECODE_NIXL_PORT_LIST+=("$port_candidate")
            if port_in_use "$port_candidate"; then
                conflict=1
            fi
        done

        if [[ $conflict -eq 0 ]]; then
            break
        fi

        PREFILL_NIXL_SIDE_CHANNEL_PORT=$((PREFILL_NIXL_SIDE_CHANNEL_PORT + stride))
        DECODE_NIXL_SIDE_CHANNEL_PORT=$((DECODE_NIXL_SIDE_CHANNEL_PORT + stride))
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
        if [[ "$gpu_index" == "$GPU_E" ]]; then
            roles+=("encoder")
        fi
        if [[ "$gpu_index" == "$GPU_P" ]]; then
            roles+=("prefill")
        fi
        if contains_value "$gpu_index" "${DECODE_GPUS[@]}"; then
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


print_layout() {
    echo "1e1pNd layout"
    echo "  gpu_encoder     : $GPU_E"
    echo "  gpu_prefill     : $GPU_P"
    echo "  gpu_decode_list : $(IFS=','; echo "${DECODE_GPUS[*]}")"
    echo "  decode_ports    : $(IFS=','; echo "${DECODE_PORT_LIST[*]}")"
    echo "  kv_connector    : $KV_CONNECTOR_NAME"
    if [[ "$KV_CONNECTOR_NAME" == "NixlConnector" ]]; then
        echo "  nixl_prefill_sc : $PREFILL_NIXL_SIDE_CHANNEL_PORT"
        echo "  nixl_decode_sc  : $(IFS=','; echo "${DECODE_NIXL_PORT_LIST[*]}")"
    else
        echo "  kv_storage_path : $KV_SHARED_STORAGE_PATH"
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


validate_benchmark
ensure_pruning_config_files
parse_decode_gpus
parse_decode_ports
build_decode_layout
resolve_kv_connector
prepare_storage_paths

trap cleanup EXIT
trap cleanup INT
trap cleanup USR1
trap cleanup TERM

if ! apply_visual_token_pruning_config "${VISUAL_TOKEN_PRUNING_METHOD:-}"; then
    die "Failed to load VISUAL_TOKEN_PRUNING_METHOD config: ${VISUAL_TOKEN_PRUNING_METHOD:-}"
fi

build_visual_token_pruning_args
reserve_nixl_side_channel_ports
print_layout


CUDA_VISIBLE_DEVICES="$GPU_E" vllm serve "$MODEL" \
    --gpu-memory-utilization "$ENCODER_GPU_MEMORY_UTILIZATION" \
    --port "$ENCODE_PORT" \
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
    >"${ENC_LOG}" 2>&1 &
PIDS+=($!)


declare -a PREFILL_ENV=("CUDA_VISIBLE_DEVICES=$GPU_P" "UCX_NET_DEVICES=all")
if [[ "$KV_CONNECTOR_NAME" == "NixlConnector" ]]; then
    PREFILL_ENV+=("VLLM_NIXL_SIDE_CHANNEL_PORT=$PREFILL_NIXL_SIDE_CHANNEL_PORT")
fi

env "${PREFILL_ENV[@]}" \
vllm serve "$MODEL" \
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
    --kv-transfer-config "$PREFILL_KV_TRANSFER_CONFIG" \
    "${VISUAL_TOKEN_PRUNING_ARGS[@]}" \
    >"${P_LOG}" 2>&1 &
PIDS+=($!)


for idx in "${!DECODE_GPUS[@]}"; do
    decode_gpu="${DECODE_GPUS[$idx]}"
    decode_port="${DECODE_PORT_LIST[$idx]}"
    decode_log="${DECODE_LOGS[$idx]}"

    declare -a DECODE_ENV=("CUDA_VISIBLE_DEVICES=$decode_gpu" "UCX_NET_DEVICES=all")
    if [[ "$KV_CONNECTOR_NAME" == "NixlConnector" ]]; then
        DECODE_ENV+=("VLLM_NIXL_SIDE_CHANNEL_PORT=${DECODE_NIXL_PORT_LIST[$idx]}")
    fi

    env "${DECODE_ENV[@]}" \
    vllm serve "$MODEL" \
        --gpu-memory-utilization "$DECODE_GPU_MEMORY_UTILIZATION" \
        --port "$decode_port" \
        "${VLLM_EAGER_ARGS[@]}" \
        --enable-request-id-headers \
        --max-model-len "$PD_MAX_MODEL_LEN" \
        --max-num-batched-tokens "$PD_MAX_NUM_BATCHED_TOKENS" \
        --max-num-seqs "$PD_MAX_NUM_SEQS" \
        --allowed-local-media-path "${GIT_ROOT}"/tests/v1/ec_connector/integration \
        --kv-transfer-config "$DECODE_KV_TRANSFER_CONFIG" \
        "${VISUAL_TOKEN_PRUNING_ARGS[@]}" \
        >"${decode_log}" 2>&1 &
    PIDS+=($!)
done


wait_for_server "$ENCODE_PORT"
wait_for_server "$PREFILL_PORT"
for port in "${DECODE_PORT_LIST[@]}"; do
    wait_for_server "$port"
done

python "${GIT_ROOT}/examples/online_serving/disaggregated_encoder/disagg_epd_proxy.py" \
    --host "0.0.0.0" \
    --port "$PROXY_PORT" \
    --encode-servers-urls "http://localhost:$ENCODE_PORT" \
    --prefill-servers-urls "http://localhost:$PREFILL_PORT" \
    --decode-servers-urls "$DECODE_URLS_CSV" \
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
