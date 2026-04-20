#!/bin/bash
set -euo pipefail

# Shared Ne1p1d runner for both BENCHMARK=simple and BENCHMARK=randommm.

# -----------------------------------------------------------------------------
# Configuration defaults
# -----------------------------------------------------------------------------

declare -a PIDS=()

MODEL="${MODEL:-Qwen/Qwen2.5-VL-3B-Instruct}"
LOG_PATH="${LOG_PATH:-./epdtest/logs}"
mkdir -p "$LOG_PATH"
BENCHMARK="${BENCHMARK:-randommm}"

ENCODE_PORT="${ENCODE_PORT:-19534}"
PREFILL_PORT="${PREFILL_PORT:-19535}"
DECODE_PORT="${DECODE_PORT:-19536}"
PROXY_PORT="${PROXY_PORT:-10001}"

GPU_E="${GPU_E:-0,1}"
GPU_P="${GPU_P:-0}"
GPU_D="${GPU_D:-1}"

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
PREFILL_DECODE_SCHEDULING_POLICY="${PREFILL_DECODE_SCHEDULING_POLICY:-priority}"
PREFILL_DECODE_REQUEST_PRIORITY="${PREFILL_DECODE_REQUEST_PRIORITY:--100}"
ENABLE_PREFILL_GPU_ENCODING="${ENABLE_PREFILL_GPU_ENCODING:-1}"
ENABLE_DECODE_GPU_ENCODING="${ENABLE_DECODE_GPU_ENCODING:-1}"
ENCODE_ROUTING_MEMORY_BUFFER="${ENCODE_ROUTING_MEMORY_BUFFER:-0.03}"
ENCODE_ROUTING_WEIGHT_SCALE="${ENCODE_ROUTING_WEIGHT_SCALE:-100}"
ENCODER_ONLY_SERVER_WEIGHT="${ENCODER_ONLY_SERVER_WEIGHT:-100}"
PROXY_IDLE_AWARE_ENCODE_ROUTING="${PROXY_IDLE_AWARE_ENCODE_ROUTING:-1}"
PROXY_IDLE_ENCODE_MAX_INFLIGHT_PER_SERVER="${PROXY_IDLE_ENCODE_MAX_INFLIGHT_PER_SERVER:-8}"
PROXY_ENCODE_REQUEST_PRIORITY="${PROXY_ENCODE_REQUEST_PRIORITY:-100}"
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

die() {
    echo "$*" >&2
    exit 1
}

is_truthy() {
    case "${1,,}" in
        1|true|yes|on)
            return 0
            ;;
        *)
            return 1
            ;;
    esac
}

is_non_negative_number() {
    [[ "${1:-}" =~ ^([0-9]+([.][0-9]+)?|[.][0-9]+)$ ]]
}

is_integer() {
    [[ "${1:-}" =~ ^-?[0-9]+$ ]]
}

assert_non_negative_number() {
    local key=$1
    local value=$2
    if ! is_non_negative_number "$value"; then
        die "$key must be a non-negative number, got: $value"
    fi
}

assert_fraction() {
    local key=$1
    local value=$2
    assert_non_negative_number "$key" "$value"
    awk -v v="$value" 'BEGIN { exit !(v <= 1.0) }' || die "$key must be <= 1.0, got: $value"
}

assert_non_negative_integer() {
    local key=$1
    local value=$2
    if ! is_integer "$value"; then
        die "$key must be an integer, got: $value"
    fi
    (( value >= 0 )) || die "$key must be >= 0, got: $value"
}

split_csv_to_array() {
    local csv="${1:-}"
    local -n out_arr=$2
    local -a raw_items=()
    local item
    out_arr=()
    IFS=',' read -r -a raw_items <<< "$csv"
    for item in "${raw_items[@]}"; do
        item="${item//[[:space:]]/}"
        [[ -n "$item" ]] && out_arr+=("$item")
    done
}

join_by_comma() {
    local -n arr=$1
    local IFS=,
    echo "${arr[*]}"
}

array_contains() {
    local needle=$1
    shift || true
    local item
    for item in "$@"; do
        if [[ "$item" == "$needle" ]]; then
            return 0
        fi
    done
    return 1
}

append_unique() {
    local -n arr=$1
    local value=$2
    if ! array_contains "$value" "${arr[@]}"; then
        arr+=("$value")
    fi
}

append_url_by_weight() {
    local -n out_arr=$1
    local url=$2
    local weight=$3
    local i
    for ((i = 0; i < weight; i++)); do
        out_arr+=("$url")
    done
}

compute_encode_headroom() {
    local gpu_util=$1
    local memory_buffer=$2
    awk -v util="$gpu_util" -v buffer="$memory_buffer" 'BEGIN {
        h = 1.0 - util - buffer;
        if (h < 0.0) h = 0.0;
        printf "%.6f", h;
    }'
}

compute_routing_weight() {
    local headroom=$1
    local scale=$2
    awk -v h="$headroom" -v s="$scale" 'BEGIN {
        w = int(h * s);
        if (h > 0.0 && w < 1) w = 1;
        if (w < 0) w = 0;
        print w;
    }'
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

export UCX_TLS=all
export UCX_NET_DEVICES=all

ulimit -n "${ULIMIT_NOFILE:-65535}" >/dev/null 2>&1 || true

GIT_ROOT=$(git rev-parse --show-toplevel)
PRUNE_CONFIG_FILE="${PRUNE_CONFIG_FILE:-$GIT_ROOT/epdtest/visual_token_pruning_configs.json}"
PRUNE_CONFIG_PARSER="${PRUNE_CONFIG_PARSER:-$GIT_ROOT/epdtest/parse_config.py}"
NE1P1D_PROXY_SCRIPT="${NE1P1D_PROXY_SCRIPT:-$GIT_ROOT/epdtest/scripts/disagg_Ne1p1d_proxy.py}"

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
ENC_LOG="$RUN_DIR/encoder.log"
P_LOG="$RUN_DIR/prefill.log"
D_LOG="$RUN_DIR/decode.log"
PROXY_LOG="$RUN_DIR/proxy.log"
ENCODER_SERVER_ENABLED=0
PREFILL_ENCODE_HEADROOM=0
DECODE_ENCODE_HEADROOM=0
PREFILL_ENCODE_WEIGHT=0
DECODE_ENCODE_WEIGHT=0
ENCODE_SERVERS_URLS_ARG=""
MONITORED_GPU_IDS_CSV=""
PREFILL_GPU_CUDA_VISIBLE_DEVICES="$GPU_P"
DECODE_GPU_CUDA_VISIBLE_DEVICES="$GPU_D"
ENCODER_ONLY_GPU_CUDA_VISIBLE_DEVICES=""
declare -a ENCODER_GPUS_ARRAY=()
declare -a PREFILL_GPUS_ARRAY=()
declare -a DECODE_GPUS_ARRAY=()
declare -a ENCODER_ONLY_GPUS_ARRAY=()
declare -a MONITORED_GPUS_ARRAY=()
declare -a ENCODE_SERVER_URLS=()

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
    if [[ -z "$MONITORED_GPU_IDS_CSV" ]]; then
        return 0
    fi
    output=$(nvidia-smi \
        --query-gpu=index,utilization.gpu,utilization.memory,memory.used,power.draw \
        --format=csv,noheader,nounits \
        -i "$MONITORED_GPU_IDS_CSV" 2>/dev/null || true)
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
        if array_contains "$gpu_index" "${ENCODER_GPUS_ARRAY[@]}"; then
            roles+=("encoder")
        fi
        if array_contains "$gpu_index" "${PREFILL_GPUS_ARRAY[@]}"; then
            roles+=("prefill")
        fi
        if array_contains "$gpu_index" "${DECODE_GPUS_ARRAY[@]}"; then
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
assert_fraction "PREFILL_GPU_MEMORY_UTILIZATION" "$PREFILL_GPU_MEMORY_UTILIZATION"
assert_fraction "DECODE_GPU_MEMORY_UTILIZATION" "$DECODE_GPU_MEMORY_UTILIZATION"
assert_fraction "ENCODER_GPU_MEMORY_UTILIZATION" "$ENCODER_GPU_MEMORY_UTILIZATION"
assert_fraction "ENCODE_ROUTING_MEMORY_BUFFER" "$ENCODE_ROUTING_MEMORY_BUFFER"
assert_non_negative_integer "ENCODE_ROUTING_WEIGHT_SCALE" "$ENCODE_ROUTING_WEIGHT_SCALE"
assert_non_negative_integer "ENCODER_ONLY_SERVER_WEIGHT" "$ENCODER_ONLY_SERVER_WEIGHT"
assert_non_negative_integer "PROXY_IDLE_ENCODE_MAX_INFLIGHT_PER_SERVER" "$PROXY_IDLE_ENCODE_MAX_INFLIGHT_PER_SERVER"
if (( PROXY_IDLE_ENCODE_MAX_INFLIGHT_PER_SERVER < 1 )); then
    die "PROXY_IDLE_ENCODE_MAX_INFLIGHT_PER_SERVER must be >= 1, got: $PROXY_IDLE_ENCODE_MAX_INFLIGHT_PER_SERVER"
fi
if ! is_integer "$PREFILL_DECODE_REQUEST_PRIORITY"; then
    die "PREFILL_DECODE_REQUEST_PRIORITY must be an integer, got: $PREFILL_DECODE_REQUEST_PRIORITY"
fi
if ! is_integer "$PROXY_ENCODE_REQUEST_PRIORITY"; then
    die "PROXY_ENCODE_REQUEST_PRIORITY must be an integer, got: $PROXY_ENCODE_REQUEST_PRIORITY"
fi
if [[ "${PREFILL_DECODE_SCHEDULING_POLICY,,}" != "priority" ]]; then
    if (( PREFILL_DECODE_REQUEST_PRIORITY != 0 || PROXY_ENCODE_REQUEST_PRIORITY != 0 )); then
        die "Non-zero request priorities require PREFILL_DECODE_SCHEDULING_POLICY=priority"
    fi
fi
if [[ ! -f "$NE1P1D_PROXY_SCRIPT" ]]; then
    die "Missing Ne1p1d proxy script: $NE1P1D_PROXY_SCRIPT"
fi

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

split_csv_to_array "$GPU_E" ENCODER_GPUS_ARRAY
split_csv_to_array "$GPU_P" PREFILL_GPUS_ARRAY
split_csv_to_array "$GPU_D" DECODE_GPUS_ARRAY

if [[ "${#ENCODER_GPUS_ARRAY[@]}" -eq 0 ]]; then
    die "GPU_E must include at least one GPU id"
fi
if [[ "${#PREFILL_GPUS_ARRAY[@]}" -eq 0 ]]; then
    die "GPU_P must include at least one GPU id"
fi
if [[ "${#DECODE_GPUS_ARRAY[@]}" -eq 0 ]]; then
    die "GPU_D must include at least one GPU id"
fi

for gpu in "${ENCODER_GPUS_ARRAY[@]}"; do
    if array_contains "$gpu" "${PREFILL_GPUS_ARRAY[@]}"; then
        continue
    fi
    if array_contains "$gpu" "${DECODE_GPUS_ARRAY[@]}"; then
        continue
    fi
    append_unique ENCODER_ONLY_GPUS_ARRAY "$gpu"
done

PREFILL_GPU_CUDA_VISIBLE_DEVICES="$(join_by_comma PREFILL_GPUS_ARRAY)"
DECODE_GPU_CUDA_VISIBLE_DEVICES="$(join_by_comma DECODE_GPUS_ARRAY)"
ENCODER_ONLY_GPU_CUDA_VISIBLE_DEVICES="$(join_by_comma ENCODER_ONLY_GPUS_ARRAY)"

for gpu in "${ENCODER_GPUS_ARRAY[@]}"; do
    append_unique MONITORED_GPUS_ARRAY "$gpu"
done
for gpu in "${PREFILL_GPUS_ARRAY[@]}"; do
    append_unique MONITORED_GPUS_ARRAY "$gpu"
done
for gpu in "${DECODE_GPUS_ARRAY[@]}"; do
    append_unique MONITORED_GPUS_ARRAY "$gpu"
done
MONITORED_GPU_IDS_CSV="$(join_by_comma MONITORED_GPUS_ARRAY)"

while port_in_use "$PREFILL_NIXL_SIDE_CHANNEL_PORT" || port_in_use "$DECODE_NIXL_SIDE_CHANNEL_PORT"; do
    PREFILL_NIXL_SIDE_CHANNEL_PORT=$((PREFILL_NIXL_SIDE_CHANNEL_PORT + 1))
    DECODE_NIXL_SIDE_CHANNEL_PORT=$((DECODE_NIXL_SIDE_CHANNEL_PORT + 1))
done

if [[ "${#ENCODER_ONLY_GPUS_ARRAY[@]}" -gt 0 ]]; then
    CUDA_VISIBLE_DEVICES="$ENCODER_ONLY_GPU_CUDA_VISIBLE_DEVICES" vllm serve "$MODEL" \
        "${HF_OVERRIDES_ARGS[@]}" \
        --gpu-memory-utilization "$ENCODER_GPU_MEMORY_UTILIZATION" \
        --port "$ENCODE_PORT" \
        "${VLLM_EAGER_ARGS[@]}" \
        --scheduling-policy "$PREFILL_DECODE_SCHEDULING_POLICY" \
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
    ENCODER_SERVER_ENABLED=1
fi

CUDA_VISIBLE_DEVICES="$PREFILL_GPU_CUDA_VISIBLE_DEVICES" UCX_NET_DEVICES=all VLLM_NIXL_SIDE_CHANNEL_PORT="$PREFILL_NIXL_SIDE_CHANNEL_PORT" \
vllm serve "$MODEL" \
    "${HF_OVERRIDES_ARGS[@]}" \
    --gpu-memory-utilization "$PREFILL_GPU_MEMORY_UTILIZATION" \
    --port "$PREFILL_PORT" \
    "${VLLM_EAGER_ARGS[@]}" \
    --scheduling-policy "$PREFILL_DECODE_SCHEDULING_POLICY" \
    --enable-request-id-headers \
    --max-model-len "$PD_MAX_MODEL_LEN" \
    --max-num-batched-tokens "$PD_MAX_NUM_BATCHED_TOKENS" \
    --max-num-seqs "$PD_MAX_NUM_SEQS" \
    --allowed-local-media-path "${GIT_ROOT}"/tests/v1/ec_connector/integration \
    --ec-transfer-config '{
        "ec_connector": "ECExampleConnector",
        "ec_role": "ec_both",
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

CUDA_VISIBLE_DEVICES="$DECODE_GPU_CUDA_VISIBLE_DEVICES" UCX_NET_DEVICES=all VLLM_NIXL_SIDE_CHANNEL_PORT="$DECODE_NIXL_SIDE_CHANNEL_PORT" \
vllm serve "$MODEL" \
    "${HF_OVERRIDES_ARGS[@]}" \
    --gpu-memory-utilization "$DECODE_GPU_MEMORY_UTILIZATION" \
    --port "$DECODE_PORT" \
    "${VLLM_EAGER_ARGS[@]}" \
    --scheduling-policy "$PREFILL_DECODE_SCHEDULING_POLICY" \
    --enable-request-id-headers \
    --max-model-len "$PD_MAX_MODEL_LEN" \
    --max-num-batched-tokens "$PD_MAX_NUM_BATCHED_TOKENS" \
    --max-num-seqs "$PD_MAX_NUM_SEQS" \
    --allowed-local-media-path "${GIT_ROOT}"/tests/v1/ec_connector/integration \
    --ec-transfer-config '{
        "ec_connector": "ECExampleConnector",
        "ec_role": "ec_both",
        "ec_connector_extra_config": {
            "shared_storage_path": "'"$EC_SHARED_STORAGE_PATH"'"
        }
    }' \
    --kv-transfer-config '{
        "kv_connector": "NixlConnector",
        "kv_role": "kv_consumer"
    }' \
    "${VISUAL_TOKEN_PRUNING_ARGS[@]}" \
    >"${D_LOG}" 2>&1 &
PIDS+=($!)

if [[ "$ENCODER_SERVER_ENABLED" -eq 1 ]]; then
    wait_for_server "$ENCODE_PORT"
fi
wait_for_server "$PREFILL_PORT"
wait_for_server "$DECODE_PORT"

if is_truthy "$ENABLE_PREFILL_GPU_ENCODING"; then
    PREFILL_ENCODE_HEADROOM=$(compute_encode_headroom "$PREFILL_GPU_MEMORY_UTILIZATION" "$ENCODE_ROUTING_MEMORY_BUFFER")
    PREFILL_ENCODE_WEIGHT=$(compute_routing_weight "$PREFILL_ENCODE_HEADROOM" "$ENCODE_ROUTING_WEIGHT_SCALE")
    if (( PREFILL_ENCODE_WEIGHT > 0 )); then
        append_url_by_weight ENCODE_SERVER_URLS "http://localhost:$PREFILL_PORT" "$PREFILL_ENCODE_WEIGHT"
    fi
fi

if is_truthy "$ENABLE_DECODE_GPU_ENCODING"; then
    DECODE_ENCODE_HEADROOM=$(compute_encode_headroom "$DECODE_GPU_MEMORY_UTILIZATION" "$ENCODE_ROUTING_MEMORY_BUFFER")
    DECODE_ENCODE_WEIGHT=$(compute_routing_weight "$DECODE_ENCODE_HEADROOM" "$ENCODE_ROUTING_WEIGHT_SCALE")
    if (( DECODE_ENCODE_WEIGHT > 0 )); then
        append_url_by_weight ENCODE_SERVER_URLS "http://localhost:$DECODE_PORT" "$DECODE_ENCODE_WEIGHT"
    fi
fi

if [[ "$ENCODER_SERVER_ENABLED" -eq 1 && "$ENCODER_ONLY_SERVER_WEIGHT" -gt 0 ]]; then
    append_url_by_weight ENCODE_SERVER_URLS "http://localhost:$ENCODE_PORT" "$ENCODER_ONLY_SERVER_WEIGHT"
fi

if [[ "${#ENCODE_SERVER_URLS[@]}" -eq 0 ]]; then
    die "No encode servers are routable. Lower PREFILL/DECODE_GPU_MEMORY_UTILIZATION, lower ENCODE_ROUTING_MEMORY_BUFFER, or add encoder-only GPUs in GPU_E."
fi

ENCODE_SERVERS_URLS_ARG="$(join_by_comma ENCODE_SERVER_URLS)"
echo "Encode routing URLs: $ENCODE_SERVERS_URLS_ARG"
echo "Encode routing weights -> prefill:$PREFILL_ENCODE_WEIGHT decode:$DECODE_ENCODE_WEIGHT encoder_only:$ENCODER_ONLY_SERVER_WEIGHT(enabled=$ENCODER_SERVER_ENABLED)"

declare -a PROXY_IDLE_ROUTING_ARGS=(
    --idle-encode-max-inflight-per-server "$PROXY_IDLE_ENCODE_MAX_INFLIGHT_PER_SERVER"
    --encode-request-priority "$PROXY_ENCODE_REQUEST_PRIORITY"
)
if is_truthy "$PROXY_IDLE_AWARE_ENCODE_ROUTING"; then
    PROXY_IDLE_ROUTING_ARGS+=(--prefer-idle-prefill-decode-for-encode)
fi

python "$NE1P1D_PROXY_SCRIPT" \
    --host "0.0.0.0" \
    --port "$PROXY_PORT" \
    --encode-servers-urls "$ENCODE_SERVERS_URLS_ARG" \
    --prefill-servers-urls "http://localhost:$PREFILL_PORT" \
    --decode-servers-urls "http://localhost:$DECODE_PORT" \
    "${PROXY_IDLE_ROUTING_ARGS[@]}" \
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

EXTRA_BODY_JSON="{\"priority\": ${PREFILL_DECODE_REQUEST_PRIORITY}}"
if [[ "${BENCH_MIN_TOKENS:-0}" =~ ^[0-9]+$ ]] && (( BENCH_MIN_TOKENS > 0 )); then
    EXTRA_BODY_JSON="{\"min_tokens\": ${BENCH_MIN_TOKENS}, \"priority\": ${PREFILL_DECODE_REQUEST_PRIORITY}}"
fi
BENCH_ARGS+=(--extra-body "$EXTRA_BODY_JSON")

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
        "priority": '"${PREFILL_DECODE_REQUEST_PRIORITY}"',
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
