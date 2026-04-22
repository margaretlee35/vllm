#!/bin/bash
set -euo pipefail

declare -a PIDS=()

###############################################################################
# User-configurable settings (override with env)
###############################################################################
MODEL="${MODEL:-Qwen/Qwen2.5-VL-3B-Instruct}"
LOG_PATH="${LOG_PATH:-./epdtest/lmms_eval}"
mkdir -p "$LOG_PATH"

# Keep benchmark + model cache off ~ (override per host)
HF_HOME="${HF_HOME:-/workspace/.hf_cache}"
HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
EVAL_OUTPUT_ROOT="${EVAL_OUTPUT_ROOT:-./epdtest/lmms_eval}"
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$HUGGINGFACE_HUB_CACHE" "$EVAL_OUTPUT_ROOT"

# Space-separated method names in run order.
METHODS="${METHODS:-visionzip cdpruner}"
BASE_PORT="${BASE_PORT:-19535}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-1800}"
DEFAULT_VLLM_LIMIT_MM_PER_PROMPT='{"image": 8, "video": 1}'
VLLM_LIMIT_MM_PER_PROMPT="${VLLM_LIMIT_MM_PER_PROMPT:-$DEFAULT_VLLM_LIMIT_MM_PER_PROMPT}"

# Visual-token pruning knobs are resolved per method from PRUNE_CONFIG_FILE.
# Direct terminal overrides for pruning knobs are intentionally ignored.

# Prune wrapper architecture. Keep this aligned with model registry.
PRUNE_WRAPPER_ARCH="${PRUNE_WRAPPER_ARCH:-Qwen2_5_VLPruneForConditionalGeneration}"
# If 1, fail run when prune method does not resolve to PRUNE_WRAPPER_ARCH.
STRICT_ARCH_CHECK="${STRICT_ARCH_CHECK:-1}"

# Extra args appended to every `vllm serve` (escape/quote carefully)
EXTRA_VLLM_ARGS="${EXTRA_VLLM_ARGS:-}"

# LMMS-Eval knobs — prefer the active venv when `source .venv/bin/activate` sets VIRTUAL_ENV
# (matches README.sy.md: `uv pip install lmms-eval` into the vLLM venv). Override: LMMS_PYTHON_BIN=/path/to/python
if [[ -z "${LMMS_PYTHON_BIN:-}" ]]; then
    if [[ -n "${VIRTUAL_ENV:-}" && -x "${VIRTUAL_ENV}/bin/python" ]]; then
        LMMS_PYTHON_BIN="${VIRTUAL_ENV}/bin/python"
    elif command -v python3 >/dev/null 2>&1; then
        LMMS_PYTHON_BIN="python3"
    else
        LMMS_PYTHON_BIN="python"
    fi
fi
LMMS_TASKS="${LMMS_TASKS:-mmmu_val}"
LMMS_LIMIT="${LMMS_LIMIT:-300}"       # set 0 for full split
LMMS_BATCH_SIZE="${LMMS_BATCH_SIZE:-1}"
LMMS_MODEL="${LMMS_MODEL:-openai_compatible}"

# Override if your lmms-eval version expects different fields.
# Tokens: {PORT} -> server port, {MODEL} -> model id
LMMS_MODEL_ARGS_TEMPLATE="${LMMS_MODEL_ARGS_TEMPLATE:-model={MODEL},base_url=http://127.0.0.1:{PORT}/v1,api_key=EMPTY,temperature=0,max_new_tokens=128}"
# Keep OpenAI-compatible backends usable by default for local vLLM.
OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
# If 1, force OpenAI-compatible eval backends to use local vLLM base_url.
FORCE_LOCAL_OPENAI_BASE_URL="${FORCE_LOCAL_OPENAI_BASE_URL:-1}"

START_TIME=$(date +"%Y%m%d_%H%M%S")
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
GIT_ROOT=$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)
PRUNE_CONFIG_FILE="${PRUNE_CONFIG_FILE:-$GIT_ROOT/epdtest/visual_token_pruning_configs.json}"
PRUNE_CONFIG_PARSER="${PRUNE_CONFIG_PARSER:-$GIT_ROOT/epdtest/parse_config.py}"

###############################################################################
# Helpers
###############################################################################
normalize_method() {
    local method="${1,,}"
    case "$method" in
        none|visionzip|cdpruner) echo "$method" ;;
        *)
            echo "Unsupported METHOD '$1'. Use one of: none, visionzip, cdpruner." >&2
            exit 2
            ;;
    esac
}

ensure_pruning_config_files() {
    if [[ ! -f "$PRUNE_CONFIG_FILE" ]]; then
        echo "Missing pruning config file: $PRUNE_CONFIG_FILE" >&2
        exit 1
    fi
    if [[ ! -f "$PRUNE_CONFIG_PARSER" ]]; then
        echo "Missing pruning config parser: $PRUNE_CONFIG_PARSER" >&2
        exit 1
    fi
}

enforce_config_only_pruning_knobs() {
    local ignored=0
    for var_name in VISUAL_TOKEN_PRUNING_RATE VISION_ZIP_DOMINANT_RATIO VISION_ZIP_ATTENTION_LAYER; do
        if [[ -n "${!var_name:-}" ]]; then
            ignored=1
        fi
        unset "$var_name" || true
    done
    if [[ "$ignored" -eq 1 ]]; then
        echo "Ignoring direct pruning env vars (VISUAL_TOKEN_PRUNING_RATE, VISION_ZIP_DOMINANT_RATIO, VISION_ZIP_ATTENTION_LAYER)."
        echo "Using method values from: $PRUNE_CONFIG_FILE"
    fi
}

resolve_visual_token_pruning_config() {
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
        exit 1
    fi

    parsed=$("$python_bin" "$PRUNE_CONFIG_PARSER" "$PRUNE_CONFIG_FILE" "$raw_method")

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
        RESOLVED_VISUAL_TOKEN_PRUNING_METHOD="none"
        RESOLVED_VISUAL_TOKEN_PRUNING_RATE=""
        RESOLVED_VISION_ZIP_DOMINANT_RATIO=""
        RESOLVED_VISION_ZIP_ATTENTION_LAYER=""
        return 0
    fi

    RESOLVED_VISUAL_TOKEN_PRUNING_METHOD="$cfg_method"
    RESOLVED_VISUAL_TOKEN_PRUNING_RATE="$cfg_rate"
    if [[ "$cfg_method" == "visionzip" ]]; then
        RESOLVED_VISION_ZIP_DOMINANT_RATIO="$cfg_dom_ratio"
        RESOLVED_VISION_ZIP_ATTENTION_LAYER="$cfg_attn_layer"
    else
        RESOLVED_VISION_ZIP_DOMINANT_RATIO=""
        RESOLVED_VISION_ZIP_ATTENTION_LAYER=""
    fi
}

port_in_use() {
    local port=$1
    ss -ltn 2>/dev/null | awk '{print $4}' | grep -Eq "[:.]${port}$"
}

wait_for_model_ready() {
    local port=$1
    local pid=$2
    local server_log=$3
    local deadline=$((SECONDS + TIMEOUT_SECONDS))
    local url_models="http://127.0.0.1:${port}/v1/models"
    while (( SECONDS < deadline )); do
        if ! kill -0 "$pid" 2>/dev/null; then
            echo "vLLM exited before readiness check on port ${port}. See log: ${server_log}" >&2
            tail -n 120 "$server_log" >&2 || true
            return 1
        fi
        if curl -fsS "$url_models" >/dev/null 2>&1; then
            return 0
        fi
        sleep 1
    done
    echo "Model did not become ready within ${TIMEOUT_SECONDS}s on port ${port}."
    return 1
}

cleanup() {
    local rc=$?
    set +e
    trap - EXIT INT TERM
    echo "Stopping background processes..."
    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid" 2>/dev/null
        fi
    done
    sleep 2
    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill -9 "$pid" 2>/dev/null
        fi
    done
    exit "$rc"
}
trap cleanup EXIT INT TERM

stop_server() {
    set +e
    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid" 2>/dev/null
        fi
    done
    sleep 2
    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill -9 "$pid" 2>/dev/null
        fi
    done
    PIDS=()
    set -e
}

verify_architecture() {
    local normalized_method="$1"
    local server_log="$2"

    if [[ "$normalized_method" == "none" ]]; then
        return 0
    fi

    local resolved_arch
    resolved_arch=$(sed -n 's/.*Resolved architecture: //p' "$server_log" | tail -n 1 || true)

    if [[ -z "$resolved_arch" ]]; then
        echo "Could not find 'Resolved architecture' in $server_log." >&2
        if [[ "$STRICT_ARCH_CHECK" == "1" ]]; then
            return 1
        fi
        return 0
    fi

    if [[ "$resolved_arch" != "$PRUNE_WRAPPER_ARCH" ]]; then
        echo "Architecture mismatch for method '$normalized_method': expected '$PRUNE_WRAPPER_ARCH', got '$resolved_arch'." >&2
        echo "Hint: verify model registry mapping and hf-overrides usage." >&2
        if [[ "$STRICT_ARCH_CHECK" == "1" ]]; then
            return 1
        fi
    fi
}

start_vllm_server() {
    local method="$1"
    local port="$2"
    local server_log="$3"

    local -a METHOD_ARGS=()
    local -a HF_OVERRIDES_ARGS=()

    local normalized_method
    normalized_method="$(normalize_method "$method")"
    resolve_visual_token_pruning_config "$normalized_method"

    case "$RESOLVED_VISUAL_TOKEN_PRUNING_METHOD" in
        none)
            ;;
        visionzip)
            METHOD_ARGS+=(
                --visual-token-pruning-method vision_zip
                --vt-prune-rate "$RESOLVED_VISUAL_TOKEN_PRUNING_RATE"
                --vision-zip-dominant-ratio "$RESOLVED_VISION_ZIP_DOMINANT_RATIO"
                --vision-zip-attention-layer "$RESOLVED_VISION_ZIP_ATTENTION_LAYER"
            )
            ;;
        cdpruner)
            METHOD_ARGS+=(
                --visual-token-pruning-method cdpruner
                --vt-prune-rate "$RESOLVED_VISUAL_TOKEN_PRUNING_RATE"
            )
            ;;
        *)
            echo "Unsupported VISUAL_TOKEN_PRUNING_METHOD from config: $RESOLVED_VISUAL_TOKEN_PRUNING_METHOD" >&2
            exit 2
            ;;
    esac

    if [[ "$RESOLVED_VISUAL_TOKEN_PRUNING_METHOD" != "none" ]]; then
        local prune_arch_json
        prune_arch_json=$(printf '{"architectures":["%s"]}' "$PRUNE_WRAPPER_ARCH")
        HF_OVERRIDES_ARGS+=(--hf-overrides "$prune_arch_json")
    fi

    # shellcheck disable=SC2206
    local -a EXTRA_ARR=()
    if [[ -n "${EXTRA_VLLM_ARGS}" ]]; then
        read -r -a EXTRA_ARR <<< "${EXTRA_VLLM_ARGS}"
    fi

    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
      vllm serve "$MODEL" \
        --tensor-parallel-size 1 \
        --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.85}" \
        --port "$port" \
        --enforce-eager \
        --enable-request-id-headers \
        --max-model-len "${MAX_MODEL_LEN:-65536}" \
        --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS:-32768}" \
        --max-num-seqs "${MAX_NUM_SEQS:-64}" \
        --limit-mm-per-prompt "$VLLM_LIMIT_MM_PER_PROMPT" \
        "${METHOD_ARGS[@]}" \
        "${HF_OVERRIDES_ARGS[@]}" \
        "${EXTRA_ARR[@]}" \
        >"${server_log}" 2>&1 &
    local server_pid=$!
    PIDS+=("$server_pid")

    wait_for_model_ready "$port" "$server_pid" "$server_log"
    verify_architecture "$normalized_method" "$server_log"

    echo "vLLM ready: method=${method} normalized=${normalized_method} port=${port}"
    if [[ "$RESOLVED_VISUAL_TOKEN_PRUNING_METHOD" == "none" ]]; then
        echo "  visual_token_pruning_method=none (disabled)"
    else
        echo "  visual_token_pruning_method=${RESOLVED_VISUAL_TOKEN_PRUNING_METHOD} rate=${RESOLVED_VISUAL_TOKEN_PRUNING_RATE}"
    fi
}

run_lmms_eval() {
    local mode="$1"
    local port="$2"

    export HF_HOME HF_DATASETS_CACHE HUGGINGFACE_HUB_CACHE OPENAI_API_KEY

    local outdir="${EVAL_OUTPUT_ROOT}/${START_TIME}_${mode}"
    mkdir -p "$outdir"

    local model_args="${LMMS_MODEL_ARGS_TEMPLATE//\{PORT\}/$port}"
    model_args="${model_args//\{MODEL\}/$MODEL}"

    # Ensure OpenAI-compatible backends are pinned to local vLLM unless
    # explicitly disabled.
    case "$LMMS_MODEL" in
        openai|openai_compatible|openai_compatible_chat|async_openai|async_openai_compatible|async_openai_compatible_chat)
            local local_base_url="http://127.0.0.1:${port}/v1"
            local canonical_model_args="model=${MODEL},base_url=${local_base_url},api_key=${OPENAI_API_KEY},temperature=0,max_new_tokens=128"
            export OPENAI_API_BASE="$local_base_url"

            if [[ "$FORCE_LOCAL_OPENAI_BASE_URL" == "1" ]]; then
                if [[ "$model_args" == *"base_url="* ]]; then
                    model_args=$(echo "$model_args" | sed -E "s|base_url=[^,]*|base_url=${local_base_url}|g")
                else
                    model_args="${model_args},base_url=${local_base_url}"
                fi
            fi

            if [[ "$model_args" != *"api_key="* ]]; then
                model_args="${model_args},api_key=${OPENAI_API_KEY}"
            fi

            # Force the `model=` field to match the served vLLM model id.
            # This avoids malformed template issues (for example model={MODEL,...).
            if [[ "$model_args" == *"model="* ]]; then
                model_args=$(echo "$model_args" | sed -E "s|model=[^,}]*|model=${MODEL}|g")
            else
                model_args="model=${MODEL},${model_args}"
            fi

            # If unresolved placeholders remain, use canonical args.
            if [[ "$model_args" == *"{MODEL"* || "$model_args" == *"{PORT"* ]]; then
                echo "Detected unresolved placeholders in LMMS model args; using canonical local OpenAI-compatible args."
                model_args="$canonical_model_args"
            fi

            # Guardrail: if args still point to api.openai.com, fail fast.
            if [[ "$model_args" == *"api.openai.com"* ]]; then
                echo "Refusing to run LMMS with remote OpenAI endpoint in local eval mode."
                echo "model_args=${model_args}"
                echo "Set FORCE_LOCAL_OPENAI_BASE_URL=1 or update LMMS_MODEL_ARGS_TEMPLATE."
                exit 2
            fi
            ;;
    esac

    local -a LIMIT_ARGS=()
    if [[ -n "${LMMS_LIMIT}" && "${LMMS_LIMIT}" != "0" ]]; then
        LIMIT_ARGS+=(--limit "$LMMS_LIMIT")
    fi

    local model_args_for_log="$model_args"
    model_args_for_log=$(echo "$model_args_for_log" | sed -E 's/(api_key=)[^,]*/\1***REDACTED***/g')
    echo "LMMS model args: ${model_args_for_log}"
    if [[ -n "${OPENAI_API_BASE:-}" ]]; then
        echo "LMMS OPENAI_API_BASE: ${OPENAI_API_BASE}"
    fi
    echo "Running LMMS-Eval (${mode})..."
    "$LMMS_PYTHON_BIN" -m lmms_eval \
        --model "$LMMS_MODEL" \
        --model_args "$model_args" \
        --tasks "$LMMS_TASKS" \
        --batch_size "$LMMS_BATCH_SIZE" \
        --output_path "$outdir" \
        --log_samples \
        "${LIMIT_ARGS[@]}"

    echo "LMMS-Eval output saved at: $outdir"
}

summarize_outputs() {
    local root="$1"
    local prefix="$2"
    "$LMMS_PYTHON_BIN" - "$root" "$prefix" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
prefix = sys.argv[2]

def collect_metrics(obj, prefix=""):
    out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            key = f"{prefix}.{k}" if prefix else k
            out.update(collect_metrics(v, key))
    elif isinstance(obj, (int, float)):
        lk = prefix.lower()
        if any(tok in lk for tok in ("acc", "score", "exact_match")):
            out[prefix] = obj
    return out

runs = sorted(p for p in root.iterdir() if p.is_dir() and p.name.startswith(prefix + "_"))
for run_dir in runs:
    json_files = sorted(run_dir.rglob("*.json"))
    if not json_files:
        print(f"[{run_dir.name}] no json outputs found")
        continue
    ranked = sorted(
        json_files,
        key=lambda p: ("result" not in p.name.lower(), "metric" not in p.name.lower(), -p.stat().st_size),
    )
    chosen = ranked[0]
    try:
        data = json.loads(chosen.read_text())
    except Exception as e:
        print(f"[{run_dir.name}] failed to parse {chosen}: {e}")
        continue
    metrics = collect_metrics(data)
    print(f"\n[{run_dir.name}] metrics from {chosen.name}")
    if not metrics:
        print("  (no accuracy-like numeric metrics auto-detected)")
        continue
    for k in sorted(metrics):
        print(f"  {k}: {metrics[k]}")
PY
}

usage() {
    cat <<'EOF'
Usage:
  bash epdtest/eval_lmms.sh [method ...]

Options (env):
  METHODS="none visionzip cdpruner"
  PRUNE_CONFIG_FILE PATH
  PRUNE_CONFIG_PARSER PATH
  LMMS_TASKS TASK_LIST
  LMMS_LIMIT INT
  -h, --help

Examples:
  bash epdtest/eval_lmms.sh none
  bash epdtest/eval_lmms.sh visionzip
  bash epdtest/eval_lmms.sh cdpruner
  bash epdtest/eval_lmms.sh none visionzip
  METHODS="visionzip cdpruner" bash epdtest/eval_lmms.sh

Notes:
  - This runs LMMS-Eval sequentially per pruning method.
  - Supported methods: none, visionzip, cdpruner.
  - Pruning knobs are always loaded from PRUNE_CONFIG_FILE.
  - Per-method outputs are written under EVAL_OUTPUT_ROOT.
EOF
}

###############################################################################
# Main
###############################################################################
# Positional methods override METHODS env for convenience.
if [[ $# -gt 0 ]]; then
    if [[ "$1" == "-h" || "$1" == "--help" ]]; then
        usage
        exit 0
    fi
    METHODS="$*"
fi

echo "Methods: ${METHODS}"
echo "Cache dirs:"
echo "  HF_HOME=$HF_HOME"
echo "  HF_DATASETS_CACHE=$HF_DATASETS_CACHE"
echo "  HUGGINGFACE_HUB_CACHE=$HUGGINGFACE_HUB_CACHE"
echo "  EVAL_OUTPUT_ROOT=$EVAL_OUTPUT_ROOT"
echo "  VLLM_LIMIT_MM_PER_PROMPT=$VLLM_LIMIT_MM_PER_PROMPT"
echo "  PRUNE_CONFIG_FILE=$PRUNE_CONFIG_FILE"
echo "  PRUNE_CONFIG_PARSER=$PRUNE_CONFIG_PARSER"

read -r -a METHOD_LIST <<< "${METHODS}"
if [[ ${#METHOD_LIST[@]} -eq 0 || -z "${METHOD_LIST[0]:-}" ]]; then
    echo "METHODS is empty. Example: METHODS='none visionzip' bash ./epdtest/eval_lmms.sh" >&2
    exit 2
fi

ensure_pruning_config_files
enforce_config_only_pruning_knobs
idx=0
for method in "${METHOD_LIST[@]}"; do
    port=$((BASE_PORT + idx))
    idx=$((idx + 1))
    log="${LOG_PATH}/lmms_eval_${method}_${START_TIME}.log"
    echo "==== LMMS-Eval run: ${method} (port ${port}) ===="
    start_vllm_server "$method" "$port" "$log"
    run_lmms_eval "$method" "$port"
    stop_server
done

echo "==== Summary (auto-detected metrics for this batch) ===="
summarize_outputs "$EVAL_OUTPUT_ROOT" "$START_TIME"

echo "Done."
