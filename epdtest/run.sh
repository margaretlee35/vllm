#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
GIT_ROOT=$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)
IMPL_DIR="$GIT_ROOT/epdtest/scripts"
VENV_ACTIVATE="$GIT_ROOT/.venv/bin/activate"
PRUNE_CONFIG_FILE="$SCRIPT_DIR/visual_token_pruning_configs.json"
PRUNE_CONFIG_PARSER="$SCRIPT_DIR/parse_config.py"

TOPOLOGY="${TOPOLOGY:-1e1pd}"
BENCHMARK="${BENCHMARK:-simple}"
MODEL="${MODEL:-Qwen/Qwen2.5-VL-3B-Instruct}"
LOG_PATH="${LOG_PATH:-$GIT_ROOT/epdtest/logs}"
IMAGES_PER_REQ="${IMAGES_PER_REQ:-1}"

if [[ ! -f "$PRUNE_CONFIG_FILE" ]]; then
    echo "Missing pruning config file: $PRUNE_CONFIG_FILE" >&2
    exit 1
fi

if [[ ! -f "$PRUNE_CONFIG_PARSER" ]]; then
    echo "Missing pruning config parser: $PRUNE_CONFIG_PARSER" >&2
    exit 1
fi

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
        export VISUAL_TOKEN_PRUNING_METHOD="none"
        export VISUAL_TOKEN_PRUNING_RATE=""
        export VISION_ZIP_DOMINANT_RATIO=""
        export VISION_ZIP_ATTENTION_LAYER=""
        return 0
    fi

    export VISUAL_TOKEN_PRUNING_METHOD="$cfg_method"

    # Allow env overrides when explicitly provided; otherwise use JSON defaults.
    export VISUAL_TOKEN_PRUNING_RATE="${VISUAL_TOKEN_PRUNING_RATE:-$cfg_rate}"
    if [[ "$cfg_method" == "visionzip" ]]; then
        export VISION_ZIP_DOMINANT_RATIO="${VISION_ZIP_DOMINANT_RATIO:-$cfg_dom_ratio}"
        export VISION_ZIP_ATTENTION_LAYER="${VISION_ZIP_ATTENTION_LAYER:-$cfg_attn_layer}"
    else
        export VISION_ZIP_DOMINANT_RATIO=""
        export VISION_ZIP_ATTENTION_LAYER=""
    fi
}

usage() {
    cat <<'EOF'
Usage:
  bash epdtest/run.sh [options]

Options:
  --topology 1e1pd|1e1p1d|1ed1p
  --benchmark simple|randommm
  --images-per-req N
  --visual-token-pruning-method visionzip|cdpruner|none
  -h, --help

Examples:
  bash epdtest/run.sh
  bash epdtest/run.sh --benchmark randommm
  bash epdtest/run.sh --topology 1e1p1d --benchmark randommm
  bash epdtest/run.sh --topology 1ed1p --benchmark randommm
  bash epdtest/run.sh --benchmark randommm --images-per-req 4
  bash epdtest/run.sh --visual-token-pruning-method visionzip
  bash epdtest/run.sh --visual-token-pruning-method none

Notes:
  - This is the main epdtest entrypoint.
  - Pruning defaults are loaded from epdtest/visual_token_pruning_configs.json.
  - The old scripts under examples/.../lovelace are preserved only as wrappers.
  - Model and log path still stay configurable through environment variables.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --topology)
            TOPOLOGY="$2"
            shift 2
            ;;
        --benchmark)
            BENCHMARK="$2"
            shift 2
            ;;
        --images-per-req)
            IMAGES_PER_REQ="$2"
            shift 2
            ;;
        --visual-token-pruning-method)
            export VISUAL_TOKEN_PRUNING_METHOD="${2,,}"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

case "$TOPOLOGY" in
    1e1pd|1e1p1d|1ed1p)
        ;;
    *)
        echo "Unsupported topology: $TOPOLOGY" >&2
        echo "Supported topology values: 1e1pd, 1e1p1d, 1ed1p" >&2
        exit 1
        ;;
esac

case "$BENCHMARK" in
    simple|default)
        BENCHMARK="simple"
        ;;
    randommm|metrics|rmm)
        BENCHMARK="randommm"
        ;;
    *)
        echo "Unsupported benchmark mode: $BENCHMARK" >&2
        exit 1
        ;;
esac

if ! apply_visual_token_pruning_config "${VISUAL_TOKEN_PRUNING_METHOD:-}"; then
    exit 1
fi

if [[ -z "${NUM_PROMPTS:-}" ]]; then
    export NUM_PROMPTS=300
fi

if [[ -z "${TIMEOUT_SECONDS:-}" ]]; then
    export TIMEOUT_SECONDS=120
fi

export TOPOLOGY
export BENCHMARK
export MODEL
export LOG_PATH

mkdir -p "$LOG_PATH"

activate_venv() {
    if [[ -n "${VIRTUAL_ENV:-}" ]]; then
        return 0
    fi

    if [[ -f "$VENV_ACTIVATE" ]]; then
        # shellcheck disable=SC1090
        source "$VENV_ACTIVATE"
    fi
}

case "${TOPOLOGY}:${BENCHMARK}" in
    1e1pd:simple)
        TARGET_SCRIPT="$IMPL_DIR/disagg_1e1pd.sh"
        ;;
    1e1pd:randommm)
        TARGET_SCRIPT="$IMPL_DIR/disagg_1e1pd.sh"
        ;;
    1e1p1d:simple)
        TARGET_SCRIPT="$IMPL_DIR/disagg_1e1p1d.sh"
        ;;
    1e1p1d:randommm)
        TARGET_SCRIPT="$IMPL_DIR/disagg_1e1p1d.sh"
        ;;
    1ed1p:simple)
        TARGET_SCRIPT="$IMPL_DIR/disagg_1ed1p.sh"
        ;;
    1ed1p:randommm)
        TARGET_SCRIPT="$IMPL_DIR/disagg_1ed1p.sh"
        ;;
    *)
        echo "Unsupported mode combination: ${TOPOLOGY}:${BENCHMARK}" >&2
        exit 1
        ;;
esac

RUN_STAMP="${RUN_STAMP:-$(date +"%Y%m%d_%H%M%S")}"
RUN_DIR="${RUN_DIR:-${LOG_PATH}/${RUN_STAMP}}"
mkdir -p "$RUN_DIR"
TARGET_OUTPUT_LOG="$RUN_DIR/target_script.log"

activate_venv
cd "$GIT_ROOT"

echo "epdtest launcher"
echo "  topology       : $TOPOLOGY"
echo "  benchmark      : $BENCHMARK"
echo "  model          : $MODEL"
echo "  log_path       : ${LOG_PATH#$GIT_ROOT/}"
echo "  run_dir        : ${RUN_DIR#$GIT_ROOT/}"
if [[ -n "${VISUAL_TOKEN_PRUNING_METHOD:-}" ]]; then
    echo "  vt_method      : $VISUAL_TOKEN_PRUNING_METHOD"
    echo "  vt_config      : ${PRUNE_CONFIG_FILE#$GIT_ROOT/}"
fi
if [[ "$BENCHMARK" == "randommm" ]]; then
    echo "  images_per_req : $IMAGES_PER_REQ"
fi
if [[ -n "${VISUAL_TOKEN_PRUNING_RATE:-}" ]]; then
    echo "  vt_rate        : $VISUAL_TOKEN_PRUNING_RATE"
fi
if [[ -n "${VISION_ZIP_DOMINANT_RATIO:-}" ]]; then
    echo "  vz_dom_ratio   : $VISION_ZIP_DOMINANT_RATIO"
fi
if [[ -n "${VISION_ZIP_ATTENTION_LAYER:-}" ]]; then
    echo "  vz_attn_layer  : $VISION_ZIP_ATTENTION_LAYER"
fi
echo "  script         : ${TARGET_SCRIPT#$GIT_ROOT/}"
echo "  target_output  : ${TARGET_OUTPUT_LOG#$GIT_ROOT/}"

bash "$TARGET_SCRIPT" 2>&1 | tee "$TARGET_OUTPUT_LOG"
