#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
GIT_ROOT=$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)
IMPL_DIR="$GIT_ROOT/epdtest/scripts"
VENV_ACTIVATE="$GIT_ROOT/.venv/bin/activate"

TOPOLOGY="${TOPOLOGY:-1e1pd}"
BENCHMARK="${BENCHMARK:-simple}"
MODEL="${MODEL:-Qwen/Qwen2.5-VL-3B-Instruct}"
LOG_PATH="${LOG_PATH:-$GIT_ROOT/epdtest/logs}"
IMAGES_PER_REQ="${IMAGES_PER_REQ:-1}"
TARGET_SCRIPT=""
RUN_STAMP=""
RUN_DIR=""
TARGET_OUTPUT_LOG=""

die() {
    echo "$*" >&2
    exit 1
}

usage() {
    cat <<'EOF'
Usage:
  bash epdtest/run.sh [options]

Options:
  --topology 1e1pd|1e1p1d|1e1pNd|Ne1p1d|1ed1p
  --benchmark simple|randommm
  --images-per-req N
  --visual-token-pruning-method visionzip|cdpruner|none
  -h, --help

Examples:
  bash epdtest/run.sh
  bash epdtest/run.sh --benchmark randommm
  bash epdtest/run.sh --topology 1e1p1d --benchmark randommm
  bash epdtest/run.sh --topology 1e1pNd --benchmark randommm
  bash epdtest/run.sh --topology Ne1p1d --benchmark randommm
  bash epdtest/run.sh --topology 1ed1p --benchmark randommm
  bash epdtest/run.sh --benchmark randommm --images-per-req 4
  bash epdtest/run.sh --visual-token-pruning-method visionzip
  bash epdtest/run.sh --visual-token-pruning-method none

Notes:
  - This is the main epdtest entrypoint.
  - Pruning defaults are loaded by each script in epdtest/scripts.
  - The old scripts under examples/.../lovelace are preserved only as wrappers.
  - Model and log path still stay configurable through environment variables.
EOF
}

require_option_value() {
    local flag="$1"
    local value="${2:-}"
    if [[ -z "$value" || "$value" == --* ]]; then
        die "Missing value for $flag"
    fi
}

parse_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --topology)
                require_option_value "$1" "${2:-}"
                TOPOLOGY="$2"
                shift 2
                ;;
            --benchmark)
                require_option_value "$1" "${2:-}"
                BENCHMARK="$2"
                shift 2
                ;;
            --images-per-req)
                require_option_value "$1" "${2:-}"
                IMAGES_PER_REQ="$2"
                shift 2
                ;;
            --visual-token-pruning-method)
                require_option_value "$1" "${2:-}"
                export VISUAL_TOKEN_PRUNING_METHOD="${2,,}"
                shift 2
                ;;
            -h|--help)
                usage
                exit 0
                ;;
            *)
                usage >&2
                die "Unknown argument: $1"
                ;;
        esac
    done
}

validate_topology() {
    case "${TOPOLOGY,,}" in
        1e1pd)
            TOPOLOGY="1e1pd"
            ;;
        1e1p1d)
            TOPOLOGY="1e1p1d"
            ;;
        1e1pnd)
            TOPOLOGY="1e1pNd"
            ;;
        ne1p1d)
            TOPOLOGY="Ne1p1d"
            ;;
        1ed1p)
            TOPOLOGY="1ed1p"
            ;;
        *)
            die "Unsupported topology: $TOPOLOGY"$'\n'"Supported topology values: 1e1pd, 1e1p1d, 1e1pNd, Ne1p1d, 1ed1p"
            ;;
    esac
}

normalize_benchmark() {
    case "$BENCHMARK" in
        simple|default)
            BENCHMARK="simple"
            ;;
        randommm|metrics|rmm)
            BENCHMARK="randommm"
            ;;
        *)
            die "Unsupported benchmark mode: $BENCHMARK"
            ;;
    esac
}

set_runtime_defaults() {
    export NUM_PROMPTS="${NUM_PROMPTS:-300}"
    export TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-120}"
}

activate_venv() {
    if [[ -n "${VIRTUAL_ENV:-}" ]]; then
        return 0
    fi

    if [[ -f "$VENV_ACTIVATE" ]]; then
        # shellcheck disable=SC1090
        source "$VENV_ACTIVATE"
    fi
}

resolve_target_script() {
    TARGET_SCRIPT="$IMPL_DIR/disagg_${TOPOLOGY}.sh"
    if [[ ! -f "$TARGET_SCRIPT" ]]; then
        die "Missing target script for topology '$TOPOLOGY': $TARGET_SCRIPT"
    fi
}

setup_run_paths() {
    RUN_STAMP="${RUN_STAMP:-$(date +"%Y%m%d_%H%M%S")}"
    RUN_DIR="${RUN_DIR:-${LOG_PATH}/${RUN_STAMP}}"
    mkdir -p "$RUN_DIR"
    TARGET_OUTPUT_LOG="$RUN_DIR/target_script.log"
}

print_launcher_info() {
    echo "epdtest launcher"
    echo "  topology       : $TOPOLOGY"
    echo "  benchmark      : $BENCHMARK"
    echo "  model          : $MODEL"
    echo "  log_path       : ${LOG_PATH#$GIT_ROOT/}"
    echo "  run_dir        : ${RUN_DIR#$GIT_ROOT/}"
    if [[ -n "${VISUAL_TOKEN_PRUNING_METHOD:-}" ]]; then
        echo "  vt_method      : $VISUAL_TOKEN_PRUNING_METHOD"
    fi
    if [[ "$BENCHMARK" == "randommm" ]]; then
        echo "  images_per_req : $IMAGES_PER_REQ"
    fi
    echo "  script         : ${TARGET_SCRIPT#$GIT_ROOT/}"
    echo "  target_output  : ${TARGET_OUTPUT_LOG#$GIT_ROOT/}"
}

main() {
    parse_args "$@"
    validate_topology
    normalize_benchmark
    set_runtime_defaults

    export TOPOLOGY
    export BENCHMARK
    export MODEL
    export LOG_PATH
    export IMAGES_PER_REQ

    mkdir -p "$LOG_PATH"
    resolve_target_script
    setup_run_paths
    activate_venv

    cd "$GIT_ROOT"
    print_launcher_info
    bash "$TARGET_SCRIPT" 2>&1 | tee "$TARGET_OUTPUT_LOG"
}

main "$@"
