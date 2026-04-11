#!/bin/bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  bash epdtest/sweep.sh visionzip
  bash epdtest/sweep.sh cdpruner
  bash epdtest/sweep.sh none
EOF
}

if [[ $# -lt 1 ]]; then
    usage >&2
    exit 1
fi

PRUNE_MODE=""
PRUNE_MODE_INPUT="${1,,}"
case "$PRUNE_MODE_INPUT" in
    visionzip)
        PRUNE_MODE="visionzip"
        export VISUAL_TOKEN_PRUNING_METHOD="visionzip"
        ;;
    cdpruner)
        PRUNE_MODE="cdpruner"
        export VISUAL_TOKEN_PRUNING_METHOD="cdpruner"
        ;;
    none)
        PRUNE_MODE="none"
        export VISUAL_TOKEN_PRUNING_METHOD="none"
        ;;
    -h|--help)
        usage
        exit 0
        ;;
    *)
        echo "Unsupported pruning method: $1" >&2
        usage >&2
        exit 1
        ;;
esac

export REPO_ROOT="${REPO_ROOT:-.}"
export TOPOLOGY="${TOPOLOGY:-1e1pd}"
export BENCHMARK="${BENCHMARK:-randommm}"
export MODEL="${MODEL:-Qwen/Qwen2.5-VL-3B-Instruct}"

export GPU_E="${GPU_E:-0}"
export GPU_PD="${GPU_PD:-1}"
export GPU_P="${GPU_P:-0}"
export GPU_D="${GPU_D:-1}"

export NUM_PROMPTS="${NUM_PROMPTS:-300}"
export TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-120}"
export LOG_PATH="${LOG_PATH:-$REPO_ROOT/epdtest/logs}"

IMAGES_PER_REQ_LIST="${IMAGES_PER_REQ_LIST:-1 2 4 8}"
RUN_STAMP=$(date +"%Y%m%d_%H%M%S")
mkdir -p "$LOG_PATH"

declare -a BASE_RUN_ARGS=(
    --topology "$TOPOLOGY"
    --benchmark "$BENCHMARK"
)

if [[ -n "${VISUAL_TOKEN_PRUNING_METHOD:-}" ]]; then
    BASE_RUN_ARGS+=(--visual-token-pruning-method "$VISUAL_TOKEN_PRUNING_METHOD")
fi

echo "epdtest: topology=$TOPOLOGY benchmark=$BENCHMARK model=$MODEL"
if [[ "$PRUNE_MODE" == "none" ]]; then
    echo "epdtest: visual_token_pruning_method=none (disabled)"
else
    echo "epdtest: visual_token_pruning_method=$VISUAL_TOKEN_PRUNING_METHOD"
fi

if [[ "$BENCHMARK" == "randommm" ]]; then
    echo "epdtest: images_per_req=$IMAGES_PER_REQ_LIST"
    for images_per_req in $IMAGES_PER_REQ_LIST; do
        export IMAGES_PER_REQ="$images_per_req"
        export RUN_STAMP
        export RUN_DIR="${LOG_PATH}/${RUN_STAMP}/ipr${images_per_req}"
        mkdir -p "$RUN_DIR"

        echo "=== IMAGES_PER_REQ=$IMAGES_PER_REQ ==="
        bash ./epdtest/run.sh \
            "${BASE_RUN_ARGS[@]}" \
            --images-per-req "$IMAGES_PER_REQ" \
            2>&1
    done
else
    export RUN_STAMP
    export RUN_DIR="${LOG_PATH}/${RUN_STAMP}"
    mkdir -p "$RUN_DIR"

    echo "=== single run ==="
    bash ./epdtest/run.sh "${BASE_RUN_ARGS[@]}" 2>&1
fi
