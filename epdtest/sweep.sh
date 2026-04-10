#!/bin/bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  bash epdtest/sweep.sh visionzip
  bash epdtest/sweep.sh cdprune
  bash epdtest/sweep.sh noprune

Accepted method aliases:
  visionzip | vision_zip
  cdprune   | cdpruner
  noprune   | no_prune | none
EOF
}

if [[ $# -lt 1 ]]; then
    usage >&2
    exit 1
fi

PRUNE_MODE=""
case "$1" in
    visionzip|vision_zip)
        PRUNE_MODE="vision_zip"
        export VISUAL_TOKEN_PRUNING_METHOD="vision_zip"
        ;;
    cdprune|cdpruner)
        PRUNE_MODE="cdpruner"
        export VISUAL_TOKEN_PRUNING_METHOD="cdpruner"
        ;;
    noprune|no_prune|none)
        PRUNE_MODE="noprune"
        export VISUAL_TOKEN_PRUNING_METHOD=""
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
export PROFILE="${PROFILE:-randommm}"
export MODEL="${MODEL:-Qwen/Qwen2.5-VL-3B-Instruct}"

export GPU_E="${GPU_E:-0}"
export GPU_PD="${GPU_PD:-1}"
export GPU_P="${GPU_P:-1}"
export GPU_D="${GPU_D:-2}"

export NUM_PROMPTS="${NUM_PROMPTS:-500}"
export TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-120}"
export LOG_PATH="${LOG_PATH:-$REPO_ROOT/epdtest/logs}"

export VISUAL_TOKEN_PRUNING_RATE="${VISUAL_TOKEN_PRUNING_RATE:-0.5}"
export VISION_ZIP_DOMINANT_RATIO="${VISION_ZIP_DOMINANT_RATIO:-0.75}"
export VISION_ZIP_ATTENTION_LAYER="${VISION_ZIP_ATTENTION_LAYER:--2}"

if [[ "$PRUNE_MODE" == "noprune" ]]; then
    export VISUAL_TOKEN_PRUNING_RATE=""
    export VISION_ZIP_DOMINANT_RATIO=""
    export VISION_ZIP_ATTENTION_LAYER=""
fi

IMAGES_PER_REQ_LIST="${IMAGES_PER_REQ_LIST:-1 2 4 8}"
RUN_STAMP=$(date +"%Y%m%d_%H%M%S")
mkdir -p "$LOG_PATH"

declare -a BASE_RUN_ARGS=(
    --topology "$TOPOLOGY"
    --profile "$PROFILE"
)

if [[ -n "${VISUAL_TOKEN_PRUNING_METHOD:-}" ]]; then
    BASE_RUN_ARGS+=(--visual-token-pruning-method "$VISUAL_TOKEN_PRUNING_METHOD")
fi

echo "epdtest: topology=$TOPOLOGY profile=$PROFILE model=$MODEL"
if [[ "$PRUNE_MODE" == "noprune" ]]; then
    echo "epdtest: visual_token_pruning_method=disabled (noprune)"
else
    echo "epdtest: visual_token_pruning_method=$VISUAL_TOKEN_PRUNING_METHOD"
fi

if [[ "$PROFILE" == "randommm" ]]; then
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
