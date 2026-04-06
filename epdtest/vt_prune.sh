
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  bash epdtest/vt_prune.sh visionzip
  bash epdtest/vt_prune.sh cdprune
  bash epdtest/vt_prune.sh noprune

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
        export VISUAL_TOKEN_PRUNING_METHOD="noprune"
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
export PROFILE="${PROFILE:-metrics}"
export MODEL="${MODEL:-Qwen/Qwen2.5-VL-3B-Instruct}"

export GPU_E="${GPU_E:-0}"
export GPU_PD="${GPU_PD:-1}"
export GPU_P="${GPU_P:-0}"
export GPU_D="${GPU_D:-1}"

export NUM_PROMPTS="${NUM_PROMPTS:-500}"
export TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-120}"
export LOG_PATH="${LOG_PATH:-$REPO_ROOT/epdtest/logs}"

export VISION_ZIP_RATE="${VISION_ZIP_RATE:-0.5}"
export VISION_ZIP_DOMINANT_RATIO="${VISION_ZIP_DOMINANT_RATIO:-0.75}"
export VISION_ZIP_ATTENTION_LAYER="${VISION_ZIP_ATTENTION_LAYER:--2}"

if [[ "$PRUNE_MODE" == "noprune" ]]; then
    # Downstream scripts default pruning to vision_zip; keep this mode local by
    # stripping all visual-token-pruning CLI args at the vllm call boundary.
    export VISION_ZIP_RATE=""
    export VISION_ZIP_DOMINANT_RATIO=""
    export VISION_ZIP_ATTENTION_LAYER=""

    vllm() {
        local -a filtered_args=()
        local skip_next=0
        local arg
        for arg in "$@"; do
            if (( skip_next )); then
                skip_next=0
                continue
            fi
            case "$arg" in
                --visual-token-pruning-method|--vt-prune-rate|--vision-zip-dominant-ratio|--vision-zip-attention-layer)
                    skip_next=1
                    continue
                    ;;
                --visual-token-pruning-method=*|--vt-prune-rate=*|--vision-zip-dominant-ratio=*|--vision-zip-attention-layer=*)
                    continue
                    ;;
            esac
            filtered_args+=("$arg")
        done
        command vllm "${filtered_args[@]}"
    }
    export -f vllm
fi

IMAGES_PER_REQ_LIST="${IMAGES_PER_REQ_LIST:-1 2 4 8}"
RUN_STAMP=$(date +"%Y%m%d_%H%M%S")
SLURM_LOG_DIR="${SLURM_LOG_DIR:-$REPO_ROOT/epdtest/slurm_logs}"
mkdir -p "$LOG_PATH" "$SLURM_LOG_DIR"

echo "epdtest: topology=$TOPOLOGY profile=$PROFILE model=$MODEL"
if [[ "$PRUNE_MODE" == "noprune" ]]; then
    echo "epdtest: visual_token_pruning_method=disabled (noprune)"
else
    echo "epdtest: visual_token_pruning_method=$VISUAL_TOKEN_PRUNING_METHOD"
fi
echo "epdtest: images_per_req=$IMAGES_PER_REQ_LIST"

for images_per_req in $IMAGES_PER_REQ_LIST; do
    export IMAGES_PER_REQ="$images_per_req"
    run_log="$SLURM_LOG_DIR/${TOPOLOGY}_${PROFILE}_ipr${images_per_req}_${RUN_STAMP}.log"

    echo "=== IMAGES_PER_REQ=$IMAGES_PER_REQ ===" | tee "$run_log"
    bash ./epdtest/run.sh \
        --topology "$TOPOLOGY" \
        --profile "$PROFILE" \
        2>&1 | tee -a "$run_log"
done
