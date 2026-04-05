#!/bin/bash
#SBATCH -J epdtest
#SBATCH -o epdtest.%j.out
#SBATCH -e epdtest.%j.err
#SBATCH -p amd-rtx
#SBATCH -N 1
#SBATCH -n 6
#SBATCH -t 24:00:00
#SBATCH --mail-user=margaretlee@utexas.edu
#SBATCH --mail-type=all

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/work2/10642/margaretlee35/stampede3/vllm}"
cd "$REPO_ROOT"
source .venv/bin/activate

if [[ "${SKIP_INSTALL:-0}" != "1" ]]; then
    VLLM_USE_PRECOMPILED=1 uv pip install --editable .
    uv pip install "vllm[bench]"
fi

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

IMAGES_PER_REQ_LIST="${IMAGES_PER_REQ_LIST:-1 2 4 8}"
RUN_STAMP=$(date +"%Y%m%d_%H%M%S")
SLURM_LOG_DIR="${SLURM_LOG_DIR:-$REPO_ROOT/epdtest/slurm_logs}"
mkdir -p "$LOG_PATH" "$SLURM_LOG_DIR"

echo "epdtest: topology=$TOPOLOGY profile=$PROFILE model=$MODEL"
echo "epdtest: images_per_req=$IMAGES_PER_REQ_LIST"

for images_per_req in $IMAGES_PER_REQ_LIST; do
    export IMAGES_PER_REQ="$images_per_req"
    run_log="$SLURM_LOG_DIR/${TOPOLOGY}_${PROFILE}_ipr${images_per_req}_${RUN_STAMP}.log"

    echo "=== IMAGES_PER_REQ=$IMAGES_PER_REQ ===" | tee "$run_log"
    uv run bash ./epdtest/run.sh \
        --topology "$TOPOLOGY" \
        --profile "$PROFILE" \
        2>&1 | tee -a "$run_log"
done
