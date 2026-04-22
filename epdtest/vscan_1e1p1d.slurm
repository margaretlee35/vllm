#!/bin/bash
#SBATCH -J vscan-1e1p1d
#SBATCH -o vscan-1e1p1d.%j.out
#SBATCH -e vscan-1e1p1d.%j.err
#SBATCH -p gpu-a100
#SBATCH -N 1
#SBATCH -n 3
#SBATCH -t 02:00:00
#SBATCH --mail-user=varun.bala@utexas.edu
#SBATCH --mail-type=all

set -euo pipefail

# ---------------------------------------------------------------------------
# Environment setup — redirect ALL caches to $SCRATCH to avoid $HOME quota
# ---------------------------------------------------------------------------
export HF_HOME=$SCRATCH/.cache/huggingface
export HF_HUB_CACHE=$SCRATCH/.cache/huggingface/hub
export HUGGINGFACE_HUB_CACHE=$SCRATCH/.cache/huggingface/hub
export XDG_CACHE_HOME=$SCRATCH/.cache
export TRITON_CACHE_DIR=$SCRATCH/.cache/triton
export UV_CACHE_DIR=$SCRATCH/.cache/uv
export VLLM_USAGE_STATS_SERVER=""
export DO_NOT_TRACK=1

# Symlink ~/.cache, ~/.triton, ~/.nv → $SCRATCH (some libs ignore env vars)
for d in .cache .triton .nv; do
    target="$SCRATCH/$d"
    link="$HOME/$d"
    mkdir -p "$target"
    if [[ -L "$link" ]]; then
        : # already a symlink, skip
    elif [[ -d "$link" ]]; then
        rm -rf "$link"
    fi
    ln -sfn "$target" "$link"
done

# Conda activation — try hook first, fall back to direct source
eval "$(conda shell.bash hook 2>/dev/null)" || true
conda activate vllm 2>/dev/null || source /work/10299/varumugam/ls6/miniforge3/envs/vllm/bin/activate

# Verify conda env is active
if [[ "$(python --version 2>&1)" != *"3.11"* ]]; then
    echo "ERROR: Python 3.11 not active. Conda activation failed." >&2
    exit 1
fi

# Fix libstdc++ mismatch with TACC system libs
export LD_PRELOAD=/work/10299/varumugam/ls6/miniforge3/envs/vllm/lib/libstdc++.so.6

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
REPO_ROOT="/work/10299/varumugam/ls6/vllm"
cd "$REPO_ROOT"

export TOPOLOGY="1e1p1d"
export BENCHMARK="randommm"
export PRUNE_MODE="vscan"
export TIMEOUT_SECONDS=600
export SERVER_READY_TIMEOUT_SECONDS=300

# 1e1p1d: encoder on GPU 0, prefill on GPU 1, decode on GPU 2
export GPU_E="0"
export GPU_P="1"
export GPU_D="2"

echo "=== VScan 1e1p1d sweep starting at $(date) ==="
echo "Node: $(hostname)"
echo "GPUs: $(nvidia-smi --query-gpu=index,name --format=csv,noheader)"
echo "Topology: $TOPOLOGY (GPU_E=$GPU_E GPU_P=$GPU_P GPU_D=$GPU_D)"
echo ""

bash ./epdtest/sweep.sh "$PRUNE_MODE"

echo ""
echo "=== VScan 1e1p1d sweep finished at $(date) ==="
