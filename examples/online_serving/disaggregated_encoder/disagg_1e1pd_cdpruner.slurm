#!/bin/bash
#SBATCH -J disagg-1e1pd-cdpruner
#SBATCH -o disagg-1e1pd-cdpruner.%j.out
#SBATCH -e disagg-1e1pd-cdpruner.%j.err
#SBATCH -p amd-rtx
#SBATCH -N 1
#SBATCH -n 6
#SBATCH -t 24:00:00

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../../.." && pwd)
cd "$REPO_ROOT"

# Force CDPruner on the 1e1pd example flow.
export VISUAL_TOKEN_PRUNING_METHOD="cdpruner"
export VISUAL_TOKEN_PRUNING_RATE="${VISUAL_TOKEN_PRUNING_RATE:-0.5}"

# Optional runtime overrides.
export MODEL="${MODEL:-Qwen/Qwen2.5-VL-3B-Instruct}"
export NUM_PROMPTS="${NUM_PROMPTS:-100}"

bash ./examples/online_serving/disaggregated_encoder/disagg_1e1pd_example.sh
