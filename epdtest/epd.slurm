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

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
cd "$REPO_ROOT"

export TOPOLOGY="${TOPOLOGY:-1e1pd}"
export BENCHMARK="${BENCHMARK:-randommm}"
export PRUNE_MODE="${PRUNE_MODE:-none}"

bash ./epdtest/sweep.sh "$PRUNE_MODE"
