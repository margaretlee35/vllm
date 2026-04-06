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

declare -a INSTALL_ARGS=()
if [[ "${SKIP_INSTALL:-0}" != "1" ]]; then
    INSTALL_ARGS+=(--install)
fi

bash ./epdtest/run.sh \
    --topology "${TOPOLOGY:-1e1pd}" \
    --profile "${PROFILE:-randommm}" \
    "${INSTALL_ARGS[@]}"
