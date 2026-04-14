#!/bin/bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  bash epdtest/sweep_compare.sh

This script sweeps randommm runs across:
  - 9 topology/GPU layouts
  - IMAGES_PER_REQ = 1, 2, 4, 8

Fixed benchmark settings:
  TIMEOUT_SECONDS=600
  BENCH_REQUEST_RATE=8
  BENCH_MAX_CONCURRENCY=32
  --benchmark randommm
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)
cd "$REPO_ROOT"

MODEL="${MODEL:-Qwen/Qwen2.5-VL-3B-Instruct}"
LOG_PATH="${LOG_PATH:-$REPO_ROOT/epdtest/logs}"
RUN_STAMP="${RUN_STAMP:-$(date +"%Y%m%d_%H%M%S")}"

TIMEOUT_SECONDS=600
BENCH_REQUEST_RATE=8
BENCH_MAX_CONCURRENCY=32
BENCHMARK="randommm"
IMAGES_PER_REQ_LIST=(1)

declare -a CASE_NAMES=(
    "1e1p1d_e0_p1_d2"
    "1e1pNd_e0_p1_d0-2"
    "1e1pNd_d_preempt_e0_p1_d0-2"
    "Ne1p1d_e0-1_p1_d2"
    "Ne1p1d_e0-2_p1_d2"
    "Ne1p1d_e0-1-2_p1_d2"
    "Ne1p1d_pd_preempt_e0-1-2_p1_d2"
    "Ne1pNd_e0-1_p1_d0-2"
    "Ne1pNd_pd_preempt_e0-1_p1_d0-2"
)

declare -a CASE_TOPOLOGIES=(
    "1e1p1d"
    "1e1pNd"
    "1e1pNd_d_preempt"
    "Ne1p1d"
    "Ne1p1d"
    "Ne1p1d"
    "Ne1p1d_pd_preempt"
    "Ne1pNd"
    "Ne1pNd_pd_preempt"
)

declare -a CASE_GPU_E=(
    "0"
    "0"
    "0"
    "0,1"
    "0,2"
    "0,1,2"
    "0,1,2"
    "0,1"
    "0,1"
)

declare -a CASE_GPU_P=(
    "1"
    "1"
    "1"
    "1"
    "1"
    "1"
    "1"
    "1"
    "1"
)

declare -a CASE_GPU_D=(
    "2"
    "0,2"
    "0,2"
    "2"
    "2"
    "2"
    "2"
    "0,2"
    "0,2"
)

SWEEP_ROOT="$LOG_PATH/$RUN_STAMP/compare"
mkdir -p "$SWEEP_ROOT"

echo "sweep_compare"
echo "  model               : $MODEL"
echo "  log_path            : ${LOG_PATH#$REPO_ROOT/}"
echo "  run_stamp           : $RUN_STAMP"
echo "  sweep_root          : ${SWEEP_ROOT#$REPO_ROOT/}"
echo "  benchmark           : $BENCHMARK"
echo "  timeout_seconds     : $TIMEOUT_SECONDS"
echo "  bench_request_rate  : $BENCH_REQUEST_RATE"
echo "  bench_max_conc      : $BENCH_MAX_CONCURRENCY"
echo "  images_per_req_list : ${IMAGES_PER_REQ_LIST[*]}"

for images_per_req in "${IMAGES_PER_REQ_LIST[@]}"; do
    echo
    echo "=== IMAGES_PER_REQ=$images_per_req ==="

    for idx in "${!CASE_NAMES[@]}"; do
        case_name="${CASE_NAMES[$idx]}"
        topology="${CASE_TOPOLOGIES[$idx]}"
        gpu_e="${CASE_GPU_E[$idx]}"
        gpu_p="${CASE_GPU_P[$idx]}"
        gpu_d="${CASE_GPU_D[$idx]}"

        run_dir="$SWEEP_ROOT/$case_name/ipr${images_per_req}"
        mkdir -p "$run_dir"

        echo "--- case=$case_name topology=$topology GPU_E=$gpu_e GPU_P=$gpu_p GPU_D=$gpu_d ---"

        TIMEOUT_SECONDS="$TIMEOUT_SECONDS" \
        BENCH_REQUEST_RATE="$BENCH_REQUEST_RATE" \
        BENCH_MAX_CONCURRENCY="$BENCH_MAX_CONCURRENCY" \
        BENCHMARK="$BENCHMARK" \
        MODEL="$MODEL" \
        IMAGES_PER_REQ="$images_per_req" \
        GPU_E="$gpu_e" \
        GPU_P="$gpu_p" \
        GPU_D="$gpu_d" \
        RUN_DIR="$run_dir" \
        bash ./epdtest/run.sh \
            --topology "$topology" \
            --benchmark "$BENCHMARK" \
            --images-per-req "$images_per_req" \
            2>&1 | tee "$run_dir/launcher.log"
    done
done

echo
echo "Sweep complete."
echo "Results root: ${SWEEP_ROOT#$REPO_ROOT/}"
