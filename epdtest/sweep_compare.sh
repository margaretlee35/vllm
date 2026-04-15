#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)
cd "$REPO_ROOT"

MODEL="${MODEL:-Qwen/Qwen2.5-VL-3B-Instruct}"
LOG_PATH="${LOG_PATH:-$REPO_ROOT/epdtest/logs/sweep}"
RUN_STAMP="${RUN_STAMP:-$(date +"%Y%m%d_%H%M%S")}" 

TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-600}"
BENCH_REQUEST_RATE="${BENCH_REQUEST_RATE:-8}"
BENCH_MAX_CONCURRENCY="${BENCH_MAX_CONCURRENCY:-32}"
BENCHMARK="${BENCHMARK:-randommm}"
IMAGES_PER_REQ_LIST="${IMAGES_PER_REQ_LIST:-1}"

# Keep UCX warning/noise down for sweep stability.
UCX_RC_RETRY_COUNT="${UCX_RC_RETRY_COUNT:-7}"
UCX_RC_VERBS_RETRY_COUNT="${UCX_RC_VERBS_RETRY_COUNT:-7}"
UCX_WARN_UNUSED_ENV_VARS="${UCX_WARN_UNUSED_ENV_VARS:-n}"
ALLOW_TEARDOWN_SEGFAULT="${ALLOW_TEARDOWN_SEGFAULT:-1}"

CASES=(
  "1e1p1d_e0_p1_d2|1e1p1d|0|1|2"
  "1e1pNd_e0_p1_d0-2|1e1pNd|0|1|0,2"
  "1e1pNd_d_preempt_e0_p1_d0-2|1e1pNd_d_preempt|0|1|0,2"
  "Ne1p1d_e0-1_p1_d2|Ne1p1d|0,1|1|2"
  "Ne1p1d_e0-2_p1_d2|Ne1p1d|0,2|1|2"
  "Ne1p1d_e0-1-2_p1_d2|Ne1p1d|0,1,2|1|2"
  "Ne1p1d_pd_preempt_e0-1-2_p1_d2|Ne1p1d_pd_preempt|0,1,2|1|2"
  "Ne1pNd_e0-1_p1_d0-2|Ne1pNd|0,1|1|0,2"
  "Ne1pNd_pd_preempt_e0-1_p1_d0-2|Ne1pNd_pd_preempt|0,1|1|0,2"
)

SWEEP_ROOT="$LOG_PATH/$RUN_STAMP/compare"
mkdir -p "$SWEEP_ROOT"

echo "sweep_compare"
echo "  model               : $MODEL"
echo "  sweep_root          : ${SWEEP_ROOT#$REPO_ROOT/}"
echo "  benchmark           : $BENCHMARK"
echo "  timeout_seconds     : $TIMEOUT_SECONDS"
echo "  bench_request_rate  : $BENCH_REQUEST_RATE"
echo "  bench_max_conc      : $BENCH_MAX_CONCURRENCY"
echo "  images_per_req_list : $IMAGES_PER_REQ_LIST"

run_one_case() {
  local case_name="$1"
  local topology="$2"
  local gpu_e="$3"
  local gpu_p="$4"
  local gpu_d="$5"
  local images_per_req="$6"
  local run_dir="$SWEEP_ROOT/$case_name/ipr${images_per_req}"
  local launcher_log="$run_dir/launcher.log"

  mkdir -p "$run_dir"
  echo "--- case=$case_name topology=$topology GPU_E=$gpu_e GPU_P=$gpu_p GPU_D=$gpu_d ---"

  set +e
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
  UCX_RC_RETRY_COUNT="$UCX_RC_RETRY_COUNT" \
  UCX_RC_VERBS_RETRY_COUNT="$UCX_RC_VERBS_RETRY_COUNT" \
  UCX_WARN_UNUSED_ENV_VARS="$UCX_WARN_UNUSED_ENV_VARS" \
  bash ./epdtest/run.sh \
    --topology "$topology" \
    --benchmark "$BENCHMARK" \
    --images-per-req "$images_per_req" \
    2>&1 | tee "$launcher_log"
  local rc=${PIPESTATUS[0]}
  set -e

  if [[ "$rc" -eq 0 ]]; then
    return 0
  fi

  if [[ "$ALLOW_TEARDOWN_SEGFAULT" == "1" ]] \
      && grep -Eq "Serving Benchmark Result|Successful requests:" "$launcher_log" \
      && grep -Eq "Segfault encountered|rtnl_tc_unregister|destroy_process_group" "$launcher_log"; then
    echo "[warn] teardown segfault after benchmark output; continuing: $case_name"
    return 0
  fi

  return "$rc"
}

for images_per_req in $IMAGES_PER_REQ_LIST; do
  echo
  echo "=== IMAGES_PER_REQ=$images_per_req ==="

  for spec in "${CASES[@]}"; do
    IFS='|' read -r case_name topology gpu_e gpu_p gpu_d <<< "$spec"
    run_one_case "$case_name" "$topology" "$gpu_e" "$gpu_p" "$gpu_d" "$images_per_req"
  done
done

echo
echo "Sweep complete."
echo "Results root: ${SWEEP_ROOT#$REPO_ROOT/}"
