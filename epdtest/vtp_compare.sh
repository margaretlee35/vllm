#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)
cd "$REPO_ROOT"

# Fixed topology for this sweep.
TOPOLOGY="1e1p1d"

MODEL="${MODEL:-Qwen/Qwen2.5-VL-3B-Instruct}"
LOG_PATH="${LOG_PATH:-$REPO_ROOT/epdtest/logs/vtp}"
RUN_STAMP="${RUN_STAMP:-$(date +"%Y%m%d_%H%M%S")}"

BENCHMARK="${BENCHMARK:-randommm}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-600}"
BENCH_REQUEST_RATE="${BENCH_REQUEST_RATE:-8}"
BENCH_MAX_CONCURRENCY="${BENCH_MAX_CONCURRENCY:-32}"
IMAGES_PER_REQ_LIST="${IMAGES_PER_REQ_LIST:-1 2 4 8}"

# Default sweep methods requested by user.
VTP_METHODS="${VTP_METHODS:-none visionzip}"
VTP_RATES="${VTP_RATES:-0.3 0.5 0.7 0.9}"

GPU_E="${GPU_E:-0}"
GPU_P="${GPU_P:-1}"
GPU_D="${GPU_D:-2}"

ALLOW_TEARDOWN_SEGFAULT="${ALLOW_TEARDOWN_SEGFAULT:-1}"

SWEEP_ROOT="$LOG_PATH/$RUN_STAMP/vtp_sweep"
mkdir -p "$SWEEP_ROOT"
TOTAL_RUNS=0
FAIL_RUNS=0

validate_method() {
  case "$1" in
    none|visionzip|cdpruner) ;;
    *)
      echo "Unsupported visual-token pruning method: $1" >&2
      echo "Supported methods: none, visionzip, cdpruner" >&2
      exit 1
      ;;
  esac
}

validate_rate() {
  if [[ ! "$1" =~ ^[0-9]*\.?[0-9]+$ ]]; then
    echo "Invalid pruning rate: $1" >&2
    echo "Use numeric values like: 0.5 0.7 0.9" >&2
    exit 1
  fi
}

run_one() {
  local method="$1"
  local images_per_req="$2"
  local rate="${3:-}"
  local rate_tag=""
  local run_dir=""

  if [[ -n "$rate" ]]; then
    rate_tag="${rate//./p}"
    run_dir="$SWEEP_ROOT/${method}/r${rate_tag}/ipr${images_per_req}"
  else
    run_dir="$SWEEP_ROOT/${method}/ipr${images_per_req}"
  fi
  local launcher_log="$run_dir/launcher.log"

  mkdir -p "$run_dir"
  echo "--- method=$method rate=${rate:-n/a} topology=$TOPOLOGY GPU_E=$GPU_E GPU_P=$GPU_P GPU_D=$GPU_D images_per_req=$images_per_req ---"

  set +e
  if [[ -n "$rate" ]]; then
    TIMEOUT_SECONDS="$TIMEOUT_SECONDS" \
    BENCH_REQUEST_RATE="$BENCH_REQUEST_RATE" \
    BENCH_MAX_CONCURRENCY="$BENCH_MAX_CONCURRENCY" \
    BENCHMARK="$BENCHMARK" \
    MODEL="$MODEL" \
    IMAGES_PER_REQ="$images_per_req" \
    ENFORCE_EAGER="1" \
    GPU_E="$GPU_E" \
    GPU_P="$GPU_P" \
    GPU_D="$GPU_D" \
    VISUAL_TOKEN_PRUNING_RATE="$rate" \
    RUN_DIR="$run_dir" \
    bash ./epdtest/run.sh \
      --topology "$TOPOLOGY" \
      --benchmark "$BENCHMARK" \
      --images-per-req "$images_per_req" \
      --visual-token-pruning-method "$method" \
      2>&1 | tee "$launcher_log"
  else
    TIMEOUT_SECONDS="$TIMEOUT_SECONDS" \
    BENCH_REQUEST_RATE="$BENCH_REQUEST_RATE" \
    BENCH_MAX_CONCURRENCY="$BENCH_MAX_CONCURRENCY" \
    BENCHMARK="$BENCHMARK" \
    MODEL="$MODEL" \
    IMAGES_PER_REQ="$images_per_req" \
    ENFORCE_EAGER="1" \
    GPU_E="$GPU_E" \
    GPU_P="$GPU_P" \
    GPU_D="$GPU_D" \
    RUN_DIR="$run_dir" \
    bash ./epdtest/run.sh \
      --topology "$TOPOLOGY" \
      --benchmark "$BENCHMARK" \
      --images-per-req "$images_per_req" \
      --visual-token-pruning-method "$method" \
      2>&1 | tee "$launcher_log"
  fi
  local rc=${PIPESTATUS[0]}
  set -e

  if [[ "$rc" -eq 0 ]]; then
    return 0
  fi

  if [[ "$ALLOW_TEARDOWN_SEGFAULT" == "1" ]] \
      && grep -Eq "Serving Benchmark Result|Successful requests:" "$launcher_log" \
      && grep -Eq "Segfault encountered|rtnl_tc_unregister|destroy_process_group" "$launcher_log"; then
    echo "[warn] teardown segfault after benchmark output; continuing: method=$method rate=${rate:-n/a} ipr=$images_per_req"
    return 0
  fi

  return "$rc"
}

launcher_log_for() {
  local method="$1"
  local images_per_req="$2"
  local rate="${3:-}"
  if [[ -n "$rate" ]]; then
    local rate_tag="${rate//./p}"
    echo "$SWEEP_ROOT/${method}/r${rate_tag}/ipr${images_per_req}/launcher.log"
  else
    echo "$SWEEP_ROOT/${method}/ipr${images_per_req}/launcher.log"
  fi
}

run_with_handling() {
  local method="$1"
  local images_per_req="$2"
  local rate="${3:-}"
  local launcher_log
  local cause=""

  TOTAL_RUNS=$((TOTAL_RUNS + 1))
  if run_one "$method" "$images_per_req" "$rate"; then
    return 0
  fi

  FAIL_RUNS=$((FAIL_RUNS + 1))
  launcher_log="$(launcher_log_for "$method" "$images_per_req" "$rate")"
  if [[ -f "$launcher_log" ]]; then
    cause="$(grep -Ei \
      "Traceback|RuntimeError|ValueError|EngineCore failed|ERROR|address already in use|port collision|No CUDA runtime|out of memory|NIXL is not available|failed to start|cannot open shared object|Segfault|KeyError" \
      "$launcher_log" 2>/dev/null | tail -n1 || true)"
  fi
  [[ -n "$cause" ]] || cause="unknown (see launcher log)"

  echo "[fail] method=$method rate=${rate:-n/a} ipr=$images_per_req cause: $cause"
  echo "       launcher: ${launcher_log#$REPO_ROOT/}"
}

echo "vtp_compare"
echo "  topology            : $TOPOLOGY"
echo "  model               : $MODEL"
echo "  benchmark           : $BENCHMARK"
echo "  timeout_seconds     : $TIMEOUT_SECONDS"
echo "  bench_request_rate  : $BENCH_REQUEST_RATE"
echo "  bench_max_conc      : $BENCH_MAX_CONCURRENCY"
echo "  enforce_eager       : 1 (fixed)"
echo "  images_per_req_list : $IMAGES_PER_REQ_LIST"
echo "  vtp_methods         : $VTP_METHODS"
echo "  vtp_rates           : $VTP_RATES"
echo "  output_root         : ${SWEEP_ROOT#$REPO_ROOT/}"

for method in $VTP_METHODS; do
  validate_method "$method"
  echo
  echo "=== METHOD=$method ==="
  if [[ "$method" == "none" ]]; then
    for images_per_req in $IMAGES_PER_REQ_LIST; do
      run_with_handling "$method" "$images_per_req"
    done
  else
    for rate in $VTP_RATES; do
      validate_rate "$rate"
      echo "--- RATE=$rate ---"
      for images_per_req in $IMAGES_PER_REQ_LIST; do
        run_with_handling "$method" "$images_per_req" "$rate"
      done
    done
  fi
done

echo
echo "VTP sweep complete."
echo "  total_runs : $TOTAL_RUNS"
echo "  fail_runs  : $FAIL_RUNS"
echo "Results root: ${SWEEP_ROOT#$REPO_ROOT/}"

if [[ "$FAIL_RUNS" -gt 0 ]]; then
  exit 1
fi
