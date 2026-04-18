#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)
cd "$REPO_ROOT"

MODEL="${MODEL:-Qwen/Qwen2.5-VL-3B-Instruct}"
LOG_PATH="${LOG_PATH:-$REPO_ROOT/epdtest/logs/vtp}"
RUN_STAMP="${RUN_STAMP:-$(date +"%Y%m%d_%H%M%S")}"
SWEEP_SUBDIR="${SWEEP_SUBDIR:-vtp_sweep}"

BENCHMARK="${BENCHMARK:-randommm}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-600}"
BENCH_REQUEST_RATE="${BENCH_REQUEST_RATE:-4}"
BENCH_MAX_CONCURRENCY="${BENCH_MAX_CONCURRENCY:-20}"
IMAGES_PER_REQ_LIST="${IMAGES_PER_REQ_LIST:-1 2 4 8}"

# Default sweep methods requested by user.
VTP_METHODS="${VTP_METHODS:-none visionzip}"
VTP_RATES="${VTP_RATES:-0.3 0.5 0.7 0.9}"
FLATTEN_NONE_LAYOUT="${FLATTEN_NONE_LAYOUT:-0}"

CASES=(
  "1e1p1d_e0_p1_d2|1e1p1d|0|1|2"
  # "1e1pNd_e0_p1_d0-2|1e1pNd|0|1|0,2"
  # "1e1pNd_d_preempt_e0_p1_d0-2|1e1pNd_d_preempt|0|1|0,2"
  # "Ne1p1d_e0-1_p1_d2|Ne1p1d|0,1|1|2"
  # "Ne1p1d_e0-2_p1_d2|Ne1p1d|0,2|1|2"
  # "Ne1p1d_e0-1-2_p1_d2|Ne1p1d|0,1,2|1|2"
  # "Ne1p1d_pd_preempt_e0-1-2_p1_d2|Ne1p1d_pd_preempt|0,1,2|1|2"
  # "Ne1pNd_e0-1_p1_d0-2|Ne1pNd|0,1|1|0,2"
  # "Ne1pNd_pd_preempt_e0-1_p1_d0-2|Ne1pNd_pd_preempt|0,1|1|0,2"
)

ALLOW_TEARDOWN_SEGFAULT="${ALLOW_TEARDOWN_SEGFAULT:-1}"

SWEEP_ROOT="$LOG_PATH/$RUN_STAMP/$SWEEP_SUBDIR"
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
  local case_name="$1"
  local topology="$2"
  local gpu_e="$3"
  local gpu_p="$4"
  local gpu_d="$5"
  local method="$6"
  local images_per_req="$7"
  local rate="${8:-}"
  local rate_tag=""
  local run_dir=""

  if [[ -n "$rate" ]]; then
    rate_tag="${rate//./p}"
    run_dir="$SWEEP_ROOT/${case_name}/${method}/r${rate_tag}/ipr${images_per_req}"
  elif [[ "$FLATTEN_NONE_LAYOUT" == "1" && "$method" == "none" ]]; then
    run_dir="$SWEEP_ROOT/${case_name}/ipr${images_per_req}"
  else
    run_dir="$SWEEP_ROOT/${case_name}/${method}/ipr${images_per_req}"
  fi
  local launcher_log="$run_dir/launcher.log"

  mkdir -p "$run_dir"
  echo "--- case=$case_name method=$method rate=${rate:-n/a} topology=$topology GPU_E=$gpu_e GPU_P=$gpu_p GPU_D=$gpu_d images_per_req=$images_per_req ---"

  set +e
  if [[ -n "$rate" ]]; then
    TIMEOUT_SECONDS="$TIMEOUT_SECONDS" \
    BENCH_REQUEST_RATE="$BENCH_REQUEST_RATE" \
    BENCH_MAX_CONCURRENCY="$BENCH_MAX_CONCURRENCY" \
    BENCHMARK="$BENCHMARK" \
    MODEL="$MODEL" \
    IMAGES_PER_REQ="$images_per_req" \
    ENFORCE_EAGER="1" \
    GPU_E="$gpu_e" \
    GPU_P="$gpu_p" \
    GPU_D="$gpu_d" \
    LOG_PATH="$run_dir" \
    VISUAL_TOKEN_PRUNING_RATE="$rate" \
    RUN_DIR="$run_dir" \
    bash ./epdtest/run.sh \
      --topology "$topology" \
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
    GPU_E="$gpu_e" \
    GPU_P="$gpu_p" \
    GPU_D="$gpu_d" \
    LOG_PATH="$run_dir" \
    RUN_DIR="$run_dir" \
    bash ./epdtest/run.sh \
      --topology "$topology" \
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
    echo "[warn] teardown segfault after benchmark output; continuing: case=$case_name method=$method rate=${rate:-n/a} ipr=$images_per_req"
    return 0
  fi

  return "$rc"
}

launcher_log_for() {
  local case_name="$1"
  local method="$2"
  local images_per_req="$3"
  local rate="${4:-}"
  if [[ -n "$rate" ]]; then
    local rate_tag="${rate//./p}"
    echo "$SWEEP_ROOT/${case_name}/${method}/r${rate_tag}/ipr${images_per_req}/launcher.log"
  elif [[ "$FLATTEN_NONE_LAYOUT" == "1" && "$method" == "none" ]]; then
    echo "$SWEEP_ROOT/${case_name}/ipr${images_per_req}/launcher.log"
  else
    echo "$SWEEP_ROOT/${case_name}/${method}/ipr${images_per_req}/launcher.log"
  fi
}

run_dir_for() {
  local case_name="$1"
  local method="$2"
  local images_per_req="$3"
  local rate="${4:-}"
  if [[ -n "$rate" ]]; then
    local rate_tag="${rate//./p}"
    echo "$SWEEP_ROOT/${case_name}/${method}/r${rate_tag}/ipr${images_per_req}"
  elif [[ "$FLATTEN_NONE_LAYOUT" == "1" && "$method" == "none" ]]; then
    echo "$SWEEP_ROOT/${case_name}/ipr${images_per_req}"
  else
    echo "$SWEEP_ROOT/${case_name}/${method}/ipr${images_per_req}"
  fi
}

resolve_prefill_run_dir() {
  local base_run_dir="$1"
  local latest_prefill=""
  local latest_dir=""

  if [[ -f "$base_run_dir/prefill.log" ]]; then
    echo "$base_run_dir"
    return 0
  fi

  latest_prefill="$(find "$base_run_dir" -mindepth 1 -maxdepth 3 -type f -name prefill.log 2>/dev/null | sort | tail -n 1 || true)"
  if [[ -n "$latest_prefill" ]]; then
    latest_dir="$(dirname "$latest_prefill")"
    echo "$latest_dir"
    return 0
  fi

  echo "$base_run_dir"
}

verify_model_routing() {
  local case_name="$1"
  local method="$2"
  local images_per_req="$3"
  local rate="${4:-}"
  local run_dir
  run_dir="$(run_dir_for "$case_name" "$method" "$images_per_req" "$rate")"
  run_dir="$(resolve_prefill_run_dir "$run_dir")"
  local prefill_log="$run_dir/prefill.log"
  local case_tag="case=$case_name method=$method rate=${rate:-n/a} ipr=$images_per_req"

  if [[ ! -f "$prefill_log" ]]; then
    echo "missing prefill.log ($case_tag): ${prefill_log#$REPO_ROOT/}" >&2
    return 1
  fi

  if [[ "$method" == "none" ]]; then
    if rg -q "Resolved architecture: Qwen2_5_VLPruneForConditionalGeneration" "$prefill_log"; then
      echo "none-method run incorrectly resolved prune architecture ($case_tag)" >&2
      rg -n "Resolved architecture:" "$prefill_log" >&2 || true
      return 1
    fi
    return 0
  fi

  if rg -q "Resolved architecture: Qwen2_5_VLPruneForConditionalGeneration" "$prefill_log"; then
    return 0
  fi

  if rg -q "Resolved architecture:" "$prefill_log"; then
    echo "expected prune architecture but resolved different architecture ($case_tag)" >&2
    rg -n "Resolved architecture:" "$prefill_log" >&2 || true
    return 1
  fi

  if rg -q "vllm/multimodal/evs.py|qwen2_5_vl.py\", line 1391|IndexError: index 0 is out of bounds" "$prefill_log"; then
    echo "detected EVS/base fallback traceback in prefill ($case_tag)" >&2
    return 1
  fi

  echo "could not confirm expected model routing from prefill log ($case_tag)" >&2
  return 1
}

run_with_handling() {
  local case_name="$1"
  local topology="$2"
  local gpu_e="$3"
  local gpu_p="$4"
  local gpu_d="$5"
  local method="$6"
  local images_per_req="$7"
  local rate="${8:-}"
  local launcher_log
  local cause=""

  TOTAL_RUNS=$((TOTAL_RUNS + 1))
  if run_one "$case_name" "$topology" "$gpu_e" "$gpu_p" "$gpu_d" "$method" "$images_per_req" "$rate"; then
    if ! verify_model_routing "$case_name" "$method" "$images_per_req" "$rate"; then
      FAIL_RUNS=$((FAIL_RUNS + 1))
      launcher_log="$(launcher_log_for "$case_name" "$method" "$images_per_req" "$rate")"
      echo "[fail] case=$case_name method=$method rate=${rate:-n/a} ipr=$images_per_req cause: model routing verification failed"
      echo "       launcher: ${launcher_log#$REPO_ROOT/}"
    fi
    return 0
  fi

  FAIL_RUNS=$((FAIL_RUNS + 1))
  launcher_log="$(launcher_log_for "$case_name" "$method" "$images_per_req" "$rate")"
  if [[ -f "$launcher_log" ]]; then
    cause="$(grep -Ei \
      "Traceback|RuntimeError|ValueError|EngineCore failed|ERROR|address already in use|port collision|No CUDA runtime|out of memory|NIXL is not available|failed to start|cannot open shared object|Segfault|KeyError" \
      "$launcher_log" 2>/dev/null | tail -n1 || true)"
  fi
  [[ -n "$cause" ]] || cause="unknown (see launcher log)"

  echo "[fail] case=$case_name method=$method rate=${rate:-n/a} ipr=$images_per_req cause: $cause"
  echo "       launcher: ${launcher_log#$REPO_ROOT/}"
}

echo "vtp_compare"
echo "  case_count           : ${#CASES[@]}"
echo "  model               : $MODEL"
echo "  benchmark           : $BENCHMARK"
echo "  timeout_seconds     : $TIMEOUT_SECONDS"
echo "  bench_request_rate  : $BENCH_REQUEST_RATE"
echo "  bench_max_conc      : $BENCH_MAX_CONCURRENCY"
echo "  enforce_eager       : 1 (fixed)"
echo "  images_per_req_list : $IMAGES_PER_REQ_LIST"
echo "  vtp_methods         : $VTP_METHODS"
echo "  vtp_rates           : $VTP_RATES"
echo "  sweep_subdir        : $SWEEP_SUBDIR"
echo "  flatten_none_layout : $FLATTEN_NONE_LAYOUT"
echo "  output_root         : ${SWEEP_ROOT#$REPO_ROOT/}"

for spec in "${CASES[@]}"; do
  IFS='|' read -r case_name topology gpu_e gpu_p gpu_d <<< "$spec"
  echo
  echo "=== CASE=$case_name topology=$topology GPU_E=$gpu_e GPU_P=$gpu_p GPU_D=$gpu_d ==="
  for method in $VTP_METHODS; do
    validate_method "$method"
    echo "--- METHOD=$method ---"
    if [[ "$method" == "none" ]]; then
      for images_per_req in $IMAGES_PER_REQ_LIST; do
        run_with_handling "$case_name" "$topology" "$gpu_e" "$gpu_p" "$gpu_d" "$method" "$images_per_req"
      done
    else
      for rate in $VTP_RATES; do
        validate_rate "$rate"
        echo "--- RATE=$rate ---"
        for images_per_req in $IMAGES_PER_REQ_LIST; do
          run_with_handling "$case_name" "$topology" "$gpu_e" "$gpu_p" "$gpu_d" "$method" "$images_per_req" "$rate"
        done
      done
    fi
  done
done

echo
echo "VTP sweep complete."
echo "  total_runs : $TOTAL_RUNS"
echo "  fail_runs  : $FAIL_RUNS"
echo "Results root: ${SWEEP_ROOT#$REPO_ROOT/}"

if [[ "$FAIL_RUNS" -gt 0 ]]; then
  exit 1
fi
