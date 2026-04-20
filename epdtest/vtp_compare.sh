#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)
cd "$REPO_ROOT"

# Core run config
MODEL="${MODEL:-Qwen/Qwen2.5-VL-3B-Instruct}"
RUN_BENCHMARK="${RUN_BENCHMARK:-randommm}"
PROXY_PORT="${PROXY_PORT:-10001}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-1800}"
SERVER_READY_TIMEOUT_SECONDS="${SERVER_READY_TIMEOUT_SECONDS:-900}"
BENCH_REQUEST_RATE="${BENCH_REQUEST_RATE:-8}"
BENCH_MAX_CONCURRENCY="${BENCH_MAX_CONCURRENCY:-32}"
NUM_PROMPTS="${NUM_PROMPTS:-300}"

# Pruning sweep
IMAGES_PER_REQ="${IMAGES_PER_REQ:-32}"
VTP_METHODS="${VTP_METHODS:-none visionzip}"
VTP_RATES="${VTP_RATES:-0.5 0.9}"

# Output layout
LOG_PATH="${LOG_PATH:-$REPO_ROOT/epdtest/logs/vtp}"
EVAL_OUTPUT_ROOT="${EVAL_OUTPUT_ROOT:-$LOG_PATH}"
RUN_STAMP="${RUN_STAMP:-$(date +"%Y%m%d_%H%M%S")}" 
RUN_ROOT="$LOG_PATH/$RUN_STAMP"
EVAL_ROOT="$EVAL_OUTPUT_ROOT/$RUN_STAMP"
TABLE_SUMMARY_LOG="$EVAL_ROOT/vtp_sweep_summary.md"

# Cases: case_name|topology|gpu_e|gpu_p|gpu_d
CASES=(
  # "1e1p1d_e0_p1_d2|1e1p1d|0|1|2"
  # "Ne1p1d_e0-1_p1_d2|Ne1p1d|0,1|1|2"
  # "Ne1p1d_e0-2_p1_d2|Ne1p1d|0,2|1|2"
  "Ne1p1d_e0-1-2_p1_d2|Ne1p1d|0,1,2|1|2"
)

infer_prune_arch_expect_re() {
  local model_lc="${MODEL,,}"
  case "$model_lc" in
    *qwen2.5-vl*)
      echo "Qwen2_5_VLPruneForConditionalGeneration|Qwen2_5_VLVisionZipForConditionalGeneration"
      ;;
    *)
      echo "PruneForConditionalGeneration|VisionZipForConditionalGeneration"
      ;;
  esac
}
PRUNE_ARCH_EXPECT_RE="${PRUNE_ARCH_EXPECT_RE:-$(infer_prune_arch_expect_re)}"

if command -v rg >/dev/null 2>&1; then
  LOG_SEARCH_TOOL="rg"
else
  LOG_SEARCH_TOOL="grep"
fi

log_match_re() {
  local pattern="$1"
  local file="$2"
  if [[ "$LOG_SEARCH_TOOL" == "rg" ]]; then
    rg -Eq "$pattern" "$file"
  else
    grep -Eq "$pattern" "$file"
  fi
}

log_match_line_numbers_re() {
  local pattern="$1"
  local file="$2"
  if [[ "$LOG_SEARCH_TOOL" == "rg" ]]; then
    rg -nE "$pattern" "$file"
  else
    grep -nE "$pattern" "$file"
  fi
}

mkdir -p "$RUN_ROOT" "$EVAL_ROOT"
: > "$TABLE_SUMMARY_LOG"
{
  echo "# VTP Sweep Summary ($RUN_STAMP)"
  echo
} >> "$TABLE_SUMMARY_LOG"

validate_method() {
  case "$1" in
    none|visionzip|cdpruner) ;;
    *)
      echo "Unsupported visual-token pruning method: $1" >&2
      echo "Supported methods: none, visionzip, cdpruner" >&2
      exit 2
      ;;
  esac
}

validate_rate() {
  [[ "$1" =~ ^[0-9]*\.?[0-9]+$ ]] || { echo "Invalid visual-token pruning rate: $1" >&2; exit 2; }
}

run_with_cmd_timeout() {
  local timeout_seconds="$1"
  shift
  if command -v timeout >/dev/null 2>&1; then
    timeout "$timeout_seconds" "$@"
  else
    "$@"
  fi
}

wait_proxy_ready() {
  local pid="$1"
  local log="$2"
  local url="http://127.0.0.1:${PROXY_PORT}/v1/models"
  local deadline=$((SECONDS + SERVER_READY_TIMEOUT_SECONDS))

  while (( SECONDS < deadline )); do
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "Case launcher exited before proxy became ready. Log: $log" >&2
      tail -n 120 "$log" >&2 || true
      return 1
    fi
    if curl -fsS "$url" >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
  done

  echo "Timed out waiting for proxy readiness. Log: $log" >&2
  tail -n 120 "$log" >&2 || true
  return 1
}

run_dir_for() {
  local case_name="$1" method="$2" images_per_req="$3" rate="${4:-}"
  if [[ -n "$rate" ]]; then
    local rate_tag="${rate//./p}"
    echo "$RUN_ROOT/${case_name}/${method}/r${rate_tag}/ipr${images_per_req}"
  else
    echo "$RUN_ROOT/${case_name}/${method}/ipr${images_per_req}"
  fi
}

launcher_log_for() {
  local case_name="$1" method="$2" images_per_req="$3" rate="${4:-}"
  echo "$(run_dir_for "$case_name" "$method" "$images_per_req" "$rate")/launcher.log"
}

resolve_prefill_run_dir() {
  local base_run_dir="$1"
  if [[ -f "$base_run_dir/prefill.log" ]]; then
    echo "$base_run_dir"
    return 0
  fi

  local latest_prefill
  latest_prefill="$(find "$base_run_dir" -mindepth 1 -maxdepth 4 -type f -name prefill.log 2>/dev/null | sort | tail -n 1 || true)"
  if [[ -n "$latest_prefill" ]]; then
    dirname "$latest_prefill"
  else
    echo "$base_run_dir"
  fi
}

verify_model_routing() {
  local case_name="$1" method="$2" images_per_req="$3" rate="${4:-}"
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
    if log_match_re "(${PRUNE_ARCH_EXPECT_RE})" "$prefill_log"; then
      echo "none-method run resolved prune architecture ($case_tag)" >&2
      log_match_line_numbers_re "Resolved architecture(s)?:" "$prefill_log" >&2 || true
      return 1
    fi
    return 0
  fi

  if log_match_re "Resolved architecture(s)?: .*(${PRUNE_ARCH_EXPECT_RE})" "$prefill_log" || \
     log_match_re "(${PRUNE_ARCH_EXPECT_RE})" "$prefill_log"; then
    return 0
  fi

  if log_match_re "Resolved architecture(s)?:" "$prefill_log"; then
    echo "expected prune architecture but resolved different architecture ($case_tag)" >&2
    log_match_line_numbers_re "Resolved architecture(s)?:" "$prefill_log" >&2 || true
    return 1
  fi

  if log_match_re "vllm/multimodal/evs.py|qwen2_5_vl.py\", line 1391|IndexError: index 0 is out of bounds" "$prefill_log"; then
    echo "detected EVS/base fallback traceback in prefill ($case_tag)" >&2
    return 1
  fi

  echo "could not confirm expected model routing from prefill log ($case_tag)" >&2
  return 1
}

run_one() {
  local case_name="$1" topology="$2" gpu_e="$3" gpu_p="$4" gpu_d="$5" method="$6" images_per_req="$7" rate="${8:-}"
  local run_dir
  run_dir="$(run_dir_for "$case_name" "$method" "$images_per_req" "$rate")"
  local launcher_log="$run_dir/launcher.log"

  mkdir -p "$run_dir"
  echo "--- case=$case_name method=$method rate=${rate:-n/a} topology=$topology GPU_E=$gpu_e GPU_P=$gpu_p GPU_D=$gpu_d ipr=$images_per_req timeout=${TIMEOUT_SECONDS}s ---"

  set +e
  if [[ -n "$rate" ]]; then
    run_with_cmd_timeout "$TIMEOUT_SECONDS" env \
    TIMEOUT_SECONDS="$TIMEOUT_SECONDS" \
    BENCH_REQUEST_RATE="$BENCH_REQUEST_RATE" \
    BENCH_MAX_CONCURRENCY="$BENCH_MAX_CONCURRENCY" \
    BENCHMARK="$RUN_BENCHMARK" \
    MODEL="$MODEL" \
    PROXY_PORT="$PROXY_PORT" \
    IMAGES_PER_REQ="$images_per_req" \
    NUM_PROMPTS="$NUM_PROMPTS" \
    ENFORCE_EAGER="1" \
    GPU_E="$gpu_e" \
    GPU_P="$gpu_p" \
    GPU_D="$gpu_d" \
    LOG_PATH="$run_dir" \
    RUN_DIR="$run_dir" \
    VISUAL_TOKEN_PRUNING_RATE="$rate" \
    bash ./epdtest/run.sh \
      --topology "$topology" \
      --benchmark "$RUN_BENCHMARK" \
      --images-per-req "$images_per_req" \
      --visual-token-pruning-method "$method" \
      > >(tee "$launcher_log") 2>&1 &
  else
    run_with_cmd_timeout "$TIMEOUT_SECONDS" env \
    TIMEOUT_SECONDS="$TIMEOUT_SECONDS" \
    BENCH_REQUEST_RATE="$BENCH_REQUEST_RATE" \
    BENCH_MAX_CONCURRENCY="$BENCH_MAX_CONCURRENCY" \
    BENCHMARK="$RUN_BENCHMARK" \
    MODEL="$MODEL" \
    PROXY_PORT="$PROXY_PORT" \
    IMAGES_PER_REQ="$images_per_req" \
    NUM_PROMPTS="$NUM_PROMPTS" \
    ENFORCE_EAGER="1" \
    GPU_E="$gpu_e" \
    GPU_P="$gpu_p" \
    GPU_D="$gpu_d" \
    LOG_PATH="$run_dir" \
    RUN_DIR="$run_dir" \
    bash ./epdtest/run.sh \
      --topology "$topology" \
      --benchmark "$RUN_BENCHMARK" \
      --images-per-req "$images_per_req" \
      --visual-token-pruning-method "$method" \
      > >(tee "$launcher_log") 2>&1 &
  fi
  local run_pid="$!"
  local ready_rc=0
  wait_proxy_ready "$run_pid" "$launcher_log" || ready_rc=$?
  if [[ "$ready_rc" -ne 0 ]]; then
    kill "$run_pid" 2>/dev/null || true
  fi
  wait "$run_pid"
  local rc=$?
  set -e

  if [[ "$ready_rc" -ne 0 ]]; then
    return "$ready_rc"
  fi

  if [[ "$rc" -eq 0 ]]; then
    return 0
  fi

  return "$rc"
}

extract_fail_cause() {
  local launcher_log="$1"
  local run_rc="${2:-}"
  case "$run_rc" in
    124)
      echo "timed out (exit code 124). Increase TIMEOUT_SECONDS."
      return
      ;;
    137)
      echo "terminated by signal (exit code 137), likely timeout/kill."
      return
      ;;
  esac
  if [[ ! -f "$launcher_log" ]]; then
    echo "unknown (launcher log missing)"
    return
  fi
  local cause
  cause="$(grep -Ei "Traceback|RuntimeError|ValueError|EngineCore failed|ERROR|address already in use|Port [0-9]+ is already in use|port collision|No CUDA runtime|out of memory|NIXL is not available|failed to start|cannot open shared object|Segfault|KeyError" "$launcher_log" 2>/dev/null | tail -n 1 || true)"
  if [[ -n "$cause" ]]; then
    echo "$cause"
  else
    echo "unknown (see launcher log)"
  fi
}

extract_serving_benchmark_block() {
  local launcher_log="$1"
  awk '
    BEGIN { in_block=0; found=0; }
    /^============ Serving Benchmark Result ============/ {
      in_block=1
      found=1
    }
    in_block { print }
    in_block && /^==================================================$/ { exit }
    END {
      if (!found) exit 1
    }
  ' "$launcher_log"
}

append_summary_entry() {
  local status="$1" case_name="$2" method="$3" rate="$4" images_per_req="$5" cause="$6" launcher_log="$7"
  {
    echo "- ${status} case=$case_name method=$method rate=${rate:-n/a} ipr=$images_per_req"
    if [[ -n "$cause" ]]; then
      echo "  - cause: $cause"
    fi
    echo "  - launcher: ${launcher_log#$REPO_ROOT/}"
    echo "  - serving_benchmark_result:"
    echo
    echo '```text'
    if [[ -f "$launcher_log" ]]; then
      if ! extract_serving_benchmark_block "$launcher_log"; then
        echo "[not found in launcher.log]"
      fi
    else
      echo "[launcher.log missing]"
    fi
    echo '```'
    echo
  } >> "$TABLE_SUMMARY_LOG"
}

run_with_handling() {
  local case_name="$1" topology="$2" gpu_e="$3" gpu_p="$4" gpu_d="$5" method="$6" images_per_req="$7" rate="${8:-}"
  local run_rc=0
  if run_one "$case_name" "$topology" "$gpu_e" "$gpu_p" "$gpu_d" "$method" "$images_per_req" "$rate"; then
    if ! verify_model_routing "$case_name" "$method" "$images_per_req" "$rate"; then
      local launcher_log
      launcher_log="$(launcher_log_for "$case_name" "$method" "$images_per_req" "$rate")"
      echo "[fail] case=$case_name method=$method rate=${rate:-n/a} ipr=$images_per_req cause: model routing verification failed"
      echo "       launcher: ${launcher_log#$REPO_ROOT/}"
      append_summary_entry "FAIL" "$case_name" "$method" "${rate:-n/a}" "$images_per_req" "model routing verification failed" "$launcher_log"
    else
      local ok_launcher_log
      ok_launcher_log="$(launcher_log_for "$case_name" "$method" "$images_per_req" "$rate")"
      append_summary_entry "OK" "$case_name" "$method" "${rate:-n/a}" "$images_per_req" "" "$ok_launcher_log"
    fi
    return 0
  else
    run_rc=$?
  fi

  local launcher_log
  launcher_log="$(launcher_log_for "$case_name" "$method" "$images_per_req" "$rate")"
  local cause
  cause="$(extract_fail_cause "$launcher_log" "$run_rc")"

  echo "[fail] case=$case_name method=$method rate=${rate:-n/a} ipr=$images_per_req cause: $cause"
  echo "       launcher: ${launcher_log#$REPO_ROOT/}"
  append_summary_entry "FAIL" "$case_name" "$method" "${rate:-n/a}" "$images_per_req" "$cause" "$launcher_log"
}

echo "vtp_compare"
echo "  case_count             : ${#CASES[@]}"
echo "  model                  : $MODEL"
echo "  run_benchmark          : $RUN_BENCHMARK"
echo "  proxy_port             : $PROXY_PORT"
echo "  timeout_seconds        : $TIMEOUT_SECONDS"
echo "  server_ready_timeout   : $SERVER_READY_TIMEOUT_SECONDS"
echo "  bench_request_rate     : $BENCH_REQUEST_RATE"
echo "  bench_max_concurrency  : $BENCH_MAX_CONCURRENCY"
echo "  images_per_req         : $IMAGES_PER_REQ"
echo "  num_prompts            : $NUM_PROMPTS"
echo "  vtp_methods            : $VTP_METHODS"
echo "  vtp_rates              : $VTP_RATES"
echo "  prune_arch_expect_re   : $PRUNE_ARCH_EXPECT_RE"
echo "  log_root               : ${RUN_ROOT#$REPO_ROOT/}"
echo "  eval_root              : ${EVAL_ROOT#$REPO_ROOT/}"
echo "  table_summary          : ${TABLE_SUMMARY_LOG#$REPO_ROOT/}"

for spec in "${CASES[@]}"; do
  IFS='|' read -r case_name topology gpu_e gpu_p gpu_d <<< "$spec"
  echo
  echo "==== CASE: $case_name ===="
  echo "topology=$topology GPU_E=$gpu_e GPU_P=$gpu_p GPU_D=$gpu_d"

  for method in $VTP_METHODS; do
    validate_method "$method"

    if [[ "$method" == "none" ]]; then
      for images_per_req in $IMAGES_PER_REQ; do
        run_with_handling "$case_name" "$topology" "$gpu_e" "$gpu_p" "$gpu_d" "$method" "$images_per_req"
      done
      continue
    fi

    for rate in $VTP_RATES; do
      validate_rate "$rate"
      for images_per_req in $IMAGES_PER_REQ; do
        run_with_handling "$case_name" "$topology" "$gpu_e" "$gpu_p" "$gpu_d" "$method" "$images_per_req" "$rate"
      done
    done
  done
done

TOTAL_RUNS=$(grep -Ec '^- (OK|FAIL) case=' "$TABLE_SUMMARY_LOG" || true)
FAIL_RUNS=$(grep -Ec '^- FAIL case=' "$TABLE_SUMMARY_LOG" || true)

echo
echo "VTP sweep complete."
echo "  total_runs : $TOTAL_RUNS"
echo "  fail_runs  : $FAIL_RUNS"
echo "Results root: ${RUN_ROOT#$REPO_ROOT/}"
echo "Table summary: ${TABLE_SUMMARY_LOG#$REPO_ROOT/}"

if [[ "$FAIL_RUNS" -gt 0 ]]; then
  exit 1
fi
