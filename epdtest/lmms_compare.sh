#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)
cd "$REPO_ROOT"

MODEL="${MODEL:-Qwen/Qwen2.5-VL-3B-Instruct}"
RUN_BENCHMARK="${RUN_BENCHMARK:-randommm}"
PROXY_PORT="${PROXY_PORT:-10001}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-600}"
BENCH_REQUEST_RATE="${BENCH_REQUEST_RATE:-4}"
BENCH_MAX_CONCURRENCY="${BENCH_MAX_CONCURRENCY:-32}"
IMAGES_PER_REQ="${IMAGES_PER_REQ:-1}"
NUM_PROMPTS="${NUM_PROMPTS:-300}"
SERVER_READY_TIMEOUT_SECONDS="${SERVER_READY_TIMEOUT_SECONDS:-900}"

LMMS_TASKS="${LMMS_TASKS:-mmmu_val}"
# Default to full-task evaluation (no --limit cap).
# Set LMMS_LIMIT to a positive integer to run a subset.
LMMS_LIMIT="${LMMS_LIMIT:-}"
LMMS_BATCH_SIZE="${LMMS_BATCH_SIZE:-1}"
LMMS_MODEL="${LMMS_MODEL:-openai_compatible}"
LMMS_MODEL_ARGS_TEMPLATE="${LMMS_MODEL_ARGS_TEMPLATE:-model={MODEL},base_url=http://127.0.0.1:{PORT}/v1,api_key=EMPTY,temperature=0,max_new_tokens=128}"
OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
# If 1, force OpenAI-compatible backends to use local vLLM endpoint.
FORCE_LOCAL_OPENAI_BASE_URL="${FORCE_LOCAL_OPENAI_BASE_URL:-1}"

# Default sweep methods requested by user.
VTP_METHODS="${VTP_METHODS:-none visionzip}"
VTP_RATES="${VTP_RATES:-0.3 0.5 0.7 0.9}"

HF_HOME="${HF_HOME:-/workspace/.hf_cache}"
HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"

LOG_PATH="${LOG_PATH:-$REPO_ROOT/epdtest/logs/lmms}"
EVAL_OUTPUT_ROOT="${EVAL_OUTPUT_ROOT:-$LOG_PATH}"
RUN_STAMP="${RUN_STAMP:-$(date +"%Y%m%d_%H%M%S")}" 
RUN_ROOT="$LOG_PATH/${RUN_STAMP}"
EVAL_ROOT="$EVAL_OUTPUT_ROOT/${RUN_STAMP}"
TABLE_SUMMARY_LOG="$EVAL_ROOT/lmms_tables_summary.md"

mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$HUGGINGFACE_HUB_CACHE" "$RUN_ROOT" "$EVAL_ROOT"

LMMS_PYTHON_BIN="${LMMS_PYTHON_BIN:-}"
if [[ -z "$LMMS_PYTHON_BIN" ]]; then
  if [[ -n "${VIRTUAL_ENV:-}" && -x "${VIRTUAL_ENV}/bin/python" ]]; then
    LMMS_PYTHON_BIN="${VIRTUAL_ENV}/bin/python"
  elif [[ -x "$REPO_ROOT/.venv/bin/python" ]]; then
    LMMS_PYTHON_BIN="$REPO_ROOT/.venv/bin/python"
  elif command -v python3 >/dev/null 2>&1; then
    LMMS_PYTHON_BIN="python3"
  else
    LMMS_PYTHON_BIN="python"
  fi
fi

REAL_VLLM_BIN="${REAL_VLLM_BIN:-$REPO_ROOT/.venv/bin/vllm}"
if [[ ! -x "$REAL_VLLM_BIN" ]]; then
  REAL_VLLM_BIN="$(command -v vllm)"
fi

CASES=(
  "1e1p1d_e0_p1_d2|1e1p1d|0|1|0"
  # "1e1pNd_e0_p1_d0-2|1e1pNd|0|1|0,2"
  # "1e1pNd_d_preempt_e0_p1_d0-2|1e1pNd_d_preempt|0|1|0,2"
  # "Ne1p1d_e0-1_p1_d2|Ne1p1d|0,1|1|2"
  # "Ne1p1d_e0-2_p1_d2|Ne1p1d|0,2|1|2"
  # "Ne1p1d_e0-1-2_p1_d2|Ne1p1d|0,1,2|1|2"
  # "Ne1p1d_pd_preempt_e0-1-2_p1_d2|Ne1p1d_pd_preempt|0,1,2|1|2"
  # "Ne1pNd_e0-1_p1_d0-2|Ne1pNd|0,1|1|0,2"
  # "Ne1pNd_pd_preempt_e0-1_p1_d0-2|Ne1pNd_pd_preempt|0,1|1|0,2"
)

CURRENT_GROUP_PID=""
CURRENT_WRAPPER_DIR=""

cleanup() {
  set +e
  if [[ -n "$CURRENT_GROUP_PID" ]]; then
    kill -- -"$CURRENT_GROUP_PID" 2>/dev/null || true
    sleep 2
    kill -9 -- -"$CURRENT_GROUP_PID" 2>/dev/null || true
    CURRENT_GROUP_PID=""
  fi
  if [[ -n "$CURRENT_WRAPPER_DIR" && -d "$CURRENT_WRAPPER_DIR" ]]; then
    rm -rf "$CURRENT_WRAPPER_DIR" 2>/dev/null || true
    CURRENT_WRAPPER_DIR=""
  fi
}
trap cleanup EXIT INT TERM

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

start_case() {
  local case_name="$1" topology="$2" gpu_e="$3" gpu_p="$4" gpu_d="$5" vtp_method="$6" vtp_rate="${7:-}"
  local case_root="$RUN_ROOT/$case_name/$vtp_method"
  if [[ -n "$vtp_rate" ]]; then
    case_root="$case_root/r${vtp_rate//./p}"
  fi
  local case_run_dir="$case_root/serve_run"
  local case_log="$case_root/launcher.log"
  local wrapper_dir="$case_root/bin"

  mkdir -p "$case_root" "$case_run_dir" "$wrapper_dir"

  cat > "$wrapper_dir/vllm" <<'WRAP'
#!/bin/bash
set -euo pipefail
if [[ "${VLLM_INTERCEPT_BENCH:-0}" == "1" && "${1:-}" == "bench" && "${2:-}" == "serve" ]]; then
  echo "[vllm-wrapper] intercepted: vllm bench serve"
  while true; do sleep 3600; done
fi
exec "${REAL_VLLM_BIN:?REAL_VLLM_BIN is not set}" "$@"
WRAP
  chmod +x "$wrapper_dir/vllm"
  CURRENT_WRAPPER_DIR="$wrapper_dir"

  echo
  echo "==== CASE: $case_name ===="
  echo "topology=$topology GPU_E=$gpu_e GPU_P=$gpu_p GPU_D=$gpu_d"
  echo "vt_method=$vtp_method vt_rate=${vtp_rate:-n/a}"
  echo "run_dir=${case_run_dir#$REPO_ROOT/}"

  if [[ -n "$vtp_rate" ]]; then
    setsid env \
      PATH="$wrapper_dir:$PATH" \
      VLLM_INTERCEPT_BENCH=1 \
      REAL_VLLM_BIN="$REAL_VLLM_BIN" \
      TIMEOUT_SECONDS="$TIMEOUT_SECONDS" \
      BENCH_REQUEST_RATE="$BENCH_REQUEST_RATE" \
      BENCH_MAX_CONCURRENCY="$BENCH_MAX_CONCURRENCY" \
      BENCHMARK="$RUN_BENCHMARK" \
      MODEL="$MODEL" \
      IMAGES_PER_REQ="$IMAGES_PER_REQ" \
      NUM_PROMPTS="$NUM_PROMPTS" \
      ENFORCE_EAGER="1" \
      GPU_E="$gpu_e" \
      GPU_P="$gpu_p" \
      GPU_D="$gpu_d" \
      LOG_PATH="$case_run_dir" \
      RUN_DIR="$case_run_dir" \
      VISUAL_TOKEN_PRUNING_RATE="$vtp_rate" \
      bash ./epdtest/run.sh \
        --topology "$topology" \
        --benchmark "$RUN_BENCHMARK" \
        --images-per-req "$IMAGES_PER_REQ" \
        --visual-token-pruning-method "$vtp_method" \
        > "$case_log" 2>&1 &
  else
    setsid env \
      PATH="$wrapper_dir:$PATH" \
      VLLM_INTERCEPT_BENCH=1 \
      REAL_VLLM_BIN="$REAL_VLLM_BIN" \
      TIMEOUT_SECONDS="$TIMEOUT_SECONDS" \
      BENCH_REQUEST_RATE="$BENCH_REQUEST_RATE" \
      BENCH_MAX_CONCURRENCY="$BENCH_MAX_CONCURRENCY" \
      BENCHMARK="$RUN_BENCHMARK" \
      MODEL="$MODEL" \
      IMAGES_PER_REQ="$IMAGES_PER_REQ" \
      NUM_PROMPTS="$NUM_PROMPTS" \
      ENFORCE_EAGER="1" \
      GPU_E="$gpu_e" \
      GPU_P="$gpu_p" \
      GPU_D="$gpu_d" \
      LOG_PATH="$case_run_dir" \
      RUN_DIR="$case_run_dir" \
      VISUAL_TOKEN_PRUNING_RATE="" \
      bash ./epdtest/run.sh \
        --topology "$topology" \
        --benchmark "$RUN_BENCHMARK" \
        --images-per-req "$IMAGES_PER_REQ" \
        --visual-token-pruning-method "$vtp_method" \
        > "$case_log" 2>&1 &
  fi

  CURRENT_GROUP_PID="$!"
  wait_proxy_ready "$CURRENT_GROUP_PID" "$case_log"
  echo "Proxy ready on port $PROXY_PORT for case $case_name"
}

stop_case() {
  cleanup
}

preflight() {
  "$LMMS_PYTHON_BIN" - "$LMMS_TASKS" <<'PY' >/dev/null
import re
import sys
from lmms_eval.tasks import TaskManager

tasks = [t for t in re.split(r"[,\s]+", sys.argv[1].strip()) if t]
TaskManager("").load_task_or_group(tasks)
PY
  echo "LMMS task preflight passed for tasks: $LMMS_TASKS"
}

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
  if [[ ! "$1" =~ ^[0-9]*\.?[0-9]+$ ]]; then
    echo "Invalid visual-token pruning rate: $1" >&2
    echo "Use numeric values like: 0.3 0.5 0.7 0.9" >&2
    exit 2
  fi
}

extract_tables() {
  local in_log="$1"
  local out_log="$2"
  awk '
    BEGIN { in_tasks=0; in_t=0; got_tasks=0; got_t=0; }
    /^\| Tasks[[:space:]]*\|/ { in_tasks=1; got_tasks=1; print; next }
    in_tasks {
      if ($0 ~ /^\|/) { print; next }
      if ($0 ~ /^[[:space:]]*$/) print ""
      in_tasks=0
    }
    /^Throughput Summary[[:space:]]*$/ { if (got_tasks) print ""; in_t=1; got_t=1; print; next }
    in_t {
      if ($0 ~ /^\|/) { print; next }
      if ($0 ~ /^[[:space:]]*$/) { print ""; next }
      in_t=0
    }
    END {
      if (!got_tasks) print "[warn] task metric table not found in lmms output."
      if (!got_t) print "[warn] throughput summary table not found in lmms output."
    }
  ' "$in_log" > "$out_log"
}

run_lmms() {
  local case_name="$1" vtp_method="$2" vtp_rate="${3:-}"
  local outdir="$EVAL_ROOT/$case_name/$vtp_method"
  if [[ -n "$vtp_rate" ]]; then
    outdir="$outdir/r${vtp_rate//./p}"
  fi
  local lmms_log="$outdir/lmms_eval.log"
  local table_log="$outdir/lmms_tables.log"
  local model_args="${LMMS_MODEL_ARGS_TEMPLATE//\{MODEL\}/$MODEL}"
  model_args="${model_args//\{PORT\}/$PROXY_PORT}"
  local local_base_url="http://127.0.0.1:${PROXY_PORT}/v1"

  mkdir -p "$outdir"
  export HF_HOME HF_DATASETS_CACHE HUGGINGFACE_HUB_CACHE OPENAI_API_KEY

  case "$LMMS_MODEL" in
    openai|openai_compatible)
      export OPENAI_API_BASE="$local_base_url"
      export OPENAI_BASE_URL="$local_base_url"
      local canonical_model_args="model=${MODEL},base_url=${local_base_url},api_key=${OPENAI_API_KEY},temperature=0,max_new_tokens=128"

      # If placeholders remain or template looks malformed, reset to canonical args.
      if [[ "$model_args" == *"{MODEL"* || "$model_args" == *"{PORT"* || "$model_args" == \{* ]]; then
        model_args="$canonical_model_args"
      fi

      if [[ "$FORCE_LOCAL_OPENAI_BASE_URL" == "1" ]]; then
        if [[ "$model_args" == *"base_url="* ]]; then
          model_args=$(echo "$model_args" | sed -E "s|base_url=[^,]*|base_url=${local_base_url}|g")
        else
          model_args="${model_args},base_url=${local_base_url}"
        fi
      fi

      if [[ "$model_args" == *"model="* ]]; then
        model_args=$(echo "$model_args" | sed -E "s|model=[^,}]*|model=${MODEL}|g")
      else
        model_args="model=${MODEL},${model_args}"
      fi

      if [[ "$model_args" != *"api_key="* ]]; then
        model_args="${model_args},api_key=${OPENAI_API_KEY}"
      fi
      ;;
  esac

  local -a limit_arg=()
  if [[ -n "$LMMS_LIMIT" ]]; then
    case "${LMMS_LIMIT,,}" in
      0|all|none) ;;
      *) limit_arg=(--limit "$LMMS_LIMIT") ;;
    esac
  fi

  local model_args_for_log
  model_args_for_log=$(echo "$model_args" | sed -E 's/(api_key=)[^,]*/\1***REDACTED***/g')
  echo "LMMS model args ($case_name/$vtp_method${vtp_rate:+/r$vtp_rate}): $model_args_for_log"
  if [[ -n "${OPENAI_API_BASE:-}" ]]; then
    echo "LMMS OPENAI_API_BASE ($case_name/$vtp_method${vtp_rate:+/r$vtp_rate}): $OPENAI_API_BASE"
  fi

  "$LMMS_PYTHON_BIN" -m lmms_eval \
    --model "$LMMS_MODEL" \
    --model_args "$model_args" \
    --tasks "$LMMS_TASKS" \
    --batch_size "$LMMS_BATCH_SIZE" \
    --output_path "$outdir" \
    --log_samples \
    "${limit_arg[@]}" 2>&1 | tee "$lmms_log"

  extract_tables "$lmms_log" "$table_log"

  {
    echo "## $case_name | method=$vtp_method | rate=${vtp_rate:--}"
    echo
    cat "$table_log"
    echo
  } >> "$TABLE_SUMMARY_LOG"

  echo "LMMS output saved: ${outdir#$REPO_ROOT/}"
  echo "LMMS raw log: ${lmms_log#$REPO_ROOT/}"
  echo "LMMS table log: ${table_log#$REPO_ROOT/}"
}

echo "lmms_compare"
echo "  model                  : $MODEL"
echo "  run_benchmark          : $RUN_BENCHMARK"
echo "  proxy_port             : $PROXY_PORT"
echo "  timeout_seconds        : $TIMEOUT_SECONDS"
echo "  bench_request_rate     : $BENCH_REQUEST_RATE"
echo "  bench_max_concurrency  : $BENCH_MAX_CONCURRENCY"
echo "  images_per_req         : $IMAGES_PER_REQ"
echo "  lmms_tasks             : $LMMS_TASKS"
LMMS_LIMIT_DISPLAY="${LMMS_LIMIT:-all}"
case "${LMMS_LIMIT_DISPLAY,,}" in
  0|all|none) LMMS_LIMIT_DISPLAY="all" ;;
esac
echo "  lmms_limit             : $LMMS_LIMIT_DISPLAY"
echo "  lmms_batch_size        : $LMMS_BATCH_SIZE"
echo "  vtp_methods            : $VTP_METHODS"
echo "  vtp_rates              : $VTP_RATES"
echo "  log_root               : ${RUN_ROOT#$REPO_ROOT/}"
echo "  eval_root              : ${EVAL_ROOT#$REPO_ROOT/}"
echo "  table_summary          : ${TABLE_SUMMARY_LOG#$REPO_ROOT/}"

: > "$TABLE_SUMMARY_LOG"
echo "# LMMS Table Summary ($RUN_STAMP)" >> "$TABLE_SUMMARY_LOG"
echo >> "$TABLE_SUMMARY_LOG"

preflight

for spec in "${CASES[@]}"; do
  IFS='|' read -r case_name topology gpu_e gpu_p gpu_d <<< "$spec"
  for vtp_method in $VTP_METHODS; do
    validate_method "$vtp_method"
    if [[ "$vtp_method" == "none" ]]; then
      start_case "$case_name" "$topology" "$gpu_e" "$gpu_p" "$gpu_d" "$vtp_method"
      run_lmms "$case_name" "$vtp_method"
      stop_case
    else
      for vtp_rate in $VTP_RATES; do
        validate_rate "$vtp_rate"
        start_case "$case_name" "$topology" "$gpu_e" "$gpu_p" "$gpu_d" "$vtp_method" "$vtp_rate"
        run_lmms "$case_name" "$vtp_method" "$vtp_rate"
        stop_case
      done
    fi
  done
done

echo
echo "Done."
