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
LOG_PATH="${LOG_PATH:-$REPO_ROOT/epdtest/logs/lmms}"
EVAL_OUTPUT_ROOT="${EVAL_OUTPUT_ROOT:-$LOG_PATH}"
RUN_STAMP="${RUN_STAMP:-$(date +"%Y%m%d_%H%M%S")}" 
RUN_ROOT="$LOG_PATH/$RUN_STAMP"
EVAL_ROOT="$EVAL_OUTPUT_ROOT/$RUN_STAMP"
TABLE_SUMMARY_LOG="$EVAL_ROOT/lmms_sweep_summary.md"

# Cases: case_name|topology|gpu_e|gpu_p|gpu_d
CASES=(
  "1e1p1d_e0_p1_d2|1e1p1d|0|1|2"
  "Ne1p1d_e0-1_p1_d2|Ne1p1d|0,1|1|2"
  # "Ne1p1d_e0-2_p1_d2|Ne1p1d|0,2|1|2"
  # "Ne1p1d_e0-1-2_p1_d2|Ne1p1d|0,1,2|1|2"
)

# LMMS config
LMMS_TASKS="${LMMS_TASKS:-mmmu_val}"
LMMS_LIMIT="${LMMS_LIMIT:-}"
LMMS_BATCH_SIZE="${LMMS_BATCH_SIZE:-1}"
OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"

# HF Cache
HF_HOME="${HF_HOME:-/workspace/.hf_cache}"
HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"


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

pick_python_bin() {
  if [[ -n "${LMMS_PYTHON_BIN:-}" ]]; then
    echo "$LMMS_PYTHON_BIN"
    return
  fi
  if [[ -n "${VIRTUAL_ENV:-}" && -x "${VIRTUAL_ENV}/bin/python" ]]; then
    echo "${VIRTUAL_ENV}/bin/python"
  elif [[ -x "$REPO_ROOT/.venv/bin/python" ]]; then
    echo "$REPO_ROOT/.venv/bin/python"
  elif command -v python3 >/dev/null 2>&1; then
    echo "python3"
  else
    echo "python"
  fi
}

pick_vllm_bin() {
  if [[ -n "${REAL_VLLM_BIN:-}" && -x "$REAL_VLLM_BIN" ]]; then
    echo "$REAL_VLLM_BIN"
  elif [[ -x "$REPO_ROOT/.venv/bin/vllm" ]]; then
    echo "$REPO_ROOT/.venv/bin/vllm"
  else
    command -v vllm
  fi
}

LMMS_PYTHON_BIN="$(pick_python_bin)"
REAL_VLLM_BIN="$(pick_vllm_bin)"

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

mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$HUGGINGFACE_HUB_CACHE" "$RUN_ROOT" "$EVAL_ROOT"

CURRENT_GROUP_PID=""
CURRENT_WRAPPER_DIR=""
CURRENT_CASE_RUN_DIR=""

cleanup_current_case() {
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
  CURRENT_CASE_RUN_DIR=""
}

cleanup_all() {
  cleanup_current_case
}
trap cleanup_all EXIT INT TERM

wait_proxy_ready() {
  local pid="$1" log="$2"
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

verify_prune_architecture_ready() {
  local case_name="$1" vtp_method="$2" vtp_rate="${3:-}"
  [[ "$vtp_method" == "none" ]] && return 0

  local case_tag="$case_name/$vtp_method${vtp_rate:+/r$vtp_rate}"
  local deadline=$((SECONDS + SERVER_READY_TIMEOUT_SECONDS))

  while (( SECONDS < deadline )); do
    if [[ -n "$CURRENT_GROUP_PID" ]] && ! kill -0 "$CURRENT_GROUP_PID" 2>/dev/null; then
      echo "Case launcher exited before prune architecture verification: $case_tag" >&2
      return 1
    fi

    local run_dir
    run_dir="$(resolve_prefill_run_dir "$CURRENT_CASE_RUN_DIR")"
    local prefill_log="$run_dir/prefill.log"

    if [[ -f "$prefill_log" ]]; then
      if log_match_re "Resolved architecture(s)?: .*(${PRUNE_ARCH_EXPECT_RE})" "$prefill_log" || \
         log_match_re "(${PRUNE_ARCH_EXPECT_RE})" "$prefill_log"; then
        echo "Prune architecture verified ($case_tag): matched /${PRUNE_ARCH_EXPECT_RE}/"
        return 0
      fi

      if log_match_re "Resolved architecture(s)?:" "$prefill_log"; then
        echo "Expected prune architecture but got different resolved architecture ($case_tag)." >&2
        log_match_line_numbers_re "Resolved architecture(s)?:" "$prefill_log" >&2 || true
        return 1
      fi

      if log_match_re "vllm/multimodal/evs.py|qwen2_5_vl.py\", line 1391|IndexError: index 0 is out of bounds" "$prefill_log"; then
        echo "Detected legacy EVS/base mRoPE traceback in prefill ($case_tag)." >&2
        tail -n 120 "$prefill_log" >&2 || true
        return 1
      fi
    fi

    sleep 1
  done

  echo "Timed out waiting for prune architecture verification: $case_tag" >&2
  return 1
}

create_vllm_wrapper() {
  local wrapper_path="$1"
  cat > "$wrapper_path" <<'WRAP'
#!/bin/bash
set -euo pipefail
if [[ "${VLLM_INTERCEPT_BENCH:-0}" == "1" && "${1:-}" == "bench" && "${2:-}" == "serve" ]]; then
  echo "[vllm-wrapper] intercepted: vllm bench serve"
  while true; do sleep 3600; done
fi
exec "${REAL_VLLM_BIN:?REAL_VLLM_BIN is not set}" "$@"
WRAP
  chmod +x "$wrapper_path"
}

start_case() {
  local case_name="$1" topology="$2" gpu_e="$3" gpu_p="$4" gpu_d="$5" vtp_method="$6" images_per_req="$7" vtp_rate="${8:-}"

  local case_root="$RUN_ROOT/$case_name/$vtp_method"
  [[ -n "$vtp_rate" ]] && case_root="$case_root/r${vtp_rate//./p}"
  case_root="$case_root/ipr${images_per_req}"

  local case_run_dir="$case_root/serve_run"
  local case_log="$case_root/launcher.log"
  local wrapper_dir="$case_root/bin"
  mkdir -p "$case_root" "$case_run_dir" "$wrapper_dir"

  create_vllm_wrapper "$wrapper_dir/vllm"
  CURRENT_WRAPPER_DIR="$wrapper_dir"
  CURRENT_CASE_RUN_DIR="$case_run_dir"

  echo
  echo "==== CASE: $case_name ===="
  echo "topology=$topology GPU_E=$gpu_e GPU_P=$gpu_p GPU_D=$gpu_d"
  echo "vt_method=$vtp_method vt_rate=${vtp_rate:-n/a}"
  echo "images_per_req=$images_per_req"
  echo "run_dir=${case_run_dir#$REPO_ROOT/}"

  local -a run_env=(
    env
    PATH="$wrapper_dir:$PATH"
    VLLM_INTERCEPT_BENCH=1
    REAL_VLLM_BIN="$REAL_VLLM_BIN"
    TIMEOUT_SECONDS="$TIMEOUT_SECONDS"
    BENCH_REQUEST_RATE="$BENCH_REQUEST_RATE"
    BENCH_MAX_CONCURRENCY="$BENCH_MAX_CONCURRENCY"
    BENCHMARK="$RUN_BENCHMARK"
    MODEL="$MODEL"
    IMAGES_PER_REQ="$images_per_req"
    NUM_PROMPTS="$NUM_PROMPTS"
    ENFORCE_EAGER=1
    GPU_E="$gpu_e"
    GPU_P="$gpu_p"
    GPU_D="$gpu_d"
    PROXY_PORT="$PROXY_PORT"
    LOG_PATH="$case_run_dir"
    RUN_DIR="$case_run_dir"
  )

  if [[ -n "$vtp_rate" ]]; then
    run_env+=(VISUAL_TOKEN_PRUNING_RATE="$vtp_rate")
  else
    run_env+=(VISUAL_TOKEN_PRUNING_RATE="")
  fi

  setsid "${run_env[@]}" \
    bash ./epdtest/run.sh \
      --topology "$topology" \
      --benchmark "$RUN_BENCHMARK" \
      --images-per-req "$images_per_req" \
      --visual-token-pruning-method "$vtp_method" \
      > "$case_log" 2>&1 &

  CURRENT_GROUP_PID="$!"
  wait_proxy_ready "$CURRENT_GROUP_PID" "$case_log"
  echo "Proxy ready on port $PROXY_PORT for case $case_name"
}

stop_case() {
  cleanup_current_case
}

preflight_lmms_tasks() {
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
      exit 2
      ;;
  esac
}

validate_rate() {
  [[ "$1" =~ ^[0-9]*\.?[0-9]+$ ]] || { echo "Invalid visual-token pruning rate: $1" >&2; exit 2; }
}

validate_images_per_req() {
  [[ "$1" =~ ^[0-9]+$ ]] || { echo "Invalid images-per-req: $1" >&2; exit 2; }
}

extract_tables() {
  local in_log="$1" out_log="$2"
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
  local case_name="$1" vtp_method="$2" images_per_req="$3" vtp_rate="${4:-}"
  local outdir="$EVAL_ROOT/$case_name/$vtp_method"
  [[ -n "$vtp_rate" ]] && outdir="$outdir/r${vtp_rate//./p}"
  outdir="$outdir/ipr${images_per_req}"

  local lmms_log="$outdir/lmms_eval.log"
  local table_log="$outdir/lmms_tables.log"
  local base_url="http://127.0.0.1:${PROXY_PORT}/v1"
  local model_args="model=${MODEL},base_url=${base_url},api_key=${OPENAI_API_KEY},temperature=0,max_new_tokens=128"

  mkdir -p "$outdir"
  export HF_HOME HF_DATASETS_CACHE HUGGINGFACE_HUB_CACHE OPENAI_API_KEY
  export OPENAI_API_BASE="$base_url"
  export OPENAI_BASE_URL="$base_url"

  local -a limit_arg=()
  if [[ -n "$LMMS_LIMIT" && ! "${LMMS_LIMIT,,}" =~ ^(0|all|none)$ ]]; then
    limit_arg=(--limit "$LMMS_LIMIT")
  fi

  echo "LMMS model args ($case_name/$vtp_method${vtp_rate:+/r$vtp_rate}/ipr${images_per_req}): model=${MODEL},base_url=${base_url},api_key=***REDACTED***,temperature=0,max_new_tokens=128"

  "$LMMS_PYTHON_BIN" -m lmms_eval \
    --model openai_compatible \
    --model_args "$model_args" \
    --tasks "$LMMS_TASKS" \
    --batch_size "$LMMS_BATCH_SIZE" \
    --output_path "$outdir" \
    --log_samples \
    "${limit_arg[@]}" 2>&1 | tee "$lmms_log"

  extract_tables "$lmms_log" "$table_log"

  {
    echo "## $case_name | method=$vtp_method | rate=${vtp_rate:--} | ipr=$images_per_req"
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
[[ "${LMMS_LIMIT_DISPLAY,,}" =~ ^(0|all|none)$ ]] && LMMS_LIMIT_DISPLAY="all"
echo "  lmms_limit             : $LMMS_LIMIT_DISPLAY"
echo "  lmms_batch_size        : $LMMS_BATCH_SIZE"
echo "  vtp_methods            : $VTP_METHODS"
echo "  vtp_rates              : $VTP_RATES"
echo "  prune_arch_expect_re   : $PRUNE_ARCH_EXPECT_RE"
echo "  log_root               : ${RUN_ROOT#$REPO_ROOT/}"
echo "  eval_root              : ${EVAL_ROOT#$REPO_ROOT/}"
echo "  table_summary          : ${TABLE_SUMMARY_LOG#$REPO_ROOT/}"

: > "$TABLE_SUMMARY_LOG"
echo "# LMMS Table Summary ($RUN_STAMP)" >> "$TABLE_SUMMARY_LOG"
echo >> "$TABLE_SUMMARY_LOG"

preflight_lmms_tasks

for spec in "${CASES[@]}"; do
  IFS='|' read -r case_name topology gpu_e gpu_p gpu_d <<< "$spec"
  for images_per_req in $IMAGES_PER_REQ; do
    validate_images_per_req "$images_per_req"
    for vtp_method in $VTP_METHODS; do
      validate_method "$vtp_method"

      if [[ "$vtp_method" == "none" ]]; then
        start_case "$case_name" "$topology" "$gpu_e" "$gpu_p" "$gpu_d" "$vtp_method" "$images_per_req"
        run_lmms "$case_name" "$vtp_method" "$images_per_req"
        stop_case
        continue
      fi

      for vtp_rate in $VTP_RATES; do
        validate_rate "$vtp_rate"
        start_case "$case_name" "$topology" "$gpu_e" "$gpu_p" "$gpu_d" "$vtp_method" "$images_per_req" "$vtp_rate"
        verify_prune_architecture_ready "$case_name" "$vtp_method" "$vtp_rate"
        run_lmms "$case_name" "$vtp_method" "$images_per_req" "$vtp_rate"
        stop_case
      done
    done
  done
done

echo
echo "Done."
