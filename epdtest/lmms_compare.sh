#!/bin/bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  bash epdtest/lmms_compare.sh

Runs LMMS evaluation across 6 topology/GPU cases:
  1) 1e1p1d (GPU_E=0 GPU_P=1 GPU_D=2)
  2) 1e1pNd (GPU_E=0 GPU_P=1 GPU_D=0,2)
  3) Ne1p1d (GPU_E=0,2 GPU_P=1 GPU_D=2)
  4) Ne1p1d (GPU_E=0,1 GPU_P=1 GPU_D=2)
  5) Ne1p1d (GPU_E=0,1,2 GPU_P=1 GPU_D=2)
  6) Ne1p1d_pd_preempt (GPU_E=0,1,2 GPU_P=1 GPU_D=2)

Notes:
  - This script follows eval_lmms.sh conventions for LMMS options/env.
  - Topology servers are launched through epdtest/run.sh.
  - The built-in benchmark stage is intercepted/held while LMMS runs.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)
cd "$REPO_ROOT"

###############################################################################
# Config (override via env)
###############################################################################
MODEL="${MODEL:-Qwen/Qwen2.5-VL-3B-Instruct}"
RUN_BENCHMARK="${RUN_BENCHMARK:-randommm}"
PROXY_PORT="${PROXY_PORT:-10001}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-600}"
BENCH_REQUEST_RATE="${BENCH_REQUEST_RATE:-8}"
BENCH_MAX_CONCURRENCY="${BENCH_MAX_CONCURRENCY:-32}"
IMAGES_PER_REQ="${IMAGES_PER_REQ:-1}"
NUM_PROMPTS="${NUM_PROMPTS:-300}"
SERVER_READY_TIMEOUT_SECONDS="${SERVER_READY_TIMEOUT_SECONDS:-900}"

LMMS_TASKS="${LMMS_TASKS:-mmmu_val}"
LMMS_LIMIT="${LMMS_LIMIT:-300}"   # 0 means no --limit
LMMS_BATCH_SIZE="${LMMS_BATCH_SIZE:-1}"
LMMS_MODEL="${LMMS_MODEL:-openai_compatible}"
LMMS_MODEL_ARGS_TEMPLATE="${LMMS_MODEL_ARGS_TEMPLATE:-model={MODEL},base_url=http://127.0.0.1:{PORT}/v1,api_key=EMPTY,temperature=0,max_new_tokens=128}"
OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
FORCE_LOCAL_OPENAI_BASE_URL="${FORCE_LOCAL_OPENAI_BASE_URL:-1}"

HF_HOME="${HF_HOME:-/workspace/.hf_cache}"
HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"

LOG_PATH="${LOG_PATH:-$REPO_ROOT/epdtest/lmms_eval}"
EVAL_OUTPUT_ROOT="${EVAL_OUTPUT_ROOT:-$REPO_ROOT/epdtest/lmms_eval}"
RUN_STAMP="${RUN_STAMP:-$(date +"%Y%m%d_%H%M%S")}"
RUN_ROOT="${LOG_PATH}/${RUN_STAMP}_compare"
EVAL_ROOT="${EVAL_OUTPUT_ROOT}/${RUN_STAMP}_compare"

mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$HUGGINGFACE_HUB_CACHE" "$RUN_ROOT" "$EVAL_ROOT"

python_supports_lmms_eval() {
    local py="$1"
    if [[ -z "$py" ]]; then
        return 1
    fi
    if [[ "$py" == */* ]]; then
        [[ -x "$py" ]] || return 1
    else
        command -v "$py" >/dev/null 2>&1 || return 1
    fi

    "$py" - <<'PY' >/dev/null 2>&1
import importlib.util
import sys
sys.exit(0 if importlib.util.find_spec("lmms_eval") is not None else 1)
PY
}

declare -a LMMS_PY_CANDIDATES=()
if [[ -n "${LMMS_PYTHON_BIN:-}" ]]; then
    LMMS_PY_CANDIDATES+=("$LMMS_PYTHON_BIN")
fi
if [[ -n "${VIRTUAL_ENV:-}" && -x "${VIRTUAL_ENV}/bin/python" ]]; then
    LMMS_PY_CANDIDATES+=("${VIRTUAL_ENV}/bin/python")
fi
if [[ -x "$REPO_ROOT/.venv/bin/python" ]]; then
    LMMS_PY_CANDIDATES+=("$REPO_ROOT/.venv/bin/python")
fi
LMMS_PY_CANDIDATES+=("python3" "python")

LMMS_PYTHON_BIN=""
for candidate in "${LMMS_PY_CANDIDATES[@]}"; do
    if python_supports_lmms_eval "$candidate"; then
        LMMS_PYTHON_BIN="$candidate"
        break
    fi
done

if [[ -z "$LMMS_PYTHON_BIN" ]]; then
    echo "Could not find a Python with lmms_eval installed." >&2
    echo "Install hint: .venv/bin/python -m pip install lmms-eval" >&2
    echo "Or set LMMS_PYTHON_BIN to a Python that has lmms_eval." >&2
    exit 2
fi

REAL_VLLM_BIN="${REAL_VLLM_BIN:-$REPO_ROOT/.venv/bin/vllm}"
if [[ ! -x "$REAL_VLLM_BIN" ]]; then
    if command -v vllm >/dev/null 2>&1; then
        REAL_VLLM_BIN="$(command -v vllm)"
    else
        echo "Could not find vllm executable. Set REAL_VLLM_BIN or activate the venv." >&2
        exit 2
    fi
fi

declare -a CASE_NAMES=(
    "1e1p1d_e0_p1_d2"
    "1e1pNd_e0_p1_d0-2"
    "Ne1p1d_e0-2_p1_d2"
    "Ne1p1d_e0-1_p1_d2"
    "Ne1p1d_e0-1-2_p1_d2"
    "Ne1p1d_pd_preempt_e0-1-2_p1_d2"
)

declare -a CASE_TOPOLOGIES=(
    "1e1p1d"
    "1e1pNd"
    "Ne1p1d"
    "Ne1p1d"
    "Ne1p1d"
    "Ne1p1d_pd_preempt"
)

declare -a CASE_GPU_E=(
    "0"
    "0"
    "0,2"
    "0,1"
    "0,1,2"
    "0,1,2"
)

declare -a CASE_GPU_P=("1" "1" "1" "1" "1" "1")
declare -a CASE_GPU_D=("2" "0,2" "2" "2" "2" "2")

CURRENT_GROUP_PID=""
CURRENT_WRAPPER_DIR=""

###############################################################################
# Helpers
###############################################################################
cleanup() {
    local rc=$?
    set +e
    trap - EXIT INT TERM
    if [[ -n "${CURRENT_GROUP_PID:-}" ]]; then
        kill -- -"${CURRENT_GROUP_PID}" 2>/dev/null || true
        sleep 2
        kill -9 -- -"${CURRENT_GROUP_PID}" 2>/dev/null || true
    fi
    if [[ -n "${CURRENT_WRAPPER_DIR:-}" && -d "${CURRENT_WRAPPER_DIR}" ]]; then
        rm -rf "${CURRENT_WRAPPER_DIR}" 2>/dev/null || true
    fi
    exit "$rc"
}
trap cleanup EXIT INT TERM

port_in_use() {
    local port="$1"
    ss -ltn 2>/dev/null | awk '{print $4}' | grep -Eq "[:.]${port}$"
}

wait_for_port_down() {
    local port="$1"
    local timeout_sec="${2:-60}"
    local deadline=$((SECONDS + timeout_sec))
    while (( SECONDS < deadline )); do
        if ! port_in_use "$port"; then
            return 0
        fi
        sleep 1
    done
    return 1
}

wait_for_proxy_ready() {
    local port="$1"
    local run_group_pid="$2"
    local run_log="$3"
    local deadline=$((SECONDS + SERVER_READY_TIMEOUT_SECONDS))
    local url="http://127.0.0.1:${port}/v1/models"

    while (( SECONDS < deadline )); do
        if ! kill -0 "$run_group_pid" 2>/dev/null; then
            echo "Case launcher exited before proxy became ready. Log: ${run_log}" >&2
            tail -n 200 "$run_log" >&2 || true
            return 1
        fi
        if curl -fsS "$url" >/dev/null 2>&1; then
            return 0
        fi
        sleep 1
    done

    echo "Timed out waiting for proxy readiness on port ${port}. Log: ${run_log}" >&2
    tail -n 200 "$run_log" >&2 || true
    return 1
}

create_vllm_wrapper() {
    local wrapper_dir="$1"
    mkdir -p "$wrapper_dir"
    cat > "${wrapper_dir}/vllm" <<'WRAP'
#!/bin/bash
set -euo pipefail
if [[ "${VLLM_INTERCEPT_BENCH:-0}" == "1" && "${1:-}" == "bench" && "${2:-}" == "serve" ]]; then
    echo "[vllm-wrapper] intercepted: vllm bench serve (holding process for LMMS eval)"
    while true; do
        sleep 3600
    done
fi
exec "${REAL_VLLM_BIN:?REAL_VLLM_BIN is not set}" "$@"
WRAP
    chmod +x "${wrapper_dir}/vllm"
}

start_case_servers() {
    local case_name="$1"
    local topology="$2"
    local gpu_e="$3"
    local gpu_p="$4"
    local gpu_d="$5"

    local case_root="${RUN_ROOT}/${case_name}"
    local case_run_dir="${case_root}/serve_run"
    local case_log="${case_root}/launcher.log"
    local wrapper_dir="${case_root}/bin"
    mkdir -p "$case_root" "$case_run_dir"

    create_vllm_wrapper "$wrapper_dir"
    CURRENT_WRAPPER_DIR="$wrapper_dir"

    echo
    echo "==== CASE: ${case_name} ===="
    echo "topology=${topology} GPU_E=${gpu_e} GPU_P=${gpu_p} GPU_D=${gpu_d}"
    echo "run_dir=${case_run_dir#$REPO_ROOT/}"

    # setsid creates an isolated process group; we can tear down all descendants
    # with kill -- -PID after LMMS finishes.
    setsid env \
        PATH="${wrapper_dir}:$PATH" \
        VLLM_INTERCEPT_BENCH=1 \
        REAL_VLLM_BIN="$REAL_VLLM_BIN" \
        TIMEOUT_SECONDS="$TIMEOUT_SECONDS" \
        BENCH_REQUEST_RATE="$BENCH_REQUEST_RATE" \
        BENCH_MAX_CONCURRENCY="$BENCH_MAX_CONCURRENCY" \
        BENCHMARK="$RUN_BENCHMARK" \
        MODEL="$MODEL" \
        IMAGES_PER_REQ="$IMAGES_PER_REQ" \
        NUM_PROMPTS="$NUM_PROMPTS" \
        GPU_E="$gpu_e" \
        GPU_P="$gpu_p" \
        GPU_D="$gpu_d" \
        LOG_PATH="$RUN_ROOT" \
        RUN_DIR="$case_run_dir" \
        bash ./epdtest/run.sh \
            --topology "$topology" \
            --benchmark "$RUN_BENCHMARK" \
            --images-per-req "$IMAGES_PER_REQ" \
            >"$case_log" 2>&1 &

    CURRENT_GROUP_PID="$!"
    wait_for_proxy_ready "$PROXY_PORT" "$CURRENT_GROUP_PID" "$case_log"
    echo "Proxy ready on port ${PROXY_PORT} for case ${case_name}"
}

stop_case_servers() {
    if [[ -n "${CURRENT_GROUP_PID:-}" ]]; then
        kill -- -"${CURRENT_GROUP_PID}" 2>/dev/null || true
        sleep 2
        kill -9 -- -"${CURRENT_GROUP_PID}" 2>/dev/null || true
        CURRENT_GROUP_PID=""
    fi

    if [[ -n "${CURRENT_WRAPPER_DIR:-}" && -d "${CURRENT_WRAPPER_DIR}" ]]; then
        rm -rf "${CURRENT_WRAPPER_DIR}" 2>/dev/null || true
        CURRENT_WRAPPER_DIR=""
    fi

    if ! wait_for_port_down "$PROXY_PORT" 120; then
        echo "Warning: proxy port ${PROXY_PORT} still appears in use; continuing." >&2
    fi
}

run_lmms_eval_for_case() {
    local case_name="$1"
    local port="$2"

    export HF_HOME HF_DATASETS_CACHE HUGGINGFACE_HUB_CACHE OPENAI_API_KEY

    local outdir="${EVAL_ROOT}/${case_name}"
    mkdir -p "$outdir"

    local model_args="${LMMS_MODEL_ARGS_TEMPLATE//\{PORT\}/$port}"
    model_args="${model_args//\{MODEL\}/$MODEL}"

    case "$LMMS_MODEL" in
        openai|openai_compatible|openai_compatible_chat|async_openai|async_openai_compatible|async_openai_compatible_chat)
            local local_base_url="http://127.0.0.1:${port}/v1"
            local canonical_args="model=${MODEL},base_url=${local_base_url},api_key=${OPENAI_API_KEY},temperature=0,max_new_tokens=128"
            export OPENAI_API_BASE="$local_base_url"

            if [[ "$FORCE_LOCAL_OPENAI_BASE_URL" == "1" ]]; then
                if [[ "$model_args" == *"base_url="* ]]; then
                    model_args=$(echo "$model_args" | sed -E "s|base_url=[^,]*|base_url=${local_base_url}|g")
                else
                    model_args="${model_args},base_url=${local_base_url}"
                fi
            fi
            if [[ "$model_args" != *"api_key="* ]]; then
                model_args="${model_args},api_key=${OPENAI_API_KEY}"
            fi
            if [[ "$model_args" == *"model="* ]]; then
                model_args=$(echo "$model_args" | sed -E "s|model=[^,}]*}?|model=${MODEL}|g")
            else
                model_args="model=${MODEL},${model_args}"
            fi
            if [[ "$model_args" == *"{MODEL"* || "$model_args" == *"{PORT"* || "$model_args" == *"{"* || "$model_args" == *"}"* ]]; then
                model_args="$canonical_args"
            fi
            ;;
    esac

    local -a LIMIT_ARGS=()
    if [[ -n "${LMMS_LIMIT}" && "${LMMS_LIMIT}" != "0" ]]; then
        LIMIT_ARGS+=(--limit "$LMMS_LIMIT")
    fi

    local model_args_for_log="$model_args"
    model_args_for_log=$(echo "$model_args_for_log" | sed -E 's/(api_key=)[^,]*/\1***REDACTED***/g')
    echo "LMMS model args (${case_name}): ${model_args_for_log}"

    "$LMMS_PYTHON_BIN" -m lmms_eval \
        --model "$LMMS_MODEL" \
        --model_args "$model_args" \
        --tasks "$LMMS_TASKS" \
        --batch_size "$LMMS_BATCH_SIZE" \
        --output_path "$outdir" \
        --log_samples \
        "${LIMIT_ARGS[@]}"

    echo "LMMS output saved: ${outdir#$REPO_ROOT/}"
}

summarize_outputs() {
    local root="$1"
    "$LMMS_PYTHON_BIN" - "$root" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])

def collect_metrics(obj, prefix=""):
    out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            key = f"{prefix}.{k}" if prefix else k
            out.update(collect_metrics(v, key))
    elif isinstance(obj, (int, float)):
        lk = prefix.lower()
        if any(tok in lk for tok in ("acc", "score", "exact_match")):
            out[prefix] = obj
    return out

case_dirs = sorted([p for p in root.iterdir() if p.is_dir()])
for case_dir in case_dirs:
    json_files = sorted(case_dir.rglob("*.json"))
    if not json_files:
        print(f"[{case_dir.name}] no json outputs found")
        continue
    ranked = sorted(
        json_files,
        key=lambda p: ("result" not in p.name.lower(), "metric" not in p.name.lower(), -p.stat().st_size),
    )
    chosen = ranked[0]
    try:
        data = json.loads(chosen.read_text())
    except Exception as e:
        print(f"[{case_dir.name}] failed to parse {chosen}: {e}")
        continue
    metrics = collect_metrics(data)
    print(f"\n[{case_dir.name}] metrics from {chosen.name}")
    if not metrics:
        print("  (no accuracy-like numeric metrics auto-detected)")
    else:
        for k in sorted(metrics):
            print(f"  {k}: {metrics[k]}")
PY
}

###############################################################################
# Main
###############################################################################
echo "lmms_compare"
echo "  model                  : $MODEL"
echo "  run_benchmark          : $RUN_BENCHMARK"
echo "  proxy_port             : $PROXY_PORT"
echo "  timeout_seconds        : $TIMEOUT_SECONDS"
echo "  bench_request_rate     : $BENCH_REQUEST_RATE"
echo "  bench_max_concurrency  : $BENCH_MAX_CONCURRENCY"
echo "  images_per_req         : $IMAGES_PER_REQ"
echo "  lmms_tasks             : $LMMS_TASKS"
echo "  lmms_limit             : $LMMS_LIMIT"
echo "  lmms_batch_size        : $LMMS_BATCH_SIZE"
echo "  log_root               : ${RUN_ROOT#$REPO_ROOT/}"
echo "  eval_root              : ${EVAL_ROOT#$REPO_ROOT/}"

for idx in "${!CASE_NAMES[@]}"; do
    case_name="${CASE_NAMES[$idx]}"
    topology="${CASE_TOPOLOGIES[$idx]}"
    gpu_e="${CASE_GPU_E[$idx]}"
    gpu_p="${CASE_GPU_P[$idx]}"
    gpu_d="${CASE_GPU_D[$idx]}"

    start_case_servers "$case_name" "$topology" "$gpu_e" "$gpu_p" "$gpu_d"
    run_lmms_eval_for_case "$case_name" "$PROXY_PORT"
    stop_case_servers
done

echo
echo "==== Summary (auto-detected metrics) ===="
summarize_outputs "$EVAL_ROOT"
echo
echo "Done."
