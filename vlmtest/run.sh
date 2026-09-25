#!/bin/bash
set -euo pipefail

# VLM serving run: encoder / prefill / decode as three servers on three GPUs.
#
# Self-contained counterpart to epdtest/run.sh, kept separate so epdtest stays
# untouched. The topology script under vlmtest/scripts/ differs from the epdtest
# one in its UCX transport selection; see the comment there.
#
# Companion run: vlmtest/preprocess.sh, for the multimodal preprocessing latency
# that is folded invisibly into TTFT here.

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
GIT_ROOT=$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)
VENV_ACTIVATE="$GIT_ROOT/.venv/bin/activate"
TARGET_SCRIPT="$SCRIPT_DIR/scripts/disagg_1e1p1d.sh"

MODEL="${MODEL:-Qwen/Qwen2.5-VL-3B-Instruct}"
LOG_PATH="${LOG_PATH:-$SCRIPT_DIR/logs}"
BENCHMARK="${BENCHMARK:-randommm}"
IMAGES_PER_REQ="${IMAGES_PER_REQ:-1}"

# --benchmark video: aiperf against the proxy, fed a JSONL of synthetic MP4s.
VIDEOS_PER_REQ="${VIDEOS_PER_REQ:-1}"
VIDEO_POOL="${VIDEO_POOL:-}"            # unset = every video unique (no reuse)
VIDEO_SIZE="${VIDEO_SIZE:-320x240}"
VIDEO_FPS="${VIDEO_FPS:-8}"
VIDEO_SECONDS="${VIDEO_SECONDS:-4}"
VIDEO_DIR="${VIDEO_DIR:-/tmp/vlmtest_videos}"
# opencv = CPU decode (vLLM default); nvdec = GPU NVDEC via PyNvVideoCodec
VIDEO_BACKEND="${VIDEO_BACKEND:-opencv}"
USER_TEXT_TOKENS="${USER_TEXT_TOKENS:-300}"
AIPERF_VENV="${AIPERF_VENV:-$SCRIPT_DIR/.venv-aiperf}"
STAGE_TRACE="${STAGE_TRACE:-0}"

# E / P / D each get their own GPU.
GPU_E="${GPU_E:-0}"
GPU_P="${GPU_P:-1}"
GPU_D="${GPU_D:-2}"

usage() {
    cat <<'EOF'
Usage:
  bash vlmtest/run.sh [options]

Options:
  --benchmark randommm|simple|video
                                synthetic random-mm, the HF VisionArena set, or
                                synthetic MP4s driven by aiperf
  --images-per-req N            images per request (randommm only)
  --videos-per-req N            videos per request (video only)
  --video-backend opencv|nvdec  video decode on CPU (default) or GPU NVDEC
  --video-pool N                unique videos; < prompts*videos => reuse
  --num-prompts N               benchmark request count
  --stage-trace                 per-request E/P/D stage breakdown (adds GPU
                                syncs: use for attribution, not throughput)
  -h, --help

Examples:
  bash vlmtest/run.sh --benchmark simple --num-prompts 8      # smoke run
  bash vlmtest/run.sh                                        # measurement run
  bash vlmtest/run.sh --images-per-req 4
  bash vlmtest/run.sh --benchmark video --num-prompts 8       # video smoke run
  bash vlmtest/run.sh --benchmark video --video-pool 50       # 300 req, reuse

Environment overrides:
  GPU_E, GPU_P, GPU_D           GPU per role (default 0, 1, 2)
  MODEL, LOG_PATH, NUM_PROMPTS, TIMEOUT_SECONDS
  NIXL_UCX_TLS, NIXL_UCX_NET_DEVICES   UCX fabric for the P->D KV transfer
  VISUAL_TOKEN_PRUNING_METHOD   vision_zip | cdpruner (unset = no pruning)
  VIDEO_SIZE (WxH), VIDEO_FPS, VIDEO_SECONDS, VIDEO_DIR, USER_TEXT_TOKENS,
  AIPERF_VENV                   video mode; run vlmtest/setup_aiperf.sh first
  NVDEC_GPU                     with --video-backend nvdec: spare GPU that all
                                three servers decode on (default: own GPU)

Pruning sweeps and the JSON-driven prune configs stay in epdtest; this folder
passes VISUAL_TOKEN_PRUNING_METHOD straight through.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --benchmark)
            BENCHMARK="$2"
            shift 2
            ;;
        --images-per-req)
            IMAGES_PER_REQ="$2"
            shift 2
            ;;
        --num-prompts)
            NUM_PROMPTS="$2"
            shift 2
            ;;
        --stage-trace)
            STAGE_TRACE=1
            shift
            ;;
        --videos-per-req)
            VIDEOS_PER_REQ="$2"
            shift 2
            ;;
        --video-pool)
            VIDEO_POOL="$2"
            shift 2
            ;;
        --video-backend)
            VIDEO_BACKEND="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

case "$VIDEO_BACKEND" in
    opencv|nvdec)
        ;;
    *)
        echo "Unsupported video backend: $VIDEO_BACKEND (expected opencv or nvdec)" >&2
        exit 1
        ;;
esac
# Read by every vLLM process launched below.
export VLLM_VIDEO_LOADER_BACKEND="$VIDEO_BACKEND"

case "${BENCHMARK,,}" in
    simple|default)
        BENCHMARK="simple"
        ;;
    randommm|rmm|metrics)
        BENCHMARK="randommm"
        ;;
    video)
        BENCHMARK="video"
        ;;
    *)
        echo "Unsupported benchmark mode: $BENCHMARK (expected simple, randommm or video)" >&2
        exit 1
        ;;
esac

# Three servers loading weights blows past a 120s default on a cold cache.
export NUM_PROMPTS="${NUM_PROMPTS:-300}"
export TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-900}"
export MODEL LOG_PATH BENCHMARK IMAGES_PER_REQ GPU_E GPU_P GPU_D STAGE_TRACE
if [[ "$VIDEO_BACKEND" == "nvdec" && -n "${NVDEC_GPU:-}" ]]; then
    export NVDEC_GPU
else
    unset NVDEC_GPU
fi

if [[ -z "${VIRTUAL_ENV:-}" && -f "$VENV_ACTIVATE" ]]; then
    # shellcheck disable=SC1090
    source "$VENV_ACTIVATE"
fi

cd "$GIT_ROOT"

# Triton JIT-compiles helpers with $CC; the default nvidia (NVHPC) module sets
# CC=nvc, which rejects gcc flags like -Wno-psabi. Fall back to gcc.
if [[ "${CC:-}" == *nvc* ]]; then
    export CC=gcc CXX=g++
fi

if [[ "$VIDEO_BACKEND" == "nvdec" ]] \
        && ! python -c "import PyNvVideoCodec" 2>/dev/null; then
    echo "--video-backend nvdec needs PyNvVideoCodec in the vLLM venv:" >&2
    echo "  source .venv/bin/activate && uv pip install pynvvideocodec==2.2.3" >&2
    exit 1
fi

mkdir -p "$LOG_PATH"
RUN_STAMP="${RUN_STAMP:-$(date +"%Y%m%d_%H%M%S")}"
# Exported so the topology script writes its per-server logs into the same
# directory as this launcher's output instead of deriving its own timestamp.
export RUN_DIR="${RUN_DIR:-${LOG_PATH}/${RUN_STAMP}}"
mkdir -p "$RUN_DIR"
TARGET_OUTPUT_LOG="$RUN_DIR/target_script.log"

if [[ "$BENCHMARK" == "video" ]]; then
    if [[ ! -x "$AIPERF_VENV/bin/aiperf" ]]; then
        echo "aiperf not found in $AIPERF_VENV; run: bash vlmtest/setup_aiperf.sh" >&2
        exit 1
    fi
    export AIPERF_BIN="$AIPERF_VENV/bin/aiperf"
    # Generated before the servers start so a bad option fails fast. The MP4
    # pool is cached in VIDEO_DIR by content key and reused across runs.
    export VIDEO_JSONL="$RUN_DIR/video_requests.jsonl"
    declare -a GEN_ARGS=(
        -n "$NUM_PROMPTS"
        --videos-per-request "$VIDEOS_PER_REQ"
        --user-text-tokens "$USER_TEXT_TOKENS"
        --video-size "${VIDEO_SIZE%x*}" "${VIDEO_SIZE#*x}"
        --fps "$VIDEO_FPS"
        --seconds "$VIDEO_SECONDS"
        --video-dir "$VIDEO_DIR"
        -o "$VIDEO_JSONL"
    )
    if [[ -n "$VIDEO_POOL" ]]; then
        GEN_ARGS+=(--videos-pool "$VIDEO_POOL")
    fi
    "$AIPERF_VENV/bin/python" "$SCRIPT_DIR/scripts/gen_video_jsonl.py" "${GEN_ARGS[@]}"
fi

echo "vlmtest launcher (1e1p1d, one GPU per role)"
echo "  model          : $MODEL"
echo "  benchmark      : $BENCHMARK"
if [[ "$BENCHMARK" == "randommm" ]]; then
    echo "  images_per_req : $IMAGES_PER_REQ"
elif [[ "$BENCHMARK" == "video" ]]; then
    echo "  videos_per_req : $VIDEOS_PER_REQ (pool ${VIDEO_POOL:-all unique})"
    echo "  video          : $VIDEO_SIZE ${VIDEO_FPS}fps ${VIDEO_SECONDS}s"
    echo "  video_backend  : $VIDEO_BACKEND${NVDEC_GPU:+ (decode on GPU $NVDEC_GPU)}"
fi
echo "  num_prompts    : $NUM_PROMPTS"
echo "  gpus           : E=$GPU_E P=$GPU_P D=$GPU_D"
if [[ -n "${VISUAL_TOKEN_PRUNING_METHOD:-}" ]]; then
    echo "  vt_method      : $VISUAL_TOKEN_PRUNING_METHOD"
fi
if [[ "$STAGE_TRACE" == "1" ]]; then
    echo "  stage_trace    : on (${RUN_DIR#$GIT_ROOT/}/trace)"
fi
echo "  run_dir        : ${RUN_DIR#$GIT_ROOT/}"

bash "$TARGET_SCRIPT" 2>&1 | tee "$TARGET_OUTPUT_LOG"
