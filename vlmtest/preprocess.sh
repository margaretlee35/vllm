#!/bin/bash
set -euo pipefail

# Multimodal preprocessing latency entrypoint.
#
# `run.sh` measures the serving side (encode / prefill / decode). It cannot tell
# you how much of TTFT is spent turning raw images into model inputs, because
# that work happens CPU-side in the API server before the request reaches any
# GPU worker, and the serving path does not export those timers.
#
# `vllm bench mm-processor` does export them, via the multimodal timing registry
# that `ObservabilityConfig.enable_mm_processor_stats` turns on. It runs offline
# (in-process `LLM`, one GPU), so use it as the preprocessing companion to a
# `run.sh` serving run rather than as a replacement for one.
#
# Reported stages (see vllm/benchmarks/mm_processor.py):
#   get_mm_hashes_ms          hashing mm items for the processor cache
#   get_cache_missing_items_ms  processor-cache lookup
#   apply_hf_processor_ms     the HF image processor itself (decode/resize/patch)
#   merge_mm_kwargs_ms        collecting processed items into engine kwargs
#   apply_prompt_updates_ms   splicing image placeholders into the prompt
#   preprocessor_total_ms     sum of the above -- the "preprocessing" number
#   encoder_forward_ms        vision tower forward, for comparison
#
# --benchmark video runs scripts/preprocess_video.py instead: same registry
# stages, plus video_decode_ms (MP4 decode + frame sampling), which the
# registry does not cover. Input is the same JSONL format run.sh sends through
# aiperf; pass --input-jsonl to reuse a serving run's file exactly.

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
GIT_ROOT=$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)
VENV_ACTIVATE="$GIT_ROOT/.venv/bin/activate"

MODEL="${MODEL:-Qwen/Qwen2.5-VL-3B-Instruct}"
LOG_PATH="${LOG_PATH:-$SCRIPT_DIR/logs}"
BENCHMARK="${BENCHMARK:-randommm}"
IMAGES_PER_REQ="${IMAGES_PER_REQ:-1}"
NUM_PROMPTS="${NUM_PROMPTS:-64}"
NUM_WARMUPS="${NUM_WARMUPS:-4}"
HF_DATASET_PATH="${HF_DATASET_PATH:-lmarena-ai/VisionArena-Chat}"
METRIC_PERCENTILES="${METRIC_PERCENTILES:-50,99}"

# --benchmark video (same knobs and defaults as run.sh)
INPUT_JSONL="${INPUT_JSONL:-}"
VIDEOS_PER_REQ="${VIDEOS_PER_REQ:-1}"
VIDEO_POOL="${VIDEO_POOL:-}"
VIDEO_SIZE="${VIDEO_SIZE:-320x240}"
VIDEO_FPS="${VIDEO_FPS:-8}"
VIDEO_SECONDS="${VIDEO_SECONDS:-4}"
VIDEO_DIR="${VIDEO_DIR:-/tmp/vlmtest_videos}"
# opencv = CPU decode (vLLM default); nvdec = GPU NVDEC via PyNvVideoCodec
VIDEO_BACKEND="${VIDEO_BACKEND:-opencv}"
USER_TEXT_TOKENS="${USER_TEXT_TOKENS:-300}"
AIPERF_VENV="${AIPERF_VENV:-$SCRIPT_DIR/.venv-aiperf}"

# Preprocessing is CPU work; it needs only one GPU for the engine that consumes
# the processed inputs. Default to the encoder GPU so a run lines up with run.sh.
GPU_PREPROC="${GPU_PREPROC:-${GPU_E:-0}}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-65536}"

# The processor cache makes repeat items nearly free, which hides the real
# per-image cost. Off by default so the numbers reflect cold preprocessing.
MM_PROCESSOR_CACHE_GB="${MM_PROCESSOR_CACHE_GB:-0}"

usage() {
    cat <<'EOF'
Usage:
  bash vlmtest/preprocess.sh [options]

Options:
  --benchmark randommm|simple|video
                                synthetic random-mm, HF VisionArena, or the
                                synthetic MP4 workload of run.sh --benchmark video
  --images-per-req N            images per request (randommm only)
  --videos-per-req N            videos per request (video only)
  --video-backend opencv|nvdec  video decode on CPU (default) or GPU NVDEC
  --video-pool N                unique videos (video only)
  --input-jsonl PATH            reuse an existing video JSONL, e.g. a serving
                                run's logs/<ts>/video_requests.jsonl
  --num-prompts N               requests to measure (default 64)
  -h, --help

Examples:
  bash vlmtest/preprocess.sh
  bash vlmtest/preprocess.sh --images-per-req 4
  bash vlmtest/preprocess.sh --benchmark simple
  NUM_PROMPTS=200 bash vlmtest/preprocess.sh --images-per-req 8
  bash vlmtest/preprocess.sh --benchmark video
  bash vlmtest/preprocess.sh --benchmark video \
      --input-jsonl vlmtest/logs/<ts>/video_requests.jsonl

Environment overrides:
  MODEL, LOG_PATH, NUM_PROMPTS, NUM_WARMUPS, GPU_PREPROC, HF_DATASET_PATH,
  MM_PROCESSOR_CACHE_GB (default 0 = cache disabled), METRIC_PERCENTILES
  VIDEO_SIZE (WxH), VIDEO_FPS, VIDEO_SECONDS, VIDEO_DIR, USER_TEXT_TOKENS,
  AIPERF_VENV                   video mode; run vlmtest/setup_aiperf.sh first
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
        --input-jsonl)
            INPUT_JSONL="$2"
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
    simple|default|hf)
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

# Resolve before the cd below, so a path relative to the caller's cwd works.
if [[ -n "$INPUT_JSONL" ]]; then
    INPUT_JSONL=$(realpath "$INPUT_JSONL")
fi

if [[ -z "${VIRTUAL_ENV:-}" && -f "$VENV_ACTIVATE" ]]; then
    # shellcheck disable=SC1090
    source "$VENV_ACTIVATE"
fi

cd "$GIT_ROOT"

# Same NVHPC workaround as run.sh: Triton JIT-compiles with $CC, and nvc
# rejects gcc flags like -Wno-psabi.
if [[ "${CC:-}" == *nvc* ]]; then
    export CC=gcc CXX=g++
fi

if [[ "$VIDEO_BACKEND" == "nvdec" ]] \
        && ! python -c "import PyNvVideoCodec" 2>/dev/null; then
    echo "--video-backend nvdec needs PyNvVideoCodec in the vLLM venv:" >&2
    echo "  source .venv/bin/activate && uv pip install pynvvideocodec==2.2.3" >&2
    exit 1
fi

RUN_STAMP="${RUN_STAMP:-$(date +"%Y%m%d_%H%M%S")}"
RUN_DIR="${RUN_DIR:-${LOG_PATH}/${RUN_STAMP}_preprocess}"
mkdir -p "$RUN_DIR"
PREPROC_LOG="$RUN_DIR/preprocess.log"
PREPROC_JSON="$RUN_DIR/preprocess.json"

if [[ "$BENCHMARK" == "video" ]]; then
    if [[ -z "$INPUT_JSONL" ]]; then
        if [[ ! -x "$AIPERF_VENV/bin/python" ]]; then
            echo "video generator needs $AIPERF_VENV; run: bash vlmtest/setup_aiperf.sh" >&2
            exit 1
        fi
        INPUT_JSONL="$RUN_DIR/video_requests.jsonl"
        declare -a GEN_ARGS=(
            -n "$NUM_PROMPTS"
            --videos-per-request "$VIDEOS_PER_REQ"
            --user-text-tokens "$USER_TEXT_TOKENS"
            --video-size "${VIDEO_SIZE%x*}" "${VIDEO_SIZE#*x}"
            --fps "$VIDEO_FPS"
            --seconds "$VIDEO_SECONDS"
            --video-dir "$VIDEO_DIR"
            -o "$INPUT_JSONL"
        )
        if [[ -n "$VIDEO_POOL" ]]; then
            GEN_ARGS+=(--videos-pool "$VIDEO_POOL")
        fi
        "$AIPERF_VENV/bin/python" "$SCRIPT_DIR/scripts/gen_video_jsonl.py" "${GEN_ARGS[@]}"
    fi

    echo "vlmtest video preprocessing benchmark"
    echo "  model          : $MODEL"
    echo "  input_jsonl    : ${INPUT_JSONL#$GIT_ROOT/}"
    echo "  num_warmups    : $NUM_WARMUPS"
    echo "  video_backend  : $VIDEO_BACKEND"
    echo "  gpu            : $GPU_PREPROC"
    echo "  mm_cache_gb    : $MM_PROCESSOR_CACHE_GB"
    echo "  run_dir        : ${RUN_DIR#$GIT_ROOT/}"

    CUDA_VISIBLE_DEVICES="$GPU_PREPROC" \
        python "$SCRIPT_DIR/scripts/preprocess_video.py" \
        --input-jsonl "$INPUT_JSONL" \
        --model "$MODEL" \
        --num-warmups "$NUM_WARMUPS" \
        --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
        --max-model-len "$MAX_MODEL_LEN" \
        --mm-processor-cache-gb "$MM_PROCESSOR_CACHE_GB" \
        --metric-percentiles "$METRIC_PERCENTILES" \
        --output-json "$PREPROC_JSON" 2>&1 | tee "$PREPROC_LOG"
    exit "${PIPESTATUS[0]}"
fi

declare -a BENCH_ARGS=(
    --model "$MODEL"
    --seed 0
    --num-prompts "$NUM_PROMPTS"
    --num-warmups "$NUM_WARMUPS"
    --enforce-eager
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
    --max-model-len "$MAX_MODEL_LEN"
    --mm-processor-cache-gb "$MM_PROCESSOR_CACHE_GB"
    --metric-percentiles "$METRIC_PERCENTILES"
    --output-json "$PREPROC_JSON"
)

if [[ "$BENCHMARK" == "randommm" ]]; then
    BENCH_ARGS+=(
        --dataset-name random-mm
        --random-mm-base-items-per-request "$IMAGES_PER_REQ"
        --random-mm-num-mm-items-range-ratio 0
        --random-mm-limit-mm-per-prompt "{\"image\": ${IMAGES_PER_REQ}, \"video\": 0}"
    )
else
    BENCH_ARGS+=(
        --dataset-name hf
        --dataset-path "$HF_DATASET_PATH"
    )
fi

echo "vlmtest preprocessing benchmark"
echo "  model          : $MODEL"
echo "  benchmark      : $BENCHMARK"
if [[ "$BENCHMARK" == "randommm" ]]; then
    echo "  images_per_req : $IMAGES_PER_REQ"
else
    echo "  dataset        : $HF_DATASET_PATH"
fi
echo "  num_prompts    : $NUM_PROMPTS (+${NUM_WARMUPS} warmup)"
echo "  gpu            : $GPU_PREPROC"
echo "  mm_cache_gb    : $MM_PROCESSOR_CACHE_GB"
echo "  run_dir        : ${RUN_DIR#$GIT_ROOT/}"

CUDA_VISIBLE_DEVICES="$GPU_PREPROC" \
    vllm bench mm-processor "${BENCH_ARGS[@]}" 2>&1 | tee "$PREPROC_LOG"
