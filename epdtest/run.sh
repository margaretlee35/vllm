#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
GIT_ROOT=$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)
IMPL_DIR="$GIT_ROOT/epdtest/scripts"
VENV_ACTIVATE="$GIT_ROOT/.venv/bin/activate"

TOPOLOGY="${TOPOLOGY:-1e1pd}"
PROFILE="${PROFILE:-simple}"
MODEL="${MODEL:-Qwen/Qwen2.5-VL-3B-Instruct}"
LOG_PATH="${LOG_PATH:-$GIT_ROOT/epdtest/logs}"
IMAGES_PER_REQ_LIST="${IMAGES_PER_REQ_LIST:-1 2 4 8}"
INSTALL="${INSTALL:-1}"

if [[ "${SKIP_INSTALL:-0}" == "1" ]]; then
    INSTALL="0"
fi

usage() {
    cat <<'EOF'
Usage:
  bash epdtest/run.sh [options]

Options:
  --topology 1e1pd|1e1p1d
  --profile simple|randommm
  --skip-install
  -h, --help

Examples:
  bash epdtest/run.sh
  bash epdtest/run.sh --profile randommm
  bash epdtest/run.sh --topology 1e1p1d --profile randommm
  bash epdtest/run.sh --skip-install

Notes:
  - This is the main epdtest entrypoint.
  - The old scripts under examples/.../lovelace are preserved only as wrappers.
  - Model, log path, and image sweep stay configurable through environment variables.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --topology)
            TOPOLOGY="$2"
            shift 2
            ;;
        --profile)
            PROFILE="$2"
            shift 2
            ;;
        --skip-install)
            INSTALL="0"
            shift
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

case "$TOPOLOGY" in
    1e1pd|pd)
        TOPOLOGY="1e1pd"
        ;;
    1e1p1d|p1d)
        TOPOLOGY="1e1p1d"
        ;;
    *)
        echo "Unsupported topology: $TOPOLOGY" >&2
        exit 1
        ;;
esac

case "$PROFILE" in
    simple|default)
        PROFILE="simple"
        ;;
    randommm|metrics|rmm)
        PROFILE="randommm"
        ;;
    *)
        echo "Unsupported profile: $PROFILE" >&2
        exit 1
        ;;
esac

if [[ -z "${NUM_PROMPTS:-}" ]]; then
    if [[ "$PROFILE" == "randommm" ]]; then
        export NUM_PROMPTS=500
    else
        export NUM_PROMPTS=100
    fi
fi

if [[ -z "${TIMEOUT_SECONDS:-}" ]]; then
    export TIMEOUT_SECONDS=120
fi

if [[ -z "${VISION_ZIP_RATE:-}" ]]; then
    export VISION_ZIP_RATE=0.5
fi

if [[ -z "${VISION_ZIP_DOMINANT_RATIO:-}" ]]; then
    export VISION_ZIP_DOMINANT_RATIO=0.75
fi

if [[ -z "${VISION_ZIP_ATTENTION_LAYER:-}" ]]; then
    export VISION_ZIP_ATTENTION_LAYER=-2
fi

export TOPOLOGY
export PROFILE
export MODEL
export LOG_PATH

mkdir -p "$LOG_PATH"

activate_venv() {
    if [[ -n "${VIRTUAL_ENV:-}" ]]; then
        return 0
    fi

    if [[ -f "$VENV_ACTIVATE" ]]; then
        # shellcheck disable=SC1090
        source "$VENV_ACTIVATE"
    fi
}

maybe_install() {
    if [[ "$INSTALL" != "1" ]]; then
        return 0
    fi

    if ! command -v uv >/dev/null 2>&1; then
        echo "uv is required unless you use --skip-install" >&2
        exit 1
    fi

    VLLM_USE_PRECOMPILED=1 uv pip install --editable .
    uv pip install "vllm[bench]"
}

case "${TOPOLOGY}:${PROFILE}" in
    1e1pd:simple)
        TARGET_SCRIPT="$IMPL_DIR/disagg_1e1pd_example.sh"
        ;;
    1e1pd:randommm)
        TARGET_SCRIPT="$IMPL_DIR/disagg_1e1pd_rmm.sh"
        ;;
    1e1p1d:simple)
        TARGET_SCRIPT="$IMPL_DIR/disagg_1e1p1d_example.sh"
        ;;
    1e1p1d:randommm)
        TARGET_SCRIPT="$IMPL_DIR/disagg_1e1p1d_rmm.sh"
        ;;
    *)
        echo "Unsupported mode combination: ${TOPOLOGY}:${PROFILE}" >&2
        exit 1
        ;;
esac

run_once() {
    local images_per_req="${1:-}"

    if [[ -n "$images_per_req" ]]; then
        export IMAGES_PER_REQ="$images_per_req"
    fi

    echo "epdtest launcher"
    echo "  topology       : $TOPOLOGY"
    echo "  profile        : $PROFILE"
    echo "  model          : $MODEL"
    echo "  log_path       : ${LOG_PATH#$GIT_ROOT/}"
    if [[ -n "${IMAGES_PER_REQ:-}" ]]; then
        echo "  images_per_req : $IMAGES_PER_REQ"
    fi
    echo "  script         : ${TARGET_SCRIPT#$GIT_ROOT/}"

    bash "$TARGET_SCRIPT"
}

activate_venv
cd "$GIT_ROOT"
maybe_install

if [[ "$PROFILE" == "randommm" ]]; then
    for images_per_req in $IMAGES_PER_REQ_LIST; do
        run_once "$images_per_req"
    done
else
    run_once "${IMAGES_PER_REQ:-}"
fi
