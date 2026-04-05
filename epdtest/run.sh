#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
GIT_ROOT=$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)
IMPL_DIR="$GIT_ROOT/epdtest/scripts"

TOPOLOGY="${TOPOLOGY:-1e1pd}"
PROFILE="${PROFILE:-simple}"

usage() {
    cat <<'EOF'
Usage:
  bash epdtest/run.sh [--topology 1e1pd|1e1p1d] [--profile simple|metrics]

Defaults:
  topology = 1e1pd
  profile  = simple

Examples:
  bash epdtest/run.sh
  bash epdtest/run.sh --topology 1e1p1d
  bash epdtest/run.sh --profile metrics
  TOPOLOGY=1e1p1d PROFILE=metrics GPU_E=0 GPU_P=1 GPU_D=2 bash epdtest/run.sh

Notes:
  - This is the direct Lovelace entrypoint.
  - The old scripts under examples/.../lovelace are preserved only as wrappers.
  - Logs and local analysis helpers live under epdtest/logs/.
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
    metrics|rmm)
        PROFILE="metrics"
        ;;
    *)
        echo "Unsupported profile: $PROFILE" >&2
        exit 1
        ;;
esac

case "${TOPOLOGY}:${PROFILE}" in
    1e1pd:simple)
        TARGET_SCRIPT="$IMPL_DIR/disagg_1e1pd_example.sh"
        ;;
    1e1pd:metrics)
        TARGET_SCRIPT="$IMPL_DIR/disagg_1e1pd_rmm.sh"
        ;;
    1e1p1d:simple)
        TARGET_SCRIPT="$IMPL_DIR/disagg_1e1p1d_example.sh"
        ;;
    1e1p1d:metrics)
        TARGET_SCRIPT="$IMPL_DIR/disagg_1e1p1d_rmm.sh"
        ;;
    *)
        echo "Unsupported mode combination: ${TOPOLOGY}:${PROFILE}" >&2
        exit 1
        ;;
esac

echo "Lovelace launcher"
echo "  topology : $TOPOLOGY"
echo "  profile  : $PROFILE"
echo "  script   : ${TARGET_SCRIPT#$GIT_ROOT/}"

cd "$GIT_ROOT"
exec bash "$TARGET_SCRIPT"
