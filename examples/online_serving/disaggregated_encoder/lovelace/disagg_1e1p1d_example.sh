#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
GIT_ROOT=$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)

echo "Routing to epdtest/run.sh (topology=1e1p1d, profile=simple)"
exec env TOPOLOGY=1e1p1d PROFILE=simple bash "$GIT_ROOT/epdtest/run.sh"
