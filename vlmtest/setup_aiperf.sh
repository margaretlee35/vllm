#!/bin/bash
set -euo pipefail

# One-time setup: aiperf (+ imageio-ffmpeg for the video generator) in its own
# venv, kept apart from the vLLM .venv so their dependency pins never collide.
#
# CC=gcc: aiperf pulls in `crick`, a C extension with no aarch64 wheel. The
# NVHPC module's CC=nvc cannot build it.

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
AIPERF_VENV="${AIPERF_VENV:-$SCRIPT_DIR/.venv-aiperf}"
AIPERF_VERSION="${AIPERF_VERSION:-0.13.0}"

uv venv -p 3.12 "$AIPERF_VENV"
CC=gcc CXX=g++ uv pip install -p "$AIPERF_VENV/bin/python" \
    "aiperf==${AIPERF_VERSION}" imageio imageio-ffmpeg

"$AIPERF_VENV/bin/aiperf" --version
echo "aiperf venv ready: $AIPERF_VENV"
