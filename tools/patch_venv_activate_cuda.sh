#!/usr/bin/env bash
set -euo pipefail

VENV_DIR="${1:-.venv}"
ACTIVATE_PATH="${VENV_DIR}/bin/activate"
MARKER_START="# >>> vllm-cuda-activate-hook >>>"
MARKER_END="# <<< vllm-cuda-activate-hook <<<"

if [[ ! -f "${ACTIVATE_PATH}" ]]; then
  echo "error: activation script not found: ${ACTIVATE_PATH}" >&2
  exit 1
fi

if grep -Fq "${MARKER_START}" "${ACTIVATE_PATH}"; then
  echo "CUDA activate hook already installed in ${ACTIVATE_PATH}"
  exit 0
fi

cat >> "${ACTIVATE_PATH}" <<'EOF'

# >>> vllm-cuda-activate-hook >>>
if [ -z "${CUDA_HOME+_}" ] || [ ! -x "${CUDA_HOME}/bin/nvcc" ]; then
    if command -v nvcc >/dev/null 2>&1 ; then
        _CUDA_NVCC_PATH="$(command -v nvcc)"
        CUDA_HOME="$(dirname "$(dirname "${_CUDA_NVCC_PATH}")")"
        unset _CUDA_NVCC_PATH
    else
        for _CUDA_CANDIDATE in \
            /opt/apps/sysnet/cuda/cuda-12.8 \
            /opt/apps/sysnet/cuda/cuda-12.5 \
            /opt/apps/sysnet/cuda/cuda-12.4 \
            /opt/apps/sysnet/cuda/cuda-12.2 \
            /opt/apps/sysnet/cuda/cuda-11.8 \
            /usr/local/cuda; do
            if [ -x "${_CUDA_CANDIDATE}/bin/nvcc" ]; then
                CUDA_HOME="${_CUDA_CANDIDATE}"
                break
            fi
        done
        unset _CUDA_CANDIDATE
    fi
fi

if [ -n "${CUDA_HOME+_}" ] && [ -x "${CUDA_HOME}/bin/nvcc" ]; then
    export CUDA_HOME

    case ":$PATH:" in
        *":${CUDA_HOME}/bin:"*) ;;
        *) PATH="${CUDA_HOME}/bin:$PATH" ;;
    esac
    export PATH

    if [ -d "${CUDA_HOME}/lib64" ]; then
        case ":${LD_LIBRARY_PATH-}:" in
            *":${CUDA_HOME}/lib64:"*) ;;
            *) LD_LIBRARY_PATH="${CUDA_HOME}/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" ;;
        esac
    fi

    if [ -d "${CUDA_HOME}/targets/x86_64-linux/lib" ]; then
        case ":${LD_LIBRARY_PATH-}:" in
            *":${CUDA_HOME}/targets/x86_64-linux/lib:"*) ;;
            *) LD_LIBRARY_PATH="${CUDA_HOME}/targets/x86_64-linux/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" ;;
        esac
    fi

    export LD_LIBRARY_PATH
fi
# <<< vllm-cuda-activate-hook <<<
EOF

echo "Installed CUDA activate hook into ${ACTIVATE_PATH}"
