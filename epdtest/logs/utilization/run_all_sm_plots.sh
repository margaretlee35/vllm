#!/bin/bash
set -euo pipefail

UTIL_DIR="/home/margaretlee35/work/vllm/epdtest/logs/utilization"
PLOT_SCRIPT="${UTIL_DIR}/plot_gpu_utilization.py"
PYTHON_BIN="${PYTHON_BIN:-python}"

shopt -s nullglob
files=("${UTIL_DIR}"/sm_*.log)

if [[ ${#files[@]} -eq 0 ]]; then
    echo "No sm_*.log files found in ${UTIL_DIR}"
    exit 0
fi

for log_file in "${files[@]}"; do
    output_file="${log_file%.log}.png"
    echo "Plotting ${log_file} -> ${output_file}"
    "${PYTHON_BIN}" "${PLOT_SCRIPT}" "${log_file}" -o "${output_file}"
done

echo "Done."
