#!/bin/bash
set -euo pipefail

UTIL_DIR="/home/margaretlee35/work/vllm/epdtest/logs/utilization"
PLOT_SCRIPT="${UTIL_DIR}/plot_gpu_utilization.py"
PYTHON_BIN="${PYTHON_BIN:-python}"
LOG_ROOT="${LOG_ROOT:-/home/margaretlee35/work/vllm/epdtest/logs}"

mapfile -t files < <(find "${LOG_ROOT}" -path "${UTIL_DIR}" -prune -o -name sm.log -type f -print | sort)

if [[ ${#files[@]} -eq 0 ]]; then
    echo "No sm.log files found under ${LOG_ROOT}"
    exit 0
fi

for log_file in "${files[@]}"; do
    mapfile -t roles < <(awk -F, 'NR > 1 {print $2}' "$log_file" | sed 's/^[[:space:]]*//; s/[[:space:]]*$//' | sort -u)
    if [[ ${#roles[@]} -eq 0 ]]; then
        echo "Skipping ${log_file}: no roles found"
        continue
    fi

    run_dir=$(dirname "$log_file")
    for role in "${roles[@]}"; do
        [[ -z "$role" ]] && continue
        output_file="${run_dir}/sm_${role}.png"
        echo "Plotting ${log_file} (${role}) -> ${output_file}"
        "${PYTHON_BIN}" "${PLOT_SCRIPT}" "${log_file}" --role "${role}" -o "${output_file}"
    done
done

echo "Done."
