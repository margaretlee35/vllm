#!/bin/bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  bash epdtest/summarize_compare_logs.sh [--run-stamp YYYYMMDD_HHMMSS]

Environment overrides:
  RUN_STAMP
  LMMS_ROOT
  SWEEP_ROOT
  VTP_ROOT
  SUMMARY_FILE
EOF
}

RUN_STAMP="${RUN_STAMP:-}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --run-stamp)
            RUN_STAMP="${2:-}"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)

find_latest_lmms_root() {
    ls -1dt "$REPO_ROOT"/epdtest/logs/lmms/*_compare 2>/dev/null | head -n1 || \
    ls -1dt "$REPO_ROOT"/epdtest/lmms_eval/*_compare 2>/dev/null | head -n1 || true
}

find_latest_sweep_root() {
    ls -1dt "$REPO_ROOT"/epdtest/logs/sweep/*/compare 2>/dev/null | head -n1 || \
    ls -1dt "$REPO_ROOT"/epdtest/logs/*/compare 2>/dev/null | head -n1 || true
}

find_latest_vtp_root() {
    ls -1dt "$REPO_ROOT"/epdtest/logs/vtp/*/vtp_sweep 2>/dev/null | head -n1 || \
    ls -1dt "$REPO_ROOT"/epdtest/logs/*/vtp_sweep 2>/dev/null | head -n1 || true
}

if [[ -n "$RUN_STAMP" ]]; then
    LMMS_ROOT_DEFAULT="$REPO_ROOT/epdtest/logs/lmms/${RUN_STAMP}_compare"
    SWEEP_ROOT_DEFAULT="$REPO_ROOT/epdtest/logs/sweep/${RUN_STAMP}/compare"
    VTP_ROOT_DEFAULT="$REPO_ROOT/epdtest/logs/vtp/${RUN_STAMP}/vtp_sweep"
else
    LMMS_ROOT_DEFAULT="$(find_latest_lmms_root)"
    SWEEP_ROOT_DEFAULT="$(find_latest_sweep_root)"
    VTP_ROOT_DEFAULT="$(find_latest_vtp_root)"
fi

LMMS_ROOT="${LMMS_ROOT:-$LMMS_ROOT_DEFAULT}"
SWEEP_ROOT="${SWEEP_ROOT:-$SWEEP_ROOT_DEFAULT}"
VTP_ROOT="${VTP_ROOT:-$VTP_ROOT_DEFAULT}"

SUMMARY_DIR="$REPO_ROOT/epdtest/slurm_logs"
mkdir -p "$SUMMARY_DIR"

if [[ -n "$RUN_STAMP" ]]; then
    SUMMARY_FILE_DEFAULT="$SUMMARY_DIR/epd_compare_summary_${RUN_STAMP}.txt"
else
    SUMMARY_FILE_DEFAULT="$SUMMARY_DIR/epd_compare_summary_latest.txt"
fi
SUMMARY_FILE="${SUMMARY_FILE:-$SUMMARY_FILE_DEFAULT}"

extract_error_line_from_files() {
    local error_line=""
    if [[ $# -gt 0 ]]; then
        error_line=$(grep -hEi \
            "Traceback|RuntimeError|ValueError|NVMLError|EngineCore failed|ERROR|address already in use|No CUDA runtime|free memory on device|NIXL is not available|failed to start|cannot open shared object" \
            "$@" 2>/dev/null | tail -n1 || true)
    fi
    echo "$error_line"
}

resolve_repo_path() {
    local path="$1"
    if [[ -z "$path" ]]; then
        return 1
    fi
    if [[ "$path" = /* ]]; then
        echo "$path"
    else
        echo "$REPO_ROOT/$path"
    fi
}

summarize_lmms() {
    local root="$1"
    echo "## LMMS Compare"
    echo "root: ${root#$REPO_ROOT/}"

    if [[ -z "$root" || ! -d "$root" ]]; then
        echo "status: root not found"
        echo
        return
    fi

    local cases=0
    local fails=0
    local case_dir
    for case_dir in "$root"/*; do
        [[ -d "$case_dir" ]] || continue
        [[ -f "$case_dir/launcher.log" ]] || continue
        cases=$((cases + 1))

        local case_name
        case_name="$(basename "$case_dir")"
        local launcher_log="$case_dir/launcher.log"
        local table_log="$case_dir/lmms_tables.log"

        local run_dir_rel
        run_dir_rel="$(awk -F':' '/run_dir[[:space:]]*:/ {sub(/^[[:space:]]*/, "", $2); print $2; exit}' "$launcher_log" || true)"
        local run_dir=""
        if [[ -n "$run_dir_rel" ]]; then
            run_dir="$REPO_ROOT/$run_dir_rel"
        fi

        local -a inspect_files=("$launcher_log")
        if [[ -n "$run_dir" && -d "$run_dir" ]]; then
            while IFS= read -r f; do
                inspect_files+=("$f")
            done < <(find "$run_dir" -maxdepth 1 -type f -name "*.log" | sort)
        fi

        local error_line
        error_line="$(extract_error_line_from_files "${inspect_files[@]}")"
        local status="OK"
        if [[ -n "$error_line" ]]; then
            status="FAIL"
            fails=$((fails + 1))
        fi

        local mmmu_acc="-"
        local avg_speed="-"
        if [[ -f "$table_log" ]]; then
            mmmu_acc="$(awk -F'|' '/\|mmmu_val\|/ {gsub(/[[:space:]]/, "", $7); print $7; exit}' "$table_log" || true)"
            avg_speed="$(awk -F'|' '/\|avg_speed/ {gsub(/[[:space:]]/, "", $3); print $3; exit}' "$table_log" || true)"
            [[ -n "$mmmu_acc" ]] || mmmu_acc="-"
            [[ -n "$avg_speed" ]] || avg_speed="-"
        fi

        echo "- ${case_name}: ${status} (mmmu_acc=${mmmu_acc}, avg_speed=${avg_speed})"
        if [[ -n "$error_line" ]]; then
            echo "  cause: $error_line"
            echo "  launcher: ${launcher_log#$REPO_ROOT/}"
            if [[ -n "$run_dir" ]]; then
                echo "  run_dir: ${run_dir#$REPO_ROOT/}"
            fi
        fi
    done

    if [[ $cases -eq 0 ]]; then
        echo "status: no case launcher logs found"
    else
        echo "cases: ${cases}, failures: ${fails}"
    fi
    echo
}

summarize_sweep() {
    local root="$1"
    echo "## Sweep Compare"
    echo "root: ${root#$REPO_ROOT/}"

    if [[ -z "$root" || ! -d "$root" ]]; then
        echo "status: root not found"
        echo
        return
    fi

    local runs=0
    local fails=0
    local run_dir
    while IFS= read -r run_dir; do
        [[ -n "$run_dir" ]] || continue
        runs=$((runs + 1))

        local case_name ipr launcher_log target_log target_log_rel target_output_rel
        case_name="$(basename "$(dirname "$run_dir")")"
        ipr="$(basename "$run_dir")"
        launcher_log="$run_dir/launcher.log"

        # Prefer the true target log path emitted by run.sh; fall back to legacy
        # location under the case run directory.
        target_output_rel="$(awk -F':' '/target_output[[:space:]]*:/ {sub(/^[[:space:]]*/, "", $2); print $2; exit}' "$launcher_log" 2>/dev/null || true)"
        target_log_rel="$(awk -F':' '/run_dir[[:space:]]*:/ {sub(/^[[:space:]]*/, "", $2); print $2; exit}' "$launcher_log" 2>/dev/null || true)"
        target_log=""

        if [[ -n "$target_output_rel" ]]; then
            target_log="$(resolve_repo_path "$target_output_rel" || true)"
        fi
        if [[ -z "$target_log" || ! -f "$target_log" ]]; then
            if [[ -n "$target_log_rel" ]]; then
                target_log="$(resolve_repo_path "$target_log_rel" || true)/target_script.log"
            else
                target_log="$run_dir/target_script.log"
            fi
        fi
        if [[ ! -f "$target_log" ]]; then
            target_log="$run_dir/target_script.log"
        fi

        local error_line
        error_line="$(extract_error_line_from_files "$launcher_log" "$target_log")"

        local req_tput out_tput failed_req
        req_tput="$(awk -F':' '/Request throughput \(req\/s\):/ {gsub(/[[:space:]]/, "", $2); print $2; exit}' "$target_log" 2>/dev/null || true)"
        out_tput="$(awk -F':' '/Output token throughput \(tok\/s\):/ {gsub(/[[:space:]]/, "", $2); print $2; exit}' "$target_log" 2>/dev/null || true)"
        failed_req="$(awk -F':' '/Failed requests:/ {gsub(/[[:space:]]/, "", $2); print $2; exit}' "$target_log" 2>/dev/null || true)"
        [[ -n "$req_tput" ]] || req_tput="-"
        [[ -n "$out_tput" ]] || out_tput="-"
        [[ -n "$failed_req" ]] || failed_req="-"

        local benchmark_done="0"
        if grep -qE "Serving Benchmark Result|Successful requests:" "$target_log" 2>/dev/null || \
           grep -qE "Serving Benchmark Result|Successful requests:" "$launcher_log" 2>/dev/null; then
            benchmark_done="1"
        fi

        local status="OK"
        if [[ "$failed_req" =~ ^[0-9]+$ ]] && (( failed_req > 0 )); then
            status="FAIL"
        elif [[ -n "$error_line" && "$benchmark_done" != "1" ]]; then
            status="FAIL"
        fi
        if [[ "$status" == "FAIL" ]]; then
            fails=$((fails + 1))
        fi

        echo "- ${case_name}/${ipr}: ${status} (failed_req=${failed_req}, req/s=${req_tput}, tok/s=${out_tput})"
        if [[ -n "$error_line" ]]; then
            echo "  cause: $error_line"
            echo "  launcher: ${launcher_log#$REPO_ROOT/}"
            if [[ -f "$target_log" ]]; then
                echo "  target: ${target_log#$REPO_ROOT/}"
            fi
        fi
    done < <(find "$root" -mindepth 2 -maxdepth 2 -type d -name "ipr*" | sort)

    if [[ $runs -eq 0 ]]; then
        echo "status: no sweep runs found"
    else
        echo "runs: ${runs}, failures: ${fails}"
    fi
    echo
}

summarize_vtp() {
    local root="$1"
    echo "## VTP Sweep"
    echo "root: ${root#$REPO_ROOT/}"

    if [[ -z "$root" || ! -d "$root" ]]; then
        echo "status: root not found"
        echo
        return
    fi

    local runs=0
    local fails=0
    local compare_tmp
    compare_tmp="$(mktemp)"
    local run_dir
    while IFS= read -r run_dir; do
        [[ -n "$run_dir" ]] || continue
        runs=$((runs + 1))

        local rel method rate ipr
        rel="${run_dir#$root/}"  # none/ipr1 OR visionzip/r0p5/ipr1
        method="${rel%%/*}"
        rate="-"
        ipr="$(basename "$run_dir")"
        if [[ "$rel" == */r*/ipr* ]]; then
            local rate_tag
            rate_tag="$(basename "$(dirname "$run_dir")")"
            if [[ "$rate_tag" == r* ]]; then
                rate="${rate_tag#r}"
                rate="${rate//p/.}"
            fi
        fi

        local launcher_log target_log target_log_rel target_output_rel
        launcher_log="$run_dir/launcher.log"

        target_output_rel="$(awk -F':' '/target_output[[:space:]]*:/ {sub(/^[[:space:]]*/, "", $2); print $2; exit}' "$launcher_log" 2>/dev/null || true)"
        target_log_rel="$(awk -F':' '/run_dir[[:space:]]*:/ {sub(/^[[:space:]]*/, "", $2); print $2; exit}' "$launcher_log" 2>/dev/null || true)"
        target_log=""

        if [[ -n "$target_output_rel" ]]; then
            target_log="$(resolve_repo_path "$target_output_rel" || true)"
        fi
        if [[ -z "$target_log" || ! -f "$target_log" ]]; then
            if [[ -n "$target_log_rel" ]]; then
                target_log="$(resolve_repo_path "$target_log_rel" || true)/target_script.log"
            else
                target_log="$run_dir/target_script.log"
            fi
        fi
        if [[ ! -f "$target_log" ]]; then
            target_log="$run_dir/target_script.log"
        fi

        local error_line
        error_line="$(extract_error_line_from_files "$launcher_log" "$target_log")"

        local req_tput out_tput total_tput failed_req mean_ttft mean_tpot
        req_tput="$(awk -F':' '/Request throughput \(req\/s\):/ {gsub(/[[:space:]]/, "", $2); print $2; exit}' "$target_log" 2>/dev/null || true)"
        out_tput="$(awk -F':' '/Output token throughput \(tok\/s\):/ {gsub(/[[:space:]]/, "", $2); print $2; exit}' "$target_log" 2>/dev/null || true)"
        total_tput="$(awk -F':' '/Total token throughput \(tok\/s\):/ {gsub(/[[:space:]]/, "", $2); print $2; exit}' "$target_log" 2>/dev/null || true)"
        mean_ttft="$(awk -F':' '/Mean TTFT \(ms\):/ {gsub(/[[:space:]]/, "", $2); print $2; exit}' "$target_log" 2>/dev/null || true)"
        mean_tpot="$(awk -F':' '/Mean TPOT \(ms\):/ {gsub(/[[:space:]]/, "", $2); print $2; exit}' "$target_log" 2>/dev/null || true)"
        failed_req="$(awk -F':' '/Failed requests:/ {gsub(/[[:space:]]/, "", $2); print $2; exit}' "$target_log" 2>/dev/null || true)"
        [[ -n "$req_tput" ]] || req_tput="-"
        [[ -n "$out_tput" ]] || out_tput="-"
        [[ -n "$total_tput" ]] || total_tput="-"
        [[ -n "$mean_ttft" ]] || mean_ttft="-"
        [[ -n "$mean_tpot" ]] || mean_tpot="-"
        [[ -n "$failed_req" ]] || failed_req="-"

        local benchmark_done="0"
        if grep -qE "Serving Benchmark Result|Successful requests:" "$target_log" 2>/dev/null || \
           grep -qE "Serving Benchmark Result|Successful requests:" "$launcher_log" 2>/dev/null; then
            benchmark_done="1"
        fi

        local status="OK"
        if [[ "$failed_req" =~ ^[0-9]+$ ]] && (( failed_req > 0 )); then
            status="FAIL"
        elif [[ -n "$error_line" && "$benchmark_done" != "1" ]]; then
            status="FAIL"
        fi
        if [[ "$status" == "FAIL" ]]; then
            fails=$((fails + 1))
        fi

        echo "- method=${method} rate=${rate} ${ipr}: ${status} (failed_req=${failed_req}, req/s=${req_tput}, tok/s=${out_tput}, ttft=${mean_ttft}, tpot=${mean_tpot})"
        if [[ -n "$error_line" ]]; then
            echo "  cause: $error_line"
            echo "  launcher: ${launcher_log#$REPO_ROOT/}"
            if [[ -f "$target_log" ]]; then
                echo "  target: ${target_log#$REPO_ROOT/}"
            fi
        fi

        printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
            "$method" "$rate" "$ipr" "$status" "$mean_ttft" "$mean_tpot" "$req_tput" "$out_tput" "$total_tput" >> "$compare_tmp"
    done < <(find "$root" -mindepth 2 -maxdepth 3 -type d -name "ipr*" | sort)

    if [[ -s "$compare_tmp" ]]; then
        echo
        echo "### VTP TTFT/TPOT/Throughput Comparison"
        echo
        echo "| method | rate | ipr | status | Mean TTFT (ms) | Mean TPOT (ms) | req/s | out tok/s | total tok/s |"
        echo "|---|---:|---:|---|---:|---:|---:|---:|---:|"
        while IFS=$'\t' read -r c_method c_rate c_ipr c_status c_ttft c_tpot c_req c_out c_total; do
            echo "| ${c_method} | ${c_rate} | ${c_ipr} | ${c_status} | ${c_ttft} | ${c_tpot} | ${c_req} | ${c_out} | ${c_total} |"
        done < "$compare_tmp"
    fi
    rm -f "$compare_tmp"

    if [[ $runs -eq 0 ]]; then
        echo "status: no vtp sweep runs found"
    else
        echo "runs: ${runs}, failures: ${fails}"
    fi
    echo
}

{
    echo "epd_compare summary"
    echo "timestamp: $(date '+%Y-%m-%d %H:%M:%S %Z')"
    if [[ -n "$RUN_STAMP" ]]; then
        echo "run_stamp: $RUN_STAMP"
    fi
    echo
    summarize_lmms "$LMMS_ROOT"
    summarize_sweep "$SWEEP_ROOT"
    summarize_vtp "$VTP_ROOT"
    echo "summary_file: ${SUMMARY_FILE#$REPO_ROOT/}"
} | tee "$SUMMARY_FILE"
