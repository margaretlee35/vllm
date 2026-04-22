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
    ls -1dt "$REPO_ROOT"/epdtest/logs/lmms/* 2>/dev/null | head -n1 || \
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
    LMMS_ROOT_DEFAULT="$REPO_ROOT/epdtest/logs/lmms/${RUN_STAMP}"
    if [[ ! -d "$LMMS_ROOT_DEFAULT" ]]; then
        LMMS_ROOT_DEFAULT="$REPO_ROOT/epdtest/logs/lmms/${RUN_STAMP}_compare"
    fi
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
    local -a existing_files=()
    local f
    for f in "$@"; do
        [[ -f "$f" ]] && existing_files+=("$f")
    done
    if [[ ${#existing_files[@]} -gt 0 ]]; then
        error_line=$(grep -hEi \
            "Traceback|RuntimeError|ValueError|NVMLError|EngineCore failed|ERROR|address already in use|No CUDA runtime|free memory on device|NIXL is not available|failed to start|cannot open shared object" \
            "${existing_files[@]}" 2>/dev/null | tail -n1 || true)
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

resolve_target_log_path() {
    local launcher_log="$1"
    local fallback_dir="${2:-}"

    local target_output_rel target_log_rel target_log run_dir_abs
    target_output_rel="$(awk -F':' '/target_output[[:space:]]*:/ {sub(/^[[:space:]]*/, "", $2); print $2; exit}' "$launcher_log" 2>/dev/null || true)"
    target_log_rel="$(awk -F':' '/run_dir[[:space:]]*:/ {sub(/^[[:space:]]*/, "", $2); print $2; exit}' "$launcher_log" 2>/dev/null || true)"
    target_log=""

    if [[ -n "$target_output_rel" ]]; then
        target_log="$(resolve_repo_path "$target_output_rel" || true)"
        if [[ -f "$target_log" ]]; then
            echo "$target_log"
            return
        fi
    fi

    if [[ -n "$target_log_rel" ]]; then
        run_dir_abs="$(resolve_repo_path "$target_log_rel" || true)"
        if [[ -n "$run_dir_abs" && -d "$run_dir_abs" ]]; then
            if [[ -f "$run_dir_abs/target_script.log" ]]; then
                echo "$run_dir_abs/target_script.log"
                return
            fi
            target_log="$(find "$run_dir_abs" -maxdepth 3 -type f -name "target_script.log" | sort | tail -n1 || true)"
            if [[ -n "$target_log" && -f "$target_log" ]]; then
                echo "$target_log"
                return
            fi
        fi
    fi

    if [[ -n "$fallback_dir" ]]; then
        if [[ -f "$fallback_dir/target_script.log" ]]; then
            echo "$fallback_dir/target_script.log"
            return
        fi
        target_log="$(find "$fallback_dir" -maxdepth 3 -type f -name "target_script.log" | sort | tail -n1 || true)"
        if [[ -n "$target_log" && -f "$target_log" ]]; then
            echo "$target_log"
            return
        fi
    fi

    echo ""
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

    local runs=0
    local fails=0
    local launcher_log
    while IFS= read -r launcher_log; do
        [[ -n "$launcher_log" ]] || continue
        runs=$((runs + 1))

        local run_root table_log eval_log rel
        run_root="$(dirname "$launcher_log")"
        table_log="$run_root/lmms_tables.log"
        eval_log="$run_root/lmms_eval.log"
        rel="${run_root#$root/}"

        local case_name method rate
        case_name="$(awk -F'/' '{print $1}' <<< "$rel")"
        method="-"
        rate="-"
        if [[ "$rel" == */* ]]; then
            method="$(awk -F'/' '{print $2}' <<< "$rel")"
        fi
        if [[ "$rel" == */*/* ]]; then
            local rate_tag
            rate_tag="$(awk -F'/' '{print $3}' <<< "$rel")"
            if [[ "$rate_tag" == r* ]]; then
                rate="${rate_tag#r}"
                rate="${rate//p/.}"
            fi
        fi

        local case_label="$case_name"
        if [[ "$method" != "-" && -n "$method" ]]; then
            case_label="$case_label/$method"
        fi
        if [[ "$rate" != "-" ]]; then
            case_label="$case_label/r$rate"
        fi

        local run_dir_rel run_dir=""
        run_dir_rel="$(awk -F':' '/run_dir[[:space:]]*:/ {sub(/^[[:space:]]*/, "", $2); print $2; exit}' "$launcher_log" 2>/dev/null || true)"
        if [[ -n "$run_dir_rel" ]]; then
            run_dir="$(resolve_repo_path "$run_dir_rel" || true)"
        fi

        local -a inspect_files=("$launcher_log" "$table_log" "$eval_log")
        if [[ -n "$run_dir" && -d "$run_dir" ]]; then
            while IFS= read -r f; do
                inspect_files+=("$f")
            done < <(find "$run_dir" -maxdepth 2 -type f -name "*.log" | sort)
        fi

        local error_line status
        error_line="$(extract_error_line_from_files "${inspect_files[@]}")"
        status="OK"

        local mmmu_acc="-" avg_speed="-"
        if [[ -f "$table_log" ]]; then
            mmmu_acc="$(awk -F'|' '/\|mmmu_val\|/ {gsub(/[[:space:]]/, "", $7); print $7; exit}' "$table_log" || true)"
            avg_speed="$(awk -F'|' '/\|avg_speed/ {gsub(/[[:space:]]/, "", $3); print $3; exit}' "$table_log" || true)"
            [[ -n "$mmmu_acc" ]] || mmmu_acc="-"
            [[ -n "$avg_speed" ]] || avg_speed="-"
        fi

        if [[ "$mmmu_acc" == "-" ]]; then
            status="FAIL"
        elif [[ -n "$error_line" ]]; then
            status="FAIL"
        fi
        if [[ "$status" == "FAIL" ]]; then
            fails=$((fails + 1))
        fi

        echo "- ${case_label}: ${status} (mmmu_acc=${mmmu_acc}, avg_speed=${avg_speed})"
        if [[ "$status" == "FAIL" ]]; then
            [[ -n "$error_line" ]] || error_line="no LMMS table metrics detected"
            echo "  cause: $error_line"
            echo "  launcher: ${launcher_log#$REPO_ROOT/}"
            if [[ -n "$run_dir" ]]; then
                echo "  run_dir: ${run_dir#$REPO_ROOT/}"
            fi
        fi
    done < <(find "$root" -mindepth 1 -maxdepth 4 -type f -name "launcher.log" | sort)

    if [[ $runs -eq 0 ]]; then
        echo "status: no case launcher logs found"
    else
        echo "cases: ${runs}, failures: ${fails}"
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

        local case_name ipr launcher_log target_log rel
        rel="${run_dir#$root/}"
        case_name="$(awk -F'/' '{print $1}' <<< "$rel")"
        ipr="$(basename "$run_dir")"
        launcher_log="$run_dir/launcher.log"

        target_log="$(resolve_target_log_path "$launcher_log" "$run_dir")"

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
    done < <(find "$root" -mindepth 2 -maxdepth 4 -type d -name "ipr*" | sort)

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

        local rel case_name method rate ipr
        rel="${run_dir#$root/}"  # case/method/ipr1 OR case/method/r0p5/ipr1
        case_name="-"
        method="none"
        rate="-"
        ipr="$(basename "$run_dir")"

        local -a parts=()
        IFS='/' read -r -a parts <<< "$rel"
        local part_count="${#parts[@]}"
        if (( part_count == 2 )); then
            # method/ipr OR case/ipr (flattened none layout)
            case "${parts[0]}" in
                none|visionzip|cdpruner)
                    method="${parts[0]}"
                    ;;
                *)
                    case_name="${parts[0]}"
                    method="none"
                    ;;
            esac
        elif (( part_count >= 3 )); then
            local maybe_rate maybe_method
            maybe_rate="${parts[$((part_count - 2))]}"
            if [[ "$maybe_rate" == r* ]]; then
                rate="${maybe_rate#r}"
                rate="${rate//p/.}"
                maybe_method="${parts[$((part_count - 3))]}"
                method="$maybe_method"
                if (( part_count >= 4 )); then
                    case_name="${parts[$((part_count - 4))]}"
                fi
            else
                method="$maybe_rate"
                case_name="${parts[$((part_count - 3))]}"
            fi
        fi

        local launcher_log target_log
        launcher_log="$run_dir/launcher.log"
        target_log="$(resolve_target_log_path "$launcher_log" "$run_dir")"

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

        error_line="${error_line//$'\t'/ }"
        local error_line_print
        error_line_print="$error_line"
        [[ -n "$error_line_print" ]] || error_line_print="-"
        local launcher_rel target_rel
        launcher_rel="${launcher_log#$REPO_ROOT/}"
        target_rel="-"
        if [[ -f "$target_log" ]]; then
            target_rel="${target_log#$REPO_ROOT/}"
        fi

        printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
            "$case_name" "$method" "$rate" "$ipr" "$status" "$mean_ttft" "$mean_tpot" "$req_tput" "$out_tput" "$total_tput" "$failed_req" "$error_line_print" "$launcher_rel" "$target_rel" >> "$compare_tmp"
    done < <(find "$root" -mindepth 2 -maxdepth 5 -type d -name "ipr*" | sort)

    if [[ -s "$compare_tmp" ]]; then
        local sorted_tmp
        sorted_tmp="$(mktemp)"
        awk -F'\t' '
            function ipr_rank(ipr, v) {
                if (ipr == "ipr1") return 1
                if (ipr == "ipr2") return 2
                if (ipr == "ipr4") return 4
                if (ipr == "ipr8") return 8
                if (ipr ~ /^ipr[0-9]+$/) {
                    v = ipr
                    sub(/^ipr/, "", v)
                    return v + 0
                }
                return 9999
            }
            function rate_rank(rate) {
                if (rate == "-") return -1
                return rate + 0
            }
            {
                # sort by ipr -> case -> method -> rate
                printf "%06d\t%s\t%s\t%010.4f\t%s\n", ipr_rank($4), $1, $2, rate_rank($3), $0
            }
        ' "$compare_tmp" | sort -t$'\t' -k1,1n -k2,2 -k3,3 -k4,4n | cut -f5- > "$sorted_tmp"

        while IFS=$'\t' read -r case_name method rate ipr status mean_ttft mean_tpot req_tput out_tput total_tput failed_req error_line launcher_rel target_rel; do
            echo "- case=${case_name} method=${method} rate=${rate} ${ipr}: ${status} (failed_req=${failed_req}, req/s=${req_tput}, tok/s=${out_tput}, ttft=${mean_ttft}, tpot=${mean_tpot})"
            if [[ -n "$error_line" && "$error_line" != "-" ]]; then
                echo "  cause: $error_line"
                echo "  launcher: $launcher_rel"
                if [[ -n "$target_rel" && "$target_rel" != "-" ]]; then
                    echo "  target: $target_rel"
                fi
            fi
        done < "$sorted_tmp"

        echo
        echo "### VTP TTFT/TPOT/Throughput Comparison"
        echo
        echo "| case | method | rate | ipr | status | Mean TTFT (ms) | Mean TPOT (ms) | req/s | out tok/s | total tok/s |"
        echo "|---|---|---:|---:|---|---:|---:|---:|---:|---:|"
        while IFS=$'\t' read -r c_case c_method c_rate c_ipr c_status c_ttft c_tpot c_req c_out c_total _c_failed _c_error _c_launcher _c_target; do
            echo "| ${c_case} | ${c_method} | ${c_rate} | ${c_ipr} | ${c_status} | ${c_ttft} | ${c_tpot} | ${c_req} | ${c_out} | ${c_total} |"
        done < "$sorted_tmp"
        rm -f "$sorted_tmp"
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
