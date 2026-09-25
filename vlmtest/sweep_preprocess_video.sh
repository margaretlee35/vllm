#!/bin/bash
set -uo pipefail

# Sweep video preprocessing latency over clip resolution x duration x decode
# backend, one preprocess.sh --benchmark video run per point, then collect the
# per-stage means into summary.csv.
#
# Runs are sequential on purpose: most stages are CPU work, and concurrent runs
# would compete for cores and inflate each other.
#
# 16 unique clips per point is enough: the processor cache is off by default, so
# a repeated clip is decoded and processed again, same as a new one.

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

SIZES="${SIZES:-320x240 640x480 1280x720 1920x1080}"
SECONDS_LIST="${SECONDS_LIST:-4 8 16}"
BACKENDS="${BACKENDS:-opencv nvdec}"
NUM_PROMPTS="${NUM_PROMPTS:-64}"
VIDEO_POOL="${VIDEO_POOL:-16}"
export VIDEO_FPS="${VIDEO_FPS:-8}"

# The model's GPU, and a second one that nvdec decodes on. Sharing the model's
# GPU makes nvdec slower than CPU decode (see README), so it is not swept.
GPU_MODEL="${GPU_MODEL:-0}"
GPU_NVDEC="${GPU_NVDEC:-3}"

SWEEP_DIR="${SWEEP_DIR:-$SCRIPT_DIR/logs/$(date +"%Y%m%d_%H%M%S")_video_sweep}"
mkdir -p "$SWEEP_DIR"
echo "sweep: sizes=[$SIZES] seconds=[$SECONDS_LIST] backends=[$BACKENDS] fps=$VIDEO_FPS"
echo "       $NUM_PROMPTS prompts, pool $VIDEO_POOL -> $SWEEP_DIR"

for backend in $BACKENDS; do
    for size in $SIZES; do
        for secs in $SECONDS_LIST; do
            point="${backend}_${size}_${secs}s"
            if [[ -f "$SWEEP_DIR/$point/preprocess.json" ]]; then
                echo "[skip] $point"
                continue
            fi
            if [[ "$backend" == "nvdec" ]]; then
                gpus="$GPU_MODEL,$GPU_NVDEC"
                nvdec_dev=1
            else
                gpus="$GPU_MODEL"
                nvdec_dev=0
            fi
            echo "[run ] $point"
            start=$SECONDS
            RUN_DIR="$SWEEP_DIR/$point" \
                GPU_PREPROC="$gpus" VLLM_NVDEC_DEVICE="$nvdec_dev" \
                VIDEO_SIZE="$size" VIDEO_SECONDS="$secs" NUM_PROMPTS="$NUM_PROMPTS" \
                bash "$SCRIPT_DIR/preprocess.sh" --benchmark video \
                --video-backend "$backend" --video-pool "$VIDEO_POOL" \
                > "$SWEEP_DIR/$point.out" 2>&1
            rc=$?
            echo "       rc=$rc, $((SECONDS - start)) s"
        done
    done
done

python3 - "$SWEEP_DIR" <<'EOF'
import csv, json, sys
from pathlib import Path

sweep = Path(sys.argv[1])
stages = ["video_decode_ms", "apply_hf_processor_ms", "get_mm_hashes_ms",
          "preprocessor_total_ms", "encoder_forward_ms"]
rows = []
for f in sorted(sweep.glob("*/preprocess.json")):
    backend, size, secs = f.parent.name.split("_")
    d = json.loads(f.read_text())
    st = d["mm_processor_stats"]
    row = {"backend": backend, "size": size, "seconds": int(secs[:-1]),
           "prompt_tokens": round(d.get("mean_prompt_tokens", 0))}
    for s in stages:
        row[s] = round(st.get(s, {}).get("mean", float("nan")), 2)
    row["preprocess_per_req_ms"] = round(d["mean_preprocess_per_request_ms"], 2)
    rows.append(row)

px = lambda s: int(s.split("x")[0]) * int(s.split("x")[1])
rows.sort(key=lambda r: (r["backend"], px(r["size"]), r["seconds"]))
if rows:
    with open(sweep / "summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    cols = list(rows[0])
    print("\n" + "  ".join(f"{c:>12.12}" for c in cols))
    for r in rows:
        print("  ".join(f"{str(r[c]):>12.12}" for c in cols))
    print(f"\n-> {sweep / 'summary.csv'}")
failed = [p.stem for p in sweep.glob("*.out") if not (sweep / p.stem / "preprocess.json").exists()]
if failed:
    print("failed points (see <point>.out):", " ".join(sorted(failed)))
EOF
