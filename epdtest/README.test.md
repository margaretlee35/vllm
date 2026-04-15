# Compare Run Guide

This file explains the comparison entrypoints:
- `epdtest/sweep_compare.sh`: serving benchmark comparison (throughput/latency)
- `epdtest/lmms_compare.sh`: LMMS accuracy + generation throughput comparison
- `epdtest/vtp_compare.sh`: visual-token-pruning sweep on `1e1p1d`
- Both support visual-token pruning method selection through `VISUAL_TOKEN_PRUNING_METHOD`

## Cases Compared

Both scripts run the same case list:
- `1e1p1d_e0_p1_d2`
- `1e1pNd_e0_p1_d0-2`
- `1e1pNd_d_preempt_e0_p1_d0-2`
- `Ne1p1d_e0-1_p1_d2`
- `Ne1p1d_e0-2_p1_d2`
- `Ne1p1d_e0-1-2_p1_d2`
- `Ne1p1d_pd_preempt_e0-1-2_p1_d2`
- `Ne1pNd_e0-1_p1_d0-2`
- `Ne1pNd_pd_preempt_e0-1_p1_d0-2`

## Visual-Token Pruning Methods

Supported methods:
- `none`
- `visionzip`
- `cdpruner`

Set method for either compare script by exporting:

```bash
VISUAL_TOKEN_PRUNING_METHOD=none
```

or inline:

```bash
VISUAL_TOKEN_PRUNING_METHOD=visionzip bash epdtest/sweep_compare.sh
```

## 1) Run Sweep Comparison

From repo root:

```bash
bash epdtest/sweep_compare.sh
```

Run one pruning method:

```bash
VISUAL_TOKEN_PRUNING_METHOD=none bash epdtest/sweep_compare.sh
VISUAL_TOKEN_PRUNING_METHOD=visionzip bash epdtest/sweep_compare.sh
VISUAL_TOKEN_PRUNING_METHOD=cdpruner bash epdtest/sweep_compare.sh
```

Run all three methods:

```bash
for m in none visionzip cdpruner; do
  RUN_STAMP="$(date +%Y%m%d_%H%M%S)_${m}" \
  VISUAL_TOKEN_PRUNING_METHOD="$m" \
  bash epdtest/sweep_compare.sh
done
```

Common overrides:

```bash
TIMEOUT_SECONDS=600 BENCH_REQUEST_RATE=8 BENCH_MAX_CONCURRENCY=32 \
IMAGES_PER_REQ_LIST="1 2 4 8" \
bash epdtest/sweep_compare.sh
```

Main knobs:
- `TIMEOUT_SECONDS` (default `600`)
- `BENCH_REQUEST_RATE` (default `8`)
- `BENCH_MAX_CONCURRENCY` (default `32`)
- `BENCHMARK` (default `randommm`)
- `IMAGES_PER_REQ_LIST` (default `1`)
- `MODEL` (default `Qwen/Qwen2.5-VL-3B-Instruct`)

## 2) Run LMMS Comparison

From repo root:

```bash
bash epdtest/lmms_compare.sh
```

Run one pruning method:

```bash
VISUAL_TOKEN_PRUNING_METHOD=none bash epdtest/lmms_compare.sh
VISUAL_TOKEN_PRUNING_METHOD=visionzip bash epdtest/lmms_compare.sh
VISUAL_TOKEN_PRUNING_METHOD=cdpruner bash epdtest/lmms_compare.sh
```

Run all three methods:

```bash
for m in none visionzip cdpruner; do
  RUN_STAMP="$(date +%Y%m%d_%H%M%S)_${m}" \
  VISUAL_TOKEN_PRUNING_METHOD="$m" \
  bash epdtest/lmms_compare.sh
done
```

Common overrides:

```bash
TIMEOUT_SECONDS=600 BENCH_REQUEST_RATE=8 BENCH_MAX_CONCURRENCY=32 \
LMMS_TASKS=mmmu_val LMMS_LIMIT=300 LMMS_BATCH_SIZE=1 \
IMAGES_PER_REQ=1 \
bash epdtest/lmms_compare.sh
```

Main knobs:
- Serving: `TIMEOUT_SECONDS`, `BENCH_REQUEST_RATE`, `BENCH_MAX_CONCURRENCY`, `RUN_BENCHMARK`, `IMAGES_PER_REQ`
- LMMS: `LMMS_TASKS` (default `mmmu_val`), `LMMS_LIMIT` (default `300`), `LMMS_BATCH_SIZE` (default `1`)
- Model/proxy: `MODEL`, `PROXY_PORT`

## 3) Run VTP Comparison

From repo root:

```bash
bash epdtest/vtp_compare.sh
```

Main knobs:
- `VTP_METHODS` (default `none visionzip`)
- `VTP_RATES` (default `0.5 0.7 0.9`, only used for non-`none` methods)
- `IMAGES_PER_REQ_LIST` (default `1 2 4 8`)
- `TIMEOUT_SECONDS`, `BENCH_REQUEST_RATE`, `BENCH_MAX_CONCURRENCY`

## Outputs

- Sweep logs: `epdtest/logs/sweep/<timestamp>/compare/...`
- LMMS logs/results: `epdtest/logs/lmms/<timestamp>_compare/...`
- LMMS table summary: `epdtest/logs/lmms/<timestamp>_compare/lmms_tables_summary.md`
- VTP logs: `epdtest/logs/vtp/<timestamp>/vtp_sweep/...`
- Slurm logs (if used): `epdtest/slurm_logs/`
