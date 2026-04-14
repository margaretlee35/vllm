# Lonestar three A100 Compare Results

## (0) Setup

### Command

```bash
bash epdtest/sweep_compare.sh
```
```bash
bash epdtest/lmms_compare.sh
```

### Fixed runtime settings

```text
BENCH_REQUEST_RATE=8
BENCH_MAX_CONCURRENCY=32
```

## (1) Sweep Comparison Table (`randommm`, `IMAGES_PER_REQUEST=1`)

| Metric | `1e1p1d_e0_p1_d2` | `1e1pNd_e0_p1_d0-2` | `1e1pNd_e0_p1_d1-2` | `1e1pNd_e0_p1_d0-1-2` | `1e1pNd_d_preempt_e0_p1_d0-1-2` | `Ne1p1d_e0-1_p1_d2` | `Ne1p1d_e0-2_p1_d2` | `Ne1p1d_e0-1-2_p1_d2` | `Ne1p1d_pd_preempt_e0-1-2_p1_d2` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Successful / Failed |  |  |  |  |  |  |  |  |  |
| Duration (s) |  |  |  |  |  |  |  |  |  |
| Request throughput (req/s) |  |  |  |  |  |  |  |  |  |
| Output throughput (tok/s) |  |  |  |  |  |  |  |  |  |
| Peak output throughput (tok/s) |  |  |  |  |  |  |  |  |  |
| Peak concurrent requests |  |  |  |  |  |  |  |  |  |
| Total throughput (tok/s) |  |  |  |  |  |  |  |  |  |
| Mean TTFT (ms) |  |  |  |  |  |  |  |  |  |
| Median TTFT (ms) |  |  |  |  |  |  |  |  |  |
| P99 TTFT (ms) |  |  |  |  |  |  |  |  |  |
| Mean TPOT (ms) |  |  |  |  |  |  |  |  |  |
| Median TPOT (ms) |  |  |  |  |  |  |  |  |  |
| P99 TPOT (ms) |  |  |  |  |  |  |  |  |  |
| Mean ITL (ms) |  |  |  |  |  |  |  |  |  |
| Median ITL (ms) |  |  |  |  |  |  |  |  |  |
| P99 ITL (ms) |  |  |  |  |  |  |  |  |  |

### Summary

Fill this section after collecting sweep results.

## (2) LMMS Comparison Table (`mmmu_val`, `limit=300`, `IMAGES_PER_REQUEST=1`)

| Metric | `1e1p1d_e0_p1_d2` | `1e1pNd_e0_p1_d0-2` | `1e1pNd_e0_p1_d1-2` | `1e1pNd_e0_p1_d0-1-2` | `1e1pNd_d_preempt_e0_p1_d0-1-2` | `Ne1p1d_e0-1_p1_d2` | `Ne1p1d_e0-2_p1_d2` | `Ne1p1d_e0-1-2_p1_d2` | `Ne1p1d_pd_preempt_e0-1-2_p1_d2` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `mmmu_acc` |  |  |  |  |  |  |  |  |  |
| `total_gen_tokens` |  |  |  |  |  |  |  |  |  |
| `total_elapsed_time (s)` |  |  |  |  |  |  |  |  |  |
| `avg_speed (tokens/s)` |  |  |  |  |  |  |  |  |  |

### Summary

Fill this section after collecting LMMS results.
