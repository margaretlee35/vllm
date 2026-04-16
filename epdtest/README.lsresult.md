# Lonestar A100 ×3 Comparisons

## (0) Setup

### Commands

```bash
VTP_METHODS=none bash epdtest/vtp_compare.sh
bash epdtest/lmms_compare.sh
```

### Fixed runtime settings

```text
BENCH_REQUEST_RATE=8 BENCH_MAX_CONCURRENCY=32
```

### Caution
P and D are always disaggregated since they both need high HBM memory usages.


## (1) Sweep Comparison Table (`randommm`, `IMAGES_PER_REQUEST=1`)

| Metric | `1e1p1d_e0_p1_d2` | `1e1pNd_e0_p1_d0-2` | `1e1pNd_d_preempt_e0_p1_d0-2` | `Ne1p1d_e0-1_p1_d2` | `Ne1p1d_e0-2_p1_d2` | `Ne1p1d_e0-1-2_p1_d2` | `Ne1p1d_pd_preempt_e0-1-2_p1_d2` | `Ne1pNd_pd_preempt_e0-1_p1_d0-2` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Successful / Failed |  |  |  |  |  |  |  |  |
| Duration (s) |  |  |  |  |  |  |  |  |
| Request throughput (req/s) |  |  |  |  |  |  |  |  |
| Output throughput (tok/s) |  |  |  |  |  |  |  |  |
| Peak output throughput (tok/s) |  |  |  |  |  |  |  |  |
| Peak concurrent requests |  |  |  |  |  |  |  |  |
| Total throughput (tok/s) |  |  |  |  |  |  |  |  |
| Mean TTFT (ms) |  |  |  |  |  |  |  |  |
| Median TTFT (ms) |  |  |  |  |  |  |  |  |
| P99 TTFT (ms) |  |  |  |  |  |  |  |  |
| Mean TPOT (ms) |  |  |  |  |  |  |  |  |
| Median TPOT (ms) |  |  |  |  |  |  |  |  |
| P99 TPOT (ms) |  |  |  |  |  |  |  |  |
| Mean ITL (ms) |  |  |  |  |  |  |  |  |
| Median ITL (ms) |  |  |  |  |  |  |  |  |
| P99 ITL (ms) |  |  |  |  |  |  |  |  |

### Summary

Fill this section after collecting sweep results.

## (2) LMMS Comparison Table (`mmmu_val`, `limit=300`, `IMAGES_PER_REQUEST=1`)

| Metric | `1e1p1d_e0_p1_d2` | `1e1pNd_e0_p1_d0-2` | `1e1pNd_d_preempt_e0_p1_d0-2` | `Ne1p1d_e0-1_p1_d2` | `Ne1p1d_e0-2_p1_d2` | `Ne1p1d_e0-1-2_p1_d2` | `Ne1p1d_pd_preempt_e0-1-2_p1_d2` | `Ne1pNd_pd_preempt_e0-1_p1_d0-2` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `mmmu_acc` | 0.4767 | 0.4767 |  |  |  |  |  |  |
| `total_gen_tokens` | 836.0000 | 837.0000 |  |  |  |  |  |  |
| `total_elapsed_time (s)` | 427.1978 | 355.4162 |  |  |  |  |  |  |
| `avg_speed (tokens/s)` | 1.9569 | 2.3550 |  |  |  |  |  |  |

### Summary

Partial update (2 / 8 cases filled):
- `1e1p1d_e0_p1_d2`
- `1e1pNd_e0_p1_d0-2`

Current LMMS run root:

```text
epdtest/lmms_eval/20260414_004218_compare
```
