# No-Prune EPD Benchmark Results

This file organizes results from:
- `bash epdtest/vt_prune.sh noprune`
- topology/profile: `1e1pd / metrics`
- prompts per run: `500`
- dataset mode: `random-mm`

## Summary Table (Transposed)

| Metric | IPR=1 | IPR=2 | IPR=4 | IPR=8 |
|---|---:|---:|---:|---:|
| Success | 500 | 500 | 500 | 499 |
| Failed | 0 | 0 | 0 | 1 |
| Duration (s) | 75.22 | 100.05 | 92.94 | 153.18 |
| Req/s | 6.65 | 5.00 | 5.38 | 3.26 |
| Output tok/s | 813.38 | 639.70 | 283.71 | 114.48 |
| Total tok/s | 7619.92 | 5757.34 | 5792.76 | 3450.35 |
| Mean TTFT (ms) | 31572.61 | 40440.04 | 18059.81 | 23930.49 |
| P99 TTFT (ms) | 73215.49 | 96687.37 | 89433.89 | 144287.01 |
| Mean TPOT (ms) | 19.07 | 34.38 | 57.47 | 105.34 |
| P99 TPOT (ms) | 48.13 | 60.46 | 109.75 | 186.03 |
| Mean ITL (ms) | 22.11 | 37.40 | 62.46 | 133.29 |
| P99 ITL (ms) | 256.43 | 879.78 | 1770.92 | 2676.47 |

## Key Trends

1. Throughput degrades at higher `IMAGES_PER_REQ`, especially at `IPR=8`.
- Req/s: `6.65 -> 5.00 -> 5.38 -> 3.26`
- Output tok/s: `813.38 -> 639.70 -> 283.71 -> 114.48`

2. Token-level latency rises with more images.
- Mean TPOT: `19.07ms -> 34.38ms -> 57.47ms -> 105.34ms`
- Mean ITL: `22.11ms -> 37.40ms -> 62.46ms -> 133.29ms`

3. Stability issue appears at `IPR=8`.
- `1/500` request failed.
- Failure signature:
  - `json.decoder.JSONDecodeError: Expecting value: line 1 column 1 (char 0)`

## Notes

1. `IPR=1/2/4` completed with `0` failures.
2. `Median TTFT (ms)` is reported as `0.00` at `IPR=4` and `IPR=8`, which looks anomalous and should be treated cautiously.