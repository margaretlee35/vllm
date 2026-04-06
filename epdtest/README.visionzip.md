# VisionZip EPD Benchmark Results

This file organizes results from:
- `bash epdtest/vt_prune.sh visionzip`
- topology/profile: `1e1pd / metrics`
- prompts per run: `500`
- dataset mode: `random-mm`

## Summary Table (Transposed)

| Metric | IPR=1 | IPR=2 | IPR=4 | IPR=8 |
|---|---:|---:|---:|---:|
| Success | 500 | 500 | 500 | 496 |
| Failed | 0 | 0 | 0 | 4 |
| Duration (s) | 70.52 | 101.17 | 158.66 | 288.28 |
| Req/s | 7.09 | 4.94 | 3.15 | 1.72 |
| Output tok/s | 907.48 | 632.61 | 403.38 | 220.23 |
| Total tok/s | 8167.35 | 5693.47 | 3630.47 | 1982.07 |
| Mean TTFT (ms) | 30139.06 | 43200.21 | 68107.77 | 129010.52 |
| P99 TTFT (ms) | 67851.55 | 97375.56 | 154072.65 | 281579.66 |
| Mean TPOT (ms) | 45.18 | 72.28 | 128.15 | 253.72 |
| P99 TPOT (ms) | 84.39 | 117.48 | 203.64 | 371.50 |
| Mean ITL (ms) | 45.65 | 94.96 | 158.02 | 308.90 |
| P99 ITL (ms) | 1860.10 | 2310.02 | 2581.03 | 2832.84 |

## Key Trends

1. Throughput drops as `IMAGES_PER_REQ` increases:
- Req/s: `7.09 -> 4.94 -> 3.15 -> 1.72`
- Output tok/s: `907.48 -> 632.61 -> 403.38 -> 220.23`

2. Latency rises with more images:
- Mean TTFT: `30.1s -> 43.2s -> 68.1s -> 129.0s`
- Mean TPOT: `45.18ms -> 72.28ms -> 128.15ms -> 253.72ms`

3. Stability issue appears at `IPR=8`:
- `4/500` requests failed.
- Failure signature in benchmark client:
  - `json.decoder.JSONDecodeError: Expecting value: line 1 column 1 (char 0)`
  - Location: `vllm/benchmarks/lib/endpoint_request_func.py`, JSON parse path during chat completions streaming.

## Notes

1. `IPR=1/2/4` runs completed with `0` failed requests.
2. `IPR=8` still produced valid benchmark output but with partial request failures.
3. If comparing methods later (for example `cdpruner` or future `sparsevlm`), reuse this exact table format for direct side-by-side comparison.
