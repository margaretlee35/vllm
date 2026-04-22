# VTP Comparison

## Benchmark
randommm

## Command

```bash
BENCH_REQUEST_RATE=8 \
BENCH_MAX_CONCURRENCY=32 \
IMAGES_PER_REQ=32 \
VTP_METHODS="none visionzip" \
VTP_RATES="0.5 0.9" \
bash epdtest/vtp_compare.sh
```

## Single Comparison Table

| Metric | `1e1p1d / none` | `1e1p1d / visionzip / 0.5` | `1e1p1d / visionzip / 0.9` | `Ne1p1d / none` | `Ne1p1d / visionzip / 0.5` | `Ne1p1d / visionzip / 0.9` |
|---|---|---|---|---|---|---|
| Topology | `1e1p1d` | `1e1p1d` | `1e1p1d` | `Ne1p1d` | `Ne1p1d` | `Ne1p1d` |
| VT Method | `none` | `visionzip` | `visionzip` | `none` | `visionzip` | `visionzip` |
| VT Rate | `n/a` | `0.5` | `0.9` | `n/a` | `0.5` | `0.9` |
| Successful | 300 | 300 | 300 | 300 | 300 | 300 |
| Failed | 0 | 0 | 0 | 0 | 0 | 0 |
| Req Rate (RPS) | 8.00 | 8.00 | 8.00 | 8.00 | 8.00 | 8.00 |
| Max Concurrency | 32 | 32 | 32 | 32 | 32 | 32 |
| Duration (s) | 346.64 | 383.23 | 376.70 | 388.08 | 304.51 | 298.61 |
| Generated Tokens | 38272 | 38272 | 38272 | 37888 | 37632 | 38144 |
| Req Throughput (req/s) | 0.87 | 0.78 | 0.80 | 0.77 | 0.99 | 1.00 |
| Output Throughput (tok/s) | 110.41 | 99.87 | 101.60 | 97.63 | 123.58 | 127.74 |
| Peak Output Throughput (tok/s) | 217.00 | 253.00 | 193.00 | 378.00 | 258.00 | 196.00 |
| Peak Concurrent Requests | 36.00 | 36.00 | 35.00 | 36.00 | 36.00 | 35.00 |
| Total Throughput (tok/s) | 996.64 | 901.47 | 917.10 | 889.21 | 1132.42 | 1156.51 |
| Mean TTFT (ms) | 31660.41 | 36080.21 | 35383.02 | 34627.43 | 26890.72 | 27190.55 |
| Median TTFT (ms) | 31527.60 | 36980.85 | 35938.59 | 15372.56 | 27226.97 | 27280.04 |
| P99 TTFT (ms) | 66948.99 | 58506.42 | 51533.96 | 100990.74 | 41502.21 | 48091.20 |
| Mean TPOT (ms) | 27.87 | 23.34 | 21.26 | 33.01 | 29.04 | 23.51 |
| Median TPOT (ms) | 27.64 | 23.03 | 21.49 | 32.21 | 28.88 | 23.63 |
| P99 TPOT (ms) | 39.56 | 31.28 | 27.46 | 51.83 | 42.19 | 31.43 |
| Mean ITL (ms) | 63.57 | 38.93 | 38.03 | 67.89 | 79.55 | 65.35 |
| Median ITL (ms) | 20.80 | 21.96 | 20.47 | 20.83 | 23.00 | 20.86 |
| P99 ITL (ms) | 697.09 | 616.44 | 611.91 | 698.60 | 770.71 | 704.47 |
