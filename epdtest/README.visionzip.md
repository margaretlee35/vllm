# VisionZip vs NoPrune EPD Benchmark Results

This file compares all 8 runs:
- VisionZip batch: `20260406_020517` (`bash epdtest/run_all.sh visionzip`)
- NoPrune batch: `20260406_021822` (`bash epdtest/run_all.sh noprune`)
- topology/profile: `1e1pd / metrics`
- prompts per run: `500`
- dataset mode: `random-mm`
- benchmark throttle: `request_rate=32`, `max_concurrency=32`

## All 8 Runs

| Method | IPR | Success | Failed | Duration (s) | Req/s | Output tok/s | Total tok/s | Mean TTFT (ms) | P99 TTFT (ms) | Mean TPOT (ms) | Mean ITL (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| VisionZip | 1 | 500 | 0 | 68.53 | 7.30 | 933.89 | 8405.04 | 1233.62 | 3034.59 | 23.85 | 23.99 |
| VisionZip | 2 | 500 | 0 | 89.61 | 5.58 | 714.21 | 6427.89 | 2142.79 | 5300.45 | 27.64 | 28.75 |
| VisionZip | 4 | 500 | 0 | 133.87 | 3.74 | 468.52 | 4293.21 | 4691.60 | 10132.45 | 28.58 | 29.61 |
| VisionZip | 8 | 498 | 2 | 229.51 | 2.17 | 276.06 | 2497.96 | 10377.91 | 21159.72 | 33.14 | 34.98 |
| NoPrune | 1 | 500 | 0 | 82.20 | 6.08 | 778.55 | 7006.94 | 1760.44 | 3866.15 | 26.55 | 26.48 |
| NoPrune | 2 | 500 | 0 | 111.47 | 4.49 | 574.13 | 5167.21 | 1743.67 | 5586.33 | 42.10 | 43.85 |
| NoPrune | 4 | 500 | 0 | 202.23 | 2.47 | 316.47 | 2848.28 | 7900.09 | 14493.40 | 39.09 | 39.22 |
| NoPrune | 8 | 500 | 0 | 347.75 | 1.44 | 173.37 | 1645.68 | 14234.29 | 27265.63 | 59.14 | 58.85 |

## Head-to-Head By IPR

| IPR | Req/s Winner | Mean TTFT Winner | Mean TPOT Winner | Mean ITL Winner | Stability |
|---:|---|---|---|---|---|
| 1 | VisionZip (`7.30` vs `6.08`) | VisionZip (`1233.62` vs `1760.44`) | VisionZip (`23.85` vs `26.55`) | VisionZip (`23.99` vs `26.48`) | Both `0` failed |
| 2 | VisionZip (`5.58` vs `4.49`) | NoPrune (`1743.67` vs `2142.79`) | VisionZip (`27.64` vs `42.10`) | VisionZip (`28.75` vs `43.85`) | Both `0` failed |
| 4 | VisionZip (`3.74` vs `2.47`) | VisionZip (`4691.60` vs `7900.09`) | VisionZip (`28.58` vs `39.09`) | VisionZip (`29.61` vs `39.22`) | Both `0` failed |
| 8 | VisionZip (`2.17` vs `1.44`) | VisionZip (`10377.91` vs `14234.29`) | VisionZip (`33.14` vs `59.14`) | VisionZip (`34.98` vs `58.85`) | VisionZip had `2` failures |

## Takeaways

1. VisionZip is faster in throughput (`Req/s`, `Output tok/s`) at all `IPR=1/2/4/8`.
2. VisionZip improves most latency metrics (TPOT/ITL, and TTFT at `IPR=1/4/8`).
3. `IPR=2` is the one exception where NoPrune has lower mean TTFT.
4. VisionZip at `IPR=8` needs stabilization (`2` failed requests), even though it is faster than NoPrune.

## LMMS-Eval (MMMU Val, limit=300)

Runs:
- `bash epdtest/lmms_eval.sh noprune`
- `bash epdtest/lmms_eval.sh visionzip`

| Method | MMMU Acc | Input Tokens | Output Tokens | Total Tokens | API Calls | Eval Elapsed (s) | Avg Speed (tok/s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| NoPrune | 0.48333 | 285,494 | 834 | 286,328 | 300 | 650.4043 | 1.2823 |
| VisionZip | 0.47667 | 159,413 | 847 | 160,260 | 300 | 451.8503 | 1.8745 |

### LMMS Comparison

| Metric | VisionZip vs NoPrune |
|---|---:|
| MMMU Acc | -0.00666 (absolute, -1.38% relative) |
| Input Tokens | -126,081 (-44.16%) |
| Total Tokens | -126,068 (-44.03%) |
| Eval Elapsed Time | -198.5540 s (-30.53%) |
| Avg Generation Speed | +0.5922 tok/s (+46.18%) |

Notes:
1. VisionZip gives a large efficiency gain (much fewer input tokens and faster eval), with a small accuracy drop on this `limit=300` slice.
2. The LMMS warning still applies: `--limit` is for testing and should not be treated as final benchmark accuracy.
