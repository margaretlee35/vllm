# LS6 Sweep Compare Results (randommm)

## 1) Sweep Setup

### Command

```bash
bash epdtest/sweep_compare.sh
```

### Fixed runtime settings

```text
TIMEOUT_SECONDS=600
BENCH_REQUEST_RATE=8
BENCH_MAX_CONCURRENCY=32
--benchmark randommm
```

### This report

```text
IMAGES_PER_REQUEST=1
```

## 2) Detailed Results (`IMAGES_PER_REQUEST=1`)

### case=1e1p1d_e0_p1_d2

#### `nvidia-smi` process snapshot

```text
+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|    0   N/A  N/A         2642070      C   VLLM::EngineCore                       9750MiB |
|    1   N/A  N/A         2642063      C   VLLM::EngineCore                      28592MiB |
|    2   N/A  N/A         2642077      C   VLLM::EngineCore                      27952MiB |
+-----------------------------------------------------------------------------------------+
```

#### Benchmark output

```text
============ Serving Benchmark Result ============
Successful requests:                     300
Failed requests:                         0
Maximum request concurrency:             32
Request rate configured (RPS):           8.00
Benchmark duration (s):                  51.38
Total input tokens:                      307200
Total generated tokens:                  38400
Request throughput (req/s):              5.84
Output token throughput (tok/s):         747.36
Peak output token throughput (tok/s):    1527.00
Peak concurrent requests:                53.00
Total token throughput (tok/s):          6726.25
---------------Time to First Token----------------
Mean TTFT (ms):                          1028.63
Median TTFT (ms):                        886.34
P99 TTFT (ms):                           3255.65
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          31.78
Median TPOT (ms):                        32.82
P99 TPOT (ms):                           37.70
---------------Inter-token Latency----------------
Mean ITL (ms):                           32.55
Median ITL (ms):                         21.13
P99 ITL (ms):                            246.36
==================================================
```

### case=1e1pNd_e0_p1_d0-2

#### `nvidia-smi` process snapshot

```text
+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|    0   N/A  N/A         2647388      C   VLLM::EngineCore                       9416MiB |
|    0   N/A  N/A         2647402      C   VLLM::EngineCore                      19378MiB |
|    1   N/A  N/A         2647409      C   VLLM::EngineCore                      28676MiB |
|    2   N/A  N/A         2647395      C   VLLM::EngineCore                      28162MiB |
+-----------------------------------------------------------------------------------------+
```

#### Benchmark output

```text
============ Serving Benchmark Result ============
Successful requests:                     300
Failed requests:                         0
Maximum request concurrency:             32
Request rate configured (RPS):           8.00
Benchmark duration (s):                  61.09
Total input tokens:                      307200
Total generated tokens:                  38400
Request throughput (req/s):              4.91
Output token throughput (tok/s):         628.54
Peak output token throughput (tok/s):    1509.00
Peak concurrent requests:                49.00
Total token throughput (tok/s):          5656.87
---------------Time to First Token----------------
Mean TTFT (ms):                          1490.13
Median TTFT (ms):                        1349.73
P99 TTFT (ms):                           4383.32
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          36.12
Median TPOT (ms):                        34.97
P99 TPOT (ms):                           71.41
---------------Inter-token Latency----------------
Mean ITL (ms):                           36.53
Median ITL (ms):                         20.54
P99 ITL (ms):                            502.68
==================================================
```

### case=Ne1p1d_e0-2_p1_d2

#### `nvidia-smi` process snapshot

```text
+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|    0   N/A  N/A         2653390      C   VLLM::EngineCore                      10172MiB |
|    1   N/A  N/A         2653404      C   VLLM::EngineCore                      27930MiB |
|    2   N/A  N/A         2653397      C   VLLM::EngineCore                      23566MiB |
|    2   N/A  N/A         2653414      C   VLLM::EngineCore                       8824MiB |
+-----------------------------------------------------------------------------------------+
```

#### Benchmark output

```text
============ Serving Benchmark Result ============
Successful requests:                     300
Failed requests:                         0
Maximum request concurrency:             32
Request rate configured (RPS):           8.00
Benchmark duration (s):                  67.47
Total input tokens:                      307200
Total generated tokens:                  38400
Request throughput (req/s):              4.45
Output token throughput (tok/s):         569.10
Peak output token throughput (tok/s):    1600.00
Peak concurrent requests:                50.00
Total token throughput (tok/s):          5121.91
---------------Time to First Token----------------
Mean TTFT (ms):                          1390.54
Median TTFT (ms):                        1120.89
P99 TTFT (ms):                           4215.69
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          42.13
Median TPOT (ms):                        40.61
P99 TPOT (ms):                           64.61
---------------Inter-token Latency----------------
Mean ITL (ms):                           42.21
Median ITL (ms):                         21.04
P99 ITL (ms):                            525.30
==================================================
```

### case=Ne1p1d_e0-1_p1_d2

#### `nvidia-smi` process snapshot

```text
+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|    0   N/A  N/A         2659958      C   VLLM::EngineCore                       9242MiB |
|    1   N/A  N/A         2659933      C   VLLM::EngineCore                       8824MiB |
|    1   N/A  N/A         2659940      C   VLLM::EngineCore                      19428MiB |
|    2   N/A  N/A         2659947      C   VLLM::EngineCore                      30108MiB |
+-----------------------------------------------------------------------------------------+
```

#### Benchmark output

```text
============ Serving Benchmark Result ============
Successful requests:                     300
Failed requests:                         0
Maximum request concurrency:             32
Request rate configured (RPS):           8.00
Benchmark duration (s):                  69.43
Total input tokens:                      307200
Total generated tokens:                  38400
Request throughput (req/s):              4.32
Output token throughput (tok/s):         553.04
Peak output token throughput (tok/s):    1536.00
Peak concurrent requests:                64.00
Total token throughput (tok/s):          4977.33
---------------Time to First Token----------------
Mean TTFT (ms):                          2986.04
Median TTFT (ms):                        2618.60
P99 TTFT (ms):                           6318.63
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          30.81
Median TPOT (ms):                        27.74
P99 TPOT (ms):                           66.75
---------------Inter-token Latency----------------
Mean ITL (ms):                           30.62
Median ITL (ms):                         21.41
P99 ITL (ms):                            283.92
==================================================
```

### case=Ne1p1d_e0-1-2_p1_d2

#### `nvidia-smi` process snapshot

```text
+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|    0   N/A  N/A         2666756      C   VLLM::EngineCore                       9058MiB |
|    1   N/A  N/A         2666742      C   VLLM::EngineCore                       8824MiB |
|    1   N/A  N/A         2666763      C   VLLM::EngineCore                      18732MiB |
|    2   N/A  N/A         2666741      C   VLLM::EngineCore                      19494MiB |
|    2   N/A  N/A         2666749      C   VLLM::EngineCore                       8824MiB |
+-----------------------------------------------------------------------------------------+
```

#### Benchmark output

```text
============ Serving Benchmark Result ============
Successful requests:                     300
Failed requests:                         0
Maximum request concurrency:             32
Request rate configured (RPS):           8.00
Benchmark duration (s):                  39.01
Total input tokens:                      307200
Total generated tokens:                  38400
Request throughput (req/s):              7.69
Output token throughput (tok/s):         984.39
Peak output token throughput (tok/s):    1767.00
Peak concurrent requests:                29.00
Total token throughput (tok/s):          8859.51
---------------Time to First Token----------------
Mean TTFT (ms):                          348.60
Median TTFT (ms):                        240.98
P99 TTFT (ms):                           1389.54
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          7.95
Median TPOT (ms):                        1.17
P99 TPOT (ms):                           29.96
---------------Inter-token Latency----------------
Mean ITL (ms):                           8.57
Median ITL (ms):                         1.17
P99 ITL (ms):                            50.59
==================================================
```

### case=Ne1p1d_pd_preempt_e0-1-2_p1_d2

#### `nvidia-smi` process snapshot

```text
+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|    0   N/A  N/A         2671498      C   VLLM::EngineCore                       9060MiB |
|    1   N/A  N/A         2671512      C   VLLM::EngineCore                       8824MiB |
|    1   N/A  N/A         2671519      C   VLLM::EngineCore                      19840MiB |
|    2   N/A  N/A         2671491      C   VLLM::EngineCore                       8824MiB |
|    2   N/A  N/A         2671505      C   VLLM::EngineCore                      20058MiB |
+-----------------------------------------------------------------------------------------+
```

#### Benchmark output

```text
============ Serving Benchmark Result ============
Successful requests:                     300
Failed requests:                         0
Maximum request concurrency:             32
Request rate configured (RPS):           8.00
Benchmark duration (s):                  38.52
Total input tokens:                      307200
Total generated tokens:                  38400
Request throughput (req/s):              7.79
Output token throughput (tok/s):         996.78
Peak output token throughput (tok/s):    1772.00
Peak concurrent requests:                25.00
Total token throughput (tok/s):          8971.05
---------------Time to First Token----------------
Mean TTFT (ms):                          296.46
Median TTFT (ms):                        243.09
P99 TTFT (ms):                           1123.64
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          5.54
Median TPOT (ms):                        1.17
P99 TPOT (ms):                           27.83
---------------Inter-token Latency----------------
Mean ITL (ms):                           6.11
Median ITL (ms):                         1.17
P99 ITL (ms):                            26.01
==================================================
```

## 3) Comparison Table (`IMAGES_PER_REQUEST=1`)

| Metric | `1e1p1d_e0_p1_d2` | `1e1pNd_e0_p1_d0-2` | `Ne1p1d_e0-2_p1_d2` | `Ne1p1d_e0-1_p1_d2` | `Ne1p1d_e0-1-2_p1_d2` | `Ne1p1d_pd_preempt_e0-1-2_p1_d2` |
|---|---:|---:|---:|---:|---:|---:|
| Successful / Failed | 300 / 0 | 300 / 0 | 300 / 0 | 300 / 0 | 300 / 0 | 300 / 0 |
| Duration (s) | 51.38 | 61.09 | 67.47 | 69.43 | 39.01 | 38.52 |
| Request throughput (req/s) | 5.84 | 4.91 | 4.45 | 4.32 | 7.69 | 7.79 |
| Output throughput (tok/s) | 747.36 | 628.54 | 569.10 | 553.04 | 984.39 | 996.78 |
| Peak output throughput (tok/s) | 1527.00 | 1509.00 | 1600.00 | 1536.00 | 1767.00 | 1772.00 |
| Peak concurrent requests | 53.00 | 49.00 | 50.00 | 64.00 | 29.00 | 25.00 |
| Total throughput (tok/s) | 6726.25 | 5656.87 | 5121.91 | 4977.33 | 8859.51 | 8971.05 |
| Mean TTFT (ms) | 1028.63 | 1490.13 | 1390.54 | 2986.04 | 348.60 | 296.46 |
| Median TTFT (ms) | 886.34 | 1349.73 | 1120.89 | 2618.60 | 240.98 | 243.09 |
| P99 TTFT (ms) | 3255.65 | 4383.32 | 4215.69 | 6318.63 | 1389.54 | 1123.64 |
| Mean TPOT (ms) | 31.78 | 36.12 | 42.13 | 30.81 | 7.95 | 5.54 |
| Median TPOT (ms) | 32.82 | 34.97 | 40.61 | 27.74 | 1.17 | 1.17 |
| P99 TPOT (ms) | 37.70 | 71.41 | 64.61 | 66.75 | 29.96 | 27.83 |
| Mean ITL (ms) | 32.55 | 36.53 | 42.21 | 30.62 | 8.57 | 6.11 |
| Median ITL (ms) | 21.13 | 20.54 | 21.04 | 21.41 | 1.17 | 1.17 |
| P99 ITL (ms) | 246.36 | 502.68 | 525.30 | 283.92 | 50.59 | 26.01 |

## 4) Summary

All six runs succeeded with `300/300` successful requests and `0` failures.

For `IMAGES_PER_REQUEST=1` under this sweep, `Ne1p1d_pd_preempt_e0-1-2_p1_d2` is the strongest overall:
- highest total throughput (`8971.05 tok/s`)
- highest request throughput (`7.79 req/s`)
- best TTFT/TPOT/ITL tail and mean latency profile among the six.
