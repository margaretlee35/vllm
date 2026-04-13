# LS6 Result Notes (randommm)

## 1) Baseline: `1e1p1d`

### Run command

```bash
BENCH_REQUEST_RATE=4 BENCH_MAX_CONCURRENCY=20 GPU_E=0 GPU_P=1 GPU_D=2 bash epdtest/run.sh --topology 1e1p1d --benchmark randommm
```

### `nvidia-smi` process snapshot

```text
+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|    0   N/A  N/A         2620503      C   VLLM::EngineCore                       8824MiB |
|    1   N/A  N/A         2620516      C   VLLM::EngineCore                      27516MiB |
|    2   N/A  N/A         2620517      C   VLLM::EngineCore                      27516MiB |
+-----------------------------------------------------------------------------------------+
```

### Benchmark output

```text
============ Serving Benchmark Result ============
Successful requests:                     300
Failed requests:                         0
Maximum request concurrency:             20
Request rate configured (RPS):           4.00
Benchmark duration (s):                  77.86
Total input tokens:                      307200
Total generated tokens:                  38400
Request throughput (req/s):              3.85
Output token throughput (tok/s):         493.20
Peak output token throughput (tok/s):    996.00
Peak concurrent requests:                30.00
Total token throughput (tok/s):          4438.83
---------------Time to First Token----------------
Mean TTFT (ms):                          700.73
Median TTFT (ms):                        665.17
P99 TTFT (ms):                           2654.22
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          32.27
Median TPOT (ms):                        32.98
P99 TPOT (ms):                           38.00
---------------Inter-token Latency----------------
Mean ITL (ms):                           32.67
Median ITL (ms):                         20.47
P99 ITL (ms):                            226.04
==================================================
```

## 2) Multi-decode: `1e1pNd`

### Run command

```bash
TIMEOUT_SECONDS=600 BENCH_REQUEST_RATE=4 BENCH_MAX_CONCURRENCY=20 \
GPU_E=0 GPU_P=1 GPU_D=0,2 \
bash epdtest/run.sh --topology 1e1pNd --benchmark randommm
```

### `nvidia-smi` process snapshot

```text
+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|    0   N/A  N/A         2591816      C   VLLM::EngineCore                       8830MiB |
|    0   N/A  N/A         2591824      C   VLLM::EngineCore                      19738MiB |
|    1   N/A  N/A         2591817      C   VLLM::EngineCore                      27686MiB |
|    2   N/A  N/A         2591831      C   VLLM::EngineCore                      27774MiB |
+-----------------------------------------------------------------------------------------+
```

### Benchmark output

```text
============ Serving Benchmark Result ============
Successful requests:                     300
Failed requests:                         0
Maximum request concurrency:             20
Request rate configured (RPS):           4.00
Benchmark duration (s):                  80.32
Total input tokens:                      307200
Total generated tokens:                  38400
Request throughput (req/s):              3.74
Output token throughput (tok/s):         478.08
Peak output token throughput (tok/s):    896.00
Peak concurrent requests:                29.00
Total token throughput (tok/s):          4302.73
---------------Time to First Token----------------
Mean TTFT (ms):                          1048.10
Median TTFT (ms):                        899.84
P99 TTFT (ms):                           3259.40
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          30.34
Median TPOT (ms):                        29.22
P99 TPOT (ms):                           48.28
---------------Inter-token Latency----------------
Mean ITL (ms):                           30.46
Median ITL (ms):                         20.74
P99 ITL (ms):                            355.12
==================================================
```

## 3) Multi-encoder: `Ne1p1d`

### Run command

```bash
TIMEOUT_SECONDS=600 BENCH_REQUEST_RATE=4 BENCH_MAX_CONCURRENCY=20 \
GPU_E=0,2 GPU_P=1 GPU_D=2 \
bash epdtest/run.sh --topology Ne1p1d --benchmark randommm
```

### `nvidia-smi` process snapshot

```text
+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|    0   N/A  N/A         2599372      C   VLLM::EngineCore                       9888MiB |
|    1   N/A  N/A         2599379      C   VLLM::EngineCore                      27804MiB |
|    2   N/A  N/A         2599358      C   VLLM::EngineCore                      20322MiB |
|    2   N/A  N/A         2599365      C   VLLM::EngineCore                       8824MiB |
+-----------------------------------------------------------------------------------------+
```

### Benchmark output

```text
============ Serving Benchmark Result ============
Successful requests:                     300
Failed requests:                         0
Maximum request concurrency:             20
Request rate configured (RPS):           4.00
Benchmark duration (s):                  83.70
Total input tokens:                      307200
Total generated tokens:                  38400
Request throughput (req/s):              3.58
Output token throughput (tok/s):         458.78
Peak output token throughput (tok/s):    926.00
Peak concurrent requests:                31.00
Total token throughput (tok/s):          4129.03
---------------Time to First Token----------------
Mean TTFT (ms):                          808.94
Median TTFT (ms):                        689.64
P99 TTFT (ms):                           2383.77
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          34.70
Median TPOT (ms):                        34.42
P99 TPOT (ms):                           48.93
---------------Inter-token Latency----------------
Mean ITL (ms):                           34.97
Median ITL (ms):                         20.49
P99 ITL (ms):                            328.15
==================================================
```

## 4) Multi-encoder variant: `Ne1p1d` (`GPU_E=0,1`)

### Run command

```bash
TIMEOUT_SECONDS=600 BENCH_REQUEST_RATE=4 BENCH_MAX_CONCURRENCY=20 \
GPU_E=0,1 GPU_P=1 GPU_D=2 \
bash epdtest/run.sh --topology Ne1p1d --benchmark randommm
```

### `nvidia-smi` process snapshot

```text
+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|    0   N/A  N/A         2611488      C   VLLM::EngineCore                       9054MiB |
|    1   N/A  N/A         2611498      C   VLLM::EngineCore                      19262MiB |
|    1   N/A  N/A         2611505      C   VLLM::EngineCore                       8824MiB |
|    2   N/A  N/A         2611477      C   VLLM::EngineCore                      28886MiB |
+-----------------------------------------------------------------------------------------+
```

### Benchmark output

```text
============ Serving Benchmark Result ============
Successful requests:                     299
Failed requests:                         1
Maximum request concurrency:             20
Request rate configured (RPS):           4.00
Benchmark duration (s):                  80.96
Total input tokens:                      306176
Total generated tokens:                  38016
Request throughput (req/s):              3.69
Output token throughput (tok/s):         469.59
Peak output token throughput (tok/s):    980.00
Peak concurrent requests:                38.00
Total token throughput (tok/s):          4251.64
---------------Time to First Token----------------
Mean TTFT (ms):                          1286.27
Median TTFT (ms):                        1082.00
P99 TTFT (ms):                           3762.20
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          30.27
Median TPOT (ms):                        29.38
P99 TPOT (ms):                           47.35
---------------Inter-token Latency----------------
Mean ITL (ms):                           30.30
Median ITL (ms):                         20.56
P99 ITL (ms):                            270.89
==================================================
```

## 5) Multi-encoder variant: `Ne1p1d` (`GPU_E=0,1,2`)

### Run command

```bash
TIMEOUT_SECONDS=600 BENCH_REQUEST_RATE=4 BENCH_MAX_CONCURRENCY=20 \
GPU_E=0,1,2 GPU_P=1 GPU_D=2 \
bash epdtest/run.sh --topology Ne1p1d --benchmark randommm
```

### `nvidia-smi` process snapshot

```text
+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|    0   N/A  N/A         2627725      C   VLLM::EngineCore                       9058MiB |
|    1   N/A  N/A         2627718      C   VLLM::EngineCore                       8824MiB |
|    1   N/A  N/A         2627732      C   VLLM::EngineCore                      19274MiB |
|    2   N/A  N/A         2627739      C   VLLM::EngineCore                       8824MiB |
|    2   N/A  N/A         2627746      C   VLLM::EngineCore                      19352MiB |
+-----------------------------------------------------------------------------------------+
```

### Benchmark output

```text
============ Serving Benchmark Result ============
Successful requests:                     300
Failed requests:                         0
Maximum request concurrency:             20
Request rate configured (RPS):           4.00
Benchmark duration (s):                  75.31
Total input tokens:                      307200
Total generated tokens:                  38400
Request throughput (req/s):              3.98
Output token throughput (tok/s):         509.88
Peak output token throughput (tok/s):    1085.00
Peak concurrent requests:                14.00
Total token throughput (tok/s):          4588.91
---------------Time to First Token----------------
Mean TTFT (ms):                          225.43
Median TTFT (ms):                        222.93
P99 TTFT (ms):                           915.13
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          3.51
Median TPOT (ms):                        1.15
P99 TPOT (ms):                           21.17
---------------Inter-token Latency----------------
Mean ITL (ms):                           3.67
Median ITL (ms):                         1.15
P99 ITL (ms):                            21.83
==================================================
```

## 6) PD-preempt multi-encoder: `Ne1p1d_pd_preempt` (`GPU_E=0,1,2`)

### Run command

```bash
TIMEOUT_SECONDS=600 BENCH_REQUEST_RATE=4 BENCH_MAX_CONCURRENCY=20 \
GPU_E=0,1,2 GPU_P=1 GPU_D=2 \
bash epdtest/run.sh --topology Ne1p1d_pd_preempt --benchmark randommm
```

### `nvidia-smi` process snapshot

```text
+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|    0   N/A  N/A         2634536      C   VLLM::EngineCore                       9054MiB |
|    1   N/A  N/A         2634522      C   VLLM::EngineCore                      19668MiB |
|    1   N/A  N/A         2634543      C   VLLM::EngineCore                       8824MiB |
|    2   N/A  N/A         2634515      C   VLLM::EngineCore                      19930MiB |
|    2   N/A  N/A         2634529      C   VLLM::EngineCore                       8824MiB |
+-----------------------------------------------------------------------------------------+
```

### Benchmark output

```text
============ Serving Benchmark Result ============
Successful requests:                     300
Failed requests:                         0
Maximum request concurrency:             20
Request rate configured (RPS):           4.00
Benchmark duration (s):                  76.96
Total input tokens:                      307200
Total generated tokens:                  38400
Request throughput (req/s):              3.90
Output token throughput (tok/s):         498.98
Peak output token throughput (tok/s):    1114.00
Peak concurrent requests:                16.00
Total token throughput (tok/s):          4490.79
---------------Time to First Token----------------
Mean TTFT (ms):                          233.95
Median TTFT (ms):                        223.99
P99 TTFT (ms):                           912.38
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          7.28
Median TPOT (ms):                        1.15
P99 TPOT (ms):                           25.01
---------------Inter-token Latency----------------
Mean ITL (ms):                           7.46
Median ITL (ms):                         1.16
P99 ITL (ms):                            23.43
==================================================
```

## 7) Comparison

| Metric | `1e1p1d` | `1e1pNd` (`GPU_D=0,2`) | `Ne1p1d` (`GPU_E=0,2`) | `Ne1p1d` (`GPU_E=0,1`) | `Ne1p1d` (`GPU_E=0,1,2`) | `Ne1p1d_pd_preempt` (`GPU_E=0,1,2`) |
|---|---:|---:|---:|---:|---:|---:|
| Successful / Failed | 300 / 0 | 300 / 0 | 300 / 0 | 299 / 1 | 300 / 0 | 300 / 0 |
| Duration (s) | 77.86 | 80.32 | 83.70 | 80.96 | 75.31 | 76.96 |
| Request throughput (req/s) | 3.85 | 3.74 | 3.58 | 3.69 | 3.98 | 3.90 |
| Output throughput (tok/s) | 493.20 | 478.08 | 458.78 | 469.59 | 509.88 | 498.98 |
| Peak output throughput (tok/s) | 996.00 | 896.00 | 926.00 | 980.00 | 1085.00 | 1114.00 |
| Peak concurrent requests | 30.00 | 29.00 | 31.00 | 38.00 | 14.00 | 16.00 |
| Total throughput (tok/s) | 4438.83 | 4302.73 | 4129.03 | 4251.64 | 4588.91 | 4490.79 |
| Mean TTFT (ms) | 700.73 | 1048.10 | 808.94 | 1286.27 | 225.43 | 233.95 |
| Median TTFT (ms) | 665.17 | 899.84 | 689.64 | 1082.00 | 222.93 | 223.99 |
| P99 TTFT (ms) | 2654.22 | 3259.40 | 2383.77 | 3762.20 | 915.13 | 912.38 |
| Mean TPOT (ms) | 32.27 | 30.34 | 34.70 | 30.27 | 3.51 | 7.28 |
| Median TPOT (ms) | 32.98 | 29.22 | 34.42 | 29.38 | 1.15 | 1.15 |
| P99 TPOT (ms) | 38.00 | 48.28 | 48.93 | 47.35 | 21.17 | 25.01 |
| Mean ITL (ms) | 32.67 | 30.46 | 34.97 | 30.30 | 3.67 | 7.46 |
| Median ITL (ms) | 20.47 | 20.74 | 20.49 | 20.56 | 1.15 | 1.16 |
| P99 ITL (ms) | 226.04 | 355.12 | 328.15 | 270.89 | 21.83 | 23.43 |

## 8) Summary

Five runs are fully successful (`300/300`), and one run (`Ne1p1d` with `GPU_E=0,1`) has a single failed request (`299/300`).

For this setup and load, `Ne1p1d` with `GPU_E=0,1,2` is still the strongest overall result. `Ne1p1d_pd_preempt` is close, with excellent TTFT and very strong throughput, but slightly lower total throughput and slower mean TPOT/ITL.
