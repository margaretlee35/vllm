# LS6 Result Notes (1e1p1d, randommm)

## Run command

```bash
BENCH_REQUEST_RATE=4 BENCH_MAX_CONCURRENCY=20 GPU_E=0 GPU_P=1 GPU_D=2 bash epdtest/run.sh --topology 1e1p1d --benchmark randommm
```

## `nvidia-smi` process snapshot

```text
+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|    0   N/A  N/A         2575250      C   VLLM::EngineCore                       9208MiB |
|    1   N/A  N/A         2575257      C   VLLM::EngineCore                      27930MiB |
|    2   N/A  N/A         2575267      C   VLLM::EngineCore                      27608MiB |
+-----------------------------------------------------------------------------------------+
```

## Benchmark output

```text
============ Serving Benchmark Result ============
Successful requests:                     300
Failed requests:                         0
Maximum request concurrency:             20
Request rate configured (RPS):           4.00
Benchmark duration (s):                  77.84
Total input tokens:                      307200
Total generated tokens:                  38400
Request throughput (req/s):              3.85
Output token throughput (tok/s):         493.29
Peak output token throughput (tok/s):    872.00
Peak concurrent requests:                27.00
Total token throughput (tok/s):          4439.65
---------------Time to First Token----------------
Mean TTFT (ms):                          533.77
Median TTFT (ms):                        544.29
P99 TTFT (ms):                           1287.53
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          21.69
Median TPOT (ms):                        21.70
P99 TPOT (ms):                           22.81
---------------Inter-token Latency----------------
Mean ITL (ms):                           22.08
Median ITL (ms):                         20.92
P99 ITL (ms):                            50.77
==================================================
```

## Summary

This run is stable at the selected load (`BENCH_REQUEST_RATE=4`,
`BENCH_MAX_CONCURRENCY=20`) with `300/300` successful requests.
