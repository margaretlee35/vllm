# VTP Sweep Summary (20260419_220639)

- FAIL case=Ne1p1d_e0-1-2_p1_d2 method=none rate=n/a ipr=32
  - cause: unknown (see launcher log)
  - launcher: epdtest/logs/vtp/20260419_220639/Ne1p1d_e0-1-2_p1_d2/none/ipr32/launcher.log
  - serving_benchmark_result:

```text
[not found in launcher.log]
```

- FAIL case=Ne1p1d_e0-1-2_p1_d2 method=visionzip rate=0.5 ipr=32
  - cause: Port 19537 is already in use. Stop stale processes or change ports.
  - launcher: epdtest/logs/vtp/20260419_220639/Ne1p1d_e0-1-2_p1_d2/visionzip/r0p5/ipr32/launcher.log
  - serving_benchmark_result:

```text
[not found in launcher.log]
```

- OK case=Ne1p1d_e0-1-2_p1_d2 method=visionzip rate=0.9 ipr=32
  - launcher: epdtest/logs/vtp/20260419_220639/Ne1p1d_e0-1-2_p1_d2/visionzip/r0p9/ipr32/launcher.log
  - serving_benchmark_result:

```text
============ Serving Benchmark Result ============
Successful requests:                     297       
Failed requests:                         3         
Maximum request concurrency:             32        
Request rate configured (RPS):           8.00      
Benchmark duration (s):                  37.54     
Total input tokens:                      304128    
Total generated tokens:                  0         
Request throughput (req/s):              7.91      
Output token throughput (tok/s):         0.00      
Peak output token throughput (tok/s):    15.00     
Peak concurrent requests:                15.00     
Total token throughput (tok/s):          8101.28   
---------------Time to First Token----------------
Mean TTFT (ms):                          0.00      
Median TTFT (ms):                        0.00      
P99 TTFT (ms):                           0.00      
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          0.00      
Median TPOT (ms):                        0.00      
P99 TPOT (ms):                           0.00      
---------------Inter-token Latency----------------
Mean ITL (ms):                           0.00      
Median ITL (ms):                         0.00      
P99 ITL (ms):                            0.00      
==================================================
```

