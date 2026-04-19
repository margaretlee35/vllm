# VTP Sweep Summary (20260418_224559)

- OK case=1e1p1d_e0_p1_d2 method=none rate=n/a ipr=1
  - launcher: epdtest/logs/vtp/20260418_224559/1e1p1d_e0_p1_d2/none/ipr1/launcher.log
  - serving_benchmark_result:

```text
============ Serving Benchmark Result ============
Successful requests:                     300       
Failed requests:                         0         
Maximum request concurrency:             32        
Request rate configured (RPS):           32.00     
Benchmark duration (s):                  36.84     
Total input tokens:                      307200    
Total generated tokens:                  38400     
Request throughput (req/s):              8.14      
Output token throughput (tok/s):         1042.22   
Peak output token throughput (tok/s):    1542.00   
Peak concurrent requests:                45.00     
Total token throughput (tok/s):          9379.99   
---------------Time to First Token----------------
Mean TTFT (ms):                          1560.53   
Median TTFT (ms):                        1286.06   
P99 TTFT (ms):                           4093.00   
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          16.90     
Median TPOT (ms):                        17.33     
P99 TPOT (ms):                           23.01     
---------------Inter-token Latency----------------
Mean ITL (ms):                           17.73     
Median ITL (ms):                         11.05     
P99 ITL (ms):                            173.23    
==================================================
```

- OK case=1e1p1d_e0_p1_d2 method=visionzip rate=0.5 ipr=1
  - launcher: epdtest/logs/vtp/20260418_224559/1e1p1d_e0_p1_d2/visionzip/r0p5/ipr1/launcher.log
  - serving_benchmark_result:

```text
============ Serving Benchmark Result ============
Successful requests:                     300       
Failed requests:                         0         
Maximum request concurrency:             32        
Request rate configured (RPS):           32.00     
Benchmark duration (s):                  34.55     
Total input tokens:                      307200    
Total generated tokens:                  38400     
Request throughput (req/s):              8.68      
Output token throughput (tok/s):         1111.39   
Peak output token throughput (tok/s):    1820.00   
Peak concurrent requests:                54.00     
Total token throughput (tok/s):          10002.48  
---------------Time to First Token----------------
Mean TTFT (ms):                          1200.61   
Median TTFT (ms):                        922.11    
P99 TTFT (ms):                           3999.16   
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          18.25     
Median TPOT (ms):                        18.39     
P99 TPOT (ms):                           24.87     
---------------Inter-token Latency----------------
Mean ITL (ms):                           19.07     
Median ITL (ms):                         11.42     
P99 ITL (ms):                            184.89    
==================================================
```

- OK case=1e1p1d_e0_p1_d2 method=visionzip rate=0.9 ipr=1
  - launcher: epdtest/logs/vtp/20260418_224559/1e1p1d_e0_p1_d2/visionzip/r0p9/ipr1/launcher.log
  - serving_benchmark_result:

```text
============ Serving Benchmark Result ============
Successful requests:                     300       
Failed requests:                         0         
Maximum request concurrency:             32        
Request rate configured (RPS):           32.00     
Benchmark duration (s):                  31.07     
Total input tokens:                      307200    
Total generated tokens:                  38400     
Request throughput (req/s):              9.66      
Output token throughput (tok/s):         1235.90   
Peak output token throughput (tok/s):    2255.00   
Peak concurrent requests:                48.00     
Total token throughput (tok/s):          11123.06  
---------------Time to First Token----------------
Mean TTFT (ms):                          764.27    
Median TTFT (ms):                        529.21    
P99 TTFT (ms):                           2452.35   
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          18.93     
Median TPOT (ms):                        19.81     
P99 TPOT (ms):                           24.80     
---------------Inter-token Latency----------------
Mean ITL (ms):                           19.70     
Median ITL (ms):                         11.36     
P99 ITL (ms):                            155.09    
==================================================
```

