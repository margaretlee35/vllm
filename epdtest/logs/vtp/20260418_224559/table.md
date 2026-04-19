# VTP Full Comparison Table (20260418_224559)

Source: `/workspace/sylee/git_repos/vllm/epdtest/logs/vtp/20260418_224559/vtp_sweep_summary.md`

| metric | 1e1p1d_e0_p1_d0 · none · r=n/a · ipr=1 | 1e1p1d_e0_p1_d0 · visionzip · r=0.5 · ipr=1 | 1e1p1d_e0_p1_d0 · visionzip · r=0.9 · ipr=1 |
|---|---|---|---|
| case | 1e1p1d_e0_p1_d0 | 1e1p1d_e0_p1_d0 | 1e1p1d_e0_p1_d0 |
| method | none | visionzip | visionzip |
| rate | n/a | 0.5 | 0.9 |
| ipr | 1 | 1 | 1 |
| status | OK | OK | OK |
| ok_reqs | 300 | 300 | 300 |
| failed_reqs | 0 | 0 | 0 |
| max_conc | 32 | 32 | 32 |
| cfg_rps | 32.00 | 32.00 | 32.00 |
| duration_s | 36.84 | 34.55 | 31.07 |
| in_tok | 307200 | 307200 | 307200 |
| out_tok | 38400 | 38400 | 38400 |
| req_s | 8.14 | 8.68 | 9.66 |
| out_tok_s | 1042.22 | 1111.39 | 1235.90 |
| peak_out_tok_s | 1542.00 | 1820.00 | 2255.00 |
| peak_conc | 45.00 | 54.00 | 48.00 |
| total_tok_s | 9379.99 | 10002.48 | 11123.06 |
| ttft_mean | 1560.53 | 1200.61 | 764.27 |
| ttft_med | 1286.06 | 922.11 | 529.21 |
| ttft_p99 | 4093.00 | 3999.16 | 2452.35 |
| tpot_mean | 16.90 | 18.25 | 18.93 |
| tpot_med | 17.33 | 18.39 | 19.81 |
| tpot_p99 | 23.01 | 24.87 | 24.80 |
| itl_mean | 17.73 | 19.07 | 19.70 |
| itl_med | 11.05 | 11.42 | 11.36 |
| itl_p99 | 173.23 | 184.89 | 155.09 |
