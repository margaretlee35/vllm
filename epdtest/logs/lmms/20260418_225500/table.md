# LMMS Comparison Table (20260418_225500)

Source: `/workspace/sylee/git_repos/vllm/epdtest/logs/lmms/20260418_225500/lmms_tables_summary.md`

| metric | 1e1p1d_e0_p1_d2 · none · r=- · ipr=1 | 1e1p1d_e0_p1_d2 · visionzip · r=0.5 · ipr=1 | 1e1p1d_e0_p1_d2 · visionzip · r=0.9 · ipr=1 |
|---|---|---|---|
| case | 1e1p1d_e0_p1_d2 | 1e1p1d_e0_p1_d2 | 1e1p1d_e0_p1_d2 |
| method | none | visionzip | visionzip |
| rate | - | 0.5 | 0.9 |
| ipr | 1 | 1 | 1 |
| mmmu_acc | 0.4667 | 0.4411 | 0.4156 |
| total_gen_tokens | 2637.0000 | 2775.0000 | 3053.0000 |
| total_elapsed_time_s | 1819.7647 | 1389.6699 | 1205.4781 |
| avg_speed_tok_s | 1.4491 | 1.9969 | 2.5326 |
