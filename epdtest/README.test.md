# Test Run Guide

This file explains how to run:
- `sweep.sh` for throughput/latency benchmark sweeps
- `eval_lmms.sh` for LMMS accuracy evaluation

No benchmark/eval result tables are stored in this file.

## Run `sweep.sh`

Run from repo root:

```bash
bash epdtest/sweep.sh visionzip
bash epdtest/sweep.sh none
bash epdtest/sweep.sh cdpruner
```

Useful overrides:

```bash
TOPOLOGY=1e1p1d BENCHMARK=randommm NUM_PROMPTS=300 bash epdtest/sweep.sh visionzip
TOPOLOGY=1e1pd BENCHMARK=simple NUM_PROMPTS=300 bash epdtest/sweep.sh none
```

Notes:
- Supported pruning modes: `visionzip`, `cdpruner`, `none`
- `sweep.sh` calls `run.sh` repeatedly (for example across `IMAGES_PER_REQ_LIST` in `randommm`)

## Run `eval_lmms.sh`

Single method:

```bash
bash epdtest/eval_lmms.sh none
bash epdtest/eval_lmms.sh visionzip
bash epdtest/eval_lmms.sh cdpruner
```

Multiple methods in one run:

```bash
METHODS="none visionzip cdpruner" bash epdtest/eval_lmms.sh
```

Common overrides:

```bash
METHODS="visionzip" VISUAL_TOKEN_PRUNING_RATE=0.4 bash epdtest/eval_lmms.sh
LMMS_TASKS=mmmu_val LMMS_LIMIT=300 bash epdtest/eval_lmms.sh visionzip
```

## Output Locations

- Sweep/serving logs: `epdtest/logs/`
- LMMS server/eval logs and outputs: `epdtest/lmms_eval/`
- Slurm logs (if used): `epdtest/slurm_logs/`
