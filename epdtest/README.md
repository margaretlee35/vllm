# epdtest

This directory is the direct entrypoint for local EPD testing.

Use this instead of remembering which legacy wrapper lives under
`examples/online_serving/disaggregated_encoder/lovelace`.

`epdtest/run.sh` is the single-run entrypoint. It chooses the topology/benchmark
script, activates `.venv` when present, and runs one configuration once.

`epdtest/sweep.sh` is the outer sweep wrapper. It is the right entrypoint for
`randommm` image-count sweeps and prune-mode batches.

## Quick Start

Default local run:

```bash
bash epdtest/run.sh
```

Choose topology explicitly:

```bash
bash epdtest/run.sh --topology 1e1pd
bash epdtest/run.sh --topology 1e1p1d
bash epdtest/run.sh --topology 1ed1p
```

Enable the heavier `randommm` flow:

```bash
bash epdtest/run.sh --benchmark randommm
bash epdtest/run.sh --topology 1e1p1d --benchmark randommm
bash epdtest/run.sh --benchmark randommm --images-per-req 4
```

All existing environment overrides still work:

```bash
GPU_E=0 GPU_PD=1 NUM_PROMPTS=50 bash epdtest/run.sh
TOPOLOGY=1e1p1d BENCHMARK=randommm GPU_E=0 GPU_P=1 GPU_D=2 bash epdtest/run.sh
MODEL=Qwen/Qwen2.5-VL-3B-Instruct LOG_PATH=/tmp/epdtest_logs bash epdtest/run.sh
BENCHMARK=randommm IMAGES_PER_REQ=4 bash epdtest/run.sh
```

Pruning-specific defaults are now loaded from:

```bash
epdtest/visual_token_pruning_configs.json
```

Visual token pruning is not enabled by default anymore. To turn it on, pass an
explicit pruning method such as `--visual-token-pruning-method visionzip`.

Example:

```bash
bash epdtest/run.sh --visual-token-pruning-method visionzip
```

Use `sweep.sh` for outer sweeps:

```bash
bash epdtest/sweep.sh visionzip
IMAGES_PER_REQ_LIST="1 2 4 8" bash epdtest/sweep.sh cdpruner
BENCHMARK=simple bash epdtest/sweep.sh none
```

## Slurm

`epdtest/epd.slurm` is intentionally minimal now. It just exports the requested
topology/benchmark/prune mode and calls `epdtest/sweep.sh`.

Default batch run:

```bash
sbatch epdtest/epd.slurm
```

Run the lighter simple benchmark mode instead of `randommm`:

```bash
sbatch --export=ALL,BENCHMARK=simple epdtest/epd.slurm
```

Run the `randommm` benchmark mode explicitly:

```bash
sbatch --export=ALL,BENCHMARK=randommm epdtest/epd.slurm
```

Run VisionZip sweep:

```bash
sbatch --export=ALL,PRUNE_MODE=visionzip epdtest/epd.slurm
```

Run the `1e1p1d` topology:

```bash
sbatch --export=ALL,TOPOLOGY=1e1p1d,GPU_E=0,GPU_P=1,GPU_D=2 epdtest/epd.slurm
```

Run with the `1ed1p` topology:

```bash
sbatch --export=ALL,TOPOLOGY=1ed1p,GPU_E=0,GPU_P=1,GPU_D=2 epdtest/epd.slurm
```

Sweep a smaller set of image counts:

```bash
sbatch --export=ALL,IMAGES_PER_REQ_LIST="1 2 4" epdtest/epd.slurm
```

Override VisionZip settings via environment (optional):

```bash
sbatch --export=ALL,VISUAL_TOKEN_PRUNING_RATE=0.5,VISION_ZIP_DOMINANT_RATIO=0.75,VISION_ZIP_ATTENTION_LAYER=-2 epdtest/epd.slurm
```

The main variables to override are now consumed by `run.sh` or `sweep.sh`:

- `TOPOLOGY`: `1e1pd`, `1e1p1d`, or `1ed1p`
- `BENCHMARK`: `simple` or `randommm`
- `PRUNE_MODE`: `none`, `visionzip`, or `cdpruner` for `sweep.sh` / slurm
- `MODEL`: VLM to serve
- `LOG_PATH`: output directory for launcher-owned logs
- `IMAGES_PER_REQ`: single-run image count for `run.sh`
- `IMAGES_PER_REQ_LIST`: space-separated sweep values for `sweep.sh` when `BENCHMARK=randommm`
- `VISUAL_TOKEN_PRUNING_METHOD`: optional explicit pruning method
- `VISUAL_TOKEN_PRUNING_RATE`: optional override of JSON default
- `GPU_E`, `GPU_PD`, `GPU_P`, `GPU_D`: GPU assignment
- `NUM_PROMPTS`: benchmark request count
- `VISION_ZIP_DOMINANT_RATIO`, `VISION_ZIP_ATTENTION_LAYER`: optional VisionZip overrides

## Notes

- `epdtest/logs/` is still the home for logs and log analysis helpers.
- Each run now creates a timestamped subdirectory like
  `epdtest/logs/20260405_153000/`. For `randommm` sweeps, each item lands under
  a nested directory like `epdtest/logs/20260405_153000/ipr1/`. Inside those
  folders you will see files such as `encoder.log`, `prefill_decode.log`,
  `prefill.log`, `decode.log`, `proxy.log`, `sm.log`, and
  `target_script.log` depending on the topology/benchmark mode.
- The old example scripts are preserved as compatibility wrappers and now route
  into this launcher.
