# epdtest

This directory is the direct entrypoint for local EPD testing.

Use this instead of remembering which legacy wrapper lives under
`examples/online_serving/disaggregated_encoder/lovelace`.

## Quick Start

Default local run:

```bash
bash epdtest/run.sh
```

Choose topology explicitly:

```bash
bash epdtest/run.sh --topology 1e1pd
bash epdtest/run.sh --topology 1e1p1d
```

Enable the heavier metrics/profiling flow:

```bash
bash epdtest/run.sh --profile metrics
bash epdtest/run.sh --topology 1e1p1d --profile metrics
```

All existing environment overrides still work:

```bash
GPU_E=0 GPU_PD=1 NUM_PROMPTS=50 bash epdtest/run.sh
TOPOLOGY=1e1p1d PROFILE=metrics GPU_E=0 GPU_P=1 GPU_D=2 bash epdtest/run.sh
```

## Slurm

`epdtest/epd.slurm` is the simplest batch entrypoint for repeated EPD runs on
cluster. It wraps `epdtest/run.sh`, loops over `IMAGES_PER_REQ_LIST`, and logs
each sweep into `epdtest/slurm_logs/`.

Default batch run:

```bash
sbatch epdtest/epd.slurm
```

Run the lighter simple profile instead of metrics:

```bash
sbatch --export=ALL,PROFILE=simple epdtest/epd.slurm
```

Run the `1e1p1d` topology:

```bash
sbatch --export=ALL,TOPOLOGY=1e1p1d,GPU_E=0,GPU_P=1,GPU_D=2 epdtest/epd.slurm
```

Rerun without reinstalling dependencies:

```bash
sbatch --export=ALL,SKIP_INSTALL=1 epdtest/epd.slurm
```

Sweep a smaller set of image counts:

```bash
sbatch --export=ALL,SKIP_INSTALL=1,IMAGES_PER_REQ_LIST="1 2 4" epdtest/epd.slurm
```

Run a quick smoke benchmark:

```bash
sbatch --export=ALL,SKIP_INSTALL=1,NUM_PROMPTS=20,TIMEOUT_SECONDS=300 epdtest/epd.slurm
```

Override the model:

```bash
sbatch --export=ALL,SKIP_INSTALL=1,MODEL="Qwen/Qwen2.5-VL-3B-Instruct" epdtest/epd.slurm
```

Override VisionZip settings:

```bash
sbatch --export=ALL,SKIP_INSTALL=1,VISION_ZIP_RATE=0.5,VISION_ZIP_DOMINANT_RATIO=0.75,VISION_ZIP_ATTENTION_LAYER=-2 epdtest/epd.slurm
```

Write logs somewhere else:

```bash
sbatch --export=ALL,SKIP_INSTALL=1,LOG_PATH=/scratch/$USER/epdtest_logs,SLURM_LOG_DIR=/scratch/$USER/epdtest_slurm epdtest/epd.slurm
```

The main variables exposed by `epd.slurm` are:

- `TOPOLOGY`: `1e1pd` or `1e1p1d`
- `PROFILE`: `simple` or `metrics`
- `MODEL`: VLM to serve
- `GPU_E`, `GPU_PD`, `GPU_P`, `GPU_D`: GPU assignment
- `NUM_PROMPTS`: benchmark request count
- `IMAGES_PER_REQ_LIST`: space-separated sweep values
- `SKIP_INSTALL`: set to `1` to skip `uv pip install`
- `VISION_ZIP_RATE`, `VISION_ZIP_DOMINANT_RATIO`, `VISION_ZIP_ATTENTION_LAYER`: VisionZip tuning
- `LOG_PATH`, `SLURM_LOG_DIR`: output locations

## Notes

- `epdtest/logs/` is still the home for logs and log analysis helpers.
- The old example scripts are preserved as compatibility wrappers and now route
  into this launcher.
