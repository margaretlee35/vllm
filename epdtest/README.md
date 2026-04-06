# epdtest

This directory is the direct entrypoint for local EPD testing.

Use this instead of remembering which legacy wrapper lives under
`examples/online_serving/disaggregated_encoder/lovelace`.

`epdtest/run.sh` is the main control surface. It chooses the topology/profile
script, activates `.venv` when present, installs dependencies by default, and
handles the `randommm` image sweep. `epdtest/epd.slurm` is now a thin batch
wrapper around that launcher.

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

Enable the heavier `randommm` flow:

```bash
bash epdtest/run.sh --profile randommm
bash epdtest/run.sh --topology 1e1p1d --profile randommm
```

Skip install when you already have the environment ready:

```bash
bash epdtest/run.sh --skip-install
```

All existing environment overrides still work:

```bash
GPU_E=0 GPU_PD=1 NUM_PROMPTS=50 bash epdtest/run.sh
TOPOLOGY=1e1p1d PROFILE=randommm GPU_E=0 GPU_P=1 GPU_D=2 bash epdtest/run.sh
MODEL=Qwen/Qwen2.5-VL-3B-Instruct LOG_PATH=/tmp/epdtest_logs bash epdtest/run.sh
PROFILE=randommm IMAGES_PER_REQ_LIST="1 2 4 8" bash epdtest/run.sh
```

## Slurm

`epdtest/epd.slurm` is intentionally minimal now. It just calls
`epdtest/run.sh` with the requested topology/profile and optionally skips the
install step.

Default batch run:

```bash
sbatch epdtest/epd.slurm
```

Run the lighter simple profile instead of `randommm`:

```bash
sbatch --export=ALL,PROFILE=simple epdtest/epd.slurm
```

Run the `randommm` profile explicitly:

```bash
sbatch --export=ALL,PROFILE=randommm epdtest/epd.slurm
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
sbatch --export=ALL,SKIP_INSTALL=1,LOG_PATH=/scratch/$USER/epdtest_logs epdtest/epd.slurm
```

The main variables to override are now consumed by `run.sh`:

- `TOPOLOGY`: `1e1pd` or `1e1p1d`
- `PROFILE`: `simple` or `randommm`
- `MODEL`: VLM to serve
- `LOG_PATH`: output directory for launcher-owned logs
- `IMAGES_PER_REQ_LIST`: space-separated sweep values for `randommm`
- `SKIP_INSTALL`: set to `1` to skip the default install
- `GPU_E`, `GPU_PD`, `GPU_P`, `GPU_D`: GPU assignment
- `NUM_PROMPTS`: benchmark request count
- `VISION_ZIP_RATE`, `VISION_ZIP_DOMINANT_RATIO`, `VISION_ZIP_ATTENTION_LAYER`: VisionZip tuning

## Notes

- `epdtest/logs/` is still the home for logs and log analysis helpers.
- The old example scripts are preserved as compatibility wrappers and now route
  into this launcher.
