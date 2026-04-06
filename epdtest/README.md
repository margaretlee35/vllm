# epdtest

This directory is the direct entrypoint for local EPD testing.

Use this instead of remembering which legacy wrapper lives under
`examples/online_serving/disaggregated_encoder/lovelace`.

`epdtest/run.sh` is the single-run entrypoint. It chooses the topology/profile
script, activates `.venv` when present, and runs one configuration once.

`epdtest/run_all.sh` is the outer sweep wrapper. It is the right entrypoint for
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
```

Enable the heavier `randommm` flow:

```bash
bash epdtest/run.sh --profile randommm
bash epdtest/run.sh --topology 1e1p1d --profile randommm
bash epdtest/run.sh --profile randommm --images-per-req 4
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

VisionZip arguments are also available directly on the launcher:

```bash
bash epdtest/run.sh --visual-token-pruning-method vision_zip --vision-zip-rate 0.5 --vision-zip-dominant-ratio 0.75 --vision-zip-attention-layer -2
```

Visual token pruning is not enabled by default anymore. To turn it on, pass an
explicit pruning method such as `--visual-token-pruning-method vision_zip`.

Use `run_all.sh` for outer sweeps:

```bash
bash epdtest/run_all.sh noprune
bash epdtest/run_all.sh visionzip
IMAGES_PER_REQ_LIST="1 2 4 8" bash epdtest/run_all.sh cdprune
```

## Slurm

`epdtest/epd.slurm` is intentionally minimal now. It just exports the requested
topology/profile/prune mode and calls `epdtest/run_all.sh`.

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

Run VisionZip sweep instead of `noprune`:

```bash
sbatch --export=ALL,PRUNE_MODE=visionzip epdtest/epd.slurm
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

Override VisionZip settings:

```bash
sbatch --export=ALL,SKIP_INSTALL=1,VISION_ZIP_RATE=0.5,VISION_ZIP_DOMINANT_RATIO=0.75,VISION_ZIP_ATTENTION_LAYER=-2 epdtest/epd.slurm
```

The main variables to override are now consumed by `run.sh` or `run_all.sh`:

- `TOPOLOGY`: `1e1pd` or `1e1p1d`
- `PROFILE`: `simple` or `randommm`
- `PRUNE_MODE`: `noprune`, `visionzip`, or `cdprune` for `run_all.sh` / slurm
- `MODEL`: VLM to serve
- `LOG_PATH`: output directory for launcher-owned logs
- `IMAGES_PER_REQ`: single-run image count for `run.sh`
- `IMAGES_PER_REQ_LIST`: space-separated sweep values for `run_all.sh`
- `SKIP_INSTALL`: set to `1` to skip the default install
- `VISUAL_TOKEN_PRUNING_METHOD`: optional explicit pruning method
- `GPU_E`, `GPU_PD`, `GPU_P`, `GPU_D`: GPU assignment
- `NUM_PROMPTS`: benchmark request count
- `VISION_ZIP_RATE`, `VISION_ZIP_DOMINANT_RATIO`, `VISION_ZIP_ATTENTION_LAYER`: VisionZip tuning

## Notes

- `epdtest/logs/` is still the home for logs and log analysis helpers.
- Each run now creates a timestamped subdirectory like
  `epdtest/logs/20260405_153000/`. For `randommm` sweeps, each item lands under
  a nested directory like `epdtest/logs/20260405_153000/ipr1/`. Inside those
  folders you will see files such as `encoder.log`, `prefill_decode.log`,
  `prefill.log`, `decode.log`, `proxy.log`, `kv.log`, `sm.log`, and
  `target_script.log` depending on the topology/profile.
- The old example scripts are preserved as compatibility wrappers and now route
  into this launcher.
