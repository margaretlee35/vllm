# Lovelace

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

## Notes

- `epdtest/logs/` is still the home for logs and log analysis helpers.
- The old example scripts are preserved as compatibility wrappers and now route
  into this launcher.
