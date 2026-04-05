# vLLM (mag)

Quick setup and run guide for the vLLM repo.

## Install

```bash
git clone {our repo .git}
cd vllm
uv venv                    # if no venv not yet created
source .venv/bin/activate
VLLM_USE_PRECOMPILED=1 uv pip install --editable .
uv pip install "vllm[bench]"
```

Optional (if you use a shared cache):

```bash
# ln -s $WORK/.cache ~/.cache
```

## Run example

```bash
uv run bash ./epdtest/run.sh
```

Run the 1e1p1d topology instead:

```bash
uv run --extra bench bash ./epdtest/run.sh --topology 1e1p1d
```

Note:
- `epdtest/` is the direct local EPD test entrypoint in this repo.
- Legacy scripts under `examples/online_serving/disaggregated_encoder/lovelace/` are compatibility wrappers that route into `epdtest/run.sh`.

## Troubleshooting

- If the encoder or PD worker fails, check `./epdtest/logs`.
- Example run output: `examples/online_serving/disaggregated_encoder/1e1pd.txt`
