# vLLM (mag)

Quick setup and run guide for the vLLM repo.

## Install

```bash
git clone {our repo .git}
cd vllm
uv venv                    # if no venv not yet created
VLLM_USE_PRECOMPILED=1 uv pip install --editable .
uv pip install "vllm[bench]"
```

Optional (if you use a shared cache):

```bash
# ln -s $WORK/.cache ~/.cache
```

## Run example

```bash
uv run bash examples/online_serving/disaggregated_encoder/disagg_1e1pd_example.sh
```

## Troubleshooting

- If the encoder or PD worker fails, check `./logs`.
- Example run output: `examples/online_serving/disaggregated_encoder/1e1pd.txt`
