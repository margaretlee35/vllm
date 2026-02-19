# install vllm
git clone {our repo .git}
cd vllm
uv venv # if no venv created
VLLM_USE_PRECOMPILED=1 uv pip install --editable .
uv pip install "vllm[bench]"

# ln -s $WORK/.cache ~/.cache

# run example
uv run bash examples/online_serving/disaggregated_encoder/disagg_1e1pd_example.sh

#if something failing with your encoder / pd worker check ./logs
#example of run result: examples/online_serving/disaggregated_encoder/1e1pd.log
