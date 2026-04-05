# vLLM Setup (Source Build)

Run these commands from `/workspace/sylee/git_repos/vllm`.

## 1) Branch setup (`sylee`)

```bash
cd /workspace/sylee/git_repos/vllm
git fetch origin
git checkout main
git pull --ff-only
```

If `origin/sylee` already exists:

```bash
git checkout -B sylee origin/sylee
```

If `origin/sylee` does not exist:

```bash
git checkout -b sylee
git push -u origin sylee
```

## 2) Install with `uv` (full build with compilation)

```bash
cd /workspace/sylee/git_repos/vllm
uv venv --python 3.12 --seed --managed-python
bash tools/patch_venv_activate_cuda.sh .venv
source .venv/bin/activate
nvcc --version
uv pip install -r requirements/build.txt --torch-backend=auto
uv pip install -e '.[bench]' --torch-backend=auto
uv pip install "nixl>=0.7.1,<0.10.0"
```

Note: "uv pip install -e . --torch-backend=auto" will take > 15 minutes

## 3) Run

Quick smoke run:

```bash
cd examples/online_serving/disaggregated_encoder
TIMEOUT_SECONDS=120 NUM_PROMPTS=1 \
GPU_E=0 GPU_PD=1 \
uv run bash ./epdtest/run.sh
```

```bash
cd examples/online_serving/disaggregated_encoder
TIMEOUT_SECONDS=120 NUM_PROMPTS=1 \
GPU_E=0 GPU_P=0 GPU_D=1 \
uv run --extra bench ./epdtest/run.sh --topology 1e1p1d
```

Run 1E1PD:

```bash
cd examples/online_serving/disaggregated_encoder
TIMEOUT_SECONDS=120 NUM_PROMPTS=1000 \
GPU_E=0 GPU_PD=1 \
uv run --extra bench bash ./epdtest/run.sh
```

Run 1E1P1D (requires NIXL):

```bash
cd examples/online_serving/disaggregated_encoder
TIMEOUT_SECONDS=120 NUM_PROMPTS=1000 \
GPU_E=0 GPU_P=0 GPU_D=1 \
uv run --extra bench ./epdtest/run.sh --topology 1e1p1d
```

Note:
`epdtest` is just the local test directory name. Change env settings as needed for a specific server.
