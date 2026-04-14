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
uv pip install lmms-eval
```

Note: "uv pip install -e . --torch-backend=auto" will take > 15 minutes