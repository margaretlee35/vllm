# VLM Test Run (E/P/D on separate GPUs + preprocessing)

Test guide for VLM serving on this repo. Two runs, meant to be read together,
plus a video-input variant of the first (section 3):

1. **Serving run** — encode / prefill / decode as three servers on three GPUs.
   Gives throughput, TTFT, TPOT, ITL.
2. **Preprocessing run** — how long it takes to turn raw images into model
   inputs, broken down by stage. This cost sits inside TTFT in run 1 but is not
   attributed there, so it needs its own measurement.

Model defaults to `Qwen/Qwen2.5-VL-3B-Instruct` throughout.

## This folder

```
vlmtest/
├── README.md                    this file
├── run.sh                       serving run: E/P/D, one GPU each
├── preprocess.sh                preprocessing latency, by stage
├── setup_aiperf.sh              one-time: aiperf venv for --benchmark video
├── scripts/
│   ├── disagg_1e1p1d.sh         topology implementation
│   ├── gen_video_jsonl.py       synthetic MP4 pool -> aiperf JSONL
│   ├── stage_breakdown.py       per-request E/P/D stage latencies from a trace
│   └── preprocess_video.py      video preprocessing latency, incl. decode
└── logs/                        run output
```

Self-contained and separate from `epdtest/`, which is left untouched. The
topology script here started as a copy of `epdtest/scripts/disagg_1e1p1d.sh` and
differs in two ways:

- **UCX transport selection** for the NIXL prefill→decode KV transfer. The
  epdtest version hard-codes `UCX_TLS=all` / `UCX_NET_DEVICES=all`, which yields
  no usable transport on some hosts. See Troubleshooting.
- **`RUN_DIR` is exported** by `run.sh`, so the launcher and the per-server logs
  always land in one directory. (In epdtest the child derives its own timestamp,
  so the two can split across directories when the run starts on a second
  boundary.)

Pruning sweeps, LMMS accuracy eval and the JSON-driven prune configs remain in
`epdtest/`; `run.sh` here only passes `VISUAL_TOKEN_PRUNING_METHOD` through.

## Setup

Install as in `README.mag.md`, plus NIXL, which the `1e1p1d` topology needs for
the prefill→decode KV transfer.

**Match the NIXL wheel to the host CUDA version.** `pip install nixl` resolves to
the CUDA 12 build, which on a CUDA 13 / aarch64 host (GB200) segfaults inside
UCX during agent construction — a zero-size `md_resources` allocation followed by
a free of a null pointer, with no Python-level error. Check CUDA and install the
matching build:

```bash
source .venv/bin/activate
nvcc --version | tail -2                       # 13.x here
uv pip install "nixl-cu13==0.9.0"              # cu12 host: "nixl-cu12==0.9.0"
```

Both satisfy vLLM's pin in `requirements/kv_connectors.txt`
(`nixl >= 0.7.1, < 0.10.0`). Verify the agent actually constructs — importing the
module is not enough, since the failure is in agent creation:

```bash
python -c "
from nixl._api import nixl_agent, nixl_agent_config
nixl_agent('probe', nixl_agent_config(num_threads=4))
print('nixl OK')"
```

If that prints `NIXL_ERR_BACKEND` or hangs, see the UCX note in Troubleshooting.

Check you have three free GPUs:

```bash
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader
```

## 1. Serving: E / P / D on separate GPUs

One GPU per role, `GPU_E=0 GPU_P=1 GPU_D=2` by default. Override any of the
three by environment variable.

Smoke run (a few requests, confirms all three servers and the proxy come up):

```bash
bash vlmtest/run.sh --benchmark simple --num-prompts 8
```

Measurement run:

```bash
bash vlmtest/run.sh
```

Vary the image count per request:

```bash
bash vlmtest/run.sh --images-per-req 4
```

What gets launched (`vlmtest/scripts/disagg_1e1p1d.sh`):

| Role | GPU | Port | KV / EC role |
|---|---|---|---|
| encoder | `GPU_E` | 19534 | EC producer (writes embeddings to `/tmp/ec_cache`) |
| prefill | `GPU_P` | 19535 | EC consumer, NIXL `kv_producer` |
| decode | `GPU_D` | 19536 | NIXL `kv_consumer` |
| proxy | — | 10001 | fans out E → P → D |

`vllm bench serve` drives the proxy on port 10001.

## 2. Preprocessing latency

```bash
bash vlmtest/preprocess.sh
```

With more images per request, and more samples:

```bash
NUM_PROMPTS=200 bash vlmtest/preprocess.sh --images-per-req 4
```

Against the same HF dataset the `simple` serving run uses:

```bash
bash vlmtest/preprocess.sh --benchmark simple
```

### Why this is a separate run

Preprocessing happens CPU-side in the API server process, before the request
reaches any GPU worker. vLLM times it via the multimodal timing registry
(`vllm/multimodal/registry.py`), gated on
`ObservabilityConfig.enable_mm_processor_stats`.

That flag has **no CLI argument** — `vllm/engine/arg_utils.py` registers the
other observability flags but not this one — so `vllm serve` cannot turn it on,
and nothing exports the timers over HTTP or Prometheus anyway. The only consumer
is `vllm bench mm-processor`, which sets the flag itself and reads the registry
in-process. That benchmark is offline and single-GPU, hence a second run.

So: the serving run tells you what the disaggregated system does; the
preprocessing run tells you what one slice of its TTFT is made of. Keep model,
dataset and `--images-per-req` the same across both if you want to compare.

### Reading the output

```
                  Stage  Mean Median   Std P50.0 P99.0
  apply_hf_processor_ms   ...           the HF image processor: decode, resize, patchify
       get_mm_hashes_ms   ...           hashing items for the processor cache
apply_prompt_updates_ms   ...           splicing image placeholders into the prompt
  preprocessor_total_ms   ...           <- "how long preprocessing takes"
     encoder_forward_ms   ...           vision tower forward, for comparison
      num_encoder_calls   ...           encoder invocations per request
```

Two more stages, `get_cache_missing_items_ms` (processor-cache lookup) and
`merge_mm_kwargs_ms` (collecting processed items into engine kwargs), appear only
when the processor cache is enabled. With the default `MM_PROCESSOR_CACHE_GB=0`
those code paths do not run, so they are absent from the table.

`preprocessor_total_ms` is the headline number. Compare it against
`encoder_forward_ms`: if preprocessing is the larger of the two, the encoder GPU
in the `1e1p1d` topology is waiting on CPU work, and adding encoder GPUs will not
help until preprocessing is addressed.

`apply_hf_processor_ms` normally dominates the total. It scales with pixel count,
so it grows with `--images-per-req` and with image resolution.

The processor cache is **disabled by default** in `preprocess.sh`
(`MM_PROCESSOR_CACHE_GB=0`), so the numbers are cold-path per-image cost. With
the cache on, repeated items are nearly free and the mean collapses — useful for
modelling a workload with duplicate images, misleading otherwise:

```bash
MM_PROCESSOR_CACHE_GB=4 bash vlmtest/preprocess.sh
```

## 3. Video input (aiperf)

Same E/P/D topology, but the load is text + synthetic MP4 videos, and the client
is [aiperf](https://github.com/ai-dynamo/aiperf), not `vllm bench serve`. The
workflow follows dynamo's
[`benchmarks/multimodal/jsonl`](https://github.com/ai-dynamo/dynamo/tree/main/benchmarks/multimodal/jsonl)
video mode: generate a pool of deterministic MP4s, write a `single_turn` JSONL
whose rows reference them, and let aiperf base64-encode each file into a
`video_url` data URL.

One-time setup. aiperf gets its own venv (`vlmtest/.venv-aiperf`) so its pins
never touch the vLLM `.venv`:

```bash
bash vlmtest/setup_aiperf.sh
```

Smoke run, then a measurement run:

```bash
bash vlmtest/run.sh --benchmark video --num-prompts 8
bash vlmtest/run.sh --benchmark video
```

Video reuse. `--video-pool` sets how many unique videos the requests draw from.
Unset, every video is unique. Smaller pools give more repeats, which the
encoder's EC cache (`/tmp/ec_cache`) turns into hits:

```bash
bash vlmtest/run.sh --benchmark video --video-pool 50         # 300 req, ~83% reuse
bash vlmtest/run.sh --benchmark video --videos-per-req 2
```

Clip shape is set through the environment, e.g.
`VIDEO_SIZE=640x480 VIDEO_FPS=8 VIDEO_SECONDS=8`. The defaults, 320x240 at 8 fps
for 4 s, come to about 1,600 video tokens per clip on Qwen2.5-VL, because vLLM
samples 32 frames per video by default. Clips are cached in `VIDEO_DIR`
(default `/tmp/vlmtest_videos`) by content key and reused across runs.

The output length is fixed at `BENCH_OUTPUT_LEN` (default 128) with `max_tokens`
+ `ignore_eos`. `min_tokens` is left out on purpose: the proxy rewrites
`max_tokens=1` for the prefill hop but passes `min_tokens` through, so prefill
would reject every request with a 400.

Results go to `vlmtest/logs/<timestamp>/aiperf/`. `profile_export_aiperf.json` /
`.csv` hold the summary (TTFT, ITL, throughput), `profile_export.jsonl` holds
per-request records, and `video_requests.jsonl` in the run directory is the exact
input.

To split one request's latency into encode, prefill and decode, read `proxy.log`.
It timestamps "encoder requests completed", "Prefill request completed" and
"Streaming completed" per request ID.

### Video preprocessing latency

The companion to section 2, for video:

```bash
bash vlmtest/preprocess.sh --benchmark video
# exactly the requests a serving run sent:
bash vlmtest/preprocess.sh --benchmark video \
    --input-jsonl vlmtest/logs/<timestamp>/video_requests.jsonl
```

This does not go through `vllm bench mm-processor`, whose stages start at the HF
processor. For video, a real share of the cost comes before that point:
`VideoMediaIO.load_base64` base64-decodes the data URL, decodes the MP4 and
samples frames (32 by default). It runs while the chat messages are parsed,
before any timing context exists, so no registered stage covers it.
`scripts/preprocess_video.py` runs the same offline `LLM` with the same registry
stages and adds a timer around that call:

```
                  Stage   Mean ...
        video_decode_ms  18.12      per video: b64 + MP4 decode + frame sampling
  apply_hf_processor_ms  64.56      per request, as in section 2
       get_mm_hashes_ms  11.79
apply_prompt_updates_ms   0.26
  preprocessor_total_ms  76.61
     encoder_forward_ms  29.52

Per-request preprocessing, mean: 94.73 ms = video decode 18.12 ms (1 video(s))
                                            + preprocessor_total 76.61 ms
```

(Numbers from a 16-request smoke run of the default 320x240 / 8 fps / 4 s
clips.) The headline figure is the last line. `preprocessor_total_ms` alone
leaves out the decode step. Even for these small clips, preprocessing (~95 ms)
takes about three times as long as the vision forward (~30 ms), so it bounds
throughput before the encoder GPU does.

Decode time scales with resolution and clip length, while vLLM samples a fixed
32 frames. So sweep `VIDEO_SIZE` and `VIDEO_SECONDS`, not just the request
count. The synthetic clips are H.264 at a low bitrate, and real footage decodes
more slowly, so treat these numbers as a lower bound.

### GPU video decode (NVDEC)

`--video-backend nvdec` switches both scripts from vLLM's default OpenCV (CPU)
video loader to an `nvdec` loader. It is registered in
`vllm/multimodal/video.py` and uses PyNvVideoCodec on the GPU's hardware
decoder. It samples the same frame indices and returns the same
`(T, H, W, 3)` uint8 RGB array as `opencv` (pixels within ±1), so nothing
downstream changes. Frames are copied back to host because the HF processor
runs on CPU. One-time install into the vLLM venv:

```bash
source .venv/bin/activate && uv pip install pynvvideocodec==2.2.3
```

```bash
bash vlmtest/preprocess.sh --benchmark video --video-backend nvdec
GPU_PREPROC=0,3 VLLM_NVDEC_DEVICE=1 \
    bash vlmtest/preprocess.sh --benchmark video --video-backend nvdec  # decode on GPU 3
bash vlmtest/run.sh --benchmark video --video-backend nvdec
NVDEC_GPU=3 bash vlmtest/run.sh --benchmark video --video-backend nvdec
```

**Put the decoder on a GPU that is not running the model.** NVDEC is its own
engine, but each frame still needs an NV12→RGB kernel and a device→host copy.
Those run in the API-server process, and without MPS they time-slice with the
engine process's model kernels on the same GPU. Measured with 64 requests of the
default clips:

| decode | `video_decode_ms` | per-request preprocessing |
|---|---|---|
| `opencv` (CPU) | 17.2 | 93.8 ms |
| `nvdec`, same GPU as the model | 37.5 | 112.5 ms |
| `nvdec`, separate GPU | 4.4 | 79.3 ms |

`NVDEC_GPU` does this for serving. Each server then sees
`CUDA_VISIBLE_DEVICES=<its GPU>,<NVDEC_GPU>` and gets `VLLM_NVDEC_DEVICE=1`, so
the model stays on the first GPU. It costs about 0.9 GB of CUDA context per
server on the NVDEC GPU. Note that all three servers decode each video, because
every one of them receives the full request and preprocesses it.

Two more points:
- Even with decode at 4 ms, `apply_hf_processor_ms` (CPU resize, normalise,
  patchify, ~63 ms) is most of the preprocessing time. GPU decode only removes
  the smaller part.
- The first decode in each thread creates a decoder (~0.7 s). Serving loads
  media on up to `VLLM_MEDIA_LOADING_THREAD_COUNT` (8) threads, so short runs
  pay this in TTFT. The decoders are then reused.

**Proxy change this depends on.** `disagg_epd_proxy.py` picks which content
items to send to the encoder by type. `video_url` was missing from that list, so
video requests skipped the encoder. Prefill then missed in the EC cache and
quietly ran the vision tower itself. Requests still succeeded, but the run was
really a 1pd + 1d setup. `video_url` is now in `MM_TYPES`. After a run, check
that `grep -c "multimodal items" proxy.log` is non-zero and that
`/tmp/ec_cache` holds one entry per unique video.

## 4. Where the time goes: per-stage trace

`--stage-trace` makes every server write timestamped stage events. The events
are then stitched together per request, which splits TTFT into the pipeline's
stages:

```bash
bash vlmtest/run.sh --benchmark video --stage-trace
BENCH_MAX_CONCURRENCY=1 BENCH_REQUEST_RATE=inf TRACE_SKIP_FIRST=4 \
    bash vlmtest/run.sh --benchmark video --num-prompts 24 --stage-trace   # no queueing
```

| stage | what | where it is measured |
|---|---|---|
| `e_decode` | video decode (incl. media thread-pool wait) | encoder API server |
| `e_hf` | HF processor: resize / normalise / patchify | encoder API server |
| `e_render_other` | chat template, tokenisation | encoder API server |
| `e_queue` | API → engine core → first scheduled | encoder scheduler |
| `e_h2d` | collate + pin + CPU→GPU copy of pixel values | encoder worker, GPU-synced |
| `e_encode` | vision tower forward | encoder worker, GPU-synced |
| `e_ec_save` | embedding GPU→CPU + safetensors write to `EC_SHARED_STORAGE_PATH` | EC connector |
| `e_to_p` | encoder response → proxy → prefill request | gap between servers |
| `p_render` | prefill re-renders the request: decode + HF processor again | prefill API server |
| `p_queue` | API → scheduled | prefill scheduler |
| `p_ec_load` | embedding file read + CPU→GPU | EC connector, GPU-synced |
| `p_prefill` | scheduled → first token, minus `p_ec_load` | prefill scheduler |
| `p_to_d` | prefill response → proxy → decode request | gap between servers |
| `d_render` | decode re-renders the request too | decode API server |
| `d_queue` | API → KV load starts | decode scheduler |
| `d_kv` | NIXL P→D KV transfer (request start → receive finished) | decode scheduler |
| `d_first_token` | KV received → first token | decode scheduler |
| `d_decode_rest` | first → last token | decode scheduler |

Output: `stage_breakdown.txt` (the table: mean, p50/p90/p99, share of TTFT),
`stage_breakdown.csv` (one row per request) and `stage_breakdown.json` in the
run directory. The raw events are in `trace/<role>-<pid>.jsonl`. To re-analyse,
for example after dropping a warm-up wave:

```bash
python vlmtest/scripts/stage_breakdown.py vlmtest/logs/<ts>/trace --skip-first 32
```

How it works: `vllm/v1/stage_trace.py` is a no-op unless
`VLLM_STAGE_TRACE_DIR` is set. Timestamps are `time.time()` on one host, so the
three servers' events share a clock. They are joined through the proxy's
request id, which every server sees in `X-Request-Id`: the encoder as
`<id>:<item>:<rand>`, the engine as `chatcmpl-<id>-<rand>`. Embedding events are
joined through the mm hash.

Caveats:
- GPU spans synchronise the device, which removes CPU/GPU overlap. Use a traced
  run for attribution, and an untraced run for throughput.
- Batched GPU stages (`e_h2d`, `e_encode`) charge the whole batch time to every
  request in it. That is the latency each request experienced, so these rows can
  add up to more than wall time.
- The encoder → prefill hand-off goes through files (`ECExampleConnector`).
  `/tmp` is on-disk XFS here, so `e_ec_save` / `p_ec_load` include page-cache
  behaviour. `EC_SHARED_STORAGE_PATH=/dev/shm/ec_cache` makes it RAM-only.

## Output locations

| What | Where |
|---|---|
| Serving run | `vlmtest/logs/<timestamp>/` — `encoder.log`, `prefill.log`, `decode.log`, `proxy.log`, `sm.log`, `target_script.log` |
| Preprocessing run | `vlmtest/logs/<timestamp>_preprocess/` — `preprocess.log`, `preprocess.json` |
| Video run | `vlmtest/logs/<timestamp>/` — as the serving run, plus `video_requests.jsonl` and `aiperf/` |
| Video preprocessing run | `vlmtest/logs/<timestamp>_preprocess/` — `preprocess.log`, `preprocess.json`, `video_requests.jsonl` (unless `--input-jsonl`) |

`sm.log` is a CSV of per-GPU SM/memory utilization sampled once a second, with a
`role` column, so you can see how busy each of the three GPUs actually was.

## Troubleshooting

- **Prefill or decode server never comes up** — usually NIXL. Check
  `vlmtest/logs/<timestamp>/prefill.log`. The script auto-bumps the NIXL
  side-channel ports if they are taken, so a stale run does not block a new one.
  Two distinct failures:
  - `!!!!!!! Segfault encountered !!!!!!!` under `uct_md_query_tl_resources`, or
    a process spinning at ~100% CPU after `Initializing NIXL worker` — wrong
    CUDA build of the wheel. See Setup.
  - `NIXL_ERR_BACKEND`, or UCX logging `no active messages transport` /
    `Destination is unreachable` — no usable UCX transport. NIXL needs an
    active-messages transport with peer error handling; shared-memory and CUDA
    transports report `no am bcopy` / `no peer failure handler`, so TCP on a
    real interface must be in `UCX_TLS`. `disagg_1e1p1d.sh` sets
    `UCX_TLS=tcp,cuda_copy,cuda_ipc` and auto-detects the first non-loopback
    interface, ignoring any inherited `UCX_NET_DEVICES` — cluster profiles often
    export an mlx5 list that belongs to a *system* UCX, and the wheel's bundled
    UCX reports those devices as unavailable. Override deliberately with
    `NIXL_UCX_TLS` / `NIXL_UCX_NET_DEVICES`:

    ```bash
    NIXL_UCX_NET_DEVICES=ibP2p1s0 bash vlmtest/run.sh
    ```

    Harmless noise either way: repeated
    `Invalid value for MLX5_DEVX_OBJECTS: 'auto'` / `Failed to read MD config`.
    That comes from the system UCX's setting being rejected by the wheel's older
    UCX, and does not prevent startup once a TCP transport is available.

    The TCP default is correctness-first, not bandwidth-first. For P→D transfer
    bandwidth on the IB fabric, point `NIXL_UCX_NET_DEVICES` at an `ib*`
    interface and confirm it beats the auto-detected one.
- **Proxy times out** — raise `TIMEOUT_SECONDS`; three servers loading weights
  takes well over the 120 s default on a cold cache.
- **Triton fails to JIT with `-Wno-psabi`** — the NVHPC module sets `CC=nvc`.
  Both `run.sh` and `preprocess.sh` fall back to `gcc`, but if you invoke
  `vllm` directly, export `CC=gcc CXX=g++` first.
- **Stale GPU memory after a failed run** — `nvidia-smi` and kill leftover
  `vllm serve` processes; the trap in the topology script misses them if the
  shell was killed with `-9`.
