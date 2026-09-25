# SPDX-License-Identifier: Apache-2.0
"""Video preprocessing latency, by stage, on an aiperf video JSONL.

`vllm bench mm-processor` reports the multimodal timing registry, which starts
at the HF processor. For video, a large piece of preprocessing happens before
that: `VideoMediaIO.load_base64` base64-decodes the data URL, demuxes and
decodes the MP4 and samples frames. It runs while chat messages are parsed,
before any timing context exists, so no registered stage covers it.

This script runs the same in-process `LLM` the benchmark uses, fed with the
exact JSONL that `run.sh --benchmark video` sends through aiperf, and wraps
`VideoMediaIO.load_base64` with a timer to add that missing stage:

  video_decode_ms          per video: b64 decode + MP4 decode + frame sampling
  apply_hf_processor_ms    ... registry stages, per request, as in
  preprocessor_total_ms        vllm bench mm-processor
  encoder_forward_ms
"""

import argparse
import base64
import json
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

import vllm.envs as envs
from vllm import LLM, SamplingParams
from vllm.benchmarks.mm_processor import (
    calculate_mm_processor_metrics,
    collect_mm_processor_stats,
)
from vllm.multimodal.media.video import VideoMediaIO

_decode_secs: list[float] = []
_orig_load_base64 = VideoMediaIO.load_base64


def _timed_load_base64(self, media_type, data):
    t0 = time.perf_counter()
    try:
        return _orig_load_base64(self, media_type, data)
    finally:
        _decode_secs.append(time.perf_counter() - t0)


VideoMediaIO.load_base64 = _timed_load_base64


def load_conversations(path: Path) -> list[list[dict]]:
    """JSONL rows -> chat messages, video first, as aiperf builds them."""
    b64_cache: dict[str, str] = {}
    convs = []
    for line in path.read_text().splitlines():
        row = json.loads(line)
        content = []
        for video in row.get("videos", []):
            if video not in b64_cache:
                b64_cache[video] = base64.b64encode(Path(video).read_bytes()).decode()
            url = f"data:video/mp4;base64,{b64_cache[video]}"
            content.append({"type": "video_url", "video_url": {"url": url}})
        content.append({"type": "text", "text": row["text"]})
        convs.append([{"role": "user", "content": content}])
    return convs


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--input-jsonl", type=Path, required=True)
    p.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    p.add_argument("--num-warmups", type=int, default=4)
    p.add_argument("--output-len", type=int, default=1)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    p.add_argument("--max-model-len", type=int, default=65536)
    p.add_argument("--mm-processor-cache-gb", type=float, default=0)
    p.add_argument("--metric-percentiles", default="50,99")
    p.add_argument("--output-json", type=Path, default=None)
    args = p.parse_args()

    convs = load_conversations(args.input_jsonl)
    videos_per_req = max(
        sum(c["type"] == "video_url" for c in conv[0]["content"]) for conv in convs
    )
    percentiles = [float(x) for x in args.metric_percentiles.split(",")]

    llm = LLM(
        model=args.model,
        seed=0,
        enforce_eager=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        mm_processor_cache_gb=args.mm_processor_cache_gb,
        limit_mm_per_prompt={"image": 0, "video": videos_per_req},
        enable_mm_processor_stats=True,
    )
    sampling = SamplingParams(temperature=0.0, max_tokens=args.output_len)

    # Warm up on the first rows; with the processor cache off (the default)
    # reusing them in the measured set gives them no advantage.
    if args.num_warmups > 0:
        print(f"Processing {args.num_warmups} warmup requests...")
        llm.chat(convs[: args.num_warmups], sampling)
    collect_mm_processor_stats(llm.llm_engine)  # drains the registry
    _decode_secs.clear()

    print(f"Processing {len(convs)} requests...")
    t0 = time.perf_counter()
    outputs = llm.chat(convs, sampling)
    total_secs = time.perf_counter() - t0

    stats = collect_mm_processor_stats(llm.llm_engine)
    stats = {"video_decode_secs": list(_decode_secs), **stats}
    metrics = calculate_mm_processor_metrics(stats, percentiles)

    rows = []
    for stage, m in metrics.items():
        row = {
            "Stage": stage,
            "Mean": f"{m['mean']:.2f}",
            "Median": f"{m['median']:.2f}",
            "Std": f"{m['std']:.2f}",
        }
        for pct in percentiles:
            row[f"P{pct}"] = f"{m[f'p{pct}']:.2f}"
        rows.append(row)
    print(
        "\nMM Processor Metrics (video_decode_ms is per video, the rest per request):"
    )
    print(pd.DataFrame(rows).to_string(index=False))

    decode_per_req = metrics["video_decode_ms"]["mean"] * videos_per_req
    proc_total = metrics.get("preprocessor_total_ms", {}).get("mean", 0.0)
    print(
        f"\nPer-request preprocessing, mean: {decode_per_req + proc_total:.2f} ms "
        f"= video decode {decode_per_req:.2f} ms ({videos_per_req} video(s)) "
        f"+ preprocessor_total {proc_total:.2f} ms"
    )
    mean_prompt_tokens = sum(len(o.prompt_token_ids) for o in outputs) / len(outputs)
    print(f"video backend: {envs.VLLM_VIDEO_LOADER_BACKEND}")
    print(f"mean prompt tokens: {mean_prompt_tokens:.0f}")
    print(f"{len(outputs)} requests in {total_secs:.2f} s")

    if args.output_json:
        result = {
            "completed": sum(o.finished for o in outputs),
            "mean_prompt_tokens": mean_prompt_tokens,
            "total_secs": total_secs,
            "videos_per_request": videos_per_req,
            "num_video_decodes": len(_decode_secs),
            "mean_preprocess_per_request_ms": decode_per_req + proc_total,
            "mm_processor_stats": metrics,
            "config": {
                "model": args.model,
                "input_jsonl": str(args.input_jsonl),
                "num_prompts": len(convs),
                "mm_processor_cache_gb": args.mm_processor_cache_gb,
                "video_backend": envs.VLLM_VIDEO_LOADER_BACKEND,
            },
            "timestamp": datetime.now().isoformat(),
        }
        args.output_json.write_text(json.dumps(result, indent=2))
        print(f"Results saved to {args.output_json}")


if __name__ == "__main__":
    main()
