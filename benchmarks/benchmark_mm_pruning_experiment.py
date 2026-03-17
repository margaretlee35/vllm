# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
r"""Run reproducible multimodal pruning speed experiments.

This script automates baseline-vs-variant comparisons for multimodal workloads:

- Encoder-focused latency via `vllm bench mm-processor` (random-mm dataset).
- End-to-end serving latency/throughput via `vllm bench serve` (random-mm).

Each variant can provide extra engine args (for example, pruning flags), and the
script emits a unified summary JSON/CSV with speedups relative to a baseline.
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import csv
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_LIMIT_MM_PER_PROMPT = '{"image": 3, "video": 0}'
DEFAULT_BUCKET_CONFIG = "{(256, 256, 1): 0.7, (720, 1280, 1): 0.3}"

LOWER_IS_BETTER_METRICS = (
    "encoder_forward_ms_mean",
    "preprocessor_total_ms_mean",
    "mm_e2el_ms_mean",
    "serve_ttft_ms_mean",
    "serve_tpot_ms_mean",
    "serve_e2el_ms_mean",
)

HIGHER_IS_BETTER_METRICS = (
    "serve_request_throughput",
    "serve_output_throughput",
    "serve_total_token_throughput",
)


@dataclass(frozen=True)
class VariantSpec:
    name: str
    extra_engine_args: list[str]


def _quote_bare_json_keys(text: str) -> str:
    return re.sub(
        r'([{\[,]\s*)([A-Za-z_][A-Za-z0-9_-]*)(\s*:)',
        r'\1"\2"\3',
        text,
    )


def _normalize_jsonish_value(value: str) -> str:
    raw = value.strip()

    with contextlib.suppress(Exception):
        json.loads(raw)
        return raw

    with contextlib.suppress(Exception):
        parsed = ast.literal_eval(raw)
        return json.dumps(parsed, separators=(",", ":"))

    if raw.startswith("{") and raw.endswith("}"):
        candidate = _quote_bare_json_keys(raw.replace("'", '"'))
        with contextlib.suppress(Exception):
            json.loads(candidate)
            return candidate

    return value


def _normalize_special_flag_values(args: list[str]) -> list[str]:
    normalized = list(args)
    for i in range(len(normalized) - 1):
        if normalized[i] == "--mm-processor-kwargs":
            normalized[i + 1] = _normalize_jsonish_value(normalized[i + 1])
    return normalized


def parse_variant_spec(spec: str) -> VariantSpec:
    if "::" in spec:
        name, args_text = spec.split("::", maxsplit=1)
    else:
        name, args_text = spec, ""

    name = name.strip()
    if not name:
        raise ValueError(f"Invalid variant spec: {spec!r}")

    return VariantSpec(
        name=name,
        extra_engine_args=_normalize_special_flag_values(shlex.split(args_text)),
    )


def parse_extra_args(raw: str) -> list[str]:
    return _normalize_special_flag_values(shlex.split(raw)) if raw.strip() else []


def resolve_vllm_command(explicit: str | None) -> list[str]:
    if explicit:
        return shlex.split(explicit)
    if shutil.which("vllm") is not None:
        return ["vllm"]
    return [sys.executable, "-m", "vllm.entrypoints.cli.main"]


def tail_text(path: Path, max_lines: int = 60) -> str:
    if not path.exists():
        return f"(missing log: {path})"
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(lines[-max_lines:])


def run_command(cmd: list[str], log_path: Path, env: dict[str, str]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        print(f"[run] {shlex.join(cmd)}")
        completed = subprocess.run(
            cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            env=env,
            check=False,
        )
    if completed.returncode != 0:
        raise RuntimeError(
            f"Command failed ({completed.returncode}): {shlex.join(cmd)}\n"
            f"Log tail ({log_path}):\n{tail_text(log_path)}"
        )


def wait_for_server(base_url: str, timeout_sec: int) -> None:
    deadline = time.time() + timeout_sec
    models_url = f"{base_url}/v1/models"
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(models_url, timeout=3) as response:
                if 200 <= response.status < 300:
                    return
        except (urllib.error.URLError, TimeoutError):
            time.sleep(2)
    raise TimeoutError(f"Server not ready in {timeout_sec}s at {models_url}")


def start_server(
    vllm_command: list[str],
    model: str,
    port: int,
    common_engine_args: list[str],
    variant_engine_args: list[str],
    env: dict[str, str],
    log_path: Path,
    timeout_sec: int,
    host: str,
) -> subprocess.Popen[str]:
    cmd = [
        *vllm_command,
        "serve",
        model,
        "--port",
        str(port),
        *common_engine_args,
        *variant_engine_args,
    ]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = log_path.open("w", encoding="utf-8")
    print(f"[run] {shlex.join(cmd)}")
    process = subprocess.Popen(
        cmd,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        env=env,
        text=True,
    )
    try:
        wait_for_server(f"http://{host}:{port}", timeout_sec=timeout_sec)
        return process
    except Exception:
        process.terminate()
        try:
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            process.kill()
        raise RuntimeError(
            "Failed to start server.\n"
            f"Log tail ({log_path}):\n{tail_text(log_path)}"
        ) from None
    finally:
        log_file.close()


def stop_server(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)


def build_mm_processor_cmd(
    vllm_command: list[str],
    args: argparse.Namespace,
    model: str,
    common_engine_args: list[str],
    variant_engine_args: list[str],
    output_json: Path,
) -> list[str]:
    cmd = [
        *vllm_command,
        "bench",
        "mm-processor",
        "--model",
        model,
        *common_engine_args,
        *variant_engine_args,
        "--dataset-name",
        "random-mm",
        "--num-prompts",
        str(args.mm_num_prompts),
        "--num-warmups",
        str(args.mm_num_warmups),
        "--random-input-len",
        str(args.random_input_len),
        "--random-output-len",
        str(args.random_output_len),
        "--random-mm-base-items-per-request",
        str(args.random_mm_base_items_per_request),
        "--random-mm-num-mm-items-range-ratio",
        str(args.random_mm_num_mm_items_range_ratio),
        "--random-mm-limit-mm-per-prompt",
        args.random_mm_limit_mm_per_prompt,
        "--random-mm-bucket-config",
        args.random_mm_bucket_config,
        "--metric-percentiles",
        args.metric_percentiles,
        "--output-json",
        str(output_json),
        "--disable-tqdm",
    ]
    return cmd


def build_serve_bench_cmd(
    vllm_command: list[str],
    args: argparse.Namespace,
    model: str,
    host: str,
    port: int,
    variant_dir: Path,
    result_filename: str,
) -> list[str]:
    cmd = [
        *vllm_command,
        "bench",
        "serve",
        "--backend",
        "openai-chat",
        "--model",
        model,
        "--host",
        host,
        "--port",
        str(port),
        "--endpoint",
        "/v1/chat/completions",
        "--dataset-name",
        "random-mm",
        "--num-prompts",
        str(args.serve_num_prompts),
        "--num-warmups",
        str(args.serve_num_warmups),
        "--request-rate",
        str(args.request_rate),
        "--metric-percentiles",
        args.metric_percentiles,
        "--percentile-metrics",
        "ttft,tpot,itl,e2el",
        "--temperature",
        "0",
        "--random-input-len",
        str(args.random_input_len),
        "--random-output-len",
        str(args.random_output_len),
        "--random-mm-base-items-per-request",
        str(args.random_mm_base_items_per_request),
        "--random-mm-num-mm-items-range-ratio",
        str(args.random_mm_num_mm_items_range_ratio),
        "--random-mm-limit-mm-per-prompt",
        args.random_mm_limit_mm_per_prompt,
        "--random-mm-bucket-config",
        args.random_mm_bucket_config,
        "--save-result",
        "--result-dir",
        str(variant_dir),
        "--result-filename",
        result_filename,
        "--disable-tqdm",
    ]
    if args.max_concurrency is not None:
        cmd.extend(["--max-concurrency", str(args.max_concurrency)])
    return cmd


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def get_stage_percentile(stage_metrics: dict[str, Any], percentile: float) -> float | None:
    candidates = [
        f"p{int(percentile)}" if percentile.is_integer() else f"p{percentile}",
        f"p{percentile}",
    ]
    for key in candidates:
        value = stage_metrics.get(key)
        if value is not None:
            return float(value)
    return None


def get_list_percentile(metric_pairs: Any, percentile: float) -> float | None:
    if not isinstance(metric_pairs, list):
        return None
    for pair in metric_pairs:
        if (
            isinstance(pair, (list, tuple))
            and len(pair) == 2
            and float(pair[0]) == percentile
        ):
            return float(pair[1])
    return None


def extract_mm_metrics(mm_result: dict[str, Any], percentile: float) -> dict[str, float | None]:
    stats = mm_result.get("mm_processor_stats", {})
    encoder = stats.get("encoder_forward_ms", {})
    preproc = stats.get("preprocessor_total_ms", {})

    return {
        "encoder_forward_ms_mean": float(encoder["mean"]) if "mean" in encoder else None,
        "encoder_forward_ms_p": get_stage_percentile(encoder, percentile),
        "preprocessor_total_ms_mean": (
            float(preproc["mean"]) if "mean" in preproc else None
        ),
        "preprocessor_total_ms_p": get_stage_percentile(preproc, percentile),
        "mm_e2el_ms_mean": (
            float(mm_result["mean_e2el_ms"]) if "mean_e2el_ms" in mm_result else None
        ),
        "mm_e2el_ms_p": get_list_percentile(
            mm_result.get("percentiles_e2el_ms"), percentile
        ),
    }


def extract_serve_metrics(
    serve_result: dict[str, Any], percentile: float
) -> dict[str, float | None]:
    return {
        "serve_request_throughput": (
            float(serve_result["request_throughput"])
            if "request_throughput" in serve_result
            else None
        ),
        "serve_output_throughput": (
            float(serve_result["output_throughput"])
            if "output_throughput" in serve_result
            else None
        ),
        "serve_total_token_throughput": (
            float(serve_result["total_token_throughput"])
            if "total_token_throughput" in serve_result
            else None
        ),
        "serve_ttft_ms_mean": (
            float(serve_result["mean_ttft_ms"]) if "mean_ttft_ms" in serve_result else None
        ),
        "serve_ttft_ms_p": get_list_percentile(
            serve_result.get("percentiles_ttft_ms"), percentile
        ),
        "serve_tpot_ms_mean": (
            float(serve_result["mean_tpot_ms"]) if "mean_tpot_ms" in serve_result else None
        ),
        "serve_tpot_ms_p": get_list_percentile(
            serve_result.get("percentiles_tpot_ms"), percentile
        ),
        "serve_e2el_ms_mean": (
            float(serve_result["mean_e2el_ms"]) if "mean_e2el_ms" in serve_result else None
        ),
        "serve_e2el_ms_p": get_list_percentile(
            serve_result.get("percentiles_e2el_ms"), percentile
        ),
    }


def compute_speedups(
    baseline_metrics: dict[str, float | None], metrics: dict[str, float | None]
) -> dict[str, float | None]:
    speedups: dict[str, float | None] = {}
    for key in LOWER_IS_BETTER_METRICS:
        base_value = baseline_metrics.get(key)
        cur_value = metrics.get(key)
        if (
            base_value is None
            or cur_value is None
            or base_value <= 0
            or cur_value <= 0
        ):
            speedups[f"{key}_speedup_vs_baseline"] = None
        else:
            speedups[f"{key}_speedup_vs_baseline"] = base_value / cur_value

    for key in HIGHER_IS_BETTER_METRICS:
        base_value = baseline_metrics.get(key)
        cur_value = metrics.get(key)
        if (
            base_value is None
            or cur_value is None
            or base_value <= 0
            or cur_value <= 0
        ):
            speedups[f"{key}_speedup_vs_baseline"] = None
        else:
            speedups[f"{key}_speedup_vs_baseline"] = cur_value / base_value

    return speedups


def write_summary_csv(rows: list[dict[str, Any]], path: Path) -> None:
    all_fields: set[str] = set()
    for row in rows:
        all_fields.update(row.keys())
    preferred = [
        "variant",
        "port",
        "mm_result_json",
        "serve_result_json",
    ]
    fields = preferred + sorted(all_fields - set(preferred))

    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def make_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run baseline-vs-variant multimodal pruning speed experiment."
    )
    parser.add_argument("--model", required=True, help="Model to benchmark.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmark_results/mm_pruning_experiment"),
        help="Directory where logs and results are written.",
    )
    parser.add_argument(
        "--variant",
        action="append",
        default=None,
        help=(
            "Variant spec in the form '<name>::<extra engine args>'. "
            "Can be passed multiple times. "
            "Example: --variant 'baseline::' "
            "--variant 'cdpruner-320::--my-prune-flag 320'"
        ),
    )
    parser.add_argument(
        "--baseline-name",
        default=None,
        help=(
            "Variant name used as baseline for speedup calculation. "
            "Defaults to the first variant."
        ),
    )
    parser.add_argument(
        "--common-engine-args",
        default="",
        help="Extra engine args applied to all variants (quoted string).",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Benchmark target host.")
    parser.add_argument(
        "--base-port",
        type=int,
        default=18000,
        help="First server port. Each variant increments by 1.",
    )
    parser.add_argument(
        "--startup-timeout-sec",
        type=int,
        default=600,
        help="Timeout for server readiness checks.",
    )
    parser.add_argument(
        "--cuda-visible-devices",
        default=None,
        help="Optional CUDA_VISIBLE_DEVICES value for local runs.",
    )
    parser.add_argument(
        "--vllm-command",
        default=None,
        help=(
            "Command used to invoke vLLM CLI. Defaults to auto-detect: "
            "'vllm' if available, otherwise "
            f"'{sys.executable} -m vllm.entrypoints.cli.main'."
        ),
    )

    parser.add_argument("--mm-num-prompts", type=int, default=50)
    parser.add_argument("--mm-num-warmups", type=int, default=5)
    parser.add_argument("--serve-num-prompts", type=int, default=100)
    parser.add_argument("--serve-num-warmups", type=int, default=10)
    parser.add_argument("--request-rate", default="inf")
    parser.add_argument("--max-concurrency", type=int, default=None)
    parser.add_argument("--metric-percentiles", default="50,90,95,99")
    parser.add_argument("--percentile-for-summary", type=float, default=99.0)

    parser.add_argument("--random-input-len", type=int, default=300)
    parser.add_argument("--random-output-len", type=int, default=40)
    parser.add_argument("--random-mm-base-items-per-request", type=int, default=2)
    parser.add_argument(
        "--random-mm-num-mm-items-range-ratio",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--random-mm-limit-mm-per-prompt",
        default=DEFAULT_LIMIT_MM_PER_PROMPT,
        help="Raw value passed to --random-mm-limit-mm-per-prompt.",
    )
    parser.add_argument(
        "--random-mm-bucket-config",
        default=DEFAULT_BUCKET_CONFIG,
        help="Raw value passed to --random-mm-bucket-config.",
    )
    parser.add_argument(
        "--skip-mm-processor",
        action="store_true",
        help="Skip the encoder-focused mm-processor benchmark.",
    )
    parser.add_argument(
        "--skip-serve",
        action="store_true",
        help="Skip the end-to-end serve benchmark.",
    )

    return parser


def main() -> None:
    parser = make_arg_parser()
    args = parser.parse_args()

    variants_raw = args.variant or ["baseline::"]
    variants = [parse_variant_spec(spec) for spec in variants_raw]
    variant_names = [v.name for v in variants]
    if len(set(variant_names)) != len(variant_names):
        raise ValueError(f"Duplicate variant names found: {variant_names}")

    baseline_name = args.baseline_name or variants[0].name
    if baseline_name not in variant_names:
        raise ValueError(f"Unknown baseline variant: {baseline_name!r}")

    common_engine_args = parse_extra_args(args.common_engine_args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    vllm_command = resolve_vllm_command(args.vllm_command)
    print(f"[info] Using vLLM command: {shlex.join(vllm_command)}")

    env = os.environ.copy()
    if args.cuda_visible_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    records: list[dict[str, Any]] = []

    for index, variant in enumerate(variants):
        port = args.base_port + index
        variant_dir = args.output_dir / variant.name
        variant_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n=== Running variant: {variant.name} (port {port}) ===")

        mm_result_path = variant_dir / "mm_processor.json"
        mm_log_path = variant_dir / "mm_processor.log"
        serve_result_path = variant_dir / "serve.json"
        serve_bench_log_path = variant_dir / "serve_bench.log"
        server_log_path = variant_dir / "server.log"

        mm_result: dict[str, Any] | None = None
        serve_result: dict[str, Any] | None = None

        if not args.skip_mm_processor:
            mm_cmd = build_mm_processor_cmd(
                vllm_command=vllm_command,
                args=args,
                model=args.model,
                common_engine_args=common_engine_args,
                variant_engine_args=variant.extra_engine_args,
                output_json=mm_result_path,
            )
            run_command(mm_cmd, log_path=mm_log_path, env=env)
            mm_result = load_json(mm_result_path)

        server_process: subprocess.Popen[str] | None = None
        try:
            if not args.skip_serve:
                server_process = start_server(
                    vllm_command=vllm_command,
                    model=args.model,
                    port=port,
                    common_engine_args=common_engine_args,
                    variant_engine_args=variant.extra_engine_args,
                    env=env,
                    log_path=server_log_path,
                    timeout_sec=args.startup_timeout_sec,
                    host=args.host,
                )
                serve_cmd = build_serve_bench_cmd(
                    vllm_command=vllm_command,
                    args=args,
                    model=args.model,
                    host=args.host,
                    port=port,
                    variant_dir=variant_dir,
                    result_filename=serve_result_path.name,
                )
                run_command(serve_cmd, log_path=serve_bench_log_path, env=env)
                serve_result = load_json(serve_result_path)
        finally:
            if server_process is not None:
                stop_server(server_process)

        metrics: dict[str, float | None] = {}
        if mm_result is not None:
            metrics.update(
                extract_mm_metrics(mm_result, percentile=args.percentile_for_summary)
            )
        if serve_result is not None:
            metrics.update(
                extract_serve_metrics(serve_result, percentile=args.percentile_for_summary)
            )

        records.append(
            {
                "variant": variant.name,
                "port": port,
                "mm_result_json": str(mm_result_path) if mm_result is not None else None,
                "serve_result_json": (
                    str(serve_result_path) if serve_result is not None else None
                ),
                **metrics,
            }
        )

    baseline_record = next(record for record in records if record["variant"] == baseline_name)
    baseline_metrics = {
        key: value
        for key, value in baseline_record.items()
        if isinstance(value, (float, int)) or value is None
    }

    summary_rows: list[dict[str, Any]] = []
    for record in records:
        metrics = {
            key: value
            for key, value in record.items()
            if isinstance(value, (float, int)) or value is None
        }
        speedups = compute_speedups(baseline_metrics, metrics)
        summary_rows.append({**record, **speedups})

    summary_json_path = args.output_dir / "summary.json"
    summary_csv_path = args.output_dir / "summary.csv"
    summary_payload = {
        "model": args.model,
        "baseline_name": baseline_name,
        "metric_percentiles": args.metric_percentiles,
        "percentile_for_summary": args.percentile_for_summary,
        "vllm_command": vllm_command,
        "common_engine_args": common_engine_args,
        "variants": [
            {"name": variant.name, "extra_engine_args": variant.extra_engine_args}
            for variant in variants
        ],
        "rows": summary_rows,
    }
    summary_json_path.write_text(
        json.dumps(summary_payload, indent=2),
        encoding="utf-8",
    )
    write_summary_csv(summary_rows, summary_csv_path)

    print("\n=== Experiment complete ===")
    print(f"Summary JSON: {summary_json_path}")
    print(f"Summary CSV:  {summary_csv_path}")


if __name__ == "__main__":
    main()
