import argparse
import csv
import re
from pathlib import Path


TIMESTAMP_PATTERN = re.compile(r"^\d{8}_\d{6}$")
METRIC_PATTERN = re.compile(
    r"Avg prompt throughput: (?P<prompt_tp>[\d.]+) tokens/s, "
    r"Avg generation throughput: (?P<gen_tp>[\d.]+) tokens/s, "
    r"Running: (?P<running>\d+) reqs, "
    r"Waiting: (?P<waiting>\d+) reqs, "
    r"GPU KV cache usage: (?P<gpu_kv>[\d.]+)%, "
    r"Prefix cache hit rate: (?P<prefix_hit>[\d.]+)%, "
    r"MM cache hit rate: (?P<mm_hit>[\d.]+)%"
)
HEADER_MAPPING = {
    "prompt_tp": "Avg prompt throughput (token/s)",
    "gen_tp": "Avg generation throughput (token/s)",
    "running": "Running (reqs)",
    "waiting": "Waiting (reqs)",
    "gpu_kv": "GPU KV cache usage (%)",
    "prefix_hit": "Prefix cache hit rate (%)",
    "mm_hit": "MM cache hit rate (%)",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Aggregate benchmark metrics from epdtest log directories.",
    )
    parser.add_argument(
        "root",
        nargs="?",
        default=".",
        help="Root logs directory to scan recursively. Defaults to current directory.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="aggregated_metrics.csv",
        help="Output CSV path. Defaults to aggregated_metrics.csv.",
    )
    return parser.parse_args()


def resolve_run_info(log_path: Path, root: Path) -> tuple[str, str]:
    rel_parts = log_path.relative_to(root).parts
    timestamp = ""
    variant = ""

    for idx, part in enumerate(rel_parts):
        if TIMESTAMP_PATTERN.match(part):
            timestamp = part
            remainder = rel_parts[idx + 1:-1]
            if remainder:
                variant = "/".join(remainder)
            break

    if not timestamp:
        timestamp = "unknown"

    return timestamp, variant


def parse_logs(root: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []

    if not root.exists():
        print(f"Error: Directory '{root}' not found.")
        return rows

    log_paths = sorted(root.rglob("target_script.log"))
    if not log_paths:
        log_paths = sorted(root.rglob("*.log"))

    for log_path in log_paths:
        timestamp, variant = resolve_run_info(log_path, root)
        with log_path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                metric_match = METRIC_PATTERN.search(line)
                if not metric_match:
                    continue

                raw_data = metric_match.groupdict()
                formatted_entry = {
                    "Log Group Timestamp": timestamp,
                    "Run Variant": variant,
                    "Source File": str(log_path.relative_to(root)),
                }
                for key, label in HEADER_MAPPING.items():
                    formatted_entry[label] = raw_data[key]
                rows.append(formatted_entry)

    return rows


def save_to_csv(rows: list[dict[str, str]], output_path: Path):
    if not rows:
        print("No matching log data found to save.")
        return

    fieldnames = [
        "Log Group Timestamp",
        "Run Variant",
        "Source File",
        "Avg prompt throughput (token/s)",
        "Avg generation throughput (token/s)",
        "Running (reqs)",
        "Waiting (reqs)",
        "GPU KV cache usage (%)",
        "Prefix cache hit rate (%)",
        "MM cache hit rate (%)",
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for entry in sorted(rows, key=lambda item: (
            item["Log Group Timestamp"],
            item["Run Variant"],
            item["Source File"],
        )):
            writer.writerow(entry)


if __name__ == "__main__":
    args = parse_args()
    root = Path(args.root).resolve()
    output_path = Path(args.output).resolve()
    rows = parse_logs(root)
    save_to_csv(rows, output_path)
    if rows:
        print(f"Done! Results written to {output_path}")
