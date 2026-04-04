import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MaxNLocator


METRIC_COLUMNS = {
    "power_draw_watts": "Power Draw (W)",
    "memory_utilization_pct": "Memory Utilization (%)",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot GPU utilization logs for one role across two utilization metrics."
    )
    parser.add_argument(
        "input_log",
        help="Path to the CSV-like GPU utilization log.",
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="Path to the output image file.",
    )
    parser.add_argument(
        "--role",
        default="prefill_decode",
        help="Role to plot. Defaults to prefill_decode.",
    )
    parser.add_argument(
        "--figsize",
        nargs=2,
        type=float,
        default=(14, 4),
        metavar=("WIDTH", "HEIGHT"),
        help="Figure size in inches. Defaults to 14 6.",
    )
    parser.add_argument(
        "--title",
        help="Optional figure title.",
    )
    return parser.parse_args()


def load_log(input_path: Path) -> pd.DataFrame:
    df = pd.read_csv(input_path)
    required_columns = {"timestamp", "role", *METRIC_COLUMNS.keys()}
    missing = sorted(required_columns - set(df.columns))
    if missing:
        raise ValueError(f"Missing required column(s): {', '.join(missing)}")

    df = df.copy()
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    for metric in METRIC_COLUMNS:
        df[metric] = pd.to_numeric(df[metric], errors="coerce")

    df = df.dropna(subset=["timestamp"])
    df = df.sort_values(["timestamp", "role"]).reset_index(drop=True)
    return df


def format_role_label(role: str) -> str:
    return role.replace("_", " ").title()


def plot_role_metric(ax, role_df: pd.DataFrame, role: str, metric: str, ylabel: str):
    x_positions = range(len(role_df))
    metric_series = role_df[metric].fillna(0)
    values = metric_series.tolist()
    if metric == "power_draw_watts":
        avg_series = metric_series[metric_series >= 100]
    else:
        avg_series = metric_series[metric_series != 0]
    avg_value = avg_series.mean() if not avg_series.empty else 0.0

    bar_color = "#F4A261" if metric == "power_draw_watts" else "#2A9D8F"
    edge_color = "#BC6C25" if metric == "power_draw_watts" else "#1B4332"

    ax.bar(
        x_positions,
        values,
        width=0.6,
        color=bar_color,
        edgecolor=edge_color,
        linewidth=0.2,
        alpha=0.85,
    )

    ax.set_xticks([])

    avg_suffix = "W" if metric == "power_draw_watts" else "%"
    ax.set_title(f"{role} - {ylabel} (avg: {avg_value:.1f}{avg_suffix})")
    ax.set_xlabel("time")
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 600 if metric == "power_draw_watts" else 100)
    ax.margins(x=0.01)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def main():
    args = parse_args()
    input_path = Path(args.input_log)
    output_path = Path(args.output)

    df = load_log(input_path)
    role = args.role

    available_roles = set(df["role"].astype(str))
    if role not in available_roles:
        raise ValueError(
            f"Requested role not found: {role}. "
            f"Available roles: {', '.join(sorted(available_roles))}"
        )

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=tuple(args.figsize), sharex=False)
    fig.patch.set_facecolor("#F8F9FA")

    role_df = df[df["role"] == role].copy()
    role_df = role_df.sort_values("timestamp")

    for col_idx, (metric, ylabel) in enumerate(METRIC_COLUMNS.items()):
        plot_role_metric(
            axes[col_idx],
            role_df,
            format_role_label(role),
            metric,
            ylabel,
        )

    if args.title:
        fig.suptitle(args.title)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
    else:
        fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
