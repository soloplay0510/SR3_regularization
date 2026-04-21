#!/usr/bin/env python3
import argparse
import os
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_sample_indices(text: str) -> List[int]:
    parts = [p.strip() for p in text.split(",") if p.strip()]
    if not parts:
        raise ValueError("No sample indices provided.")
    out = []
    for p in parts:
        v = int(p)
        if v < 1:
            raise ValueError(f"Sample index must be >= 1, got {v}")
        out.append(v)
    return out


def build_long_table(summary_csv: str, metric: str, series_label: str) -> Tuple[pd.DataFrame, List[str]]:
    summary = pd.read_csv(summary_csv)
    metric_upper = metric.upper()
    metric_lower = metric.lower()
    mean_col = f"{metric_upper}_mean"
    std_col = f"{metric_upper}_std"

    rows = []
    label_order = []
    for _, r in summary.iterrows():
        label = str(r["label"])
        label_order.append(label)
        results_dir = str(r["results_dir"])
        stat_csv = os.path.join(results_dir, f"eval_{metric_lower}_statistics.csv")
        if not os.path.isfile(stat_csv):
            raise FileNotFoundError(f"Missing statistics file: {stat_csv}")
        stat_df = pd.read_csv(stat_csv)
        if "sample_index" not in stat_df.columns or mean_col not in stat_df.columns or std_col not in stat_df.columns:
            raise ValueError(
                f"{stat_csv} must contain columns: sample_index, {mean_col}, {std_col}"
            )
        for _, srow in stat_df.iterrows():
            rows.append(
                {
                    "label": label,
                    "checkpoint_epoch": r.get("checkpoint_epoch"),
                    "checkpoint_iter": r.get("checkpoint_iter"),
                    "results_dir": results_dir,
                    "sample_index": int(srow["sample_index"]),
                    "mean": float(srow[mean_col]),
                    "std": float(srow[std_col]),
                    "series": series_label,
                }
            )

    long_df = pd.DataFrame(rows)
    # Keep a stable checkpoint order for plots.
    long_df = long_df.sort_values(
        by=["checkpoint_epoch", "checkpoint_iter", "label"],
        na_position="last",
    ).reset_index(drop=True)
    return long_df, label_order


def plot_per_sample(long_df: pd.DataFrame, metric: str, sample_indices: List[int], out_dir: str, label_order: List[str]) -> None:
    os.makedirs(out_dir, exist_ok=True)
    y_label = metric.upper()
    series_names = long_df["series"].dropna().unique().tolist()

    for sample_idx in sample_indices:
        sub = long_df[long_df["sample_index"] == sample_idx].copy()
        if sub.empty:
            print(f"[WARN] sample_index={sample_idx} not found, skipping.")
            continue

        x = np.arange(len(label_order))

        plt.figure(figsize=(11, 4.5))
        for series_name in series_names:
            series_sub = sub[sub["series"] == series_name][["label", "mean", "std"]].drop_duplicates("label")
            series_sub = series_sub.set_index("label").reindex(label_order)
            y = series_sub["mean"].to_numpy(dtype=float)
            ystd = series_sub["std"].to_numpy(dtype=float)
            plt.plot(x, y, marker="o", linewidth=1.8, label=f"{series_name} mean")
            plt.fill_between(x, y - ystd, y + ystd, alpha=0.18, label=f"{series_name} ±1 std")

        plt.xticks(x, label_order, rotation=45, ha="right")
        plt.xlabel("Checkpoint")
        plt.ylabel(y_label)
        plt.title(f"{y_label} across checkpoints (sample index {sample_idx}, overlaid models)")
        plt.grid(alpha=0.25)
        plt.legend()
        plt.tight_layout()

        out_png = os.path.join(
            out_dir, f"{metric.lower()}_across_checkpoints_sample_{sample_idx}.png"
        )
        plt.savefig(out_png, dpi=160)
        plt.close()
        print(f"Saved: {out_png}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Plot metric mean±std across checkpoints for selected sample indices."
    )
    ap.add_argument(
        "--summary-csv",
        default=None,
        help="Path to one checkpoint_eval_summary.csv (single-series mode)",
    )
    ap.add_argument(
        "--summary-csvs",
        nargs="*",
        default=None,
        help="Multiple summary CSVs for overlaid comparison mode",
    )
    ap.add_argument(
        "--series-labels",
        default=None,
        help='Comma-separated legend labels matching --summary-csvs, e.g. "vgg1E-1,no vgg"',
    )
    ap.add_argument(
        "--metric",
        choices=["psnr", "ssim", "sam"],
        default="psnr",
        help="Metric to plot",
    )
    ap.add_argument(
        "--sample-indices",
        default="1,2,3",
        help="Comma-separated sample indices, e.g. 1,2,3",
    )
    ap.add_argument(
        "--output-dir",
        default=None,
        help="Output folder for generated plots and merged CSV",
    )
    args = ap.parse_args()

    csv_paths = []
    series_labels = []
    if args.summary_csvs:
        csv_paths = list(args.summary_csvs)
        if args.series_labels is None:
            series_labels = [f"series_{i+1}" for i in range(len(csv_paths))]
        else:
            series_labels = [s.strip() for s in args.series_labels.split(",") if s.strip()]
            if len(series_labels) != len(csv_paths):
                raise ValueError("--series-labels count must match --summary-csvs count")
    elif args.summary_csv:
        csv_paths = [args.summary_csv]
        series_labels = ["model"]
    else:
        raise ValueError("Provide --summary-csv or --summary-csvs.")

    out_dir = args.output_dir
    if out_dir is None:
        out_dir = os.path.join(os.path.dirname(csv_paths[0]), "sample_across_checkpoints")

    sample_indices = parse_sample_indices(args.sample_indices)
    all_long = []
    label_order = None
    for i, path in enumerate(csv_paths):
        long_df_i, order_i = build_long_table(path, args.metric, series_labels[i])
        all_long.append(long_df_i)
        if label_order is None:
            label_order = order_i
    long_df = pd.concat(all_long, ignore_index=True)
    # Keep only unique labels in first-seen order.
    label_order = list(dict.fromkeys(label_order))

    os.makedirs(out_dir, exist_ok=True)
    merged_csv = os.path.join(out_dir, f"{args.metric}_across_checkpoints_long.csv")
    long_df.to_csv(merged_csv, index=False)
    print(f"Saved: {merged_csv}")

    plot_per_sample(long_df, args.metric, sample_indices, out_dir, label_order)


if __name__ == "__main__":
    main()
