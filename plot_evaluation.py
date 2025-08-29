#!/usr/bin/env python3
import argparse
import os
import re
from typing import Dict, List, Tuple
import numpy as np
import matplotlib.pyplot as plt
import csv

# ---------- Regex ----------
PAT = re.compile(
    r"<epoch:\s*(\d+),\s*iter:\s*([0-9,]+)>\s*(ssim|psnr):\s*([\deE+\-.]+)"
)

# ---------- Parsing ----------
def parse_log_file(path: str) -> Dict[str, Dict[str, List]]:
    """Return {'psnr': {'epochs': [], 'iters': [], 'values': []}, 'ssim': {...}}."""
    data: Dict[str, Dict[str, List]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            m = PAT.search(line)
            if not m:
                continue
            epoch_s, iter_s, metric, val_s = m.groups()
            epoch = int(epoch_s)
            iters = int(iter_s.replace(",", ""))
            val = float(val_s)
            if metric not in data:
                data[metric] = {"epochs": [], "iters": [], "values": []}
            data[metric]["epochs"].append(epoch)
            data[metric]["iters"].append(iters)
            data[metric]["values"].append(val)
    return data

def aggregate_metrics(files_data: Dict[str, Dict[str, Dict[str, List]]]) -> List[str]:
    s = set()
    for d in files_data.values():
        s.update(d.keys())
    return sorted(s)

def sort_by_key(keys: List[int], values: List[float]) -> Tuple[List[int], List[float]]:
    if not keys:
        return [], []
    arr = sorted(zip(keys, values), key=lambda t: t[0])
    k, v = zip(*arr)
    return list(k), list(v)

def moving_average(y: List[float], k: int) -> List[float]:
    if k is None or k <= 1 or len(y) == 0:
        return y
    k = min(k, len(y))
    kern = np.ones(k) / k
    return np.convolve(np.array(y, dtype=float), kern, mode="same").tolist()

# ---------- Plot helpers ----------
def annotate_best(ax, xs, ys, fmt="{:.3f}", dx=3, dy_ratio=0.02):
    if not xs:
        return
    i = int(np.argmax(ys))
    ax.scatter([xs[i]], [ys[i]], s=80, zorder=5)
    dy = (max(ys) - min(ys) + 1e-9) * dy_ratio
    ax.annotate(fmt.format(ys[i]),
                xy=(xs[i], ys[i]),
                xytext=(xs[i] + dx, ys[i] + dy),
                arrowprops=dict(arrowstyle="->"))

# ---------- CSV ----------
def write_csv(out_csv: str, files_data: Dict[str, Dict[str, Dict[str, List]]]):
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "metric", "epoch", "iter", "value"])
        for fname, md in files_data.items():
            for metric, d in md.items():
                for e, it, v in zip(d["epochs"], d["iters"], d["values"]):
                    w.writerow([fname, metric, e, it, v])
    print(f"CSV written to {out_csv}")

# ---------- Plotters ----------
def plot_grid(files_data, metrics, output_path, epoch_only, smooth, mark_best):
    metric_names = metrics
    n = len(metric_names)
    if n == 0:
        raise RuntimeError("No metrics to plot.")
    if epoch_only:
        fig, axes = plt.subplots(n, 1, figsize=(9, 4 * n))
        if n == 1:
            axes = [axes]
        for r, metric in enumerate(metric_names):
            ax = axes[r]
            for fname, mdata in files_data.items():
                if metric not in mdata: 
                    continue
                e = mdata[metric]["epochs"]
                v = mdata[metric]["values"]
                e_s, v_s = sort_by_key(e, moving_average(v, smooth))
                if not e_s:
                    continue
                ax.plot(e_s, v_s, label=fname, linestyle="-", marker=None)
                if mark_best:
                    annotate_best(ax, e_s, v_s, fmt="{:.3f}")
            mu = metric.upper()
            ax.set_title(f"{mu} vs Epoch")
            ax.set_xlabel("Epoch")
            ax.set_ylabel(mu)
            ax.grid(True, linestyle="--", alpha=0.6)
            ax.legend()
        fig.tight_layout()
    else:
        fig, axes = plt.subplots(n, 2, figsize=(14, 4 * n))
        if n == 1:
            axes = axes.reshape(1, -1)
        for r, metric in enumerate(metric_names):
            ax_e = axes[r][0]
            ax_i = axes[r][1]
            for fname, mdata in files_data.items():
                if metric not in mdata:
                    continue
                e = mdata[metric]["epochs"]
                it = mdata[metric]["iters"]
                v = mdata[metric]["values"]
                e_s, v_e = sort_by_key(e, moving_average(v, smooth))
                it_s, v_i = sort_by_key(it, moving_average(v, smooth))
                if e_s:
                    ax_e.plot(e_s, v_e, label=fname, linestyle="-")
                    if mark_best:
                        annotate_best(ax_e, e_s, v_e, fmt="{:.3f}")
                if it_s:
                    ax_i.plot(it_s, v_i, label=fname, linestyle="-")
                    if mark_best:
                        annotate_best(ax_i, it_s, v_i, fmt="{:.3f}")
            mu = metric.upper()
            ax_e.set_title(f"{mu} vs Epoch"); ax_e.set_xlabel("Epoch"); ax_e.set_ylabel(mu); ax_e.grid(True, linestyle="--", alpha=0.6)
            ax_i.set_title(f"{mu} vs Iteration"); ax_i.set_xlabel("Iteration"); ax_i.set_ylabel(mu); ax_i.grid(True, linestyle="--", alpha=0.6)
            ax_e.legend(); ax_i.legend()
        fig.tight_layout()

    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        plt.savefig(output_path, dpi=300)
        print(f"Figure saved to {output_path}")
    else:
        plt.show()

def plot_dual_y(files_data, output_path, smooth, mark_best, labels):
    # Dual-y supports PSNR and SSIM together vs epoch (epoch only).
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("PSNR (dB)", color="tab:blue")
    ax2 = ax1.twinx()
    ax2.set_ylabel("SSIM", color="tab:orange")

    # draw PSNR then SSIM for each file
    for fname in labels:
        mdata = files_data.get(fname, {})
        # PSNR
        if "psnr" in mdata and mdata["psnr"]["epochs"]:
            e = mdata["psnr"]["epochs"]
            v = mdata["psnr"]["values"]
            e_s, v_s = sort_by_key(e, moving_average(v, smooth))
            ax1.plot(e_s, v_s, label=f"PSNR - {fname}")
            if mark_best:
                annotate_best(ax1, e_s, v_s, fmt="{:.2f}", dx=2)
        # SSIM
        if "ssim" in mdata and mdata["ssim"]["epochs"]:
            e = mdata["ssim"]["epochs"]
            v = mdata["ssim"]["values"]
            e_s, v_s = sort_by_key(e, moving_average(v, smooth))
            ax2.plot(e_s, v_s, linestyle="--", marker=None, label=f"SSIM - {fname}")
            if mark_best:
                annotate_best(ax2, e_s, v_s, fmt="{:.3f}", dx=2)

    ax1.grid(True, linestyle="--", alpha=0.6)
    # merge legends
    l1, lab1 = ax1.get_legend_handles_labels()
    l2, lab2 = ax2.get_legend_handles_labels()
    ax1.legend(l1 + l2, lab1 + lab2, loc="best")

    fig.suptitle("PSNR & SSIM vs Epoch (dual y)")
    fig.tight_layout()
    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        plt.savefig(output_path, dpi=300)
        print(f"Figure saved to {output_path}")
    else:
        plt.show()

# ---------- Main ----------
def main():
    p = argparse.ArgumentParser(
        description="Combine PSNR/SSIM plotting: supports epoch/iteration grids or dual-y overlay."
    )
    p.add_argument("logfiles", nargs="+", help="Paths to one or more log files.")
    p.add_argument("--labels", nargs="+",
                   help="Legend labels matching the order of logfiles (default: basenames).")
    p.add_argument("--metrics", nargs="+", choices=["psnr", "ssim"],
                   help="Subset of metrics to plot (default: all found).")
    p.add_argument("--output", type=str, default=None,
                   help="Image output path (e.g., plots/plot.png). If omitted, shows interactively.")
    p.add_argument("--epoch-only", action="store_true",
                   help="Plot only vs epoch (omit iteration column).")
    p.add_argument("--dual-y", action="store_true",
                   help="Use a single chart with PSNR & SSIM on dual y-axes (epoch only).")
    p.add_argument("--smooth", type=int, default=0,
                   help="Moving-average window (e.g., 5). 0/1 disables smoothing.")
    p.add_argument("--mark-best", action="store_true",
                   help="Mark and annotate the best points.")
    p.add_argument("--max-epoch", type=int, default=None,
                   help="Keep only points with epoch <= this value.")
    p.add_argument("--out-csv", type=str, default=None,
                   help="Optional CSV to write all parsed points.")
    args = p.parse_args()

    # Labels
    base_labels = [os.path.basename(f) for f in args.logfiles]
    if args.labels:
        if len(args.labels) != len(args.logfiles):
            p.error("Number of --labels must match number of logfiles.")
        base_labels = args.labels

    # Parse
    files_data: Dict[str, Dict[str, Dict[str, List]]] = {}
    for path, lab in zip(args.logfiles, base_labels):
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        parsed = parse_log_file(path)
        # Filter by max epoch
        if args.max_epoch is not None:
            for mname, d in parsed.items():
                keep = [(e, it, v) for e, it, v in zip(d["epochs"], d["iters"], d["values"]) if e <= args.max_epoch]
                if keep:
                    e, it, v = zip(*keep)
                    parsed[mname]["epochs"] = list(e)
                    parsed[mname]["iters"] = list(it)
                    parsed[mname]["values"] = list(v)
                else:
                    parsed[mname]["epochs"] = []
                    parsed[mname]["iters"] = []
                    parsed[mname]["values"] = []
        files_data[lab] = parsed

    # Determine metrics to plot
    all_metrics = aggregate_metrics(files_data)
    metrics = [m for m in all_metrics if (args.metrics is None or m in args.metrics)]

    # Optional CSV
    if args.out_csv:
        # Flatten to filename keys rather than labels for reproducibility
        write_csv(args.out_csv, files_data)

    # Choose plotter
    if args.dual_y:
        # dual-y implies epoch-only
        plot_dual_y(files_data, args.output, args.smooth, args.mark_best, labels=base_labels)
    else:
        plot_grid(files_data, metrics, args.output, args.epoch_only, args.smooth, args.mark_best)

if __name__ == "__main__":
    main()
