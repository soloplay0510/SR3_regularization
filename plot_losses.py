import re
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import os

# --------- parsing ---------
LOG_LINE = re.compile(
    r"""^(?P<ts>\d{2}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\.\d{3})\s+-\s+INFO:\s+
        <\s*epoch:\s*(?P<epoch>\d+)\s*,\s*iter:\s*(?P<iter>[^>]+)\s*>\s+
        (?P<metrics>.+)$
    """,
    re.VERBOSE,
)
PAIR = re.compile(r"([A-Za-z0-9_]+):\s*([-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?)")

def parse_log(log_path):
    rows, total_epoch_lines, unmatched = [], 0, []
    with open(log_path, "r", encoding="utf-8") as f:
        for raw in f:
            s = raw.strip()
            if "<epoch:" not in s:
                continue
            total_epoch_lines += 1
            m = LOG_LINE.search(s)
            if not m:
                unmatched.append(s); continue
            epoch = int(m.group("epoch"))
            iter_digits = re.sub(r"\D", "", m.group("iter"))
            if not iter_digits:
                unmatched.append(s); continue
            it = int(iter_digits)
            metrics = {}
            for key, val, *_ in PAIR.findall(m.group("metrics")):
                try: metrics[key] = float(val)
                except ValueError: pass
            rows.append({"epoch": epoch, "iter": it, **metrics})
    if not rows:
        raise RuntimeError(f"No matching lines in {log_path}")
    print(f"[{log_path}] Found {total_epoch_lines} lines with '<epoch:', matched {len(rows)}.")
    if unmatched:
        print("  Unmatched example(s):")
        for u in unmatched[:3]: print("   >>", u)
    df = pd.DataFrame(rows).sort_values(["epoch", "iter"]).reset_index(drop=True)
    df["step"] = range(1, len(df) + 1)
    return df

# --------- plotting ---------
def _auto_metric_list(df):
    exclude = {"epoch", "iter", "step"}
    metrics = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
    priority = ["total", "loss", "loss_noise", "loss_TV1", "loss_TV2", "loss_TVF", "loss_wave_l1"]
    return list(dict.fromkeys([*priority, *metrics]))

def plot_losses_multi(
    dfs,
    labels,
    metrics=None,
    out_path="plots/loss_curves.png",
    title=None,
    xaxis="iter",
    smooth=0,
    yscale="linear",
    ybase=10.0,
    linthresh=1e-3,
    linscale=1.0,
    ymin=None,
    ymax=None,
    xmin=None,
    xmax=None,
):
    # metrics default: union across all dfs
    if metrics is None:
        union = []
        for df in dfs:
            for m in _auto_metric_list(df):
                if m not in union: union.append(m)
        metrics = union

    def xvec(df):
        if xaxis == "iter": return df["iter"].values
        if xaxis == "step": return df["step"].values
        if xaxis == "epoch": return df["epoch"].values
        raise ValueError("--xaxis must be one of iter|step|epoch")

    plt.figure(figsize=(10, 6))
    for df, lab in zip(dfs, labels):
        xv = xvec(df)
        for m in metrics:
            if m in df.columns:
                y = df[m].values
                if isinstance(smooth, int) and smooth > 1:
                    y = (pd.Series(y).rolling(window=smooth, min_periods=1, center=True).mean().to_numpy())
                plt.plot(xv, y, label=f"{lab}:{m}")

    plt.xlabel({"iter":"Iteration","step":"Step","epoch":"Epoch"}[xaxis])
    plt.ylabel("Loss / Metric value")
    plt.title(title or f"Metrics vs {xaxis}")

    if ymin is not None or ymax is not None:
        plt.ylim(bottom=ymin, top=ymax)
    if xmin is not None or xmax is not None:
        plt.xlim(left=xmin, right=xmax)

    if yscale == "log":
        plt.yscale("log", base=ybase)
    elif yscale == "symlog":
        plt.yscale("symlog", linthresh=linthresh, linscale=linscale, base=ybase)
    else:
        plt.yscale("linear")

    plt.legend(loc="best", fontsize=9)
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=220)
    plt.close()
    print(f"Saved plot to {out_path}")

# --------- cli ---------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse training log(s) and plot losses/metrics.")
    parser.add_argument("log_files", nargs="+", help="Path(s) to log file(s) (1 or more).")
    parser.add_argument("--labels", nargs="+", default=None,
                        help="Labels for each log (default: basename).")
    parser.add_argument("--out_csv", default="plots/losses_parsed.csv",
                        help="Output CSV (single log) or base path for multiple.")
    parser.add_argument("--out_png", default="plots/loss_curves.png", help="Output plot file")
    parser.add_argument("--metrics", nargs="+", default=["total","loss","loss_noise","loss_TV1","loss_TV2","loss_TVF","loss_wave_l1"],
                        help="Metrics to plot (space separated).")
    parser.add_argument("--xaxis", choices=["iter","step","epoch"], default="iter",
                        help="X-axis for plotting. Default: iter (best for comparing logs).")
    parser.add_argument("--smooth", type=int, default=0,
                        help="Centered moving-average window in steps (0/1 = off).")

    # Y-axis controls
    parser.add_argument("--yscale", choices=["linear","log","symlog"], default="linear",
                        help="Y-axis scale.")
    parser.add_argument("--ybase", type=float, default=10.0,
                        help="Log base for log/symlog (default 10).")
    parser.add_argument("--linthresh", type=float, default=1e-3,
                        help="symlog: linear range around 0.")
    parser.add_argument("--linscale", type=float, default=1.0,
                        help="symlog: length of the linear region in decades.")
    parser.add_argument("--ymin", type=float, default=None,
                        help="Lower Y limit.")
    parser.add_argument("--ymax", type=float, default=None,
                        help="Upper Y limit.")

    # NEW: iteration filtering and x-axis clipping
    parser.add_argument("--iter-min", type=int, default=None,
                        help="Keep only rows with iter >= this value.")
    parser.add_argument("--iter-max", type=int, default=None,
                        help="Keep only rows with iter <= this value.")
    parser.add_argument("--xmin", type=float, default=None,
                        help="Left X limit (visual only).")
    parser.add_argument("--xmax", type=float, default=None,
                        help="Right X limit (visual only).")

    # Optional helper: list metrics
    parser.add_argument("--list-metrics", action="store_true",
                        help="Print numeric metric keys found in each log and exit.")

    args = parser.parse_args()

    if args.iter_min is not None and args.iter_max is not None and args.iter_min > args.iter_max:
        raise SystemExit("--iter-min must be <= --iter-max")

    # Parse logs
    dfs = [parse_log(p) for p in args.log_files]

    # Apply iteration filtering (recompute step after)
    def _filter_iter(df, imin, imax):
        if imin is not None:
            df = df[df["iter"] >= imin]
        if imax is not None:
            df = df[df["iter"] <= imax]
        df = df.sort_values(["epoch", "iter"]).reset_index(drop=True)
        df["step"] = range(1, len(df) + 1)
        return df

    dfs = [_filter_iter(df, args.iter_min, args.iter_max) for df in dfs]

    # Labels
    if args.labels:
        if len(args.labels) != len(dfs):
            raise SystemExit("Number of --labels must match number of log files.")
        labels = args.labels
    else:
        labels = [Path(p).name for p in args.log_files]

    # List metrics and exit, if requested
    if args.list_metrics:
        for path, df, lab in zip(args.log_files, dfs, labels):
            cols = [c for c in df.columns
                    if c not in ("epoch", "iter", "step")
                    and pd.api.types.is_numeric_dtype(df[c])]
            print(f"[{lab}] {', '.join(cols) if cols else '(none)'}")
        raise SystemExit(0)

    # Save CSV(s)
    out_csv_path = Path(args.out_csv)
    if len(dfs) == 1:
        out_csv_path.parent.mkdir(parents=True, exist_ok=True)
        dfs[0].to_csv(out_csv_path, index=False)
        print(f"Saved parsed CSV to {out_csv_path}")
    else:
        base = out_csv_path.stem
        ext = out_csv_path.suffix or ".csv"
        out_dir = out_csv_path.parent if out_csv_path.parent.as_posix() not in ("", ".") else Path("plots")
        out_dir.mkdir(parents=True, exist_ok=True)
        for lab, df in zip(labels, dfs):
            safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", lab)
            df.to_csv(out_dir / f"{base}_{safe}{ext}", index=False)
            print(f"Saved parsed CSV to {out_dir / f'{base}_{safe}{ext}'}")

    # Plot
    plot_losses_multi(
        dfs, labels,
        metrics=args.metrics,
        out_path=args.out_png,
        xaxis=args.xaxis,
        smooth=args.smooth,
        yscale=args.yscale,
        ybase=args.ybase,
        linthresh=args.linthresh,
        linscale=args.linscale,
        ymin=args.ymin,
        ymax=args.ymax,
        xmin=args.xmin,
        xmax=args.xmax,
    )
