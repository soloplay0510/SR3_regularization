import re
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Match the log line; be permissive with spacing and iter formatting
LOG_LINE = re.compile(
    r"""^(?P<ts>\d{2}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\.\d{3})\s+-\s+INFO:\s+
        <\s*epoch:\s*(?P<epoch>\d+)\s*,\s*iter:\s*(?P<iter>[^>]+)\s*>\s+
        (?P<metrics>.+)$
    """,
    re.VERBOSE,
)

# key: value with scientific/decimal numbers, e.g. total: 1.4575e+05
PAIR = re.compile(r"([A-Za-z0-9_]+):\s*([-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?)")

def parse_log(log_path):
    rows = []
    total_epoch_lines = 0
    unmatched_lines = []

    with open(log_path, "r", encoding="utf-8") as f:
        for raw in f:
            s = raw.strip()
            if "<epoch:" not in s:
                continue
            total_epoch_lines += 1
            m = LOG_LINE.search(s)
            if not m:
                unmatched_lines.append(s)
                continue

            epoch = int(m.group("epoch"))
            # strip EVERYTHING not a digit from the iter field (handles commas, thin spaces, etc.)
            iter_raw = m.group("iter")
            iter_digits = re.sub(r"\D", "", iter_raw)
            if not iter_digits:
                unmatched_lines.append(s)
                continue
            it = int(iter_digits)

            metrics = {}
            for key, val, *_ in PAIR.findall(m.group("metrics")):
                try:
                    metrics[key] = float(val)
                except ValueError:
                    pass

            rows.append({"epoch": epoch, "iter": it, **metrics})

    if not rows:
        raise RuntimeError("No matching log lines found. Check the regex or example format.")

    print(f"Found {total_epoch_lines} lines containing '<epoch:', matched {len(rows)}.")
    if unmatched_lines:
        print(f"Unmatched example (first 3):")
        for u in unmatched_lines[:3]:
            print("  >>", u)

    df = pd.DataFrame(rows).sort_values(["epoch", "iter"]).reset_index(drop=True)
    df["step"] = range(1, len(df) + 1)
    return df

def plot_losses(df, metrics=None, out_path="plots/loss_curves.png", title=None):
    if metrics is None:
        exclude = {"epoch", "iter", "step"}
        metrics = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
        priority = ["total", "loss", "loss_noise", "loss_TV1", "loss_TV2"]
        metrics = list(dict.fromkeys([*priority, *metrics]))

    plt.figure(figsize=(9, 5))
    for m in metrics:
        if m in df.columns:
            plt.plot(df["step"], df[m], label=m)
    # X ticks only at epoch starts
    epoch_starts = df.groupby("epoch")["step"].min().tolist()
    epoch_labels = [str(e) for e in sorted(df["epoch"].unique())]
    plt.xticks(epoch_starts, epoch_labels)

    plt.xlabel("Epoch")
    # plt.xscale('log')


    plt.ylabel("Loss")
    plt.title(title or "Loss components vs iterations (x ticks = epochs)")
    plt.legend(loc="best")
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved plot to {out_path}")

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse training log and plot losses.")
    parser.add_argument("log_file", help="Path to the log file")
    parser.add_argument("--out_csv", default="plots/losses_parsed.csv", help="Output CSV file")
    parser.add_argument("--out_png", default="plots/loss_curves.png", help="Output plot file")
    args = parser.parse_args()

    df = parse_log(args.log_file)
    print(f"Parsed {len(df)} rows from {args.log_file}")
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    print(f"Saved parsed losses to {args.out_csv}")

    plot_losses(df, out_path=args.out_png)

