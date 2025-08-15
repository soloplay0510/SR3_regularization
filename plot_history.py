"""
plot_history.py
================

This script reads one or more log files containing model evaluation history and
produces plots of PSNR/SSIM values versus epoch and iteration counts. Each
log line is expected to follow a pattern similar to the following example:

    25-08-11 19:17:24.643 - INFO: <epoch: 47, iter: 350,000> ssim: 6.9082e-01

The script will extract the epoch, iteration (with commas removed) and the
numeric metric value following either the ``ssim`` or ``psnr`` label. It
supports reading multiple files at once (three are typical) and will overlay
curves from each file on the same axes for easy comparison.

Usage
-----

Run the script from the command line, passing one or more log file paths:

.. code:: bash

    python plot_history.py run1.log run2.log run3.log

The resulting figure will display a row for each metric present in the logs
(``ssim`` and/or ``psnr``). For each metric, the left column shows the metric
value versus epoch, while the right column shows the metric value versus
iteration. Each line corresponds to one input file and is labelled by the
file's base name.

If an input file contains both SSIM and PSNR entries, both will be plotted.

Dependencies
------------

This script requires ``matplotlib`` and ``numpy``. Both are installed by
default in most Python environments, but if not, they can be installed via
``pip install matplotlib numpy``.
"""

import argparse
import os
import re
from collections import defaultdict
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt


def parse_log_file(filepath: str) -> Dict[str, Dict[str, List]]:
    """Parse a single log file for metrics.

    Parameters
    ----------
    filepath : str
        Path to the log file to parse.

    Returns
    -------
    Dict[str, Dict[str, List]]
        A nested dictionary keyed first by metric name (e.g. ``ssim`` or
        ``psnr``), then by the list of epochs, iterations and metric values.
        For example::

            {
                'ssim': {
                    'epochs': [1, 2, 3],
                    'iters': [1000, 2000, 3000],
                    'values': [0.67, 0.70, 0.72]
                },
                'psnr': {
                    ...
                }
            }

    Notes
    -----
    Only lines that match the expected pattern are parsed; all others are
    silently ignored.
    """
    # Regular expression to capture epoch, iteration, metric name and value.
    # It looks for patterns like `<epoch: 47, iter: 350,000> ssim: 6.9082e-01`.
    pattern = re.compile(
        r"<epoch:\s*(\d+),\s*iter:\s*([0-9,]+)>\s*(ssim|psnr):\s*([\deE+\-.]+)"
    )

    metrics_data: Dict[str, Dict[str, List]] = {}

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            match = pattern.search(line)
            if not match:
                continue  # skip lines that don't conform

            epoch_str, iter_str, metric_name, value_str = match.groups()

            # Convert strings to appropriate types
            epoch = int(epoch_str)
            iteration = int(iter_str.replace(",", ""))
            value = float(value_str)

            # Initialize nested dicts on first encounter
            if metric_name not in metrics_data:
                metrics_data[metric_name] = {
                    "epochs": [],
                    "iters": [],
                    "values": [],
                }

            metrics_data[metric_name]["epochs"].append(epoch)
            metrics_data[metric_name]["iters"].append(iteration)
            metrics_data[metric_name]["values"].append(value)

    return metrics_data


def aggregate_metrics(files_data: Dict[str, Dict[str, Dict[str, List]]]) -> List[str]:
    """Get a sorted list of all metric names present across all files.

    Parameters
    ----------
    files_data : Dict[str, Dict[str, Dict[str, List]]]
        A mapping from file base name to its parsed metrics data.

    Returns
    -------
    List[str]
        Sorted list of unique metric names.
    """
    metrics = set()
    for data in files_data.values():
        metrics.update(data.keys())
    return sorted(metrics)


def sort_by_key(keys: List[int], values: List[float]) -> Tuple[List[int], List[float]]:
    """Sort two parallel lists based on the first (keys) and return sorted copies.

    Parameters
    ----------
    keys : List[int]
        Values to sort by (e.g. epochs or iterations).
    values : List[float]
        Corresponding metric values to sort in the same order.

    Returns
    -------
    Tuple[List[int], List[float]]
        The sorted keys and values.
    """
    combined = list(zip(keys, values))
    combined.sort(key=lambda pair: pair[0])
    sorted_keys, sorted_values = zip(*combined)
    return list(sorted_keys), list(sorted_values)


def plot_metrics(
    files_data: Dict[str, Dict[str, Dict[str, List]]],
    output_path: str = None,
    epoch_only: bool = False,
) -> None:
    """Create figures plotting metrics from multiple files.

    By default, this function produces two subplots per metric: one plotting the
    metric versus epoch and another plotting the metric versus iteration. When
    ``epoch_only`` is set to ``True``, only the epoch-based plots are
    generated (one subplot per metric).

    Parameters
    ----------
    files_data : Dict[str, Dict[str, Dict[str, List]]]
        Parsed metric data keyed by file base name.
    output_path : str, optional
        If provided, the figure will be saved to this path instead of being
        displayed interactively. When ``epoch_only`` is ``True`` the figure
        will contain one column; otherwise it will contain two columns.
    epoch_only : bool, default False
        If True, plots only metric versus epoch (no iteration plots).
    """
    # Determine which metrics are present
    metric_names = aggregate_metrics(files_data)
    if not metric_names:
        raise RuntimeError("No valid metrics found in the provided logs.")

    n_metrics = len(metric_names)

    if epoch_only:
        # One column: metric versus epoch
        fig, axes = plt.subplots(n_metrics, 1, figsize=(8, 4 * n_metrics))
        # Normalize axes to a 2D-like structure for consistency
        if n_metrics == 1:
            axes = [axes]
        for row_idx, metric in enumerate(metric_names):
            ax = axes[row_idx]
            for fname, mdata in files_data.items():
                if metric not in mdata:
                    continue
                epochs = mdata[metric]["epochs"]
                values = mdata[metric]["values"]
                sorted_epochs, sorted_values = sort_by_key(epochs, values)
                ax.plot(
                    sorted_epochs,
                    sorted_values,
                    # marker="o",
                    label=fname,
                    linestyle="-",
                )
            metric_name_upper = metric.upper()
            ax.set_title(f"{metric_name_upper} vs Epoch")
            ax.set_xlabel("Epoch")
            ax.set_ylabel(metric_name_upper)
            ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
            ax.legend()
        fig.tight_layout()
    else:
        # Two columns: metric versus epoch and iteration
        fig, axes = plt.subplots(n_metrics, 2, figsize=(12, 4 * n_metrics))
        # Normalize to 2D structure
        if n_metrics == 1:
            axes = axes.reshape(1, -1)
        for row_idx, metric in enumerate(metric_names):
            # Left column: metric versus epoch
            ax_epoch = axes[row_idx][0]
            # Right column: metric versus iteration
            ax_iter = axes[row_idx][1]
            for fname, mdata in files_data.items():
                if metric not in mdata:
                    continue
                epochs = mdata[metric]["epochs"]
                iters = mdata[metric]["iters"]
                values = mdata[metric]["values"]
                # Sort for neat plotting
                sorted_epochs, sorted_values_epoch = sort_by_key(epochs, values)
                sorted_iters, sorted_values_iter = sort_by_key(iters, values)
                # Plot against epochs
                ax_epoch.plot(
                    sorted_epochs,
                    sorted_values_epoch,
                    # marker="o",
                    label=fname,
                    linestyle="-",
                )
                # Plot against iterations
                ax_iter.plot(
                    sorted_iters,
                    sorted_values_iter,
                    # marker="o",
                    label=fname,
                    linestyle="-",
                )
            metric_name_upper = metric.upper()
            ax_epoch.set_title(f"{metric_name_upper} vs Epoch")
            ax_epoch.set_xlabel("Epoch")
            ax_epoch.set_ylabel(metric_name_upper)
            ax_epoch.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
            ax_epoch.legend()
            ax_iter.set_title(f"{metric_name_upper} vs Iteration")
            ax_iter.set_xlabel("Iteration")
            ax_iter.set_ylabel(metric_name_upper)
            ax_iter.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
            ax_iter.legend()
        fig.tight_layout()
    # Save or show
    if output_path:
        plt.savefig(output_path, dpi=300)
        print(f"Figure saved to {output_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Plot PSNR/SSIM metrics against epochs and iterations for multiple log files. "
            "Each log line should include an epoch and iteration specification, and a "
            "metric name (ssim or psnr) followed by its value."
        )
    )
    parser.add_argument(
        "logfiles",
        nargs="+",
        help="Paths to log files containing evaluation history.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help=(
            "Optional path to save the generated plot as an image file (e.g. 'plot.png'). "
            "If omitted, the plot will be shown interactively."
        ),
    )
    parser.add_argument(
        "--epoch-only",
        action="store_true",
        help=(
            "Plot only metric versus epoch (omit iteration plots). When specified, the output "
            "figure will contain one column per metric instead of two."
        ),
    )

    parser.add_argument(
        "--max-epoch",
        type=int,
        default=None,
        help=(
            "If provided, only data points with epoch less than or equal to this value will be "
            "plotted. This can be used to truncate the curves at a particular epoch."
        ),
    )
    args = parser.parse_args()

    files_data: Dict[str, Dict[str, Dict[str, List]]] = {}
    for filepath in args.logfiles:
        if not os.path.isfile(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
        # Use the base name for legend labels
        base_name = os.path.basename(filepath)
        parsed = parse_log_file(filepath)
        # Optionally filter by max epoch
        if args.max_epoch is not None:
            for metric_name, mdata in parsed.items():
                # Zip up the lists and filter
                filtered = [
                    (e, i, v)
                    for e, i, v in zip(
                        mdata["epochs"], mdata["iters"], mdata["values"]
                    )
                    if e <= args.max_epoch
                ]
                if filtered:
                    epochs, iters, values = zip(*filtered)
                    parsed[metric_name]["epochs"] = list(epochs)
                    parsed[metric_name]["iters"] = list(iters)
                    parsed[metric_name]["values"] = list(values)
                else:
                    # If no data remains for this metric, clear it entirely
                    parsed[metric_name]["epochs"] = []
                    parsed[metric_name]["iters"] = []
                    parsed[metric_name]["values"] = []
        files_data[base_name] = parsed

    # Remove any metrics that have become empty after filtering
    # (This prevents empty plots from being drawn.)
    for fname, mdata in list(files_data.items()):
        metrics_to_delete = [
            metric_name
            for metric_name, info in mdata.items()
            if not info["epochs"]
        ]
        for metric_name in metrics_to_delete:
            del files_data[fname][metric_name]
    
    plot_metrics(files_data, output_path=args.output, epoch_only=args.epoch_only)


if __name__ == "__main__":
    main()