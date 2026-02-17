#!/usr/bin/env python3
import argparse
import csv
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd


CKPT_GEN_RE = re.compile(r"^I(\d+)_E(\d+)_gen\.pth$")


@dataclass
class EvalResult:
    label: str
    checkpoint_prefix: str
    checkpoint_iter: Optional[int]
    checkpoint_epoch: Optional[int]
    experiment_dir: str
    results_dir: str
    avg_psnr_of_avg_sr: float
    avg_ssim_of_avg_sr: float
    psnr_run_mean_avg: float
    psnr_run_std_avg: float
    ssim_run_mean_avg: float
    ssim_run_std_avg: float


def load_config_with_comments(path: str) -> dict:
    json_str = ""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            json_str += line.split("//")[0] + "\n"
    return json.loads(json_str)


def normalize_checkpoint_prefix(path_or_prefix: str) -> str:
    p = path_or_prefix.strip()
    if p.endswith("_gen.pth"):
        return p[:-8]
    if p.endswith("_opt.pth"):
        return p[:-8]
    return p


def ckpt_sort_key(prefix: str) -> Tuple[int, int, str]:
    base = os.path.basename(prefix)
    m = re.match(r"^I(\d+)_E(\d+)$", base)
    if not m:
        return (10**18, 10**18, base)
    return (int(m.group(2)), int(m.group(1)), base)


def discover_checkpoints(
    checkpoint_dir: str,
    limit: Optional[int],
    pick: str = "last",
    start_index: int = 1,
    index_step: int = 1,
) -> List[str]:
    prefixes: List[str] = []
    for name in os.listdir(checkpoint_dir):
        m = CKPT_GEN_RE.match(name)
        if not m:
            continue
        prefixes.append(os.path.join(checkpoint_dir, name[:-8]))  # remove "_gen.pth"
    prefixes.sort(key=ckpt_sort_key)
    start0 = max(int(start_index) - 1, 0)
    if start0 > 0:
        prefixes = prefixes[start0:]
    step = max(int(index_step), 1)
    prefixes = prefixes[::step]
    if limit is not None:
        if pick == "first":
            prefixes = prefixes[:limit]
        else:
            prefixes = prefixes[-limit:]
    return prefixes


def list_existing_experiment_dirs(prefix_name: str) -> List[str]:
    root = "experiments"
    if not os.path.isdir(root):
        return []
    out = []
    for d in os.listdir(root):
        p = os.path.join(root, d)
        if os.path.isdir(p) and d.startswith(prefix_name + "_"):
            out.append(p)
    out.sort()
    return out


def run_one_checkpoint(
    sr_py: str,
    base_cfg_path: str,
    checkpoint_prefix: str,
    out_dir: str,
    n_run: int,
    data_len: Optional[int],
    gpu_ids: Optional[str],
    mkl_threading_layer: Optional[str],
    idx: int,
) -> EvalResult:
    base_cfg = load_config_with_comments(base_cfg_path)
    label = os.path.basename(checkpoint_prefix)

    run_name = f"{base_cfg['name']}_eval_{idx:02d}_{label}"
    before = set(list_existing_experiment_dirs(run_name))

    cfg = base_cfg
    cfg["phase"] = "val"
    cfg["name"] = run_name
    cfg["path"]["resume_state"] = checkpoint_prefix
    cfg["datasets"]["val"]["n_run"] = int(n_run)
    if data_len is not None:
        cfg["datasets"]["val"]["data_len"] = int(data_len)

    tmp_cfg_path = os.path.join(out_dir, "tmp_configs", f"{run_name}.json")
    os.makedirs(os.path.dirname(tmp_cfg_path), exist_ok=True)
    with open(tmp_cfg_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    cmd = [sys.executable, sr_py, "-p", "val", "-c", tmp_cfg_path]
    if gpu_ids is not None:
        cmd.extend(["-gpu", gpu_ids])
    run_env = os.environ.copy()
    if mkl_threading_layer:
        run_env["MKL_THREADING_LAYER"] = mkl_threading_layer
    subprocess.run(cmd, check=True, env=run_env)

    after = set(list_existing_experiment_dirs(run_name))
    new_dirs = sorted(after - before)
    if not new_dirs:
        raise RuntimeError(f"Could not locate new experiment directory for {label}.")
    experiment_dir = new_dirs[-1]
    results_dir = os.path.join(experiment_dir, "results")

    train_log = os.path.join(experiment_dir, "logs", "train.log")
    avg_psnr = float("nan")
    avg_ssim = float("nan")
    with open(train_log, "r", encoding="utf-8") as f:
        for line in f:
            m_psnr = re.search(r"# Validation # PSNR:\s*([0-9eE+\-.]+)", line)
            if m_psnr:
                avg_psnr = float(m_psnr.group(1))
            m_ssim = re.search(r"# Validation # SSIM:\s*([0-9eE+\-.]+)", line)
            if m_ssim:
                avg_ssim = float(m_ssim.group(1))

    psnr_csv = os.path.join(results_dir, "eval_psnr_statistics.csv")
    ssim_csv = os.path.join(results_dir, "eval_ssim_statistics.csv")
    if not (os.path.isfile(psnr_csv) and os.path.isfile(ssim_csv)):
        raise RuntimeError(f"Missing evaluation CSV files in {results_dir}")

    psnr_df = pd.read_csv(psnr_csv)
    ssim_df = pd.read_csv(ssim_csv)

    m = re.match(r"^I(\d+)_E(\d+)$", label)
    ckpt_iter = int(m.group(1)) if m else None
    ckpt_epoch = int(m.group(2)) if m else None

    return EvalResult(
        label=label,
        checkpoint_prefix=checkpoint_prefix,
        checkpoint_iter=ckpt_iter,
        checkpoint_epoch=ckpt_epoch,
        experiment_dir=experiment_dir,
        results_dir=results_dir,
        avg_psnr_of_avg_sr=avg_psnr,
        avg_ssim_of_avg_sr=avg_ssim,
        psnr_run_mean_avg=float(psnr_df["PSNR_mean"].mean()),
        psnr_run_std_avg=float(psnr_df["PSNR_std"].mean()),
        ssim_run_mean_avg=float(ssim_df["SSIM_mean"].mean()),
        ssim_run_std_avg=float(ssim_df["SSIM_std"].mean()),
    )


def save_summary_csv(rows: List[EvalResult], out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "label",
            "checkpoint_prefix",
            "checkpoint_epoch",
            "checkpoint_iter",
            "experiment_dir",
            "results_dir",
            "avg_psnr_of_avg_sr",
            "avg_ssim_of_avg_sr",
            "psnr_run_mean_avg",
            "psnr_run_std_avg",
            "ssim_run_mean_avg",
            "ssim_run_std_avg",
        ])
        for r in rows:
            w.writerow([
                r.label,
                r.checkpoint_prefix,
                r.checkpoint_epoch,
                r.checkpoint_iter,
                r.experiment_dir,
                r.results_dir,
                r.avg_psnr_of_avg_sr,
                r.avg_ssim_of_avg_sr,
                r.psnr_run_mean_avg,
                r.psnr_run_std_avg,
                r.ssim_run_mean_avg,
                r.ssim_run_std_avg,
            ])


def plot_summary(rows: List[EvalResult], out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    labels = [r.label for r in rows]
    x = list(range(len(rows)))

    plt.figure(figsize=(10, 4))
    plt.plot(x, [r.avg_psnr_of_avg_sr for r in rows], marker="o", label="PSNR(avg SR)")
    plt.plot(x, [r.psnr_run_mean_avg for r in rows], marker="s", label="PSNR(run mean)")
    plt.xticks(x, labels, rotation=45, ha="right")
    plt.ylabel("PSNR")
    plt.title("PSNR Across Checkpoints")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "summary_psnr.png"), dpi=160)
    plt.close()

    plt.figure(figsize=(10, 4))
    plt.plot(x, [r.avg_ssim_of_avg_sr for r in rows], marker="o", label="SSIM(avg SR)")
    plt.plot(x, [r.ssim_run_mean_avg for r in rows], marker="s", label="SSIM(run mean)")
    plt.xticks(x, labels, rotation=45, ha="right")
    plt.ylabel("SSIM")
    plt.title("SSIM Across Checkpoints")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "summary_ssim.png"), dpi=160)
    plt.close()

    plt.figure(figsize=(10, 4))
    plt.plot(x, [r.psnr_run_std_avg for r in rows], marker="o", label="PSNR run std")
    plt.plot(x, [r.ssim_run_std_avg for r in rows], marker="s", label="SSIM run std")
    plt.xticks(x, labels, rotation=45, ha="right")
    plt.ylabel("Std over runs")
    plt.title("Stochasticity (Mean Std) Across Checkpoints")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "summary_stochasticity.png"), dpi=160)
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Evaluate multiple checkpoints with sr.py -p val and summarize PSNR/SSIM mean+std."
    )
    ap.add_argument("--config", required=True, help="Base config path (e.g., config/celebahq.json)")
    ap.add_argument("--sr-script", default="sr.py", help="Path to sr.py")
    ap.add_argument("--checkpoint-prefixes", nargs="*", default=None,
                    help="Checkpoint prefixes or *_gen.pth paths, e.g. .../I10000_E1")
    ap.add_argument("--checkpoint-dir", default=None,
                    help="Directory with checkpoint files (I*_E*_gen.pth).")
    ap.add_argument("--limit", type=int, default=10,
                    help="When using --checkpoint-dir, keep N checkpoints after sorting (default: 10).")
    ap.add_argument("--pick", choices=["first", "last"], default="last",
                    help='For --checkpoint-dir + --limit, choose earliest N ("first") or latest N ("last").')
    ap.add_argument("--start-index", type=int, default=1,
                    help="1-based start index after sorting checkpoints (default: 1).")
    ap.add_argument("--index-step", type=int, default=1,
                    help="Pick every Nth checkpoint after start index (default: 1).")
    ap.add_argument("--n-run", type=int, default=100, help="n_run for each checkpoint evaluation.")
    ap.add_argument("--data-len", type=int, default=None,
                    help="Optional override for datasets.val.data_len.")
    ap.add_argument("--gpu-ids", default=None, help='GPU ids passed to sr.py, e.g. "0" or "0,1".')
    ap.add_argument("--mkl-threading-layer", default="GNU",
                    help='Subprocess MKL threading layer for sr.py (default: "GNU").')
    ap.add_argument("--output-dir", default="plots/checkpoint_eval", help="Summary output directory.")
    args = ap.parse_args()

    ckpts: List[str] = []
    if args.checkpoint_prefixes:
        ckpts.extend([normalize_checkpoint_prefix(c) for c in args.checkpoint_prefixes])
    if args.checkpoint_dir:
        ckpts.extend(
            discover_checkpoints(
                args.checkpoint_dir,
                args.limit,
                pick=args.pick,
                start_index=args.start_index,
                index_step=args.index_step,
            )
        )
    ckpts = sorted(set(ckpts), key=ckpt_sort_key)

    if not ckpts:
        raise ValueError("No checkpoints provided. Use --checkpoint-prefixes or --checkpoint-dir.")

    os.makedirs(args.output_dir, exist_ok=True)
    results: List[EvalResult] = []
    for i, ckpt in enumerate(ckpts, start=1):
        print(f"[{i}/{len(ckpts)}] Evaluating {ckpt}")
        result = run_one_checkpoint(
            sr_py=args.sr_script,
            base_cfg_path=args.config,
            checkpoint_prefix=ckpt,
            out_dir=args.output_dir,
            n_run=args.n_run,
            data_len=args.data_len,
            gpu_ids=args.gpu_ids,
            mkl_threading_layer=args.mkl_threading_layer,
            idx=i,
        )
        print(f"    -> avgSR PSNR={result.avg_psnr_of_avg_sr:.4f}, SSIM={result.avg_ssim_of_avg_sr:.4f}")
        print(f"    -> run PSNR std={result.psnr_run_std_avg:.4f}, SSIM std={result.ssim_run_std_avg:.4f}")
        results.append(result)

    results = sorted(
        results,
        key=lambda r: (
            r.checkpoint_epoch if r.checkpoint_epoch is not None else 10**18,
            r.checkpoint_iter if r.checkpoint_iter is not None else 10**18,
            r.label,
        ),
    )

    summary_csv = os.path.join(args.output_dir, "checkpoint_eval_summary.csv")
    save_summary_csv(results, summary_csv)
    plot_summary(results, args.output_dir)
    print(f"Saved summary: {summary_csv}")
    print(f"Saved plots: {args.output_dir}/summary_psnr.png, summary_ssim.png, summary_stochasticity.png")


if __name__ == "__main__":
    main()
