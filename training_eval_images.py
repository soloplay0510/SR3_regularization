#!/usr/bin/env python3
import argparse, os, re, math
from typing import Dict, List, Tuple
from PIL import Image

# Filenames like: <leadnum>_<idx>_sr.png  e.g., 10000_1_sr.png
FNPAT = re.compile(r"^(\d+)_([0-9]+)_sr\.png$")

def load_groups(scan_root: str, verbose: bool = False) -> Dict[str, List[Tuple[int, str]]]:
    """Return groups[idx] = [(leadnum, fullpath), ...] sorted by leadnum ascending."""
    groups: Dict[str, List[Tuple[int, str]]] = {}
    matched = 0
    for dirpath, _, filenames in os.walk(scan_root):
        for fn in filenames:
            m = FNPAT.match(fn)
            if not m:
                continue
            leadnum = int(m.group(1))
            idx = m.group(2)
            full = os.path.join(dirpath, fn)
            groups.setdefault(idx, []).append((leadnum, full))
            matched += 1
    for idx in groups:
        groups[idx].sort(key=lambda x: x[0])
    if verbose:
        if matched:
            all_leads = [l for lst in groups.values() for (l, _) in lst]
            print(f"[{scan_root}] matched {matched} files | idxs={len(groups)} | "
                  f"leadnum range={min(all_leads)}..{max(all_leads)}")
        else:
            print(f"[{scan_root}] matched 0 files.")
    return groups

def open_and_uniform_height(paths: List[str], target_h: int = None):
    """Open images; resize to same height (target_h or max of set); enforce identical tile size."""
    imgs = []
    for p in paths:
        with Image.open(p) as im:
            imgs.append(im.copy())
    if not imgs:
        return [], 0, 0
    max_h = target_h if target_h else max(im.height for im in imgs)
    resized = []
    for im in imgs:
        if im.height != max_h:
            new_w = int(round(im.width * max_h / im.height))
            resized.append(im.resize((new_w, max_h)))
            im.close()
        else:
            resized.append(im)
    w, h = resized[0].size
    resized = [im if im.size == (w, h) else im.resize((w, h)) for im in resized]
    return resized, w, h

def make_grid(images: List[Image.Image], per_row: int, bg="white"):
    if not images:
        return None
    w, h = images[0].size
    n = len(images)
    rows = math.ceil(n / per_row)
    canvas = Image.new("RGB", (per_row * w, rows * h), bg)
    for k, im in enumerate(images):
        r, c = divmod(k, per_row)
        canvas.paste(im, (c * w, r * h))
    return canvas

def stack_vertical(images: List[Image.Image], spacing: int = 12, bg="white"):
    if not images:
        return None
    widths = [im.width for im in images]
    heights = [im.height for im in images]
    W = max(widths)
    H = sum(heights) + spacing * (len(images) - 1)
    out = Image.new("RGB", (W, H), bg)
    y = 0
    for im in images:
        x = (W - im.width) // 2
        out.paste(im, (x, y))
        y += im.height + spacing
    return out

def save_image(im: Image.Image, out_path: str):
    if im is None:
        print("Nothing to save.")
        return
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    im.save(out_path)
    print(f"Saved: {out_path}")

def parse_idx_list(values: List[str]) -> List[str]:
    """Accept space- or comma-separated idx values and return as strings."""
    out = []
    for v in values:
        out.extend([s for s in v.split(",") if s != ""])
    # normalize numeric strings (e.g., '01' -> '1')
    return [str(int(s)) for s in out]

def main():
    ap = argparse.ArgumentParser(
        description=("Scan results/<number> subfolders, gather ALL images (all leadnums) for each "
                     "idx across all input results dirs, and output grids. Filenames must be "
                     "<leadnum>_<idx>_sr.png.")
    )
    ap.add_argument("results_dirs", nargs="+", help="One or more '.../results' directories.")
    ap.add_argument("--min-subdir", type=int, default=None,
                    help="Include numeric subfolders >= this (e.g., 2).")
    ap.add_argument("--max-subdir", type=int, default=None,
                    help="Include numeric subfolders <= this (e.g., 90).")
    ap.add_argument("--idx", nargs="+", default=None,
                    help="Only include these idx groups (e.g., --idx 1 2 3 or --idx 1,2,3).")
    ap.add_argument("--separate-groups", action="store_true",
                    help="Save one image per idx instead of one big stacked image.")
    ap.add_argument("--output-dir", default="plots", help="Output directory.")
    ap.add_argument("--per-row", type=int, default=10, help="Tiles per row in each idx grid.")
    ap.add_argument("--target-height", type=int, default=None,
                    help="Force tile height (px). Default: auto per idx.")
    ap.add_argument("--prefix", default="", help="Output filename prefix.")
    ap.add_argument("--spacing", type=int, default=12, help="Vertical spacing between idx grids.")
    ap.add_argument("--verbose", action="store_true", help="Print scan summary.")
    args = ap.parse_args()

    # 1) Choose numeric subfolders under each results dir
    selected_scan_roots: Dict[str, List[str]] = {}
    for rd in args.results_dirs:
        if not os.path.isdir(rd):
            raise FileNotFoundError(f"Not a directory: {rd}")
        subs = []
        for name in sorted(os.listdir(rd), key=lambda s: (not s.isdigit(), int(s) if s.isdigit() else s)):
            p = os.path.join(rd, name)
            if not os.path.isdir(p) or not name.isdigit():
                continue
            num = int(name)
            if args.min_subdir is not None and num < args.min_subdir:
                continue
            if args.max_subdir is not None and num > args.max_subdir:
                continue
            subs.append(p)
        if not subs:
            if args.verbose:
                print(f"[WARN] {rd}: no numeric subfolders matched; scanning {rd} itself.")
            subs = [rd]
        selected_scan_roots[rd] = subs
        if args.verbose:
            nums = [os.path.basename(p) for p in subs if os.path.basename(p).isdigit()]
            print(f"[SELECT] {rd}: subdirs -> {nums or 'NONE'}")

    # 2) Collect ALL images per idx across everything
    global_groups: Dict[str, List[Tuple[int, str]]] = {}
    for rd in args.results_dirs:
        for scan_root in selected_scan_roots[rd]:
            g = load_groups(scan_root, verbose=args.verbose)
            for idx, items in g.items():
                global_groups.setdefault(idx, []).extend(items)

    if not global_groups:
        print("No matching images found.")
        return

    # Optional: filter by idx list
    if args.idx:
        wanted = set(parse_idx_list(args.idx))
        global_groups = {k: v for k, v in global_groups.items() if k in wanted}
        if args.verbose:
            print(f"[FILTER] Keeping idx groups: {sorted(wanted, key=int)}")

    if not global_groups:
        print("No matching idx groups after filtering.")
        return

    # 3) Build per-idx grids (all leadnums), then either stack or save separately
    per_idx_grids: List[Tuple[str, Image.Image]] = []
    for idx in sorted(global_groups.keys(), key=lambda x: int(x) if x.isdigit() else x):
        items = sorted(global_groups[idx], key=lambda t: t[0])  # by leadnum
        paths = [p for _, p in items]
        imgs, w, h = open_and_uniform_height(paths, target_h=args.target_height)
        if not imgs:
            continue
        grid = make_grid(imgs, per_row=args.per_row)
        if grid:
            per_idx_grids.append((idx, grid))

    os.makedirs(args.output_dir, exist_ok=True)

    if args.separate_groups:
        # One file per idx
        for idx, grid in per_idx_grids:
            out_name = f"{args.prefix}group_{idx}.png"
            save_image(grid, os.path.join(args.output_dir, out_name))
    else:
        # One big image stacking all idx grids
        final = stack_vertical([g for _, g in per_idx_grids], spacing=args.spacing)
        out_name = f"{args.prefix}ALL_IDX_merged.png"
        save_image(final, os.path.join(args.output_dir, out_name))

if __name__ == "__main__":
    main()
