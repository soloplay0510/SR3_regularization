import os
import re
from PIL import Image
import math

root_dir = "experiments/celeb_tv0_stdrelu_250825_151239/results"
output_dir = "plots"
os.makedirs(output_dir, exist_ok=True)

# filename pattern: <leadnum>_<idx>_sr.png  e.g., 10000_1_sr.png
pat = re.compile(r"^(\d+)_([0-9]+)_sr\.png$")

# Collect: groups[idx] = list of (leadnum_int, fullpath)
groups = {}

for sub in os.listdir(root_dir):
    subpath = os.path.join(root_dir, sub)
    if not os.path.isdir(subpath):
        continue
    for fn in os.listdir(subpath):
        m = pat.match(fn)
        if not m:
            continue
        leadnum = int(m.group(1))   # sort key inside a group
        idx = m.group(2)
        groups.setdefault(idx, []).append((leadnum, os.path.join(subpath, fn)))

# Make combined grid image per idx, sorted by leadnum
for idx, items in groups.items():
    items.sort(key=lambda x: x[0])    # sort by leading number
    imgs = [Image.open(p) for _, p in items]

    # Resize all to same height (for neat grid)
    max_h = max(im.height for im in imgs)
    resized = [im.resize((int(im.width * max_h / im.height), max_h)) if im.height != max_h else im
               for im in imgs]

    per_row = 10
    n = len(resized)
    rows = math.ceil(n / per_row)
    img_w, img_h = resized[0].size  # after resize, all should be consistent

    total_w = per_row * img_w
    total_h = rows * img_h

    canvas = Image.new("RGB", (total_w, total_h), "white")

    for k, im in enumerate(resized):
        row = k // per_row
        col = k % per_row
        x = col * img_w
        y = row * img_h
        canvas.paste(im, (x, y))

    out = os.path.join(output_dir, f"relu_group_{idx}_sr_2.png")
    canvas.save(out)
    print(f"Saved: {out}")
