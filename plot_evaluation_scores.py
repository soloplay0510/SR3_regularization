import re
import matplotlib.pyplot as plt
import numpy as np


logfile = "experiments/celeb_tv0_stdrelu_250824_162052/logs/val.log"   # change this to your log filename

epochs, psnr, ssim = [], [], []

# Regex patterns
epoch_pat = re.compile(r"epoch:\s*([0-9]+)")
psnr_pat = re.compile(r"psnr:\s*([\d\.e\+\-]+)")
ssim_pat = re.compile(r"ssim:\s*([\d\.e\+\-]+)")

with open(logfile, "r") as f:
    for line in f:
        e = epoch_pat.search(line)
        p = psnr_pat.search(line)
        s = ssim_pat.search(line)
        if e and p:  # line with PSNR
            epochs.append(int(e.group(1)))
            psnr.append(float(p.group(1)))
        elif e and s:  # line with SSIM
            ssim.append(float(s.group(1)))

epochs, psnr, ssim = np.array(epochs), np.array(psnr), np.array(ssim)

# Find best
best_psnr_idx = np.argmax(psnr)
best_ssim_idx = np.argmax(ssim)
best_psnr_epoch, best_psnr_val = epochs[best_psnr_idx], psnr[best_psnr_idx]
best_ssim_epoch, best_ssim_val = epochs[best_ssim_idx], ssim[best_ssim_idx]

# Plot with epochs on x-axis
fig, ax1 = plt.subplots(figsize=(12,6))

ax1.set_xlabel("Epoch")
ax1.set_ylabel("PSNR (dB)", color="tab:blue")
ax1.plot(epochs, psnr, marker="o", color="tab:blue", label="PSNR")
ax1.scatter(best_psnr_epoch, best_psnr_val, s=120, color="blue", zorder=5)
ax1.annotate(f"Best PSNR\n{best_psnr_val:.2f} @ epoch {best_psnr_epoch}",
             xy=(best_psnr_epoch, best_psnr_val),
             xytext=(best_psnr_epoch+2, best_psnr_val+0.8),
             arrowprops=dict(arrowstyle="->"))
ax1.tick_params(axis="y", labelcolor="tab:blue")
ax1.grid(True)

ax2 = ax1.twinx()
ax2.set_ylabel("SSIM", color="tab:orange")
ax2.plot(epochs, ssim, marker="o", color="tab:orange", label="SSIM")
ax2.scatter(best_ssim_epoch, best_ssim_val, s=120, color="orange", zorder=5)
ax2.annotate(f"Best SSIM\n{best_ssim_val:.3f} @ epoch {best_ssim_epoch}",
             xy=(best_ssim_epoch, best_ssim_val),
             xytext=(best_ssim_epoch-10, best_ssim_val-0.08),
             arrowprops=dict(arrowstyle="->"))
ax2.tick_params(axis="y", labelcolor="tab:orange")

fig.suptitle("PSNR & SSIM vs Epoch (Best Checkpoints Marked)")
fig.tight_layout()
plt.savefig("plots/psnr_ssim_vs_epoch.png", dpi=300)
