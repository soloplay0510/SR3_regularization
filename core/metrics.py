import os
import math
import numpy as np
import cv2
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import pandas as pd


def tensor2img(tensor, out_type=np.uint8, min_max=(-1, 1)):
    '''
    Converts a torch Tensor into an image Numpy array
    Input: 4D(B,(3/1),H,W), 3D(C,H,W), or 2D(H,W), any range, RGB channel order
    Output: 3D(H,W,C) or 2D(H,W), [0,255], np.uint8 (default)
    '''
    tensor = tensor.squeeze().float().cpu().clamp_(*min_max)  # clamp
    tensor = (tensor - min_max[0]) / \
        (min_max[1] - min_max[0])  # to range [0,1]
    n_dim = tensor.dim()
    if n_dim == 4:
        n_img = len(tensor)
        img_np = make_grid(tensor, nrow=int(
            math.sqrt(n_img)), normalize=False).numpy()
        img_np = np.transpose(img_np, (1, 2, 0))  # HWC, RGB
    elif n_dim == 3:
        img_np = tensor.numpy()
        img_np = np.transpose(img_np, (1, 2, 0))  # HWC, RGB
    elif n_dim == 2:
        img_np = tensor.numpy()
    else:
        raise TypeError(
            'Only support 4D, 3D and 2D tensor. But received with dimension: {:d}'.format(n_dim))
    if out_type == np.uint8:
        img_np = (img_np * 255.0).round()
        # Important. Unlike matlab, numpy.unit8() WILL NOT round by default.
    return img_np.astype(out_type)


def save_img(img, img_path, mode='RGB'):
    cv2.imwrite(img_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    # cv2.imwrite(img_path, img)


def calculate_psnr(img1, img2):
    # img1 and img2 have range [0, 255]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))


def ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calculate_ssim(img1, img2):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')
def plot_eval_metrics(all_metric_means,all_metric_stds,runs,folder, metric = "PSNR"):
    x = np.arange(1, len(all_metric_means)+1)  # sample indices
    # PSNR curve
    plt.figure(figsize=(7,4))
    plt.plot(x, all_metric_means, marker='o', label=metric+" mean")
    plt.fill_between(x,
                    np.array(all_metric_means) - np.array(all_metric_stds),
                    np.array(all_metric_means) + np.array(all_metric_stds),
                    alpha=0.3, label="±1 std")
    plt.xticks(x)
    plt.xlabel("Sample index")
    plt.ylabel(metric)
    plt.title(metric+f" across samples (mean ± std over {runs} runs)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(folder+f'/eval_{metric.lower()}_curve.png')
    df = pd.DataFrame({
    "sample_index": range(1, len(all_metric_means)+1),
    f"{metric}_mean": all_metric_means,
    f"{metric}_std": all_metric_stds
    })
    df.to_csv(folder+f"/eval_{metric.lower()}_statistics.csv", index=False)
    print(f"Saved results to eval_{metric.lower()}_results.csv")


def _to_numpy(x):
    import torch
    if isinstance(x, np.ndarray):
        return x
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    raise TypeError("Input must be a NumPy array or PyTorch tensor.")

def calculate_sam(est,ref,mask=None, eps=1e-12, return_stats=True):
    import torch

    """
    Compute SAM per pixel between ref and est (H, W, C). Returns (sam_map_deg, stats)
    where sam_map_deg is in degrees. If return_stats=False, returns only the map.

    Args:
        ref: ground-truth image, shape (H, W, C) or (H, W) in np/torch. Any dtype.
        est: estimated image, same shape as ref.
        mask: optional boolean array (H, W); True = include pixel in metrics.
        eps: small constant to avoid divide by zero.
        return_stats: whether to also return dict(mean/median/std in degrees).

    Notes:
        - SAM is arccos( <r,e> / (||r||*||e||) ), applied per pixel across channels.
        - Works with any number of channels C ≥ 1 (grayscale is fine).
        - Scaling both ref and est by the same positive scalar at each pixel
          does not change the angle; but per-channel scaling does.
    """
    ref_np = _to_numpy(ref).astype(np.float64)
    est_np = _to_numpy(est).astype(np.float64)

    if ref_np.ndim == 2:  # (H, W) -> (H, W, 1)
        ref_np = ref_np[..., None]
    if est_np.ndim == 2:
        est_np = est_np[..., None]

    if ref_np.shape != est_np.shape:
        raise ValueError(f"Shape mismatch: ref {ref_np.shape} vs est {est_np.shape}")
    if ref_np.shape[-1] < 1:
        raise ValueError("Channel dimension must be ≥ 1.")

    # Per-pixel dot product and norms across channels
    dot = np.sum(ref_np * est_np, axis=-1)                       # (H, W)
    ref_norm = np.sqrt(np.sum(ref_np * ref_np, axis=-1))         # (H, W)
    est_norm = np.sqrt(np.sum(est_np * est_np, axis=-1))         # (H, W)

    denom = np.maximum(ref_norm * est_norm, eps)                 # avoid /0
    cosang = np.clip(dot / denom, -1.0, 1.0)                     # numeric safety
    sam_map_rad = np.arccos(cosang)                              # radians
    sam_map_deg = np.degrees(sam_map_rad)                        # degrees

    if mask is not None:
        mask = _to_numpy(mask).astype(bool)
        if mask.shape != sam_map_deg.shape:
            raise ValueError(f"Mask shape {mask.shape} must match image spatial shape {sam_map_deg.shape}")
        valid = mask
    else:
        # Exclude pixels where either vector is (near) zero-length
        valid = (ref_norm > eps) & (est_norm > eps)

    if return_stats:
        vals = sam_map_deg[valid]
        stats = {
            "mean_deg": float(np.mean(vals)) if vals.size else float("nan"),
            "median_deg": float(np.median(vals)) if vals.size else float("nan"),
            "std_deg": float(np.std(vals, ddof=0)) if vals.size else float("nan"),
            "num_valid": int(vals.size),
        }
    if valid.any():
        vmin, vmax = np.percentile(sam_map_deg[valid], [1, 99])  # robust scaling
        sam_norm = (sam_map_deg - vmin) / (vmax - vmin + 1e-8)
    else:
        sam_norm = np.zeros_like(sam_map_deg)

    sam_norm = np.clip(sam_norm, 0.0, 1.0)

    # Convert to torch tensor so tensor2img can handle it
    sam_map_tensor = torch.from_numpy(sam_norm)

    return (sam_map_tensor, stats) if return_stats else sam_map_tensor
   