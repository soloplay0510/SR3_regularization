import torch
import torch.nn.functional as F

def TV1(img, epsilon=1e-6):
    """
    Computes the smooth total variation (TV) loss for a batch of images.

    Args:
        img (Tensor): shape (N, C, H, W) — batch of images
        epsilon (float): small constant to smooth the square root

    Returns:
        Scalar tensor representing the TV loss
    """
    # Horizontal and vertical differences
    dx = img[:, :, :, 1:] - img[:, :, :, :-1]  # (N, C, H, W-1)
    dy = img[:, :, 1:, :] - img[:, :, :-1, :]  # (N, C, H-1, W)

    # Pad to keep same shape
    dx = F.pad(dx, (0, 1, 0, 0))  # pad right
    dy = F.pad(dy, (0, 0, 0, 1))  # pad bottom
    
    # Smooth total variation (differentiable)
    tv = torch.abs(dx)+ torch.abs(dy) 

    # tv = torch.sqrt(dx**2 + dy**2 + epsilon)
    return tv.mean()


def TV2(img, epsilon=1e-6):
    """
    Computes the smooth total variation (TV) loss for a batch of images.

    Args:
        img (Tensor): shape (N, C, H, W) — batch of images
        epsilon (float): small constant to smooth the square root

    Returns:
        Scalar tensor representing the TV loss
    """
    # Horizontal and vertical differences
    dx = img[:, :, :, 1:] - img[:, :, :, :-1]  # (N, C, H, W-1)
    dy = img[:, :, 1:, :] - img[:, :, :-1, :]  # (N, C, H-1, W)

    # Pad to keep same shape
    dx = F.pad(dx, (0, 1, 0, 0))  # pad right
    dy = F.pad(dy, (0, 0, 0, 1))  # pad bottom
    
    # Smooth total variation (differentiable)
    # tv = torch.abs(dx)+ torch.abs(dy) 

    tv = torch.sqrt(dx**2 + dy**2 + epsilon)
    return tv.mean()

from scipy.special import comb
def frac_diff_1d(img, alpha, axis, n_terms=20):
    """
    Fractional finite difference along one axis for tensor (N, C, H, W).
    axis: 2 (height) or 3 (width).
    Uses zero padding instead of periodic boundary.
    """

    # Grünwald–Letnikov fractional binomial coefficients
    w = [((-1)**k) * comb(alpha, k) for k in range(n_terms)]
    w = torch.tensor(w, dtype=img.dtype, device=img.device)

    out = torch.zeros_like(img)
    for k in range(n_terms):
        if k == 0:
            shifted = img
        else:
            # shift with zero padding (no wraparound)
            pad_shape = [0, 0, 0, 0]  # [left, right, top, bottom]
            if axis == 2:   # vertical
                pad_shape[2] = k
            elif axis == 3: # horizontal
                pad_shape[0] = k
            padded = F.pad(img, pad_shape, mode="constant", value=0)
            if axis == 2:
                shifted = padded[:, :, :-k, :]
            else:
                shifted = padded[:, :, :, :-k]

        out = out + w[k] * shifted

    return out


def FTV(img, alpha=1.0, n_terms=20, epsilon=1e-6):
    """
    Fractional-order gradients. Matches isotropic TV when alpha=1.
    """
    dx = frac_diff_1d(img, alpha, axis=3, n_terms=n_terms)  # width
    dy = frac_diff_1d(img, alpha, axis=2, n_terms=n_terms)  # height

    # isotropic TV: sqrt(sum(dx^2 + dy^2) across channels)
    grad_mag = torch.sqrt(dx.pow(2) + dy.pow(2) + epsilon)
    grad_mag = grad_mag.sum(dim=1)  # sum across channels
    return grad_mag.mean()



import pywt
def waveL1(img, wname='haar'):
    """
    Compute wavelet L1 loss for a batch of images.
    
    Args:
        img: torch.Tensor of shape (N, C, H, W)
        wname: wavelet name (default: 'haar') refer to https://pywavelets.readthedocs.io/en/latest/ref/wavelets.html
    
    Returns:
        Scalar torch.Tensor (mean L1 norm of wavelet coefficients)
    """
    N, C, H, W = img.shape
    coeff_list = []

    # loop over batch and channels
    for n in range(N):
        for c in range(C):
            # convert slice to numpy for pywt
            arr = img[n, c].detach().cpu().numpy()
            coeffs2 = pywt.dwt2(arr, wname)
            cA, (cH, cV, cD) = coeffs2

            # concatenate coefficients into one array
            coeffs_all = np.concatenate(
                [cA.flatten(), cH.flatten(), cV.flatten(), cD.flatten()]
            )
            coeff_list.append(torch.tensor(coeffs_all, device=img.device, dtype=img.dtype))
    
    coeffs = torch.cat(coeff_list)
    return coeffs.abs().mean()