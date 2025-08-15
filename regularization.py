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