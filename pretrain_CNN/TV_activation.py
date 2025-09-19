import torch.nn as nn
import torch.nn.functional as F
class TVLeakyReLU(nn.Module):
    def __init__(self, n_channel, nb_iterations=10, alpha=0.2, tau=0.25, lam_init=0.2):
        super().__init__()
        self.n_channel = n_channel
        self.nb_iterations = nb_iterations
        self.alpha = float(alpha)
        self.tau = float(tau)
        # Make lambda a learnable parameter
        self.lam = nn.Parameter(torch.tensor(lam_init, dtype=torch.float32),requires_grad=False)
        # Optional: Add constraints to keep lambda positive
        # self.lam = nn.Parameter(torch.tensor(lam_init, dtype=torch.float32))
    def get_lambda(self):
        """Get the current lambda value, optionally with constraints"""
        return torch.clamp(self.lam, min=0.0, max=1.0)
    @staticmethod
    def _grad2d(u):
        u_right = torch.roll(u, shifts=-1, dims=3)
        px = u_right - u
        px[..., -1] = 0.0
        u_down = torch.roll(u, shifts=-1, dims=2)
        py = u_down - u
        py[..., -1, :] = 0.0
        return px, py
    @staticmethod
    def _div2d(px, py):
        px_left = torch.roll(px, shifts=1, dims=3)
        div_x = px - px_left
        div_x[..., 0] = px[..., 0]
        py_up = torch.roll(py, shifts=1, dims=2)
        div_y = py - py_up
        div_y[..., 0, :] = py[..., 0, :]
        return div_x + div_y
    @staticmethod
    def _proj_unit_ball(px, py, eps=1e-12):
        norm = torch.sqrt(px*px + py*py + eps)
        scale = torch.clamp(norm, min=1.0)
        px = px / scale
        py = py / scale
        return px, py
    def forward(self, z):
        u = F.leaky_relu(z, negative_slope=self.alpha)
        px = torch.zeros_like(u)
        py = torch.zeros_like(u)
        # Get the current lambda value
        current_lam = self.get_lambda()
        for _ in range(self.nb_iterations):
            gx, gy = self._grad2d(u)
            px = px + current_lam * self.tau * gx
            py = py + current_lam * self.tau * gy
            px, py = self._proj_unit_ball(px, py)
            div_p = self._div2d(px, py)
            u = F.leaky_relu(z + current_lam * div_p, negative_slope=self.alpha)
        return u