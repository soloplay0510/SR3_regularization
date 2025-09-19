#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import torch, torch.nn as nn
import math
import torch.nn.functional as F

class STDReLu(nn.Module):
    def __init__(self, n_channel,nb_iterations=10, nb_kerhalfsize=1.0):

        """
        :param nb_iterations: iterations number
        :param nb_kerhalfsize: the half size of neighborhood
        """
        super(STDReLu, self).__init__()
        self.nb_iterations = nb_iterations
        self.n_channel = n_channel
        self.ker_halfsize = nb_kerhalfsize
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Learnable version: sigma of Gaussian function; entropic parameter epsilon; regularization parameter lam

        self.nb_sigma = nn.Parameter(torch.FloatTensor([5.0] * n_channel).view(n_channel, 1, 1))
        self.lam = nn.Parameter(torch.FloatTensor([1.0])).to(self.device)
            
        self.relu = nn.ReLU(True)

    def forward(self, o):
        u = self.relu(o)
        ker = self.STD_Kernel( self.nb_sigma, self.ker_halfsize)
        ker = ker.to(self.device)
        for i in range(self.nb_iterations):
            q = F.conv2d(1.0 - 2.0 * u, ker, padding=int(self.ker_halfsize), groups=self.n_channel)
            # 2. relu
            u = self.relu(o - self.lam * q)
        return u

    def STD_Kernel(self, sigma, halfsize):
        x, y = torch.meshgrid(torch.arange(-halfsize, halfsize + 1), torch.arange(-halfsize, halfsize + 1))
        x = x.to(self.device)
        y = y.to(self.device)
        sigma = sigma.to(self.device)
        ker = torch.exp(-(x.float()**2 + y.float()**2) / (2.0*sigma*sigma))
        ker = ker / (0.2*math.pi*sigma*sigma)
        ker = ker.unsqueeze(1)

        return ker

class STDLeakyReLu(nn.Module):
    def __init__(self, n_channel, nb_iterations=10, nb_kerhalfsize=1, alpha=0.2):
        super(STDLeakyReLu, self).__init__()
        self.nb_iterations = nb_iterations
        self.n_channel = n_channel
        self.ker_halfsize = nb_kerhalfsize
        self.alpha = alpha
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Learnable parameters
        self.nb_sigma = nn.Parameter(torch.FloatTensor([5.0] * n_channel).view(n_channel, 1, 1), requires_grad=True)
        self.lam = nn.Parameter(torch.FloatTensor([1.0]), requires_grad=True)
    def forward(self, o):
        # Initialize u^0 using standard LeakyReLU
        u = F.leaky_relu(o, negative_slope=self.alpha)
        ker = self.STD_Kernel(self.nb_sigma, self.ker_halfsize) #.to(o.device)
        for _ in range(self.nb_iterations):
            # Subgradient of STD prior: P(y^t)
            q = F.conv2d(1.0 - 2.0 * u, ker, padding=self.ker_halfsize, groups=self.n_channel)
            # Gradient descent update
            u = F.leaky_relu(o-self.lam*q,negative_slope=self.alpha)
        return u
    def STD_Kernel(self, sigma, halfsize):
        x, y = torch.meshgrid(torch.arange(-halfsize, halfsize + 1),
                              torch.arange(-halfsize, halfsize + 1),
                              indexing='ij')
        x = x.to(self.device)
        y = y.to(self.device)
        ker = torch.exp(-(x.float() ** 2 + y.float() ** 2) / (2.0 * sigma * sigma))
        ker = ker / (0.2 * math.pi * sigma * sigma)
        ker = ker.unsqueeze(1)  # shape: [C, 1, K, K] for grouped conv
        return ker
    
class STDSLReLU(nn.Module):
    """
    STD-regularized Smooth LeakyReLU
      u^{t+1} = T_{alpha,beta}( o - lam * (K_sigma * (1 - 2 u^t)) ),
    with T_{alpha,beta}(x) = alpha*x + (1-alpha)/beta * softplus(beta*x).
    Learnables:
      - nb_sigma: [C,1,1] per-channel sigma for K
      - lam:      scalar
    """
    def __init__(self, n_channel, nb_iterations=10, nb_kerhalfsize=1, alpha=0.2, beta=10.0):
        super().__init__()
        self.nb_iterations = int(nb_iterations)
        self.n_channel = int(n_channel)
        self.ker_halfsize = int(nb_kerhalfsize)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.nb_sigma = nn.Parameter(torch.full((n_channel, 1, 1), 5.0), requires_grad=True)  # [C,1,1]
        self.lam = nn.Parameter(torch.tensor([1.0], dtype=torch.float32), requires_grad=True) # scalar
    @staticmethod
    def _slrelu(x, alpha, beta):
        # T_{alpha,beta}(x)
        return alpha * x + (1.0 - alpha) * F.softplus(beta * x) / beta
    def forward(self, o: torch.Tensor) -> torch.Tensor:
        # Expect [N,C,H,W]
        assert o.dim() == 4, f"Input must be 4D [N,C,H,W], got {tuple(o.shape)}"
        assert o.size(1) == self.n_channel, f"Channel mismatch {o.size(1)} vs {self.n_channel}"
        device = o.device
        # u^0: smooth LeakyReLU (not raw LeakyReLU)
        u = self._slrelu(o, self.alpha, self.beta)
        # Build Gaussian kernel [C,1,K,K] on the correct device
        ker = self._std_kernel(self.nb_sigma.to(device), self.ker_halfsize, device)
        assert ker.dim() == 4 and ker.size(0) == self.n_channel and ker.size(1) == 1, \
            f"Kernel must be [C,1,K,K], got {tuple(ker.shape)}"
        lam = self.lam.to(device)
        for _ in range(self.nb_iterations):
            # q = K * (1 - 2u)
            assert u.dim() == 4, f"u must be 4D, got {tuple(u.shape)}"
            q = F.conv2d(1.0 - 2.0 * u, ker, stride=1,
                         padding=self.ker_halfsize, groups=self.n_channel)
            u = self._slrelu(o - lam * q, self.alpha, self.beta)
        return u
    def _std_kernel(self, sigma: torch.Tensor, halfsize: int, device: torch.device) -> torch.Tensor:
        """
        Per-channel 2D Gaussian kernels with L1 normalization.
        sigma: [C,1,1] -> reshape to [C,1,1,1] for broadcasting.
        return: [C,1,K,K]
        """
        C = sigma.size(0)
        k = 2 * halfsize + 1
        # Explicit grid as [1,1,K,K]
        xs = torch.arange(-halfsize, halfsize + 1, device=device, dtype=torch.float32)
        ys = torch.arange(-halfsize, halfsize + 1, device=device, dtype=torch.float32)
        xg = xs.view(1, 1, 1, k).expand(1, 1, k, k)   # [1,1,K,K]
        yg = ys.view(1, 1, k, 1).expand(1, 1, k, k)   # [1,1,K,K]
        r2 = xg.pow(2) + yg.pow(2)                    # [1,1,K,K]
        sigma = sigma.view(C, 1, 1, 1).clamp_min(1e-6)
        ker = torch.exp(-r2 / (2.0 * sigma.pow(2)))   # -> [C,1,K,K]
        ker = ker / (ker.sum(dim=(2, 3), keepdim=True) + 1e-12)
        return ker