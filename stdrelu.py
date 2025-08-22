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
        self.lam = nn.Parameter(torch.FloatTensor([1.0]))       
        self.relu = nn.ReLU(True)

    def forward(self, o):
        u = self.relu(o)
        ker = self.STD_Kernel( self.nb_sigma, self.ker_halfsize).cuda()
        for i in range(self.nb_iterations):
            q = F.conv2d(1.0 - 2.0 * u, ker, padding=int(self.ker_halfsize), groups=self.n_channel)
            # 2. relu
            u = self.relu(o - self.lam * q)
        return u

    def STD_Kernel(self, sigma, halfsize):
        x, y = torch.meshgrid(torch.arange(-halfsize, halfsize + 1), torch.arange(-halfsize, halfsize + 1))
        x = x.to(self.device)
        y = y.to(self.device)
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
