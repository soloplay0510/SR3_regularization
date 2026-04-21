import torch
import torch.nn as nn
import torch.nn.functional as F
from TV_activation import TVLeakyReLU

class SimpleCNN(nn.Module):
    def __init__(self, scale_factor=4, use_TVrelu=False):
        super(SimpleCNN, self).__init__()
        self.scale_factor = scale_factor
        self.use_TVrelu = use_TVrelu

        # convolution layers
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1, bias=True)

        self.tvleaky = TVLeakyReLU(n_channel=32)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        nn.init.normal_(self.conv3.weight, mean=0.0, std=0.001)
        if self.conv3.bias is not None:
            nn.init.constant_(self.conv3.bias, 0)

    def forward(self, x):
        x_skip = x
        x_up = F.interpolate(x_skip, scale_factor=self.scale_factor,
                             mode='bicubic', align_corners=False)

        x = self.conv1(x_up)
        x = self.relu1(x)
        x = self.conv2(x)

        if self.use_TVrelu:
            x = self.tvleaky(x)
        else:
            x = self.relu2(x)

        x = self.conv3(x)

        return x + x_up


if __name__ == '__main__':
    x = torch.randn(7, 3, 32, 32)
    net = SimpleCNN(use_TVrelu=False)
    y = net(x)
    print(y.shape)
