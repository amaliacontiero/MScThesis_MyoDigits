import torch.nn as nn
import torch.nn.functional as F


class _ConvBlock(nn.Module):
    """Conv1d + GELU + Dropout. No BatchNorm; bias enabled. Supports dilation."""
    def __init__(self, cin, cout, k=5, stride=1, dilation=1, dropout=0.1):
        super().__init__()
        pad = dilation * (k // 2)
        self.net = nn.Sequential(
            nn.Conv1d(cin, cout, kernel_size=k, stride=stride,
                      padding=pad, dilation=dilation, bias=True),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class _ResidBlock(nn.Module):
    """Two-layer residual Conv1d block. No BatchNorm; bias enabled. Supports dilation."""
    def __init__(self, c, k=5, dilation=1, dropout=0.1):
        super().__init__()
        pad = dilation * (k // 2)
        self.conv1 = nn.Conv1d(c, c, kernel_size=k, padding=pad, dilation=dilation, bias=True)
        self.conv2 = nn.Conv1d(c, c, kernel_size=k, padding=pad, dilation=dilation, bias=True)
        self.drop  = nn.Dropout(dropout)

    def forward(self, x):
        r = x
        x = F.gelu(self.conv1(x))
        x = self.drop(x)
        x = self.conv2(x)
        return F.gelu(x + r)
