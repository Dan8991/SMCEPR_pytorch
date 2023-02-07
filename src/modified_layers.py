import torch as th
import torch.nn as nn

class EntropyLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias)

    def forward(self, x, W, b=None):
        assert W.shape == self.weight.shape
        assert b is None or b.shape == self.bias.shape

        W = nn.Parameter(W)
        if b is not None:
            b = nn.Parameter(b)

        self.weight = W
        self.bias = b
        x = super().forward(x)
        return x

class EntropyConv2d(nn.Conv2d):

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode='zeros'
    ):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode)

    def forward(self, x, W, b=None):
        assert W.shape == self.weight.shape
        assert b is None or b.shape == self.bias.shape

        W = nn.Parameter(W)
        if b is not None:
            b = nn.Parameter(b)

        self.weight = W
        self.bias = b
        x = super().forward(x)
        return x
