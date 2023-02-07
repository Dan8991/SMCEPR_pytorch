import torch as th
import torch.nn as nn

class EntropyLinear(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, W, b=None):
        x = th.matmul(x, W.T) 
        if b is not None:
            x += b
        return x

# not gonna work since assigning new parameters breaks the graph
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
        self.weight.requires_grad = False
        self.bias.requires_grad = False

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
