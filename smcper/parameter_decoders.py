import torch as th 
import torch.nn as nn
import math
from torch.autograd import Function
import numpy as np
from compressai.entropy_models import EntropyBottleneck

class StraightThrough(Function):

    @staticmethod
    def forward(ctx, tensor):
        return tensor.round()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

class ConvDecoder(nn.Module):

    '''
    In the original paper they say the they have self.W.shape = (IO, IO)
    however I find this puzzling since usually IOxIO > IOxHW which means
    that the number of parameters for the decoder is much larger than the number 
    of parameters in the layers. Additionally, the authors say that it is each 
    filter that is seen as a sample from the learned probability distribution
    so I assume that the IOxIO was a typo and instead use a matrix with shape 
    HWxHW (same for the bias)
    '''
    def __init__(self, kernel_size):
        super(ConvDecoder, self).__init__()

        if type(kernel_size) == int:
            kernel_size = (kernel_size, kernel_size)

        self.kernel_size = kernel_size
        self.HW = kernel_size[0] * kernel_size[1]
        self.bias = nn.Parameter(th.zeros(self.HW), requires_grad=True)
        self.weight = nn.Parameter(th.ones(self.HW, self.HW), requires_grad=True)

    def compress(self):
        bw = self.weight.detach().cpu().numpy().tobytes()
        bb = self.bias.detach().cpu().numpy().tobytes()
        return bw + bb

    def decompress(self, b):
        l = self.HW

        w = th.from_numpy(
            np.frombuffer(b[4:4 + l * l * 4],
            dtype=np.float32
        ).copy()).view(l, l)

        b = th.from_numpy(
            np.frombuffer(b[4 + l * l * 4:],
            dtype=np.float32
        ).copy()).view(l)

        self.weight.data = w.to(self.weight.device)
        self.bias.data = b.to(self.bias.device)

    def get_model_size(self):
        return (self.weight.numel() + self.bias.numel()) * 2

    def forward(self, x):
        i = x.shape[0]
        o = x.shape[1]
        IO = i * o
        x = x.view(IO, self.HW)
        return th.matmul(x + self.bias, self.weight).T.reshape(
            i,
            o,
            self.kernel_size[0],
            self.kernel_size[1]
        )

class AffineDecoder(nn.Module):

    def __init__(self, l):
        super().__init__()
        self.l = l
        self.bias = nn.Parameter(th.zeros(1, l), requires_grad=True)
        self.weight = nn.Parameter(th.ones(l, l) * (- 4), requires_grad=True)

    def get_model_size(self):
        return (self.weight.numel() + self.bias.numel()) * 2

    def forward(self, x):
        return th.matmul(x + self.bias, th.exp(self.weight))

    def compress(self):
        bw = self.weight.detach().cpu().numpy().tobytes()
        bb = self.bias.detach().cpu().numpy().tobytes()
        return bw + bb

    def decompress(self, b):
        l = self.l

        w = th.from_numpy(
            np.frombuffer(b[4:4 + l * l * 4],
            dtype=np.float32
        ).copy()).view(l, l)

        b = th.from_numpy(
            np.frombuffer(b[4 + l * l * 4:],
            dtype=np.float32
        ).copy()).view(1, l)

        self.weight.data = w.to(self.weight.device)
        self.bias.data = b.to(self.bias.device)

class LinearDecoder(nn.Module):

    def __init__(self, span, fan_in):
        span_const = math.sqrt(6.0) / math.sqrt(fan_in) * 2 / span
        super(LinearDecoder, self).__init__()
        self.bias = nn.Parameter(th.zeros(1), requires_grad=True)
        self.weight = nn.Parameter(th.ones(1) * span_const, requires_grad=True)  

    def get_model_size(self):
        return (self.weight.numel() + self.bias.numel()) * 2

    def compress(self):
        bw = self.weight.detach().cpu().numpy().tobytes()
        bb = self.bias.detach().cpu().numpy().tobytes()
        return bw + bb

    def decompress(self, b):
        w = th.from_numpy(np.frombuffer(b[4:8], dtype=np.float32).copy()).view(1)
        b = th.from_numpy(np.frombuffer(b[8:], dtype=np.float32).copy()).view(1)
        self.weight.data = w.to(self.weight.device)
        self.bias.data = b.to(self.bias.device)

        
    def forward(self, x):
        w_out = (x + self.bias) * self.weight
        return w_out



