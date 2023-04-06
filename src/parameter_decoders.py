import torch as th 
import torch.nn as nn
import math
from torch.autograd import Function
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
        self.b = nn.Parameter(th.zeros(self.HW), requires_grad=True)
        self.w = nn.Parameter(th.ones(self.HW, self.HW), requires_grad=True)

    def get_model_size(self):
        return (self.w.numel() + self.b.numel()) * 2

    def forward(self, x):
        i = x.shape[0]
        o = x.shape[1]
        IO = i * o
        x = x.view(IO, self.HW)
        return th.matmul(x + self.b, self.w).T.reshape(
            i,
            o,
            self.kernel_size[0],
            self.kernel_size[1]
        )

class AffineDecoder(nn.Module):

    def __init__(self, l):
        super().__init__()
        self.l = l
        self.b = nn.Parameter(th.zeros(1, l), requires_grad=True)
        self.w = nn.Parameter(th.ones(l, l) * (- 4), requires_grad=True)

    def get_model_size(self):
        return (self.w.numel() + self.b.numel()) * 2

    def forward(self, x):
        return th.matmul(x + self.b, th.exp(self.w))

class LinearDecoder(nn.Module):

    def __init__(self, span, fan_in):
        span_const = math.sqrt(6.0) / math.sqrt(fan_in) * 2 / span
        super(LinearDecoder, self).__init__()
        self.b = nn.Parameter(th.zeros(1), requires_grad=True)
        self.w = nn.Parameter(th.ones(1) * span_const, requires_grad=True)  

    def get_model_size(self):
        return (self.w.numel() + self.b.numel()) * 2
        
    def forward(self, x):
        w_out = (x + self.b) * self.w
        return w_out



