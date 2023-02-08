import torch as th 
import torch.nn as nn
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
    def __init__(self, in_channel, out_channel, kernel_size):
        super(ConvDecoder, self).__init__()

        if type(kernel_size) == int:
            kernel_size = (kernel_size, kernel_size)

        self.kernel_size = kernel_size
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.IO = in_channel * out_channel
        self.HW = kernel_size[0] * kernel_size[1]
        self.b = nn.Parameter(th.zeros(self.HW), requires_grad=True)
        self.w = nn.Parameter(th.ones(self.HW, self.HW), requires_grad=True)

    def forward(self, x):
        x = x.view(self.IO, self.HW)
        return th.matmul(x + self.b, self.w).T.reshape(
            self.out_channel,
            self.in_channel,
            self.kernel_size[0],
            self.kernel_size[1]
        )

class LinearDecoder(nn.Module):

    def __init__(self):
        super(LinearDecoder, self).__init__()
        self.b = nn.Parameter(th.zeros(1), requires_grad=True)
        self.w = nn.Parameter(th.ones(1), requires_grad=True)  
        self.entropy_bottleneck = EntropyBottleneck(1)
        self.ste = StraightThrough()

    def update(self, force=False):
        self.entropy_bottleneck.update(force)
        
    def forward(self, x):
        x_hat, x_likelyhoods = self.entropy_bottleneck(x.unsqueeze(1))
        w_out = (self.ste.apply(x) + self.b) * self.w
        return w_out, x_likelyhoods



