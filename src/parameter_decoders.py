import torch as th 
import torch.nn as nn

class ConvDecoder(nn.Module):

    def __init__(self, in_channel, out_channel, kernel_size):
        super(ConvDecoder, self).__init__()

        if type(kernel_size) == int:
            kernel_size = (kernel_size, kernel_size)

        self.kernel_size = kernel_size
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.IO = in_channel * out_channel
        self.HW = kernel_size[0] * kernel_size[1]
        self.b = nn.Parameter(th.zeros(self.IO), requires_grad=True)
        self.w = nn.Parameter(th.ones(self.IO, self.IO), requires_grad=True)

    def forward(self, x):
        x = x.view(self.IO, self.HW).T
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
        
    def forward(self, x):
        w_out = (x + self.b) * self.w
        return w_out



