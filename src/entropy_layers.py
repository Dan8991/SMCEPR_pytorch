import torch as th
import torch.nn as nn
from torch.autograd import Function
from compressai.entropy_models import EntropyBottleneck
from torch.nn import functional as F

class StraightThrough(Function):

    @staticmethod
    def forward(ctx, tensor):
        return tensor.round()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

class EntropyLinear(nn.Module):

    def __init__(self, in_features, out_features, weight_decoder, bias_decoder=None, ema_decay=0.999):
        super(EntropyLinear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.ema_decay = ema_decay

        self.w = nn.Parameter(th.randn(out_features, in_features), requires_grad=True)
        self.b = nn.Parameter(th.zeros(out_features), requires_grad=True)
        self.ema_w = th.zeros_like(self.w)
        self.ema_b = th.zeros_like(self.b)

        self.weight_decoder = weight_decoder
        self.bias_decoder = bias_decoder

        self.entropy_bottleneck_w = EntropyBottleneck(1)
        self.entropy_bottleneck_b = EntropyBottleneck(1)
        self.ste = StraightThrough()

    def get_non_entropy_parameters(self):

        w_params = [self.w]
        b_params = [self.b]

        return w_params + b_params

    def get_entropy_parameters(self):
        entropy_w_params = list(self.entropy_bottleneck_w.parameters()) 
        entropy_b_params = list(self.entropy_bottleneck_b.parameters())
        return entropy_w_params + entropy_b_params

    def forward(self, x):

        if self.w.is_cuda:
            self.ema_to_device("cuda")
        else:
            self.ema_to_device("cpu")
                
        if self.training:

            self.ema_w = self.ema_w.detach()
            self.ema_b = self.ema_b.detach()

            self.ema_w = self.ema_decay * self.ema_w + (1 - self.ema_decay) * self.w
            self.ema_b = self.ema_decay * self.ema_b + (1 - self.ema_decay) * self.b
            
        _, likelyhoods_w = self.entropy_bottleneck_w(self.ema_w.unsqueeze(0).unsqueeze(0))
        _, likelyhoods_b = self.entropy_bottleneck_b(self.ema_b.unsqueeze(0).unsqueeze(0))

        w_hat = self.ste.apply(self.ema_w)
        b_hat = self.ste.apply(self.ema_b)

        w_hat = self.weight_decoder(w_hat)
        b_hat = self.bias_decoder(b_hat)

        out = F.linear(x, w_hat, b_hat)


        rate = - th.log2(likelyhoods_w).sum()
        rate = rate - th.log2(likelyhoods_b).sum()
        bpp = rate / (w_hat.numel() + b_hat.numel()) 

        return out, bpp

    def ema_to_device(self, device):
        self.ema_w = self.ema_w.to(device)
        self.ema_b = self.ema_b.to(device)
    
    def update(self, force=False):
        print("update", force)
        self.entropy_bottleneck_w.update(force=force)
        self.entropy_bottleneck_b.update(force=force)

    def get_compressed_params_size(self):

        parameters_size = 0
       
        self.entropy_bottleneck_w.eval()
        self.entropy_bottleneck_b.eval()

        self.ema_to_device("cpu")

        #computing the size of the compressed parameters
        strings = self.entropy_bottleneck_w.compress(self.ema_w.unsqueeze(0).unsqueeze(0).round())
        parameters_size += len(strings[0])

        strings = self.entropy_bottleneck_b.compress(self.ema_b.unsqueeze(0).unsqueeze(0).round())
        parameters_size += len(strings[0])

        tables_size = 0
        #computing the size of the tables
        tables_size += self.entropy_bottleneck_w.quantiles.numel() * 4
        tables_size += self.entropy_bottleneck_w._quantized_cdf.numel() * 4
        tables_size += self.entropy_bottleneck_b.quantiles.numel() * 4
        tables_size += self.entropy_bottleneck_b._quantized_cdf.numel() * 4

        self.entropy_bottleneck_w.train()
        self.entropy_bottleneck_b.train()

        self.ema_to_device("cuda" if self.w.is_cuda else "cpu")

        print("parameters_size: ", parameters_size)
        return parameters_size, tables_size

    def get_model_size(self):
        return (self.w.numel() + self.b.numel()) * 4


