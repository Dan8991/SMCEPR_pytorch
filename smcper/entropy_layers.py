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

class EntropyLayer(nn.Module):

    def __init__(
        self,
        in_features,
        out_features,
        l,
        weight_decoder,
        bias_decoder=None,
        ema_decay=0.999,
        extra_args={},
    ):

        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.l = l
        self.weight_decoder = weight_decoder
        self.bias_decoder = bias_decoder
        self.ema_decay = ema_decay

        self.w = nn.Parameter(th.randn(in_features * out_features, l), requires_grad=True)
        self.ema_w = nn.Parameter(th.zeros_like(self.w))

        if bias_decoder:
            self.b = nn.Parameter(
                th.randn(out_features, 1),
                requires_grad=True,
            )

        self.ema_b = nn.Parameter(th.zeros_like(self.b))

        self.entropy_bottleneck_w = EntropyBottleneck(l)
        if bias_decoder:
            self.entropy_bottleneck_b = EntropyBottleneck(1)

        self.ste = StraightThrough()
        self.layer_func = None

        self.extra_args = extra_args
        # apply init_weights
        self.init_weights()

    def init_weights(self):
        pass

    def get_non_entropy_parameters(self):
        params = [self.w]
        if self.bias_decoder:
            params += [self.b]
        return params

    def get_entropy_parameters(self):
        params = list(self.entropy_bottleneck_w.parameters())
        if self.bias_decoder:
            params += list(self.entropy_bottleneck_b.parameters())
        return params
    
    def update(self, force=False):
        self.entropy_bottleneck_w.update(force=force)
        if self.bias_decoder:
            self.entropy_bottleneck_b.update(force=force)

    def get_compressed_params_size(self):
        device = self.w.device
        self.to("cpu")
        strings = self.entropy_bottleneck_w.compress(self.ema_w.T.unsqueeze(0).round())
        parameters_size = len(strings[0])
        tables_size = self.entropy_bottleneck_w.quantiles.numel() * 2
        tables_size += self.entropy_bottleneck_w._quantized_cdf.numel() * 2
        if self.bias_decoder:
            strings = self.entropy_bottleneck_b.compress(self.ema_b.T.unsqueeze(0).round())
            parameters_size += len(strings[0])
            tables_size += self.entropy_bottleneck_b.quantiles.numel() * 2
            tables_size += self.entropy_bottleneck_b._quantized_cdf.numel() * 2
            self.entropy_bottleneck_b.train()

        self.entropy_bottleneck_w.train()
        self.to(device)

        return parameters_size, tables_size

    def get_model_size(self):
        if self.bias_decoder:
            return self.w.numel() * 4 + self.b.numel() * 4
        else:
            return self.w.numel() * 4

    def get_weight_and_bias(self, w, b):
        raise Exception("Implement get_weight_and_bias in a child class")

    def get_ema_and_rate(self):
        if self.training:
            detatched_w = self.ema_w.detach()
            self.ema_w = nn.Parameter(self.ema_decay * detatched_w + (1 - self.ema_decay) * self.w)
            if self.bias_decoder:
                detatched_b = self.ema_b.detach()
                self.ema_b = nn.Parameter(self.ema_decay * detatched_b + (1 - self.ema_decay) * self.b)

        if self.training:
            b = self.b
            w = self.w
        else:
            b = self.ema_b
            w = self.ema_w
        _, likelyhoods_w = self.entropy_bottleneck_w(w.T.unsqueeze(0))
        w_hat = self.ste.apply(w)
        w_hat = self.weight_decoder(w_hat)
        rate_w = - th.log2(likelyhoods_w).sum() / w_hat.numel()
        rate_b = 0
        b_hat = None
        if self.bias_decoder:
            _, likelyhoods_b = self.entropy_bottleneck_b(b.T.unsqueeze(0))
            b_hat = self.ste.apply(b)
            b_hat = self.bias_decoder(b_hat)
            rate_b = - th.log2(likelyhoods_b).sum() / b_hat.numel()

        return w_hat, b_hat, rate_w, rate_b

    def forward(self, x):

        w_hat, b_hat, rate_w, rate_b = self.get_ema_and_rate()
        w_hat, b_hat = self.get_weight_and_bias(w_hat, b_hat)
        return self.layer_func(x, w_hat, bias=b_hat, **self.extra_args), rate_w + rate_b
        

class EntropyLinear(EntropyLayer):

    def __init__(
        self,
        in_features,
        out_features,
        weight_decoder,
        bias_decoder=None,
        ema_decay=0.999
    ):
        super().__init__(
            in_features,
            out_features,
            1,
            weight_decoder,
            bias_decoder=bias_decoder,
            ema_decay=ema_decay
        )

        self.layer_func = F.linear

    def init_weights(self):
        nn.init.xavier_uniform_(self.w.view(self.out_features, self.in_features))
        self.w.data /= th.exp(th.tensor(-4))
        nn.init.zeros_(self.b)

    def get_weight_and_bias(self, w, b):
        if b is not None:
            b = b.view(self.out_features)
        w = w.view(self.out_features, self.in_features)
        return w, b

class EntropyConv2d(EntropyLayer):

    def __init__(
        self,
        kernel_size,
        in_features,
        out_features,
        weight_decoder,
        padding=0,
        stride=1,
        bias_decoder=None,
        ema_decay=0.999
    ):
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)

        super().__init__(
            in_features,
            out_features,
            kernel_size[0] * kernel_size[1],
            weight_decoder,
            bias_decoder=bias_decoder,
            ema_decay=ema_decay,
        )

        self.in_features = in_features
        self.out_features = out_features
        self.kernel_size = kernel_size
        self.extra_args = {
            "padding": padding,
            "stride": stride
        }
        self.layer_func = F.conv2d

    def init_weights(self):
        nn.init.xavier_uniform_(
            self.w.view(
                self.out_features,
                self.in_features,
                *self.kernel_size
            )
        )
        self.w.data /= th.exp(th.tensor(-4))

    def get_weight_and_bias(self, w, b):
        if b is not None:
            b = b.view(self.out_features)
        w = w.view(
            self.out_features,
            self.in_features,
            self.kernel_size[0],
            self.kernel_size[1]
        )
        return w, b


