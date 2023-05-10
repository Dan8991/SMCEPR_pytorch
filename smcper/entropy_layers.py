import torch as th
import torch.nn as nn
from torch.autograd import Function
from compressai.entropy_models import EntropyBottleneck
from torch.nn import functional as F
import io
import numpy as np
from smcper.io_utils import tensor_to_byte, get_next_tensor, get_bytestream

class StraightThrough(Function):

    @staticmethod
    def forward(ctx, tensor):
        return tensor.round()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

class CustomEntropyBottleneck(EntropyBottleneck):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def update(self, force, super_up=True) -> bool:
        # Check if we need to update the bottleneck parameters, the offsets are
        # only computed and stored when the conditonal model is update()'d.
        if super_up:
            return super().update(force)
        else:
            if self._offset.numel() > 0 and not force:
                return False

            medians = self.quantiles[:, 0, 1]

            minima = medians - self.quantiles[:, 0, 0]
            minima = th.ceil(minima).int()
            minima = th.clamp(minima, min=0)

            maxima = self.quantiles[:, 0, 2] - medians
            maxima = th.ceil(maxima).int()
            maxima = th.clamp(maxima, min=0)

            self._offset = -minima

            pmf_start = medians - minima
            pmf_length = maxima + minima + 1

            self._cdf_length = pmf_length + 2
            return True

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

        self.weight = nn.Parameter(th.randn(in_features * out_features, l), requires_grad=True)
        self.ema_w = nn.Parameter(th.zeros_like(self.weight))

        if bias_decoder:
            self.bias = nn.Parameter(
                th.randn(out_features, 1),
                requires_grad=True,
            )

        self.ema_b = nn.Parameter(th.zeros_like(self.bias))

        self.entropy_bottleneck_w = CustomEntropyBottleneck(l)
        if bias_decoder:
            self.entropy_bottleneck_b = CustomEntropyBottleneck(1)

        self.ste = StraightThrough()
        self.layer_func = None

        self.extra_args = extra_args
        # apply init_weights
        self.init_weights()

    def init_weights(self):
        pass

    def get_non_entropy_parameters(self):
        params = [self.weight]
        if self.bias_decoder:
            params += [self.bias]
        return params

    def get_entropy_parameters(self):
        params = list(self.entropy_bottleneck_w.parameters())
        if self.bias_decoder:
            params += list(self.entropy_bottleneck_b.parameters())
        return params
    
    def update(self, force=False, super_up=True):
        self.entropy_bottleneck_w.update(force=force, super_up=super_up)
        if self.bias_decoder:
            self.entropy_bottleneck_b.update(force=force, super_up=super_up)

    def get_compressed_params_size(self):
        device = self.weight.device
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

    def compress(self):
        device = self.weight.device
        self.to("cpu")
        strings_w = self.entropy_bottleneck_w.compress(
            self.ema_w.T.unsqueeze(0).round()
        )
        quantiles_w = self.entropy_bottleneck_w.quantiles.half()
        quantized_cdf_w = self.entropy_bottleneck_w._quantized_cdf
        if self.bias_decoder:
            strings_bias = self.entropy_bottleneck_b.compress(
                self.ema_b.T.unsqueeze(0).round()
            )
            quantiles_b = self.entropy_bottleneck_b.quantiles.half()
            quantized_cdf_b = self.entropy_bottleneck_b._quantized_cdf
            self.entropy_bottleneck_b.train()

        self.entropy_bottleneck_w.train()
        self.to(device)

        b = tensor_to_byte(quantiles_w)
        b += tensor_to_byte(quantized_cdf_w)
        b += tensor_to_byte(quantiles_b)
        b += tensor_to_byte(quantized_cdf_b)
        b += int.to_bytes(len(strings_w[0]), 4, byteorder="little")
        b += strings_w[0]
        b += int.to_bytes(len(strings_bias[0]), 4, byteorder="little")
        b += strings_bias[0]

        return b

    def decompress(self, b):
        device = self.weight.device
        self.to("cpu")

        quantiles_w, b = get_next_tensor(b, dtype=np.float16)
        quantized_cdf_w, b = get_next_tensor(b, dtype=np.int32)

        quantiles_b, b = get_next_tensor(b, dtype=np.float16)
        quantized_cdf_b, b = get_next_tensor(b, dtype=np.int32)

        quantized_cdf_w = quantized_cdf_w.squeeze(-1)
        quantized_cdf_b = quantized_cdf_b.squeeze(-1)

        strings_w, b = get_bytestream(b)
        strings_b, b = get_bytestream(b)

        strings_w = [strings_w]
        strings_b = [strings_b]

        self.entropy_bottleneck_w.quantiles.data = quantiles_w.float()
        self.entropy_bottleneck_w._quantized_cdf.data = quantized_cdf_w
        self.entropy_bottleneck_w._cdf_length = th.tensor([quantized_cdf_w.size(1)])

        self.entropy_bottleneck_b.quantiles.data = quantiles_b.float()
        self.entropy_bottleneck_b._quantized_cdf.data = quantized_cdf_b
        self.entropy_bottleneck_b._cdf_length = th.tensor([quantized_cdf_b.size(1)])

        self.ema_w.data = self.entropy_bottleneck_w.decompress(
            strings_w, 
            self.ema_w.shape
        )[0][0]
        self.ema_b.data = self.entropy_bottleneck_b.decompress(
            strings_b,
            self.ema_b.shape
        )[0][0]

        self.to(device)
        return b


    def get_model_size(self):
        if self.bias_decoder:
            return self.weight.numel() * 4 + self.bias.numel() * 4
        else:
            return self.weight.numel() * 4

    def get_weight_and_bias(self, w, b):
        raise Exception("Implement get_weight_and_bias in a child class")

    def get_ema_and_rate(self):
        if self.training:
            detatched_w = self.ema_w.detach()
            self.ema_w = nn.Parameter(self.ema_decay * detatched_w + (1 - self.ema_decay) * self.weight)
            if self.bias_decoder:
                detatched_b = self.ema_b.detach()
                self.ema_b = nn.Parameter(self.ema_decay * detatched_b + (1 - self.ema_decay) * self.bias)

        if self.training:
            b = self.bias
            w = self.weight
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
        nn.init.xavier_uniform_(self.weight.view(self.out_features, self.in_features))
        self.weight.data /= th.exp(th.tensor(-4))
        nn.init.zeros_(self.bias)

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
            self.weight.view(
                self.out_features,
                self.in_features,
                *self.kernel_size
            )
        )
        self.weight.data /= th.exp(th.tensor(-4))

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
