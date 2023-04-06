from modified_layers import EntropyLinear, EntropyConv2d
from parameter_decoders import ConvDecoder, LinearDecoder
import torch as th

linear_layer = EntropyLinear(10, 10)
x = th.randn(10)
W = th.zeros(10, 10)
b = th.zeros(10)

y = linear_layer(x, W, b)

assert y.shape == (10,)
assert th.allclose(y, th.zeros(10))

x = th.randn(10)
W = th.ones(10, 10)
b = th.zeros(10)

y = linear_layer(x, W, b)

assert th.abs(y.sum().item() - (x.sum() * 10)) < 1e-5

conv_layer = EntropyConv2d(1, 1, 3)
x = th.randn(1, 1, 3, 3)
W = th.zeros(1, 1, 3, 3)
b = th.zeros(1)

y = conv_layer(x, W, b)

assert y.shape == (1, 1, 1, 1)
assert th.allclose(y, th.zeros(1, 1, 1, 1))

x = th.randn(1, 1, 3, 3)
W = th.ones(1, 1, 3, 3)
b = th.zeros(1)

y = conv_layer(x, W, b)

assert th.abs(y.sum().item() - (x.sum())) < 1e-5

x = th.randn(1, 3, 10, 10)
conv_layer = EntropyConv2d(3, 20, 3)
conv_dec = ConvDecoder(3, 20, 3)
lin_dec = LinearDecoder(is_bias=True)
w_param = th.randn(9, 60)
b_param = th.randn(20)
W = conv_dec(w_param)
b = lin_dec(b_param, 20)
y = conv_layer(x, W, b)


x = th.randn(10)
lin_layer = EntropyLinear(10, 20)
wdec = LinearDecoder()
bdec = LinearDecoder(is_bias=True)
w_param = th.randn(200)
b_param = th.randn(20)
W = wdec(w_param, 10, 20)
b = bdec(b_param, 20)
y = lin_layer(x, W, b)
