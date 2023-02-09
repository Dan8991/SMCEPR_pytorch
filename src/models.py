from torch import nn
import torch as th
from parameter_decoders import ConvDecoder, LinearDecoder
from torch.nn.init import kaiming_uniform_, _calculate_fan_in_and_fan_out
from torch.nn.init import uniform_
from entropy_layers import EntropyLinear
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt

class LeNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 300)
        self.fc2 = nn.Linear(300, 100)
        self.fc3 = nn.Linear(100, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class EntropyLeNet(nn.Module):

    def __init__(self, span=10):

        super().__init__()

        # activations
        self.relu = nn.ReLU()

        #creating the decoders for the linear layers
        #The fan in should be the smallest input size for the group of layers considered by the decoder
        self.wdec = LinearDecoder(span=span, fan_in=300.0)
        self.bdec = LinearDecoder(span=span, fan_in=100.0)
        self.cdec = LinearDecoder(span=span, fan_in=100.0)
        self.ema_decay = 0

        # creating the entropic linear layers
        self.fc1 = EntropyLinear(784, 300, self.wdec, self.bdec, self.ema_decay)
        self.fc2 = EntropyLinear(300, 100, self.wdec, self.bdec, self.ema_decay)
        self.fc3 = EntropyLinear(100, 10, self.cdec, self.bdec, self.ema_decay)

        self.init_weights(self.fc1, span, 300.0)
        self.init_weights(self.fc2, span, 300.0)
        self.init_weights(self.fc3, span, 100.0)

    def init_weights(self, layer, span, min_in):

        fan_in = layer.w.data.size()[1]

        const_max = math.sqrt(6.0) / math.sqrt(min_in)
        default_const = math.sqrt(6.0) / math.sqrt(fan_in)

        mult = span / 2 / const_max * default_const

        # initialize weights
        layer.w.data.uniform_(-mult, mult)

        # initialize bias
        layer.b.data.uniform_(-mult, mult)

    def get_rate(self):

        p1, t1 = self.fc1.get_compressed_params_size()
        p2, t2 = self.fc2.get_compressed_params_size()
        p3, t3 = self.fc3.get_compressed_params_size()

        d1 = self.wdec.get_model_size()
        d2 = self.bdec.get_model_size()
        d3 = self.cdec.get_model_size()

        parameters_size = p1 + p2 + p3 
        entropy_model_size = t1 + t2 + t3 + d1 + d2 + d3
        return parameters_size, entropy_model_size

    def get_original_size(self):

        parameters_size = self.fc1.get_model_size()
        parameters_size += self.fc2.get_model_size()
        parameters_size += self.fc3.get_model_size()

        return parameters_size

    def update(self, force=False):
        self.fc1.update(force=force)
        self.fc2.update(force=force)
        self.fc3.update(force=force)

    def forward(self, x):

        x = x.view(-1, 784)
        rate = 0

        x, bpp = self.fc1(x)
        rate = rate + bpp
        x = self.relu(x)

        x, bpp = self.fc2(x)
        rate = rate + bpp
        x = self.relu(x)

        x, bpp = self.fc3(x)
        rate = rate + bpp
        return x, rate

class CafeLeNet(nn.Module):

    def __init__(self):

        super().__init__()
        self.conv1 = nn.Conv2d(1, 20, 5)
        self.conv2 = nn.Conv2d(20, 50, 5)
        self.max_pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.max_pool(self.relu(self.conv1(x)))
        x = self.max_pool(self.relu(self.conv2(x)))
        x = x.view(-1, 4*4*50)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class EntropyCafeLeNet(nn.Module):

    def __init__(self):

        super().__init__()

        # activations
        self.max_pool = nn.MaxPool2d(2)
        self.relu = nn.ReLU()

        # creating parameters for conv layers
        self.conv_params = [(1, 20, 5), (20, 50, 5)]
        
        self.conv_w_param = nn.ParameterList([
            nn.Parameter(
                th.randn(o, i, k, k), requires_grad=True
            ) for i, o, k in self.conv_params
        ])

        self.conv_b_param = nn.ParameterList([
            nn.Parameter(
                th.zeros(o), requires_grad=True
            ) for i, o, k in self.conv_params
        ])

        # creating the decoders for the conv layers
        self.wdecs = nn.ModuleList([
            ConvDecoder(*self.conv_params[0]),
            ConvDecoder(*self.conv_params[1])
        ])

        # creating the decoders for the linear layers
        self.wdec_lin = LinearDecoder()

        # creating the parameters for the biases 
        self.wdec_bias = LinearDecoder()

        # creating parameters for linear layers
        self.lin_params = [(4*4*50, 500), (500, 10)]

        self.lin_w_param = nn.ParameterList([
            nn.Parameter(
                th.randn(o, i), requires_grad=True
            ) for i, o in self.lin_params
        ])

        self.lin_b_param = nn.ParameterList([
            nn.Parameter(
                th.zeros(o), requires_grad=True
            ) for i, o in self.lin_params
        ])

        for w, b in zip(self.conv_w_param, self.conv_b_param):
            self.init_weights(w, b)

        for w, b in zip(self.lin_w_param, self.lin_b_param):
            self.init_weights(w, b)

    def get_rate(self):
        parameters_size = 0
        decoders_size = 0
        for w, b in zip(self.conv_w_param, self.conv_b_param):
            parameters_size += w.numel() + b.numel()

        for w, b in zip(self.lin_w_param, self.lin_b_param):
            parameters_size += w.numel() + b.numel()

        for param in self.wdecs.parameters():
            decoders_size += param.numel()

        for param in self.wdec_lin.parameters():
            decoders_size += param.numel()

        for param in self.wdec_bias.parameters():
            decoders_size += param.numel()

        return parameters_size * 4, decoders_size * 4

    def init_weights(self, w, b):

        # initialize weights
        kaiming_uniform_(w, a=math.sqrt(5))

        # initialize bias
        fan_in, _ = _calculate_fan_in_and_fan_out(w)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        uniform_(b, -bound, bound)

    def forward(self, x):
        for w, b, wdec in zip(self.conv_w_param, self.conv_b_param, self.wdecs):
            new_w, rate = wdec(w)
            new_b, rate = self.wdec_bias(b)
            x = self.max_pool(self.relu(F.conv2d(x, w, b)))

        x = x.view(-1, 4*4*50)
        for i, (w, b) in enumerate(zip(self.lin_w_param, self.lin_b_param)):
            new_w = self.wdec_lin(w)
            new_b = self.wdec_bias(b)
            x = F.linear(x, w, b)
            if i == 0:
                x = self.relu(x)
        return x

def train_step(model, x, y, opt, device, lambda_RD):
    x = x.to(device)
    y = y.to(device)
    opt.zero_grad()
    y_hat, rate = model(x)
    loss = nn.CrossEntropyLoss()(y_hat, y) 
    loss = loss + lambda_RD * rate
    loss.backward()
    opt.step()
    return loss.item(), rate

def test_step(model, test_dataloader, device):
    cum_loss = 0
    cum_acc = 0
    num_steps = 0
    model.cpu()
    model.eval()
    model.update(force=True)
    parameters_size, tables_size = model.get_rate()
    compressed_size = parameters_size + tables_size
    original_size = model.get_original_size()
    print("Rate:" + str(compressed_size))
    print(f"Parameters Size: {parameters_size}, Tables Size: {tables_size}")
    print("Original Rate:" + str(original_size))
    print("Compression Ratio:" + str(original_size / compressed_size))
    model.to("cuda" if th.cuda.is_available() else "cpu")

    for x,y in test_dataloader:
        with th.no_grad():
            x = x.to(device)
            y = y.to(device)
            y_hat, rate = model(x)
            loss = nn.CrossEntropyLoss()(y_hat, y)

        cum_loss += loss.item()
        cum_acc += (y_hat.argmax(dim=1) == y).float().mean().item()
        num_steps += 1

    model.train()
    print("Test loss: ", cum_loss / num_steps)
    print("Test accuracy: ", cum_acc / num_steps)


