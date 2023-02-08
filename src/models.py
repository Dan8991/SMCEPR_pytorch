from torch import nn
import torch as th
from parameter_decoders import ConvDecoder, LinearDecoder
from torch.nn.init import kaiming_uniform_, _calculate_fan_in_and_fan_out
from torch.nn.init import uniform_
import torch.nn.functional as F
import math

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

    def __init__(self):

        super().__init__()

        # activations
        self.relu = nn.ReLU()

        #creating the decoders for the linear layers
        self.wdec = LinearDecoder()
        self.bdec = LinearDecoder()

        # creating the parameters for the linear layers
        self.io_list = [(784, 300), (300, 100), (100, 10)]
        self.w_param = nn.ParameterList([nn.Parameter(th.randn(o, i), requires_grad=True) for i, o in self.io_list])
        self.b_param = nn.ParameterList([nn.Parameter(th.zeros(o), requires_grad=True) for i, o in self.io_list])

        # for w, b in zip(self.w_param, self.b_param):
            # self.init_weights(w, b)

    def init_weights(self, w, b):
        # initialize weights
        kaiming_uniform_(w, a=math.sqrt(5))

        # initialize bias
        fan_in, _ = _calculate_fan_in_and_fan_out(w)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        uniform_(b, -bound, bound)

    def get_model_size(self):
        parameters_size = 0
        decoders_size = 0
        for w, b in zip(self.w_param, self.b_param):
            parameters_size += w.numel() + b.numel()

        for param in self.wdec.parameters():
            decoders_size += param.numel()

        for param in self.bdec.parameters():
            decoders_size += param.numel()

        return parameters_size * 4, decoders_size * 4

    def get_rate(self):

        self.wdec.eval()
        self.bdec.eval()

        parameters_size = 0
        decoders_size = 0
        for w, b in zip(self.w_param, self.b_param):
            strings = self.wdec.entropy_bottleneck.compress(
                w.unsqueeze(0).unsqueeze(1).round()
            )
            parameters_size += len(strings[0])

            strings = self.bdec.entropy_bottleneck.compress(b.unsqueeze(0).unsqueeze(1).round())
            parameters_size += len(strings[0])

        for param in self.wdec.parameters():
            decoders_size += param.numel()

        for param in self.bdec.parameters():
            decoders_size += param.numel()

        decoders_size += self.wdec.entropy_bottleneck.quantiles.numel()
        decoders_size += self.wdec.entropy_bottleneck._quantized_cdf.numel()
        decoders_size += self.bdec.entropy_bottleneck.quantiles.numel()
        decoders_size += self.bdec.entropy_bottleneck._quantized_cdf.numel()

        self.wdec.train()
        self.bdec.train()

        return parameters_size * 4, decoders_size * 4

    def update(self, force=False):
        self.wdec.update(force=force)
        self.bdec.update(force=force)

    def forward(self, x):
        x = x.view(-1, 784)
        rate = 0
        for i, (w, b) in enumerate(zip(self.w_param, self.b_param)):
            new_w, likelyhoods_w = self.wdec(w)
            new_b, likelyhoods_b = self.bdec(b)
            new_rate = - th.log2(likelyhoods_w).sum()
            new_rate = new_rate  - th.log2(likelyhoods_b).sum() 
            new_rate = new_rate / (w.numel() + b.numel())
            rate = rate + new_rate
            x = F.linear(x, new_w, new_b)
            if i < 2:
                x = self.relu(x)
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
    model.update(force=True)
    model_size, decoder_size = model.get_rate()
    uncompressed_model_size, uncompressed_decoder_size = model.get_model_size()
    print("Rate:" + str(model_size + decoder_size))
    print("Rate parts: " + str(model_size) + " " + str(decoder_size))
    print("Uncompressed Rate:" + str(uncompressed_model_size + uncompressed_decoder_size))
    print("Uncompressed Rate parts: " + str(uncompressed_model_size) + " " + str(uncompressed_decoder_size))
    print("Compression Ratio:" + str((model_size + decoder_size) / (uncompressed_model_size + uncompressed_decoder_size)))
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

    print("Test loss: ", cum_loss / num_steps)
    print("Test accuracy: ", cum_acc / num_steps)


