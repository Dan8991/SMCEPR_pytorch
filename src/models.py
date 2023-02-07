from torch import nn
import torch as th
from parameter_decoders import ConvDecoder, LinearDecoder
from modified_layers import EntropyLinear, EntropyConv2d

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
        self.io_list = [(784, 300), (300, 100), (100, 10)]
        self.fcs = nn.ModuleList([EntropyLinear() for i, o in self.io_list])
        self.relu = nn.ReLU()
        self.wdec = LinearDecoder()
        self.bdec = LinearDecoder(is_bias=True)
        self.w_param = nn.ParameterList([nn.Parameter(th.randn(i * o, 1), requires_grad=True) for i, o in self.io_list])
        self.b_param = nn.ParameterList([nn.Parameter(th.zeros(o, 1), requires_grad=True) for i, o in self.io_list])

    def forward(self, x):
        x = x.view(-1, 784)
        for i, (l, w, b, (in_features, out_features)) in enumerate(zip(self.fcs, self.w_param, self.b_param, self.io_list)):
            new_w = self.wdec(w, in_features, out_features)
            new_b = self.bdec(b, out_features)
            a = l(x, new_w, new_b)
            if i < 2:
                x = self.relu(a)
        return a

def train_step(model, x, y, opt, device):
    x = x.to(device)
    y = y.to(device)
    opt.zero_grad()
    y_hat = model(x)
    loss = nn.CrossEntropyLoss()(y_hat, y)
    loss.backward()
    opt.step()
    return loss.item()

def test_step(model, test_dataloader, device):
    cum_loss = 0
    cum_acc = 0
    num_steps = 0
    for x,y in test_dataloader:
        with th.no_grad():
            x = x.to(device)
            y = y.to(device)
            y_hat = model(x)
            loss = nn.CrossEntropyLoss()(y_hat, y)

        cum_loss += loss.item()
        cum_acc += (y_hat.argmax(dim=1) == y).float().mean().item()
        num_steps += 1

    print("Test loss: ", cum_loss / num_steps)
    print("Test accuracy: ", cum_acc / num_steps)


