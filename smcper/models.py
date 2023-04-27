from torch import nn
import torch as th
from parameter_decoders import ConvDecoder, LinearDecoder, AffineDecoder
from torch.nn.init import kaiming_uniform_, _calculate_fan_in_and_fan_out
from torch.nn.init import uniform_
from entropy_layers import EntropyLinear, EntropyConv2d
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt
from torch.utils.data import Dataset

class LeNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 300)
        self.fc2 = nn.Linear(300, 100)
        self.fc3 = nn.Linear(100, 10)
        self.relu = nn.LeakyReLU()

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
        self.relu = nn.LeakyReLU()

        #creating the decoders for the linear layers
        #The fan in should be the smallest input size for the group of layers considered by the decoder
        self.wdec = AffineDecoder(1)
        self.bdec = AffineDecoder(1)
        self.cdec = AffineDecoder(1)
        self.ema_decay = 0

        # creating the entropic linear layers
        self.fc1 = EntropyLinear(784, 300, self.wdec, self.bdec, self.ema_decay)
        self.fc2 = EntropyLinear(300, 100, self.wdec, self.bdec, self.ema_decay)
        self.fc3 = EntropyLinear(100, 10, self.cdec, self.bdec, self.ema_decay)

        # self.init_weights(self.fc1, span, 300.0)
        # self.init_weights(self.fc2, span, 300.0)
        # self.init_weights(self.fc3, span, 100.0)
        
        self.dropout = nn.Dropout(0.1)

    # def init_weights(self, layer, span, min_in):

        # fan_in = layer.w.data.size()[1]

        # const_max = math.sqrt(6.0) / math.sqrt(min_in)
        # default_const = math.sqrt(6.0) / math.sqrt(fan_in)

        # mult = span / 2 / const_max * default_const

        # # initialize weights
        # layer.w.data.uniform_(-mult, mult)

        # # initialize bias
        # layer.b.data.uniform_(-mult, mult)


    def get_non_entropy_parameters(self):

        fc1_params = self.fc1.get_non_entropy_parameters()
        fc2_params = self.fc2.get_non_entropy_parameters()
        fc3_params = self.fc3.get_non_entropy_parameters()

        decoder_params = list(self.wdec.parameters())
        decoder_params += list(self.bdec.parameters())
        decoder_params += list(self.cdec.parameters())

        return fc1_params + fc2_params + fc3_params + decoder_params

    def get_entropy_parameters(self):

        fc1_params = self.fc1.get_entropy_parameters()
        fc2_params = self.fc2.get_entropy_parameters()
        fc3_params = self.fc3.get_entropy_parameters()

        return fc1_params + fc2_params + fc3_params

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
        x = self.dropout(self.relu(x))

        x, bpp = self.fc2(x)
        rate = rate + bpp
        x = self.dropout(self.relu(x))

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
        self.relu = nn.LeakyReLU()

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
        self.relu = nn.LeakyReLU()

        self.conv_decoder = AffineDecoder(25)
        # creating the decoders for the linear layers
        self.wdec_lin = AffineDecoder(1)
        # creating the parameters for the biases 
        self.wdec_bias = AffineDecoder(1)
        self.ema_decay = 0
        
        self.conv1 = EntropyConv2d(
            5,
            1,
            20,
            weight_decoder=self.conv_decoder,
            bias_decoder=self.wdec_bias,
            ema_decay=self.ema_decay
        )
        self.conv2 = EntropyConv2d(
            5,
            20,
            50,
            weight_decoder=self.conv_decoder,
            bias_decoder=self.wdec_bias,
            ema_decay=self.ema_decay
        )
        self.fc1 = EntropyLinear(
            4*4*50,
            500,
            self.wdec_lin,
            self.wdec_bias,
            self.ema_decay
        )
        self.fc2 = EntropyLinear(
            500,
            10,
            self.wdec_lin,
            self.wdec_bias,
            self.ema_decay
        )

        # for w, b in zip(self.conv_w_param, self.conv_b_param):
            # self.init_weights(w, b)

        # for w, b in zip(self.lin_w_param, self.lin_b_param):
            # self.init_weights(w, b)


    def get_non_entropy_parameters(self):

        fc1_params = self.fc1.get_non_entropy_parameters()
        fc2_params = self.fc2.get_non_entropy_parameters()
        conv1_params = self.conv1.get_non_entropy_parameters()
        conv2_params = self.conv2.get_non_entropy_parameters()

        decoder_params = list(self.conv_decoder.parameters())
        decoder_params += list(self.wdec_lin.parameters())
        decoder_params += list(self.wdec_bias.parameters())

        return fc1_params + fc2_params + conv1_params + conv2_params + decoder_params

    def get_entropy_parameters(self):

        fc1_params = self.fc1.get_entropy_parameters()
        fc2_params = self.fc2.get_entropy_parameters()
        conv1_params = self.conv1.get_entropy_parameters()
        conv2_params = self.conv2.get_entropy_parameters()

        return fc1_params + fc2_params + conv1_params + conv2_params    

    def get_rate(self):

        p1, t1 = self.fc1.get_compressed_params_size()
        p2, t2 = self.fc2.get_compressed_params_size()
        p3, t3 = self.conv1.get_compressed_params_size()
        p4, t4 = self.conv2.get_compressed_params_size()

        d1 = self.conv_decoder.get_model_size()
        d2 = self.wdec_lin.get_model_size()
        d3 = self.wdec_bias.get_model_size()

        parameters_size = p1 + p2 + p3 + p4
        entropy_model_size = t1 + t2 + t3 + t4 + d1 + d2 + d3
        return parameters_size, entropy_model_size

    def get_original_size(self):

        parameters_size = self.fc1.get_model_size()
        parameters_size += self.fc2.get_model_size()
        parameters_size += self.conv1.get_model_size()
        parameters_size += self.conv2.get_model_size()

        return parameters_size

    def update(self, force=False):
        self.fc1.update(force=force)
        self.fc2.update(force=force)
        self.conv1.update(force=force)
        self.conv2.update(force=force)

    def forward(self, x):
        rate = 0

        x, bpp = self.conv1(x)
        rate = rate + bpp
        x = self.max_pool(self.relu(x))

        x, bpp = self.conv2(x)
        rate = rate + bpp
        x = self.max_pool(self.relu(x))
        
        x = x.view(-1, 4*4*50)

        x, bpp = self.fc1(x)
        rate = rate + bpp
        x = self.relu(x)

        x, bpp = self.fc2(x)
        rate = rate + bpp
        return x, rate

def train_step(model, x, y, opt, device, lambda_RD, criterion):
    x = x.to(device)
    y = y.to(device)
    opt.zero_grad()
    y_hat, rate = model(x)
    loss = criterion(y_hat, y) 
    loss = loss + lambda_RD * rate
    loss.backward()
    opt.step()
    return loss.item(), rate

def test_step(model, test_dataloader, device, lambda_RD, criterion):
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

    final_loss = 0
    i = 0
    for x,y in test_dataloader:
        with th.no_grad():
            x = x.to(device)
            y = y.to(device)
            y_hat, rate = model(x)
            loss = criterion(y_hat, y)
            final_loss += loss.item() + lambda_RD * rate.item()

        cum_loss += loss.item()
        cum_acc += (y_hat.argmax(dim=1) == y).float().mean().item()
        num_steps += 1

    model.train()
    cum_loss = cum_loss / num_steps
    cum_acc = cum_acc / num_steps
    final_loss = final_loss / num_steps
    print(
        f"Test loss: {cum_loss:.3f}, "
        f"Test accuracy: {cum_acc:.3f}, "
        f"Final Loss: {final_loss:.3f}"
    )
    return final_loss / num_steps


class TensorDataset(Dataset):

    def __init__(self, x, y):

        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):

        return self.x[idx].float().unsqueeze(0) / 255, self.y[idx]

