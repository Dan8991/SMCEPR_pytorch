from torch import nn
import torch as th

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


