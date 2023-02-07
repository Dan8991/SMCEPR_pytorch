import torch as th
from torchvision.datasets import MNIST
from torch.optim import Adam
import matplotlib.pyplot as plt
from torchvision import transforms
from models import LeNet, train_step, test_step
from torch.utils.data import DataLoader
from tqdm import tqdm

train_data = MNIST("../datasets", train=True, download=True, transform=transforms.ToTensor())
test_data = MNIST("../datasets", train=False, download=True, transform=transforms.ToTensor())

iterations = 200000
batch_size = 32
lr = 0.001

model = LeNet()
device = th.device("cuda:0" if th.cuda.is_available() else "cpu")
model = model.to(device)
opt = Adam(model.parameters(), lr=lr)

train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

i = 0
while i < iterations:
    cum_loss = 0
    for x, y in tqdm(train_dataloader):
        x = x.to(device)
        y = y.to(device)
        loss = train_step(model, x, y, opt, device)
        cum_loss += loss
        i += 1
    print(f"Train Loss: {cum_loss / 10000}")
    test_step(model, test_dataloader, device)
