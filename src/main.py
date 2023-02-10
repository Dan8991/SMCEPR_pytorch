import torch as th
from torchvision.datasets import MNIST
from torch.optim import Adam
import matplotlib.pyplot as plt
from torchvision import transforms
from models import EntropyLeNet, LeNet, CafeLeNet, EntropyCafeLeNet, train_step, test_step
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

train_data = MNIST("../datasets", train=True, download=True, transform=transforms.ToTensor())
test_data = MNIST("../datasets", train=False, download=True, transform=transforms.ToTensor())

iterations = 200000
batch_size = 1024
lr = 0.001

model = EntropyLeNet()
print("Model size and decoder size for the network:", model.get_original_size())
device = th.device("cuda:0" if th.cuda.is_available() else "cpu")
model = model.to(device)
opt = Adam([
    {"params": model.get_non_entropy_parameters(), "lr":lr},
    {"params": model.get_entropy_parameters(), "lr":1e-4}
])

train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=os.cpu_count())
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=os.cpu_count())

criterion = th.nn.CrossEntropyLoss().to(device)
i = 0
lambda_RD = 1
while i < iterations:
    cum_loss = 0
    cum_rate = 0
    num_steps = 0
    for x, y in tqdm(train_dataloader):
        x = x.to(device)
        y = y.to(device)
        loss, rate = train_step(
            model,
            x,
            y,
            opt,
            device,
            lambda_RD,
            criterion
        )
        cum_loss += loss
        cum_rate += rate
        i += 1
        num_steps += 1
        if i % 1000 == 0:
            test_step(model, test_dataloader, device, criterion)
    print(f"Train Loss: {cum_loss / num_steps}, {cum_rate / num_steps}")
