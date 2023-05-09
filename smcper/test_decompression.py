import torch as th
from torchvision.datasets import MNIST
from torch.optim import Adam
import matplotlib.pyplot as plt
from torchvision import transforms
from models import EntropyLeNet, LeNet, CafeLeNet, EntropyCafeLeNet, train_step, test_step, TensorDataset, EntropyCafeLeNet
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import os
import numpy as np

train_data = MNIST("../datasets", train=True, download=True, transform=transforms.ToTensor())
train_subset, val_subset = th.utils.data.random_split(
        train_data, [50000, 10000], generator=th.Generator().manual_seed(1))

X_train = train_subset.dataset.data[train_subset.indices]
y_train = train_subset.dataset.targets[train_subset.indices]

X_val = val_subset.dataset.data[val_subset.indices]
y_val = val_subset.dataset.targets[val_subset.indices]

train_data = TensorDataset(X_train, y_train)
val_data = TensorDataset(X_val, y_val)

test_data = MNIST("../datasets", train=False, download=True, transform=transforms.ToTensor())

batch_size = 128

model = EntropyLeNet()
print("Model size and decoder size for the network:", model.get_original_size())
device = th.device("cuda:0" if th.cuda.is_available() else "cpu")

model.update()

with open("tmp/compressed.bin", "rb") as f:
    b = f.read()

model.decompress(b)

model.update(force=True, super_up=False)
model = model.to(device)
test_dataloader = DataLoader(
    test_data,
    batch_size=batch_size,
    shuffle=False,
    num_workers=os.cpu_count()
)

criterion = th.nn.CrossEntropyLoss().to(device)
lambda_RD = 1

print("testing")
test_loss = test_step(model, tqdm(test_dataloader), device, lambda_RD, criterion)
