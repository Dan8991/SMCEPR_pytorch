import torch as th
from torchvision.datasets import MNIST
from torch.optim import Adam
import matplotlib.pyplot as plt
from torchvision import transforms
from models import EntropyLeNet, LeNet, CafeLeNet, EntropyCafeLeNet, train_step, test_step, TensorDataset
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
val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=os.cpu_count())
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=os.cpu_count())

criterion = th.nn.CrossEntropyLoss().to(device)
i = 0
lambda_RD = 0.5
best_model = None
best_loss = np.inf
epoch = 0
iters_without_best = 0
while i < iterations:
    cum_loss = 0
    cum_rate = 0
    num_steps = 0
    epoch += 1
    for x, y in train_dataloader:
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
    if epoch % 5 == 0:
        val_loss = test_step(
            model,
            val_dataloader,
            device,
            lambda_RD,
            criterion
        )
        if val_loss < best_loss:
            print("FOUND BEST MODEL")
            best_loss = val_loss
            best_model = model.state_dict()
            iters_without_best = 0
        else:
            iters_without_best += 1

        print(f"Train Loss: {cum_loss / num_steps}, {cum_rate / num_steps}")
        print()
    if iters_without_best > 10:
        break

model.load_state_dict(best_model)
test_loss = test_step(model, test_dataloader, device, lambda_RD, criterion)
