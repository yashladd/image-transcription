"""
Training code for EMNIST classification model
"""
import string
import sys
from copy import deepcopy
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

EPOCHS = 150
LEARNING_RATE = 1e-2
BATCH_SIZE = 1024
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {DEVICE} device")

fix_transform = transforms.Compose(
    [
        lambda img: transforms.functional.rotate(img, -90),
        transforms.functional.hflip,
        transforms.ToTensor(),
    ]
)

training_data = datasets.EMNIST(
    root="data",
    split="letters",
    train=True,
    download=True,
    transform=fix_transform,
    target_transform=lambda x: x - 1,
)

test_data = datasets.EMNIST(
    root="data",
    split="letters",
    train=False,
    download=True,
    transform=fix_transform,
    target_transform=lambda x: x - 1,
)

print(training_data)
print(test_data)

figure = plt.figure(figsize=(8, 8))
cols, rows = 2, 3
for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(training_data), size=(1,)).item()
    img, label = training_data[sample_idx]
    figure.add_subplot(rows, cols, i)
    plt.title(list(string.ascii_uppercase)[label])
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="gray")
plt.suptitle("EMNIST images")
plt.savefig("examples.png")

train_dataloader = DataLoader(training_data, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True)


def train_loop(dataloader, model, loss_fn, optimizer):
    # set model to train mode
    model.train()

    iter_loader = iter(dataloader)
    tbar_loader = tqdm(iter_loader, dynamic_ncols=True)

    for X, y in tbar_loader:
        # move images to GPU if needed
        X, y = X.to(DEVICE), y.to(DEVICE)

        # zero gradients from previous step
        optimizer.zero_grad()

        # compute prediction and loss
        pred = model(X)  # Remember forward()? This calls that.
        loss = loss_fn(pred, y)

        # backpropagation
        loss.backward()
        optimizer.step()

        tbar_loader.set_description(f"train loss: {loss.item():.5f}")


def test_loop(dataloader, model, loss_fn):
    # set model to eval mode
    model.eval()

    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    iter_loader = iter(dataloader)
    tbar_loader = tqdm(iter_loader, dynamic_ncols=True)

    with torch.no_grad():
        for X, y in tbar_loader:
            # move images to GPU if needed
            X, y = X.to(DEVICE), y.to(DEVICE)

            # compute prediction and loss
            pred = model(X)
            loss = loss_fn(pred, y).item()
            test_loss += loss

            # compare predictions and labels
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

            tbar_loader.set_description(f"test loss: {loss:.5f}")

    test_loss /= num_batches
    correct /= size
    print(
        f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n"
    )

    return correct


# initialize model
model = models.resnet50()
model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
model.to(DEVICE)

# initialize loss function
loss_fn = nn.CrossEntropyLoss()

# initialize optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.97)

best = -1
best_state = None
try:
    for t in range(EPOCHS):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(train_dataloader, model, loss_fn, optimizer)
        curr = test_loop(test_dataloader, model, loss_fn)
        if curr > best:
            best = curr
            best_state = deepcopy(model.state_dict())
        scheduler.step()

except KeyboardInterrupt:
    print("saving model and exiting...")
    torch.save(best_state, "model.pth")
    sys.exit(-1)

# save model
print("saving model...")
torch.save(best_state, "model.pth")
print("Done!")
