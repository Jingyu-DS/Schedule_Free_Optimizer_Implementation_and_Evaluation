# !pip install schedulefree

# Imports
from __future__ import print_function
import torch
import torchvision
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import gzip
import math
import os
import time
import shutil
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from sklearn.datasets import make_moons
from torch.utils.data import TensorDataset
import schedulefree
import argparse
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

# Utility function
def plot_loss(loss_list, lr):
    plt.figure(figsize=(10, 6))
    plt.plot(loss_list, label='Training Loss')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.title(f'Training Loss Over Steps for Learning Rate {lr}')
    plt.legend()
    plt.grid()
    plt.show()

# Create the nets
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

def train(model, device, train_loader, optimizer, epoch, loss_list):
    model.train()
    optimizer.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        loss_list.append(loss.item())
        if batch_idx % 50 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def test(model, optimizer, device, test_loader):
    model.eval()
    optimizer.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def main():
    # Training settings
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 128
    learning_rate = [0.001, 0.01, 0.05, 0.5]
    epochs = 5

    train_kwargs = {'batch_size': batch_size}
    test_kwargs = {'batch_size': batch_size}

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    dataset1 = datasets.MNIST('../data', train=True, download=True,
                       transform=transform)
    dataset2 = datasets.MNIST('../data', train=False,
                       transform=transform)

    for lr in learning_rate:
      train_loader = torch.utils.data.DataLoader(dataset1, batch_size=batch_size, shuffle=True)
      test_loader = torch.utils.data.DataLoader(dataset2, batch_size=batch_size, shuffle=False)

      model = Net().to(device)
      optimizer = schedulefree.AdamWScheduleFree(model.parameters(), lr=lr) # Things to change

      loss_list = []
      for epoch in range(1, epochs + 1):
          train(model, device, train_loader, optimizer, epoch, loss_list)
          test(model, optimizer, device, test_loader)

      plot_loss(loss_list, lr)


if __name__ == '__main__':
    main()

