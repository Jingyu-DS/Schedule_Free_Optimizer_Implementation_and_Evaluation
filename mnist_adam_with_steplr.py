import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False, num_workers=2)

class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(28*28, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = SimpleNet().to(device)

optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = StepLR(optimizer, step_size=3, gamma=0.1)

num_epochs = 5

# Lists to store losses
train_losses = []
test_losses = []
test_accuracies = []

def train(epoch):
    model.train()
    running_loss = 0.0
    total = 0
    correct = 0

    for batch_idx, (data, targets) in enumerate(train_loader):
        data, targets = data.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(data)
        loss = F.cross_entropy(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * data.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total
    print(f"Train Epoch: {epoch} | Loss: {epoch_loss:.4f} | Accuracy: {epoch_acc:.2f}%")
    return epoch_loss

def test():
    model.eval()
    test_loss = 0.0
    total = 0
    correct = 0

    with torch.no_grad():
        for data, targets in test_loader:
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)
            loss = F.cross_entropy(outputs, targets)
            test_loss += loss.item() * data.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    test_loss = test_loss / total
    accuracy = 100. * correct / total
    return test_loss, accuracy

for epoch in range(1, num_epochs+1):
    train_loss = train(epoch)
    valid_loss, valid_acc = test()
    print(f"Test Epoch: {epoch} | Loss: {valid_loss:.4f} | Accuracy: {valid_acc:.2f}%")

    # Save metrics
    train_losses.append(train_loss)
    test_losses.append(valid_loss)
    test_accuracies.append(valid_acc)

    # Step the scheduler at the end of each epoch
    scheduler.step()
