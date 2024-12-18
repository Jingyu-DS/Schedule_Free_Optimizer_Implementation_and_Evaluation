# !pip install schedulefree

import tensorflow_datasets as tfds
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Load smallNORB from TFDS
dataset, info = tfds.load("smallnorb", with_info=True, as_supervised=False)
train_tfds = dataset['train']
test_tfds = dataset['test']

num_classes = info.features['label_category'].num_classes  # Should be 5
image_shape = info.features['image'].shape  # (96,96,1)
image_height, image_width, _ = image_shape

# Convert TFDS datasets to PyTorch-compatible datasets
class SmallNORBDataset(Dataset):
    def __init__(self, tfds_dataset):
        self.data = list(tfds_dataset)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        example = self.data[idx]
        # Extract left image and label
        # image_left: shape (96,96,1) and uint8
        img = example['image'].numpy()
        label = example['label_category'].numpy()

        # Convert to torch tensor and normalize
        img = torch.tensor(img, dtype=torch.float32) / 255.0  # now (96,96,1)
        img = img.permute(2, 0, 1)  # Change to (1,96,96) for consistency

        return img, label

train_dataset = SmallNORBDataset(train_tfds)
test_dataset = SmallNORBDataset(test_tfds)

# Create DataLoaders
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# Define a simple model
class MyModel(nn.Module):
    def __init__(self, num_classes=5):
        super(MyModel, self).__init__()
        # Flatten 96*96 = 9216 features
        self.fc = nn.Linear(96*96, num_classes)

    def forward(self, x):
        # x: (batch_size, 1, 96, 96)
        x = x.view(x.size(0), -1)  # Flatten to (batch_size, 9216)
        x = self.fc(x)             # (batch_size, num_classes)
        return x

model = MyModel(num_classes=num_classes)

criterion = nn.CrossEntropyLoss()
optimizer = schedulefree.SGDScheduleFree(model.parameters(), lr=0.01)

n_epochs = 100
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

for epoch in range(n_epochs):
    model.train()
    optimizer.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == y_batch).sum().item()
        total += y_batch.size(0)

    train_loss = running_loss / len(train_loader)
    train_acc = correct / total
    print(f"Epoch [{epoch+1}/{n_epochs}] - Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}")

model.eval()
correct = 0
total = 0

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        outputs = model(X_batch)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == y_batch).sum().item()
        total += y_batch.size(0)

test_acc = correct / total
print(f"Test Accuracy: {test_acc:.4f}")
