import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

iris = load_iris()
X = iris.data
y = iris.target

binary_indices = y < 2
X = X[binary_indices]
y = y[binary_indices]
print(len(X))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

batch_size = 16
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

class LogisticRegression(nn.Module):
    def __init__(self, input_dim):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.linear(x))

input_dim = X_train.shape[1]
model = LogisticRegression(input_dim)
loss = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

n_epochs = 100
for epoch in range(n_epochs):
    model.train()
    epoch_loss = 0
    correct = 0
    total = 0

    for X_batch, y_batch in train_loader:
        y_pred = model(X_batch)
        l = loss(y_pred, y_batch)

        # Backward pass and optimization
        optimizer.zero_grad()
        l.backward()
        optimizer.step()

        epoch_loss += l.item()

        predicted = (y_pred > 0.5).float()  # Convert probabilities to binary predictions
        correct += (predicted == y_batch).float().sum().item()
        total += y_batch.size(0)

    accuracy = correct / total
    print(f"Epoch {epoch+1}/{n_epochs}, Loss: {epoch_loss/len(train_loader):.4f}, Accuracy: {accuracy:.4f}")

model.eval()
with torch.no_grad():
    y_pred = model(X_test_tensor)
    y_pred_class = (y_pred > 0.5).float()
    accuracy = (y_pred_class == y_test_tensor).float().mean().item()
    print(f"Test Accuracy: {accuracy:.4f}")

