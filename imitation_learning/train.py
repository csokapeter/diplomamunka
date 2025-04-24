import pandas as pd
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler

df = pd.read_csv('expert_act_prep_5_action.csv', header=None)

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

X_train, X_eval, y_train, y_eval = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_eval_tensor = torch.tensor(X_eval, dtype=torch.float32)
y_eval_tensor = torch.tensor(y_eval, dtype=torch.long)

y_train = y_train.astype(int)
class_counts = np.bincount(y_train)
class_weights = 1.0 / class_counts
sample_weights = class_weights[y_train]
sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

batch_size = 256
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
eval_dataset = TensorDataset(X_eval_tensor, y_eval_tensor)

train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)

class MLP(nn.Module):
    def __init__(self, input_size, num_classes, dropout_rate=0.5):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(256, 256)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(256, 256)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.output = nn.Linear(256, num_classes)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.tanh(self.fc1(x))
        x = self.dropout1(x)
        x = self.tanh(self.fc2(x))
        x = self.dropout1(x)
        x = self.tanh(self.fc3(x))
        x = self.dropout1(x)
        x = self.output(x)
        return x

input_size = X.shape[1]
num_classes = len(np.unique(y))
model = MLP(input_size, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 300
train_losses, eval_losses = [], []

for epoch in tqdm(range(epochs)):
    model.train()
    running_loss = 0.0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    train_losses.append(running_loss / len(train_loader))

    model.eval()
    eval_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_X, batch_y in eval_loader:
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            eval_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == batch_y).sum().item()
            total += batch_y.size(0)

    eval_losses.append(eval_loss / len(eval_loader))
    accuracy = correct / total * 100
    print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_losses[-1]:.4f}, Eval Loss: {eval_losses[-1]:.4f}, Accuracy: {accuracy:.2f}%')

torch.save(model.state_dict(), 'pretrained_tanh.pth')
print('Model weights saved to pretrained.pth')

plt.plot(range(1, epochs+1), train_losses, label='Train Loss')
plt.plot(range(1, epochs+1), eval_losses, label='Eval Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Evaluation Loss')
plt.legend()
plt.show()
