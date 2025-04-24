import torch
import torch.nn as nn
import numpy as np

class MLP(nn.Module):
    def __init__(self, input_size, num_classes, dropout_rate=0.5, act='relu'):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(256, 256)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(256, 256)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.output = nn.Linear(256, num_classes)
        self.act = nn.ReLU()
        if act == 'tanh':
            self.act = nn.Tanh()

    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.dropout1(x)
        x = self.act(self.fc2(x))
        x = self.dropout1(x)
        x = self.act(self.fc3(x))
        x = self.dropout1(x)
        x = self.output(x)
        return x
    
    def select_action(self, obs, valid_action_mask):
        valid_indices = np.where(valid_action_mask)[0]
        obs = torch.tensor(obs, dtype=torch.float32)
        dist = self.forward(obs)
        action = int(torch.argmax(dist))

        if action in valid_indices:
            return action
        else:
            return np.random.choice(valid_indices)
