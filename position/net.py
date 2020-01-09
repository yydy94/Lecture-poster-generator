
import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self, hidden1=100, hidden2=50):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(4, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
  
    def forward(self, input):
        out = self.relu(self.fc1(input))
        out = self.relu(self.fc2(out))
        out = self.sigmoid(self.fc3(out))
        return out

