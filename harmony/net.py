
import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self, kind):
        super(Net, self).__init__()
        self.kind = kind
        if self.kind == 0:
            self.cnn = nn.Conv2d(3, 3, 3, stride=2)
            self.fc1 = nn.Linear(5 * 5, 1)
            self.fc2 = nn.Linear(3, 1)
        else:
            self.cnn = nn.Conv2d(3, 1, 3, stride=2)
            # print(self.cnn.weight.shape)
            self.fc1 = nn.Linear(5 * 5, 1)
    
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, input):
        if self.kind == 0:
            out = self.relu(self.cnn(input))
            out = out.view(-1, 3, 5 * 5)
            out = self.relu(self.fc1(out))
            out = out.view(-1, 3)
            out = self.sigmoid(self.fc2(out))
        else:
            out = self.relu(self.cnn(input))
                # print(out.shape)
            out = out.view(-1, 5 * 5)
            out = self.sigmoid(self.fc1(out))
    
        return out

