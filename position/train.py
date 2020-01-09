
import csv
import torch
import torch.nn as nn
import numpy as np

from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader, SubsetRandomSampler
from torch.optim import Adam
from net import Net

'''
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
'''

with open('train_data.csv', 'r') as f:
    reader = csv.reader(f)
    rows = [row for row in reader]

data_size = len(rows) - 1

data = [[float(x) for x in i] for i in rows[1:]]
data = torch.tensor(data).float()

dataset = TensorDataset(data)
indices = list(range(data_size))
np.random.seed(2017)
np.random.shuffle(indices)
torch.manual_seed(2017)

val_size = data_size // 10
train_idx, val_idx = indices[val_size:], indices[:val_size]

train_sampler = SubsetRandomSampler(train_idx)
val_sampler = SubsetRandomSampler(val_idx)

train_loader = DataLoader(dataset, batch_size=32, sampler=train_sampler)
val_loader = DataLoader(dataset, batch_size=val_size, sampler=val_sampler)

net = Net()
optim = Adam(net.parameters(), lr=0.001)
criterion = nn.MSELoss()

epochs = 500
for i in range(epochs):
    for j, batch in enumerate(train_loader):
        # print(batch)
        batch = batch[0]
        input, label = batch[:, :-1], batch[:, -1]
        label = label.reshape(-1, 1)
        # print(input)
        output = net(input)

        loss = criterion(output, label)
        optim.zero_grad()
        loss.backward()
        optim.step()

    with torch.no_grad():
        for j, batch in enumerate(val_loader):
            batch = batch[0]
            input, label = batch[:, :-1], batch[:, -1]
            label = label.reshape(-1, 1)            
            output = net(input)            

            loss = criterion(output, label)
            print('#{} loss: {}'.format(i, loss))

torch.save(net.state_dict(), 'model.pkl')
