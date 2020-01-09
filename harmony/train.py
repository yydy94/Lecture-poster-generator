
import csv
import torch

import numpy as np
import torch.nn as nn
import matplotlib.image as mpimg
from net import Net

from torch.utils.data import TensorDataset, DataLoader, SubsetRandomSampler
from torch.optim import Adam

'''
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
            # out = self.relu(self.cnn(input))
            out = input.view(-1, 3, 5 * 5)
            out = self.relu(self.fc1(out))
            out = out.view(-1, 3)
            out = self.sigmoid(self.fc2(out))
        else:
            out = self.relu(self.cnn(input))
            # print(out.shape)
            out = out.view(-1, 5 * 5)
            out = self.sigmoid(self.fc1(out))

        return out
'''

seed = 2019
kind = 1

torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

with open('raw_data.csv', 'r') as f:
    reader = csv.reader(f)
    rows = [row for row in reader]

data_size = len(rows) - 1
labels = []
data = []

avg_pool = nn.AdaptiveAvgPool2d(11)

for i in range(1, data_size + 1):
    x, y, dx, dy, label, image_id = [int(rows[i][j]) for j in range(6)]

    rgb = torch.tensor(mpimg.imread('images/{}.jpg'.format(image_id))).float()
    # print(rgb.shape)
    # rgb = rgb.unsqueeze(0)

    rgb = rgb.transpose(0, 2)
    rgb = rgb.transpose(1, 2)
    # rgb = rgb.squeeze(-1)
    # print(rgb.shape)

    rgb = rgb[0:3, y:y + dy, x:x + dx]
    for j in range(3):
        rgb[j] = abs(rgb[j] - rgb[j].mean()) / 255
   
    # if label == 1: print(rgb)
    rgb = np.array(avg_pool(rgb))
   
    # if label == 1: print(rgb)

    data.append(rgb)
    labels.append(label)

data = torch.tensor(data).float()
labels = torch.tensor(labels).float().view(-1, 1)

dataset = TensorDataset(data, labels)
indices = list(range(data_size))
np.random.seed(seed)
np.random.shuffle(indices)

val_size = data_size // 10
print("val_size: ", val_size, data_size)
train_idx, val_idx = indices[val_size:], indices[:val_size]

train_sampler = SubsetRandomSampler(train_idx)
val_sampler = SubsetRandomSampler(val_idx)
print("dataset: {}".format(len(dataset)))


train_loader = DataLoader(dataset, batch_size=8, sampler=train_sampler)
val_loader = DataLoader(dataset, batch_size=val_size, sampler=val_sampler)

net = Net(kind)
# net.load_state_dict(torch.load('model-{}.pkl'.format(4000)))
optim = Adam(net.parameters(), lr=0.001)

criterion = nn.BCELoss()
criterion

epochs = 500
for i in range(epochs):
    losses = 0
    items = 0
    for j, batch in enumerate(train_loader):
        input, label = batch
        
        output = net(input)
        # print(label)
        output = output.view(-1)
        label = label.view(-1)

        loss = criterion(output, label)
        losses += len(label) * loss.item()
        items += len(label)

        optim.zero_grad()
        loss.backward()
        optim.step()
    
    losses /= items

    with torch.no_grad():
        for j, batch in enumerate(val_loader):
            input, label = batch
            output = net(input)
            output = output.view(-1)
            label = label.view(-1)
            
            loss = criterion(output, label)
            print('#{} train_loss: {} val_loss: {}'.format(i, losses, loss))
    

# torch.save(net.state_dict(), 'new_model-{}.pkl'.format(epochs))
