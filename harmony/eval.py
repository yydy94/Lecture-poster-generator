
import torch
import torch.nn as nn 
import matplotlib.image as mpimg
import numpy as np

from net import Net

def eval_harmony(x, y, dx, dy, image_id):
    rgb = torch.Tensor(mpimg.imread('images/{}.jpg'.format(image_id))).float()
  
    rgb = rgb.transpose(0, 2)
    rgb = rgb.transpose(1, 2)
 
    rgb = rgb[0:3, y:y + dy, x:x + dx]
    rgb = abs(rgb - rgb.mean()) / 255
    rgb = avg_pool(rgb)
    rgb.unsqueeze_(0)

    return net(rgb)[0][0]



net = Net(0)
net.load_state_dict(torch.load('model-6000.pkl'))
avg_pool = nn.AdaptiveAvgPool2d(5)

