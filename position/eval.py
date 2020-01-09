
import torch

from net import Net

def eval_importance(x, y, dx, dy, lengthx, lengthy):
    x /= lengthx
    dx /= lengthx
    y /= lengthy
    dy /= lengthy
    input = torch.Tensor([x, y, dx, dy]).float()
    input.unsqueeze_(0)
    return net(input)[0][0]    


net = Net()
net.load_state_dict(torch.load('model.pkl'))
