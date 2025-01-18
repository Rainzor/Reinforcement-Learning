import numpy as np
import torch

def soft_update(net, target_net, tau=0.01):
    """
    软更新：target_net = tau * net + (1 - tau) * target_net
    """
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

def compute_advantage(gamma, lmbda, td_delta):
    td_delta = td_delta.detach().numpy()
    advantage_list = []
    advantage = 0.0
    for delta in td_delta[::-1]:
        advantage = gamma * lmbda * advantage + delta
        advantage_list.append(advantage)
    advantage_list.reverse()
    advantage_list = np.array(advantage_list)
    return torch.tensor(advantage_list, dtype=torch.float)
