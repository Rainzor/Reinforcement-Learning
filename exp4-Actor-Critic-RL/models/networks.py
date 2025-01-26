import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Normal

class PolicyNet(nn.Module):
    """
    策略网络：输出对每个离散动作的 log 概率。
    """

    def __init__(self, obs_dim, act_dim, hidden_dim=128):
        super(PolicyNet, self).__init__()
        self.net = nn.Sequential(
                        nn.Linear(obs_dim, hidden_dim),
                        nn.ReLU(), 
                        # nn.Linear(hidden_dim, hidden_dim), 
                        # nn.ReLU(), 
                        nn.Linear(hidden_dim, act_dim))

    def forward(self, obs):
        """
        返回 logits（未经过 softmax）
        """
        logits = self.net(obs)
        return logits, torch.log_softmax(logits, dim=-1)

    def take_action(self, obs, deterministic=False):
        """
        给定单个状态，返回离散动作。
        如果 deterministic=True，则选取概率最大的动作。
        否则根据概率分布随机采样动作。
        """
        with torch.no_grad():
            logits, _ = self.forward(obs)
            probs = torch.softmax(logits, dim=-1)
            if deterministic:
                action = torch.argmax(probs, dim=-1)
            else:
                # 按照多项式分布进行随机采样
                action_dist = torch.distributions.Categorical(probs)
                action = action_dist.sample()
        return action.item()

    def get_log_probs(self, obs):
        """
        给定一个batch状态，返回动作的 log_probs 和对应的概率分布。
        log_probs: [batch_size, act_dim]
        probs:     [batch_size, act_dim]
        """
        logits, log_probs = self.forward(obs)
        probs = torch.softmax(logits, dim=-1)
        return log_probs, probs



class PolicyNetContinuous(torch.nn.Module):
    def __init__(self, obs_dim, hidden_dim, act_dim, action_bound):
        super(PolicyNetContinuous, self).__init__()
        self.fc1 = torch.nn.Linear(obs_dim, hidden_dim)
        self.fc_mu = torch.nn.Linear(hidden_dim, act_dim)
        self.fc_std = torch.nn.Linear(hidden_dim, act_dim)
        self.action_bound = action_bound

    def forward(self, x, deterministic=False):
        x = F.relu(self.fc1(x))
        mu = self.fc_mu(x)
        std = F.softplus(self.fc_std(x))
        if deterministic:
            log_prob = None
            action = torch.tanh(mu) * self.action_bound
            return action, log_prob
        else:
            dist = Normal(mu, std)
            normal_sample = dist.rsample()  # rsample()是重参数化采样
            log_prob = dist.log_prob(normal_sample)
            action = torch.tanh(normal_sample)
            # 计算tanh_normal分布的对数概率密度
            log_prob = log_prob - torch.log(1 - torch.tanh(action).pow(2) + 1e-7)
            action = action * self.action_bound
            return action, log_prob
    
    def take_action(self, obs, deterministic=False):
        """
        给定单个状态，返回连续动作。
        如果 deterministic=True，则选取均值作为动作。
        否则根据均值和标准差进行正态分布采样。
        """
        with torch.no_grad():
            x = F.relu(self.fc1(obs))
            mu = self.fc_mu(x)
            std = F.softplus(self.fc_std(x))
            if deterministic:
                action = torch.tanh(mu) * self.action_bound
            else:
                dist = Normal(mu, std)
                action = torch.tanh(dist.sample()) * self.action_bound
        return [action.item()]

    def get_log_probs(self, obs):
        """
        给定一个batch状态，返回动作的 log_probs 和对应的概率分布。
        log_probs: [batch_size, act_dim]
        probs:     [batch_size, act_dim]
        """
        _, log_probs = self.forward(obs)
        probs = torch.exp(log_probs)
        return log_probs, probs
    
    def get_probs(self, obs, action):
        """
        给定一个batch状态和动作，返回动作的概率。
        """
        x = F.relu(self.fc1(obs))
        mu = self.fc_mu(x)
        std = F.softplus(self.fc_std(x))
        dist = Normal(mu, std)
        action = action / self.action_bound
        action_a = torch.atanh(action)
        log_prob = dist.log_prob(action_a)
        log_prob = log_prob - torch.log(1 - action.pow(2) + 1e-7)
        return torch.exp(log_prob)


class QNetwork(nn.Module):
    """
    Q 网络：Q(s, a)的预测。
    """

    def __init__(self, obs_dim, act_dim, hidden_dim=128):
        super(QNetwork, self).__init__()
        self.net = nn.Sequential(
                            nn.Linear(obs_dim, hidden_dim), 
                            nn.ReLU(), 
                            nn.Linear(hidden_dim, hidden_dim), 
                            nn.ReLU(), 
                            nn.Linear(hidden_dim, act_dim))
        self.num_actions = act_dim

    def forward(self, obs):
        """
        返回 Q(s, a) 对于每个动作的预测 [batch_size, act_dim]
        """
        return self.net(obs)

class QValueNet(torch.nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim=128):
        super(QValueNet, self).__init__()
        self.fc1 = torch.nn.Linear(obs_dim + act_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x, a):
        # print(x.shape, a.shape)
        cat = torch.cat([x, a], dim=1) # 拼接状态和动作
        x = F.relu(self.fc1(cat))
        x = F.relu(self.fc2(x))
        return self.fc_out(x)

class ValueNet(nn.Module):
    """
    值网络：V(s)的预测。
    """

    def __init__(self, obs_dim, hidden_dim=128):
        super(ValueNet, self).__init__()
        self.net = nn.Sequential(
                        nn.Linear(obs_dim, hidden_dim), 
                        nn.ReLU(), 
                        # nn.Linear(hidden_dim, hidden_dim), 
                        # nn.ReLU(), 
                        nn.Linear(hidden_dim, 1))

    def forward(self, obs):
        """
        返回 V(s) 的预测值
        """
        return self.net(obs)
