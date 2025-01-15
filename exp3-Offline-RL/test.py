import os
import random
from collections import deque

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import argparse
from tqdm import tqdm
import time
# =========================
# 超参数设置
# =========================
ENV_NAME = "CartPole-v1"

SEED = 42

class PolicyNetwork(nn.Module):
    """
    策略网络：输出对每个离散动作的 log 概率。
    """

    def __init__(self, obs_dim, act_dim, hidden_dim=128):
        super(PolicyNetwork, self).__init__()
        self.net = nn.Sequential(nn.Linear(obs_dim, hidden_dim), 
                                nn.ReLU(), 
                                nn.Linear(hidden_dim, hidden_dim), 
                                nn.ReLU(), 
                                nn.Linear(hidden_dim, act_dim))

    def forward(self, obs):
        """
        返回 logits（未经过 softmax）
        """
        return self.net(obs)

    def get_action(self, obs, deterministic=False):
        """
        给定单个状态，返回离散动作。
        如果 deterministic=True，则选取概率最大的动作。
        否则根据概率分布随机采样动作。
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(obs)
            # 获取各动作的概率分布
            probs = torch.softmax(logits, dim=-1)
            if deterministic:
                action = torch.argmax(probs, dim=-1)
            else:
                # 按照多项式分布进行随机采样
                action = torch.multinomial(probs, 1)
        self.train()
        return action.item()

    def get_log_probs(self, obs):
        """
        给定一个batch状态，返回动作的 log_probs 和对应的概率分布。
        log_probs: [batch_size, act_dim]
        probs:     [batch_size, act_dim]
        """
        with torch.no_grad():
            self.eval()
            logits = self.forward(obs)
            log_probs = torch.log_softmax(logits, dim=-1)
            probs = torch.softmax(logits, dim=-1)
        self.train()
        return log_probs, probs

def parse_args():
    parser = argparse.ArgumentParser(description="Render the agent in the environment.")
    parser.add_argument("--model_path",'-m', type=str, default="model.pth", help="The path to the model file.")
    args = parser.parse_args()
    return args

def main():
    # 设置随机种子，便于结果复现
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)


    # =========================
    # 创建环境
    # =========================
    env = gym.make(ENV_NAME, render_mode="human")
    env.reset(seed=SEED)  # 在 reset 时设置随机种子
    env.action_space.seed(SEED)

    obs_dim = env.observation_space.shape[0]  # 4
    act_dim = env.action_space.n  # 2 (离散动作: 左 or 右)

    args = parse_args()
    # 初始化策略网络
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy_net = PolicyNetwork(obs_dim, act_dim).to(device)
    policy_net.load_state_dict(torch.load(args.model_path, weights_only=True))
    policy_net.eval()

    state , _ = env.reset()
    while True:
        env.render()
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        action = policy_net.get_action(state_tensor, deterministic=True)
        next_state, reward, done, truncated, _ = env.step(action)
        state = next_state
        if done or truncated:
            break
        time.sleep(0.1)
    env.close()

if __name__ == "__main__":
    main()