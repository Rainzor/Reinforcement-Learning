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
from torch.utils.tensorboard import SummaryWriter
from dataclasses import dataclass

# =========================
# 超参数设置
# =========================
ENV_NAME = "CartPole-v1"
GAMMA = 0.99  # 折扣因子
TAU = 0.01  # 软更新系数
LR_Q = 3e-4  # Q 网络学习率
LR_POLICY = 3e-4  # 策略网络学习率
ALPHA = 0.2  # SAC 温度系数 (决定了熵项的比重)
BATCH_SIZE = 64  # 批大小
MEMORY_SIZE = 100000  # Replay Buffer 大小
MAX_EPISODES = 400  # 训练轮数
MAX_STEPS = 500  # 每个episode最大步数
START_STEPS = 1000  # 随机探索步数
UPDATE_AFTER = 1000  # 准备好一定量数据再开始更新
UPDATE_EVERY = 50  # 每隔多少步更新一次
SAVE_DATASET_EVERY = 50
SEED = 42
HIDDEN_DIM = 128
OUTPUT = "outputs"  # 模型和数据输出目录

@dataclass
class TrainConfig:
    env_name: str = ENV_NAME
    seed: int = SEED
    batch_size: int = BATCH_SIZE
    memory_size: int = MEMORY_SIZE
    max_episodes: int = MAX_EPISODES
    max_steps: int = MAX_STEPS
    start_steps: int = START_STEPS
    update_after: int = UPDATE_AFTER
    update_every: int = UPDATE_EVERY
    save_dataset_every: int = SAVE_DATASET_EVERY
    output: str = OUTPUT

@dataclass
class ModelConfig:
    obs_dim: int
    act_dim: int
    hidden_dim: int = HIDDEN_DIM
    gamma: float = GAMMA
    tau: float = TAU
    alpha: float = ALPHA
    lr_q: float = LR_Q
    lr_policy: float = LR_POLICY


def soft_update(net, target_net, tau=TAU):
    """
    软更新：target_net = tau * net + (1 - tau) * target_net
    """
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

# =========================
# 定义网络
# =========================


class PolicyNetwork(nn.Module):
    """
    策略网络：输出对每个离散动作的 log 概率。
    """

    def __init__(self, obs_dim, act_dim, hidden_dim=128):
        super(PolicyNetwork, self).__init__()
        self.net = nn.Sequential(
                        nn.Linear(obs_dim, hidden_dim),
                        nn.ReLU(), 
                        nn.Linear(hidden_dim, hidden_dim), 
                        nn.ReLU(), 
                        nn.Linear(hidden_dim, act_dim))

    def forward(self, obs):
        """
        返回 logits（未经过 softmax）
        """
        return self.net(obs)

    def take_action(self, obs, deterministic=False):
        """
        给定单个状态，返回离散动作。
        如果 deterministic=True，则选取概率最大的动作。
        否则根据概率分布随机采样动作。
        """
        with torch.no_grad():
            logits = self.forward(obs)
            # 获取各动作的概率分布
            probs = torch.softmax(logits, dim=-1)
            if deterministic:
                action = torch.argmax(probs, dim=-1)
            else:
                # 按照多项式分布进行随机采样
                action = torch.multinomial(probs, 1)
        return action.item()

    def get_log_probs(self, obs):
        """
        给定一个batch状态，返回动作的 log_probs 和对应的概率分布。
        log_probs: [batch_size, act_dim]
        probs:     [batch_size, act_dim]
        """
        logits = self.forward(obs)
        log_probs = torch.log_softmax(logits, dim=-1)
        probs = torch.softmax(logits, dim=-1)
        return log_probs, probs

class QNetwork(nn.Module):
    """
    Q 网络：Q(s, a)的预测。
    """

    def __init__(self, obs_dim, act_dim, hidden_dim=128):
        super(QNetwork, self).__init__()
        self.net = nn.Sequential(nn.Linear(obs_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, act_dim))
        self.num_actions = act_dim

    def forward(self, obs):
        """
        返回 Q(s, a) 对于每个动作的预测 [batch_size, act_dim]
        """
        return self.net(obs)
    
    def take_action(self, obs, deterministic=False):
        """
        给定单个状态，返回离散动作。
        如果 deterministic=True，则选取概率最大的动作。
        否则根据概率分布随机采样动作。
        """
        with torch.no_grad():
            logits = self.forward(obs)
            # 获取各动作的概率分布
            probs = torch.softmax(logits, dim=-1)

            if deterministic or np.random.random() > EPSILON:
                action = torch.argmax(probs, dim=-1).item()
            else:
                action = np.random.randint(0, self.num_actions)  # Random integer between 0 and num_actions - 1
        return action

class SAC(nn.Module):
    def __init__(self, model_config):
        super(SAC, self).__init__()
        self.model_config = model_config
        self.q1 = QNetwork(model_config.obs_dim, model_config.act_dim, model_config.hidden_dim)
        self.q2 = QNetwork(model_config.obs_dim, model_config.act_dim, model_config.hidden_dim)
        self.q1_target = QNetwork(model_config.obs_dim, model_config.act_dim, model_config.hidden_dim)
        self.q2_target = QNetwork(model_config.obs_dim, model_config.act_dim, model_config.hidden_dim)
        self.policy = PolicyNetwork(model_config.obs_dim, model_config.act_dim, model_config.hidden_dim)

        # 拷贝参数到 target网络
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        # 优化器
        self.q1_optimizer = optim.Adam(self.q1.parameters(), lr=model_config.lr_q)
        self.q2_optimizer = optim.Adam(self.q2.parameters(), lr=model_config.lr_q)
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=model_config.lr_policy)

    def forward(self, obs):
        pass

    def update(self, data):
        states, actions, rewards, next_states, dones = data
        with torch.no_grad():
            # 下一个状态的 log_probs, probs
            next_log_probs, next_probs = self.policy.get_log_probs(next_states)
            # 计算下一个状态中, 对应各个动作的 Q 值
            q1_next = self.q1_target(next_states)  # shape [batch_size, act_dim]
            q2_next = self.q2_target(next_states)  # shape [batch_size, act_dim]

            # 取两个Q网络的最小值
            min_q_next = torch.min(q1_next, q2_next)  # [batch_size, act_dim]

            # 对离散空间，目标值 = r + γ * E_{a' ~ π}[ Q(s',a') - α * log π(a'|s') ]
            # 其中 E_{a' ~ π}[·] 可以用 sum(prob * ·)
            V_next = (next_probs * (min_q_next - self.model_config.alpha * next_log_probs)).sum(dim=-1, keepdim=True)
            target_q = rewards + self.model_config.gamma * (1 - dones) * V_next
        
        q1_values = self.q1(states).gather(dim=1, index=actions)  # [batch_size, 1]
        q2_values = self.q2(states).gather(dim=1, index=actions)  # [batch_size, 1]
        q1_loss = nn.MSELoss()(q1_values, target_q)
        q2_loss = nn.MSELoss()(q2_values, target_q)

        self.q1_optimizer.zero_grad()
        q1_loss.backward()
        self.q1_optimizer.step()

        self.q2_optimizer.zero_grad()
        q2_loss.backward()
        self.q2_optimizer.step()

        # --------------------------
        # 3) 更新 策略网络
        # --------------------------
        log_probs, probs = self.policy.get_log_probs(states)
        # 计算 Q(s,a) 的最小值 (针对所有动作)
        q1_vals = self.q1(states)
        q2_vals = self.q2(states)
        min_q = torch.min(q1_vals, q2_vals)  # [batch_size, act_dim]

        # 期望 J(π) = E_{s ~ D}[ E_{a ~ π}[ α * log π(a|s) - Q(s,a) ] ]
        # 其中对离散动作的期望可以写成 sum(π(a|s)*[α * log π(a|s) - Q(s,a)])
        # 注意这里 log_probs 的 shape = [batch_size, act_dim]
        policy_loss = (probs * (self.model_config.alpha * log_probs - min_q)).sum(dim=1).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        # --------------------------
        # 4) 软更新 target 网络
        # --------------------------
        soft_update(self.q1, self.q1_target, TAU)
        soft_update(self.q2, self.q2_target, TAU)
    
    def take_action(self, obs, deterministic=False):
        return self.policy.take_action(obs, deterministic=deterministic)

# =========================
# 定义经验回放池
# =========================
class ReplayBuffer:
    def __init__(self, capacity=MEMORY_SIZE):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size=BATCH_SIZE):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states, dtype=np.float32), np.array(actions), np.array(rewards, dtype=np.float32), np.array(next_states, dtype=np.float32), np.array(dones, dtype=np.float32))

    def save(self, filename):
        """
        保存回放池中的所有数据到文件中。
        """
        states, actions, rewards, next_states, dones = zip(*self.buffer)
        np.savez_compressed(
            filename,
            states=np.array(states, dtype=np.float32),
            actions=np.array(actions),
            rewards=np.array(rewards, dtype=np.float32),
            next_states=np.array(next_states, dtype=np.float32),
            dones=np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)

class Trainer:

    def __init__(self, env, agent, writer, config):
        self.config = config
        self.env = env

        self.obs_dim = env.observation_space.shape[0]  # 4
        self.act_dim = env.action_space.n  # 2 (离散动作: 左 or 右)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.agent = agent.to(self.device)
        
        self.replay_buffer = ReplayBuffer(config.memory_size)
        self.writer = writer
        self.global_step = 0


    def train_step(self):
        if len(self.replay_buffer) < self.config.batch_size:
            return
        
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.config.batch_size)
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(-1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(-1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(-1).to(self.device)

        self.agent.update((states, actions, rewards, next_states, dones))
    
    def evaluate(self, n_episodes=3):
        eval_reward_list = []
        for _ in range(n_episodes):
            step = 0
            state, _ = self.env.reset()
            done = False
            eval_reward = 0
            while not done and step < self.config.max_steps:
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                action = self.agent.take_action(state_tensor, deterministic=True)
                next_state, reward, done, _, _ = self.env.step(action)
                step += 1
                eval_reward += reward
                state = next_state
            eval_reward_list.append(eval_reward)
        reward = np.mean(eval_reward_list)
        self.writer.add_scalar("Reward/eval", reward, self.global_step)
        return reward
    
    def train(self):
        state, _ = self.env.reset()
        train_reward = 0
        done = False
        for t in range(self.config.max_steps):
            self.global_step += 1

            # 在前 START_STEPS 步，随机选择动作
            if self.global_step < self.config.start_steps:
                action = self.env.action_space.sample()
            else:
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                action = self.agent.take_action(state_tensor, deterministic=False)

            next_state, reward, done, _, _ = self.env.step(action)
            self.replay_buffer.push(state, action, reward, next_state, done)

            state = next_state
            train_reward += reward

            # 在一定步数之后，开始更新网络
            if (self.global_step >= self.config.update_after) and \
                (self.global_step % self.config.update_every == 0):
                for _ in range(self.config.update_every):
                    self.train_step()

            if done:
                break
        self.writer.add_scalar("Reward/train", train_reward, self.global_step)
        
        return train_reward
        
def train_on_policy_agent(env, agent, num_episodes):
    return_list = []
    for i in range(10):
        with tqdm(total=int(num_episodes/10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes/10)):
                episode_return = 0
                transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}
                state = env.reset()
                done = False
                while not done:
                    action = agent.take_action(state)
                    next_state, reward, done, _ = env.step(action)
                    transition_dict['states'].append(state)
                    transition_dict['actions'].append(action)
                    transition_dict['next_states'].append(next_state)
                    transition_dict['rewards'].append(reward)
                    transition_dict['dones'].append(done)
                    state = next_state
                    episode_return += reward
                return_list.append(episode_return)
                agent.update(transition_dict)
                if (i_episode+1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (num_episodes/10 * i + i_episode+1), 'return': '%.3f' % np.mean(return_list[-10:])})
                pbar.update(1)
    return return_list

def train_off_policy_agent(env, agent, num_episodes, replay_buffer, minimal_size, batch_size):
    return_list = []
    for i in range(10):
        with tqdm(total=int(num_episodes/10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes/10)):
                episode_return = 0
                state = env.reset()
                done = False
                while not done:
                    action = agent.take_action(state)
                    next_state, reward, done, _ = env.step(action)
                    replay_buffer.add(state, action, reward, next_state, done)
                    state = next_state
                    episode_return += reward
                    if replay_buffer.size() > minimal_size:
                        b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                        transition_dict = {'states': b_s, 'actions': b_a, 'next_states': b_ns, 'rewards': b_r, 'dones': b_d}
                        agent.update(transition_dict)
                return_list.append(episode_return)
                if (i_episode+1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (num_episodes/10 * i + i_episode+1), 'return': '%.3f' % np.mean(return_list[-10:])})
                pbar.update(1)
    return return_list



def main():

    # 设置随机种子，便于结果复现
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    env_name = ENV_NAME
    env = gym.make(env_name)
    env.reset(seed=SEED)
    env.action_space.seed(SEED)
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    train_config = TrainConfig()
    model_config = ModelConfig(obs_dim=state_dim, act_dim=action_dim)
    agent = SAC(model_config)
    writer = SummaryWriter( "logs")
    trainer = Trainer(env=env, agent=agent, writer=writer, config=train_config)

    # =========================
    # 训练循环
    # =========================
    global_step = 0
    best_reward = -np.inf
    eval_reward = 0
    with tqdm(total=MAX_EPISODES) as pbar:
        for episode in range(MAX_EPISODES):
            train_reward = trainer.train() 
            # 打印训练进度
            if (episode + 1) % 10 == 0:
                eval_reward = trainer.evaluate()
            pbar.set_postfix({
                'global_step': trainer.global_step,
                "train_reward": train_reward,
                "eval_reward": eval_reward
            })
            pbar.update(1)

    # =========================
    # 测试结果
    # =========================
    test_reward = trainer.evaluate(n_episodes=10)
    print(f"Final evaluation reward (10 episodes): {test_reward:.2f}")

if __name__ == "__main__":
    main()