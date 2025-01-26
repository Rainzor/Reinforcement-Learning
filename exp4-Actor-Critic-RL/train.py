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
import time

from models import *

# =========================
# 超参数设置
# =========================
ENV_NAME = "CartPole-v1"
GAMMA = 0.99  # 折扣因子
LAMBDA = 0.95  # GAE 参数
EPSILON = 0.2  # PPO 算法参数
TAU = 0.01  # 软更新系数
LR_Q = 3e-4  # Q 网络学习率
LR_V = 1e-3  # V 网络学习率
LR_POLICY = 3e-4  # 策略网络学习率
ALPHA = 0.2  # SAC 温度系数 (决定了熵项的比重)
NOISE = 0.1  # 动作噪声
BATCH_SIZE = 64  # 批大小
MEMORY_SIZE = 100000  # Replay Buffer 大小
MAX_EPISODES = 400  # 训练轮数
ONLINE_EPOCHS = 10 # PPO 算法中的在线轮数
MAX_STEPS = 500  # 每个episode最大步数
MAX_GLOBAL_STEPS = 1e5  # 最大训练步数
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
    batch_size: int = BATCH_SIZE
    memory_size: int = MEMORY_SIZE
    max_episodes: int = MAX_EPISODES
    max_steps: int = MAX_STEPS
    max_global_steps: int = MAX_GLOBAL_STEPS
    start_steps: int = START_STEPS
    update_after: int = UPDATE_AFTER
    update_every: int = UPDATE_EVERY
    save_dataset_every: int = SAVE_DATASET_EVERY
    output: str = OUTPUT
    on_policy: bool = False
    continuous: bool = False

@dataclass
class ModelConfig:
    obs_dim: int
    act_dim: int
    act_limit: float = 1.0
    hidden_dim: int = HIDDEN_DIM
    gamma: float = GAMMA
    epsilon: float = EPSILON
    lmbda: float = LAMBDA
    tau: float = TAU
    noise: float = NOISE
    alpha: float = ALPHA
    lr_q: float = LR_Q
    lr_v: float = LR_V
    lr_policy: float = LR_POLICY

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
        return (np.array(states, dtype=np.float32), np.array(actions,dtype=np.float32), np.array(rewards, dtype=np.float32), np.array(next_states, dtype=np.float32), np.array(dones, dtype=np.float32))

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
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.agent = agent.to(self.device)
        
        self.replay_buffer = ReplayBuffer(config.memory_size)
        self.writer = writer
        self.global_step = 0

    def evaluate(self, n_episodes=3):
        eval_reward_list = []
        for _ in range(n_episodes):
            step = 0
            state, _ = self.env.reset()
            done = False
            eval_reward = 0
            truncated = False
            while not done and not truncated:
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                action = self.agent.take_action(state_tensor, deterministic=True)
                next_state, reward, done, truncated, _ = self.env.step(action)
                step += 1
                eval_reward += reward
                state = next_state
            eval_reward_list.append(eval_reward)
        reward = np.mean(eval_reward_list)
        return reward    
   
    def train(self, on_policy=True):
        if on_policy:
            return self._train_on_policy()
        else:
            return self._train_off_policy()
 
    def train_step(self, transition):
        states = torch.FloatTensor(transition['states']).to(self.device)
        if self.config.continuous:
            actions = torch.FloatTensor(transition['actions']).view(-1,1).to(self.device)
        else:
            actions = torch.LongTensor(transition['actions']).view(-1,1).to(self.device)
        rewards = torch.FloatTensor(transition['rewards']).view(-1,1).to(self.device)
        next_states = torch.FloatTensor(transition['next_states']).to(self.device)
        dones = torch.FloatTensor(transition['dones']).view(-1,1).to(self.device)

        self.agent.update((states, actions, rewards, next_states, dones))

    def _train_on_policy(self):
        state, _ = self.env.reset()
        train_reward = 0
        done = False
        transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}
        done = False
        truncated = False

        while not done and not truncated:
            self.global_step += 1

            # 在前 START_STEPS 步，随机选择动作
            if self.global_step < self.config.start_steps:
                action = self.env.action_space.sample()
            else:
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                action = self.agent.take_action(state_tensor, deterministic=False)

            next_state, reward, done, truncated, _ = self.env.step(action)
            transition_dict['states'].append(state)
            transition_dict['actions'].append(action)
            transition_dict['next_states'].append(next_state)
            transition_dict['rewards'].append(reward)
            transition_dict['dones'].append(done)

            state = next_state
            train_reward += reward

        # 更新网络 ON-POLICY
        self.agent.update(transition_dict)
        return train_reward

    def _train_off_policy(self):
        state, _ = self.env.reset()
        train_reward = 0
        done = False
        truncated = False
        while not done and not truncated:
            self.global_step += 1

            # 在前 START_STEPS 步，随机选择动作
            if self.global_step < self.config.start_steps:
                action = self.env.action_space.sample()
            else:
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                action = self.agent.take_action(state_tensor, deterministic=False)

            next_state, reward, done, truncated, _ = self.env.step(action)
            self.replay_buffer.push(state, action, reward, next_state, done)

            state = next_state
            train_reward += reward

            # 在一定步数之后，开始更新网络, OFF-POLICY
            if (self.global_step >= self.config.update_after) and \
                (self.global_step % self.config.update_every == 0):
                for _ in range(self.config.update_every):
                    b_s, b_a, b_r, b_ns, b_d = self.replay_buffer.sample(self.config.batch_size)
                    transition = {'states': b_s, 'actions': b_a, 'rewards': b_r, 'next_states': b_ns, 'dones': b_d}
                    self.train_step(transition)      
        return train_reward

def parse_args():
    parser = argparse.ArgumentParser(description="Actor-Critic RL")

    parser.add_argument("--model", '-m',type=str, default="sac", help="The name of the model.")
    parser.add_argument("--env", '-e', type=str, default=ENV_NAME, help="The name of the environment.")
    parser.add_argument("--continuous", '-c', action="store_true", help="Use continuous action space.")
    parser.add_argument("--epochs", '-n', type=int, default=MAX_EPISODES, help="The number of training epochs.")
    parser.add_argument("--max_steps", '-s', type=int, default=MAX_STEPS, help="The maximum number of steps in each episode.")

    parser.add_argument("--output", '-o', type=str, default=OUTPUT, help="The output directory.")
    parser.add_argument("--lr_q", type=float, default=LR_Q, help="The learning rate of Q network.")
    parser.add_argument("--lr_v", type=float, default=LR_V, help="The learning rate of V network.")
    parser.add_argument("--lr_p", type=float, default=LR_POLICY, help="The learning rate of policy network.")

    parser.add_argument("--tag", type=str, default=None, help="The tag of the experiment.")

    parser.add_argument("--seed", type=int, default=SEED, help="Random seed.")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="The batch size.")

    args = parser.parse_args()
    return args

def main(args):

    # 设置随机种子，便于结果复现
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    env_name = args.env
    env = gym.make(env_name)
    env.reset(seed=args.seed)
    env.action_space.seed(args.seed)
    
    if args.continuous:
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        act_limit = env.action_space.high[0] # assume symmetric
    else:
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n
        act_limit = 1.0
    timenow = time.strftime("%Y-%m-%d-%H-%M", time.localtime())
    if args.tag is not None:
        output_dir = os.path.join(args.output, f"{env_name}-{args.model}-{args.tag}",timenow)
    else:
        output_dir = os.path.join(args.output, f"{env_name}-{args.model}", timenow)

    os.makedirs(output_dir, exist_ok=True)
    train_config = TrainConfig(
        env_name=env_name,
        batch_size=args.batch_size,
        max_episodes=args.epochs,
        output=output_dir
    )
    model_config = ModelConfig(obs_dim=state_dim, act_dim=action_dim, act_limit=act_limit)
    if not args.continuous:
        train_config.continuous = False
        if args.model == "ac":
            agent = ActorCritic(model_config)
            train_config.on_policy = True
            print("Using Actor-Critic Agent")
        elif args.model == "ppo":
            agent = PPO(model_config)
            train_config.on_policy = True
            print("Using PPO Agent")
        elif args.model == "ac-off-policy":
            agent = ActorCriticOffPolicy(model_config)
            train_config.on_policy = False
            print("Using Actor-Critic Off-Policy Agent")
        elif args.model == "sac":
            agent = SAC(model_config)
            train_config.on_policy = False
            print("Using SAC Agent")
        else:
            raise ValueError("Unknown model name")
    else:
        train_config.continuous = True
        if args.model == "sac":
            agent = SACContinuous(model_config)
            train_config.on_policy = False
            print("Using SAC Continuous Agent")
        elif args.model == "ddpg":
            agent = DDPG(model_config)
            train_config.on_policy = False
            print("Using DDPG Continuous Agent")
        elif args.model == "td3":
            agent = TD3(model_config)
            train_config.on_policy = False
            print("Using TD3 Continuous Agent")
        else:
            raise ValueError("Unknown model name")

    timenow = time.strftime("%Y-%m-%d-%H-%M", time.localtime())
    log_dir = os.path.join(output_dir, "logs")
    writer = SummaryWriter( output_dir)
    trainer = Trainer(env=env, agent=agent, writer=writer, config=train_config)

    # =========================
    # 训练循环
    # =========================
    global_step = 0
    best_reward = -np.inf
    eval_reward = 0
    with tqdm(total=train_config.max_episodes) as pbar:
        for episode in range(train_config.max_episodes):
            train_reward = trainer.train(on_policy=False)
            writer.add_scalar("Reward/train", train_reward, episode)
            # 打印训练进度
            if (episode + 1) % 10 == 0:
                eval_reward = trainer.evaluate()
                writer.add_scalar("Reward/eval", eval_reward, episode)
            pbar.set_postfix({
                'global_step': trainer.global_step,
                "train_reward": train_reward,
                "eval_reward": eval_reward
            })
            pbar.update(1)

            if trainer.global_step >= MAX_GLOBAL_STEPS and MAX_GLOBAL_STEPS > 0:
                print(f"Reach max global steps {MAX_GLOBAL_STEPS} at episode {episode}")
                break

    # =========================
    # 测试结果
    # =========================
    test_reward = trainer.evaluate(n_episodes=10)
    print(f"Final evaluation reward (10 episodes): {test_reward:.2f}")

if __name__ == "__main__":
    args = parse_args()
    main(args)