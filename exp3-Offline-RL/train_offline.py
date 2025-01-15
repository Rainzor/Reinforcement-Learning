import os
import random
from collections import deque

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import time
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

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
MAX_GLOBAL_STEPS = 50000  # 最大训练步数
MAX_STEPS = 500  # 每个episode最大步数
START_STEPS = 1000  # 随机探索步数
UPDATE_AFTER = 1000  # 准备好一定量数据再开始更新
UPDATE_EVERY = 50  # 每隔多少步更新一次
Q_NETWORK_ITERATION =50 # Q 网络更新次数
SAVE_DATASET_EVERY = 50
EPSILON = 0.01  # epsilon-greedy
SEED = 42
PATIENCE = 50  # 早停等待轮数
OUTPUT = "outputs"
DATASET_SIZE = 8192

os.makedirs(OUTPUT, exist_ok=True)

# =========================
# CQL 参数
# =========================
BETA = 5  # CQL 中的超参数
N_ACTIONS = 4  # CQL 中的超参数，对动作空间进行采样的数量


class PolicyNetwork(nn.Module):
    """
    策略网络：输出对每个离散动作的 log 概率。
    """

    def __init__(self, obs_dim, act_dim, hidden_dim=128):
        super(PolicyNetwork, self).__init__()
        self.net = nn.Sequential(nn.Linear(obs_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, act_dim))

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
        self.act_dim = act_dim
        self.net = nn.Sequential(nn.Linear(obs_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, act_dim))

    def forward(self, obs):
        """
        返回 Q(s, a) 对于每个动作的预测 [batch_size, act_dim]
        """
        return self.net(obs)
    
    def get_action(self, obs, EPSILON=0.01):
        if np.random.random() > EPSILON:  # Greedy policy
            action_value = self.forward(obs)
            action = torch.argmax(action_value).item()
        else:
            # Random policy
            action = np.random.randint(0, self.act_dim)  # Random integer
        return action


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
        
    def read(self, filename, load_size=None):
        """
        从文件中读取数据到回放池中。
        """
        data = np.load(filename)
        states = data["states"]
        actions = data["actions"]
        rewards = data["rewards"]
        next_states = data["next_states"]
        dones = data["dones"]
        if load_size is not None and load_size < len(states):
            shuffle_idx = np.random.permutation(len(states))
            shuffle_idx = shuffle_idx[:load_size]
            states = states[shuffle_idx]
            actions = actions[shuffle_idx]
            rewards = rewards[shuffle_idx]
            next_states = next_states[shuffle_idx]
            dones = dones[shuffle_idx]
        print(f"Load data from {filename}, size: {len(states)}")
        for i in range(len(states)):
            self.push(states[i], actions[i], rewards[i], next_states[i], dones[i])

    def __len__(self):
        return len(self.buffer)

def parse_args():
    parser = argparse.ArgumentParser(description="CQL")
    parser.add_argument("--dataset", type=str, default="dataset_episode_350.npz", help="dataset path")
    parser.add_argument("--load_size", type=int, default=None, help="load size of dataset")

    parser.add_argument("--epochs",'-n', type=int, default=MAX_EPISODES, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="batch size")    
    parser.add_argument("--beta", type=float, default=BETA, help="CQL hyperparameter")
    parser.add_argument("--gamma", type=float, default=GAMMA, help="discount factor")
    parser.add_argument("--n_actions", type=int, default=N_ACTIONS, help="CQL hyperparameter")

    parser.add_argument("--tag", type=str, default="CQL", help="tag for the experiment")
    parser.add_argument("--patience", type=int, default=PATIENCE, help="early stopping patience")
    parser.add_argument("--method", type=str, default="DQL", choices=["SAC", "DQL"], help="method to use")
    parser.add_argument("--seed", type=int, default=SEED, help="random seed")

    args = parser.parse_args()
    return args

def main(args):
    # 设置随机种子，便于结果复现
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


    # =========================
    # 创建环境
    # =========================
    env = gym.make(ENV_NAME)  # 移除 seed 参数
    env.reset(seed=args.seed)  # 在 reset 时设置随机种子
    env.action_space.seed(args.seed)

    obs_dim = env.observation_space.shape[0]  # 4
    act_dim = env.action_space.n  # 2 (离散动作: 左 or 右)


    # =========================
    # 定义网络
    # =========================

    # 初始化两个Q网络和对应的target网络，以及一个策略网络
    q1 = QNetwork(obs_dim, act_dim)
    q2 = QNetwork(obs_dim, act_dim)
    q1_target = QNetwork(obs_dim, act_dim)
    q2_target = QNetwork(obs_dim, act_dim)

    policy = PolicyNetwork(obs_dim, act_dim)

    # 拷贝参数到 target网络
    q1_target.load_state_dict(q1.state_dict())
    q2_target.load_state_dict(q2.state_dict())

    # 优化器
    q1_optimizer = optim.Adam(q1.parameters(), lr=LR_Q)
    q2_optimizer = optim.Adam(q2.parameters(), lr=LR_Q)
    policy_optimizer = optim.Adam(policy.parameters(), lr=LR_POLICY)

    replay_buffer = ReplayBuffer(MEMORY_SIZE)
    replay_buffer.read(args.dataset, args.load_size)

    # 将网络移动到 GPU 或 CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    q1.to(device)
    q2.to(device)
    q1_target.to(device)
    q2_target.to(device)
    policy.to(device)


    def soft_update(net, target_net, tau=TAU):
        """
        软更新：target_net = tau * net + (1 - tau) * target_net
        """
        for param, target_param in zip(net.parameters(), target_net.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
    
    def hard_update(net, target_net):
        """
        硬更新：target_net = net
        """
        target_net.load_state_dict(net.state_dict())

    def SAC(data):
        """
        SAC 算法
        """
        states, actions, rewards, next_states, dones = data
        # --------------------------
        # 1) 计算目标 Q 值
        # --------------------------
        with torch.no_grad():
            # 下一个状态的 log_probs, probs
            next_log_probs, next_probs = policy.get_log_probs(next_states)
            # 计算下一个状态中, 对应各个动作的 Q 值
            q1_next = q1_target(next_states)  # shape [batch_size, act_dim]
            q2_next = q2_target(next_states)  # shape [batch_size, act_dim]

            # 取两个Q网络的最小值
            min_q_next = torch.min(q1_next, q2_next)  # [batch_size, act_dim]

            # 对离散空间，目标值 = r + γ * E_{a' ~ π}[ Q(s',a') - α * log π(a'|s') ]
            # 其中 E_{a' ~ π}[·] 可以用 sum(prob * ·)
            V_next = (next_probs * (min_q_next - ALPHA * next_log_probs)).sum(dim=-1, keepdim=True)
            target_q = rewards + args.gamma * (1 - dones) * V_next

        # --------------------------
        # 2) 更新 Q1, Q2
        # --------------------------
        q1_values = q1(states).gather(dim=1, index=actions)  # [batch_size, 1]
        q2_values = q2(states).gather(dim=1, index=actions)  # [batch_size, 1]
        q1_loss = nn.MSELoss()(q1_values, target_q)
        q2_loss = nn.MSELoss()(q2_values, target_q)


        #! CQL 中的额外损失, 用于约束 Q(s,a) 的值
        # 计算 Q1, Q2 对所有动作的值
        if args.beta > 0:
            action_unif = torch.randint(0, act_dim, (BATCH_SIZE, args.n_actions)).to(device)
            q1_unif = q1(states).gather(dim=1, index=action_unif) # [batch_size, N_ACTIONS]
            q2_unif = q2(states).gather(dim=1, index=action_unif) # [batch_size, N_ACTIONS]

            cql1_loss = (torch.logsumexp(q1_unif, dim=1)).mean() - q1_values.mean()
            cql2_loss = (torch.logsumexp(q2_unif, dim=1)).mean() - q2_values.mean()

            qf1_loss = q1_loss + args.beta * cql1_loss
            qf2_loss = q2_loss + args.beta * cql2_loss
        else:
            qf1_loss = q1_loss
            qf2_loss = q2_loss

        q1_optimizer.zero_grad()
        qf1_loss.backward()
        q1_optimizer.step()

        q2_optimizer.zero_grad()
        qf2_loss.backward()
        q2_optimizer.step()

        # --------------------------
        # 3) 更新 策略网络
        # --------------------------
        log_probs, probs = policy.get_log_probs(states)
        # 计算 Q(s,a) 的最小值 (针对所有动作)
        q1_vals = q1(states)
        q2_vals = q2(states)
        min_q = torch.min(q1_vals, q2_vals)  # [batch_size, act_dim]

        # 期望 J(π) = E_{s ~ D}[ E_{a ~ π}[ α * log π(a|s) - Q(s,a) ] ]
        # 其中对离散动作的期望可以写成 sum(π(a|s)*[α * log π(a|s) - Q(s,a)])
        # 注意这里 log_probs 的 shape = [batch_size, act_dim]
        policy_loss = (probs * (ALPHA * log_probs - min_q)).sum(dim=1).mean()

        policy_optimizer.zero_grad()
        policy_loss.backward()
        policy_optimizer.step()

        # --------------------------
        # 4) 软更新 target 网络
        # --------------------------
        soft_update(q1, q1_target, TAU)
        soft_update(q2, q2_target, TAU)
    
    def DQL(data):
        """
        Deep Q Learning 算法
        """
        states, actions, rewards, next_states, dones = data

        q_eval = q1(states).gather(dim=1, index=actions)  # [batch_size, 1]

        with torch.no_grad():
            actions_eval = q1(next_states).argmax(dim=1).unsqueeze(-1)  # [batch_size, 1]
            q_next = q1_target(next_states).gather(dim=1, index=actions_eval)  # [batch_size, 1]
            q_target = rewards + args.gamma * q_next * (1 - dones)
        
        loss = nn.MSELoss()(q_eval, q_target)
        
        #! CQL 中的额外损失, 用于约束 Q(s,a) 的值
        if args.beta > 0:
            # action_unif = torch.arange(0, act_dim)\
            #                 .unsqueeze(0).repeat(BATCH_SIZE, 1).to(device)  # [batch_size, act_dim]
            action_unif = torch.randint(0, act_dim, (BATCH_SIZE, args.n_actions)).to(device)
            q1_unif = q1(states).gather(dim=1, index=action_unif)

            cql1_loss = (torch.logsumexp(q1_unif, dim=1)).mean() - q_eval.mean()

            loss = loss + args.beta * cql1_loss

        q1_optimizer.zero_grad()
        loss.backward()
        q1_optimizer.step()

    def train_step():
        """
        从 replay buffer 抽样并更新一次网络
        """
        if len(replay_buffer) < BATCH_SIZE:
            return

        states, actions, rewards, next_states, dones = replay_buffer.sample(BATCH_SIZE)

        states = torch.FloatTensor(states).to(device)
        actions = torch.LongTensor(actions).unsqueeze(-1).to(device)  # shape [batch_size, 1]
        rewards = torch.FloatTensor(rewards).unsqueeze(-1).to(device)  # shape [batch_size, 1]
        next_states = torch.FloatTensor(next_states).to(device)
        dones = torch.FloatTensor(dones).unsqueeze(-1).to(device)

        if args.method == "SAC":
            SAC((states, actions, rewards, next_states, dones))
            soft_update(q1, q1_target, TAU)
            soft_update(q2, q2_target, TAU)
        elif args.method == "DQL":
            DQL((states, actions, rewards, next_states, dones))
            if global_step % Q_NETWORK_ITERATION == 0:
                hard_update(q1, q1_target)


    def evaluate_policy(n_episodes=5):
        """
        测试当前策略：在测试时使用确定性策略（选最大概率动作）
        返回平均回合总奖励
        """
        eval_rewards = []
        for _ in range(n_episodes):
            step = 0
            state, _ = env.reset()
            done = False
            episode_reward = 0
            while not done and step < MAX_STEPS:
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                with torch.no_grad():
                    if args.method == "DQL":
                        action = q1.get_action(state_tensor, 0)
                    else:
                        action = policy.get_action(state_tensor, deterministic=True)
                next_state, reward, done, _, _ = env.step(action)
                step += 1
                episode_reward += reward
                state = next_state
            eval_rewards.append(episode_reward)
        return np.mean(eval_rewards)

    # =========================
    # 训练循环
    # =========================
    global_step = 0
    best_reward = -np.inf
    timenow = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime(time.time()))
    save_dir = os.path.join(OUTPUT, f"{args.method}_{args.tag}_{timenow}")
    os.makedirs(save_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=save_dir)
    patience = 0
    with tqdm(total=args.epochs) as pbar:
        for episode in range(args.epochs):
            state, _ = env.reset()
            episode_reward = 0
            done = False
            for t in range(MAX_STEPS):
                global_step += 1

                # 在前 START_STEPS 步，随机选择动作
                if global_step < START_STEPS:
                    action = env.action_space.sample()
                else:
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                    with torch.no_grad():
                        if args.method == "DQL":
                            action = q1.get_action(state_tensor, EPSILON)
                        else:
                            action = policy.get_action(state_tensor, deterministic=False)

                next_state, reward, done, _, _ = env.step(action)

                state = next_state
                episode_reward += reward

                # 在一定步数之后，开始更新网络
                if (global_step >= UPDATE_AFTER) and (global_step % UPDATE_EVERY == 0):
                    for _ in range(UPDATE_EVERY):
                        train_step()

                if done:
                    writer.add_scalar("reward/train", episode_reward, global_step)
                    break
                    # 定期保存经验池数据集
                

            # 打印训练进度
            if (episode + 1) % 10 == 0:
                eval_reward = evaluate_policy(n_episodes=3)
                # print(f"Episode: {episode+1}, Step: {global_step}, TrainEpisodeReward: {episode_reward:.2f}, EvalReward: {eval_reward:.2f}")
                pbar.set_postfix({"Step": global_step, "TrainEpisodeReward": episode_reward, "EvalReward": eval_reward})
                writer.add_scalar("reward/eval", eval_reward, global_step)
                if eval_reward > best_reward:
                    patience = 0
                    best_reward = eval_reward
                    torch.save(policy.state_dict(), os.path.join(save_dir, "best_policy.pth"))
                    print(f"\nBest policy saved at episode {episode+1}, reward: {best_reward:.2f}")
                    
            if patience >= args.patience and args.patience > 0 and global_step > 10000:
                print(f"\nEarly stopping at episode {episode}")
                break
            if global_step >= MAX_GLOBAL_STEPS:
                print(f"\nMax steps reached at episode {episode}")
                break
            pbar.update(1)
            patience += 1

    # =========================
    # 测试结果
    # =========================
    test_reward = evaluate_policy(n_episodes=10)
    print(f"Final evaluation reward (10 episodes): {test_reward:.2f}")
    writer.add_scalar("reward/test", test_reward, global_step)
    writer.close()

if __name__ == "__main__":
    args = parse_args()
    main(args)