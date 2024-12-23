import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import gymnasium as gym
from torch.utils.tensorboard import SummaryWriter
import collections
import argparse

# 超参数
EPISODES = 2000  # 训练/测试幕数
BATCH_SIZE = 64
LR = 0.00025
GAMMA = 0.98
SAVING_IETRATION = 1000  # 保存Checkpoint的间隔
MEMORY_CAPACITY = 10000  # Memory的容量
MIN_CAPACITY = 500  # 开始学习的下限
Q_NETWORK_ITERATION = 10  # 同步target network的间隔
EPSILON = 0.01  # epsilon-greedy
SEED = 0
MODEL_PATH = ''
SAVE_PATH_PREFIX = './log/dqn/'
TEST = False  # 用于控制当前行为是在训练还是在测试
ENV = 'CartPole-v1'



# 选择一个实验环境

# Classica Control
# env = gym.make('CartPole-v1', render_mode="human" if TEST else None)
# env = gym.make('MountainCar-v0', render_mode="human" if TEST else None)
# ......

# LunarLander
# env = gym.make("LunarLander-v2",continuous=False,gravity=-10.0,enable_wind=True,wind_power=15.0,turbulence_power=1.5,render_mode="human" if TEST else None)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", '-e', type=str, default=ENV)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--model", type=str, default=MODEL_PATH)
    parser.add_argument("--save_path", type=str, default=SAVE_PATH_PREFIX)

    parser.add_argument("--episodes",'-n', type=int, default=EPISODES)
    parser.add_argument("--batch_size", '-b', type=int, default=BATCH_SIZE)
    parser.add_argument("--learning_rate", '-lr', type=float, default=LR)
    parser.add_argument("--gamma", '-g', type=float, default=GAMMA)
    parser.add_argument("--saving_iteration", '-si', type=int, default=SAVING_IETRATION)
    parser.add_argument("--memory_capacity", '-mc', type=int, default=MEMORY_CAPACITY)
    parser.add_argument("--min_capacity", '-minc', type=int, default=MIN_CAPACITY)
    parser.add_argument("--q_network_iteration", '-qi', type=int, default=Q_NETWORK_ITERATION)
    parser.add_argument("--epsilon", '-eps', type=float, default=EPSILON)
    
    return parser.parse_args()

class Model(nn.Module):

    def __init__(self, num_inputs=4):
        # TODO 输入的维度为 NUM_STATES，输出的维度为 NUM_ACTIONS
        super(Model, self).__init__()

    def forward(self, x):
        # TODO
        return x


class Data:

    def __init__(self, state, action, reward, next_state, done):
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.done = done


class Memory:
    """用于 Experience Replay"""

    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def set(self, data):
        # TODO
        pass

    def get(self, batch_size):
        # TODO
        pass


class DQN():
    """docstring for DQN"""

    def __init__(self, config):
        super(DQN, self).__init__()
        self.eval_net, self.target_net = Model().to(device), Model().to(device)
        self.learn_step_counter = 0
        self.memory_counter = 0
        self.memory = Memory(capacity=MEMORY_CAPACITY)
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()
        self.device = config['device']
        self.num_actions = config['num_actions']
        self.num_states = config['num_states']
        self.env_a_shape = config['env_a_shape']

    def choose_action(self, state, EPSILON=1.0):
        state = torch.tensor(state, dtype=torch.float).to(device)
        ENV_A_SHAPE = self.env_a_shape
        NUM_ACTIONS = self.num_actions

        if np.random.random() > EPSILON:  # random number
            # greedy policy
            action_value = self.eval_net.forward(state)
            action = torch.argmax(action_value).item()
            action = action if ENV_A_SHAPE == 0 else action.reshape(
                ENV_A_SHAPE)
        else:
            # random policy
            action = np.random.randint(0, NUM_ACTIONS)  # int random number
            action = action if ENV_A_SHAPE == 0 else action.reshape(
                ENV_A_SHAPE)
        return action

    def store_transition(self, data):
        self.memory.set(data)
        self.memory_counter += 1

    def learn(self, Q_NETWORK_ITERATION=10):
        # update the parameters
        if self.learn_step_counter % Q_NETWORK_ITERATION == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        if self.learn_step_counter % SAVING_IETRATION == 0:
            self.save_train_model(self.learn_step_counter)

        self.learn_step_counter += 1

        # TODO 实现 Q network 的更新过程

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def save_train_model(self, epoch):
        torch.save(self.eval_net.state_dict(),
                   f"{SAVE_PATH_PREFIX}ckpt/{epoch}.pth")

    def load_net(self, file):
        self.eval_net.load_state_dict(torch.load(file))
        self.target_net.load_state_dict(torch.load(file))


def main():
    args = get_args()
    writer = SummaryWriter(f'{args.save_path}')

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    os.makedirs(f"{args.save_path}ckpt", exist_ok=True)

    # 创建环境
    env = gym.make(args.env)

    num_actions = env.action_space.n  # 2
    num_states = env.observation_space.shape[0]  # 4
    env_a_shape = 0 if np.issubdtype(
        type(env.action_space.sample()),
        np.integer) else env.action_space.sample().shape  # 0, to confirm the shape
    
    config = {
        'num_actions': num_actions,
        'num_states': num_states,
        'env_a_shape': env_a_shape,
        'device': device
    }
    dqn = DQN(config)

    if args.test:
        dqn.load_net(args.model)
    for i in range(args.episodes):
        print("EPISODE: ", i)
        state, info = env.reset(seed=args.seed)

        ep_reward = 0
        while True:
            action = dqn.choose_action(
                state=state,
                EPSILON=args.epsilon if not args.test else 0)  # choose best action
            next_state, reward, done, truncated, info = env.step(
                action)  # observe next state and reward
            dqn.store_transition(Data(state, action, reward, next_state, done))
            ep_reward += reward
            if arg.test:
                env.render()
            if dqn.memory_counter >= args.min_capacity and not arg.test:
                dqn.learn(args.q_network_iteration)
                if done:
                    print("episode: {} , the episode reward is {}".format(
                        i, round(ep_reward, 3)))
            if done:
                if arg.test:
                    print("episode: {} , the episode reward is {}".format(
                        i, round(ep_reward, 3)))
                break
            state = next_state
        writer.add_scalar('reward', ep_reward, global_step=i)


if __name__ == '__main__':
    main()
