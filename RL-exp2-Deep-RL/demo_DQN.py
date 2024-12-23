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
from tqdm import tqdm

# Hyperparameters
EPISODES = 2000  # Number of training/testing episodes
BATCH_SIZE = 64
LR = 0.00025
GAMMA = 0.98
TAU = 0.005
SAVING_IETRATION = 10000  # Interval for saving checkpoints
MEMORY_CAPACITY = 10000  # Capacity of replay memory
MIN_CAPACITY = 500  # Minimum memory before learning starts
Q_NETWORK_ITERATION = 10  # Interval for syncing target network
EPSILON = 0.01  # epsilon-greedy
SEED = 0
MODEL_PATH = ''
SAVE_PATH_PREFIX = './logs'
TEST = False  # Flag to control training or testing mode
ENV = 'CartPole-v1'

# Choose an experimental environment
# Classic Control
# env = gym.make('CartPole-v1', render_mode="human" if TEST else None)
# env = gym.make('MountainCar-v0', render_mode="human" if TEST else None)
# ......
# LunarLander
# env = gym.make("LunarLander-v2", continuous=False, gravity=-10.0, enable_wind=True, wind_power=15.0, turbulence_power=1.5, render_mode="human" if TEST else None)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", '-e', type=str, default=ENV)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--model", type=str, default=MODEL_PATH)
    parser.add_argument("--save_path", type=str, default=SAVE_PATH_PREFIX)

    parser.add_argument("--episodes", '-n', type=int, default=EPISODES)
    parser.add_argument("--batch_size", '-b', type=int, default=BATCH_SIZE)
    parser.add_argument("--learning_rate", '-lr', type=float, default=LR)
    parser.add_argument("--gamma", '-g', type=float, default=GAMMA)
    parser.add_argument("--tau", '-t', type=float, default=TAU)
    parser.add_argument("--saving_iteration", '-si', type=int, default=SAVING_IETRATION)
    parser.add_argument("--memory_capacity", '-mc', type=int, default=MEMORY_CAPACITY)
    parser.add_argument("--min_capacity", '-minc', type=int, default=MIN_CAPACITY)
    parser.add_argument("--q_network_iteration", '-qi', type=int, default=Q_NETWORK_ITERATION)
    parser.add_argument("--epsilon", '-eps', type=float, default=EPSILON)
    
    # New argument to choose the algorithm
    parser.add_argument("--algorithm", '-alg', type=str, choices=['DQN', 'DDQN'], default='DQN', help="Choose between DQN and DDQN algorithms")
    
    return parser.parse_args()

class Model(nn.Module):

    def __init__(self, num_inputs=4, num_actions=2):
        # Input dimension is num_inputs, output dimension is num_actions
        super(Model, self).__init__()
        self.fc1 = nn.Linear(num_inputs, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, num_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class Data:

    def __init__(self, state, action, reward, next_state, done):
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.done = done

class Memory:
    """Experience Replay Memory"""

    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def set(self, data):
        self.buffer.append(data)

    def get(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states = torch.tensor([data.state for data in batch], dtype=torch.float)
        actions = torch.tensor([data.action for data in batch], dtype=torch.long).unsqueeze(1)
        rewards = torch.tensor([data.reward for data in batch], dtype=torch.float).unsqueeze(1)
        next_states = torch.tensor([data.next_state for data in batch], dtype=torch.float)
        dones = torch.tensor([data.done for data in batch], dtype=torch.float).unsqueeze(1)
        return states, actions, rewards, next_states, dones

class DQN():
    """Deep Q-Network"""

    def __init__(self, config):
        super(DQN, self).__init__()
        self.device = config['device']
        self.num_actions = config['num_actions']
        self.num_states = config['num_states']
        self.env_a_shape = config['env_a_shape']
        self.save_path = config['save_path']
        self.q_network_iteration = config['q_network_iteration']
        self.saving_iteration = config['saving_iteration']

        self.eval_net = Model(num_inputs=self.num_states, num_actions=self.num_actions).to(self.device)
        self.target_net = Model(num_inputs=self.num_states, num_actions=self.num_actions).to(self.device)
        self.target_net.load_state_dict(self.eval_net.state_dict())
        self.target_net.eval()

        self.learn_step_counter = 0
        self.memory_counter = 0
        self.memory = Memory(config['memory_capacity'])
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=config['learning_rate'])
        self.loss_func = nn.MSELoss()

    def choose_action(self, state, EPSILON=0.001):
        state = torch.tensor(state, dtype=torch.float).to(self.device)
        ENV_A_SHAPE = self.env_a_shape
        NUM_ACTIONS = self.num_actions

        if np.random.random() > EPSILON:  # Greedy policy
            with torch.no_grad():
                action_value = self.eval_net.forward(state)
            action = torch.argmax(action_value).item()
            action = action if ENV_A_SHAPE == 0 else np.array(action).reshape(ENV_A_SHAPE)
        else:
            # Random policy
            action = np.random.randint(0, NUM_ACTIONS)  # Random integer
            action = action if ENV_A_SHAPE == 0 else np.array(action).reshape(ENV_A_SHAPE)
        return action

    def store_transition(self, data):
        self.memory.set(data)
        self.memory_counter += 1

    def learn(self, BATCH_SIZE=BATCH_SIZE, GAMMA=GAMMA):
        # Update the target network
        if self.learn_step_counter % self.q_network_iteration == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        if self.learn_step_counter % self.saving_iteration == 0:
            self.save_train_model(self.learn_step_counter)

        self.learn_step_counter += 1

        # Sample a batch of transitions
        states, actions, rewards, next_states, dones = self.memory.get(BATCH_SIZE)

        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        # Current Q values
        q_eval = self.eval_net(states).gather(1, actions)

        # Next Q values from target network
        with torch.no_grad():
            q_next = self.target_net(next_states).max(1, keepdim=True)[0]
            q_target = rewards + GAMMA * q_next * (1 - dones)

        # Compute loss
        loss = self.loss_func(q_eval, q_target)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def save_train_model(self, epoch):
        ckpt_path = os.path.join(self.save_path, 'ckpt')
        os.makedirs(ckpt_path, exist_ok=True)
        torch.save(self.eval_net.state_dict(),
                     os.path.join(ckpt_path, f"{epoch}.pth"))

    def load_net(self, file):
        self.eval_net.load_state_dict(torch.load(file, map_location=self.device))
        self.target_net.load_state_dict(torch.load(file, map_location=self.device))



def main():
    args = get_args()
    save_path = os.path.join(args.save_path, args.env, args.algorithm)
    writer = None if args.test else SummaryWriter(save_path)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create environment
    if args.env == 'CartPole-v1':
        env = gym.make('CartPole-v1', render_mode="human" if args.test else None)
    elif args.env == 'MountainCar-v0':
        env = gym.make('MountainCar-v0', render_mode="human" if args.test else None)
    elif args.env == 'LunarLander-v2':
        env = gym.make("LunarLander-v2", continuous=False, gravity=-10.0, enable_wind=True, wind_power=15.0, turbulence_power=1.5, render_mode="human" if args.test else None)

    num_actions = env.action_space.n  # e.g., 2 for CartPole-v1
    num_states = env.observation_space.shape[0]  # e.g., 4 for CartPole-v1
    env_a_shape = 0 if np.issubdtype(
        type(env.action_space.sample()),
        np.integer) else env.action_space.sample().shape  # 0, to confirm the shape

    config = {
        'num_actions': num_actions,
        'num_states': num_states,
        'env_a_shape': env_a_shape,
        'device': device,
        'memory_capacity': args.memory_capacity,
        'learning_rate': args.learning_rate,
        'epsilon': args.epsilon,
        'epsilon_decay': 1000,  # You can make this configurable if desired
        'epsilon_min': 0.01,  # You can make this configurable if desired
        'save_path': save_path,
        'q_network_iteration': args.q_network_iteration,
        'saving_iteration': args.saving_iteration
    }

    # Instantiate the chosen algorithm
    if args.algorithm == 'DQN':
        agent = DQN(config)
        print("Using DQN Algorithm")
    elif args.algorithm == 'DDQN':
        agent = DDQN(config)
        print("Using DDQN Algorithm")
    else:
        raise ValueError("Unsupported algorithm type. Choose either 'DQN' or 'DDQN'.")

    if args.test:
        if args.model == '':
            raise ValueError("Please provide a model path for testing using --model")
        agent.load_net(args.model)

    with tqdm(range(args.episodes)) as pbar:
        for i in range(args.episodes):
            # print(f"EPISODE: {i+1}/{args.episodes}")
            state, info = env.reset(seed=args.seed)
            state = np.array(state)  # Ensure state is a NumPy array

            ep_reward = 0
            while True:
                action = agent.choose_action(
                    state=state,
                    EPSILON=args.epsilon if not args.test else 0)  # choose best action

                next_state, reward, done, truncated, info = env.step(action)  # observe next state and reward

                agent.store_transition(Data(state, action, reward, next_state, done))
                ep_reward += reward

                if args.test:
                    env.render()

                if agent.memory_counter >= args.min_capacity and not args.test:
                    agent.learn(BATCH_SIZE=args.batch_size, GAMMA=args.gamma)

                if done or truncated:
                    if args.test:
                        pbar.set_postfix({'Test Reward': round(ep_reward, 3)})
                    else:
                        # print(f"Train Episode: {i+1} , Reward: {round(ep_reward, 3)}")
                        pbar.set_postfix({'Reward': round(ep_reward, 3)})
                    break

                state = next_state
            pbar.update(1)
            if writer:
                writer.add_scalar('Reward', ep_reward, global_step=i)

    env.close()
    if writer:
        writer.close()

if __name__ == '__main__':
    main()
