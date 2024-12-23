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
import time

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
    parser.add_argument("--scheduler", '-s', action="store_true")
    parser.add_argument("--patience", '-p', type=int, default=200)

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
        states = torch.tensor(np.array([data.state for data in batch]), dtype=torch.float)
        actions = torch.tensor(np.array([data.action for data in batch]), dtype=torch.long).unsqueeze(1)
        rewards = torch.tensor(np.array([data.reward for data in batch]), dtype=torch.float).unsqueeze(1)
        next_states = torch.tensor(np.array([data.next_state for data in batch]), dtype=torch.float)
        dones = torch.tensor(np.array([data.done for data in batch]), dtype=torch.float).unsqueeze(1)
        return states, actions, rewards, next_states, dones

class DQN():
    """Deep Q-Network"""

    def __init__(self, config, method='DQN'):
        super(DQN, self).__init__()
        self.device = config['device']
        self.num_actions = config['num_actions']
        self.num_states = config['num_states']
        self.env_a_shape = config['env_a_shape']
        self.save_path = config['save_path']
        self.q_network_iteration = config['q_network_iteration']
        self.saving_iteration = config['saving_iteration']
        self.method = method

        self.eval_net = Model(num_inputs=self.num_states, num_actions=self.num_actions).to(self.device)
        self.target_net = Model(num_inputs=self.num_states, num_actions=self.num_actions).to(self.device)
        self.target_net.load_state_dict(self.eval_net.state_dict())
        self.target_net.eval()

        self.learn_step_counter = 0
        self.memory_counter = 0
        self.memory = Memory(config['memory_capacity'])
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=config['learning_rate'])
        if config['scheduler']:
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=5000, gamma=0.95)
        else:
            self.scheduler = None
        self.loss_func = nn.MSELoss()

    def choose_action(self, state, EPSILON=0.01):
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

        # DQN: Action Selection and Evaluation using eval_net
        # Double DQN: Action Selection using eval_net, Action Evaluation using target_net
        with torch.no_grad():
            if self.method == 'DQN':
                q_next = self.target_net(next_states).max(1, keepdim=True)[0]
            elif self.method == 'DDQN':
                # Select the best action based on eval_net
                actions_eval = self.eval_net(next_states).argmax(1, keepdim=True)
                # Evaluate the selected actions using target_net
                q_next = self.target_net(next_states).gather(1, actions_eval)
            q_target = rewards + GAMMA * q_next * (1 - dones)

        # Compute loss
        loss = self.loss_func(q_eval, q_target)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        if self.scheduler:
            self.scheduler.step()
        return loss.item()

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
    if args.test:
        args.episodes = 1
    timenow = time.strftime("%Y-%m-%d-%H-%M", time.localtime())
    save_path = os.path.join(args.save_path, args.env, args.algorithm, timenow)
    writer = None if args.test else SummaryWriter(save_path)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create environment
    if args.env == 'CartPole-v1':
        env = gym.make('CartPole-v1', render_mode="human" if args.test else None)
    # elif args.env == "Pendulum-v1":
    #     env = gym.make("Pendulum-v1", g=9.81, render_mode="human" if args.test else None)
    elif args.env == 'Acrobot-v1':
        env = gym.make('Acrobot-v1', render_mode="human" if args.test else None)
    elif args.env == 'LunarLander-v3':
        env = gym.make("LunarLander-v3", continuous=False, gravity=-10.0, enable_wind=True, wind_power=15.0, turbulence_power=1.5, render_mode="human" if args.test else None)
    else:
        assert False, "Please choose a valid environment: CartPole-v1, Acrobot-v1, LunarLander-v3"

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
        'scheduler': args.scheduler,
        'save_path': save_path,
        'q_network_iteration': args.q_network_iteration,
        'saving_iteration': args.saving_iteration
    }

    # Instantiate the chosen algorithm
    args.algorithm = args.algorithm.upper()
    agent = DQN(config, method=args.algorithm)
    print(f"Using {args.algorithm} Algorithm")

    if args.test:
        if args.model == '':
            raise ValueError("Please provide a model path for testing using --model")
        agent.load_net(args.model)

    with tqdm(range(args.episodes)) as pbar:
        best_reward = -np.inf
        early_stopping = 0
        for i in range(args.episodes):
            if early_stopping == args.patience:
                print(f"Early Stopping at Episode {i}, Best Reward: {best_reward}")
                break
            # print(f"EPISODE: {i+1}/{args.episodes}")
            state, info = env.reset(seed=args.seed)
            state = np.array(state)  # Ensure state is a NumPy array
            ep_reward = 0
            loss = 0
            count = 0
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
                    loss = loss + agent.learn(BATCH_SIZE=args.batch_size, GAMMA=args.gamma)
                    count += 1

                if done or truncated:
                    if args.test:
                        pbar.set_postfix({'Test Reward': round(ep_reward, 3)})
                    else:
                        # print(f"Train Episode: {i+1} , Reward: {round(ep_reward, 3)}")
                        if done and ep_reward > best_reward:
                            best_reward = ep_reward
                            early_stopping = 0
                            agent.save_train_model("best")
                        pbar.set_postfix({'Loss': loss / count if count != 0 else 0,
                                        'Reward': round(ep_reward, 3),        
                                        'Best Reward': round(best_reward, 3)})
                    break
                state = next_state
            pbar.update(1)
            early_stopping += 1
            if writer:
                writer.add_scalar('Reward', ep_reward, global_step=i)
                writer.add_scalar('Loss', loss / count if count != 0 else 0, global_step=i)
    agent.save_train_model("final")
    env.close()
    if writer:
        writer.add_hparams(vars(args), {'hparam/Reward': best_reward})
        writer.close()

if __name__ == '__main__':
    main()
