import gym
import matplotlib
import numpy as np
import sys
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from collections import defaultdict

if "../" not in sys.path:
  sys.path.append("../") 
from lib.envs.blackjack import BlackjackEnv
from lib import plotting

matplotlib.style.use('ggplot')

#create env
env = BlackjackEnv()

def make_epsilon_greedy_policy(Q, epsilon, nA):
    """
    Creates an epsilon-greedy policy based on a given Q-function and epsilon.
    
    Args:
        Q: A dictionary that maps from state -> action-values.
            Each value is a numpy array of length nA (see below)
        epsilon: The probability to select a random action . float between 0 and 1.
        nA: Number of actions in the environment.
    
    Returns:
        A function that takes the observation as an argument and returns
        the probabilities for each action in the form of a numpy array of length nA.
    
    """
    def policy_fn(observation):
        A = np.ones(nA, dtype=float) * epsilon / nA
        best_action = np.argmax(Q[observation])
        A[best_action] += (1.0 - epsilon)
        return A
    return policy_fn

first_stats = {}
stats_list = range(1000, 1000001, 1000)
def mc(env, num_episodes, discount_factor=1.0, epsilon=0.1):
    """
    Monte Carlo Control using Epsilon-Greedy policies.
    Finds an optimal epsilon-greedy policy.
    
    Args:
        env: OpenAI gym environment.
        num_episodes: Number of episodes to sample.
        discount_factor: Gamma discount factor.
        epsilon: Chance to sample a random action. Float between 0 and 1.
    
    Returns:
        A tuple (Q, policy).
        Q is a dictionary mapping state -> action values.
        policy is a function that takes an observation as an argument and returns
        action probabilities
    """
    
    # Keeps track of sum and count of returns for each state-action pair
    returns_sum = defaultdict(float)
    returns_count = defaultdict(float)
    
    # The final action-value function.
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    
    # The policy we're following
    policy = make_epsilon_greedy_policy(Q, epsilon, env.action_space.n)
    
    

    for i_episode in range(1, num_episodes + 1):
        # Print out which episode we're on, useful for debugging.
        if i_episode % 1000 == 0:
            print("\rEpisode {}/{}.".format(i_episode, num_episodes), end="")
            sys.stdout.flush()
        
        if i_episode in stats_list:
            V_first_visit = defaultdict(float)
            for state, actions in Q.items():
                V_first_visit[state] = np.max(actions)
            first_stats[i_episode] = V_first_visit
        
        # Step 1: Generate an episode.
        # Initialize an empty episode list
        episode = []
        state = env.reset()
        
        while True:
            # Choose action based on epsilon-greedy policy
            action_probs = policy(state)
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            
            # Take action in the environment
            next_state, reward, done, _ = env.step(action)
            episode.append((state, action, reward))
            
            if done:
                break
            state = next_state

        # Step 2: Find all (state, action) pairs visited in this episode
        # We use a set to avoid duplicate pairs
        sa_in_episode = set((x[0], x[1]) for x in episode)

        for state, action in sa_in_episode:
            # Step 3: Calculate the return (G) for the first occurrence of each (state, action) pair
            first_occurrence_idx = next(i for i, x in enumerate(episode) if x[0] == state and x[1] == action)

            # Sum up all rewards since the first occurrence
            G = 0
            G_discounted = 1
            for i, x in enumerate(episode[first_occurrence_idx:]):
                G += x[2] * G_discounted
                G_discounted *= discount_factor
            
            # Incremental averaging
            # returns_sum[(state, action)] += G
            returns_count[(state, action)] += 1.0
            Q[state][action] += (G - Q[state][action]) / returns_count[(state, action)]
        
        # Update policy using the improved Q values
        policy = make_epsilon_greedy_policy(Q, epsilon, env.action_space.n)
    
    return Q, policy

every_stats = {}

def mc_every_visit(env, num_episodes, discount_factor=1.0, epsilon=0.1):
    """
    Monte Carlo Control using Epsilon-Greedy policies.
    Finds an optimal epsilon-greedy policy with every-visit updates.
    
    Args:
        env: OpenAI gym environment.
        num_episodes: Number of episodes to sample.
        discount_factor: Gamma discount factor.
        epsilon: Chance to sample a random action. Float between 0 and 1.
    
    Returns:
        A tuple (Q, policy).
        Q is a dictionary mapping state -> action values.
        policy is a function that takes an observation as an argument and returns
        action probabilities
    """
    
    # Keeps track of sum and count of returns for each state-action pair
    returns_count = defaultdict(float)
    
    # The final action-value function.
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    
    # The policy we're following
    policy = make_epsilon_greedy_policy(Q, epsilon, env.action_space.n)
    
    for i_episode in range(1, num_episodes + 1):
        # Print out which episode we're on, useful for debugging.
        if i_episode % 1000 == 0:
            print("\rEpisode {}/{}.".format(i_episode, num_episodes), end="")
            sys.stdout.flush()
        
        if i_episode in stats_list:
            V_every_visit = defaultdict(float)
            for state, actions in Q.items():
                V_every_visit[state] = np.max(actions)
            every_stats[i_episode] = V_every_visit

        # Step 1: Generate an episode.
        # Initialize an empty episode list
        episode = []
        state = env.reset()
        
        while True:
            # Choose action based on epsilon-greedy policy
            action_probs = policy(state)
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            
            # Take action in the environment
            next_state, reward, done, _ = env.step(action)
            episode.append((state, action, reward))
            
            if done:
                break
            state = next_state

        # Step 2: Initialize return G and reverse loop through the episode
        G = 0.0
        for state, action, reward in reversed(episode):
            G = reward + discount_factor * G  # Calculate cumulative reward

            # Incremental mean update for Q[state][action]
            returns_count[(state, action)] += 1.0
            Q[state][action] += (G - Q[state][action]) / returns_count[(state, action)]
        
        # Update policy using the improved Q values
        policy = make_epsilon_greedy_policy(Q, epsilon, env.action_space.n)
    
    return Q, policy

import time

# Run and time the first-visit MC method
Q_first_visit, policy_first_visit = mc(env, num_episodes=1000000, epsilon=0.1)
Q_every_visit, policy_every_visit = mc_every_visit(env, num_episodes=1000000, epsilon=0.1)


def get_values(V):
    min_x = min(k[0] for k in V.keys())
    max_x = max(k[0] for k in V.keys())
    min_y = min(k[1] for k in V.keys())
    max_y = max(k[1] for k in V.keys())

    x_range = np.arange(min_x, max_x + 1)
    y_range = np.arange(min_y, max_y + 1)
    X, Y = np.meshgrid(x_range, y_range)

    # Find values for all (x, y) coordinates in both value functions
    Z1_noace = np.apply_along_axis(lambda _: V.get((_[0], _[1], False), 0), 2, np.dstack([X, Y]))
    Z1_ace = np.apply_along_axis(lambda _: V.get((_[0], _[1], True), 0), 2, np.dstack([X, Y]))
    return Z1_noace, Z1_ace


reference_V = defaultdict(float)
for state, actions in Q_first_visit.items():
    reference_V[state] = np.max(actions)

reference_noace, reference_ace = get_values(reference_V)

first_noace_mse = []
first_ace_mse = []
every_noace_mse = []
every_ace_mse = []

for i in stats_list:
    first_noace, first_ace = get_values(first_stats[i])
    every_noace, every_ace = get_values(every_stats[i])

    first_noace_mse.append(np.sum((first_noace - reference_noace) ** 2))
    first_ace_mse.append(np.sum((first_ace - reference_ace) ** 2))
    every_noace_mse.append(np.sum((every_noace - reference_noace) ** 2))
    every_ace_mse.append(np.sum((every_ace - reference_ace) ** 2))

figure, axes = plt.subplots(1, 2, figsize=(20, 10))

# log x scale for better visualization
axes[0].set_xscale('log')
axes[1].set_xscale('log')

axes[0].plot(stats_list, first_noace_mse, label='First-visit MC')
axes[0].plot(stats_list, every_noace_mse, label='Every-visit MC')

axes[1].plot(stats_list, first_ace_mse, label='First-visit MC')
axes[1].plot(stats_list, every_ace_mse, label='Every-visit MC')

axes[0].set_title('No Usable Ace')
axes[1].set_title('Usable Ace')
axes[0].set_xlabel('Number of episodes')
axes[1].set_xlabel('Number of episodes')
axes[0].set_ylabel('MSE')
axes[1].set_ylabel('MSE')

axes[0].legend()
axes[1].legend()
plt.savefig('mc_comparison.png')



