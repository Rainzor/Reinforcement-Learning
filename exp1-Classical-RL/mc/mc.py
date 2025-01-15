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

def plot_value_functions(V1, V2, title1="First-Visit MC", title2="Every-Visit MC"):
    """
    Plots two value functions side by side as surface plots for comparison.
    """
    min_x = min(k[0] for k in V1.keys())
    max_x = max(k[0] for k in V1.keys())
    min_y = min(k[1] for k in V1.keys())
    max_y = max(k[1] for k in V1.keys())

    x_range = np.arange(min_x, max_x + 1)
    y_range = np.arange(min_y, max_y + 1)
    X, Y = np.meshgrid(x_range, y_range)

    # Find values for all (x, y) coordinates in both value functions
    Z1_noace = np.apply_along_axis(lambda _: V1.get((_[0], _[1], False), 0), 2, np.dstack([X, Y]))
    Z1_ace = np.apply_along_axis(lambda _: V1.get((_[0], _[1], True), 0), 2, np.dstack([X, Y]))
    Z2_noace = np.apply_along_axis(lambda _: V2.get((_[0], _[1], False), 0), 2, np.dstack([X, Y]))
    Z2_ace = np.apply_along_axis(lambda _: V2.get((_[0], _[1], True), 0), 2, np.dstack([X, Y]))

    def plot_surfaces(X, Y, Z1, Z2, title1, title2, subtitle):
        fig = plt.figure(figsize=(20, 10))
        
        # Plot for the first value function
        ax1 = fig.add_subplot(121, projection='3d')
        surf1 = ax1.plot_surface(X, Y, Z1, rstride=1, cstride=1,
                                 cmap=matplotlib.cm.coolwarm, vmin=-1.0, vmax=1.0)
        ax1.set_xlabel('Player Sum')
        ax1.set_ylabel('Dealer Showing')
        ax1.set_zlabel('Value')
        ax1.set_title(f"{title1} - {subtitle}")
        fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=5)
        ax1.view_init(ax1.elev, -120)

        # Plot for the second value function
        ax2 = fig.add_subplot(122, projection='3d')
        surf2 = ax2.plot_surface(X, Y, Z2, rstride=1, cstride=1,
                                 cmap=matplotlib.cm.coolwarm, vmin=-1.0, vmax=1.0)
        ax2.set_xlabel('Player Sum')
        ax2.set_ylabel('Dealer Showing')
        ax2.set_zlabel('Value')
        ax2.set_title(f"{title2} - {subtitle}")
        fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=5)
        ax2.view_init(ax2.elev, -120)
        plt.show()

    # Plot no usable ace side by side
    plot_surfaces(X, Y, Z1_noace, Z2_noace, title1, title2, "No Usable Ace")
    
    # Plot usable ace side by side
    plot_surfaces(X, Y, Z1_ace, Z2_ace, title1, title2, "Usable Ace")

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

# def mc(env, num_episodes, discount_factor=1.0, epsilon=0.1):
#     """
#     Monte Carlo Control using Epsilon-Greedy policies.
#     Finds an optimal epsilon-greedy policy.
    
#     Args:
#         env: OpenAI gym environment.
#         num_episodes: Number of episodes to sample.
#         discount_factor: Gamma discount factor.
#         epsilon: Chance the sample a random action. Float betwen 0 and 1.
    
#     Returns:
#         A tuple (Q, policy).
#         Q is a dictionary mapping state -> action values.
#         policy is a function that takes an observation as an argument and returns
#         action probabilities
#     """
    
#     # Keeps track of sum and count of returns for each state
#     # to calculate an average. We could use an array to save all
#     # returns (like in the book) but that's memory inefficient.
#     returns_sum = defaultdict(float)
#     returns_count = defaultdict(float)
    
#     # The final action-value function.
#     # A nested dictionary that maps state -> (action -> action-value).
#     Q = defaultdict(lambda: np.zeros(env.action_space.n))
    
#     # The policy we're following
#     policy = make_epsilon_greedy_policy(Q, epsilon, env.action_space.n)
    
#     for i_episode in range(1, num_episodes + 1):
#         # Print out which episode we're on, useful for debugging.
#         if i_episode % 1000 == 0:
#             print("\rEpisode {}/{}.".format(i_episode, num_episodes), end="")
#             sys.stdout.flush()

# #############################################Implement your code###################################################################################################
#         # step 1 : Generate an episode.
#             # An episode is an array of (state, action, reward) tuples
#         # step 2 : Find all (state, action) pairs we've visited in this episode
#         # step 3 : Calculate average return for this state over all sampled episodes      
#  #############################################Implement your code end###################################################################################################

#     return Q, policy

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

num_episodes = 500000

import time

# Run and time the first-visit MC method
start_time_first_visit = time.time()
Q_first_visit, policy_first_visit = mc(env, num_episodes=num_episodes, epsilon=0.1)
end_time_first_visit = time.time()
time_first_visit = end_time_first_visit - start_time_first_visit

print("\nTime taken for first-visit MC: {:.2f} seconds".format(time_first_visit))

# Run and time the every-visit MC method
start_time_every_visit = time.time()
Q_every_visit, policy_every_visit = mc_every_visit(env, num_episodes=num_episodes, epsilon=0.1)
end_time_every_visit = time.time()
time_every_visit = end_time_every_visit - start_time_every_visit


print("\nTime taken for every-visit MC: {:.2f} seconds".format(time_every_visit))

# Create value functions from the action-value functions
V_first_visit = defaultdict(float)
V_every_visit = defaultdict(float)

for state, actions in Q_first_visit.items():
    V_first_visit[state] = np.max(actions)

for state, actions in Q_every_visit.items():
    V_every_visit[state] = np.max(actions)

# Plot the resulting value functions for comparison


plot_value_functions(V_first_visit, V_every_visit, title1="First-Visit MC", title2="Every-Visit MC")

