import gym
import itertools
import matplotlib
import numpy as np
import pandas as pd
import sys

if "../" not in sys.path:
  sys.path.append("../") 

from collections import defaultdict
from lib.envs.cliff_walking import CliffWalkingEnv
from lib import plotting

matplotlib.style.use('ggplot')

env = CliffWalkingEnv()

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

# def sarsa(env, num_episodes, discount_factor=1.0, alpha=0.5, epsilon=0.1):
#     """
#     SARSA algorithm: On-policy TD control. Finds the optimal epsilon-greedy policy.
    
#     Args:
#         env: OpenAI environment.
#         num_episodes: Number of episodes to run for.
#         discount_factor: Gamma discount factor.
#         alpha: TD learning rate.
#         epsilon: Chance the sample a random action. Float betwen 0 and 1.
    
#     Returns:
#         A tuple (Q, stats).
#         Q is the optimal action-value function, a dictionary mapping state -> action values.
#         stats is an EpisodeStats object with two numpy arrays for episode_lengths and episode_rewards.
#     """
    
#     # The final action-value function.
#     # A nested dictionary that maps state -> (action -> action-value).
#     Q = defaultdict(lambda: np.zeros(env.action_space.n))
    
#     # Keeps track of useful statistics
#     stats = plotting.EpisodeStats(
#         episode_lengths=np.zeros(num_episodes),
#         episode_rewards=np.zeros(num_episodes))

#     # The policy we're following
#     policy = make_epsilon_greedy_policy(Q, epsilon, env.action_space.n)
    
#     for i_episode in range(num_episodes):
#         # Print out which episode we're on, useful for debugging.
#         if (i_episode + 1) % 100 == 0:
#             print("\rEpisode {}/{}.".format(i_episode + 1, num_episodes), end="")
#             sys.stdout.flush()

#         # Reset the environment
#         state = env.reset()

#         # One step in the environment
#         for t in itertools.count():
# #########################################Implement your code here#######################################################################################
#             # step 1 : Take a step( 1 line code, tips : env.step() )
            
#             # step 2 : Pick the next action
            
#             # Update statistics
#             stats.episode_rewards[i_episode] += reward
#             stats.episode_lengths[i_episode] = t
            
#             # step 3 : TD Update
#             # compute Q value      

# #########################################Implement your code here end#####################################################################################
#     return Q, stats

def sarsa(env, num_episodes, discount_factor=1.0, alpha=0.5, epsilon=0.1):
    """
    SARSA algorithm: On-policy TD control. Finds the optimal epsilon-greedy policy.
    
    Args:
        env: OpenAI environment.
        num_episodes: Number of episodes to run for.
        discount_factor: Gamma discount factor.
        alpha: TD learning rate.
        epsilon: Chance to sample a random action. Float between 0 and 1.
    
    Returns:
        A tuple (Q, stats).
        Q is the optimal action-value function, a dictionary mapping state -> action values.
        stats is an EpisodeStats object with two numpy arrays for episode_lengths and episode_rewards.
    """
    
    # The final action-value function.
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    
    # Keeps track of useful statistics
    stats = plotting.EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes))

    # The policy we're following
    policy = make_epsilon_greedy_policy(Q, epsilon, env.action_space.n)
    
    for i_episode in range(num_episodes):
        # Print out which episode we're on, useful for debugging.
        if (i_episode + 1) % 100 == 0:
            print("\rEpisode {}/{}.".format(i_episode + 1, num_episodes), end="")
            sys.stdout.flush()

        # Reset the environment and pick the first action
        state = env.reset()
        action_probs = policy(state)
        action = np.random.choice(np.arange(len(action_probs)), p=action_probs)

        # One step in the environment
        for t in itertools.count():
            # Step 1: Take a step in the environment
            next_state, reward, done, _ = env.step(action)
            
            # Step 2: Pick the next action based on the epsilon-greedy policy
            next_action_probs = policy(next_state)
            next_action = np.random.choice(np.arange(len(next_action_probs)), p=next_action_probs)
            
            # Update statistics
            stats.episode_rewards[i_episode] += reward
            stats.episode_lengths[i_episode] = t
            
            # Step 3: TD Update
            # Compute the TD target and update Q value
            td_target = reward + discount_factor * Q[next_state][next_action]
            td_delta = td_target - Q[state][action]
            Q[state][action] += alpha * td_delta
            
            if done:
                break
            
            # Move to the next state and action
            state = next_state
            action = next_action
    
    return Q, stats

stats_filename = "../result/sarsa_stats.npz"

# Run Double Q-Learning algorithm
Q, stats = sarsa(env, 500)

# # Convert stats to numpy arrays and save them
# np.savez(stats_filename, episode_lengths=np.array(stats.episode_lengths), episode_rewards=np.array(stats.episode_rewards))
# print(f"\nStatistics saved to {stats_filename}")

# Plot episode statistics
plotting.plot_episode_stats(stats)