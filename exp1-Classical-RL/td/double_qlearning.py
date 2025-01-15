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
        Q: A function that takes a state as input and returns action-values for that state.
        epsilon: The probability to select a random action. float between 0 and 1.
        nA: Number of actions in the environment.
    
    Returns:
        A function that takes the observation as an argument and returns
        the probabilities for each action in the form of a numpy array of length nA.
    """
    def policy_fn(observation):
        A = np.ones(nA, dtype=float) * epsilon / nA
        q_values = Q(observation)
        best_action = np.argmax(q_values)
        A[best_action] += (1.0 - epsilon)
        return A
    return policy_fn

def double_q_learning(env, num_episodes, discount_factor=1.0, alpha=0.5, epsilon=0.1):
    """
    Double Q-Learning algorithm: Off-policy TD control. Finds the optimal greedy policy
    while following an epsilon-greedy policy.
    
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
    
    # Initialize two Q-value dictionaries for Double Q-Learning
    Q1 = defaultdict(lambda: np.zeros(env.action_space.n))
    Q2 = defaultdict(lambda: np.zeros(env.action_space.n))

    # Keeps track of useful statistics
    stats = plotting.EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes))
    
    # The policy we're following, using the combined Q-values
    policy = make_epsilon_greedy_policy(lambda s: Q1[s] + Q2[s], epsilon, env.action_space.n)
    
    for i_episode in range(num_episodes):
        # Print out which episode we're on, useful for debugging.
        if (i_episode + 1) % 100 == 0:
            print("\rEpisode {}/{}.".format(i_episode + 1, num_episodes), end="")
            sys.stdout.flush()
        
        # Reset the environment
        state = env.reset()

        for t in itertools.count():
            # Select action according to epsilon-greedy policy
            action_probs = policy(state)
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            next_state, reward, done, _ = env.step(action)

            # Update statistics
            stats.episode_rewards[i_episode] += reward
            stats.episode_lengths[i_episode] = t
            
            # Decide randomly whether to update Q1 or Q2
            if np.random.rand() < 0.5:
                # Update Q1 using the best action according to Q1, evaluated using Q2
                best_next_action = np.argmax(Q1[next_state])
                td_target = reward + discount_factor * Q2[next_state][best_next_action]
                td_delta = td_target - Q1[state][action]
                Q1[state][action] += alpha * td_delta
            else:
                # Update Q2 using the best action according to Q2, evaluated using Q1
                best_next_action = np.argmax(Q2[next_state])
                td_target = reward + discount_factor * Q1[next_state][best_next_action]
                td_delta = td_target - Q2[state][action]
                Q2[state][action] += alpha * td_delta

            if done:
                break  # End the episode if done
            
            # Move to the next state
            state = next_state

    # Combine Q1 and Q2 to get the final Q values
    Q = defaultdict(lambda: np.zeros(env.action_space.n),
                    {state: Q1[state] + Q2[state] for state in set(Q1) | set(Q2)})
    
    return Q, stats

stats_filename = "../result/double_q_learning_stats.npz"

# Run Double Q-Learning algorithm
Q, stats = double_q_learning(env, 500)

# # Convert stats to numpy arrays and save them
# np.savez(stats_filename, episode_lengths=np.array(stats.episode_lengths), episode_rewards=np.array(stats.episode_rewards))
# print(f"\nStatistics saved to {stats_filename}")


# Plot episode statistics
plotting.plot_episode_stats(stats, noshow=True)


