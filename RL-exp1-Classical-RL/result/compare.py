import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

def plot_comparison(sarsa_stats_file, qlearning_stats_file, double_qlearning_stats_file, smoothing_window=10, noshow=False, save_dir="../result"):
    # Load stats from .npz files
    sarsa_data = np.load(sarsa_stats_file)
    qlearning_data = np.load(qlearning_stats_file)
    double_qlearning_data = np.load(double_qlearning_stats_file)

    sarsa_lengths = sarsa_data["episode_lengths"]
    qlearning_lengths = qlearning_data["episode_lengths"]
    double_qlearning_lengths = double_qlearning_data["episode_lengths"]

    sarsa_rewards = sarsa_data["episode_rewards"]
    qlearning_rewards = qlearning_data["episode_rewards"]
    double_qlearning_rewards = double_qlearning_data["episode_rewards"]

    # Create directory for saving plots if it doesn't exist
    if noshow and not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Plot episode lengths over time
    fig1 = plt.figure(figsize=(10,5))
    plt.plot(sarsa_lengths, label="SARSA")
    plt.plot(qlearning_lengths, label="Q-Learning")
    plt.plot(double_qlearning_lengths, label="Double Q-Learning")
    plt.xlabel("Episode")
    plt.ylabel("Episode Length")
    plt.title("Episode Length over Time")
    plt.legend()
    if noshow:
        fig1.savefig(os.path.join(save_dir, "episode_length_comparison.png"))
        plt.close(fig1)
    else:
        plt.show()

    # Plot smoothed episode rewards over time
    fig2 = plt.figure(figsize=(10,5))
    sarsa_rewards_smoothed = pd.Series(sarsa_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()
    qlearning_rewards_smoothed = pd.Series(qlearning_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()
    double_qlearning_rewards_smoothed = pd.Series(double_qlearning_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()
    plt.plot(sarsa_rewards_smoothed, label="SARSA")
    plt.plot(qlearning_rewards_smoothed, label="Q-Learning")
    plt.plot(double_qlearning_rewards_smoothed, label="Double Q-Learning")
    plt.xlabel("Episode")
    plt.ylabel("Episode Reward (Smoothed)")
    plt.title("Episode Reward over Time (Smoothed over window size {})".format(smoothing_window))
    plt.legend()
    if noshow:
        fig2.savefig(os.path.join(save_dir, "episode_reward_comparison.png"))
        plt.close(fig2)
    else:
        plt.show()

    # Plot time steps per episode
    fig3 = plt.figure(figsize=(10,5))
    plt.plot(np.cumsum(sarsa_lengths), np.arange(len(sarsa_lengths)), label="SARSA")
    plt.plot(np.cumsum(qlearning_lengths), np.arange(len(qlearning_lengths)), label="Q-Learning")
    plt.plot(np.cumsum(double_qlearning_lengths), np.arange(len(double_qlearning_lengths)), label="Double Q-Learning")
    plt.xlabel("Time Steps")
    plt.ylabel("Episode")
    plt.title("Episode per Time Step")
    plt.legend()
    if noshow:
        fig3.savefig(os.path.join(save_dir, "episode_per_time_step_comparison.png"))
        plt.close(fig3)
    else:
        plt.show()

    return fig1, fig2, fig3

sarsa_stats_file = "./sarsa_stats.npz"
qlearning_stats_file = "./q_learning_stats.npz"
double_qlearning_stats_file = "./double_q_learning_stats.npz"

plot_comparison(sarsa_stats_file, qlearning_stats_file, double_qlearning_stats_file, noshow=True)