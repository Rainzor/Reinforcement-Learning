import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.distributions import Normal
import numpy as np

from .networks import QValueNet, QNetwork, PolicyNetContinuous, PolicyNet
from .utils import soft_update

class DDPG(nn.Module):
    def __init__(self, model_config):
        super(DDPG, self).__init__()
        self.model_config = model_config
        self.actor = PolicyNetContinuous(model_config.obs_dim, model_config.hidden_dim, model_config.act_dim, model_config.act_limit)
        self.critic = QValueNet(model_config.obs_dim, model_config.act_dim, model_config.hidden_dim)
        self.target_actor = PolicyNetContinuous(model_config.obs_dim, model_config.hidden_dim, model_config.act_dim, model_config.act_limit)
        self.target_critic = QValueNet(model_config.obs_dim, model_config.act_dim, model_config.hidden_dim)

        # 拷贝参数到 target网络
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=model_config.lr_policy)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=model_config.lr_q)

        self.gamma = model_config.gamma
        self.tau = model_config.tau
        self.policy_noise = model_config.noise * model_config.act_limit
        self.action_dim = model_config.act_dim
        self.action_bound = model_config.act_limit
    
    def forward(self, obs):
        return self.actor(obs)[0]
    
    def take_action(self, obs, deterministic=False):
        if deterministic:
            return self.actor.take_action(obs, deterministic=True)
        else:
            action = self.actor.take_action(obs, deterministic=True)[0]
            noise = np.random.normal(0, self.policy_noise, self.action_dim)
            action = np.clip(action + noise, -self.action_bound, self.action_bound)
            return action

    def update(self, transition):
        states, actions, rewards, next_states, dones = transition
        with torch.no_grad():
            next_actions, _ = self.target_actor(next_states, deterministic=True)
            next_actions = next_actions.view(-1, 1)
            q_next = self.target_critic(next_states, next_actions)  # shape [batch_size, 1]
            q_target = rewards + self.gamma * (1 - dones) * q_next # shape [batch_size, 1]

        q_values = self.critic(states, actions)  # [batch_size, 1]
        # Critic Loss: Minimize TD error
        critic_loss = nn.MSELoss()(q_values, q_target)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Actor Loss: Maximize Q value
        actions, log_probs = self.actor(states)
        q_values = self.critic(states, actions) # [batch_size, 1]
        actor_loss = -(q_values).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # --------------------------
        # Soft Update
        # --------------------------
        soft_update(self.actor, self.target_actor, self.tau)
        soft_update(self.critic, self.target_critic, self.tau) 