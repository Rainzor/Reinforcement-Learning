import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from .networks import PolicyNet, QValueNet, PolicyNetContinuous, QNetwork
from .utils import soft_update

class TD3(nn.Module):
    def __init__(self, model_config):
        super(TD3, self).__init__()
        self.model_config = model_config
        self.actor = PolicyNetContinuous(model_config.obs_dim, model_config.hidden_dim, model_config.act_dim, model_config.act_limit)
        self.critic1 = QValueNet(model_config.obs_dim, model_config.act_dim, model_config.hidden_dim)
        self.critic2 = QValueNet(model_config.obs_dim, model_config.act_dim, model_config.hidden_dim)
        self.target_actor = copy.deepcopy(self.actor)
        self.target_critic1 = copy.deepcopy(self.critic1)
        self.target_critic2 = copy.deepcopy(self.critic2)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=model_config.lr_policy)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=model_config.lr_q)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=model_config.lr_q)

        self.gamma = model_config.gamma
        self.tau = model_config.tau
        self.policy_noise = model_config.noise*model_config.act_limit
        self.noise_clip = 0.5*model_config.act_limit
        self.policy_delay = 2
        self.action_dim = model_config.act_dim
        self.action_bound = model_config.act_limit
        self.global_step = 0
    
    def forward(self, obs):
        return self.actor(obs)[0]
    
    def take_action(self, obs, deterministic=False):
        if deterministic:
            return self.actor.take_action(obs, deterministic=True)
        else:
            action = self.actor.take_action(obs, deterministic=False)[0]
            noise = np.random.normal(0, self.policy_noise, self.action_dim)
            action = np.clip(action + noise, -self.action_bound, self.action_bound)
            return action


    def update(self, transition):
        self.global_step += 1
        states, actions, rewards, next_states, dones = transition
        actions = actions.view(-1, 1).float()
        with torch.no_grad():
            noise = (torch.randn_like(actions) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_actions, _ = self.target_actor(next_states, deterministic=True)
            next_actions = (next_actions + noise).clamp(-self.action_bound, self.action_bound)

            q_next1 = self.target_critic1(next_states, next_actions)  # shape [batch_size, 1]
            q_next2 = self.target_critic2(next_states, next_actions)  # shape [batch_size, 1]
            q_next = torch.min(q_next1, q_next2)

            q_target = rewards + self.gamma * (1 - dones) * q_next # shape [batch_size, 1]

        q_values1 = self.critic1(states, actions)  # [batch_size, 1]
        q_values2 = self.critic2(states, actions)  # [batch_size, 1]
        # Critic Loss: Minimize TD error
        # 双Q网络，取最小值
        critic_loss1 = nn.MSELoss()(q_values1, q_target)
        critic_loss2 = nn.MSELoss()(q_values2, q_target)

        self.critic1_optimizer.zero_grad()
        critic_loss1.backward()
        self.critic1_optimizer.step()

        self.critic2_optimizer.zero_grad()
        critic_loss2.backward()
        self.critic2_optimizer.step()

        # Actor Loss: Maximize Q value
        # 只更新 Actor/Policy 网络
        if self.global_step % self.policy_delay == 0:
            actions_pred, _ = self.actor(states, deterministic=True)
            actor_loss = -self.critic1(states, actions_pred).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
        
            # Soft Update
            soft_update(self.actor, self.target_actor, self.tau)
            soft_update(self.critic1, self.target_critic1, self.tau)
            soft_update(self.critic2, self.target_critic2, self.tau)