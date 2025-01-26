import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import numpy as np

from .networks import PolicyNet, ValueNet
from .utils import compute_advantage
class PPO(nn.Module):
    def __init__(self, model_config, epochs=10):
        super(PPO, self).__init__()
        self.model_config = model_config
        self.actor = PolicyNet(model_config.obs_dim, model_config.act_dim, model_config.hidden_dim)
        self.critic = ValueNet(model_config.obs_dim, model_config.hidden_dim)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=model_config.lr_policy)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=model_config.lr_v)
        self.epochs = epochs
        self.gamma = model_config.gamma
        self.lmbda = model_config.lmbda
        self.epsilon = model_config.epsilon
    
    def forward(self, obs):
        return self.actor(obs)

    def update(self, transition):
        states, actions, rewards, next_states, dones = transition
        
        td_target = rewards + self.gamma * (1 - dones) * self.critic(next_states)
        # Actor Loss
        td_error = td_target - self.critic(states)
        advantage = compute_advantage(self.gamma, self.lmbda, td_error.cpu()).to(states.device)
        log_probs_old, _ = self.actor.get_log_probs(states)
        log_probs_old = log_probs_old.gather(dim=1, index=actions).detach()

        for _ in range(self.epochs):
            log_probs, _ = self.actor.get_log_probs(states)
            log_probs = log_probs.gather(dim=1, index=actions)
            ratio = (log_probs - log_probs_old).exp()
            actor_loss1 = ratio * advantage
            actor_loss2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon ) * advantage
            actor_loss = -torch.min(actor_loss1, actor_loss2).mean()
            critic_loss = nn.MSELoss()(self.critic(states), td_target.detach())

            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            
            actor_loss.backward()
            critic_loss.backward()

            self.actor_optimizer.step()
            self.critic_optimizer.step()

    def take_action(self, obs, deterministic=False):
        return self.actor.take_action(obs, deterministic=deterministic)



