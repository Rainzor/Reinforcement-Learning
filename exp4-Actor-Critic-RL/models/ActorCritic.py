import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from .networks import PolicyNet, ValueNet, QNetwork
from .utils import soft_update



class ActorCritic(nn.Module):
    def __init__(self, model_config):
        super(ActorCritic, self).__init__()
        self.model_config = model_config
        self.actor = PolicyNet(model_config.obs_dim, model_config.act_dim, model_config.hidden_dim)
        self.critic = ValueNet(model_config.obs_dim, model_config.hidden_dim)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=model_config.lr_policy)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=model_config.lr_v)

        self.gamma = model_config.gamma

    def forward(self, obs):
        return self.actor(obs)[0]

    def update(self, transition):
        states, actions, rewards, next_states, dones = transition
        
        td_target = rewards + self.gamma * (1 - dones) * self.critic(next_states)
        td_eval = self.critic(states)
        # Actor Loss
        td_error = td_target - td_eval # TD  [batch_size, 1]
        log_probs, probs = self.actor.get_log_probs(states)
        # On-Policy, action is sampled from the current policy  a ~ π(a|s)
        log_probs = log_probs.gather(dim=1, index=actions) # [batch_size, 1]
        actor_loss = -(log_probs * td_error.detach()).mean()

        # Critic Loss
        critic_loss = nn.MSELoss()(td_eval, td_target.detach())

        self.critic_optimizer.zero_grad()
        self.actor_optimizer.zero_grad()

        critic_loss.backward()
        actor_loss.backward()

        self.critic_optimizer.step()
        self.actor_optimizer.step()

    def take_action(self, obs, deterministic=False):
        return self.actor.take_action(obs, deterministic=deterministic)


class ActorCriticOffPolicy(nn.Module):
    def __init__(self, model_config):
        super(ActorCriticOffPolicy, self).__init__()
        self.model_config = model_config
        self.actor = PolicyNet(model_config.obs_dim, model_config.act_dim, model_config.hidden_dim)
        self.critic = QNetwork(model_config.obs_dim, model_config.act_dim, model_config.hidden_dim)
        
        self.target_actor = PolicyNet(model_config.obs_dim, model_config.act_dim, model_config.hidden_dim)
        self.target_critic = QNetwork(model_config.obs_dim, model_config.act_dim, model_config.hidden_dim)

        # 拷贝参数到 target网络
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=model_config.lr_policy)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=model_config.lr_q)

        self.gamma = model_config.gamma
        self.tau = model_config.tau

    def forward(self, obs):
        return self.actor(obs)[0]

    def update(self, transition):
        states, actions, rewards, next_states, dones = transition
        with torch.no_grad():
            next_log_probs, next_probs = self.target_actor.get_log_probs(next_states)
            q_next = self.target_critic(next_states)  # shape [batch_size, act_dim]
            V_next = (next_probs * q_next).sum(dim=-1, keepdim=True) # shape [batch_size, 1]
            q_target = rewards + self.gamma * (1 - dones) * V_next # shape [batch_size, 1]

        q_values = self.critic(states).gather(dim=1, index=actions)  # [batch_size, 1]
        # Critic Loss: Minimize TD error
        critic_loss = nn.MSELoss()(q_values, q_target)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Actor Loss: Maximize Q value
        log_probs, probs = self.actor.get_log_probs(states) 
        q_values = self.critic(states).detach() # [batch_size, act_dim]
        actor_loss = -(probs * q_values).sum(dim=-1).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # --------------------------
        # Soft Update
        # --------------------------
        soft_update(self.actor, self.target_actor, self.tau)
        soft_update(self.critic, self.target_critic, self.tau)
    
    def take_action(self, obs, deterministic=False):
        return self.actor.take_action(obs, deterministic=deterministic)
