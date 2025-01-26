import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import numpy as np

from .networks import QValueNet, QNetwork, PolicyNetContinuous, PolicyNet
from .utils import soft_update

TAU = 0.005

class SAC(nn.Module):
    def __init__(self, model_config):
        super(SAC, self).__init__()
        self.model_config = model_config
        self.critic1 = QNetwork(model_config.obs_dim, model_config.act_dim, model_config.hidden_dim)
        self.critic2 = QNetwork(model_config.obs_dim, model_config.act_dim, model_config.hidden_dim)
        self.critic1_target = QNetwork(model_config.obs_dim, model_config.act_dim, model_config.hidden_dim)
        self.critic2_target = QNetwork(model_config.obs_dim, model_config.act_dim, model_config.hidden_dim)
        self.actor = PolicyNet(model_config.obs_dim, model_config.act_dim, model_config.hidden_dim)

        # 拷贝参数到 target网络
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        # 优化器
        self.q1_optimizer = optim.Adam(self.critic1.parameters(), lr=model_config.lr_q)
        self.q2_optimizer = optim.Adam(self.critic2.parameters(), lr=model_config.lr_q)
        self.policy_optimizer = optim.Adam(self.actor.parameters(), lr=model_config.lr_policy)

        self.gamma = model_config.gamma
        self.tau = model_config.tau
        self.alpha = model_config.alpha

    def forward(self, obs):
        return self.actor(obs)[0]

    def update(self, transition):
        states, actions, rewards, next_states, dones = transition
        with torch.no_grad():
            # 下一个状态的 log_probs, probs
            next_log_probs, next_probs = self.actor.get_log_probs(next_states)
            # 计算下一个状态中, 对应各个动作的 Q 值
            q1_next = self.critic1_target(next_states)  # shape [batch_size, act_dim]
            q2_next = self.critic2_target(next_states)  # shape [batch_size, act_dim]

            # 取两个Q网络的最小值
            min_q_next = torch.min(q1_next, q2_next)  # [batch_size, act_dim]

            # 对离散空间，目标值 = r + γ * E_{a' ~ π}[ Q(s',a') - α * log π(a'|s') ]
            # 其中 E_{a' ~ π}[·] 可以用 sum(prob * ·)
            V_next = (next_probs * (min_q_next - self.alpha * next_log_probs)).sum(dim=-1, keepdim=True)
            q_target = rewards + self.gamma * (1 - dones) * V_next
        
        q1_values = self.critic1(states).gather(dim=1, index=actions)  # [batch_size, 1]
        q2_values = self.critic2(states).gather(dim=1, index=actions)  # [batch_size, 1]
        q1_loss = nn.MSELoss()(q1_values, q_target)
        q2_loss = nn.MSELoss()(q2_values, q_target)

        self.q1_optimizer.zero_grad()
        q1_loss.backward()
        self.q1_optimizer.step()

        self.q2_optimizer.zero_grad()
        q2_loss.backward()
        self.q2_optimizer.step()

        # --------------------------
        # 3) 更新 策略网络
        # --------------------------
        log_probs, probs = self.actor.get_log_probs(states) # contains parameters of the policy network
        # 计算 Q(s,a) 的最小值 (针对所有动作)
        q1_vals = self.critic1(states)
        q2_vals = self.critic2(states)
        min_q = torch.min(q1_vals, q2_vals)  # [batch_size, act_dim]

        # 期望 J(π) = E_{s ~ D}[ E_{a ~ π}[ α * log π(a|s) - Q(s,a) ] ]
        # 其中对离散动作的期望可以写成 sum(π(a|s)*[α * log π(a|s) - Q(s,a)])
        # 注意这里 log_probs 的 shape = [batch_size, act_dim]
        # Off-policy 的更新公式
        policy_loss = (probs * (self.alpha * log_probs - min_q)).sum(dim=1).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        # --------------------------
        # 4) 软更新 target 网络
        # --------------------------
        soft_update(self.critic1, self.critic1_target, self.tau)
        soft_update(self.critic2, self.critic2_target, self.tau)
    
    def take_action(self, obs, deterministic=False):
        return self.actor.take_action(obs, deterministic=deterministic)

class SACContinuous(nn.Module):
    def __init__(self, model_config):
        super(SACContinuous, self).__init__()
        self.model_config = model_config
        self.critic1 = QValueNet(model_config.obs_dim, model_config.act_dim, model_config.hidden_dim)
        self.critic2 = QValueNet(model_config.obs_dim, model_config.act_dim, model_config.hidden_dim)
        self.critic1_target = QValueNet(model_config.obs_dim, model_config.act_dim, model_config.hidden_dim)
        self.critic2_target = QValueNet(model_config.obs_dim, model_config.act_dim, model_config.hidden_dim)
        self.actor = PolicyNetContinuous(model_config.obs_dim, model_config.hidden_dim, model_config.act_dim, model_config.act_limit)

        # 拷贝参数到 target网络
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        # 优化器
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=model_config.lr_q)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=model_config.lr_q)
        self.policy_optimizer = optim.Adam(self.actor.parameters(), lr=model_config.lr_policy)

        # self.alpha = model_config.alpha

        self.log_alpha = torch.tensor(np.log(model_config.alpha), requires_grad=True, dtype=torch.float)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=model_config.lr_policy)

        self.gamma = model_config.gamma
        self.tau = model_config.tau
        self.target_entropy = -torch.prod(torch.Tensor(model_config.act_dim)).item()  # -dim(A)


    def forward(self, obs):
        return self.actor(obs)[0]

    def update(self, transition):
        states, actions, rewards, next_states, dones = transition
        # actions = actions.squeeze(-1)
        with torch.no_grad():
            # 下一个状态的 log_probs, probs
            next_actions, next_log_probs = self.actor(next_states) # shape [batch_size, 1]
            # 计算下一个状态确定性策略的 Q 值
            q1_next = self.critic1_target(next_states, next_actions)  # shape [batch_size, 1]
            q2_next = self.critic2_target(next_states, next_actions)  # shape [batch_size, 1]

            # 取两个Q网络的最小值
            min_q_next = torch.min(q1_next, q2_next) # [batch_size, 1]

            V_next = (min_q_next - self.log_alpha.exp() * next_log_probs)
            q_target = rewards + self.gamma * (1 - dones) * V_next # shape [batch_size, 1]
        
        q1_values = self.critic1(states, actions)  # [batch_size, 1]
        q2_values = self.critic2(states, actions)  # [batch_size, 1]

        critic1_loss = nn.MSELoss()(q1_values, q_target)
        critic2_loss = nn.MSELoss()(q2_values, q_target)

        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()
        
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()

        # --------------------------
        # 3) 更新 策略网络
        # --------------------------
        actions, log_probs = self.actor(states) # contains parameters of the policy network
        # 计算 Q(s,a) 的最小值 (针对所有动作)
        q1_vals = self.critic1(states, actions)
        q2_vals = self.critic2(states, actions)
        min_q = torch.min(q1_vals, q2_vals)  # [batch_size, 1]

        actor_loss = (self.log_alpha.exp() * log_probs - min_q).mean()

        self.policy_optimizer.zero_grad()
        actor_loss.backward()
        self.policy_optimizer.step()

        # --------------------------
        # 4) 更新 α
        # --------------------------
        alpha_loss = -(self.log_alpha.exp() * (log_probs + self.target_entropy).detach()).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        # --------------------------
        # 5) 软更新 target 网络
        # --------------------------
        soft_update(self.critic1, self.critic1_target, self.tau)
        soft_update(self.critic2, self.critic2_target, self.tau)

    def take_action(self, obs, deterministic=False):
        return self.actor.take_action(obs, deterministic=deterministic)


