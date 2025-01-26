# README

## 1. Classical Reinforcement Learning

- SARSA
  $$
  Q(s,a) \leftarrow Q(s,a)+\alpha[R(s,a,s')+\gamma Q(s',a')-Q(s,a)]
  $$
  
- Q-Learning
  $$
  Q(s,a)\leftarrow Q(s,a)+\alpha[R(s,a,s')+\gamma \max_{a'}Q(s',a')-Q(s,a)]
  $$
  
- Double Q-Learning
  $$
  Q_1(s, a) \leftarrow Q_1(s, a) + \alpha \left[ R(s, a, s') + \gamma Q_2 \left(s', \arg \max_{a'} Q_1(s', a') \right) - Q_1(s, a) \right]
  $$
  

<div align=center>
    <img style = "
        border-radius: 0.3125em;
        box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
       	src="./assets/episode_reward_comparison.png"
        width = "60%">
    <br>
</div>


## 2. Deep Q-Leaning

- Deep Q Value Network
  $$
  loss(\phi)=\frac{1}{2N}\sum_{i=1}^N\left[Q(s_i,a_i|\phi)-(R_i+\gamma\max_{a'}Q(s_i',a'|\phi)\right]
  $$

- Double DQN
  $$
  loss(\phi|\phi^-)=\frac{1}{2N}\sum_{i=1}^N\left[Q_{e}(s_i,a_i|\phi)-(R_i+\gamma Q_{t}(s_i',a_i'|\phi^-)\right]\\
  a'_i = \arg\max_ {a'} Q _ e(s_ i',a'|\phi)
  $$

- Dueling DQN
  $$
  Q(s,a|\phi,\alpha,\beta) = V(s|\alpha,\phi)+A(s,a|\beta,\phi)-\frac{1}{|\mathcal{A}|}\sum_{\alpha'}A(s,a'|\beta,\phi)
  $$
  
<div align=center>
    <img style = "
        border-radius: 0.3125em;
        box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
       	src="./assets/image-20241226225303209.png"
        width = "100%">
    <br>
</div>

## 3. Offline Reinforcement Learning

- Conservative Q-Learning

$$
\hat Q^{k+1}\leftarrow \arg \min_Q \beta\cdot \mathbb E_{s\sim \mathcal D}\left[\log\sum_a \exp(Q(s,a))-\mathbb E_{ a\sim \hat \pi(a|s)}Q(s,a)\right] + \frac{1}{2} \cdot \mathbb E_{(s,a,s')\sim \mathcal D}\left[\left(Q(s,a)-\mathcal B^\pi \hat Q^{k}(s,a)\right)^2\right]
$$

$$
\mathcal B^\pi Q(s,a) = r(s,a) +\gamma \mathbb E_{a'\sim \pi(a|s')} (Q(s',a')), \quad s'= P(s,a)
$$

<div align=center>
    <img style = "
        border-radius: 0.3125em;
        box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
       	src="./assets/beta.png"
        width = "50%">
    <br>
</div>


## 4. Actor Critic Reinforcement Learning

- PPO: Proximal Policy Optimization
- DDPG: Deep Deterministic Policy Gradient
- TD3: Twin Delayed DDPG
- SAC: Soft Actor-Critic

<div align=center>
    <img style = "
        border-radius: 0.3125em;
        box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
       	src="./assets/Discrete.png"
        width = "40%">
    <img style = "
        border-radius: 0.3125em;
        box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
       	src="./assets/Continuous.png"
        width = "40%">
    <br>
</div>
