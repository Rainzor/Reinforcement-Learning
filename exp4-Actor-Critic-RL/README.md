# Lab4: Actor Critic RL

> SA24229016 王润泽
>
> Keywords: Model-Free、Policy-based、Deep Reinforcement Learning

## 1. Abstract



## 2. Introduction

### 2.1 Basic Concepts

强化学习（Reinforcement Learning, RL）是一种通过与环境交互来学习如何在不同状态下采取合适的行动，以最大化累计奖励的机器学习方法。

在强化学习中，**智能体（Agent）**在每个时刻观察到当前**状态（State）**，执行一个**行动（Action）**，并从环境中获得一个**反馈奖励（Reward）**。通过多次尝试和反馈，智能体逐步学习出一套最优**策略（Policy）**，即在不同状态下采取的最优行动选择方法。

<div align=center>
    <img style = "
        border-radius: 0.3125em;
        box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
       	src="./assets/image-20241114174114511.png"
        width = "70%">
    <br>
    <div style = "
        color: orange;
        border-bottom: 1px solid #d9d9d9;
        display: inline-block;
        color: #999;
        text-align: center;
        padding: 2px;">
        Figure 1. Reinforcement Learning
    </div>
    <p> </p>
</div>

----

#### 2.1.1 Definitions：

- **状态（State）** $s_t\in \mathcal S$ ：Environment 在某一时刻 $t$ 描述 **Agent** 的状态

- **观察（Observation）** $o_t\in \mathcal O$：**Agent** 对环境状态的部分或全部感知 $O \subseteq S$

- **动作（Action）** $a_t\in \mathcal A$：**Agent** 在给定状态下的可能行为。

- **即时奖励（Reward）** $R_t\in R: \mathcal S\times \mathcal A \times \mathcal S \rightarrow \mathcal R$ ：反馈信号，用于指示 **Agent** 采取的动作是否有利于实现目标。

- **转移概率 (Transition Probability)** $P: \mathcal S\times \mathcal A\rightarrow \mathcal R $，表示在给定当前状态 $s$ 和执行动作 $a$ 后，转移到其他状态 $s'$ 的概率$P(s'|s,a)$。
  
  通常概率 $P$ 由外部环境本身决定，在 Model-Free RL 环境下，给定 $(s,a)$ ，转移后的状态 $s'$ 是确定的：
  $$
  P(s'|s,a) = \delta(s'-\hat s)
  $$
  
- **策略（Policy）**：**Agent** 根据当前的状态 $s_t$ 会选取不同的行为 $a_t$，选择的方法叫做策略(Policy)，通常有两种类型：

  - deterministic：$a_t = \mu(s_t)$，根据状态 $s_t$ ，唯一确定行动 $a_t$
  - stochastic: $a_t \sim \pi(\cdot|s_t)$，根据状态 $s_t$ ，按概率选择行动 $a_t$

- **轨迹 (Trajectory)**: **Agent** 从初始状态 $s_0$ 出发，按照给定 **Policy** $\pi$ 下，得到的系列状态动作对：
  
  $$
  \tau = \{(s_t,a_t)\}_{t=0}^N
  $$
  
  通常在给定策略 $\pi$ 和转移概率 $P$ 下，这样的轨迹 $\tau$ 符合分布 $\rho_\pi$，即：
  
  $$
  \tau\sim \rho_\pi
  $$

- **值函数（Value Function）**：用于评估给定**状态 $s$ 或状态-动作对 $(s,a)$ **的累积奖励， 是对未来收益的预期：
  
  $$
  G(\tau) = \sum_{t=0}^{\infty}R (s_t,a_t,s_{t+1})
  $$
  
  在多数情况下，我们更强调当下收益，同时保证未来收益足够大，因此采用**值函数**定义如下：
  
  $$
  G(\tau) = \sum_{t=0}^{\infty} \gamma^t R(s_t,a_t,s_{t+1}),\quad\gamma\in(0,1)
  $$
  常用的值函数有以下两类定义：**==Bellman Equation==** 
  
  - **Value Function：** 指在当前状态 $s$ 下，执行策略 $\pi$ 获得的累积奖励：
    
    $$
    \begin{aligned}
    V^{\pi}(s)&=\mathbb E_{\tau\sim \rho_\pi}[G(\tau)|s_0=s]\\
    &=\mathbb E_{(s_t,a_t)\sim \rho_\pi}  \left[ \sum_{t=0}^{\infty} \gamma^t R(s_t,a_t,s_{t+1}) |s_0=s\right]\\
    &= \mathbb E_{a\sim \pi(\cdot|s)} \left[\mathbb E_{s'\sim P(\cdot|s,a)}\left[R(s,a,s')+\gamma V^\pi(s')\right]\right]
    \end{aligned}
    $$
  
  - **Action-Value Function:** 指在给定状态 $s$ 和当下执行的行动 $a$ 后得到的累积奖励：
    $$
    \begin{aligned}
    Q^{\pi}(s,a)&=\mathbb E_{\tau\sim\rho_\pi}[G(\tau)|s_0=s,a_0=a]\\
    &=\mathbb E_{s'\sim P(\cdot|s,a)}\left[R(s,a,s')+\gamma \mathbb E_{a'\sim \pi(\cdot{|s'})}[Q^{\pi}(s',a')])\right]
    \end{aligned}
    $$
  - **Relation:**   
    $$
    V^{\pi}(s)=\mathbb E_{a\sim \pi(\cdot|s)} \left[Q^\pi(s,a)\right]
    $$
    
    $$
    Q^{\pi}(s,a) =\mathbb E_{s'\sim P(\cdot|s,a)}\left[R(s,a,s')+\gamma V^\pi(s') \right]
    $$



### 2.2 Reinforcement Learning Algorithms

#### 2.2.1 Classification

基于强化学习的算法分类有很多，一种分类方式是以：是否对环境建模(Model)为分类依据：

- **Model-Based RL：**学习构建环境的 $P(s'|s,a)$ 和 $R(s,a,s')$ 来进行决策，通过建立环境的模型，规划一系列行动。
- **Model-Free RL:** 直接与环境交互，根据获取到的收益，找到最优策略 $\pi(a|s)$ 指导 Agent 行为。在 Model-Free RL中，给定 $a,s$ 下， $R,s'$ 是已知的（可计算的），无需优化建模推断。

> If you want a way to check if an RL algorithm is model-based or model-free, ask yourself this question: after learning, can the agent make predictions about what the next state and reward will be before it takes each action?
>
> If the agent can do, it's model-based.

**我们主要探讨 Model-Free RL**

![../_images/rl_algorithms_9_15.svg](https://spinningup.openai.com/en/latest/_images/rl_algorithms_9_15.svg)

---

#### 2.2.2 Model-Free RL

> Targets: Optimal Value Functions

粗略来说，**Model-Free 强化学习任务就是找到一个策略 $\pi$ ，使得回报 $G$ 尽可能多。**

定义最优策略 $\pi^*$:

$$
\pi^* = \arg\max_\pi V^\pi(s),\quad \forall s\in \mathcal S
$$

最优策略 $\pi^*$ 对应的价值函数即为：最优价值函数 (Optimal Value Functions):

$$
V^*(s) = \max_\pi V^\pi(s),\quad \forall s\in  \mathcal S
$$

对应的也存在一个最优的行为价值函数：

$$
Q^*(s,a)=  \max_\pi Q^\pi(s,a),\quad \forall s\in  \mathcal S, a\in\mathcal A
$$
基于以上定义，可以将 **Model-Free RL** 划分为以下两类学习算法：

- **Policy-Based:** 定义合适的目标函数 $J(\theta)$ ，直接对策略 $\pi_\theta(a|s)$ 进行建模和优化
- **Value-Based:** 依据 **Bellman** 通过构造并逼近价值函数 $V_\phi(s)$ or $Q_{\phi}(s,a)$ ,间接学习最佳策略 $\mu(s)=\arg \max_a Q(s,a)$

> Value-Based 在 [Lab1](../exp1-Classical-RL), [Lab2](../exp2-Deep-Q-Learning) 中都已介绍，分别是关于经典Q-Learning算法，Deep Q-Learning 算法，在此不多赘述。

#### 2.2.3 Policy-Based Method: Policy Gradient



## 3. Algorithms: Actor Critic RL



## 4. Experiments

## 5. Conclusions





