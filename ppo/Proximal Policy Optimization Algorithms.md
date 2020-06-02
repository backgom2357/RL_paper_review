# Proximal Policy Optimization Algorithms

Algorithm: PPO
Author: John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, Oleg Klimov
Last Edited By: Sung Yeul Back

# Introduction

- Q-Learning
    - 간단한 문제도 해결 실패
- Vanilla Policy Gradient
    - Poor data efficiency
    - Poor robustness
- TRPO
    - 복잡함
    - noise 또는 parameter sharing을 포함하는 구조와 호환 X

TRPO의 data efficiency와 performance를 가지면서 더 간단한 방법 제시

여러 다른 version of surrogate objective 비교

PPO 와 다른 방법 비교

# Background: Policy Optimization

## Policy Gradient Method

일반적으로 쓰이는 gradient estimator

$$\hat{g} = \hat\Bbb{E} \biggl[ \nabla_\theta log \pi_\theta (a_t|s_t)\hat{A}_t \biggl]$$

## Trust Region Method

![Proximal%20Policy%20Optimization%20Algorithms%206afb2db05c424f1f84335e05e6255c17/Untitled.png](Proximal%20Policy%20Optimization%20Algorithms%206afb2db05c424f1f84335e05e6255c17/Untitled.png)

TRPO에서는 실제로는 constraint 대신 penalty를 사용

![Proximal%20Policy%20Optimization%20Algorithms%206afb2db05c424f1f84335e05e6255c17/Untitled%201.png](Proximal%20Policy%20Optimization%20Algorithms%206afb2db05c424f1f84335e05e6255c17/Untitled%201.png)

목표를 달성하기 위해 고정된 β를 선택하거나 위 식을 SGD로 optimize하는것은 충분 X

# Clipped Surrogate Objective

$$r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}, \text{ so } r(\theta_{old})=1$$

아래의 식을 maximize

![Proximal%20Policy%20Optimization%20Algorithms%206afb2db05c424f1f84335e05e6255c17/Untitled%202.png](Proximal%20Policy%20Optimization%20Algorithms%206afb2db05c424f1f84335e05e6255c17/Untitled%202.png)

contraint가 없으면 policy update가 너무 커짐

논문에서 제안하는 것

![Proximal%20Policy%20Optimization%20Algorithms%206afb2db05c424f1f84335e05e6255c17/Untitled%203.png](Proximal%20Policy%20Optimization%20Algorithms%206afb2db05c424f1f84335e05e6255c17/Untitled%203.png)

- 1st term : 기존의 L CPI
- 2nd term : clipping the probability ratio

여기서 r = 1이면

$$L^{CLIP}(\theta) = L^{CPI}(\theta)$$

![Proximal%20Policy%20Optimization%20Algorithms%206afb2db05c424f1f84335e05e6255c17/Untitled%204.png](Proximal%20Policy%20Optimization%20Algorithms%206afb2db05c424f1f84335e05e6255c17/Untitled%204.png)

![Proximal%20Policy%20Optimization%20Algorithms%206afb2db05c424f1f84335e05e6255c17/Untitled%205.png](Proximal%20Policy%20Optimization%20Algorithms%206afb2db05c424f1f84335e05e6255c17/Untitled%205.png)

interpolate between initial policy parameter and the updated policy parameter

# Adaptive KL Penalty Coefficient

![Proximal%20Policy%20Optimization%20Algorithms%206afb2db05c424f1f84335e05e6255c17/Untitled%206.png](Proximal%20Policy%20Optimization%20Algorithms%206afb2db05c424f1f84335e05e6255c17/Untitled%206.png)

KL divergence에  penalty를 주는 새로운 방법

perform은 clipped surrogate objective보다 나쁘지만 중요한 baseline을 사용해서 논문에 포함

d는 드물게 튀고, β는 빠르게 수렴

# Algorithm

기존의 방법에서 L PG 대신에 L CLIP 또는 L KLPEN으로 대체

NN Architecture를 사용하면 policy surrogate와 value function을 합한 loss function을 사용

여기에 entropy bonus term도 exploration을 위해 추가

Policy gradient 실행 방법

- 주어진 policy로 T timestep run
- collected samples로 update

이 방법은 timestep T 구간만 보는 advantage estimator가 필요

이것을 일반화 해서 truncated version of generalized advantage estimation 사용

![Proximal%20Policy%20Optimization%20Algorithms%206afb2db05c424f1f84335e05e6255c17/Untitled%207.png](Proximal%20Policy%20Optimization%20Algorithms%206afb2db05c424f1f84335e05e6255c17/Untitled%207.png)

λ=1이면,

![Proximal%20Policy%20Optimization%20Algorithms%206afb2db05c424f1f84335e05e6255c17/Untitled%208.png](Proximal%20Policy%20Optimization%20Algorithms%206afb2db05c424f1f84335e05e6255c17/Untitled%208.png)

![Proximal%20Policy%20Optimization%20Algorithms%206afb2db05c424f1f84335e05e6255c17/Untitled%209.png](Proximal%20Policy%20Optimization%20Algorithms%206afb2db05c424f1f84335e05e6255c17/Untitled%209.png)

# Experiments

## Comparison of Surrogate Objectives

![Proximal%20Policy%20Optimization%20Algorithms%206afb2db05c424f1f84335e05e6255c17/Untitled%2010.png](Proximal%20Policy%20Optimization%20Algorithms%206afb2db05c424f1f84335e05e6255c17/Untitled%2010.png)

7환경에 각 3 무작위 seed

## Comparison to Other Algorithms in the Continuous Domain

![Proximal%20Policy%20Optimization%20Algorithms%206afb2db05c424f1f84335e05e6255c17/Untitled%2011.png](Proximal%20Policy%20Optimization%20Algorithms%206afb2db05c424f1f84335e05e6255c17/Untitled%2011.png)

## Showcase in the Continuous Domain: Humanoid Running and Steering

![Proximal%20Policy%20Optimization%20Algorithms%206afb2db05c424f1f84335e05e6255c17/Untitled%2012.png](Proximal%20Policy%20Optimization%20Algorithms%206afb2db05c424f1f84335e05e6255c17/Untitled%2012.png)

## Comparison to Other Algorithms on the Atari Domain

![Proximal%20Policy%20Optimization%20Algorithms%206afb2db05c424f1f84335e05e6255c17/Untitled%2013.png](Proximal%20Policy%20Optimization%20Algorithms%206afb2db05c424f1f84335e05e6255c17/Untitled%2013.png)