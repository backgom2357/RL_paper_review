# Scalable trust-region method for deep reinforcement learning using Kronecker-factored appreoximation

Algorithm: ACKTR
Author: Yuhuai Wu, Elman Mansimov. Shun Lian, Roger Grosse, Jimmy Ba
Last Edited By: Sung Yeul Back

# Introduction

- Neural network를 control policy에 적용해 좋은 결과를 얻을 수 있었지만 훈련시간이 길어졌습니다.
- 극복을 위해 분산 처리 방법이 제안되었지만 병렬도가 높아질 수록  sample efficiency가 빠르게 줄어들었습니다.

- RL에서 sample efficiency는 주된 고민거리인데 특히 실제 세상과 interact하는 Agent의 경우 이 문제는 더 크게 다가옵니다.
- 대안으로 Natural gradient method가 있습니다.

- 하지만 Natural gradient는 다루기 어려웠고 TRPO에서 다른 방법을 제안했지만 간단한 parameter까지 계산 step이 많아 큰 스케일의 모델에는 사용하기 어려웠습니다.

- K-FAC은 natural gradient를 scalable approximation한 것입니다.
- large scale의 지도학습에서 빠른 훈련 속도를 보여주어서 policy optimizaiton에 적용시켜보았다고 합니다.

# Background

## Reinforcement learning and actor-critic methods

- policy gradient

$$\nabla_\theta\mathcal{J}(\theta)=\Bbb{E}_\pi[\sum_{t=0}^\infty \Psi^t \nabla_\theta \log \pi_\theta(a_t|s_t)]]$$

- Advantage function

$$A^\pi(s_t,a_t)=\sum_{i=0}^{k-1}( \gamma^i r(s_{t+i},a_{t+i})+\gamma^k V_\phi^\pi (s_{t+k}))-V_\phi^\pi(s_t)$$

- Value network는 TD update를 따랐습니다.

## Natural gradient using Kronecker-factored approximation

- minimize

$$\mathcal{J}(\theta+\Delta \theta), \text{ subjext to the constraint }||\Delta\theta||_B < 1, \\ \text{where }||x||_B = (x^TBx)^{1/2} $$

- positive semidefinite matrix

    $$x^TBx \geq 0$$

- The solution to the constraint optimization problem

$$\theta \leftarrow \theta - \epsilon B^{-1} \nabla_\theta \mathcal{J}$$

- Fisher matrix를 쓰는 natural gradient가 안정적이고 효율적이긴 업데이트를 해주지만 역행렬 계산은 실용적이지 않습니다.

### Kronecker-factored approximate curvature

![Scalable%20trust%20region%20method%20for%20deep%20reinforcemen%2066a37eaa0b3b43f789c0f356cedf8478/Untitled.png](Scalable%20trust%20region%20method%20for%20deep%20reinforcemen%2066a37eaa0b3b43f789c0f356cedf8478/Untitled.png)

![Scalable%20trust%20region%20method%20for%20deep%20reinforcemen%2066a37eaa0b3b43f789c0f356cedf8478/Untitled%201.png](Scalable%20trust%20region%20method%20for%20deep%20reinforcemen%2066a37eaa0b3b43f789c0f356cedf8478/Untitled%201.png)

- FIM의 역행렬을 Block으로 나누어서 근사합니다.

- 이 방법을 통해 계산량을 줄였습니다.

# Method

## Natural gradient in actor-critic

- actor

    $$\nabla_\theta \mathcal{J}(\theta) = \Bbb{E}_{\pi_\theta}[\nabla_\theta \log \pi_\theta(s,a) A^{\pi_\theta}(s,a)]  \\ = \Bbb{E}_{\pi_\theta}[\nabla_\theta \log \pi_\theta(s,a) \nabla_\theta \log \pi_\theta(s,a)^T w] \\ = Fw$$

    $$F^{-1}\nabla_\theta \mathcal{J}(\theta) =\nabla_\theta^\text{nat} \mathcal{J}(\theta) = w$$

    - critic parameters의 방향으로 actor parameters 업데이트 합니다.

- critic
    - Least-squares function approximation problem
    - Gaussian observation model에서는 Gauss-Newton matrix가 Fisher matrix에 해당합니다.
    - Gauss-Newton matrix

    $$G:= \Bbb{E}[J^TJ], \\ \text{where J is the Jacobian of the mapping from parameter to outputs}$$

    - Gaussian distribution

    $$p(v|s_t) \sim \mathcal{N}(v; V(s_t),\sigma^2)$$

    - 이를 적용해 Fisher matrix를 정의합니다.  (Gauss-Newton Matrix)
    - 마찬가지로 K-FAC을 사용할 수 있습니다.

- Actor와 Critic이 분리되어 있다면,
    - 각각 K-FAC을 적용하여 업데이트 합니다.
    - 훈련의 불안정함을 피하기 위해 networks의 lower layer를 공유합니다.

    $$p(a,v|s) = \pi (a|s)p(v|s)$$

    - 를 이용해 Fisher metric을 정립하고  이를 적용시킨 Fisher matrix를 K-FAC로 근사합니다.

- 추가로
    - factorized Tikhonov damping approach (티호노브 정칙화)
    - Distributed second-order optimization using Kronecker-factored approximates를 적용해 계산 시간 감소시킵니다.

## Step-size selection and trust-region optimization

- Update의 크기를 trust-region방법을 통해 조정합니다.

# Experiments

## Discrete control

- 6개의 atari 2600 games

![Scalable%20trust%20region%20method%20for%20deep%20reinforcemen%2066a37eaa0b3b43f789c0f356cedf8478/Untitled%202.png](Scalable%20trust%20region%20method%20for%20deep%20reinforcemen%2066a37eaa0b3b43f789c0f356cedf8478/Untitled%202.png)

![Scalable%20trust%20region%20method%20for%20deep%20reinforcemen%2066a37eaa0b3b43f789c0f356cedf8478/Untitled%203.png](Scalable%20trust%20region%20method%20for%20deep%20reinforcemen%2066a37eaa0b3b43f789c0f356cedf8478/Untitled%203.png)

- sample efficiency 차이
- ACKTR이 더 빠른 시간 안에 수렴합니다.

![Scalable%20trust%20region%20method%20for%20deep%20reinforcemen%2066a37eaa0b3b43f789c0f356cedf8478/Untitled%204.png](Scalable%20trust%20region%20method%20for%20deep%20reinforcemen%2066a37eaa0b3b43f789c0f356cedf8478/Untitled%204.png)

## Continuous control

- MuJoCo

![Scalable%20trust%20region%20method%20for%20deep%20reinforcemen%2066a37eaa0b3b43f789c0f356cedf8478/Untitled%205.png](Scalable%20trust%20region%20method%20for%20deep%20reinforcemen%2066a37eaa0b3b43f789c0f356cedf8478/Untitled%205.png)

![Scalable%20trust%20region%20method%20for%20deep%20reinforcemen%2066a37eaa0b3b43f789c0f356cedf8478/Untitled%206.png](Scalable%20trust%20region%20method%20for%20deep%20reinforcemen%2066a37eaa0b3b43f789c0f356cedf8478/Untitled%206.png)

- Threshold를 넘기까지의 시간이 짧습니다.

![Scalable%20trust%20region%20method%20for%20deep%20reinforcemen%2066a37eaa0b3b43f789c0f356cedf8478/Untitled%207.png](Scalable%20trust%20region%20method%20for%20deep%20reinforcemen%2066a37eaa0b3b43f789c0f356cedf8478/Untitled%207.png)

- A2C보다는 확실하게 뛰어난 성능을 보여줍니다.

## A better norm for critic optimization?

- Euclidean norm vs Gauss-Newton norm

![Scalable%20trust%20region%20method%20for%20deep%20reinforcemen%2066a37eaa0b3b43f789c0f356cedf8478/Untitled%208.png](Scalable%20trust%20region%20method%20for%20deep%20reinforcemen%2066a37eaa0b3b43f789c0f356cedf8478/Untitled%208.png)

- 어떤 norm을 써도 A2C보다 좋은 성능을 보여주었습니다.
- 하지만, Gauss-Newton norm을 사용했을때 sample efficiency와 훈련의 끝에 얻는 episode rewards가 더 뛰어났습니다.

## How does ACKTR compare with A2C in wall-clock time?

![Scalable%20trust%20region%20method%20for%20deep%20reinforcemen%2066a37eaa0b3b43f789c0f356cedf8478/Untitled%209.png](Scalable%20trust%20region%20method%20for%20deep%20reinforcemen%2066a37eaa0b3b43f789c0f356cedf8478/Untitled%209.png)

## How do ACKTR and A2C perform with different batch sizes?

![Scalable%20trust%20region%20method%20for%20deep%20reinforcemen%2066a37eaa0b3b43f789c0f356cedf8478/Untitled%2010.png](Scalable%20trust%20region%20method%20for%20deep%20reinforcemen%2066a37eaa0b3b43f789c0f356cedf8478/Untitled%2010.png)