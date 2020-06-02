# 2. Deep Recurrent Q-Learning for Partially Observable MDPs

Algorithm: Deep Recurrent Q-Learning
Author: Hausknecht and Stone, 2015
Last Edited By: Sung Yeul Back
URL: https://arxiv.org/abs/1507.06527

# 배경

- DQN의 한계
    - DQN은 게임에서의 깜박임 때문에 정보 손실를 방지하고자 4 frames를 input으로 사용 (MDPs로 변환)
    - 4 frames를 넘어가면 게임은 POMDPs (Partially Observable MDPs)가 되며 기존 DQN은 해당 게임에서 좋지 않은 성능을 보입니다.
    - 실제 세상에는 fully observable환경보다 Partially Observable 환경이 대부분입니다.

- 가설
    - 이런 불완전한 정보가 주어질 때 RNN을 활용한 DQN (DRQN) 을 쓰면 POMDPs를 기존 DQN보다 더 잘 처리할 수 있다는 가설을 세웁니다.
    - 실제로, full observations에서 훈련하고, partial observations에서 평가한 결과 DRQN이 DQN보다 좋은 결과를 냈습니다.

## Deep Q-learning

### Q-Learning

- Model-free off-policy alogrithm

$$Q(s,a) := Q(s,a) + \alpha (r+\gamma \underset{a'}{\operatorname{max}} Q(s',a') - Q(s,a))$$

- Approximate the Q-values

$$L(s,a|\theta_i) = (r+\gamma\underset{a'}{\operatorname{max}} Q(s',a'|\theta_i) - Q(s,a|\theta_i))^2  \\ \theta_{i+1} = \theta_i + \alpha \nabla_\theta L(\theta_i)$$

### Experience Replay

$$\mathcal{D} = \{e_1, \dots, e_N\}  \\ e_t = (s_t, a_t, r_t, s_{t+1})$$

### Seperate Target Network

$$\hat{Q}(s,a|\theta^-)$$

### Loss of the network

$$L(s,a|\theta_i) = \Bbb{E}_{(s_t, a_t, r_t, s_{t+1}) \sim U(\mathcal{D})}\biggl[r+\gamma\underset{a'}{\operatorname{max}} \hat{Q}(s',a'|\theta^-) - Q(s,a|\theta_i)\biggr]^2   \\ \text{where U is 'sample uniformly'}$$

## Partial Observability

- 기존의 MDP 는

$$(\mathcal{S, A, P, R, \gamma})$$

- POMPD는

$$(\mathcal{S,A,P,R,\Omega,O,\gamma)}$$

- Agent는 true system state 대신 Ω로부터 obseravtion o를 받습니다.
- o 는 probability distribution O(s)로부터 생성됩니다.

- Vanilla Deep Q-learning으론 POMDP를 해결할 수 없는데 아래와 같기 때문입니다.

$$Q(o,a|\theta) \neq Q(s,a|\theta) $$

- 그래서 DQN에 recurrency를 추가해 이 두개의 차이를 줄이는 것이 목적입니다.

# 설계

![2%20Deep%20Recurrent%20Q%20Learning%20for%20Partially%20Observab%2099a680f375f748f8a1a91a551b57ef92/Untitled.png](2%20Deep%20Recurrent%20Q%20Learning%20for%20Partially%20Observab%2099a680f375f748f8a1a91a551b57ef92/Untitled.png)

- DQN의 구조를 유지하고 마지막 fully  connected layer를 fully connected LSTM layer로 바꿔주었습니다.

# Stable Recurrent Updates

### Bootstrapped Sequential Updates

- Replay memory에서 무작위로 episode를 선택해 episode의 처음부터 끝까지 updates를 시킵니다.
- 각 timestep의 target은 target Q-network로부터 생성된 값입니다.
- RNN의 hidden state는 episode에 걸쳐 전파되고 계산됩니다

- Sequential updates는 episode의 시작부터 hidden state를 전파할 수 있다는 이점이 있지만
- DQN의 random sampling policy에 위배됩니다.

### Bootstrapped Random Updates

- Replay memory에서 무작위로 episode를 선택한뒤 episode의 임의의 지점부터 unroll iterations timesteps 까지 updates를 시킵니다.
- 각 timestep의 target은 target Q-network로부터 생성된 값입니다.
- RNN의 초기 state는 update 시작때 0으로 초기화합니다.

- Randim updates는 random sampling policy를 잘 따르지만
- LSTM의 hidden state는 각 update 시작에 0으로 초기화해야함으로 역전파까지의 timestep보다 긴 time scale을 가진 정보는 잘 학습하지 못합니다.

- 두 방법의 성능차는 크게나지 않습니다.

# Atari Games: MDP or POMDP?

- Atari의 많은 게임들이 partially observable 이었지만 DQN은 게임의 4 frames를 묶음으로써 full state를 얻을 수 있게 되었습니다.

# Flickering Atari Games

### Flickering Pong

- 매 timestep마다 0.5의 확률로 화면이 안보입니다.
- POMDP

- 게임을 마스터 하기위해선 공의 속도와 위치를 화면이 보이지 않아도 예측할 수 있어야 합니다.

![2%20Deep%20Recurrent%20Q%20Learning%20for%20Partially%20Observab%2099a680f375f748f8a1a91a551b57ef92/Untitled%201.png](2%20Deep%20Recurrent%20Q%20Learning%20for%20Partially%20Observab%2099a680f375f748f8a1a91a551b57ef92/Untitled%201.png)

### 결과

- 10-frames DQN은 object의 속도를 잡아냈습니다.
- 1-frame DRQN도 좋은 성능을 보여주었습니다.
    - DRQN의 convolutional layers는 속도를 잡아내지 못했으나
    - 상위의 recurrent layer가 해당 부분을 보상해주었습니다.

- DRQN은 10 timesteps에 대해 backpropagation through time을 사용해서 10-frames DQN과 1-frame DRQN은 같은 history of game screens을 가질 수 있었습니다.

### 종합해 보면

- Partially observability를 처리할 때,
    - Non-recurrent deep network with a long history of observations
    - Recurrent deep network with a single observation
- 라는 선택지들이 생겼습니다.

# 평가

![2%20Deep%20Recurrent%20Q%20Learning%20for%20Partially%20Observab%2099a680f375f748f8a1a91a551b57ef92/Untitled%202.png](2%20Deep%20Recurrent%20Q%20Learning%20for%20Partially%20Observab%2099a680f375f748f8a1a91a551b57ef92/Untitled%202.png)

# MDP에서 POMDP로의 일반화

- 일반적인 MDP에서 훈련한뒤 POMDP애서 평가를 한 결과
- DQN 보다 DRQN이 더 좋은 성능을 보였습니다.

![2%20Deep%20Recurrent%20Q%20Learning%20for%20Partially%20Observab%2099a680f375f748f8a1a91a551b57ef92/Untitled%203.png](2%20Deep%20Recurrent%20Q%20Learning%20for%20Partially%20Observab%2099a680f375f748f8a1a91a551b57ef92/Untitled%203.png)