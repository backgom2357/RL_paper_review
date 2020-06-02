# 6. Rainbow: Combining Improvements in Deep Reinforcement Learning

Algorithm: Rainbow DQN
Author: M. Hessel, J. Modayil, H.Hasselt, T.Schaul, G.Ostrovski, W.Dabney, D.Horgan, B.Piot, M.Azar, D.Silber
Last Edited By: Sung Yeul Back
URL: https://arxiv.org/abs/1710.02298

# Introduction

DQN의 6 extensions

- DDQN
- PER
- Dueling DQN
- Multi-step
- Distributional DQN
- Noisy DQN

서로 다른 문제를 해결

같은 framework를 사용했기 때문에 결합가능

이 논문에서는

1. 어떻게 결합하였는지
2. 결합 후 성능이 얼마나 향상됬는지
3. Ablation study를 통해 각 성분의 영향력

# Background

강화학습은 Agent의 문제를 해결함에 있어 주어진 환경에서 보상을 최대화하기위한 행동을 학습하는 것

# Extensions to DQN

DQN의 한계를 극복하기 위해 나왔던 extensions

## Double DQN

Convolution Q-learning은 equation의 maximization 단계 에서 야기되는 overestimation bias에 영향

Maximization 단계에서 action select와 evaluat를 분리

## Prioritized reply

학습에 더 유리한 samples로 학습

TD error와 같은 지표를 통해 우선순위 설정

## Dueling networks

신경망의 새 구조 제시

Single fully connected layer를 2개의 fully connected layers로

각각 value와 advantage functions를 추정

## Multi-step learning

기존 Q-learning에서는 Accumulated reward를 다음스탭까지만 고려

Multi-step learning에서는 n-step 까지 고려

조절 가능

## Distributional RL

## Noisy Nets

e-greedy policies 의 한계를 linear layer에 noisy를 추가함으로 극복

# The Integrated Agent

1. Distributional + multi-step
2. DDQN + Multi-step Distributional loss
3. The KL loss를 지표로 한 Prioritized replay
4. return distributions을 사용한 Dueling network
5. Apply noisy net

 

# Experimental Methods

## Evaluation Methodology

57개의 Atari 2600 games 에서 평가

Mnih et al.(2015)의 학습과 평가를 사용

Agents' scores are normalized, per game

평가의 끝에서 best agent snapshot를 재평가하기 위해 2가지 방법을 사용

- no-ops starts regime
- human starts regime

이 두 regime에서의 차는 그 agent가 trajectories에 over-fit한 정도를 암시

## Hyper-parameter tuning

Rainbow는 많은 hyper-parameter를 포함

먼저 기존의 논문에서 소개된 것을 사용

그리고 수동으로 가장 민감한 hyper-parameter 조정

DQN에선 learning update를 200K frames 이후부터 실시했다면, 

prioitized reply를 사용했을땐 80K frames 부터 가능

- Noisy net

    fully greedily

    noisy stream을 초기화 하기 위한 hyper-parameter 0.5

- Adam optimizer를 사용

    learning rate의 선택에 RMSProp보다 덜 민감하기 때문

- Reply prioritization

    priority exponent w 0.5

    sampling exponent β 0.4 ~ 1

- Multi-step learning

    n = 3

# Analysis

1. Rainbow 와 다른 agents
2. Ablation study

## Comparision to published baselines

![6%20Rainbow%20Combining%20Improvements%20in%20Deep%20Reinforce%2068501706d4c745859c4757503f11569a/Untitled.png](6%20Rainbow%20Combining%20Improvements%20in%20Deep%20Reinforce%2068501706d4c745859c4757503f11569a/Untitled.png)

7M frames 만에 DQN의 최종 성능을 돌파

44M frames 후에는 다른 baselines보다 더 뛰어난 성능

![6%20Rainbow%20Combining%20Improvements%20in%20Deep%20Reinforce%2068501706d4c745859c4757503f11569a/Untitled%201.png](6%20Rainbow%20Combining%20Improvements%20in%20Deep%20Reinforce%2068501706d4c745859c4757503f11569a/Untitled%201.png)

## Learning speed

7M frames 까지 가는데 10시간 이하

200M frames 까지 약 10 일

## Ablation studies

The full Rainbow combination에서 성분 하나씩 빼며 그성분이 얼마나 영향을 주는지 비교

![6%20Rainbow%20Combining%20Improvements%20in%20Deep%20Reinforce%2068501706d4c745859c4757503f11569a/Untitled%202.png](6%20Rainbow%20Combining%20Improvements%20in%20Deep%20Reinforce%2068501706d4c745859c4757503f11569a/Untitled%202.png)

![6%20Rainbow%20Combining%20Improvements%20in%20Deep%20Reinforce%2068501706d4c745859c4757503f11569a/Untitled%203.png](6%20Rainbow%20Combining%20Improvements%20in%20Deep%20Reinforce%2068501706d4c745859c4757503f11569a/Untitled%203.png)

Prioritized replay와 multi-step learning이 Rainbow에서 가장 중요

Noisy nets을 제외했을 때 몇몇 게임에선 성능이 크게 감소

Double Q-learning을 제외했을때 가끔 the support of the distribution을 벗어나 underestimate을 야기

the support of the distribution이 늘어났을때 DDQN의 중요도도 증가

![6%20Rainbow%20Combining%20Improvements%20in%20Deep%20Reinforce%2068501706d4c745859c4757503f11569a/Untitled%204.png](6%20Rainbow%20Combining%20Improvements%20in%20Deep%20Reinforce%2068501706d4c745859c4757503f11569a/Untitled%204.png)

# Discussion

성능을 향상시킬 수 있는 것들은 더 많이 존재

여기서는 Q-learning family만 집중

- Learning efficiency
    - Optimality tightening (He et al. 2016)
    - Eligibility traces (Sutton 1988)
- Data efficiency
    - Episodic control (Blundell et al. 2016)
- Exploration methods
    - Bootstrapped DQN (Osband et al. 2016)
    - intrinsic motivation (Stadie, Levine, and Abbeel 2015)
    - count-based exploration (Bellemare et al. 2016)
- Learning speed
    - A3C (Mnih et al. 2016)
    - Gorila (Nair et al. 2015)
    - Evolution Strategies (Salimans et al. 2017)
- Hierarchical RL
    - h-DQN (Kulkarni et al. 2016a)
    - Feudal Networks (Vezhnevets et al. 2017)
- State representation
    - pixel control or feature control (Jaderberg et al. 2016)
    - supervised predictions (Dosovitskiy and Koltun 2016)
    - successor features (Kulkarni et al. 2016b)
- Removing domain modifications
    - Pop-Art normalization (van Hasselt et al. 2016)
    - Fine-grained action repetition (Sharma, Lakshminarayanan, and Ravindran 2017)
    - A recurrent state network(Hausknecht and Stone 2015)