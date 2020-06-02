# A Distributional Perspective on Reinforcement Learning

Algorithm: Distributional RL
Author: Marc G. Bellemare, Will Dabney, Remi Munos
Last Edited By: Sung Yeul Back

# Abstract

Distribution을 return으로 하는 Reinforcement Learning

# Introduction

행동에 별다른 제약이 없다는 Q 또는 value를 maximize하는 것을 목표로 했습니다.

이 논문에서는 이런 기본 개념을 넘어 Distributional perspective를 써야하는 이유를 다룹니다.

기대값이 Q 값 (distribution의 평균값) 인 the random return Z가 이 연구의 주제

A recursive equation으로 표현하면

![A%20Distributional%20Perspective%20on%20Reinforcement%20Lear%20b57c5c9fae084786b475d7655db81623/Untitled.png](A%20Distributional%20Perspective%20on%20Reinforcement%20Lear%20b57c5c9fae084786b475d7655db81623/Untitled.png)

이 *distributional Bellman equation*은 reward R, next state-action (X', A'), 그리고 이것의 random return Z(X', A')로 특징되는 Z의 distribution입니다.

이걸 *the value distribution* 이라 부릅니다다.

The distributional perspective는 전부터 있었지만 특정 케이스를 제외하고는 잘 안쓰였습니다.

### Contraction of the policy evaluation Bellman operator

Bellman operator의 value distributon 버전은 Wasserstein metric의 maximal form으로 표현된 contraction입니다.

- What is Bellman operator

    Bellman operator는 모든 state s에 대한 value를 새로운 value로 mapping

    Bellman operator는 Contraction mapping

    Contraiction mapping : mapping 할수록 두 점의 거리가 가까워지는 mapping

    출처 : [http://www.modulabs.co.kr/RL_Practice/10307](http://www.modulabs.co.kr/RL_Practice/10307)

- Wasserstein metric?

    간단하게, distributions 사이에 metric (거리, 각도) 혹은 norm을 구할 수 있게 해주는 metric space

- Kullback-Leibler divergence?

    두 확률분포의 차이를 계산하는데 사용되는 함수로, 어떤 이상적인 분포에 대해, 그 분포를 근사하는 다른 분포를 사용해 샘플링을 한다면 발생할수 있는 정보 엔트로피 차이를 계산.

    비대칭으로 두 값의 위치를 바꾸면 함수값이 달라지므로 거리함수는 아닙니다다.

    ![A%20Distributional%20Perspective%20on%20Reinforcement%20Lear%20b57c5c9fae084786b475d7655db81623/Untitled%201.png](A%20Distributional%20Perspective%20on%20Reinforcement%20Lear%20b57c5c9fae084786b475d7655db81623/Untitled%201.png)

    출처 : [https://ko.wikipedia.org/wiki/쿨백-라이블러_발산](https://ko.wikipedia.org/wiki/%EC%BF%A8%EB%B0%B1-%EB%9D%BC%EC%9D%B4%EB%B8%94%EB%9F%AC_%EB%B0%9C%EC%82%B0)

- Kolmogorov distance

    ![A%20Distributional%20Perspective%20on%20Reinforcement%20Lear%20b57c5c9fae084786b475d7655db81623/Untitled%202.png](A%20Distributional%20Perspective%20on%20Reinforcement%20Lear%20b57c5c9fae084786b475d7655db81623/Untitled%202.png)

    ![A%20Distributional%20Perspective%20on%20Reinforcement%20Lear%20b57c5c9fae084786b475d7655db81623/Untitled%203.png](A%20Distributional%20Perspective%20on%20Reinforcement%20Lear%20b57c5c9fae084786b475d7655db81623/Untitled%203.png)

    출처 : [https://en.wikipedia.org/wiki/Kolmogorov–Smirnov_test](https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Smirnov_test)

### Instability in the control setting

~~nonstationary policy가 model 학습에 더 좋다?~~

### Better approximations

알고리즘적 관점에서 distribution 근사가 expectation 근사보다 학습에 이점이 있습니다.

Distribution일때 학습을 더 안정적이게 해주는 여러 방법을 쓸 수 있습니다.

The distributional perspective를 사용한 DQN은 실제도 더 좋은 성능을 보여주었습니다.

> Learn a guess from a guess - Sutton & Barto(1999)

→" Distribution을 쓰는 것이 scaler value를 쓰는 것보다 좋다."

# Setting

MDP (X, A, R, P, γ)

## Bellman's Equations

return Z^π := 감가된 보상들의 합

![A%20Distributional%20Perspective%20on%20Reinforcement%20Lear%20b57c5c9fae084786b475d7655db81623/Untitled%204.png](A%20Distributional%20Perspective%20on%20Reinforcement%20Lear%20b57c5c9fae084786b475d7655db81623/Untitled%204.png)

Bellman's equation으로 표현된 value function

![A%20Distributional%20Perspective%20on%20Reinforcement%20Lear%20b57c5c9fae084786b475d7655db81623/Untitled%205.png](A%20Distributional%20Perspective%20on%20Reinforcement%20Lear%20b57c5c9fae084786b475d7655db81623/Untitled%205.png)

이 논문에서는 value functions 을 벡터로 생각합니다.

Bellman operator와 optimality operator를 다음과 같이 정의합니다.

![A%20Distributional%20Perspective%20on%20Reinforcement%20Lear%20b57c5c9fae084786b475d7655db81623/Untitled%206.png](A%20Distributional%20Perspective%20on%20Reinforcement%20Lear%20b57c5c9fae084786b475d7655db81623/Untitled%206.png)

이 둘은 contraction mapping입니다.

# The Distributional Bellman Operators

expectation을 빼고 full distribution을 대체해 쓰려고 합니다.

Z^π를 state-action pair에서 distributions over returns로 mapping하는 함수라고 하고 이를 value distribution이라 부릅니다.

## Distribution Equations

- The probability space : 공간 전체의 측도가 1인 공간
    - Pr(Ω) = 1
    - Ω : sample space
    - F : measureable space, 사건
    - Pr : probability

Ω는 experiment에서 가능한 모든 결과의 space라고 생각하면 됩니다.

$$||u||_p$$

- Lebesgue space
    - 절대값의 p승이 르배그 적분 가능한 가측 함수들의 동치류들로 구성된 norm-space

![A%20Distributional%20Perspective%20on%20Reinforcement%20Lear%20b57c5c9fae084786b475d7655db81623/Untitled%207.png](A%20Distributional%20Perspective%20on%20Reinforcement%20Lear%20b57c5c9fae084786b475d7655db81623/Untitled%207.png)

이것을 value function의 vector에 적용을 하면,

![A%20Distributional%20Perspective%20on%20Reinforcement%20Lear%20b57c5c9fae084786b475d7655db81623/Untitled%208.png](A%20Distributional%20Perspective%20on%20Reinforcement%20Lear%20b57c5c9fae084786b475d7655db81623/Untitled%208.png)

이라 했을 때

(1≤ p ≤ infinity)

![A%20Distributional%20Perspective%20on%20Reinforcement%20Lear%20b57c5c9fae084786b475d7655db81623/Untitled%209.png](A%20Distributional%20Perspective%20on%20Reinforcement%20Lear%20b57c5c9fae084786b475d7655db81623/Untitled%209.png)

(p = infinity)

![A%20Distributional%20Perspective%20on%20Reinforcement%20Lear%20b57c5c9fae084786b475d7655db81623/Untitled%2010.png](A%20Distributional%20Perspective%20on%20Reinforcement%20Lear%20b57c5c9fae084786b475d7655db81623/Untitled%2010.png)

라고 정의합니다.

c.d.f of a random variable U

![A%20Distributional%20Perspective%20on%20Reinforcement%20Lear%20b57c5c9fae084786b475d7655db81623/Untitled%2011.png](A%20Distributional%20Perspective%20on%20Reinforcement%20Lear%20b57c5c9fae084786b475d7655db81623/Untitled%2011.png)

inverse c.d.f

![A%20Distributional%20Perspective%20on%20Reinforcement%20Lear%20b57c5c9fae084786b475d7655db81623/Untitled%2012.png](A%20Distributional%20Perspective%20on%20Reinforcement%20Lear%20b57c5c9fae084786b475d7655db81623/Untitled%2012.png)

![A%20Distributional%20Perspective%20on%20Reinforcement%20Lear%20b57c5c9fae084786b475d7655db81623/Untitled%2013.png](A%20Distributional%20Perspective%20on%20Reinforcement%20Lear%20b57c5c9fae084786b475d7655db81623/Untitled%2013.png)

- random variable U가 V 규칙에 의해 분산

### The Wasserstein Metric

main tool, metric between cumulative distribution functions

![A%20Distributional%20Perspective%20on%20Reinforcement%20Lear%20b57c5c9fae084786b475d7655db81623/Untitled%2014.png](A%20Distributional%20Perspective%20on%20Reinforcement%20Lear%20b57c5c9fae084786b475d7655db81623/Untitled%2014.png)

![A%20Distributional%20Perspective%20on%20Reinforcement%20Lear%20b57c5c9fae084786b475d7655db81623/Untitled%2015.png](A%20Distributional%20Perspective%20on%20Reinforcement%20Lear%20b57c5c9fae084786b475d7655db81623/Untitled%2015.png)

두 c.d.f인 F와 G에 대해

![A%20Distributional%20Perspective%20on%20Reinforcement%20Lear%20b57c5c9fae084786b475d7655db81623/Untitled%2016.png](A%20Distributional%20Perspective%20on%20Reinforcement%20Lear%20b57c5c9fae084786b475d7655db81623/Untitled%2016.png)

라고 정의를 합니다.

![A%20Distributional%20Perspective%20on%20Reinforcement%20Lear%20b57c5c9fae084786b475d7655db81623/Untitled%2017.png](A%20Distributional%20Perspective%20on%20Reinforcement%20Lear%20b57c5c9fae084786b475d7655db81623/Untitled%2017.png)

![A%20Distributional%20Perspective%20on%20Reinforcement%20Lear%20b57c5c9fae084786b475d7655db81623/Untitled%2018.png](A%20Distributional%20Perspective%20on%20Reinforcement%20Lear%20b57c5c9fae084786b475d7655db81623/Untitled%2018.png)