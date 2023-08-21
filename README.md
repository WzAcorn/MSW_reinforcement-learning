# MSW_reinforcement-learning


# 💻 A2C 알고리즘 개요

소개
A2C(Advantage Actor-Critic)는 강화학습(Reinforcement Learning)에서 널리 사용되는 알고리즘 중 하나입니다. 이 알고리즘은 강화학습에서 에이전트가 환경과 상호작용하며 특정 작업을 수행하는 정책(Policy)를 개선하는 데 사용됩니다. A2C는 정책 기반(Policy-based)과 가치 기반(Value-based) 강화학습 알고리즘의 장점을 결합한 형태로, 안정적이고 높은 성능을 보입니다.

Actor-Critic 알고리즘
먼저, Actor-Critic 알고리즘에 대해 간략히 설명하겠습니다. 이 알고리즘은 크게 두 가지 구성 요소로 이루어집니다.

Actor: 정책을 학습하는 부분으로, 주어진 상태에서 어떤 행동을 취할지 결정하는 확률적 정책을 만들고 업데이트합니다.
Critic: 가치 함수를 학습하는 부분으로, 현재 상태의 가치를 추정하여 보상의 기댓값을 계산하고, 이를 통해 Actor를 학습합니다.
A2C(Advantage Actor-Critic) 알고리즘
A2C는 Actor-Critic 알고리즘의 확장된 버전으로, 주로 병렬 환경에서 효율적으로 학습할 수 있는 장점을 갖고 있습니다. A2C는 여러 에이전트가 서로 다른 경험을 쌓고 이를 모아 하나의 큰 배치로 처리하는 방식으로 학습됩니다.

Advantage: A2C에서는 장점(Advantage)을 사용합니다. 장점은 현재 상태에서 어떤 행동이 기대 보상보다 얼마나 더 나은지를 나타냅니다. 이를 통해 에이전트가 더 나은 행동을 선택하도록 돕습니다.

에피소드 별 학습: A2C는 에피소드 단위로 학습이 이루어집니다. 에피소드는 에이전트가 시작 상태에서 종료 상태까지 상호작용하는 과정을 말합니다.

병렬 환경: 여러 에이전트가 서로 다른 환경에서 상호작용하고 그 경험을 모으는 병렬 구조를 활용하여 학습 효율을 향상시킵니다.

정책 업데이트: A2C는 에피소드가 끝난 후 한 번의 정책 업데이트가 이루어집니다. 정책 업데이트는 정책 그래디언트를 사용하여 이루어지며, 장점을 활용하여 행동의 상대적 가치를 반영합니다.

A2C 알고리즘의 장점
높은 학습 효율성: 병렬 환경을 활용하여 빠른 학습이 가능합니다.
안정성: Advantage를 사용하여 학습의 안정성을 향상시킵니다.
높은 성능: Policy-based와 Value-based의 장점을 결합하여 높은 성능을 달성합니다.
A2C 알고리즘의 단점
Hyperparameter 튜닝의 어려움: 알고리즘 성능에 영향을 주는 하이퍼파라미터가 많아서 적절한 조정이 필요합니다.
샘플 효율성: A2C는 여러 에이전트로부터 얻은 샘플을 활용하지만, 샘플 효율성은 아직 부족한 편입니다.
결론
A2C는 강화학습에서 널리 사용되는 알고리즘으로, 안정성과 성능 면에서 우수한 결과를 보여줍니다. 병렬 환경과 Advantage를 활용하여 빠른 학습과 효율적인 정책 업데이트를 가능하게 합니다. 그러나 하이퍼파라미터 조정과 샘플 효율성에 주의해야 합니다.


![UIgroup](https://github.com/ilovegalio/MSW_reinforcement-learning/assets/77008882/95ad9fdb-6af2-4450-8f73-a8acb73e0a40)
![게임화면](https://github.com/ilovegalio/MSW_reinforcement-learning/assets/77008882/562ee366-9e2a-4b57-8c04-feaf9c8df9f5)
![인게임전체](https://github.com/ilovegalio/MSW_reinforcement-learning/assets/77008882/8c233955-5bab-48d3-a0de-5a62250543e7)
 
# 결과

![Picture](https://github.com/ilovegalio/MSW_reinforcement-learning/assets/77008882/1789c17f-51af-4f3d-a0b9-92394f105e66) (학습 실패)

MSW 환경에서 python으로 데이터를 통신할 수 있는 방법을 찾지 못해 상태와 보상을 게임화면 좌상단의 이미지로 입력받음.
이 과정에서 현재 상태와 점수가 딜레이가 발생, 쓰레기 값들이 들어가 제대로 된 학습이 일어나지 않음.
