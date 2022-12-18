import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import MapleEnv

class FCDAP(nn.Module):
    def __init__(self,input_dim, output_dim, hidden_dims=(32,32), activation_fc=F.relu):
        super(FCDAP, self).__init__()
        self.activation_fc = activation_fc
        # 입력계층->은닉계층1
        self.input_layer = nn.Linear(input_dim, hidden_dims[0])
        # 은닉계층
        self.hidden_layers = nn.ModuleList() # 신경망 모듈 리스트 초기화
        # 은닉계층-1 개 추가
        for i in range(len(hidden_dims)-1):
            hidden_layer = nn.Linear(hidden_dims[i], hidden_dims[i+1]) # 은닉계층 추가
            self.hidden_layers.append(hidden_layer) # 모듈 리스트에 추가
        self.output_layer = nn.Linear(hidden_dims[-1], output_dim) # 마지막 은닉계층->출력계층

        # CUDA(GPU) 가용하면 GPU로 디바이스 설정
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda:0"
        self.device = torch.device(device) # 신경망 계산 디바이스 설정
        self.to(self.device) # 모듈의 파라미터/버퍼 등을 해당 디바이스로 이동(캐스트)
        
        # 입력이 파이토치 텐서가 아니면 텐서로 변환
    def _format(self, state):
        x = state
        if not isinstance(x, torch.Tensor): # 파이토치 텐서가 아니면 텐서로 변환
            x = torch.tensor(x, device=self.device, dtype=torch.float32)
            x = x.unsqueeze(0) # 첫번째 차원 추가
        return x
    
        # 신경망 순전파 함수
    def forward(self, state):
        x = self._format(state) # 텐서로 변환
        x = self.activation_fc(self.input_layer(x)) # 입력계층
        self.i =0
        for hidden_layer in self.hidden_layers: # 은닉계층들
            self.i +=1
            x = self.activation_fc(hidden_layer(x))
            x = self.output_layer(x) # 출력계층
        return x
    
    # 순전파 수행 후 행동 확률(정책)과 로그 계산
    def full_pass(self, state):
        # 각 행동의 확률(신경망 출력) 예측
        logits = self.forward(state)  # 순전파 함수 호출
        # 행동의 카테고리형 확률 분포 생성
        dist = torch.distributions.Categorical(logits=logits)
        # 확률 분포로부터 샘플링(행동 선택)
        action = dist.sample()
        # 지정된 행동 확률(정책)에 대한 로그 값
        logpa = dist.log_prob(action).unsqueeze(-1)
        # 엔트로피 계산
        entropy = dist.entropy().unsqueeze(-1)
        is_exploratory = action != np.argmax(logits.detach().numpy())  # 탐색 여부
        return action.item(), is_exploratory.item(), logpa, entropy
    
    # 출력의 확률 분포로부터 샘플링하여 행동 선택
    def select_action(self, state):
        logits = self.forward(state)  # 각 행동의 확률(신경망 출력) 예측
        dist = torch.distributions.Categorical(logits=logits)  # 행동의 카테고리형 확률 분포 생성
        action = dist.sample()  # 확률 분포로부터 샘플링(행동 선택)
        return action.item()  # 행동의 인덱스 값 리턴

    # 그리디 행동 선택
    def select_greedy_action(self, state):
        logits = self.forward(state)
        return np.argmax(logits.detach().numpy())
    
class FCV(nn.Module):
    def __init__(self, input_dim, # 입력 차원
        hidden_dims=(32,32), activation_fc=F.relu): # 은닉계층, 활성화 함수
        super(FCV, self).__init__()
        self.activation_fc = activation_fc
        self.input_layer = nn.Linear(input_dim, hidden_dims[0])
        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_dims)-1):
            hidden_layer = nn.Linear(hidden_dims[i], hidden_dims[i+1])
            self.hidden_layers.append(hidden_layer)
            self.output_layer = nn.Linear(hidden_dims[-1], 1)
        # 입력이 파이토치 텐서가 아니면 텐서로 변환
        # CUDA(GPU) 가용하면 GPU로 디바이스 설정
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda:0"
        self.device = torch.device(device)  # 신경망 계산 디바이스 설정
        self.to(self.device)  # 모듈의 파라미터/버퍼 등을 해당 디바이스로 이동(캐스트)
    def _format(self, state):
        x = state
        if not isinstance(x, torch.Tensor):  # 파이토치 텐서가 아니면 텐서로 변환
            x = torch.tensor(x, device=self.device, dtype=torch.float32)
            x = x.unsqueeze(0)  # 첫번째 차원 추가
        return x
        # 신경망 순전파 함수
    def forward(self, state):
        x = self._format(state)  # 텐서로 변환
        x = self.activation_fc(self.input_layer(x))  # 입력계층
        self.i = 0
        for hidden_layer in self.hidden_layers:  # 은닉계층들
            self.i += 1
            x = self.activation_fc(hidden_layer(x))
            x = self.output_layer(x)  # 출력계층
        return x

# A2C 에이전트
class A2C():
    def __init__(self,
        policy_model_fn, # 정책신경망 구성 람다함수
        policy_model_max_grad_norm, # 정책 경사 절단 임계값
        policy_optimizer_fn, # 정책신경망 최적화기법 람다함수
        policy_optimizer_lr, # 정책신경망 학습 속도
        value_model_fn, # 가치신경망 구성 람다함수
        value_model_max_grad_norm, # 가치 경사 절단 임계값
        value_optimizer_fn, # 가치 신경망 최적화기법 람다함수
        value_optimizer_lr, # 가치 신경망 학습 속도
        entropy_loss_weight, # 엔트로피 가중치
        max_n_steps): # 최대 부투스트래핑 스텝 수
        # 정책 신경망 속성 초기화
        self.policy_model_fn = policy_model_fn
        self.policy_model_max_grad_norm = policy_model_max_grad_norm
        self.policy_optimizer_fn = policy_optimizer_fn
        self.policy_optimizer_lr = policy_optimizer_lr
        # 가치 신경망 속성 초기화
        self.value_model_fn = value_model_fn
        self.value_model_max_grad_norm = value_model_max_grad_norm
        self.value_optimizer_fn = value_optimizer_fn
        self.value_optimizer_lr = value_optimizer_lr
        # 엔트로피 가중치, 최대 n 스텝 초기화
        self.entropy_loss_weight = entropy_loss_weight
        self.max_n_steps = max_n_steps
        self.render = False

    # 신경망 학습 모델
    def optimize_model(self):
        # n-스텝들의 할인율, 리턴값 계산 후 텐서 변환
        T = len(self.rewards)
        discounts = np.logspace(0, T, num=T, base=self.gamma, endpoint=False)  # n-스텝 할인율
        returns = np.array([np.sum(discounts[:T - t] * self.rewards[t:]) for t in range(T)])  # n-스텝 리턴
        discounts = torch.FloatTensor(discounts[:-1]).unsqueeze(1)  # 텐서 변환
        returns = torch.FloatTensor(returns[:-1]).unsqueeze(1)  # 텐서 변환
        
        # 로그/엔트로피/상태가치의 리스트의 개별 텐서를 결합
        logpas = torch.cat(self.logpas)
        entropies = torch.cat(self.entropies)
        values = torch.cat(self.values)
        # 정책신경망 손실함수 계산
        value_error = returns - values  # 어드밴티지
        policy_loss = -(value_error.detach() * logpas).mean()  # 정책 손실함수 평균
        entropy_loss = -entropies.mean()  # 엔트로피 평균
        loss = policy_loss + self.entropy_loss_weight * entropy_loss  # 정책신경망 손실함수
        # 정책신경망 최적화(역전파)
        self.policy_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(), self.policy_model_max_grad_norm)  # 경사 절단
        self.policy_optimizer.step()
        # 가치신경망 손실함수 평균계산
        value_loss = value_error.pow(2).mul(0.5).mean()
        # 가치신경망 최적화(역전파)
        self.value_optimizer.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.value_model.parameters(), self.value_model_max_grad_norm)  # 경사 절단
        self.value_optimizer.step()

    # 행동 선택(순전파 예측) 후 다음 상태 전이
    def interaction_step(self, state, env):
        # 한 스텝 정책 신경망 순전파 수행 후 행동 확률(정책)과 로그 수집
        action, is_exploratory, logpa, entropy = self.policy_model.full_pass(state)
        # 행동 수행, 상태 전이
        new_state, reward, is_terminal,is_truncated,_ = env.step(action)
        # 현재 상태의 행동 확률(정책) 로그, 엔트로피, 상태-가치 추가
        self.logpas.append(logpa)
        self.rewards.append(reward)
        self.entropies.append(entropy)
        self.values.append(self.value_model(state))
        self.episode_reward[-1] += reward  # 보상 추가
        self.episode_timestep[-1] += 1  # 타임 스텝 증가
        self.episode_exploration[-1] += int(is_exploratory)  # 탐색 횟수 증가
        return new_state, is_terminal, is_truncated

    # 신경망 훈련
    def train(self, env, seed, gamma, max_minutes, max_episodes, goal_mean_100_reward):
        # 속성 초기화
        training_start, last_debug_time = time.time(), float('-inf')
        self.seed = seed
        self.gamma = gamma
        torch.manual_seed(self.seed);
        np.random.seed(self.seed);
        random.seed(self.seed)
        nS, nA = 288, 2
        self.episode_timestep = []
        self.episode_reward = []
        self.episode_seconds = []
        self.episode_exploration = []
        self.evaluation_scores = []
        # 정책 신경망, 최적화 객체 생성
        self.policy_model = self.policy_model_fn(nS, nA)
        self.policy_optimizer = self.policy_optimizer_fn(self.policy_model, self.policy_optimizer_lr)
        # 가치 신경망, 최적화 객체 생성
        self.value_model = self.value_model_fn(nS)
        self.value_optimizer = self.value_optimizer_fn(self.value_model, self.value_optimizer_lr)
        # 통계 변수 초기화
        result = np.empty((max_episodes, 5))  # 결과 리스트
        result[:] = np.nan
        training_time = 0  # 학습 시간
        step, n_steps_start = 0, 0  # 현재 스텝, n 스텝 시작
        # 에피소드 반복
        for episode in range(1, max_episodes + 1):
            episode_start = time.time()  # 에피소드 시작 시간
            # 상태, 통계 변수 초기화
            state, _ = env.reset()
            is_terminal=False
            self.episode_reward.append(0.0)
            self.episode_timestep.append(0.0)
            self.episode_exploration.append(0.0)
            # 행동 로그, 엔트로피, 보상, 상태 가치 리스트 초기화
            self.logpas, self.entropies, self.rewards, self.values = [], [], [], []
            # 종료 상태 도달 시까지 반복
            while not is_terminal:
                # 행동 수행하고 상태 전이
                state, is_terminal, is_truncated = self.interaction_step(state, env)
                # n 스텝 도달 또는 종료이면 최적화(역전파) 수행
                if is_terminal or step - n_steps_start == self.max_n_steps:
                    # 마지막 스텝의 예측 가치를 보상 리스트에 추가
                    is_failure = is_terminal and not is_truncated
                    next_value = 0 if is_failure else self.value_model(state).detach().item()
                    self.rewards.append(next_value)
                    # 최적화(역전파) 수행
                    self.optimize_model()
                    # n 스텝 저장 변수 리스트 초기화
                    self.logpas, self.entropies, self.rewards, self.values = [], [], [], []
                    n_steps_start = step
            # 에피소드 당 통계 출력
            episode_elapsed = time.time() - episode_start  # 경과시간
            self.episode_seconds.append(episode_elapsed)  # 경과시간 리스트에 추가
            training_time += episode_elapsed  # 훈련시간
            total_step = int(np.sum(self.episode_timestep))  # 누적 타임 스텝
            # 한 에피소드 훈련 종료 후 그리디 행동 선택 적용하여 에피소드의 누적 보상 계산
            evaluation_score, _ = self.evaluate(self.policy_model, env)
            self.evaluation_scores.append(evaluation_score)  # 누적 보상 리스트에 추가
            # 마지막 1/10 스텝 평균 훈련 보상/표준편차
            mean_10_reward = np.mean(self.episode_reward[-1:])
            std_10_reward = np.std(self.episode_reward[-1:])
            mean_100_reward = np.mean(self.episode_reward[-10:])
            std_100_reward = np.std(self.episode_reward[-10:])
            # 마지막 1100 스텝 평균 평가 보상/표준편차
            mean_100_eval_score = np.mean(self.evaluation_scores[-10:])
            std_100_eval_score = np.std(self.evaluation_scores[-10:])
            # 마지막 100스텝 탐색 수/표준편차 출력
            lst_100_exp_rat = np.array(
                self.episode_exploration[-10:]) / np.array(self.episode_timestep[-10:])
            mean_100_exp_rat = np.mean(lst_100_exp_rat)
            std_100_exp_rat = np.std(lst_100_exp_rat)
            # 경과 시간 계산
            wallclock_elapsed = time.time() - training_start
            # 에피소드 당 결과 값 리스트에 저장
            result[episode - 1] = total_step, mean_100_reward, \
                                  mean_100_eval_score, training_time, wallclock_elapsed
            LEAVE_PRINT_EVERY_N_SECS = 60   # 행 출력 지속 시간
            ERASE_LINE = '\x1b[2K'  # 행 삭제
            reached_debug_time = time.time() - last_debug_time >= LEAVE_PRINT_EVERY_N_SECS
            # 디버그 출력 지속시간 초과 플래그 설정
            reached_max_minutes = wallclock_elapsed >= max_minutes * 60  # 최대 제한 시간 초과 플래그 설정
            reached_max_episodes = episode >= max_episodes  # 최대 제한 에피소드 수 초과 플래그 설정
            reached_goal_mean_reward = mean_100_eval_score >= goal_mean_100_reward
            # 에피소드 마지막 100스텝 최대 보상 초과 플래그 설정
            # 훈련 중지 플래그 설정: 최대 수행 시간이나 에피소드 초과 시 또는 마지막 10 스텝 평균 최대 보상 도달 시
            training_is_over = reached_max_minutes or reached_max_episodes or reached_goal_mean_reward
            # 에피소드 출력 메시지: 경과시간, 에피소드 번호, 스텝 수, 마지막 1/10 스텝 평균 훈련 보상/표준편차,
            # 마지막 10 스텝 평균 탐색 수/표준편차, 마지막 10 스텝 평균 평가 보상/표준편차
            elapsed_str = time.strftime("%H:%M:%S", time.gmtime(time.time() - training_start))
            debug_message = 'el {}, ep {:04}, ts {:06}, '
            debug_message += 'ar 1 {:05.1f}\u00B1{:05.1f}, '
            debug_message += '10 {:05.1f}\u00B1{:05.1f}, '
            debug_message += 'ex 10 {:02.1f}\u00B1{:02.1f}, '
            debug_message += 'ev {:05.1f}\u00B1{:05.1f}'
            debug_message = debug_message.format(
                elapsed_str, episode - 1, total_step, mean_10_reward, std_10_reward,
                mean_100_reward, std_100_reward, mean_100_exp_rat, std_100_exp_rat,
                mean_100_eval_score, std_100_eval_score)
            print(debug_message, end='\r', flush=True)

            # 디버그 출력 시간 초과 또는 훈련 종료 시 출력 메시지 화면 출력
            if reached_debug_time or training_is_over:
                print(ERASE_LINE + debug_message, flush=True)
                last_debug_time = time.time()
            if training_is_over:  # 훈련 종료 시 출력
                if reached_max_minutes: print(u'--> reached_max_minutes \u2715')
                if reached_max_episodes: print(u'--> reached_max_episodes \u2715')
                if reached_goal_mean_reward: print(u'--> reached_goal_mean_reward \u2713')
                break
            # 훈련 종료 후 모델 평가 수행
        final_eval_score, score_std = self.evaluate(self.policy_model, env, n_episodes=3)
        wallclock_time = time.time() - training_start  # 총 훈련 시간(모델 평가 포함)
        print('Training complete.')
        print('Final evaluation score {:.2f}\u00B1{:.2f} in {:.2f}s training time,'
              ' {:.2f}s wall-clock time.\n'.format(final_eval_score, score_std, training_time, wallclock_time))
        return result, final_eval_score, training_time, wallclock_time

    # 에피소드/훈련 종료 후 그리디 전략으로 보상 계산
    def evaluate(self, eval_policy_model, eval_env, n_episodes=1, greedy=True):
        rs = []
        for _ in range(n_episodes):
            s,_= eval_env.reset()
            done=False
            rs.append(0)
            while not done:
                if greedy:  # 그리디 전략으로 행동 선택
                    a = eval_policy_model.select_greedy_action(s)
                else:  # 정책 경사법 확률분포로 행동 선택
                    a = eval_policy_model.select_action(s)
                s, r, done,_, _ = eval_env.step(a)
                rs[-1] += r

        return np.mean(rs), np.std(rs)

    # 학습 후 그리디 전략으로 렌더링
    def render_after_train(self, r_env, n_episodes=1):
        for _ in range(n_episodes):
            s, _ = r_env.reset()
            done = False
            while not done:
                r_env.render()  # 렌더링
                a = self.policy_model.select_greedy_action(s)  # 그리디 전략으로 행동 수행하고 상태 전이
                s, r, done,_, _ = r_env.step(a)


                
## A2C 메인 루틴
env=MapleEnv.MapleEnv()
## A2C 메인 루틴
a2c_results = []
# 각기 다른 시드 값으로 수행

# 환경 세팅 파라미터
environment_settings = {
        'env_name': 'CartPole-v1', # 환경 이름
        'gamma': 1.00, # 감가율(할인율)
        'max_minutes': 360, # 최대 수행 시간
        'max_episodes': 500, # 최대 에피소드 수
        'goal_mean_100_reward': 500 # 마지막 100 스텝 평균 최대 보상
}
    # 신경망/최적화 기법 람다함수, 경사 절단 임계값
policy_model_fn = lambda nS, nA: FCDAP(nS, nA, hidden_dims=(128, 64))
policy_model_max_grad_norm = 1  # 정책신경망 경사 절단 임계값
policy_optimizer_fn = lambda net, lr: optim.Adam(net.parameters(), lr=lr)
policy_optimizer_lr = 0.0005
value_model_fn = lambda nS: FCV(nS, hidden_dims=(256, 128))
value_model_max_grad_norm = float('inf')  # 가치신경망 경사 절단 임계값
value_optimizer_fn = lambda net, lr: optim.RMSprop(net.parameters(), lr=lr)
value_optimizer_lr = 0.0007
entropy_loss_weight = 0.001  # 엔트로피
max_n_steps = 5  # 최대 부트스트랩 스텝 수
# 환경 파라미터 지정 및 환경 생성
env_name, gamma, max_minutes, \
max_episodes, goal_mean_100_reward = environment_settings.values()
# A2C 에이전트 생성
agent = A2C(policy_model_fn, policy_model_max_grad_norm, policy_optimizer_fn, policy_optimizer_lr,
                value_model_fn, value_model_max_grad_norm, value_optimizer_fn, value_optimizer_lr,
                entropy_loss_weight, max_n_steps)
    # A2C 신경망 훈련
result, final_eval_score, training_time, wallclock_time = agent.train(env, 12, gamma, max_minutes,
                                                                          max_episodes, goal_mean_100_reward)
a2c_results.append(result)  # 시드 결과 리스트에 추가
# 시드 결과 리스트를 넘파이 배열로 변환
a2c_results = np.array(a2c_results)

# 학습 후 카트폴 동작 렌더링
agent.render_after_train(env)
# 그래픽용 자료 추출
a2c_max_t, a2c_max_r, a2c_max_s, a2c_max_sec, a2c_max_rt = np.max(a2c_results, axis=0).T
a2c_min_t, a2c_min_r, a2c_min_s, a2c_min_sec, a2c_min_rt = np.min(a2c_results, axis=0).T
a2c_mean_t, a2c_mean_r, a2c_mean_s, a2c_mean_sec, a2c_mean_rt = np.mean(a2c_results, axis=0).T
a2c_x =500
# 에피소드 당 (마지막 100스탭) 평균 보상(훈련,평가), 총 스텝 수, 훈련/경과 시간 그래픽
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
plt.style.use('fivethirtyeight')
params = {
    'figure.figsize': (15, 8),
    'font.size': 24,
    'legend.fontsize': 20,
    'axes.titlesize': 28,
    'axes.labelsize': 24,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20
}
pylab.rcParams.update(params)
# 서브그래프 분할
fig, axs = plt.subplots(3, 1, figsize=(15,30), sharey=False, sharex=True)
# 마지막 100 스텝 훈련 시 평균 보상
axs[0].plot(a2c_max_r, 'y', linewidth=1)
axs[0].plot(a2c_min_r, 'y', linewidth=1)
axs[0].plot(a2c_mean_r, 'y', label='a2c', linewidth=2)
axs[0].fill_between(a2c_x, a2c_min_r, a2c_max_r, facecolor='y', alpha=0.3)
# 마지막 100 스텝 평가 시 평균 보상
axs[1].plot(a2c_max_s, 'y', linewidth=1)
axs[1].plot(a2c_min_s, 'y', linewidth=1)
axs[1].plot(a2c_mean_s, 'y', label='a2c', linewidth=2)
axs[1].fill_between(a2c_x, a2c_min_s, a2c_max_s, facecolor='y', alpha=0.3)
# 총 타임 스텝 수
axs[2].plot(a2c_max_t, 'y', linewidth=1)
axs[2].plot(a2c_min_t, 'y', linewidth=1)
axs[2].plot(a2c_mean_t, 'y', label='Da2c', linewidth=2)
axs[2].fill_between(a2c_x, a2c_min_t, a2c_max_t, facecolor='y', alpha=0.3)

# 차트 제목
axs[0].set_title('a2c: Moving Avg Reward (Training)')
axs[1].set_title('a2c: Moving Avg Reward (Evaluation)')
axs[2].set_title('a2c: Total Steps')
plt.xlabel('Episodes')
axs[0].legend(loc='upper left')
plt.show()

