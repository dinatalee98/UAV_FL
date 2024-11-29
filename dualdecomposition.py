import numpy as np
from scipy.optimize import minimize
# 문제 설정
N = 10   # 라운드 수
K = 5    # IoT 디바이스 수
B_total = 20  # 전체 대역폭 (예: MHz)
epsilon = 0.1  # 통신 시간 제한
eta = 0.01     # 학습률
D_k = np.random.randint(100, 200, size=K)  # 각 디바이스의 데이터 크기

# 라그랑지 승수 초기화
lambda_n = np.zeros(N)  # 대역폭 제약에 대한 승수
mu_k = np.zeros(K)      # 에너지 제약에 대한 승수

# 초기 스케줄링 및 대역폭 값
a_k_n = np.random.rand(K, N)
b_k_n = np.full((K, N), B_total / K)
def optimize_a(a_k_n, mu_k, D_k, N, K):
    for k in range(K):
        for n in range(N):
            # 목적 함수 계산 (예: 간단한 선형 모델로 설정)
            gradient = (4 * D_k[k]**2) / (N * np.sum(D_k)) - mu_k[k]
            # 경사하강법 업데이트
            a_k_n[k, n] = max(0, min(1, a_k_n[k, n] - eta * gradient))
    return a_k_n
def optimize_b(b_k_n, lambda_n, mu_k, B_total, N, K):
    for n in range(N):
        for k in range(K):
            # 목적 함수: 대역폭과 에너지 소모를 최소화
            gradient = -lambda_n[n] + mu_k[k] * (b_k_n[k, n] / B_total)
            # 경사하강법 업데이트
            b_k_n[k, n] = max(0, b_k_n[k, n] - eta * gradient)

        # 대역폭 합 제약 적용
        b_k_n[:, n] = np.clip(b_k_n[:, n], 0, B_total / K)
    return b_k_n
def update_lagrange_multipliers(lambda_n, mu_k, a_k_n, b_k_n, D_k, N, K, B_total, epsilon):
    gamma = 0.01  # 학습률

    # 대역폭 제약 갱신
    for n in range(N):
        lambda_n[n] = max(0, lambda_n[n] + gamma * (np.sum(b_k_n[:, n]) - B_total))

    # 에너지 제약 갱신
    for k in range(K):
        energy_usage = np.sum(a_k_n[k, :] * b_k_n[k, :])  # 가상 에너지 모델
        mu_k[k] = max(0, mu_k[k] + gamma * (energy_usage - epsilon))

    return lambda_n, mu_k
def dual_decomposition(N, K, B_total, epsilon, max_iterations=100):
    # 초기화
    a_k_n = np.random.rand(K, N)
    b_k_n = np.full((K, N), B_total / K)
    lambda_n = np.zeros(N)
    mu_k = np.zeros(K)

    for iteration in range(max_iterations):
        # Subproblem 최적화
        a_k_n = optimize_a(a_k_n, mu_k, D_k, N, K)
        b_k_n = optimize_b(b_k_n, lambda_n, mu_k, B_total, N, K)

        # 라그랑지 승수 업데이트
        lambda_n, mu_k = update_lagrange_multipliers(lambda_n, mu_k, a_k_n, b_k_n, D_k, N, K, B_total, epsilon)

        # 수렴 조건 확인 (예: 변화량이 작으면 중단)
        if iteration > 1 and np.max(np.abs(a_k_n - prev_a_k_n)) < 1e-3 and np.max(np.abs(b_k_n - prev_b_k_n)) < 1e-3:
            break
        prev_a_k_n = a_k_n.copy()
        prev_b_k_n = b_k_n.copy()

    return a_k_n, b_k_n, lambda_n, mu_k

a_k_n_opt, b_k_n_opt, lambda_opt, mu_opt = dual_decomposition(N, K, B_total, epsilon)

# 결과 확인
print("Optimized Scheduling Variables (A):")
print(a_k_n_opt)

print("\nOptimized Bandwidth Allocation (B):")
print(b_k_n_opt)
