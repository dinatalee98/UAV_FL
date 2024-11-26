import numpy as np
import cvxpy as cp

K = 40 # number of IoT devices
N = 30 # number of round

c = 3e8

c_k = 10
f_k = 5 
alpha_k = 10 ** -28
zeta1 = 9.61
zeta2 = 0.16
H_u = 100
f_c = 2
eta_LoS = 1
eta_NLoS = 20
alpha_0 = -60
beta = 2.2
K_bar = 3
sigma_sq = -110
b_k = 80

E_k = np.full(K, 1)
D_k = np.full(K, 1)
B = 10
kappa = 10
eta = 0.001
epsilon = 0.001
D = np.sum(D_k)
F_w0 = 1
F_star = 0
p_k = np.full(K, 1)
s = 10

location = np.ones((K, N))


def euclidean_distance(uav_pos, device_pos, altitude): # System model d_k(n)
    return np.sqrt(np.sum((uav_pos - device_pos) ** 2) - altitude ** 2)

def computation_time_energy(k, n, a): # Eq 3 & 4
    t_comp = (c_k * D_k[k]) / f_k
    E_comp = a[k][n] * (alpha_k / 2) * c_k * D_k[k] * (f_k ** 2)
    return t_comp, E_comp

def P_LoS(k, n):
    theta_k = np.arctan(H_u / location[k][n]) # d[k][n] distance between UAV and IoT
    return 1 / (1 + zeta1 * np.exp(-zeta2 * (180 / np.pi * theta_k - zeta1)))

def PL_LoS(k, n):
    return 20 * np.log10((4 * np.pi * f_c * location[k][n]) / c) + eta_LoS

def PL_NLoS(k, n):
    return 20 * np.log10((4 * np.pi * f_c * location[k][n]) / c) + eta_NLoS

def path_loss(k, n, P_LoS): # Eq 6 & 7 & 8
    return P_LoS * PL_LoS(k, n) + (1 - P_LoS) * PL_NLoS(k, n)


def channel_coefficient(k, n): # Eq 9, 10, 11
    PL_LoS_k_n = PL_LoS(k, n)
    PL_NLoS_k_n = PL_NLoS(k, n)

    large_scale_fading = alpha_0 * location[k][n] ** beta
    small_scale_fading = np.sqrt((K_bar / (K_bar+1)) * PL_LoS_k_n) + np.sqrt((1 / (K_bar+1)) * PL_NLoS_k_n)
    h_k = np.sqrt(large_scale_fading * small_scale_fading)
    return h_k


def data_rate_and_energy(k, n, a): # Eq 12, 13, 14
    h_k = channel_coefficient(k, n)
    
    R_k = b_k * np.log2(1 + (p_k[n] * np.linalg.norm(h_k) ** 2) / sigma_sq)
    t_comm = (a[k][n] * s) / R_k
    E_comm = (t_comm * (sigma_sq) * (2 ** (a[k][n] * s / (b_k * t_comm)) - 1)) / np.linalg.norm(h_k) ** 2
    return R_k, t_comm, E_comm


####################

#Optimize part

####################




def optimize_resource_allocation():
    # 결정 변수
    A = cp.Variable((K, N), boolean=True)  # a_k[n]

    # 목적 함수
    objective_term1 = (2/(N * eta)) * (F_w0 - F_star)
    objective_term2 = (4*K*kappa/(N*D**2)) * cp.sum(cp.sum(cp.multiply((1-A[:, 1:]), D_k**2)))
    objective = objective_term1 + objective_term2

    constraints = []
    
    # Constraint 1: Binary constraint는 이미 boolean=True로 처리됨
    
    # Constraint 2: Bandwidth constraint
    constraints.append(b_k * N <= B)
    
    # Constraint 3: Energy constraint
    for k in range(K):
        total_energy = 0
        for n in range(N):
            # 각 시간 단계에서의 통신 및 계산 에너지 계산
            _, E_comp = computation_time_energy(k, n, A)
            _, _, E_comm = data_rate_and_energy(k, n, A)
            total_energy += E_comm + E_comp
        constraints.append(total_energy <= E_k)
    
    # Constraint 4: Communication time constraint
    for k in range(K):
        for n in range(N):
            _, t_comm, _ = data_rate_and_energy(k, n, A)
            constraints.append(t_comm <= epsilon)
    
    # 최적화 문제 정의
    problem = cp.Problem(cp.Minimize(objective), constraints)
    
    # 문제 해결
    problem.solve(solver=cp.MOSEK)  # or another appropriate solver
    return A.value
    